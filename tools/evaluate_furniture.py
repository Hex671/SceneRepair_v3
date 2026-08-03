from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scene_repair_v3 import (
    ActionRange,
    FurnitureRepairNetwork,
    FurnitureGraphBuilder,
    FurnitureVocabularies,
    FrozenFunctionalPartnerRules,
    ModelConfig,
    apply_pose_deltas,
    hard_constraint_report,
    joint_translation_repair,
    collate_graphs,
)
from scene_repair_v3.checkpoint import load_checkpoint
from scene_repair_v3.data import CorruptionConfig
from scene_repair_v3.dataset import FurnitureCorruptionDataset, make_collate_fn


def _group(k: int) -> str:
    if k <= 1:
        return str(k)
    if k <= 3:
        return "2-3"
    if k <= 5:
        return "4-5"
    return "6-7+"


def _empty() -> dict[str, object]:
    return {
        "objects": 0, "correct": 0, "scenes": 0, "whole_correct": 0,
        "translation_sum": 0.0, "translation_count": 0,
        "yaw_sum": 0.0, "yaw_count": 0,
        "confusion": torch.zeros((4, 4), dtype=torch.long),
        "target_geometry_clear": 0, "prediction_geometry_clear": 0,
        "neural_raw_geometry_clear": 0,
        "projected_geometry_clear": 0, "violations_before": 0,
        "violations_neural_raw": 0,
        "violations_prediction": 0, "violations_projected": 0,
        "refinement_iterations_sum": 0, "refinement_iterations_max": 0,
        "refinement_nonconverged": 0, "max_final_translation_m": 0.0,
    }


def _finish(value: dict[str, object]) -> dict[str, object]:
    confusion: torch.Tensor = value.pop("confusion")
    tp = confusion.diag().double()
    precision = tp / confusion.sum(dim=0).clamp_min(1)
    recall = tp / confusion.sum(dim=1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        **value,
        "action_accuracy": value["correct"] / max(value["objects"], 1),
        "action_macro_f1": float(f1.mean()),
        "whole_scene_action_success": value["whole_correct"] / max(value["scenes"], 1),
        "translation_mae_m": value["translation_sum"] / max(value["translation_count"], 1),
        "yaw_mae_rad": value["yaw_sum"] / max(value["yaw_count"], 1),
        "target_geometry_clear_rate": value["target_geometry_clear"] / max(value["scenes"], 1),
        "prediction_geometry_clear_rate": value["prediction_geometry_clear"] / max(value["scenes"], 1),
        "neural_raw_geometry_clear_rate": value["neural_raw_geometry_clear"] / max(value["scenes"], 1),
        "projected_geometry_clear_rate": value["projected_geometry_clear"] / max(value["scenes"], 1),
        "mean_violations_before": value["violations_before"] / max(value["scenes"], 1),
        "mean_violations_prediction": value["violations_prediction"] / max(value["scenes"], 1),
        "mean_violations_neural_raw": value["violations_neural_raw"] / max(value["scenes"], 1),
        "mean_violations_projected": value["violations_projected"] / max(value["scenes"], 1),
        "mean_refinement_iterations": value["refinement_iterations_sum"] / max(value["scenes"], 1),
        "confusion_matrix_rows_true_cols_pred": confusion.tolist(),
    }


def _best_model_step(scene, proposal):
    """Select a scalar step along the learned joint trajectory only."""
    current = hard_constraint_report(scene).total
    best_scene, best_score = scene, (current, 0.0)
    for scale in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        candidate = apply_pose_deltas(scene, proposal * scale)
        total = hard_constraint_report(candidate).total
        score = (total, abs(scale - 1.0))
        if score < best_score:
            best_scene, best_score = candidate, score
    return best_scene


def _coordinate_model_steps(scene, proposal):
    """Greedily apply individual learned directions that reduce violations."""
    result = scene
    furniture_count = len(scene.furniture)
    for _ in range(max(1, furniture_count * 2)):
        current_total = hard_constraint_report(result).total
        if current_total == 0:
            break
        best_scene = result
        best_score = (current_total, math.inf, math.inf)
        for index in range(furniture_count):
            item_delta = torch.zeros_like(proposal)
            for scale in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5):
                item_delta[index] = proposal[index] * scale
                candidate = apply_pose_deltas(result, item_delta)
                total = hard_constraint_report(candidate).total
                score = (total, abs(scale - 1.0), index)
                if score < best_score:
                    best_scene, best_score = candidate, score
                item_delta[index].zero_()
        if best_score[0] >= current_total:
            break
        result = best_scene
    return result


def _geometry_action_search(scene, raw_delta):
    """Choose action components using the learned delta and visible constraints."""
    result = scene
    count = len(scene.furniture)
    for _ in range(max(1, count * 2)):
        current_total = hard_constraint_report(result).total
        if current_total == 0:
            break
        best_scene, best_score = result, (current_total, math.inf, math.inf)
        for index in range(count):
            for action in (0, 1, 2, 3):
                for scale in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
                    proposal = torch.zeros_like(raw_delta)
                    if action in (1, 3):
                        proposal[index, :2] = raw_delta[index, :2] * scale
                    if action in (2, 3):
                        proposal[index, 2] = raw_delta[index, 2] * scale
                    candidate = apply_pose_deltas(result, proposal)
                    total = hard_constraint_report(candidate).total
                    score = (total, abs(scale - 1.0), index)
                    if score < best_score:
                        best_scene, best_score = candidate, score
        if best_score[0] >= current_total:
            break
        result = best_scene
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Furniture checkpoint by changed-object count K")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--functional-rules", type=Path, required=True)
    parser.add_argument("--compatibility-rules", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples-per-scene", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--max-translation-m", type=float, default=3.0)
    parser.add_argument("--model-passes", type=int, default=1)
    parser.add_argument("--line-search", action="store_true")
    parser.add_argument("--coordinate-line-search", action="store_true")
    parser.add_argument("--geometry-action-search", action="store_true")
    parser.add_argument(
        "--constraint-aware-model",
        action="store_true",
        help="Evaluate the complete model contract, including internal refinement",
    )
    parser.add_argument(
        "--legacy-constraint-gate",
        action="store_true",
        help="Use the observable_v3 per-object identity gate for legacy checkpoints",
    )
    parser.add_argument(
        "--label-mode",
        choices=(
            "observable_joint_v4",
            "observable_v3",
            "semantic_restore_v1",
            "semantic_v2",
            "legacy_random",
        ),
        default="observable_joint_v4",
    )
    args = parser.parse_args()
    if args.model_passes <= 0:
        parser.error("--model-passes must be positive")
    if args.constraint_aware_model and (
        args.model_passes != 1
        or args.line_search
        or args.coordinate_line_search
        or args.geometry_action_search
    ):
        parser.error(
            "--constraint-aware-model is a complete one-pass inference contract "
            "and cannot be combined with caller-side search or repeated passes"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = FurnitureVocabularies.load(args.vocab)
    rules = FrozenFunctionalPartnerRules.load(args.functional_rules, args.compatibility_rules)
    builder = FurnitureGraphBuilder(vocab)
    dataset = FurnitureCorruptionDataset(
        args.manifest, split=args.split, vocabularies=vocab, rules=rules,
        base_seed=args.seed, samples_per_scene=args.samples_per_scene,
        corruption=CorruptionConfig(label_mode=args.label_mode),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda", collate_fn=make_collate_fn(),
    )
    model = FurnitureRepairNetwork(vocab, ModelConfig()).to(device)
    action_range = ActionRange(args.max_translation_m, math.pi)
    checkpoint_extra = load_checkpoint(args.checkpoint, model, action_range, map_location=device)
    model.eval()
    metrics = defaultdict(_empty)
    metrics["overall"]
    with torch.no_grad():
        for graph, targets, metadata in loader:
            graph, targets = graph.to(device), targets.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(graph)
            prediction = (
                output.action_type_logits.argmax(dim=-1)
                if not args.legacy_constraint_gate
                else __import__(
                    "scene_repair_v3.model", fromlist=["constraint_gated_action_types"]
                ).constraint_gated_action_types(output, graph)
            )
            normalized = output.primary_delta_normalized
            predicted_delta = torch.stack((
                normalized[:, 0] * action_range.max_translation_m,
                normalized[:, 1] * action_range.max_translation_m,
                normalized[:, 2] * action_range.max_yaw_rad,
            ), dim=-1)
            predicted_scenes = []
            for graph_id in range(graph.num_graphs):
                mask = output.furniture_graph_index == graph_id
                actions = prediction[mask]
                proposal = predicted_delta[mask].clone()
                move_enabled = (actions == 1) | (actions == 3)
                rotate_enabled = (actions == 2) | (actions == 3)
                proposal[:, :2] *= move_enabled.unsqueeze(-1)
                proposal[:, 2] *= rotate_enabled
                source_scene = metadata["corrupted_scenes"][graph_id]
                predicted_scenes.append(
                    _geometry_action_search(
                        source_scene, predicted_delta[mask]
                    )
                    if args.geometry_action_search
                    else _coordinate_model_steps(
                        source_scene, predicted_delta[mask]
                    )
                    if args.coordinate_line_search
                    else _best_model_step(source_scene, proposal)
                    if args.line_search else apply_pose_deltas(source_scene, proposal)
                )
            neural_raw_scenes = list(predicted_scenes)
            complete_output = None
            if args.constraint_aware_model:
                complete_output = model.refine_output(
                    graph,
                    metadata["corrupted_scenes"],
                    output,
                    action_range,
                )
                neural_raw_scenes = list(complete_output.neural_scenes)
                predicted_scenes = list(complete_output.scenes)
            for _ in range(args.model_passes - 1):
                active = [
                    index for index, scene in enumerate(predicted_scenes)
                    if hard_constraint_report(scene).total > 0
                ]
                if not active:
                    break
                refinement_graph = collate_graphs(
                    [builder.build(predicted_scenes[index]) for index in active]
                ).to(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    refinement_output = model(refinement_graph)
                refinement_actions = refinement_output.action_type_logits.argmax(dim=-1)
                refinement_normalized = refinement_output.primary_delta_normalized
                refinement_delta = torch.stack((
                    refinement_normalized[:, 0] * action_range.max_translation_m,
                    refinement_normalized[:, 1] * action_range.max_translation_m,
                    refinement_normalized[:, 2] * action_range.max_yaw_rad,
                ), dim=-1)
                for refinement_graph_id, scene_index in enumerate(active):
                    mask = refinement_output.furniture_graph_index == refinement_graph_id
                    actions = refinement_actions[mask]
                    proposal = refinement_delta[mask].clone()
                    move_enabled = (actions == 1) | (actions == 3)
                    rotate_enabled = (actions == 2) | (actions == 3)
                    proposal[:, :2] *= move_enabled.unsqueeze(-1)
                    proposal[:, 2] *= rotate_enabled
                    current_scene = predicted_scenes[scene_index]
                    candidate = (
                        _geometry_action_search(
                            current_scene, refinement_delta[mask]
                        )
                        if args.geometry_action_search
                        else _coordinate_model_steps(
                            current_scene, refinement_delta[mask]
                        )
                        if args.coordinate_line_search
                        else _best_model_step(current_scene, proposal)
                        if args.line_search else apply_pose_deltas(current_scene, proposal)
                    )
                    if (
                        hard_constraint_report(candidate).total
                        < hard_constraint_report(current_scene).total
                    ):
                        predicted_scenes[scene_index] = candidate
            for graph_id in range(graph.num_graphs):
                mask = output.furniture_graph_index == graph_id
                eligible = targets.eligibility[mask]
                truth = targets.action_types[mask][eligible]
                pred = prediction[mask][eligible]
                delta_truth = targets.delta_m_rad[mask][eligible]
                delta_pred = predicted_delta[mask][eligible]
                current_scene = metadata["corrupted_scenes"][graph_id]
                target_scene = apply_pose_deltas(current_scene, targets.delta_m_rad[mask])
                predicted_scene = predicted_scenes[graph_id]
                neural_raw_scene = neural_raw_scenes[graph_id]
                target_report = hard_constraint_report(target_scene)
                neural_raw_report = hard_constraint_report(neural_raw_scene)
                predicted_report = hard_constraint_report(predicted_scene)
                projected_report = joint_translation_repair(predicted_scene).report
                before_report = hard_constraint_report(current_scene)
                k = int((truth != 0).sum())
                for name in ("overall", _group(k)):
                    value = metrics[name]
                    value["objects"] += int(truth.numel())
                    value["correct"] += int((truth == pred).sum())
                    value["scenes"] += 1
                    value["whole_correct"] += int(bool((truth == pred).all()))
                    value["confusion"] += torch.bincount(
                        (truth * 4 + pred).cpu(), minlength=16
                    ).reshape(4, 4)
                    translation = (truth == 1) | (truth == 3)
                    rotation = (truth == 2) | (truth == 3)
                    value["translation_sum"] += float(torch.linalg.vector_norm(
                        delta_pred[translation, :2] - delta_truth[translation, :2], dim=-1
                    ).sum())
                    value["translation_count"] += int(translation.sum())
                    yaw_error = torch.atan2(
                        torch.sin(delta_pred[rotation, 2] - delta_truth[rotation, 2]),
                        torch.cos(delta_pred[rotation, 2] - delta_truth[rotation, 2]),
                    ).abs()
                    value["yaw_sum"] += float(yaw_error.sum())
                    value["yaw_count"] += int(rotation.sum())
                    value["target_geometry_clear"] += int(target_report.total == 0)
                    value["neural_raw_geometry_clear"] += int(neural_raw_report.total == 0)
                    value["prediction_geometry_clear"] += int(predicted_report.total == 0)
                    value["projected_geometry_clear"] += int(projected_report.total == 0)
                    value["violations_before"] += before_report.total
                    value["violations_neural_raw"] += neural_raw_report.total
                    value["violations_prediction"] += predicted_report.total
                    value["violations_projected"] += projected_report.total
                    if complete_output is not None:
                        iterations = complete_output.refinement_iterations[graph_id]
                        value["refinement_iterations_sum"] += iterations
                        value["refinement_iterations_max"] = max(
                            value["refinement_iterations_max"], iterations
                        )
                        value["refinement_nonconverged"] += int(
                            not complete_output.refinement_converged[graph_id]
                        )
                        final_delta = complete_output.final_delta_m_rad[mask]
                        value["max_final_translation_m"] = max(
                            value["max_final_translation_m"],
                            float(torch.linalg.vector_norm(
                                final_delta[:, :2], dim=-1
                            ).max()),
                        )
    report = {
        "schema_version": "scene_repair_v3_furniture_evaluation_v1",
        "checkpoint": str(args.checkpoint), "checkpoint_extra": checkpoint_extra,
        "split": args.split, "seed": args.seed,
        "samples_per_scene": args.samples_per_scene,
        "model_passes": args.model_passes,
        "line_search": args.line_search,
        "coordinate_line_search": args.coordinate_line_search,
        "geometry_action_search": args.geometry_action_search,
        "constraint_aware_model": args.constraint_aware_model,
        "inference_contract": model.inference_contract() if args.constraint_aware_model else None,
        "legacy_constraint_gate": args.legacy_constraint_gate,
        "metrics": {name: _finish(metrics[name]) for name in ("overall", "0", "1", "2-3", "4-5", "6-7+")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
