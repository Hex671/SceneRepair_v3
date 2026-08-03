from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scene_repair_v3 import (
    ActionRange,
    FunctionalLayoutEvaluator,
    FunctionalSceneContract,
    FurnitureRepairNetwork,
    FurnitureVocabularies,
    FrozenFunctionalPartnerRules,
    ModelConfig,
    SFURRules,
    apply_pose_deltas,
)
from scene_repair_v3.checkpoint import load_checkpoint
from scene_repair_v3.data import CorruptionConfig, load_scene_json
from scene_repair_v3.dataset import FurnitureCorruptionDataset, make_collate_fn


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "scenes": 0,
        "passed": 0,
        "hard_passed": 0,
        "functional_score_sum": 0.0,
        "relation_rate_sum": 0.0,
        "clearance_rate_sum": 0.0,
        "reachability_rate_sum": 0.0,
        "relation_checks": 0,
        "clearance_checks": 0,
        "path_checks": 0,
        "failure_kinds": Counter(),
    }


def _add(bucket: dict[str, object], report) -> None:
    bucket["scenes"] += 1
    bucket["passed"] += int(report.scene_pass)
    bucket["hard_passed"] += int(report.hard_constraints_pass)
    bucket["functional_score_sum"] += report.functional_score
    bucket["relation_rate_sum"] += report.relation_pass_rate
    bucket["clearance_rate_sum"] += report.clearance_pass_rate
    bucket["reachability_rate_sum"] += report.reachability_rate
    bucket["relation_checks"] += report.relation_checks
    bucket["clearance_checks"] += report.clearance_checks
    bucket["path_checks"] += report.path_checks
    bucket["failure_kinds"].update(value.kind for value in report.violations)


def _finish(bucket: dict[str, object]) -> dict[str, object]:
    scenes = max(int(bucket["scenes"]), 1)
    return {
        "scenes": bucket["scenes"],
        "scene_functional_passes": bucket["passed"],
        "sfur": bucket["passed"] / scenes,
        "hard_constraint_scene_pass_rate": bucket["hard_passed"] / scenes,
        "mean_functional_score": bucket["functional_score_sum"] / scenes,
        "mean_functional_relation_pass_rate": bucket["relation_rate_sum"] / scenes,
        "mean_interaction_clearance_pass_rate": bucket["clearance_rate_sum"] / scenes,
        "mean_core_reachability": bucket["reachability_rate_sum"] / scenes,
        "relation_checks": bucket["relation_checks"],
        "clearance_checks": bucket["clearance_checks"],
        "path_checks": bucket["path_checks"],
        "failure_kinds": dict(bucket["failure_kinds"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Scene Functional Usability Rate")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--functional-rules", type=Path, required=True)
    parser.add_argument("--compatibility-rules", type=Path, required=True)
    parser.add_argument("--sfur-rules", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples-per-scene", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--max-translation-m", type=float, default=3.0)
    parser.add_argument("--model-passes", type=int, default=1)
    parser.add_argument("--functional-refinement", action="store_true")
    parser.add_argument(
        "--action-policy",
        choices=("hard_constraint_gated", "semantic", "dense_delta"),
        default="hard_constraint_gated",
    )
    args = parser.parse_args()
    if args.model_passes <= 0:
        parser.error("--model-passes must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = FurnitureVocabularies.load(args.vocab)
    partner_rules = FrozenFunctionalPartnerRules.load(
        args.functional_rules, args.compatibility_rules
    )
    dataset = FurnitureCorruptionDataset(
        args.manifest,
        split=args.split,
        vocabularies=vocab,
        rules=partner_rules,
        base_seed=args.seed,
        samples_per_scene=args.samples_per_scene,
        corruption=CorruptionConfig(label_mode="semantic_restore_v1"),
    )
    dataset.set_epoch(0)
    contracts = {
        str(raw["scene_id"]): FunctionalSceneContract.from_raw_scene(raw)
        for raw in (
            load_scene_json(dataset.scene_root / row["path"]) for row in dataset.rows
        )
    }
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=make_collate_fn(),
    )
    action_range = ActionRange(args.max_translation_m, math.pi)
    model = FurnitureRepairNetwork(vocab, ModelConfig()).to(device)
    checkpoint_extra = load_checkpoint(
        args.checkpoint, model, action_range, map_location=device
    )
    model.eval()
    evaluator = FunctionalLayoutEvaluator(SFURRules.load(args.sfur_rules))
    names = ("corrupted", "clean_reference", "neural", "complete_model")
    metrics = {
        name: defaultdict(_empty)
        for name in names
    }
    runtime_audit = {
        "evaluated_scene_count": 0,
        "max_cumulative_translation_m": 0.0,
        "max_abs_cumulative_yaw_rad": 0.0,
        "functional_refinement_iteration_sum": 0,
        "functional_refinement_iteration_max": 0,
        "functional_refinement_unconverged_scenes": 0,
        "residual_hard_constraint_scenes": 0,
        "residual_hard_constraint_count": 0,
    }
    with torch.no_grad():
        for graph, targets, metadata in loader:
            graph = graph.to(device)
            origin_scenes = tuple(metadata["corrupted_scenes"])
            batch_contracts = tuple(contracts[value] for value in metadata["scene_ids"])
            complete = model.repair_batch(
                graph,
                origin_scenes,
                action_range,
                action_policy=args.action_policy,
                semantic_passes=args.model_passes,
                functional_contracts=(
                    batch_contracts if args.functional_refinement else None
                ),
                functional_origin_scenes=(
                    origin_scenes if args.functional_refinement else None
                ),
                sfur_rules=(evaluator.rules if args.functional_refinement else None),
            )
            current_scenes = complete.scenes
            final_delta = complete.final_delta_m_rad.detach().float().cpu()
            if final_delta.numel():
                runtime_audit["max_cumulative_translation_m"] = max(
                    runtime_audit["max_cumulative_translation_m"],
                    float(torch.linalg.vector_norm(final_delta[:, :2], dim=-1).max()),
                )
                runtime_audit["max_abs_cumulative_yaw_rad"] = max(
                    runtime_audit["max_abs_cumulative_yaw_rad"],
                    float(final_delta[:, 2].abs().max()),
                )
            iterations = complete.functional_refinement_iterations
            converged = complete.functional_refinement_converged
            runtime_audit["evaluated_scene_count"] += len(current_scenes)
            runtime_audit["functional_refinement_iteration_sum"] += sum(iterations)
            runtime_audit["functional_refinement_iteration_max"] = max(
                runtime_audit["functional_refinement_iteration_max"],
                max(iterations, default=0),
            )
            runtime_audit["functional_refinement_unconverged_scenes"] += sum(
                not value for value in converged
            )
            residual_totals = [value.total for value in complete.reports]
            runtime_audit["residual_hard_constraint_scenes"] += sum(
                value > 0 for value in residual_totals
            )
            runtime_audit["residual_hard_constraint_count"] += sum(residual_totals)
            for graph_id, scene_id in enumerate(metadata["scene_ids"]):
                mask = complete.neural_output.furniture_graph_index == graph_id
                corrupted = metadata["corrupted_scenes"][graph_id]
                clean_reference = apply_pose_deltas(
                    corrupted, targets.delta_m_rad[mask.cpu()]
                )
                scenes = {
                    "corrupted": corrupted,
                    "clean_reference": clean_reference,
                    "neural": complete.neural_scenes[graph_id],
                    "complete_model": current_scenes[graph_id],
                }
                k = int(metadata["sampled_changed_counts"][graph_id])
                group = _group(k)
                contract = contracts[scene_id]
                for name, candidate in scenes.items():
                    report = evaluator.evaluate(candidate, contract)
                    _add(metrics[name]["overall"], report)
                    _add(metrics[name][group], report)

    evaluated_scenes = max(int(runtime_audit["evaluated_scene_count"]), 1)
    runtime_audit["mean_functional_refinement_iterations"] = (
        runtime_audit.pop("functional_refinement_iteration_sum") / evaluated_scenes
    )
    runtime_audit["all_actions_within_range"] = (
        runtime_audit["max_cumulative_translation_m"]
        <= args.max_translation_m + 1.0e-6
        and runtime_audit["max_abs_cumulative_yaw_rad"] <= math.pi + 1.0e-6
    )
    report = {
        "schema_version": "scene_repair_v3_sfur_evaluation_v2",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _file_sha256(args.checkpoint),
        "checkpoint_extra": checkpoint_extra,
        "split": args.split,
        "seed": args.seed,
        "samples_per_scene": args.samples_per_scene,
        "action_policy": args.action_policy,
        "model_passes": args.model_passes,
        "functional_refinement": args.functional_refinement,
        "sfur_rule_schema": "scene_repair_v3_sfur_rules_v1",
        "sfur_rules_sha256": _file_sha256(args.sfur_rules),
        "runtime_audit": runtime_audit,
        "metrics": {
            name: {
                group: _finish(metrics[name][group])
                for group in ("overall", "0", "1", "2-3", "4-5", "6-7+")
            }
            for name in names
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
