from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scene_repair_v3 import (
    ActionRange,
    ActionType,
    FunctionalLayoutEvaluator,
    FunctionalSceneContract,
    FurnitureRepairNetwork,
    FurnitureVocabularies,
    FrozenFunctionalPartnerRules,
    ModelConfig,
    SFURRules,
    apply_pose_deltas,
    decode_actions,
)
from scene_repair_v3.checkpoint import load_checkpoint
from scene_repair_v3.data import CorruptionConfig, load_scene_json
from scene_repair_v3.dataset import FurnitureCorruptionDataset, make_collate_fn
from scene_repair_v3.geometry import (
    OrientedRectangle,
    room_signed_margins,
    signed_separation_and_escape,
    wrap_yaw,
)
from scene_repair_v3.model import _furniture_constraint_summary


ACTION_NAMES = ("KEEP", "TRANSLATE", "ROTATE", "BOTH")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _k_group(k: int) -> str:
    if k <= 1:
        return str(k)
    if k <= 3:
        return "2-3"
    if k <= 5:
        return "4-5"
    return "6+"


def _pose(item: dict, delta: torch.Tensor | None = None) -> dict:
    dx = dy = dyaw = 0.0
    if delta is not None:
        dx, dy, dyaw = (float(value) for value in delta)
    return {
        "x": item["x"] + dx,
        "y": item["y"] + dy,
        "yaw": wrap_yaw(item["yaw"] + dyaw),
        "width": item["width"],
        "depth": item["depth"],
        "height": item["height"],
    }


def _hard_violations(room: dict, openings: list[dict], furniture: list[dict]) -> dict:
    rectangles = [
        OrientedRectangle(v["x"], v["y"], v["width"], v["depth"], v["yaw"])
        for v in furniture
    ]
    boundary = sum(
        min(room_signed_margins(rect, room["length"], room["width"])) < -1.0e-6
        for rect in rectangles
    )
    collision = 0
    for left in range(len(rectangles)):
        for right in range(left + 1, len(rectangles)):
            separation, _, _ = signed_separation_and_escape(
                rectangles[left], rectangles[right]
            )
            collision += separation < -1.0e-6
    opening_count = 0
    for opening in openings:
        opening_rect = OrientedRectangle(
            opening["x"], opening["y"], opening["width"], opening["depth"], 0.0
        )
        low_z = opening["z"] - opening["height"] * 0.5
        high_z = opening["z"] + opening["height"] * 0.5
        for item, rect in zip(furniture, rectangles):
            xy_separation, _, _ = signed_separation_and_escape(opening_rect, rect)
            z_separation = max(low_z - item["height"], -high_z)
            opening_count += max(xy_separation, z_separation) < -1.0e-6
    return {
        "boundary": int(boundary),
        "collision": int(collision),
        "opening": int(opening_count),
        "total": int(boundary + collision + opening_count),
    }


def _violation_tags(value: dict) -> list[str]:
    return [name for name in ("boundary", "collision", "opening") if value[name]] or ["clean"]


def _record_graphs(
    graph,
    targets,
    output,
    decoded,
    metadata,
    vocab,
    action_range: ActionRange,
    *,
    complete_output=None,
    evaluator: FunctionalLayoutEvaluator | None = None,
    contracts: list[FunctionalSceneContract] | None = None,
) -> list[dict]:
    normalized = output.primary_delta_normalized
    raw_delta = torch.stack(
        [
            normalized[:, 0] * action_range.max_translation_m,
            normalized[:, 1] * action_range.max_translation_m,
            normalized[:, 2] * action_range.max_yaw_rad,
        ],
        dim=-1,
    )
    summary = _furniture_constraint_summary(graph)
    records: list[dict] = []
    for graph_id in range(graph.num_graphs):
        furniture_mask = output.furniture_graph_index == graph_id
        furniture_rows = furniture_mask.nonzero(as_tuple=False).flatten()
        projected_scene = (
            complete_output.scenes[graph_id]
            if complete_output is not None
            else None
        )
        room = {
            "type": vocab.room_types.values[int(graph.room_type_ids[graph_id])],
            "length": float(graph.room_features[graph_id, 0]),
            "width": float(graph.room_features[graph_id, 1]),
        }
        opening_node_mask = graph.graph_index[graph.opening_indices] == graph_id
        opening_rows = opening_node_mask.nonzero(as_tuple=False).flatten()
        openings: list[dict] = []
        for row in opening_rows.tolist():
            node_index = int(graph.opening_indices[row])
            values = graph.opening_geometry[row]
            room_scale = max(room["length"], room["width"])
            openings.append(
                {
                    "id": graph.object_ids[node_index],
                    "kind": vocab.opening_kinds.values[int(graph.opening_kind_ids[row])],
                    "x": float(values[0]) * room["length"],
                    "y": float(values[1]) * room["width"],
                    "z": float(values[2]) * room_scale,
                    "width": float(values[3]) * room["length"],
                    "depth": float(values[4]) * room["width"],
                    "height": float(values[5]) * room_scale,
                }
            )

        furniture: list[dict] = []
        sampled_actions = metadata["sampled_action_types"][graph_id]
        for local_index, row in enumerate(furniture_rows.tolist()):
            node_index = int(graph.furniture_indices[row])
            values = graph.furniture_geometry[row]
            room_scale = max(room["length"], room["width"])
            target_action = int(targets.action_types[row])
            if projected_scene is None:
                final_delta = decoded.delta_m_rad[row]
                predicted_action = int(decoded.action_types[row])
            else:
                source_item = metadata["corrupted_scenes"][graph_id].furniture[
                    len(furniture)
                ]
                final_item = projected_scene.furniture[len(furniture)]
                final_delta = torch.tensor([
                    final_item.x_m - source_item.x_m,
                    final_item.y_m - source_item.y_m,
                    wrap_yaw(final_item.yaw_rad - source_item.yaw_rad),
                ])
                has_translation = float(torch.linalg.vector_norm(final_delta[:2])) > 1.0e-6
                has_rotation = abs(float(final_delta[2])) > 1.0e-6
                predicted_action = (
                    3 if has_translation and has_rotation else
                    1 if has_translation else
                    2 if has_rotation else 0
                )
            item = {
                "id": graph.object_ids[node_index],
                "category": vocab.categories.values[int(graph.furniture_category_ids[row])],
                "x": float(values[0]) * room["length"],
                "y": float(values[1]) * room["width"],
                "yaw": math.atan2(float(values[2]), float(values[3])),
                "width": float(values[4]) * room["length"],
                "depth": float(values[5]) * room["width"],
                "height": float(values[6]) * room_scale,
                "target_action": ACTION_NAMES[target_action],
                "corruption_action": ACTION_NAMES[int(sampled_actions[local_index])],
                "predicted_action": ACTION_NAMES[predicted_action],
                "action_correct": target_action == predicted_action,
                "target_delta": [round(float(v), 5) for v in targets.delta_m_rad[row]],
                "predicted_delta": [round(float(v), 5) for v in final_delta],
                "raw_delta": [round(float(v), 5) for v in raw_delta[row]],
                "constraint": [round(float(v), 6) for v in summary[row]],
            }
            furniture.append(item)
        current_poses = [_pose(item) for item in furniture]
        predicted_poses = (
            [_pose(item, decoded.delta_m_rad[row]) for item, row in zip(furniture, furniture_rows.tolist())]
            if projected_scene is None
            else [
                {
                    "x": value.x_m, "y": value.y_m, "yaw": value.yaw_rad,
                    "width": value.bbox_width_m, "depth": value.bbox_depth_m,
                    "height": value.bbox_height_m,
                }
                for value in projected_scene.furniture
            ]
        )
        target_poses = [
            _pose(item, targets.delta_m_rad[row])
            for item, row in zip(furniture, furniture_rows.tolist())
        ]
        before = _hard_violations(room, openings, current_poses)
        after = _hard_violations(room, openings, predicted_poses)
        target_after = _hard_violations(room, openings, target_poses)
        sfur_before = sfur_after = sfur_target = None
        if (
            evaluator is not None
            and contracts is not None
            and projected_scene is not None
        ):
            contract = contracts[graph_id]
            corrupted_scene = metadata["corrupted_scenes"][graph_id]
            target_scene = apply_pose_deltas(
                corrupted_scene, targets.delta_m_rad[furniture_mask].detach().cpu()
            )
            sfur_before = evaluator.evaluate(corrupted_scene, contract).to_dict()
            sfur_target = evaluator.evaluate(target_scene, contract).to_dict()
            sfur_after = evaluator.evaluate(projected_scene, contract).to_dict()
        k = sum(item["target_action"] != "KEEP" for item in furniture)
        all_correct = all(item["action_correct"] for item in furniture)
        has_translation_corruption = any(
            item["corruption_action"] in {"TRANSLATE", "BOTH"}
            for item in furniture
        )
        has_rotation_corruption = any(
            item["corruption_action"] in {"ROTATE", "BOTH"}
            for item in furniture
        )
        disturbance_tags = [
            name
            for name, present in (
                ("translation", has_translation_corruption),
                ("rotation", has_rotation_corruption),
            )
            if present
        ] or ["none"]
        records.append(
            {
                "scene_id": metadata["scene_ids"][graph_id],
                "seed": metadata["seeds"][graph_id],
                "room": room,
                "openings": openings,
                "furniture": furniture,
                "current_poses": current_poses,
                "predicted_poses": predicted_poses,
                "target_poses": target_poses,
                "k": k,
                "k_group": _k_group(k),
                "violations_before": before,
                "violations_after": after,
                "violations_target": target_after,
                "violation_tags": _violation_tags(before),
                "action_all_correct": all_correct,
                "geometry_improved": after["total"] < before["total"],
                "geometry_cleared": after["total"] == 0,
                "has_rotation_target": any(item["target_action"] == "ROTATE" for item in furniture),
                "has_rotation_corruption": has_rotation_corruption,
                "has_translation_corruption": has_translation_corruption,
                "disturbance_tags": disturbance_tags,
                "sfur_before": sfur_before,
                "sfur_after": sfur_after,
                "sfur_target": sfur_target,
            }
        )
    return records


def _select_diverse(records: list[dict], count: int) -> list[dict]:
    selected: list[dict] = []
    remaining = list(records)
    counters = {
        "room": Counter(),
        "k": Counter(),
        "violation": Counter(),
        "status": Counter(),
        "disturbance": Counter(),
    }
    review_limit = max(1, count // 4)
    success_limit = count - review_limit
    while remaining and len(selected) < count:
        def score(record: dict) -> tuple[float, str, int]:
            sfur_after = record.get("sfur_after") or {}
            status = (
                "success"
                if sfur_after.get("scene_functional_pass", False)
                else "review"
            )
            value = 4.0 / (1 + counters["room"][record["room"]["type"]])
            value += 3.0 / (1 + counters["k"][record["k_group"]])
            value += sum(
                2.5 / (1 + counters["violation"][tag])
                for tag in record["violation_tags"]
            )
            value += 2.0 / (1 + counters["status"][status])
            value += sum(
                2.0 / (1 + counters["disturbance"][tag])
                for tag in record["disturbance_tags"]
            )
            if record["violations_after"]["total"] < record["violations_before"]["total"]:
                value += 0.5
            return value, record["scene_id"], int(record["seed"])

        eligible = [
            record
            for record in remaining
            if (
                counters["status"]["success"] < success_limit
                if (record.get("sfur_after") or {}).get(
                    "scene_functional_pass", False
                )
                else counters["status"]["review"] < review_limit
            )
        ]
        if not eligible:
            break
        best = max(eligible, key=score)
        remaining.remove(best)
        selected.append(best)
        status = (
            "success"
            if (best.get("sfur_after") or {}).get("scene_functional_pass", False)
            else "review"
        )
        counters["room"][best["room"]["type"]] += 1
        counters["k"][best["k_group"]] += 1
        counters["status"][status] += 1
        for tag in best["violation_tags"]:
            counters["violation"][tag] += 1
        for tag in best["disturbance_tags"]:
            counters["disturbance"][tag] += 1
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Export diverse Furniture prediction examples")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--functional-rules", type=Path, required=True)
    parser.add_argument("--compatibility-rules", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sfur-rules", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=24)
    parser.add_argument("--samples-per-scene", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--label-mode",
        choices=(
            "semantic_restore_v1",
            "observable_joint_v4",
            "observable_v3",
            "semantic_v2",
            "legacy_random",
        ),
        default="observable_joint_v4",
    )
    parser.add_argument("--max-translation-m", type=float, default=3.0)
    parser.add_argument(
        "--constraint-aware-model",
        action="store_true",
        help="Export the complete model output with its internal constraint layer",
    )
    parser.add_argument("--model-passes", type=int, default=1)
    parser.add_argument(
        "--functional-refinement",
        action="store_true",
        help="Use SFUR functional refinement; requires --sfur-rules",
    )
    args = parser.parse_args()
    if args.model_passes <= 0:
        parser.error("--model-passes must be positive")
    if args.functional_refinement and args.sfur_rules is None:
        parser.error("--functional-refinement requires --sfur-rules")
    if args.functional_refinement and not args.constraint_aware_model:
        parser.error("--functional-refinement requires --constraint-aware-model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = FurnitureVocabularies.load(args.vocab)
    rules = FrozenFunctionalPartnerRules.load(args.functional_rules, args.compatibility_rules)
    dataset = FurnitureCorruptionDataset(
        args.manifest,
        split="test",
        vocabularies=vocab,
        rules=rules,
        corruption=CorruptionConfig(label_mode=args.label_mode),
        base_seed=args.seed,
        samples_per_scene=args.samples_per_scene,
    )
    contract_by_scene_id = {
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
    model = FurnitureRepairNetwork(vocab, ModelConfig()).to(device).eval()
    action_range = ActionRange(args.max_translation_m, math.pi)
    checkpoint_extra = load_checkpoint(
        args.checkpoint, model, action_range, map_location=device
    )
    evaluator = (
        FunctionalLayoutEvaluator(SFURRules.load(args.sfur_rules))
        if args.sfur_rules is not None else None
    )
    records: list[dict] = []
    with torch.no_grad():
        for graph, targets, metadata in loader:
            graph = graph.to(device)
            targets = targets.to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(graph)
            decoded = decode_actions(
                output,
                action_range,
                action_types=output.action_type_logits.argmax(dim=-1),
            )
            batch_contracts = [
                contract_by_scene_id[scene_id] for scene_id in metadata["scene_ids"]
            ]
            complete_output = (
                model.repair_batch(
                    graph,
                    metadata["corrupted_scenes"],
                    action_range,
                    action_policy=(
                        "dense_delta"
                        if args.label_mode == "semantic_restore_v1"
                        else "hard_constraint_gated"
                    ),
                    semantic_passes=args.model_passes,
                    functional_contracts=(
                        batch_contracts if args.functional_refinement else None
                    ),
                    functional_origin_scenes=(
                        metadata["corrupted_scenes"]
                        if args.functional_refinement else None
                    ),
                    sfur_rules=(
                        evaluator.rules if args.functional_refinement else None
                    ),
                )
                if args.constraint_aware_model
                else None
            )
            records.extend(
                _record_graphs(
                    graph, targets, output, decoded, metadata, vocab, action_range,
                    complete_output=complete_output,
                    evaluator=evaluator,
                    contracts=batch_contracts,
                )
            )
    selected = _select_diverse(records, args.examples)
    payload = {
        "schema_version": "scene_repair_v3_prediction_examples_v3_sfur",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_extra": checkpoint_extra,
        "test_seed": args.seed,
        "candidate_count": len(records),
        "example_count": len(selected),
        "selection": "greedy diversity over room type, K, violation type, action status and corruption components",
        "label_mode": args.label_mode,
        "constraint_aware_model": args.constraint_aware_model,
        "model_passes": args.model_passes,
        "functional_refinement": args.functional_refinement,
        "inference_contract": (
            model.inference_contract() if args.constraint_aware_model else None
        ),
        "examples": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
        encoding="utf-8",
    )
    summary = {
        "candidate_count": len(records),
        "example_count": len(selected),
        "room_types": Counter(v["room"]["type"] for v in selected),
        "k_groups": Counter(v["k_group"] for v in selected),
        "violation_tags": Counter(tag for v in selected for tag in v["violation_tags"]),
        "sfur_status": Counter(
            "success"
            if (v.get("sfur_after") or {}).get("scene_functional_pass", False)
            else "review"
            for v in selected
        ),
        "translation_corruption_examples": sum(v["has_translation_corruption"] for v in selected),
        "rotation_corruption_examples": sum(v["has_rotation_corruption"] for v in selected),
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=dict))


if __name__ == "__main__":
    main()
