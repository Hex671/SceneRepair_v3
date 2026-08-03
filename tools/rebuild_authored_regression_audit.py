from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def edge_set(scene: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (str(edge["source_id"]), str(edge["target_id"]))
        for edge in scene.get("functional_partners", [])
    }


def run(root: Path, baseline_path: Path, scene_ids: list[str]) -> None:
    baseline = load(baseline_path)
    baseline_by_id = {row["scene_id"]: row for row in baseline["scenes"]}
    scenes: list[dict[str, Any]] = []
    relation_changes: list[dict[str, Any]] = []
    all_negative_pairs: list[dict[str, Any]] = []
    vocabulary_coverage: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        scene = load(root / "scenes" / f"{scene_id}.json")
        validation = load(root / "validation" / f"{scene_id}.json")
        table = validation["geometry_review_tables"]["pairwise_signed_separation_table"]
        negatives = [row for row in table if row["signed_separation_m"] < 0.0]
        all_negative_pairs.extend(
            {"scene_id": scene_id, **row} for row in negatives
        )
        current_edges = edge_set(scene)
        baseline_edges = {
            tuple(edge) for edge in baseline_by_id.get(scene_id, {}).get("functional_partner_edges", [])
        }
        added = sorted(current_edges - baseline_edges)
        removed = sorted(baseline_edges - current_edges)
        if added or removed:
            relation_changes.append(
                {
                    "scene_id": scene_id,
                    "baseline_edge_count": len(baseline_edges),
                    "current_edge_count": len(current_edges),
                    "added_edges": [list(edge) for edge in added],
                    "removed_edges": [list(edge) for edge in removed],
                }
            )
        completeness = validation["functional_partner_completeness"]
        vocabulary_coverage.append(
            {
                "scene_id": scene_id,
                "passed": completeness["passed"],
                "expected_edge_count": completeness["expected_edge_count"],
                "actual_edge_count": completeness["actual_edge_count"],
                "missing_edges": completeness["missing_edges"],
                "compatibility_matches": completeness[
                    "frozen_compatibility_matches"
                ],
            }
        )
        scenes.append(
            {
                "scene_id": scene_id,
                "furniture_count": len(scene["furniture"]),
                "functional_partner_edge_count": len(current_edges),
                "minimum_signed_separation_m": min(
                    row["signed_separation_m"] for row in table
                ) if table else None,
                "negative_pair_count": len(negatives),
                "hard_pass": validation["passed"],
                "functional_partner_coverage_valid": completeness["passed"],
            }
        )
    report = {
        "schema_version": "authored_dataset_regression_audit_v1",
        "baseline": str(baseline_path),
        "scene_ids": scene_ids,
        "scenes": scenes,
        "all_negative_pairs": all_negative_pairs,
        "negative_pair_count": len(all_negative_pairs),
        "relation_changes": relation_changes,
        "functional_partner_vocabulary_coverage": vocabulary_coverage,
        "strict_acceptance_min_signed_separation_m": 0.0,
        "numeric_separation_epsilon_m": 1e-9,
    }
    out = root / "validation" / "regression_20260728_final_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--scene-id", action="append", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(Path(args.output_root), Path(args.baseline), args.scene_id)
