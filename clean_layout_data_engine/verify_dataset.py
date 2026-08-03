"""Verify every accepted clean scene and its immutable audit evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from clean_layout_data_engine.storage import atomic_write_json, load_json, sha256_file, utc_now


REQUIRED_CHECKS = {
    "schema_valid",
    "hssd_metadata_valid",
    "bbox_valid",
    "geometry_hard_valid",
    "room_bounds_valid",
    "collision_valid",
    "opening_clearance_valid",
    "path_reachability_valid",
    "use_clearance_valid",
    "frozen_functional_partner_rules_used",
    "no_handwritten_functional_edges",
    "all_model_input_fields_available",
    "functional_partner_coverage_valid",
    "diversity_novelty_pass",
}


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"manifest line {line_number} is not an object")
        rows.append(value)
    return rows


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def verify(output_root: Path, *, minimum_scenes: int = 1) -> dict[str, Any]:
    if minimum_scenes < 1:
        raise ValueError("minimum_scenes must be at least 1")
    manifest_path = output_root / "accepted" / "manifest.jsonl"
    scenes_dir = output_root / "accepted" / "scenes"
    rows = _manifest_rows(manifest_path)
    errors: list[dict[str, Any]] = []
    if len(rows) < minimum_scenes:
        errors.append(
            {
                "type": "minimum_scene_count_not_met",
                "required": minimum_scenes,
                "actual": len(rows),
            }
        )
    seen: set[str] = set()
    verified: list[dict[str, Any]] = []
    for row in rows:
        scene_id = str(row.get("scene_id", ""))
        if not scene_id or scene_id in seen:
            errors.append({"scene_id": scene_id, "type": "duplicate_or_empty_scene_id"})
            continue
        seen.add(scene_id)
        paths = {
            key: Path(str(row.get(key, "")))
            for key in (
                "scene_path",
                "validation_path",
                "semantic_review_path",
                "diversity_report_path",
            )
        }
        scene_errors: list[str] = []
        for key, path in paths.items():
            if not _inside(output_root, path):
                scene_errors.append(f"{key} is outside production root")
            elif not path.is_file():
                scene_errors.append(f"{key} is missing")
        if not scene_errors:
            scene = load_json(paths["scene_path"])
            validation = load_json(paths["validation_path"])
            review = load_json(paths["semantic_review_path"])
            diversity = load_json(paths["diversity_report_path"])
            if scene.get("scene_id") != scene_id:
                scene_errors.append("scene_id mismatch in scene file")
            if sha256_file(paths["scene_path"]) != row.get("scene_sha256"):
                scene_errors.append("scene SHA-256 mismatch")
            checks = validation.get("checks", {})
            missing_checks = sorted(REQUIRED_CHECKS - set(checks))
            failed_checks = sorted(
                key for key in REQUIRED_CHECKS if key in checks and not checks[key]
            )
            if not validation.get("passed") or missing_checks or failed_checks:
                scene_errors.append(
                    f"hard validation incomplete: missing={missing_checks}, failed={failed_checks}"
                )
            if not review.get("passed") or review.get("scene_id") != scene_id:
                scene_errors.append("semantic review did not pass")
            if not diversity.get("novelty_pass") or diversity.get("scene_id") != scene_id:
                scene_errors.append("diversity review did not pass")
        if scene_errors:
            errors.append(
                {"scene_id": scene_id, "type": "evidence_failure", "details": scene_errors}
            )
        else:
            verified.append(
                {
                    "scene_id": scene_id,
                    "room_type": row.get("room_type"),
                    "scene_sha256": row["scene_sha256"],
                }
            )

    files = {path.stem for path in scenes_dir.glob("*.json")} if scenes_dir.exists() else set()
    unmanifested = sorted(files - seen)
    missing_scene_files = sorted(seen - files)
    if unmanifested:
        errors.append({"type": "unmanifested_scene_files", "scene_ids": unmanifested})
    if missing_scene_files:
        errors.append({"type": "manifest_scenes_missing", "scene_ids": missing_scene_files})
    return {
        "schema_version": "clean_layout_dataset_integrity_report_v1",
        "passed": not errors,
        "manifest_scene_count": len(rows),
        "minimum_scene_count": minimum_scenes,
        "verified_scene_count": len(verified),
        "room_type_counts": {
            room_type: sum(value["room_type"] == room_type for value in verified)
            for room_type in ("bedroom", "living_room", "dining_room", "home_office")
        },
        "errors": errors,
        "verified_scenes": verified,
        "verified_at": utc_now(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--minimum-scenes", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        report = verify(args.output_root.resolve(), minimum_scenes=args.minimum_scenes)
        report_path = args.report or args.output_root / "audits" / "dataset_integrity.json"
        atomic_write_json(report_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Dataset verifier error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
