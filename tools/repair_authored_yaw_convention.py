"""Repair authored layouts that used local +Y instead of the HSSD local -Y front."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any


REVISION_OLD = "rev_000_pre_yaw_convention_correction"
REVISION_NEW = "rev_001_local_minus_y_yaw_corrected"

# Each selected object has an explicit authored rationale and clearance zone that
# identify the intended front.  The incorrect values are exactly pi radians away.
REPAIRS: dict[str, dict[str, float]] = {
    "bedroom_authored_002": {
        "nightstand_sleep_west": 0.0,
        "wardrobe_clothes_east": -math.pi * 0.5,
        "cabinet_dressing_south": -math.pi,
        "armchair_daylight_south_east": 0.8 - math.pi,
    },
    "dining_room_authored_003": {
        "bench_east_diners": -math.pi * 0.5,
        "chair_west_diner": math.pi * 0.5,
        "cabinet_dishes_west": math.pi * 0.5,
        "console_serving_south": -math.pi,
    },
    "home_office_authored_002": {
        "desk_work_east": -math.pi * 0.5,
        "chair_work_east": math.pi * 0.5,
        "bookcase_reference_north": 0.0,
        "cabinet_files_south_east": -math.pi,
        "armchair_window_reading": -2.35 + math.pi,
    },
    "living_room_authored_002": {
        "sofa_conversation_north": 0.0,
        "armchair_conversation_east": 0.8 - math.pi,
        "console_serving_south_east": -math.pi,
    },
}


def _wrap_yaw(value: float) -> float:
    value = (value + math.pi) % (2.0 * math.pi) - math.pi
    return -math.pi if abs(value - math.pi) < 1.0e-12 else value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _front(yaw: float) -> list[float]:
    return [round(math.sin(yaw), 6), round(-math.cos(yaw), 6)]


def _artifact_paths(root: Path, scene_id: str) -> dict[str, Path]:
    return {
        "authored_proposal": root / "authored_proposals" / f"{scene_id}.json",
        "scene": root / "scenes" / f"{scene_id}.json",
        "validation": root / "validation" / f"{scene_id}.json",
        "diversity": root / "diversity" / f"{scene_id}.json",
        "diversity_legacy": root / "validation" / f"{scene_id}_diversity.json",
        "functional_partner_build": (
            root / "validation" / f"functional_partner_build_{scene_id}.json"
        ),
        "visualization": root / "visualizations" / f"{scene_id}.png",
        "independent_review": root / "reviews" / f"{scene_id}_independent.json",
        "reviewer_scene": root / "reviews" / "reviewer_packet" / f"{scene_id}_scene.json",
        "reviewer_geometry": (
            root / "reviews" / "reviewer_packet" / f"{scene_id}_geometry.json"
        ),
    }


def _copy_artifacts(root: Path, scene_id: str, destination: Path) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    for name, source in _artifact_paths(root, scene_id).items():
        if not source.exists():
            continue
        target = destination / source.name
        shutil.copy2(source, target)
        hashes[name] = _sha256(source)
    return hashes


def _correction_rows(scene_id: str, proposal: dict[str, Any]) -> list[dict[str, Any]]:
    new_yaws = REPAIRS[scene_id]
    authored = {item["object_id"]: item for item in proposal["furniture"]}
    rows: list[dict[str, Any]] = []
    for object_id, new_yaw in new_yaws.items():
        item = authored[object_id]
        old_yaw = float(item["yaw_rad"])
        if abs(abs(_wrap_yaw(new_yaw - old_yaw)) - math.pi) > 1.0e-6:
            raise ValueError(
                f"{scene_id}/{object_id} is not an exact pi yaw correction: "
                f"old={old_yaw}, new={new_yaw}"
            )
        rows.append(
            {
                "object_id": object_id,
                "old_yaw_rad": old_yaw,
                "new_yaw_rad": _wrap_yaw(new_yaw),
                "old_local_minus_y_front_xy": _front(old_yaw),
                "new_local_minus_y_front_xy": _front(_wrap_yaw(new_yaw)),
            }
        )
    return rows


def _write_revision(
    path: Path,
    *,
    scene_id: str,
    revision_id: str,
    status: str,
    reason: str,
    rows: list[dict[str, Any]],
    hashes: dict[str, str],
    supersedes: str | None = None,
    superseded_by: str | None = None,
) -> None:
    value: dict[str, Any] = {
        "schema_version": "authored_scene_revision_record_v1",
        "scene_id": scene_id,
        "revision_id": revision_id,
        "status": status,
        "reason": reason,
        "yaw_corrections": rows,
        "artifact_sha256": hashes,
    }
    if supersedes is not None:
        value["supersedes"] = supersedes
    if superseded_by is not None:
        value["superseded_by"] = superseded_by
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def archive_before(root: Path) -> None:
    for scene_id in REPAIRS:
        proposal_path = root / "authored_proposals" / f"{scene_id}.json"
        proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
        destination = root / "revisions" / scene_id / REVISION_OLD
        hashes = _copy_artifacts(root, scene_id, destination)
        _write_revision(
            destination / "revision.json",
            scene_id=scene_id,
            revision_id=REVISION_OLD,
            status="superseded_yaw_semantic_error",
            reason=(
                "The proposal used local +Y as the furniture front although the "
                "dataset contract defines HSSD local -Y as the front."
            ),
            rows=_correction_rows(scene_id, proposal),
            hashes=hashes,
            superseded_by=REVISION_NEW,
        )


def apply_repairs(root: Path) -> None:
    audit: dict[str, Any] = {
        "schema_version": "authored_yaw_convention_repair_audit_v1",
        "coordinate_contract": "HSSD local -Y is the semantic front",
        "scenes": [],
    }
    for scene_id, new_yaws in REPAIRS.items():
        proposal_path = root / "authored_proposals" / f"{scene_id}.json"
        proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
        rows = _correction_rows(scene_id, proposal)
        by_id = {item["object_id"]: item for item in proposal["furniture"]}
        for object_id, new_yaw in new_yaws.items():
            by_id[object_id]["yaw_rad"] = _wrap_yaw(new_yaw)
        for zone in proposal["use_clearance_zones"]:
            new_yaw = new_yaws.get(zone["owner_id"])
            if new_yaw is not None:
                zone["yaw_rad"] = _wrap_yaw(new_yaw)
        proposal_path.write_text(
            json.dumps(proposal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        audit["scenes"].append({"scene_id": scene_id, "yaw_corrections": rows})
    (root / "validation" / "yaw_convention_repair_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_pending_review(root: Path, scene_id: str) -> None:
    scene = json.loads(
        (root / "scenes" / f"{scene_id}.json").read_text(encoding="utf-8")
    )
    review = {
        "schema_version": "authored_semantic_review_pending_v1",
        "scene_id": scene_id,
        "semantic_review_status": "pending_independent_review",
        "independent_review_completed": False,
        "independent_review_pass": False,
        "scene_layout_hash": scene["audit"]["layout_hash"],
        "reason": (
            "The local -Y yaw convention correction changed rendered furniture "
            "front directions. The previous review is archived and cannot attest "
            "to the corrected artifacts."
        ),
        "isolation_packet": [
            f"reviews/reviewer_packet/{scene_id}_scene.json",
            f"reviews/reviewer_packet/{scene_id}_geometry.json",
            f"visualizations/{scene_id}.png",
            f"validation/{scene_id}.json",
            f"diversity/{scene_id}.json",
        ],
        "final_verdict": "pending_independent_review",
    }
    payload = json.dumps(review, ensure_ascii=False, indent=2) + "\n"
    (root / "reviews" / f"{scene_id}_independent.json").write_text(
        payload, encoding="utf-8"
    )
    if scene_id == "living_room_authored_002":
        (root / "reviews" / f"{scene_id}_pending_independent_review.json").write_text(
            payload, encoding="utf-8"
        )


def finalize(root: Path) -> None:
    audit = json.loads(
        (root / "validation" / "yaw_convention_repair_audit.json").read_text(
            encoding="utf-8"
        )
    )
    rows_by_scene = {
        value["scene_id"]: value["yaw_corrections"] for value in audit["scenes"]
    }
    for scene_id in REPAIRS:
        _write_pending_review(root, scene_id)
        destination = root / "revisions" / scene_id / REVISION_NEW
        hashes = _copy_artifacts(root, scene_id, destination)
        _write_revision(
            destination / "revision.json",
            scene_id=scene_id,
            revision_id=REVISION_NEW,
            status="corrected_pending_independent_review",
            reason=(
                "Yaw values and their use-clearance-zone orientations now follow "
                "the HSSD local -Y front convention; fresh independent review is "
                "required because the rendered arrows changed."
            ),
            rows=rows_by_scene[scene_id],
            hashes=hashes,
            supersedes=REVISION_OLD,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--stage", choices=("archive-before", "apply", "finalize"), required=True
    )
    args = parser.parse_args()
    if args.stage == "archive-before":
        archive_before(args.data_root)
    elif args.stage == "apply":
        apply_repairs(args.data_root)
    else:
        finalize(args.data_root)


if __name__ == "__main__":
    main()
