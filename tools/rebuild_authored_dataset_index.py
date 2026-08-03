from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


RECORD_SCHEMA = "authored_clean_layout_manifest_record_v2"
STATUS_SCHEMA = "authored_clean_layout_dataset_status_v2"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _closest_summary(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "scene_id": value["scene_id"],
        "room_type": value["room_type"],
        "overall_similarity": value["overall_similarity"],
        "template_risk": value["template_risk"],
        "best_symmetry_transform": value["best_symmetry_transform"],
    }


def _review_passed(review: dict[str, Any]) -> bool:
    status = str(review.get("semantic_review_status", "")).strip().lower()
    verdict = str(review.get("final_verdict", "")).strip().lower()
    passing_statuses = {
        "semantic_review_pass",
        "pass",
        "complete_pass",
        "accepted",
        "approved",
    }
    passing_verdicts = {"semantic_review_pass", "pass", "complete_pass", "accept", "accepted"}
    return verdict in passing_verdicts and (
        not status or status in passing_statuses
    )


def _artifact_paths(scene_id: str) -> dict[str, str]:
    return {
        "proposal": f"authored_proposals/{scene_id}.json",
        "scene": f"scenes/{scene_id}.json",
        "validation": f"validation/{scene_id}.json",
        "diversity": f"diversity/{scene_id}.json",
        "visualization": f"visualizations/{scene_id}.png",
        "independent_review": f"reviews/{scene_id}_independent.json",
        "reviewer_geometry": f"reviews/reviewer_packet/{scene_id}_geometry.json",
        "functional_partner_build": (
            f"validation/functional_partner_build_{scene_id}.json"
        ),
    }


def build_record(root: Path, scene_id: str) -> dict[str, Any]:
    artifacts = _artifact_paths(scene_id)
    artifact_paths = {key: root / value for key, value in artifacts.items()}
    missing = [str(path) for path in artifact_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required artifacts for {scene_id}: {missing}")
    proposal = _load(artifact_paths["proposal"])
    scene = _load(artifact_paths["scene"])
    validation = _load(artifact_paths["validation"])
    diversity = _load(artifact_paths["diversity"])
    review = _load(artifact_paths["independent_review"])
    functional_build = _load(artifact_paths["functional_partner_build"])
    if proposal.get("functional_partners") is not None:
        raise ValueError(f"{scene_id} proposal contains handwritten relations")
    if functional_build.get("handwritten_edges") is not False:
        raise ValueError(f"{scene_id} functional build provenance is invalid")
    if functional_build.get("edge_count") != len(scene["functional_partners"]):
        raise ValueError(f"{scene_id} functional edge count mismatch")
    checks = dict(validation["checks"])
    checks["semantic_review_pass"] = _review_passed(review)
    all_pass = bool(validation["passed"] and all(checks.values()))
    status = "accepted_clean" if all_pass else "hard_valid_pending_independent_review"
    if not validation["passed"]:
        status = "rejected"
    provenance = scene["audit"]["provenance"]
    builder = scene["audit"]["functional_partner_builder"]
    revision_root = root / "revisions" / scene_id
    revision_paths = sorted(revision_root.glob("rev_*/revision.json"))
    revision = _load(revision_paths[-1]) if revision_paths else None
    return {
        "schema_version": RECORD_SCHEMA,
        "scene_id": scene_id,
        "room_type": scene["room_type"],
        "status": status,
        "furniture_count": len(scene["furniture"]),
        "functional_partner_edge_count": len(scene["functional_partners"]),
        "checks": checks,
        "semantic_review_status": review.get("semantic_review_status"),
        "diversity_status": (
            "novelty_pass" if diversity["novelty_pass"] else "novelty_fail"
        ),
        "closest_global": _closest_summary(diversity.get("closest_global")),
        "closest_same_room_type": _closest_summary(
            diversity.get("closest_same_room_type")
        ),
        "source_versions": {
            "proposal_schema": proposal["schema_version"],
            "scene_schema": scene["schema_version"],
            "validation_schema": validation["schema_version"],
            "validation_pipeline": validation["pipeline_version"],
            "layout_rules_schema": scene["audit"]["rules_version"],
            "layout_rules_sha256": provenance["layout_rules_sha256"],
            "functional_partner_rules_schema": builder["rules_schema_version"],
            "functional_partner_asset_rules_sha256": builder["rules_sha256"],
            "functional_partner_rules_file_sha256": provenance[
                "functional_rules_sha256"
            ],
            "functional_partner_compatibility_schema": builder[
                "compatibility_schema_version"
            ],
            "functional_partner_compatibility_rules_sha256": builder[
                "compatibility_rules_sha256"
            ],
            "functional_partner_compatibility_file_sha256": provenance[
                "functional_compatibility_sha256"
            ],
            "semantic_mapping_hash": scene["audit"]["semantic_package_hash"],
            "semantic_mapping_file_sha256": provenance[
                "semantic_mapping_sha256"
            ],
            "vocabulary_sha256": provenance["vocabulary_sha256"],
            "hssd_lookup_sha256": provenance["hssd_lookup_sha256"],
        },
        "artifacts": artifacts,
        "artifact_sha256": {
            key: _sha256(path) for key, path in artifact_paths.items()
        },
        "revision": revision,
    }


def run(root: Path, scene_ids: list[str]) -> None:
    records = [build_record(root, scene_id) for scene_id in scene_ids]
    manifest_path = root / "clean_manifest.jsonl"
    manifest_path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for value in records
        ),
        encoding="utf-8",
    )
    ledger_rows = [
        {
            "schema_version": "authored_layout_diversity_ledger_record_v2",
            "scene_id": value["scene_id"],
            "room_type": value["room_type"],
            "diversity_status": value["diversity_status"],
            "closest_global": value["closest_global"],
            "closest_same_room_type": value["closest_same_room_type"],
            "diversity_artifact_sha256": value["artifact_sha256"]["diversity"],
        }
        for value in records
    ]
    (root / "diversity_ledger.jsonl").write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for value in ledger_rows
        ),
        encoding="utf-8",
    )
    status = {
        "schema_version": STATUS_SCHEMA,
        "accepted_clean_count": sum(
            value["status"] == "accepted_clean" for value in records
        ),
        "hard_valid_pending_independent_review_count": sum(
            value["status"] == "hard_valid_pending_independent_review"
            for value in records
        ),
        "rejected_count": sum(value["status"] == "rejected" for value in records),
        "scene_records": records,
    }
    (root / "dataset_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild uniform authored-layout indexes and artifact hashes."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scene-id", action="append", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(Path(args.output_root), args.scene_id)
