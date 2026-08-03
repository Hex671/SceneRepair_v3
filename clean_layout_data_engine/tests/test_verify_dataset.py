from __future__ import annotations

import json

from clean_layout_data_engine.storage import atomic_write_json, atomic_write_jsonl, sha256_file
from clean_layout_data_engine.verify_dataset import REQUIRED_CHECKS, verify


def _accepted_fixture(tmp_path):
    root = tmp_path / "production"
    scene_id = "scene_001"
    scene_path = root / "accepted" / "scenes" / f"{scene_id}.json"
    validation_path = root / "batches" / "batch_001" / "staging" / "validation" / f"{scene_id}.json"
    review_path = root / "batches" / "batch_001" / "staging" / "reviews" / "semantic" / f"{scene_id}.json"
    diversity_path = root / "batches" / "batch_001" / "staging" / "diversity" / f"{scene_id}.json"
    atomic_write_json(scene_path, {"scene_id": scene_id})
    atomic_write_json(
        validation_path,
        {"scene_id": scene_id, "passed": True, "checks": {key: True for key in REQUIRED_CHECKS}},
    )
    atomic_write_json(review_path, {"scene_id": scene_id, "passed": True})
    atomic_write_json(diversity_path, {"scene_id": scene_id, "novelty_pass": True})
    atomic_write_jsonl(
        root / "accepted" / "manifest.jsonl",
        [
            {
                "scene_id": scene_id,
                "room_type": "bedroom",
                "scene_path": str(scene_path),
                "scene_sha256": sha256_file(scene_path),
                "validation_path": str(validation_path),
                "semantic_review_path": str(review_path),
                "diversity_report_path": str(diversity_path),
            }
        ],
    )
    return root, scene_path


def test_dataset_verifier_accepts_complete_evidence(tmp_path) -> None:
    root, _ = _accepted_fixture(tmp_path)
    report = verify(root)
    assert report["passed"]
    assert report["verified_scene_count"] == 1


def test_dataset_verifier_detects_scene_tampering(tmp_path) -> None:
    root, scene_path = _accepted_fixture(tmp_path)
    scene_path.write_text(json.dumps({"scene_id": "scene_001", "changed": True}), encoding="utf-8")
    report = verify(root)
    assert not report["passed"]
    assert "SHA-256 mismatch" in str(report["errors"])


def test_dataset_verifier_rejects_empty_dataset(tmp_path) -> None:
    root = tmp_path / "production"
    atomic_write_jsonl(root / "accepted" / "manifest.jsonl", [])
    report = verify(root)
    assert not report["passed"]
    assert report["errors"][0]["type"] == "minimum_scene_count_not_met"
