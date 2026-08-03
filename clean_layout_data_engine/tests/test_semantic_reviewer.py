from __future__ import annotations

from clean_layout_data_engine.semantic_reviewer import (
    _compact_review_inputs,
    _review_prompt,
    _validate_review,
)


def _scene():
    return {"furniture": [{"object_id": "desk"}, {"object_id": "chair"}]}


def _passing_review():
    return {
        "verdict": "pass",
        "scene_summary": "valid work group",
        "global_issues": [],
        "object_reviews": [
            {
                "object_id": "desk",
                "verdict": "pass",
                "issue_codes": [],
                "explanation": "usable",
            },
            {
                "object_id": "chair",
                "verdict": "pass",
                "issue_codes": [],
                "explanation": "faces desk",
            },
        ],
        "repair_instructions": [],
    }


def test_semantic_review_requires_exact_object_coverage() -> None:
    review = _passing_review()
    review["object_reviews"].pop()

    passed, errors = _validate_review(review, _scene())

    assert not passed
    assert "object review coverage does not match scene furniture" in errors


def test_semantic_review_rejects_internally_inconsistent_pass() -> None:
    review = _passing_review()
    review["global_issues"] = [
        {
            "code": "orientation",
            "severity": "major",
            "affected_object_ids": ["chair"],
            "explanation": "wrong facing",
        }
    ]

    passed, errors = _validate_review(review, _scene())

    assert not passed
    assert "pass verdict conflicts with issue/object fields" in errors


def test_semantic_review_accepts_consistent_full_pass() -> None:
    passed, errors = _validate_review(_passing_review(), _scene())
    assert passed
    assert not errors


def test_review_prompt_removes_provenance_but_keeps_layout_evidence() -> None:
    scene = {
        "scene_id": "scene",
        "room_type": "living_room",
        "length": 5.0,
        "width": 4.0,
        "furniture": [
            {
                "object_id": "sofa",
                "category": "sofa",
                "bbox": {"width": 2.0, "depth": 0.8, "height": 0.8},
                "x": 0.0,
                "y": 0.0,
                "yaw_rad": 0.0,
                "audit": {"bbox_source_path": "unneeded"},
            }
        ],
        "functional_partners": [
            {
                "source_id": "sofa",
                "target_id": "table",
                "orientation_mode": "face_target",
                "audit": {"rules_sha256": "unneeded"},
            }
        ],
        "openings": [],
        "audit": {"use_clearance_zones": [], "path_targets": []},
    }
    geometry = {
        "pairwise_signed_separation_table": [],
        "yaw_direction_table": [],
        "functional_partner_match_table": [{"rules_sha256": "unneeded"}],
    }

    compact_scene, compact_geometry = _compact_review_inputs(scene, geometry)
    prompt = _review_prompt(scene, geometry)

    assert "audit" not in compact_scene["furniture"][0]
    assert "audit" not in compact_scene["functional_partners"][0]
    assert "functional_partner_match_table" not in compact_geometry
    assert "bbox_source_path" not in prompt
    assert "pairwise_signed_separation_table" in prompt
