from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from scene_repair_v3 import (
    FrozenFunctionalPartnerRules,
    FurnitureInput,
    RoomInput,
    SceneInput,
)


def _furniture(
    object_id: str, category: str, hssd_id: str | None
) -> FurnitureInput:
    return FurnitureInput(
        object_id,
        0.0,
        0.0,
        0.0,
        0.6,
        0.6,
        0.8,
        category,
        hssd_id=hssd_id,
    )


def _write_rules(tmp_path):
    rules = {
        "desk-annotated": {
            "source_category": "desk",
            "partner_categories": ["chair"],
        },
        "table-annotated": {
            "source_category": "coffee table",
            "partner_categories": ["armchair", "side table"],
        },
    }
    serialized = json.dumps(
        rules, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload = {
        "schema_version": "hssd_functional_partner_rules_v1",
        "source": {"file": "lookup.json.gz", "sha256": "0" * 64},
        "policy": {
            "asset_keyed_only": True,
            "category_inheritance": False,
            "commonsense_completion": False,
            "missing_annotation_behavior": "emit_no_edge",
        },
        "asset_count_in_source": 3,
        "annotated_asset_count": 2,
        "rules_sha256": hashlib.sha256(serialized).hexdigest(),
        "rules": rules,
    }
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_builder_uses_only_the_exact_source_asset_annotation(tmp_path) -> None:
    rules = FrozenFunctionalPartnerRules.load(_write_rules(tmp_path))
    furniture = (
        _furniture("desk_0", "desk", "HSSD:desk-annotated"),
        _furniture("desk_1", "desk", "desk-unannotated"),
        _furniture("chair_0", "chair", "chair-unannotated"),
    )
    edges = rules.build_edges(furniture)
    assert [(edge.source_id, edge.target_id) for edge in edges] == [
        ("desk_0", "chair_0")
    ]
    assert edges[0].orientation_mode == "none"
    assert edges[0].target_strategy == "none"
    assert edges[0].desired_gap_range_m is None


def test_missing_hssd_id_never_inherits_a_category_rule(tmp_path) -> None:
    rules = FrozenFunctionalPartnerRules.load(_write_rules(tmp_path))
    furniture = (
        _furniture("desk_0", "desk", None),
        _furniture("chair_0", "chair", "chair-unannotated"),
    )
    assert rules.build_edges(furniture) == ()


def test_partner_matching_is_lexical_not_commonsense_completion(tmp_path) -> None:
    rules = FrozenFunctionalPartnerRules.load(_write_rules(tmp_path))
    furniture = (
        _furniture("table_0", "coffee_table", "table-annotated"),
        _furniture("side_0", "side_table", "side-unannotated"),
        _furniture("seat_0", "chair", "chair-unannotated"),
    )
    edges = rules.build_edges(furniture)
    assert [(edge.source_id, edge.target_id) for edge in edges] == [
        ("table_0", "side_0")
    ]


def test_apply_refuses_to_mix_handwritten_and_frozen_edges(
    tmp_path, sample_scene
) -> None:
    rules = FrozenFunctionalPartnerRules.load(_write_rules(tmp_path))
    with pytest.raises(ValueError, match="refusing to mix"):
        rules.apply(sample_scene)


def test_repository_snapshot_is_loadable() -> None:
    rules = FrozenFunctionalPartnerRules.load(
        "configs/functional_partner_rules_hssd_v1.json"
    )
    assert len(rules) == 10050
    assert rules.asset_count_in_source == 10963


def test_frozen_dining_table_compatibility_requires_semantic_evidence() -> None:
    rules = FrozenFunctionalPartnerRules.load(
        "configs/functional_partner_rules_hssd_v1.json",
        "configs/functional_partner_category_compatibility_v1.json",
    )
    source = replace(
        _furniture(
            "bench_0",
            "bench",
            "396cb5be76282bba5590f45f0bacdd9d79475a93",
        ),
        family="seating",
        functions=("sit",),
        family_valid=True,
        functions_valid=True,
    )
    valid_table = replace(
        _furniture("table_0", "table", "unannotated-table"),
        family="support_surface",
        functions=("dining_surface", "support_objects"),
        family_valid=True,
        functions_valid=True,
    )
    invalid_table = replace(valid_table, functions=("support_objects",))
    matches = rules.build_edge_matches((source, valid_table))
    assert [(value.source_id, value.target_id) for value in matches] == [
        ("bench_0", "table_0")
    ]
    assert matches[0].annotated_partner_category == "dining_table"
    assert matches[0].category_match_mode == "frozen_semantic_compatibility"
    assert (
        matches[0].compatibility_rule_id
        == "dining_table_to_table_with_dining_surface_v1"
    )
    assert rules.build_edges((source, invalid_table)) == ()


def test_compatibility_never_replaces_missing_source_asset_annotation() -> None:
    rules = FrozenFunctionalPartnerRules.load(
        "configs/functional_partner_rules_hssd_v1.json",
        "configs/functional_partner_category_compatibility_v1.json",
    )
    source = replace(
        _furniture("bench_0", "bench", "unannotated-bench"),
        family="seating",
        functions=("sit",),
        family_valid=True,
        functions_valid=True,
    )
    table = replace(
        _furniture("table_0", "table", "unannotated-table"),
        family="support_surface",
        functions=("dining_surface",),
        family_valid=True,
        functions_valid=True,
    )
    assert rules.build_edges((source, table)) == ()
