from __future__ import annotations

import json
from pathlib import Path

from clean_layout_data_engine.run_pilot import _brief
from tools.validate_authored_scene import _compare_scene


ROOT = Path(__file__).resolve().parents[2]


def _strategy() -> dict:
    return json.loads(
        (ROOT / "clean_layout_data_engine" / "pipeline_config.json").read_text(
            encoding="utf-8"
        )
    )["diversity_strategy"]


def _profile(room_type: str) -> dict:
    return json.loads(
        (ROOT / "clean_layout_data_engine" / "pipeline_config.json").read_text(
            encoding="utf-8"
        )
    )["room_profiles"][room_type]


def _rules() -> dict:
    return json.loads(
        (ROOT / "configs" / "authored_layout_validation_rules_v2.json").read_text(
            encoding="utf-8"
        )
    )["deduplication"]


def _scene(
    *, scene_id: str, chair_x: float, chair_y: float = 1.4, door_wall: str = "south"
) -> dict:
    opening_centers = {
        "south": [0.0, -2.5, 1.025],
        "north": [0.0, 2.5, 1.025],
        "east": [3.0, 0.0, 1.025],
        "west": [-3.0, 0.0, 1.025],
    }
    return {
        "scene_id": scene_id,
        "room_type": "dining_room",
        "length": 6.0,
        "width": 5.0,
        "openings": [
            {
                "opening_id": "door_00",
                "kind": "door",
                "wall": door_wall,
                "clearance_center_xyz_m": opening_centers[door_wall],
                "clearance_size_xyz_m": [0.9, 0.8, 2.05],
                "interior_normal_xy": [0.0, 1.0],
            }
        ],
        "furniture": [
            {
                "object_id": "table_00",
                "category": "table",
                "x": 0.0,
                "y": 0.0,
                "yaw_rad": 0.0,
                "bbox": {"width": 1.6, "depth": 0.9, "height": 0.75},
            },
            {
                "object_id": "chair_00",
                "category": "chair",
                "x": chair_x,
                "y": chair_y,
                "yaw_rad": 0.0,
                "bbox": {"width": 0.5, "depth": 0.5, "height": 0.9},
            },
        ],
        "functional_partners": [
            {"source_id": "chair_00", "target_id": "table_00"}
        ],
        "audit": {"primary_zone_goal_xy_m": [0.0, 0.0]},
    }


def test_brief_spreads_openings_and_topologies() -> None:
    strategy = _strategy()
    profile = _profile("dining_room")
    briefs = [_brief(profile, "dining_room", seed, strategy, []) for seed in range(40)]

    assert {value["diversity_design"]["door_wall"] for value in briefs} == {
        "north",
        "south",
        "east",
        "west",
    }
    assert len({value["diversity_design"]["primary_layout_topology"] for value in briefs}) >= 4
    assert any(
        value["diversity_design"]["window_pattern"] == "none" for value in briefs
    )
    for brief in briefs:
        count = brief["target_furniture_count"]
        lower, upper = brief["furniture_count_range"]
        assert lower <= count <= upper
        assert sum(item["kind"] == "door" for item in brief["openings"]) == 1


def test_brief_prefers_undercovered_design_values() -> None:
    strategy = _strategy()
    profile = _profile("home_office")
    coverage = [
        {
            "audit": {
                "diversity_design": {
                    "room_scale": "small",
                    "aspect_mode": "balanced",
                    "door_wall": "north",
                    "door_offset_band": "center",
                    "window_pattern": "single_adjacent",
                    "primary_layout_topology": "wall_workstation",
                    "primary_anchor": "north",
                    "furniture_density": "sparse",
                    "window_offset_band": "center",
                }
            }
        }
        for _ in range(5)
    ]
    brief = _brief(profile, "home_office", 123, strategy, coverage)
    design = brief["diversity_design"]
    assert design["room_scale"] != "small"
    assert design["primary_layout_topology"] != "wall_workstation"


def test_functional_edge_similarity_alone_is_not_a_template() -> None:
    candidate = _scene(scene_id="candidate", chair_x=-1.8)
    existing = _scene(scene_id="existing", chair_x=0.4, chair_y=-1.2)
    comparison = _compare_scene(candidate, existing, _rules())

    assert comparison["functional_partner_graph_similarity"] == 1.0
    assert comparison["template_risk"] is False


def test_mirrored_copy_with_matching_openings_is_a_template() -> None:
    candidate = _scene(scene_id="candidate", chair_x=-1.4)
    existing = _scene(scene_id="existing", chair_x=1.4)
    comparison = _compare_scene(candidate, existing, _rules())

    assert comparison["best_symmetry_transform"] == "mirror_x"
    assert comparison["opening_signature_similarity"] == 1.0
    assert comparison["template_risk"] is True


def test_different_opening_condition_prevents_template_rejection() -> None:
    candidate = _scene(scene_id="candidate", chair_x=-1.4, door_wall="south")
    existing = _scene(scene_id="existing", chair_x=-1.4, door_wall="east")
    comparison = _compare_scene(candidate, existing, _rules())

    assert comparison["opening_signature_similarity"] == 0.0
    assert comparison["template_risk"] is False
