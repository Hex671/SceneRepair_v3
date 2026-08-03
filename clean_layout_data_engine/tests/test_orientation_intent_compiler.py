from __future__ import annotations

import math

import pytest

from clean_layout_data_engine.orientation_intent_compiler import (
    _axis_only_yaw,
    compile_intent_proposal,
    compile_orientation,
    validate_functional_geometry,
    validate_clearance_zone_geometry,
    validate_dining_surface_geometry,
    validate_semantic_intents,
)


def _record(hssd_id: str, category: str, bbox: tuple[float, float, float]):
    return {
        "hssd_id": hssd_id,
        "category": category,
        "interaction_clearance": {
            "nonartic_clearance_v2": {"object_bbox_m": list(bbox)}
        },
    }


def _rules():
    return {
        "use_clearance": {
            "wardrobe_front_depth_m": 0.78,
            "cabinet_front_depth_m": 0.7,
            "bookcase_front_depth_m": 0.62,
            "desk_front_depth_m": 0.82,
            "seat_approach_depth_m": 0.38,
            "chair_behind_depth_m": 0.48,
        }
    }


def test_wall_and_face_object_intents_use_local_minus_y_front() -> None:
    desk = {
        "object_id": "desk",
        "x": 0.0,
        "y": 1.0,
        "orientation_intent": {"mode": "back_to_wall", "wall": "north"},
    }
    chair = {
        "object_id": "chair",
        "x": 0.0,
        "y": 0.0,
        "orientation_intent": {
            "mode": "face_object",
            "target_object_id": "desk",
        },
    }
    objects = {"desk": desk, "chair": chair}
    desk_yaw, _ = compile_orientation(
        desk, objects_by_id=objects, bbox={"width": 1.4, "depth": 0.7}
    )
    chair_yaw, _ = compile_orientation(
        chair, objects_by_id=objects, bbox={"width": 0.6, "depth": 0.6}
    )
    assert desk_yaw == 0.0
    assert chair_yaw == -math.pi


@pytest.mark.parametrize(
    ("bbox", "long_axis", "expected"),
    [
        ({"width": 1.5, "depth": 0.7}, "east_west", 0.0),
        ({"width": 1.5, "depth": 0.7}, "north_south", math.pi * 0.5),
        ({"width": 0.7, "depth": 1.5}, "east_west", math.pi * 0.5),
        ({"width": 0.7, "depth": 1.5}, "north_south", 0.0),
    ],
)
def test_axis_only_aligns_the_actual_long_bbox_axis(
    bbox: dict[str, float], long_axis: str, expected: float
) -> None:
    assert _axis_only_yaw({"long_axis": long_axis}, bbox) == expected


def test_compile_generates_yaw_clearances_and_path_targets() -> None:
    records = {
        "desk_asset": _record("desk_asset", "desk", (1.4, 0.7, 0.75)),
        "chair_asset": _record("chair_asset", "chair", (0.6, 0.6, 0.9)),
    }
    mapping = {
        "source_category_to_frozen": {"desk": "desk", "chair": "chair"}
    }
    intent = {
        "schema_version": "authored_furniture_intent_proposal_v1",
        "scene_id": "test_scene",
        "room_type": "home_office",
        "coordinate_frame": "room_local_z_up",
        "quaternion_order": "wxyz",
        "yaw_range": "[-pi,pi)",
        "room": {
            "length_m": 5.0,
            "width_m": 4.0,
            "organization": "test",
            "daily_activities": ["work"],
            "main_path": "door to desk",
            "protected_open_space": "center",
        },
        "openings": [],
        "furniture": [
            {
                "object_id": "desk",
                "hssd_id": "desk_asset",
                "category": "desk",
                "x": 0.0,
                "y": 1.2,
                "z": 0.0,
                "orientation_intent": {"mode": "back_to_wall", "wall": "north"},
                "clearance_intents": [
                    {
                        "clearance_id": "desk_access",
                        "type": "front_access",
                        "purpose": "seated work",
                        "allowed_overlap_object_ids": ["chair"],
                    }
                ],
                "placement_rationale": "north wall desk",
            },
            {
                "object_id": "chair",
                "hssd_id": "chair_asset",
                "category": "chair",
                "x": 0.0,
                "y": 0.4,
                "z": 0.0,
                "orientation_intent": {
                    "mode": "face_object",
                    "target_object_id": "desk",
                },
                "clearance_intents": [
                    {
                        "clearance_id": "chair_pullout",
                        "type": "rear_pullout",
                        "purpose": "chair pullout",
                        "allowed_overlap_object_ids": [],
                    }
                ],
                "placement_rationale": "faces desk",
            },
        ],
        "path_target_intents": [
            {
                "target_id": "desk_access_target",
                "clearance_id": "desk_access",
                "purpose": "primary work zone",
            }
        ],
        "primary_target_id": "desk_access_target",
        "authorship": {"author": "test"},
    }
    compiled, report = compile_intent_proposal(intent, records, mapping, _rules())
    by_id = {item["object_id"]: item for item in compiled["furniture"]}
    assert by_id["desk"]["yaw_rad"] == 0.0
    assert by_id["chair"]["yaw_rad"] == -math.pi
    assert len(compiled["use_clearance_zones"]) == 2
    assert compiled["primary_zone_goal_xy_m"][1] < by_id["desk"]["y"]
    assert report["passed"]


def test_desk_chair_functional_group_rejects_excessive_gap() -> None:
    furniture = [
        {
            "object_id": "desk",
            "category": "desk",
            "orientation_intent": {"mode": "back_to_wall"},
        },
        {
            "object_id": "chair",
            "category": "chair",
            "orientation_intent": {
                "mode": "face_object",
                "target_object_id": "desk",
            },
        },
    ]
    geometry = {
        "desk": {
            "object_id": "desk",
            "category": "desk",
            "x": 0.3,
            "y": 1.2,
            "yaw_rad": 0.0,
            "bbox": {"width": 1.4, "depth": 0.7},
        },
        "chair": {
            "object_id": "chair",
            "category": "chair",
            "x": 0.0,
            "y": 0.0,
            "yaw_rad": -math.pi,
            "bbox": {"width": 0.6, "depth": 0.6},
        },
    }
    with pytest.raises(ValueError, match="usable desk workstation"):
        validate_functional_geometry(furniture, geometry)


def test_lounge_functional_groups_reject_unreachable_tables() -> None:
    furniture = []
    geometry = {
        "sofa": {
            "object_id": "sofa",
            "category": "sofa",
            "x": -2.0,
            "y": 0.0,
            "yaw_rad": 0.0,
            "bbox": {"width": 1.8, "depth": 0.8},
        },
        "coffee_table": {
            "object_id": "coffee_table",
            "category": "coffee_table",
            "x": 0.3,
            "y": 0.0,
            "yaw_rad": 0.0,
            "bbox": {"width": 1.0, "depth": 0.6},
        },
    }
    with pytest.raises(ValueError, match="usable coffee-table group"):
        validate_functional_geometry(furniture, geometry)

    geometry["coffee_table"]["x"] = -0.55
    with pytest.raises(ValueError, match="usable coffee-table group"):
        validate_functional_geometry(furniture, geometry)

    geometry.pop("coffee_table")
    geometry["side_table"] = {
        "object_id": "side_table",
        "category": "side_table",
        "x": 0.3,
        "y": 0.0,
        "yaw_rad": 0.0,
        "bbox": {"width": 0.5, "depth": 0.5},
    }
    with pytest.raises(ValueError, match="within seated reach"):
        validate_functional_geometry(furniture, geometry)


def test_coffee_table_must_be_in_sofa_front_sector() -> None:
    geometry = {
        "sofa": {
            "object_id": "sofa",
            "category": "sofa",
            "x": 0.0,
            "y": 0.0,
            "yaw_rad": 0.0,
            "bbox": {"width": 2.0, "depth": 0.8},
        },
        "coffee_table": {
            "object_id": "coffee_table",
            "category": "coffee_table",
            "x": 1.0,
            "y": -1.2,
            "yaw_rad": 0.0,
            "bbox": {"width": 0.8, "depth": 0.6},
        },
    }
    with pytest.raises(ValueError, match="must be in front"):
        validate_functional_geometry([], geometry)


def test_armchair_coffee_table_seated_reach_is_bounded() -> None:
    furniture = [
        _semantic_item(
            "armchair",
            "armchair",
            orientation={
                "mode": "face_object",
                "target_object_id": "coffee_table",
            },
        )
    ]
    geometry = {
        "armchair": {
            "object_id": "armchair",
            "category": "armchair",
            "x": 0.0,
            "y": 0.0,
            "yaw_rad": 0.0,
            "bbox": {"width": 0.8, "depth": 0.8},
        },
        "coffee_table": {
            "object_id": "coffee_table",
            "category": "coffee_table",
            "x": 1.9,
            "y": 0.0,
            "yaw_rad": 0.0,
            "bbox": {"width": 1.0, "depth": 0.6},
        },
    }
    with pytest.raises(ValueError, match="coffee-table seat"):
        validate_functional_geometry(furniture, geometry)

    geometry["coffee_table"]["x"] = 1.15
    with pytest.raises(ValueError, match="coffee-table seat"):
        validate_functional_geometry(furniture, geometry)


def test_unrelated_use_clearance_zones_cannot_overlap() -> None:
    zones = [
        {
            "zone_id": "chair_pullout",
            "owner_id": "chair",
            "center_xy_m": [0.0, 0.0],
            "size_xy_m": [1.0, 1.0],
            "yaw_rad": 0.0,
            "allowed_overlap_object_ids": [],
        },
        {
            "zone_id": "console_access",
            "owner_id": "console",
            "center_xy_m": [0.25, 0.0],
            "size_xy_m": [1.0, 1.0],
            "yaw_rad": 0.0,
            "allowed_overlap_object_ids": [],
        },
    ]
    with pytest.raises(ValueError, match="use-clearance zones"):
        validate_clearance_zone_geometry(zones)

    zones[0]["allowed_overlap_object_ids"] = ["console"]
    validate_clearance_zone_geometry(zones)


def test_dining_room_rejects_low_table_asset() -> None:
    geometry = {
        "dining_table": {
            "object_id": "dining_table",
            "category": "table",
            "bbox": {"width": 1.0, "depth": 1.0, "height": 0.406},
        }
    }
    with pytest.raises(ValueError, match="conventional dining surface"):
        validate_dining_surface_geometry("dining_room", geometry)

    geometry["dining_table"]["bbox"]["height"] = 0.75
    validate_dining_surface_geometry("dining_room", geometry)


def _semantic_item(
    object_id: str,
    category: str,
    *,
    orientation: dict | None = None,
    clearances: list[dict] | None = None,
) -> dict:
    return {
        "object_id": object_id,
        "category": category,
        "orientation_intent": orientation
        or {"mode": "orientation_irrelevant"},
        "clearance_intents": clearances or [],
    }


def test_accessory_furniture_cannot_invent_dedicated_clearance() -> None:
    furniture = [
        _semantic_item(
            "side_table",
            "side_table",
            clearances=[
                {
                    "clearance_id": "side_table_access",
                    "type": "side_access_left",
                    "purpose": "reach tabletop",
                    "allowed_overlap_object_ids": [],
                }
            ],
        )
    ]
    with pytest.raises(ValueError, match="must not have a dedicated"):
        validate_semantic_intents(furniture)


def test_seat_cannot_face_an_accessory() -> None:
    furniture = [
        _semantic_item(
            "armchair",
            "armchair",
            orientation={
                "mode": "face_object",
                "target_object_id": "side_table",
            },
        ),
        _semantic_item("side_table", "side_table"),
    ]
    with pytest.raises(ValueError, match="must not face accessory"):
        validate_semantic_intents(furniture)


def test_clearance_overlap_exception_is_limited_to_approved_pairs() -> None:
    furniture = [
        _semantic_item(
            "armchair",
            "armchair",
            clearances=[
                {
                    "clearance_id": "armchair_access",
                    "type": "front_access",
                    "purpose": "seat approach",
                    "allowed_overlap_object_ids": ["side_table"],
                }
            ],
        ),
        _semantic_item("side_table", "side_table"),
    ]
    with pytest.raises(ValueError, match="not an approved functional overlap"):
        validate_semantic_intents(furniture)
