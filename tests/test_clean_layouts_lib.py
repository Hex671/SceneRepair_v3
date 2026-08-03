from __future__ import annotations

import math

from tools.clean_layouts_lib import (
    OrientedRectangle,
    SceneVerifier,
    front_vector,
    grid_path_exists,
    hssd_bbox,
    signed_separation,
    yaw_facing,
    yaw_to_quaternion_wxyz,
)
from tools.validate_authored_scene import _geometry_review_tables


def test_static_hssd_bbox_keeps_z_up_axis_order() -> None:
    record = {
        "interaction_clearance": {
            "nonartic_clearance_v2": {"object_bbox_m": [1.2, 0.7, 0.8]}
        }
    }
    assert hssd_bbox(record) == {
        "width": 1.2,
        "depth": 0.7,
        "height": 0.8,
        "source_path": "interaction_clearance.nonartic_clearance_v2.object_bbox_m",
        "source_axis_order": "room_z_up_[x,y,z]",
        "axis_conversion": "identity",
    }


def test_articulated_hssd_bbox_converts_y_up_to_z_up() -> None:
    record = {
        "interaction_clearance": {
            "official_combined_clearance": {
                "obj_aabb": {"min": [-0.6, 0.0, -0.3], "max": [0.6, 2.0, 0.3]}
            }
        }
    }
    bbox = hssd_bbox(record)
    assert bbox is not None
    assert (bbox["width"], bbox["depth"], bbox["height"]) == (1.2, 0.6, 2.0)


def test_yaw_front_and_quaternion_contract() -> None:
    assert front_vector(0.0) == (0.0, -1.0)
    assert front_vector(math.pi * 0.5)[0] == 1.0
    assert yaw_facing((0.0, 0.0), (1.0, 0.0)) == math.pi * 0.5
    assert yaw_to_quaternion_wxyz(math.pi * 0.5) == [
        round(math.sqrt(0.5), 10),
        0.0,
        0.0,
        round(math.sqrt(0.5), 10),
    ]


def test_sat_reports_penetration_and_separation() -> None:
    source = OrientedRectangle(0.0, 0.0, 2.0, 1.0, 0.0)
    overlapping = OrientedRectangle(0.8, 0.0, 1.0, 1.0, 0.0)
    separated = OrientedRectangle(2.0, 0.0, 1.0, 1.0, 0.0)
    assert signed_separation(source, overlapping) < 0.0
    assert signed_separation(source, separated) > 0.0


def test_strict_verifier_rejects_sub_five_millimeter_penetration() -> None:
    verifier = SceneVerifier(
        {},
        {},
        {
            "geometry": {
                "accepted_min_signed_separation_m": 0.0,
                "numeric_separation_epsilon_m": 1.0e-9,
            }
        },
    )
    scene = {
        "furniture": [
            {
                "object_id": "left",
                "x": 0.0,
                "y": 0.0,
                "yaw_rad": 0.0,
                "bbox": {"width": 1.0, "depth": 1.0, "height": 1.0},
            },
            {
                "object_id": "right",
                "x": 0.9955,
                "y": 0.0,
                "yaw_rad": 0.0,
                "bbox": {"width": 1.0, "depth": 1.0, "height": 1.0},
            },
        ]
    }
    result = verifier._check_collisions(scene)
    assert not result["passed"]
    assert result["violations"][0]["signed_separation_m"] == -0.0045


def test_geometry_review_yaw_uses_local_minus_y_front() -> None:
    scene = {
        "furniture": [
            {
                "object_id": "armchair_0",
                "x": 0.0,
                "y": 0.0,
                "yaw_rad": -math.pi * 0.5,
                "bbox": {"width": 0.7, "depth": 0.8, "height": 0.9},
            },
            {
                "object_id": "table_0",
                "x": -1.0,
                "y": 0.0,
                "yaw_rad": 0.0,
                "bbox": {"width": 0.5, "depth": 0.5, "height": 0.5},
            },
        ]
    }
    tables = _geometry_review_tables(
        scene,
        {
            "geometry": {
                "accepted_min_signed_separation_m": 0.0,
                "numeric_separation_epsilon_m": 1.0e-9,
            }
        },
    )
    armchair = tables["yaw_direction_table"][0]
    assert armchair["dominant_cardinal_direction"] == "west"
    assert armchair["local_minus_y_front_vector_xy"] == [-1.0, -0.0]


def test_grid_path_rejects_wall_of_obstacles() -> None:
    open_path, _ = grid_path_exists(
        4.0,
        4.0,
        [],
        (-1.4, 0.0),
        (1.4, 0.0),
        grid_m=0.1,
        human_radius_m=0.25,
    )
    blocked_path, _ = grid_path_exists(
        4.0,
        4.0,
        [OrientedRectangle(0.0, 0.0, 0.4, 4.0, 0.0)],
        (-1.4, 0.0),
        (1.4, 0.0),
        grid_m=0.1,
        human_radius_m=0.25,
    )
    assert open_path
    assert not blocked_path
