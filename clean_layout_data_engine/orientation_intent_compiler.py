"""Compile semantic furniture orientation/clearance intents into geometry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from tools.clean_layouts_lib import (
    bed_side_zone,
    behind_zone,
    front_vector,
    front_zone,
    hssd_bbox,
    load_hssd_lookup,
    object_rect,
    OrientedRectangle,
    signed_separation,
    wrap_yaw,
    yaw_facing,
)


INTENT_SCHEMA_VERSION = "authored_furniture_intent_proposal_v1"
COMPILED_SCHEMA_VERSION = "authored_furniture_proposal_v1"

ORIENTATION_MODES = {
    "back_to_wall",
    "face_object",
    "face_point",
    "face_direction",
    "axis_only",
    "orientation_irrelevant",
}
CLEARANCE_TYPES = {
    "front_access",
    "rear_pullout",
    "side_access_left",
    "side_access_right",
    "none",
}

# These objects are reached from adjacent seating or circulation. Giving them a
# dedicated rectangular access zone invents constraints that do not exist in a
# normal layout and encourages the author to "solve" collisions with overlap
# exceptions.
NO_DEDICATED_CLEARANCE_CATEGORIES = {
    "coffee_table",
    "floor_lamp",
    "ottoman",
    "side_table",
}

# A seat may be near these accessories, but they are not meaningful gaze/work
# targets. Use face_point/face_direction for an informal reading seat instead.
INVALID_SEATING_FACE_TARGET_CATEGORIES = {
    "floor_lamp",
    "nightstand",
    "side_table",
}
SEATING_CATEGORIES = {"armchair", "bench", "chair", "sofa"}

ALLOWED_CLEARANCE_OVERLAP_CATEGORY_PAIRS = {
    ("bed", "nightstand"),
    ("desk", "armchair"),
    ("desk", "chair"),
    ("table", "bench"),
    ("table", "chair"),
}

DESK_CHAIR_MAX_EDGE_GAP_M = 0.35
SOFA_COFFEE_TABLE_MAX_EDGE_GAP_M = 0.65
SOFA_COFFEE_TABLE_MIN_EDGE_GAP_M = 0.35
SIDE_TABLE_SEATING_MAX_EDGE_GAP_M = 0.35
# A facing lounge chair needs the table within seated reach. The blind reviewer
# consistently rejects 0.8 m as too far even though it is collision-free.
ARMCHAIR_COFFEE_TABLE_MAX_EDGE_GAP_M = 0.60
ARMCHAIR_COFFEE_TABLE_MIN_EDGE_GAP_M = 0.35
SOFA_COFFEE_TABLE_MAX_LATERAL_TO_FORWARD_RATIO = 0.75
DINING_TABLE_MIN_HEIGHT_M = 0.65
DINING_TABLE_MAX_HEIGHT_M = 0.90

_WALL_YAWS = {
    "north": 0.0,
    "south": -math.pi,
    "east": -math.pi * 0.5,
    "west": math.pi * 0.5,
}
_DIRECTION_YAWS = {
    "north": -math.pi,
    "south": 0.0,
    "east": math.pi * 0.5,
    "west": -math.pi * 0.5,
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _required(value: dict[str, Any], fields: set[str], context: str) -> None:
    missing = sorted(fields - set(value))
    if missing:
        raise ValueError(f"{context} missing required fields: {missing}")


def _axis_only_yaw(intent: dict[str, Any], bbox: dict[str, Any]) -> float:
    long_axis = intent.get("long_axis")
    if long_axis not in {"east_west", "north_south"}:
        raise ValueError("axis_only requires long_axis east_west or north_south")
    local_long_is_x = float(bbox["width"]) >= float(bbox["depth"])
    want_room_x = long_axis == "east_west"
    return 0.0 if local_long_is_x == want_room_x else math.pi * 0.5


def compile_orientation(
    item: dict[str, Any],
    *,
    objects_by_id: dict[str, dict[str, Any]],
    bbox: dict[str, Any],
) -> tuple[float, str]:
    intent = item["orientation_intent"]
    mode = intent.get("mode")
    if mode not in ORIENTATION_MODES:
        raise ValueError(f"{item['object_id']} has unsupported orientation mode {mode!r}")
    if mode == "back_to_wall":
        wall = intent.get("wall")
        if wall not in _WALL_YAWS:
            raise ValueError(f"{item['object_id']} back_to_wall requires a valid wall")
        return _WALL_YAWS[wall], f"back_to_wall:{wall}"
    if mode == "face_direction":
        direction = intent.get("direction")
        if direction not in _DIRECTION_YAWS:
            raise ValueError(f"{item['object_id']} requires a cardinal direction")
        return _DIRECTION_YAWS[direction], f"face_direction:{direction}"
    if mode == "face_object":
        target_id = intent.get("target_object_id")
        target = objects_by_id.get(str(target_id))
        if target is None or target is item:
            raise ValueError(f"{item['object_id']} has invalid face_object target")
        yaw = yaw_facing(
            (float(item["x"]), float(item["y"])),
            (float(target["x"]), float(target["y"])),
        )
        return yaw, f"face_object:{target_id}"
    if mode == "face_point":
        target = intent.get("target_xy_m")
        if not isinstance(target, list) or len(target) != 2:
            raise ValueError(f"{item['object_id']} face_point requires target_xy_m")
        yaw = yaw_facing(
            (float(item["x"]), float(item["y"])),
            (float(target[0]), float(target[1])),
        )
        return yaw, "face_point"
    if mode == "axis_only":
        return _axis_only_yaw(intent, bbox), f"axis_only:{intent.get('long_axis')}"
    return 0.0, "orientation_irrelevant:canonical_zero"


def _clearance_depth(
    category: str, clearance_type: str, rules: dict[str, Any]
) -> float:
    values = rules["use_clearance"]
    if clearance_type == "rear_pullout":
        return float(values["chair_behind_depth_m"])
    field_by_category = {
        "wardrobe": "wardrobe_front_depth_m",
        "cabinet": "cabinet_front_depth_m",
        "bookcase": "bookcase_front_depth_m",
        "desk": "desk_front_depth_m",
    }
    field = field_by_category.get(category)
    if field is not None:
        return float(values[field])
    return float(values["seat_approach_depth_m"])


def _compile_clearance(
    intent: dict[str, Any],
    obj: dict[str, Any],
    rules: dict[str, Any],
) -> dict[str, Any] | None:
    clearance_type = intent.get("type")
    if clearance_type not in CLEARANCE_TYPES:
        raise ValueError(
            f"{obj['object_id']} has unsupported clearance type {clearance_type!r}"
        )
    if clearance_type == "none":
        return None
    allowed = intent.get("allowed_overlap_object_ids", [])
    purpose = str(intent.get("purpose", "")).strip()
    clearance_id = str(intent.get("clearance_id", "")).strip()
    if not purpose or not clearance_id:
        raise ValueError(f"{obj['object_id']} clearance needs id and purpose")
    if clearance_type == "front_access":
        zone = front_zone(
            obj,
            _clearance_depth(obj["category"], clearance_type, rules),
            purpose,
            allowed=allowed,
        )
    elif clearance_type == "rear_pullout":
        zone = behind_zone(
            obj,
            _clearance_depth(obj["category"], clearance_type, rules),
            purpose,
            allowed=allowed,
        )
    else:
        side = "left" if clearance_type.endswith("left") else "right"
        zone = bed_side_zone(obj, side)
        zone["purpose"] = purpose
        zone["allowed_overlap_object_ids"] = list(allowed)
    zone["zone_id"] = clearance_id
    return zone


def validate_semantic_intents(furniture: list[dict[str, Any]]) -> None:
    """Reject geometrically valid intents that encode implausible semantics."""

    objects_by_id = {str(item["object_id"]): item for item in furniture}
    for item in furniture:
        object_id = str(item["object_id"])
        category = str(item["category"])
        active_clearances = [
            value
            for value in item.get("clearance_intents", [])
            if value.get("type") != "none"
        ]
        if category in NO_DEDICATED_CLEARANCE_CATEGORIES and active_clearances:
            raise ValueError(
                f"{object_id} category {category} must not have a dedicated "
                "use-clearance zone; access it from adjacent circulation"
            )

        orientation = item.get("orientation_intent", {})
        if category in SEATING_CATEGORIES and orientation.get("mode") == "face_object":
            target_id = str(orientation.get("target_object_id"))
            target = objects_by_id.get(target_id)
            if target is not None and target.get("category") in (
                INVALID_SEATING_FACE_TARGET_CATEGORIES
            ):
                raise ValueError(
                    f"{object_id} must not face accessory {target_id} "
                    f"({target['category']}); face a real activity target, "
                    "cardinal direction, or explicit room point"
                )

        for clearance in active_clearances:
            for allowed_id in clearance.get("allowed_overlap_object_ids", []):
                target = objects_by_id.get(str(allowed_id))
                if target is None:
                    raise ValueError(
                        f"{object_id} clearance allows unknown object {allowed_id!r}"
                    )
                pair = (category, str(target["category"]))
                if pair not in ALLOWED_CLEARANCE_OVERLAP_CATEGORY_PAIRS:
                    raise ValueError(
                        f"{object_id} clearance may not overlap {allowed_id}: "
                        f"category pair {pair} is not an approved functional overlap"
                    )


def validate_functional_geometry(
    furniture: list[dict[str, Any]], geometry_objects: dict[str, dict[str, Any]]
) -> None:
    """Enforce small, high-confidence geometric contracts for active groups."""

    for item in furniture:
        orientation = item["orientation_intent"]
        if (
            item["category"] != "chair"
            or orientation.get("mode") != "face_object"
        ):
            continue
        target_id = str(orientation.get("target_object_id"))
        target = geometry_objects.get(target_id)
        source = geometry_objects.get(str(item["object_id"]))
        if source is None or target is None or target["category"] != "desk":
            continue
        gap = signed_separation(object_rect(source), object_rect(target))
        if gap < -1e-6 or gap > DESK_CHAIR_MAX_EDGE_GAP_M:
            raise ValueError(
                f"{item['object_id']} must form a usable desk workstation with "
                f"{target_id}: edge gap is {gap:.3f} m, required 0.000.."
                f"{DESK_CHAIR_MAX_EDGE_GAP_M:.3f} m"
            )

    sofas = [
        value for value in geometry_objects.values() if value["category"] == "sofa"
    ]
    coffee_tables = [
        value
        for value in geometry_objects.values()
        if value["category"] == "coffee_table"
    ]
    if len(sofas) == 1 and coffee_tables:
        sofa = sofas[0]
        front_x, front_y = front_vector(float(sofa["yaw_rad"]))
        candidates = []
        for table in coffee_tables:
            delta_x = float(table["x"]) - float(sofa["x"])
            delta_y = float(table["y"]) - float(sofa["y"])
            forward = delta_x * front_x + delta_y * front_y
            lateral = abs(delta_x * -front_y + delta_y * front_x)
            candidates.append(
                (
                    signed_separation(object_rect(sofa), object_rect(table)),
                    forward,
                    lateral,
                    table,
                )
            )
        nearest_gap, forward, lateral, table = min(candidates, key=lambda value: value[0])
        if (
            nearest_gap < SOFA_COFFEE_TABLE_MIN_EDGE_GAP_M
            or nearest_gap > SOFA_COFFEE_TABLE_MAX_EDGE_GAP_M
        ):
            raise ValueError(
                f"{sofa['object_id']} must have a usable coffee-table group: "
                f"nearest edge gap is {nearest_gap:.3f} m, required "
                f"{SOFA_COFFEE_TABLE_MIN_EDGE_GAP_M:.3f}.."
                f"{SOFA_COFFEE_TABLE_MAX_EDGE_GAP_M:.3f} m"
            )
        if (
            forward <= 0.0
            or lateral > forward * SOFA_COFFEE_TABLE_MAX_LATERAL_TO_FORWARD_RATIO
        ):
            raise ValueError(
                f"{table['object_id']} must be in front of {sofa['object_id']}: "
                f"forward={forward:.3f} m, lateral={lateral:.3f} m"
            )

    for item in furniture:
        orientation = item["orientation_intent"]
        if (
            item["category"] != "armchair"
            or orientation.get("mode") != "face_object"
        ):
            continue
        target_id = str(orientation.get("target_object_id"))
        source = geometry_objects.get(str(item["object_id"]))
        target = geometry_objects.get(target_id)
        if source is None or target is None or target["category"] != "coffee_table":
            continue
        gap = signed_separation(object_rect(source), object_rect(target))
        if (
            gap < ARMCHAIR_COFFEE_TABLE_MIN_EDGE_GAP_M
            or gap > ARMCHAIR_COFFEE_TABLE_MAX_EDGE_GAP_M
        ):
            raise ValueError(
                f"{item['object_id']} faces {target_id} but is not a usable "
                f"coffee-table seat: edge gap is {gap:.3f} m, required "
                f"{ARMCHAIR_COFFEE_TABLE_MIN_EDGE_GAP_M:.3f}.."
                f"{ARMCHAIR_COFFEE_TABLE_MAX_EDGE_GAP_M:.3f} m"
            )

    seating = [
        value
        for value in geometry_objects.values()
        if value["category"] in {"sofa", "armchair"}
    ]
    for side_table in (
        value
        for value in geometry_objects.values()
        if value["category"] == "side_table"
    ):
        if not seating:
            continue
        nearest_gap = min(
            signed_separation(object_rect(side_table), object_rect(seat))
            for seat in seating
        )
        if nearest_gap < -1e-6 or nearest_gap > SIDE_TABLE_SEATING_MAX_EDGE_GAP_M:
            raise ValueError(
                f"{side_table['object_id']} must be within seated reach of a sofa "
                f"or armchair: nearest edge gap is {nearest_gap:.3f} m, required "
                f"0.000..{SIDE_TABLE_SEATING_MAX_EDGE_GAP_M:.3f} m"
            )


def validate_dining_surface_geometry(
    room_type: str,
    geometry_objects: dict[str, dict[str, Any]],
) -> None:
    """Keep low coffee/side tables out of a conventional dining-room role."""
    if room_type != "dining_room":
        return
    tables = [
        value
        for value in geometry_objects.values()
        if value["category"] == "table"
    ]
    if not tables:
        raise ValueError("dining_room must contain a table dining surface")
    for table in tables:
        height = float(table["bbox"]["height"])
        if not DINING_TABLE_MIN_HEIGHT_M <= height <= DINING_TABLE_MAX_HEIGHT_M:
            raise ValueError(
                f"{table['object_id']} is not a conventional dining surface: "
                f"HSSD bbox height is {height:.3f} m, required "
                f"{DINING_TABLE_MIN_HEIGHT_M:.2f}..{DINING_TABLE_MAX_HEIGHT_M:.2f} m"
            )


def validate_clearance_zone_geometry(zones: list[dict[str, Any]]) -> None:
    """Prevent unrelated activities from claiming the same physical space."""

    for index, left in enumerate(zones):
        left_rect = OrientedRectangle(
            float(left["center_xy_m"][0]),
            float(left["center_xy_m"][1]),
            float(left["size_xy_m"][0]),
            float(left["size_xy_m"][1]),
            float(left["yaw_rad"]),
        )
        for right in zones[index + 1 :]:
            if left["owner_id"] == right["owner_id"]:
                continue
            shared_owner = (
                right["owner_id"] in left.get("allowed_overlap_object_ids", [])
                or left["owner_id"] in right.get("allowed_overlap_object_ids", [])
            )
            if shared_owner:
                continue
            right_rect = OrientedRectangle(
                float(right["center_xy_m"][0]),
                float(right["center_xy_m"][1]),
                float(right["size_xy_m"][0]),
                float(right["size_xy_m"][1]),
                float(right["yaw_rad"]),
            )
            overlap = signed_separation(left_rect, right_rect)
            if overlap < -1e-6:
                raise ValueError(
                    f"use-clearance zones {left['zone_id']} ({left['owner_id']}) "
                    f"and {right['zone_id']} ({right['owner_id']}) overlap by "
                    f"{-overlap:.3f} m without an approved shared-owner pair"
                )


def compile_intent_proposal(
    intent: dict[str, Any],
    records: dict[str, dict[str, Any]],
    semantic_mapping: dict[str, Any],
    rules: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _required(
        intent,
        {
            "schema_version",
            "scene_id",
            "room_type",
            "coordinate_frame",
            "quaternion_order",
            "yaw_range",
            "room",
            "openings",
            "furniture",
            "path_target_intents",
            "primary_target_id",
            "authorship",
        },
        "intent proposal",
    )
    if intent["schema_version"] != INTENT_SCHEMA_VERSION:
        raise ValueError("unsupported intent proposal schema")
    if intent["coordinate_frame"] != "room_local_z_up":
        raise ValueError("coordinate_frame must be room_local_z_up")
    if intent["quaternion_order"] != "wxyz" or intent["yaw_range"] != "[-pi,pi)":
        raise ValueError("unsupported rotation contract")
    furniture = intent["furniture"]
    if not isinstance(furniture, list) or not 1 <= len(furniture) <= 17:
        raise ValueError("furniture must contain 1..17 objects")
    objects_by_id = {str(item["object_id"]): item for item in furniture}
    if len(objects_by_id) != len(furniture):
        raise ValueError("furniture object_id values must be unique")
    validate_semantic_intents(furniture)

    compiled_furniture: list[dict[str, Any]] = []
    geometry_objects: dict[str, dict[str, Any]] = {}
    orientation_rows: list[dict[str, Any]] = []
    source_to_frozen = semantic_mapping["source_category_to_frozen"]
    for item in furniture:
        _required(
            item,
            {
                "object_id",
                "hssd_id",
                "category",
                "x",
                "y",
                "z",
                "orientation_intent",
                "clearance_intents",
                "placement_rationale",
            },
            f"furniture {item.get('object_id')}",
        )
        hssd_id = str(item["hssd_id"]).lower()
        record = records.get(hssd_id)
        if record is None:
            raise ValueError(f"{item['object_id']} uses unknown HSSD id {hssd_id}")
        bbox = hssd_bbox(record)
        if bbox is None:
            raise ValueError(f"{item['object_id']} has no supported HSSD bbox")
        source_category = str(record.get("category", "")).lower()
        expected_category = source_to_frozen.get(source_category)
        if expected_category != item["category"]:
            raise ValueError(
                f"{item['object_id']} category mismatch: expected {expected_category!r}"
            )
        yaw, rule = compile_orientation(
            item, objects_by_id=objects_by_id, bbox=bbox
        )
        yaw = wrap_yaw(yaw)
        compiled_item = {
            "object_id": item["object_id"],
            "hssd_id": hssd_id,
            "category": item["category"],
            "x": float(item["x"]),
            "y": float(item["y"]),
            "z": float(item["z"]),
            "yaw_rad": yaw,
            "placement_rationale": item["placement_rationale"],
        }
        compiled_furniture.append(compiled_item)
        geometry_objects[item["object_id"]] = {**compiled_item, "bbox": bbox}
        orientation_rows.append(
            {
                "object_id": item["object_id"],
                "orientation_intent": item["orientation_intent"],
                "computed_yaw_rad": round(yaw, 12),
                "local_minus_y_front_vector_xy": [
                    round(value, 9) for value in front_vector(yaw)
                ],
                "computation_rule": rule,
            }
        )

    validate_functional_geometry(furniture, geometry_objects)
    validate_dining_surface_geometry(intent["room_type"], geometry_objects)

    zones: list[dict[str, Any]] = []
    zone_rows: list[dict[str, Any]] = []
    zones_by_id: dict[str, dict[str, Any]] = {}
    for item in furniture:
        obj = geometry_objects[item["object_id"]]
        for clearance_intent in item["clearance_intents"]:
            zone = _compile_clearance(clearance_intent, obj, rules)
            if zone is None:
                continue
            if zone["zone_id"] in zones_by_id:
                raise ValueError(f"duplicate clearance id {zone['zone_id']!r}")
            zones.append(zone)
            zones_by_id[zone["zone_id"]] = zone
            zone_rows.append(
                {
                    "owner_id": item["object_id"],
                    "clearance_intent": clearance_intent,
                    "computed_clearance_geometry": {
                        "center_xy_m": zone["center_xy_m"],
                        "size_xy_m": zone["size_xy_m"],
                        "yaw_rad": zone["yaw_rad"],
                    },
                }
            )

    validate_clearance_zone_geometry(zones)

    path_targets: list[dict[str, Any]] = []
    for path_intent in intent["path_target_intents"]:
        zone = zones_by_id.get(str(path_intent["clearance_id"]))
        if zone is None:
            raise ValueError(
                f"path target {path_intent['target_id']} references unknown clearance"
            )
        path_targets.append(
            {
                "target_id": path_intent["target_id"],
                "goal_xy_m": zone["center_xy_m"],
                "purpose": path_intent["purpose"],
            }
        )
    primary = next(
        (
            value
            for value in path_targets
            if value["target_id"] == intent["primary_target_id"]
        ),
        None,
    )
    if primary is None:
        raise ValueError("primary_target_id must reference a path target")

    compiled = {
        "schema_version": COMPILED_SCHEMA_VERSION,
        "scene_id": intent["scene_id"],
        "room_type": intent["room_type"],
        "coordinate_frame": intent["coordinate_frame"],
        "quaternion_order": intent["quaternion_order"],
        "yaw_range": intent["yaw_range"],
        "room": intent["room"],
        "openings": intent["openings"],
        "furniture": compiled_furniture,
        "use_clearance_zones": zones,
        "path_targets": path_targets,
        "primary_zone_goal_xy_m": primary["goal_xy_m"],
        "authorship": {
            **intent["authorship"],
            "layout_decisions_by_python": False,
            "orientation_and_clearance_compiled_by_python": True,
            "functional_partners": (
                "omitted_by_design; generated only by FrozenFunctionalPartnerRules"
            ),
        },
    }
    report = {
        "schema_version": "orientation_intent_compile_report_v1",
        "scene_id": intent["scene_id"],
        "passed": True,
        "coordinate_contract": "room_local_z_up; HSSD local -Y front",
        "orientation_rows": orientation_rows,
        "clearance_rows": zone_rows,
    }
    return compiled, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intent", required=True, type=Path)
    parser.add_argument("--compiled", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--hssd-lookup", required=True, type=Path)
    parser.add_argument("--semantic-mapping", required=True, type=Path)
    parser.add_argument("--layout-rules", required=True, type=Path)
    args = parser.parse_args()
    compiled, report = compile_intent_proposal(
        _load_json(args.intent),
        load_hssd_lookup(args.hssd_lookup),
        _load_json(args.semantic_mapping),
        _load_json(args.layout_rules),
    )
    args.compiled.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.compiled.write_text(
        json.dumps(compiled, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
