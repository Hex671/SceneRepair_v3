from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from tools.clean_layouts_lib import (
    front_vector,
    hssd_bbox,
    load_hssd_lookup,
    rotate_xy,
    wrap_yaw,
    yaw_facing,
    yaw_to_quaternion_wxyz,
)


INTENT_SCHEMA_VERSION = "authored_furniture_intent_proposal_v1"
COMPILED_SCHEMA_VERSION = "authored_furniture_proposal_v1"
REPORT_SCHEMA_VERSION = "orientation_intent_compile_report_v1"


class OrientationCompiler:
    def __init__(
        self,
        records: dict[str, dict[str, Any]],
        semantic_mapping: dict[str, Any],
        validation_rules: dict[str, Any],
    ) -> None:
        self.records = records
        self.semantic_mapping = semantic_mapping
        self.rules = validation_rules

    def compile(
        self, proposal: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._validate_proposal(proposal)
        authored = proposal["furniture"]
        by_id = {item["object_id"]: item for item in authored}
        compiled_furniture: list[dict[str, Any]] = []
        report_objects: list[dict[str, Any]] = []

        for item in authored:
            record, bbox = self._asset(item)
            yaw, rule = self._compile_yaw(item, bbox, by_id)
            quaternion = yaw_to_quaternion_wxyz(yaw)
            compiled_furniture.append(
                {
                    "object_id": item["object_id"],
                    "hssd_id": item["hssd_id"],
                    "category": item["category"],
                    "x": float(item["x"]),
                    "y": float(item["y"]),
                    "z": float(item["z"]),
                    "yaw_rad": yaw,
                    "rotation_wxyz": quaternion,
                    "placement_rationale": item["placement_rationale"],
                }
            )
            clearance = self._compile_clearance(item, bbox, yaw, authored)
            report_objects.append(
                {
                    "object_id": item["object_id"],
                    "hssd_id": record["hssd_id"],
                    "orientation_intent": item["orientation_intent"],
                    "computed_yaw_rad": yaw,
                    "computed_quaternion_wxyz": quaternion,
                    "local_minus_y_front_vector_xy": [
                        round(value, 10) for value in front_vector(yaw)
                    ],
                    "computation_rule": rule,
                    "clearance_intent": item["clearance_intent"],
                    "computed_clearance_geometry": clearance,
                }
            )

        zones = [
            item["computed_clearance_geometry"]
            for item in report_objects
            if item["computed_clearance_geometry"] is not None
        ]
        opening_defaults = self.rules["openings"]
        openings = []
        for opening in proposal["openings"]:
            is_door = opening["kind"] == "door"
            openings.append(
                {
                    **opening,
                    "opening_height_m": float(
                        opening_defaults[
                            "door_height_m" if is_door else "window_height_m"
                        ]
                    ),
                    "sill_height_m": (
                        0.0
                        if is_door
                        else float(opening_defaults["window_sill_height_m"])
                    ),
                    "clearance_depth_m": float(
                        opening_defaults[
                            "door_clearance_depth_m"
                            if is_door
                            else "window_clearance_depth_m"
                        ]
                    ),
                }
            )

        compiled = {
            "schema_version": COMPILED_SCHEMA_VERSION,
            "scene_id": proposal["scene_id"],
            "room_type": proposal["room_type"],
            "coordinate_frame": proposal["coordinate_frame"],
            "quaternion_order": "wxyz",
            "yaw_range": "[-pi,pi)",
            "room": proposal["room"],
            "openings": openings,
            "furniture": compiled_furniture,
            "use_clearance_zones": zones,
            "path_targets": proposal["path_targets"],
            "primary_zone_goal_xy_m": proposal["primary_zone_goal_xy_m"],
            "authorship": {
                **proposal["authorship"],
                "orientation_and_clearance_compiled": True,
                "compiler_modified_furniture_identity_or_xy": False,
            },
        }
        compile_report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "scene_id": proposal["scene_id"],
            "source_schema_version": INTENT_SCHEMA_VERSION,
            "coordinate_contract": {
                "coordinate_frame": "room_local_z_up",
                "yaw_axis": "+Z",
                "semantic_front": "local_-Y",
                "yaw_range": "[-pi,pi)",
                "quaternion_order": "wxyz",
            },
            "compiler_invariants": {
                "furniture_order_preserved": True,
                "furniture_identity_preserved": True,
                "hssd_id_preserved": True,
                "xy_preserved": True,
                "yaw_authored_by_model": False,
                "quaternion_authored_by_model": False,
                "clearance_geometry_authored_by_model": False,
            },
            "objects": report_objects,
            "passed": True,
            "errors": [],
        }
        return compiled, compile_report

    def _validate_proposal(self, proposal: dict[str, Any]) -> None:
        required = {
            "schema_version",
            "scene_id",
            "room_type",
            "coordinate_frame",
            "room",
            "openings",
            "furniture",
            "path_targets",
            "primary_zone_goal_xy_m",
            "authorship",
        }
        missing = sorted(required - set(proposal))
        if missing:
            raise ValueError(f"missing intent proposal fields: {missing}")
        if proposal["schema_version"] != INTENT_SCHEMA_VERSION:
            raise ValueError("unsupported intent proposal schema")
        if proposal["coordinate_frame"] != "room_local_z_up":
            raise ValueError("coordinate_frame must be room_local_z_up")
        furniture = proposal["furniture"]
        if not isinstance(furniture, list) or not furniture:
            raise ValueError("furniture must be a non-empty list")
        allowed_fields = {
            "object_id",
            "hssd_id",
            "category",
            "x",
            "y",
            "z",
            "orientation_intent",
            "clearance_intent",
            "placement_rationale",
        }
        required_fields = allowed_fields
        ids: set[str] = set()
        for index, item in enumerate(furniture):
            if set(item) != required_fields:
                raise ValueError(
                    f"furniture[{index}] fields must be exactly {sorted(required_fields)}"
                )
            if item["object_id"] in ids:
                raise ValueError(f"duplicate object_id {item['object_id']!r}")
            ids.add(item["object_id"])
            if "yaw_rad" in item or "rotation_wxyz" in item:
                raise ValueError("intent furniture must not contain yaw or quaternion")
            if item["clearance_intent"] not in {
                "front_access",
                "rear_pullout",
                "side_access_left",
                "side_access_right",
                "none",
            }:
                raise ValueError(
                    f"unsupported clearance intent {item['clearance_intent']!r}"
                )
        for item in furniture:
            intent = item["orientation_intent"]
            if not isinstance(intent, dict) or "type" not in intent:
                raise ValueError(f"{item['object_id']} has invalid orientation_intent")
            target = intent.get("target_object_id")
            if intent["type"] == "face_object" and target not in ids:
                raise ValueError(
                    f"{item['object_id']} references unknown target {target!r}"
                )

    def _asset(
        self, item: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        hssd_id = str(item["hssd_id"]).lower()
        record = self.records.get(hssd_id)
        if record is None:
            raise ValueError(f"unknown HSSD id {hssd_id}")
        bbox = hssd_bbox(record)
        if bbox is None:
            raise ValueError(f"HSSD id {hssd_id} has no supported bbox")
        source_category = str(record.get("category", "")).lower()
        frozen = self.semantic_mapping["source_category_to_frozen"].get(
            source_category
        )
        if frozen != item["category"]:
            raise ValueError(
                f"{item['object_id']} category {item['category']!r} does not "
                f"match annotated category {source_category!r}"
            )
        return record, bbox

    def _compile_yaw(
        self,
        item: dict[str, Any],
        bbox: dict[str, Any],
        by_id: dict[str, dict[str, Any]],
    ) -> tuple[float, str]:
        intent = item["orientation_intent"]
        kind = intent["type"]
        direction_yaws = {
            "south": 0.0,
            "north": -math.pi,
            "west": -math.pi * 0.5,
            "east": math.pi * 0.5,
        }
        if kind == "back_to_wall":
            wall = intent.get("wall")
            inward = {
                "north": "south",
                "south": "north",
                "east": "west",
                "west": "east",
            }.get(wall)
            if inward is None:
                raise ValueError(f"unsupported wall {wall!r}")
            return wrap_yaw(direction_yaws[inward]), f"back_to_wall:{wall}->front_{inward}"
        if kind == "face_direction":
            direction = intent.get("direction")
            if direction not in direction_yaws:
                raise ValueError(f"unsupported direction {direction!r}")
            return wrap_yaw(direction_yaws[direction]), f"face_direction:front_{direction}"
        if kind == "face_object":
            target_id = intent["target_object_id"]
            target = by_id[target_id]
            yaw = yaw_facing(
                (float(item["x"]), float(item["y"])),
                (float(target["x"]), float(target["y"])),
            )
            return yaw, f"yaw_facing:target_object:{target_id}"
        if kind == "face_point":
            target = intent.get("target_xy_m")
            if not isinstance(target, list) or len(target) != 2:
                raise ValueError("face_point requires target_xy_m with two values")
            yaw = yaw_facing(
                (float(item["x"]), float(item["y"])),
                (float(target[0]), float(target[1])),
            )
            return yaw, f"yaw_facing:target_point:{target}"
        if kind == "axis_only":
            long_axis = intent.get("long_axis")
            if long_axis not in {"east_west", "north_south"}:
                raise ValueError(f"unsupported long_axis {long_axis!r}")
            native_axis = (
                "east_west"
                if float(bbox["width"]) >= float(bbox["depth"])
                else "north_south"
            )
            yaw = 0.0 if native_axis == long_axis else math.pi * 0.5
            return wrap_yaw(yaw), f"axis_only:native_{native_axis}->target_{long_axis}"
        if kind == "orientation_irrelevant":
            return 0.0, "orientation_irrelevant:canonical_yaw_0"
        raise ValueError(f"unsupported orientation intent {kind!r}")

    def _compile_clearance(
        self,
        item: dict[str, Any],
        bbox: dict[str, Any],
        yaw: float,
        authored: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        intent = item["clearance_intent"]
        if intent == "none":
            return None
        category = item["category"]
        defaults = self.rules["use_clearance"]
        depth_keys = {
            "wardrobe": "wardrobe_front_depth_m",
            "cabinet": "cabinet_front_depth_m",
            "bookcase": "bookcase_front_depth_m",
            "desk": "desk_front_depth_m",
            "sofa": "seat_approach_depth_m",
            "armchair": "seat_approach_depth_m",
            "chair": "chair_behind_depth_m",
            "bench": "seat_approach_depth_m",
            "ottoman": "seat_approach_depth_m",
        }
        x, y = float(item["x"]), float(item["y"])
        width, depth = float(bbox["width"]), float(bbox["depth"])
        allowed = [
            other["object_id"]
            for other in authored
            if other["orientation_intent"].get("type") == "face_object"
            and other["orientation_intent"].get("target_object_id")
            == item["object_id"]
            and intent == "front_access"
        ]
        if intent in {"front_access", "rear_pullout"}:
            if category not in depth_keys:
                raise ValueError(
                    f"no frozen clearance depth default for {category}/{intent}"
                )
            clearance_depth = float(defaults[depth_keys[category]])
            direction = front_vector(yaw)
            sign = 1.0 if intent == "front_access" else -1.0
            distance = depth * 0.5 + clearance_depth * 0.5
            center = [
                round(x + sign * direction[0] * distance, 6),
                round(y + sign * direction[1] * distance, 6),
            ]
            size = [round(width, 6), round(clearance_depth, 6)]
        else:
            if category != "bed":
                raise ValueError(
                    f"side access currently requires frozen bed_side_width_m; got {category}"
                )
            side_width = float(defaults["bed_side_width_m"])
            local_sign = 1.0 if intent == "side_access_left" else -1.0
            dx, dy = rotate_xy(
                local_sign * (width * 0.5 + side_width * 0.5), 0.0, yaw
            )
            center = [round(x + dx, 6), round(y + dy, 6)]
            size = [round(side_width, 6), round(depth, 6)]
        return {
            "zone_id": f"use_{item['object_id']}_{intent}",
            "owner_id": item["object_id"],
            "purpose": intent,
            "center_xy_m": center,
            "size_xy_m": size,
            "yaw_rad": round(wrap_yaw(yaw), 10),
            "allowed_overlap_object_ids": allowed,
        }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def run(args: argparse.Namespace) -> int:
    intent_path = Path(args.intent)
    compiled_path = Path(args.compiled_output)
    report_path = Path(args.report_output)
    compiler = OrientationCompiler(
        load_hssd_lookup(Path(args.hssd_lookup)),
        _load_json(Path(args.semantic_mapping)),
        _load_json(Path(args.layout_rules)),
    )
    compiled, report = compiler.compile(_load_json(intent_path))
    compiled_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    compiled_path.write_text(
        json.dumps(compiled, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "scene_id": compiled["scene_id"],
                "furniture_count": len(compiled["furniture"]),
                "clearance_zone_count": len(compiled["use_clearance_zones"]),
                "compiled_output": str(compiled_path),
                "report_output": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile structured orientation and clearance intent."
    )
    parser.add_argument("--intent", required=True)
    parser.add_argument("--compiled-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--hssd-lookup", required=True)
    parser.add_argument("--semantic-mapping", required=True)
    parser.add_argument("--layout-rules", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
