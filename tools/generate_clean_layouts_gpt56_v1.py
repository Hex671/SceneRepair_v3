from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from tools.clean_layouts_lib import (
    AssetPool,
    SceneVerifier,
    bed_side_zone,
    behind_zone,
    canonical_json_hash,
    front_vector,
    front_zone,
    layout_hash,
    load_hssd_lookup,
    make_furniture,
    make_opening,
    near_duplicate_distance,
    object_rect,
    offset_from_object,
    place_in_front,
    relation,
    render_topdown,
    signed_separation,
    wall_pose,
    yaw_facing,
)


SEMANTIC_VERSION = "furniture_semantic_mapping_gpt56_v1"
RULES_VERSION = "furniture_layout_rules_gpt56_v1"
VERIFIER_VERSION = "clean_furniture_verifier_gpt56_v1"
GENERATOR_VERSION = "clean_furniture_generator_gpt56_v1"
DATASET_SCHEMA_VERSION = "scene_repair_v3_clean_furniture_scene_v1"


SOURCE_CATEGORY_TO_FROZEN = {
    "armchair": "armchair",
    "bed": "bed",
    "bench": "bench",
    "bookcase": "bookcase",
    "cabinet": "cabinet",
    "chair": "chair",
    "coffee table": "coffee_table",
    "console table": "console_table",
    "daybed": "bed",
    "desk": "desk",
    "dining table": "table",
    "double bed": "bed",
    "end table": "side_table",
    "floor lamp": "floor_lamp",
    "king bed": "bed",
    "nightstand": "nightstand",
    "ottoman": "ottoman",
    "sofa": "sofa",
    "swivel chair": "chair",
    "table": "table",
    "tv stand": "tv_bench",
    "wardrobe": "wardrobe",
}

FROZEN_CATEGORY_SEMANTICS = {
    "armchair": {"family": "seating", "functions": ["sit"]},
    "bed": {"family": "sleep_surface", "functions": ["sleep"]},
    "bench": {"family": "seating", "functions": ["sit"]},
    "bookcase": {"family": "storage", "functions": ["store", "support_objects"]},
    "cabinet": {"family": "storage", "functions": ["store", "contain"]},
    "chair": {"family": "seating", "functions": ["sit"]},
    "coffee_table": {"family": "support_surface", "functions": ["support_objects"]},
    "console_table": {"family": "support_surface", "functions": ["support_objects", "store"]},
    "desk": {"family": "support_surface", "functions": ["work_surface", "support_objects"]},
    "floor_lamp": {"family": "lighting", "functions": ["illuminate"]},
    "nightstand": {"family": "support_surface", "functions": ["support_objects", "store"]},
    "ottoman": {"family": "seating", "functions": ["sit", "foot_support"]},
    "side_table": {"family": "support_surface", "functions": ["support_objects"]},
    "sofa": {"family": "seating", "functions": ["sit"]},
    "table": {"family": "support_surface", "functions": ["dining_surface", "support_objects"]},
    "tv_bench": {"family": "support_surface", "functions": ["media_support", "support_objects"]},
    "wardrobe": {"family": "storage", "functions": ["store", "contain"]},
}

SOURCE_CATEGORIES = {
    category: sorted(
        source
        for source, mapped in SOURCE_CATEGORY_TO_FROZEN.items()
        if mapped == category
    )
    for category in FROZEN_CATEGORY_SEMANTICS
}
SOURCE_CATEGORIES["desk"] = ["desk"]

TARGET_BBOX = {
    "armchair": (0.78, 0.82, 0.9),
    "bed": (1.65, 2.05, 0.8),
    "bench": (1.25, 0.48, 0.55),
    "bookcase": (1.0, 0.36, 1.8),
    "cabinet": (1.15, 0.46, 1.35),
    "chair": (0.55, 0.58, 0.85),
    "coffee_table": (1.1, 0.7, 0.42),
    "console_table": (1.2, 0.4, 0.78),
    "desk": (1.35, 0.65, 0.76),
    "floor_lamp": (0.35, 0.35, 1.55),
    "nightstand": (0.52, 0.46, 0.55),
    "ottoman": (0.62, 0.55, 0.42),
    "side_table": (0.52, 0.5, 0.55),
    "sofa": (2.05, 0.9, 0.85),
    "table": (1.65, 0.9, 0.76),
    "tv_bench": (1.65, 0.45, 0.65),
    "wardrobe": (1.45, 0.6, 2.0),
}


def semantic_mapping() -> dict[str, Any]:
    payload = {
        "schema_version": SEMANTIC_VERSION,
        "frozen_vocab_source": "configs/furniture_vocab_bootstrap_v1.json",
        "hssd_id_is_model_feature": False,
        "source_category_to_frozen": SOURCE_CATEGORY_TO_FROZEN,
        "frozen_category_semantics": FROZEN_CATEGORY_SEMANTICS,
        "bbox_sources": {
            "static_z_up": "interaction_clearance.nonartic_clearance_v2.object_bbox_m",
            "official_articulated": "interaction_clearance.official_combined_clearance.obj_aabb",
            "retrieved_articulated": "interaction_clearance.articulated_swept_volume.bbox",
        },
        "bbox_axis_contract": {
            "room_output": "width=x, depth=y, height=z",
            "static_z_up": "identity [x,y,z]",
            "articulated_asset": "asset [x,y_up,z] -> room [x,z,y]",
        },
        "canonical_front_policy": {
            "asset_local_front_axis": "HSSD canonical_orientation_axis",
            "room_layout_front_convention": "local -Y rotated by geometric yaw",
            "semantic_front_used_only_when_flagged": True,
            "saved_model_pose": "geometric yaw only",
        },
    }
    payload["semantic_package_hash"] = canonical_json_hash(payload)
    return payload


def layout_rules() -> dict[str, Any]:
    return {
        "schema_version": RULES_VERSION,
        "verifier_version": VERIFIER_VERSION,
        "scene": {
            "single_room_only": True,
            "allowed_opening_kinds": ["door", "window"],
            "max_furniture": 17,
            "furniture_stage_only": True,
        },
        "transform": {
            "coordinate_frame": "room_local_z_up",
            "room_center": [0.0, 0.0, 0.0],
            "quaternion_order": "wxyz",
            "yaw_axis": "+Z",
            "yaw_unit": "radian",
            "yaw_range": "[-pi,pi)",
            "quaternion_norm_tolerance": 1.0e-6,
            "quaternion_component_tolerance": 1.0e-6,
        },
        "identity_bbox": {"absolute_tolerance_m": 1.0e-6},
        "geometry": {
            "room_bounds_tolerance_m": 1.0e-6,
            "penetration_tolerance_m": 0.005,
        },
        "openings": {
            "door_clearance_depth_m": 0.8,
            "window_clearance_depth_m": 0.5,
            "door_height_m": 2.05,
            "window_height_m": 1.2,
            "window_sill_height_m": 0.85,
        },
        "functional_relations": {
            "orientation_tolerance_deg": 22.5,
            "gap_tolerance_m": 0.035,
            "instance_endpoints_required": True,
        },
        "use_clearance": {
            "wardrobe_front_depth_m": 0.78,
            "cabinet_front_depth_m": 0.7,
            "bookcase_front_depth_m": 0.62,
            "desk_front_depth_m": 0.82,
            "seat_approach_depth_m": 0.38,
            "chair_behind_depth_m": 0.48,
            "bed_side_width_m": 0.62,
            "required_zone_categories": {
                "bedroom": ["bed", "wardrobe"],
                "living_room": ["sofa"],
                "dining_room": ["chair"],
                "home_office": ["desk", "chair"],
            },
        },
        "path": {
            "grid_resolution_m": 0.1,
            "human_radius_m": 0.28,
            "required": "door to primary functional zone",
        },
        "deduplication": {
            "exact_hash": "sha256 of quantized room/opening/furniture geometry",
            "near_duplicate_distance_threshold": 0.035,
            "room_dimension_prefilter_m": 0.35,
        },
    }


class SceneDesigner:
    def __init__(
        self,
        scene_id: str,
        room_type: str,
        length: float,
        width: float,
        pool: AssetPool,
        mapping: dict[str, Any],
        rules: dict[str, Any],
        *,
        design_summary: str,
    ) -> None:
        self.pool = pool
        self.mapping = mapping
        self.rules = rules
        self.scene: dict[str, Any] = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "scene_id": scene_id,
            "stage": "furniture",
            "coordinate_frame": "room_local_z_up",
            "quaternion_order": "wxyz",
            "yaw_range": "[-pi,pi)",
            "room_type": room_type,
            "length": float(length),
            "width": float(width),
            "validity": {
                "scene_id": True,
                "single_room": True,
                "room_type": True,
                "dimensions": True,
                "coordinate_contract": True,
                "stage_scope": True,
            },
            "openings": [],
            "furniture": [],
            "functional_partners": [],
            "audit": {
                "generator_version": GENERATOR_VERSION,
                "rules_version": RULES_VERSION,
                "semantic_package_hash": mapping["semantic_package_hash"],
                "design_summary": design_summary,
                "provenance": {
                    "layout_author": "GPT-5.6 independent layout reasoning",
                    "qwen_layout_used_as_answer": False,
                    "hssd_usage": "asset semantics, canonical orientation metadata, and bbox lookup only",
                },
                "use_clearance_zones": [],
                "primary_zone_goal_xy_m": [0.0, 0.0],
                "gpt_semantic_review": {
                    "status": "pending_visual_review",
                    "requires_human_review": True,
                    "notes": [],
                },
            },
        }
        self._counts: Counter[str] = Counter()

    @property
    def length(self) -> float:
        return float(self.scene["length"])

    @property
    def width(self) -> float:
        return float(self.scene["width"])

    def opening(self, kind: str, wall: str, along: float, *, opening_width: float) -> None:
        opening_rules = self.rules["openings"]
        if kind == "door":
            height = opening_rules["door_height_m"]
            sill = 0.0
            depth = opening_rules["door_clearance_depth_m"]
        else:
            height = opening_rules["window_height_m"]
            sill = opening_rules["window_sill_height_m"]
            depth = opening_rules["window_clearance_depth_m"]
        self.scene["openings"].append(
            make_opening(
                f"{kind}_{sum(value['kind'] == kind for value in self.scene['openings']):02d}",
                kind,
                wall,
                along,
                self.length,
                self.width,
                opening_width=opening_width,
                opening_height=height,
                sill_height=sill,
                clearance_depth=depth,
            )
        )

    def add_at(
        self,
        category: str,
        x: float,
        y: float,
        yaw: float,
        *,
        target_bbox: tuple[float, float, float] | None = None,
        reuse_group: str | None = None,
        rationale: str,
    ) -> dict[str, Any]:
        record, bbox = self.pool.select(
            category, target_bbox or TARGET_BBOX[category], reuse_group=reuse_group
        )
        self._counts[category] += 1
        object_id = f"{category}_{self._counts[category]:02d}"
        semantic = {"category": category, **self.mapping["frozen_category_semantics"][category]}
        obj = make_furniture(
            object_id, record, bbox, semantic, x, y, yaw, rationale=rationale
        )
        self.scene["furniture"].append(obj)
        return obj

    def add_wall(
        self,
        category: str,
        wall: str,
        along: float,
        *,
        target_bbox: tuple[float, float, float] | None = None,
        reuse_group: str | None = None,
        wall_gap: float = 0.08,
        rationale: str,
    ) -> dict[str, Any]:
        record, bbox = self.pool.select(
            category, target_bbox or TARGET_BBOX[category], reuse_group=reuse_group
        )
        x, y, yaw = wall_pose(
            wall, along, self.length, self.width, bbox, wall_gap=wall_gap
        )
        self._counts[category] += 1
        object_id = f"{category}_{self._counts[category]:02d}"
        semantic = {"category": category, **self.mapping["frozen_category_semantics"][category]}
        obj = make_furniture(
            object_id, record, bbox, semantic, x, y, yaw, rationale=rationale
        )
        self.scene["furniture"].append(obj)
        return obj

    def add_relative(
        self,
        category: str,
        target: dict[str, Any],
        local_x: float,
        local_y: float,
        yaw: float,
        *,
        target_bbox: tuple[float, float, float] | None = None,
        reuse_group: str | None = None,
        rationale: str,
    ) -> dict[str, Any]:
        x, y = offset_from_object(target, local_x, local_y)
        return self.add_at(
            category,
            x,
            y,
            yaw,
            target_bbox=target_bbox,
            reuse_group=reuse_group,
            rationale=rationale,
        )

    def partner(
        self,
        source: dict[str, Any],
        target: dict[str, Any],
        orientation_mode: str,
        target_strategy: str,
        desired_gap_range: tuple[float, float] | None,
        *,
        rationale: str,
    ) -> None:
        self.scene["functional_partners"].append(
            relation(
                source["object_id"],
                target["object_id"],
                orientation_mode,
                target_strategy,
                desired_gap_range,
                rationale=rationale,
            )
        )

    def zone(self, zone: dict[str, Any]) -> None:
        self.scene["audit"]["use_clearance_zones"].append(zone)

    def goal(self, point: tuple[float, float]) -> None:
        self.scene["audit"]["primary_zone_goal_xy_m"] = [
            round(point[0], 6),
            round(point[1], 6),
        ]


def add_bedroom_pair(designer: SceneDesigner, bed: dict[str, Any], count: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for side in range(count):
        sign = -1.0 if side == 0 else 1.0
        record, bbox = designer.pool.select(
            "nightstand", TARGET_BBOX["nightstand"], reuse_group=f"{designer.scene['scene_id']}_nightstands"
        )
        local_x = sign * (
            float(bed["bbox"]["width"]) * 0.5 + float(bbox["width"]) * 0.5 + 0.09
        )
        local_y = float(bed["bbox"]["depth"]) * 0.5 - float(bbox["depth"]) * 0.5
        x, y = offset_from_object(bed, local_x, local_y)
        designer._counts["nightstand"] += 1
        object_id = f"nightstand_{designer._counts['nightstand']:02d}"
        semantic = {"category": "nightstand", **designer.mapping["frozen_category_semantics"]["nightstand"]}
        obj = make_furniture(
            object_id,
            record,
            bbox,
            semantic,
            x,
            y,
            float(bed["yaw_rad"]),
            rationale="At the bed head, reachable from the sleeping surface without narrowing the foot circulation zone.",
        )
        designer.scene["furniture"].append(obj)
        designer.partner(
            obj,
            bed,
            "aligned",
            "place_beside",
            (0.07, 0.16),
            rationale="Bedside support is aligned with and immediately beside this bed instance.",
        )
        result.append(obj)
    return result


def build_bedrooms(pool: AssetPool, mapping: dict[str, Any], rules: dict[str, Any]) -> list[dict[str, Any]]:
    specs = [
        ("bedroom_001", 5.4, 4.5, "north", -0.45, "south", 1.75, "east", -1.25, "west", -1.2, 2, "bench"),
        ("bedroom_002", 5.0, 4.2, "west", 0.45, "south", 1.35, "north", 1.25, "east", 0.45, 1, "bench"),
        ("bedroom_003", 5.8, 4.8, "north", 0.35, "west", -1.45, "east", 1.35, "west", 1.05, 2, "armchair"),
        ("bedroom_004", 4.8, 4.0, "east", 0.25, "south", -1.25, "west", 1.1, "north", -1.35, 1, "bookcase"),
        ("bedroom_005", 5.6, 4.35, "south", -0.45, "west", 1.25, "north", 1.35, "east", 1.2, 2, "bench"),
        ("bedroom_006", 5.25, 4.65, "west", -0.4, "north", 1.55, "east", -1.25, "south", 0.65, 1, "none"),
        ("bedroom_007", 5.45, 4.25, "east", -0.45, "north", -1.7, "west", -1.1, "west", 0.55, 2, "desk"),
        ("bedroom_008", 6.0, 4.7, "north", -0.7, "east", -1.35, "west", 1.25, "east", 1.25, 2, "ottoman"),
    ]
    scenes: list[dict[str, Any]] = []
    for scene_id, length, width, bed_wall, bed_along, door_wall, door_along, window_wall, window_along, storage_wall, storage_along, nightstands, extra in specs:
        d = SceneDesigner(
            scene_id,
            "bedroom",
            length,
            width,
            pool,
            mapping,
            rules,
            design_summary=f"Bed anchored to the {bed_wall} wall with asymmetric supporting furniture and an opening-aware storage wall.",
        )
        d.opening("door", door_wall, door_along, opening_width=0.92)
        d.opening("window", window_wall, window_along, opening_width=1.45)
        bed_target = (1.7, 2.05, 0.82) if scene_id not in {"bedroom_003", "bedroom_008"} else (2.05, 2.15, 0.72)
        bed = d.add_wall(
            "bed",
            bed_wall,
            bed_along,
            target_bbox=bed_target,
            rationale="The head side is near a solid wall while the local -Y front points into the room for foot access.",
        )
        add_bedroom_pair(d, bed, nightstands)
        storage = d.add_wall(
            "wardrobe",
            storage_wall,
            storage_along,
            target_bbox=(1.35, 0.58, 1.95),
            rationale="Placed on an opening-free wall segment with its access side facing the room.",
        )
        d.zone(front_zone(storage, rules["use_clearance"]["wardrobe_front_depth_m"], "wardrobe_door_and_standing_clearance"))
        d.zone(bed_side_zone(bed, "left"))
        if nightstands >= 2:
            d.zone(bed_side_zone(bed, "right"))
        front = front_vector(float(bed["yaw_rad"]))
        bed_goal = (
            float(bed["x"]) + front[0] * (float(bed["bbox"]["depth"]) * 0.5 + 0.72),
            float(bed["y"]) + front[1] * (float(bed["bbox"]["depth"]) * 0.5 + 0.72),
        )
        d.goal(bed_goal)

        if extra in {"bench", "ottoman"}:
            extra_category = extra
            record, bbox = d.pool.select(extra_category, TARGET_BBOX[extra_category])
            x, y = place_in_front(bed, bbox, 0.48)
            d._counts[extra_category] += 1
            obj = make_furniture(
                f"{extra_category}_{d._counts[extra_category]:02d}",
                record,
                bbox,
                {"category": extra_category, **mapping["frozen_category_semantics"][extra_category]},
                x,
                y,
                float(bed["yaw_rad"]),
                rationale="A deliberate foot-of-bed seat/support element, separated from the bed egress zone.",
            )
            d.scene["furniture"].append(obj)
            d.partner(obj, bed, "aligned", "place_in_front", (0.42, 0.56), rationale="Foot-of-bed furniture is centered on and aligned with this bed.")
        elif extra == "desk":
            desk_wall = "south" if bed_wall != "south" and door_wall != "south" else "north"
            desk = d.add_wall("desk", desk_wall, -1.35 if scene_id == "bedroom_007" else 1.2, rationale="Compact secondary work zone uses a free wall segment rather than crowding the bed.")
            record, bbox = d.pool.select("chair", TARGET_BBOX["chair"])
            x, y = place_in_front(desk, bbox, 0.32)
            chair = d.add_at("chair", x, y, yaw_facing((x, y), (desk["x"], desk["y"])), target_bbox=(bbox["width"], bbox["depth"], bbox["height"]), rationale="Task chair faces its specific desk and leaves a pull-out path behind it.")
            d.partner(chair, desk, "facing", "face_target", (0.28, 0.4), rationale="This chair is the work seat for this desk instance.")
            d.zone(front_zone(desk, rules["use_clearance"]["desk_front_depth_m"], "desk_seated_work_clearance", allowed=[chair["object_id"]]))
            d.zone(behind_zone(chair, 0.45, "chair_pull_out_and_entry_clearance"))
        elif extra == "armchair":
            free_wall = "south" if door_wall != "south" and bed_wall != "south" else "east"
            arm_along = 1.25 if scene_id == "bedroom_003" else (-1.35 if free_wall == "south" else 1.2)
            arm = d.add_wall("armchair", free_wall, arm_along, wall_gap=0.25, rationale="Reading chair forms a secondary zone near daylight without forcing symmetry around the bed.")
            side = d.add_relative("side_table", arm, float(arm["bbox"]["width"]) * 0.5 + 0.38, 0.0, float(arm["yaw_rad"]), rationale="Reachable side support for the reading chair.")
            d.partner(side, arm, "aligned", "place_beside", (0.08, 0.35), rationale="The side table is beside this armchair instance.")
            d.zone(front_zone(arm, 0.4, "armchair_approach_clearance"))
        elif extra == "bookcase":
            bookcase_wall = "west" if scene_id == "bedroom_004" else "north"
            bookcase_along = -0.35 if scene_id == "bedroom_004" else 1.45
            bookcase = d.add_wall("bookcase", bookcase_wall, bookcase_along, rationale="Narrow book storage occupies an unused wall segment and faces inward.")
            d.zone(front_zone(bookcase, rules["use_clearance"]["bookcase_front_depth_m"], "bookcase_browsing_clearance"))
        scenes.append(d.scene)
    return scenes


def build_living_rooms(pool: AssetPool, mapping: dict[str, Any], rules: dict[str, Any]) -> list[dict[str, Any]]:
    specs = [
        ("living_room_001", 6.2, 4.8, "north", -0.6, "south", 1.55, "west", -1.5, 2, True, "console_table"),
        ("living_room_002", 5.8, 4.5, "west", 0.35, "east", -1.2, "south", 1.55, 1, True, "floor_lamp"),
        ("living_room_003", 6.6, 5.0, "south", 0.5, "north", -1.65, "east", 1.55, 2, False, "console_table"),
        ("living_room_004", 5.6, 4.4, "east", -0.35, "west", 1.2, "north", 1.45, 1, True, "ottoman"),
        ("living_room_005", 6.0, 4.7, "north", 0.65, "south", -1.55, "east", -1.45, 2, False, "floor_lamp"),
        ("living_room_006", 6.4, 4.9, "west", -0.4, "east", 1.35, "north", -1.65, 1, True, "console_table"),
        ("living_room_007", 5.9, 4.9, "south", -0.55, "north", 1.55, "west", 1.55, 2, True, "floor_lamp"),
        ("living_room_008", 6.8, 5.1, "east", 0.5, "west", -1.45, "south", -1.75, 2, False, "console_table"),
    ]
    result: list[dict[str, Any]] = []
    opposite = {"north": "south", "south": "north", "east": "west", "west": "east"}
    for scene_id, length, width, sofa_wall, sofa_along, door_wall, door_along, window_wall, window_along, arm_count, side_table, extra in specs:
        d = SceneDesigner(scene_id, "living_room", length, width, pool, mapping, rules, design_summary=f"Conversation group organized from a {sofa_wall}-wall sofa toward a central table and media wall, with deliberately asymmetric secondary seating.")
        d.opening("door", door_wall, door_along, opening_width=0.95)
        d.opening("window", window_wall, window_along, opening_width=1.65)
        sofa = d.add_wall("sofa", sofa_wall, sofa_along, target_bbox=(2.05, 0.9, 0.85), rationale="Primary seating has a protected back and faces the room's shared activity center.")
        record, bbox = d.pool.select("coffee_table", TARGET_BBOX["coffee_table"])
        coffee_x, coffee_y = place_in_front(sofa, bbox, 0.58)
        d._counts["coffee_table"] += 1
        coffee = make_furniture("coffee_table_01", record, bbox, {"category": "coffee_table", **mapping["frozen_category_semantics"]["coffee_table"]}, coffee_x, coffee_y, float(sofa["yaw_rad"]), rationale="Centered within reach of the sofa but outside its approach clearance.")
        d.scene["furniture"].append(coffee)
        d.partner(sofa, coffee, "facing", "face_target", (0.52, 0.66), rationale="The primary sofa faces and reaches this coffee table instance.")
        tv_wall = opposite[sofa_wall]
        tv_along = (-door_along * 0.85) if door_wall == tv_wall else (-sofa_along * 0.35)
        tv = d.add_wall("tv_bench", tv_wall, tv_along, rationale="Media support sits on the opposing wall so the sofa's sight line remains direct.")
        d.partner(sofa, tv, "facing", "face_target", None, rationale="The primary sofa is oriented toward this media support instance.")
        d.partner(coffee, tv, "aligned", "align_with_target", None, rationale="Coffee table and media support share the main room axis.")
        d.zone(front_zone(sofa, rules["use_clearance"]["seat_approach_depth_m"], "sofa_sit_stand_clearance"))

        lateral_axis = object_rect(coffee).axes[0]
        for idx in range(arm_count):
            sign = -1.0 if idx == 0 else 1.0
            record, arm_bbox = d.pool.select("armchair", TARGET_BBOX["armchair"])
            distance = float(coffee["bbox"]["width"]) * 0.5 + float(arm_bbox["depth"]) * 0.5 + 0.4
            x = float(coffee["x"]) + lateral_axis[0] * sign * distance
            y = float(coffee["y"]) + lateral_axis[1] * sign * distance
            d._counts["armchair"] += 1
            arm = make_furniture(f"armchair_{d._counts['armchair']:02d}", record, arm_bbox, {"category": "armchair", **mapping["frozen_category_semantics"]["armchair"]}, x, y, yaw_facing((x, y), (coffee["x"], coffee["y"])), rationale="Angled secondary seating closes the conversation group without mirroring every item.")
            d.scene["furniture"].append(arm)
            d.partner(arm, coffee, "facing", "face_target", (0.3, 0.72), rationale="This armchair faces the shared coffee table.")
            d.zone(front_zone(arm, 0.34, "armchair_sit_stand_clearance"))
        side_table_sign = 0.0
        if side_table:
            local_sign = -1.0 if scene_id in {"living_room_002", "living_room_006"} else 1.0
            side_table_sign = local_sign
            table = d.add_relative("side_table", sofa, local_sign * (float(sofa["bbox"]["width"]) * 0.5 + 0.38), 0.12, float(sofa["yaw_rad"]), rationale="A single reachable side surface keeps the seating group useful without forced bilateral symmetry.")
            d.partner(table, sofa, "aligned", "place_beside", (0.08, 0.38), rationale="This side table serves the adjacent sofa end.")
        if extra == "console_table":
            free_wall = window_wall if window_wall != sofa_wall else door_wall
            along = -window_along * 0.45
            console = d.add_wall("console_table", free_wall, along, rationale="Slim perimeter storage occupies a circulation-safe wall segment outside the main seating cluster.")
        elif extra == "floor_lamp":
            lamp_sign = -side_table_sign if side_table_sign else 1.0
            lamp = d.add_relative("floor_lamp", sofa, lamp_sign * (float(sofa["bbox"]["width"]) * 0.5 + 0.38), float(sofa["bbox"]["depth"]) * 0.15, float(sofa["yaw_rad"]), rationale="Task lighting is adjacent to, rather than isolated from, the primary seating.")
            d.partner(lamp, sofa, "none", "place_beside", (0.05, 0.45), rationale="The floor lamp serves this sofa end.")
        else:
            lamp = d.add_relative("floor_lamp", sofa, -float(sofa["bbox"]["width"]) * 0.5 - 0.38, 0.08, float(sofa["yaw_rad"]), rationale="A single task light supports the seating group without occupying the coffee-table corridor.")
            d.partner(lamp, sofa, "none", "place_beside", (0.05, 0.45), rationale="The floor lamp serves this sofa end.")
        candidate_goals = [
            offset_from_object(coffee, float(coffee["bbox"]["width"]) * 0.5 + 0.65, 0.0),
            offset_from_object(coffee, -float(coffee["bbox"]["width"]) * 0.5 - 0.65, 0.0),
        ]
        door = next(value for value in d.scene["openings"] if value["kind"] == "door")
        door_xy = door["clearance_center_xyz_m"][:2]
        d.goal(min(candidate_goals, key=lambda point: math.hypot(point[0] - door_xy[0], point[1] - door_xy[1])))
        result.append(d.scene)
    return result


def _add_dining_chairs(d: SceneDesigner, table: dict[str, Any], count: int) -> list[dict[str, Any]]:
    record, bbox = d.pool.select("chair", (0.52, 0.56, 0.84), reuse_group=f"{d.scene['scene_id']}_dining_chairs")
    chair_width, chair_depth = float(bbox["width"]), float(bbox["depth"])
    placements: list[tuple[float, float]] = []
    if count == 4:
        placements = [(0.0, 1.0), (0.0, -1.0), (1.0, 0.0), (-1.0, 0.0)]
    else:
        placements = [(-0.28, 1.0), (0.28, 1.0), (-0.28, -1.0), (0.28, -1.0), (1.0, 0.0), (-1.0, 0.0)]
    chairs: list[dict[str, Any]] = []
    table_width, table_depth = float(table["bbox"]["width"]), float(table["bbox"]["depth"])
    for local_u, local_v in placements:
        if abs(local_v) > 0.5:
            local_x = local_u * table_width
            local_y = local_v * (table_depth * 0.5 + chair_depth * 0.5 + 0.43)
        else:
            local_x = local_u * (table_width * 0.5 + chair_depth * 0.5 + 0.43)
            local_y = 0.0
        x, y = offset_from_object(table, local_x, local_y)
        table_yaw = float(table["yaw_rad"])
        if local_v > 0.5:
            chair_yaw = table_yaw
        elif local_v < -0.5:
            chair_yaw = table_yaw + math.pi
        elif local_u > 0.5:
            chair_yaw = table_yaw - math.pi * 0.5
        else:
            chair_yaw = table_yaw + math.pi * 0.5
        d._counts["chair"] += 1
        chair = make_furniture(f"chair_{d._counts['chair']:02d}", record, bbox, {"category": "chair", **d.mapping["frozen_category_semantics"]["chair"]}, x, y, chair_yaw, rationale="Dining seat is parallel to its table edge, faces its place setting, and has pull-out space behind.")
        d.scene["furniture"].append(chair)
        d.pool._used[record["hssd_id"]] += 1 if chairs else 0
        d.partner(chair, table, "facing", "face_target", (0.31, 0.49), rationale="This chair is assigned to this dining table instance.")
        d.zone(behind_zone(chair, 0.45, "dining_chair_pull_out_clearance"))
        chairs.append(chair)
    return chairs


def build_dining_rooms(pool: AssetPool, mapping: dict[str, Any], rules: dict[str, Any]) -> list[dict[str, Any]]:
    specs = [
        ("dining_room_001", 5.4, 4.4, 0.0, 4, "south", 1.7, "north", -1.35, "east", 0.9, "cabinet"),
        ("dining_room_002", 5.8, 5.2, math.pi * 0.5, 6, "west", -1.45, "east", 1.35, "north", 1.25, "console_table"),
        ("dining_room_003", 5.1, 4.2, 0.0, 4, "east", -1.3, "west", 1.25, "north", -1.2, "cabinet"),
        ("dining_room_004", 6.0, 5.3, math.pi * 0.5, 6, "south", -1.75, "north", 1.55, "west", 1.15, "console_table"),
        ("dining_room_005", 5.6, 4.5, 0.0, 6, "north", -1.65, "south", 1.35, "east", -1.05, "cabinet"),
        ("dining_room_006", 5.2, 4.7, math.pi * 0.5, 4, "west", 1.45, "east", -1.35, "south", -1.15, "console_table"),
        ("dining_room_007", 6.2, 4.9, 0.0, 6, "east", 1.55, "west", -1.45, "north", -1.45, "cabinet"),
    ]
    scenes: list[dict[str, Any]] = []
    for scene_id, length, width, table_yaw, chair_count, door_wall, door_along, window_wall, window_along, storage_wall, storage_along, storage_cat in specs:
        d = SceneDesigner(scene_id, "dining_room", length, width, pool, mapping, rules, design_summary=f"A {chair_count}-seat dining group is centered for circulation, with service storage on a separate wall and no decorative filler.")
        d.opening("door", door_wall, door_along, opening_width=0.95)
        d.opening("window", window_wall, window_along, opening_width=1.55)
        table_target = (1.65, 0.9, 0.76) if chair_count == 4 else (2.0, 0.92, 0.76)
        table = d.add_at("table", 0.0, 0.08 if scene_id in {"dining_room_003", "dining_room_006"} else 0.0, table_yaw, target_bbox=table_target, rationale="The dining surface is central enough for chair pull-out clearance on all occupied sides.")
        chairs = _add_dining_chairs(d, table, chair_count)
        storage = d.add_wall(storage_cat, storage_wall, storage_along, rationale="Dining service storage is reachable but separated from chair pull-out zones.")
        d.zone(front_zone(storage, 0.62 if storage_cat == "console_table" else 0.7, "dining_service_storage_clearance"))
        open_side = chairs[0]
        goal_front = front_vector(float(open_side["yaw_rad"]))
        d.goal((float(open_side["x"]) - goal_front[0] * 0.72, float(open_side["y"]) - goal_front[1] * 0.72))
        scenes.append(d.scene)
    return scenes


def build_home_offices(pool: AssetPool, mapping: dict[str, Any], rules: dict[str, Any]) -> list[dict[str, Any]]:
    specs = [
        ("home_office_001", 4.8, 4.0, "north", -0.55, "south", 1.35, "east", -1.1, "west", 0.9, "bookcase", True),
        ("home_office_002", 5.2, 4.2, "west", 0.35, "east", -1.35, "north", 1.25, "south", -1.15, "cabinet", False),
        ("home_office_003", 5.0, 4.5, "south", 0.65, "north", -1.45, "west", 1.25, "east", -1.15, "bookcase", True),
        ("home_office_004", 4.7, 4.1, "east", -0.45, "west", 1.25, "south", 1.1, "north", 0.8, "cabinet", False),
        ("home_office_005", 5.4, 4.3, "north", 0.75, "east", -1.35, "west", -1.2, "south", 1.25, "bookcase", True),
        ("home_office_006", 5.1, 4.6, "west", -0.55, "south", 1.45, "east", 1.25, "north", 1.15, "cabinet", True),
        ("home_office_007", 5.6, 4.4, "south", -0.75, "west", -1.45, "north", 1.3, "east", 1.15, "bookcase", False),
    ]
    scenes: list[dict[str, Any]] = []
    for scene_id, length, width, desk_wall, desk_along, door_wall, door_along, window_wall, window_along, storage_wall, storage_along, storage_cat, lounge in specs:
        d = SceneDesigner(scene_id, "home_office", length, width, pool, mapping, rules, design_summary=f"Primary desk uses the {desk_wall} wall, with a dedicated work chair, accessible storage, and {'a small reading zone' if lounge else 'a compact support surface'}.")
        d.opening("door", door_wall, door_along, opening_width=0.92)
        d.opening("window", window_wall, window_along, opening_width=1.45)
        desk = d.add_wall("desk", desk_wall, desk_along, target_bbox=(1.35, 0.64, 0.76), rationale="The work surface uses a solid wall segment and faces its chair into the room.")
        record, bbox = d.pool.select("chair", (0.58, 0.6, 0.86))
        x, y = place_in_front(desk, bbox, 0.34)
        d._counts["chair"] += 1
        chair = make_furniture("chair_01", record, bbox, {"category": "chair", **mapping["frozen_category_semantics"]["chair"]}, x, y, yaw_facing((x, y), (desk["x"], desk["y"])), rationale="Task chair faces this desk and retains a pull-out/entry zone behind.")
        d.scene["furniture"].append(chair)
        d.partner(chair, desk, "facing", "face_target", (0.3, 0.4), rationale="This chair is the working seat for this exact desk.")
        d.zone(front_zone(desk, rules["use_clearance"]["desk_front_depth_m"], "desk_seated_work_clearance", allowed=[chair["object_id"]]))
        d.zone(behind_zone(chair, rules["use_clearance"]["chair_behind_depth_m"], "task_chair_pull_out_clearance"))
        storage = d.add_wall(storage_cat, storage_wall, storage_along, rationale="Reference/storage furniture occupies an opening-free wall and faces accessible room space.")
        d.zone(front_zone(storage, 0.65, "office_storage_access_clearance"))
        if lounge:
            free_wall = window_wall
            arm_along = 1.15 if scene_id == "home_office_001" else (-0.1 if scene_id == "home_office_006" else -window_along * 0.45)
            arm = d.add_wall("armchair", free_wall, arm_along, wall_gap=0.24, rationale="A secondary reading chair sits near daylight but outside the work-chair movement envelope.")
            side = d.add_relative("side_table", arm, float(arm["bbox"]["width"]) * 0.5 + 0.38, 0.0, float(arm["yaw_rad"]), rationale="Small reachable support for the reading chair.")
            d.partner(side, arm, "aligned", "place_beside", (0.08, 0.34), rationale="This side table belongs to the adjacent reading chair.")
            d.zone(front_zone(arm, 0.38, "reading_chair_approach_clearance"))
        else:
            support_wall = window_wall
            console = d.add_wall("console_table", support_wall, -window_along * 0.5, rationale="A slim printer/reference surface adds function without creating an isolated center object.")
            d.zone(front_zone(console, 0.48, "office_support_surface_access"))
        chair_front = front_vector(float(chair["yaw_rad"]))
        d.goal((float(chair["x"]) - chair_front[0] * 0.72, float(chair["y"]) - chair_front[1] * 0.72))
        scenes.append(d.scene)
    return scenes


def write_readme(output_dir: Path, mapping: dict[str, Any], rules: dict[str, Any]) -> None:
    text = f"""# GPT-5.6 Furniture Pilot Clean Layouts v1

This directory contains 30 independently designed, single-room Furniture-stage clean layouts for SceneRepair_v3. Existing Qwen layouts were not used as target answers.

## Contract

- Scene schema: `{DATASET_SCHEMA_VERSION}`
- Coordinate frame: `room_local_z_up`, room center `(0,0)`, x in `[-length/2,length/2]`, y in `[-width/2,width/2]`
- Quaternion order: `wxyz`; yaw is geometric rotation around `+Z`, radians, wrapped to `[-pi,pi)`
- Openings: door and window only; clearance AABBs follow SceneExpert depths (door 0.8 m, window 0.5 m)
- Furniture: at most 17 movable items; no manipulands, wall decoration, or ceiling objects
- HSSD IDs: lookup/provenance/split identity only, never a model feature
- Semantic package hash: `{mapping['semantic_package_hash']}`
- Rules version: `{rules['schema_version']}`
- Verifier version: `{rules['verifier_version']}`

## Bbox provenance

Static HSSD bbox values come from `interaction_clearance.nonartic_clearance_v2.object_bbox_m` in Z-up `[x,y,z]`. Official/retrieved articulated records use asset `[x,y_up,z]`; output width/depth/height is deterministically converted to room `[x,z,y]`. Every object records its source path and conversion under `audit`.

## Verification

`validation_report.json` separates deterministic geometry/identity/relation/path checks from GPT semantic visual review. The deterministic verifier checks finite transforms, HSSD identity and bbox equality, room bounds, Furniture OBB penetration, opening clearance, instance-level functional partner orientation/gap, declared use clearances, path reachability, and exact/near duplicate layouts.

## Known limitations

- The pilot uses rectangular rooms and 2D floor-plan OBB checks; it does not run mesh-level collision or physics simulation.
- HSSD canonical front is reliable only where its semantic-front flag is true. Final training pose remains geometric yaw.
- Grid reachability approximates a 0.56 m diameter person and does not model articulated door swing.
- GPT semantic review judges functional plausibility from top-down plots; it is not a deterministic proof and is listed separately in the report.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def summarize(scenes: list[dict[str, Any]], reports: list[dict[str, Any]], pool: AssetPool) -> dict[str, Any]:
    room_types = Counter(scene["room_type"] for scene in scenes)
    categories = Counter(obj["category"] for scene in scenes for obj in scene["furniture"])
    families = Counter(obj["family"] for scene in scenes for obj in scene["furniture"])
    functions = Counter(function for scene in scenes for obj in scene["furniture"] for function in obj["functions"])
    counts = [len(scene["furniture"]) for scene in scenes]
    lengths = [float(scene["length"]) for scene in scenes]
    widths = [float(scene["width"]) for scene in scenes]
    all_hssd = [obj["hssd_id"] for scene in scenes for obj in scene["furniture"]]
    check_names = list(reports[0]["checks"]) if reports else []
    return {
        "accepted_scene_count": len(scenes),
        "room_type_coverage": dict(sorted(room_types.items())),
        "category_coverage": dict(sorted(categories.items())),
        "family_coverage": dict(sorted(families.items())),
        "function_coverage": dict(sorted(functions.items())),
        "furniture_count_distribution": {
            "min": min(counts), "max": max(counts), "mean": round(mean(counts), 3), "histogram": dict(sorted(Counter(counts).items()))
        },
        "room_dimension_distribution_m": {
            "length": {"min": min(lengths), "max": max(lengths), "mean": round(mean(lengths), 3)},
            "width": {"min": min(widths), "max": max(widths), "mean": round(mean(widths), 3)},
        },
        "deterministic_check_pass_rates": {
            name: round(sum(report["checks"][name]["passed"] for report in reports) / len(reports), 6)
            for name in check_names
        },
        "unique_hssd_ids": len(set(all_hssd)),
        "hssd_placements": len(all_hssd),
        "hssd_repeat_placement_rate": round(1.0 - len(set(all_hssd)) / len(all_hssd), 6),
        "hssd_use_count_histogram": dict(
            sorted(Counter(Counter(all_hssd).values()).items())
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    workspace = Path(__file__).resolve().parents[2]
    parser.add_argument("--hssd-lookup", type=Path, default=workspace / "hssd-annotations" / "data" / "hssd_annotation_lookup.json.gz")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "clean_layouts_gpt56_v1")
    parser.add_argument(
        "--semantic-reviews",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "clean_layout_semantic_reviews_gpt56_v1.json",
    )
    args = parser.parse_args()

    mapping = semantic_mapping()
    rules = layout_rules()
    records = load_hssd_lookup(args.hssd_lookup)
    pool = AssetPool(records, SOURCE_CATEGORIES)
    scenes = [
        *build_bedrooms(pool, mapping, rules),
        *build_living_rooms(pool, mapping, rules),
        *build_dining_rooms(pool, mapping, rules),
        *build_home_offices(pool, mapping, rules),
    ]
    if len(scenes) != 30:
        raise RuntimeError(f"expected 30 scene recipes, got {len(scenes)}")
    review_package = json.loads(args.semantic_reviews.read_text(encoding="utf-8"))
    reviewed_ids = set(review_package["reviewed_scene_ids"])
    scene_ids = {scene["scene_id"] for scene in scenes}
    if reviewed_ids != scene_ids:
        raise RuntimeError("semantic review package does not cover exactly the 30 pilot scenes")
    default_review = review_package["default_review"]
    overrides = review_package.get("scene_overrides", {})
    for scene in scenes:
        scene["audit"]["gpt_semantic_review"] = {
            **default_review,
            **overrides.get(scene["scene_id"], {}),
            "review_schema_version": review_package["schema_version"],
            "review_basis": review_package["review_basis"],
        }

    verifier = SceneVerifier(records, mapping, rules)
    accepted: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    hashes: set[str] = set()
    near_threshold = rules["deduplication"]["near_duplicate_distance_threshold"]
    for scene in scenes:
        report = verifier.verify(scene)
        current_hash = layout_hash(scene)
        scene["audit"]["layout_hash"] = current_hash
        duplicate_reason: dict[str, Any] | None = None
        if current_hash in hashes:
            duplicate_reason = {"type": "exact_layout_duplicate"}
        else:
            for previous in accepted:
                distance = near_duplicate_distance(scene, previous)
                if distance < near_threshold:
                    duplicate_reason = {"type": "near_layout_duplicate", "target_scene_id": previous["scene_id"], "distance": distance}
                    break
        if duplicate_reason:
            report["passed"] = False
            report["violations"].append(duplicate_reason)
        if report["passed"]:
            accepted.append(scene)
            reports.append(report)
            hashes.add(current_hash)
        else:
            rejected.append({"scene_id": scene["scene_id"], "violations": report["violations"]})

    output_dir = args.output.resolve()
    if output_dir.exists():
        for name in ("scenes", "visualizations"):
            target = output_dir / name
            if target.exists():
                shutil.rmtree(target)
        for name in ("clean_manifest.jsonl", "semantic_mapping.json", "layout_rules_v1.json", "validation_report.json", "README.md"):
            target = output_dir / name
            if target.exists():
                target.unlink()
    (output_dir / "scenes").mkdir(parents=True, exist_ok=True)
    (output_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    (output_dir / "semantic_mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "layout_rules_v1.json").write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_rows: list[dict[str, Any]] = []
    for scene, report in zip(accepted, reports):
        scene_path = output_dir / "scenes" / f"{scene['scene_id']}.json"
        scene_path.write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")
        render_topdown(scene, output_dir / "visualizations" / f"{scene['scene_id']}.png")
        manifest_rows.append({
            "scene_id": scene["scene_id"],
            "room_type": scene["room_type"],
            "hssd_ids": [obj["hssd_id"] for obj in scene["furniture"]],
            "category_combination": sorted(obj["category"] for obj in scene["furniture"]),
            "rules_version": RULES_VERSION,
            "semantic_package_hash": mapping["semantic_package_hash"],
            "verifier_version": VERIFIER_VERSION,
            "layout_hash": scene["audit"]["layout_hash"],
        })
    manifest_rows.sort(key=lambda row: row["scene_id"])
    (output_dir / "clean_manifest.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in manifest_rows), encoding="utf-8")
    summary = summarize(accepted, reports, pool) if accepted else {}
    validation = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "verifier_version": VERIFIER_VERSION,
        "rules_version": RULES_VERSION,
        "semantic_package_hash": mapping["semantic_package_hash"],
        "candidate_count": len(scenes),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "deterministic_overall_pass_rate": round(len(accepted) / len(scenes), 6),
        "rejection_reasons": dict(sorted(Counter(v["type"] for item in rejected for v in item["violations"]).items())),
        "rejected_candidates": rejected,
        "scene_reports": [
            {"scene_id": scene["scene_id"], **report, "gpt_semantic_review": scene["audit"]["gpt_semantic_review"]}
            for scene, report in zip(accepted, reports)
        ],
        "summary": summary,
        "deterministic_conclusions": [
            "identity_bbox", "finite_transform", "room_bounds", "furniture_collision", "opening_clearance", "functional_relations", "use_clearance", "path_reachability", "layout_hash_deduplication"
        ],
        "gpt_semantic_judgment_scope": [
            "human-use plausibility", "non-mechanical wall placement", "functional grouping", "orientation interpretation", "visual circulation quality"
        ],
        "gpt_semantic_review_counts": dict(
            sorted(
                Counter(
                    scene["audit"]["gpt_semantic_review"]["status"]
                    for scene in accepted
                ).items()
            )
        ),
        "manual_judgment_required_scene_ids": [
            scene["scene_id"]
            for scene in accepted
            if scene["audit"]["gpt_semantic_review"]["requires_human_review"]
        ],
    }
    (output_dir / "validation_report.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(output_dir, mapping, rules)
    print(json.dumps({"output": str(output_dir), "accepted": len(accepted), "rejected": len(rejected), "rejection_reasons": validation["rejection_reasons"], "summary": summary}, ensure_ascii=False, indent=2))
    return 0 if len(accepted) == 30 else 2


if __name__ == "__main__":
    raise SystemExit(main())
