from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from scene_repair_v3 import (
    FrozenFunctionalPartnerRules,
    FurnitureGraphBuilder,
    FurnitureInput,
    FurnitureVocabularies,
    OpeningInput,
    RoomInput,
    SceneInput,
)
from tools.clean_layouts_lib import (
    OrientedRectangle,
    SceneVerifier,
    canonical_json_hash,
    front_vector,
    grid_path_exists,
    hssd_bbox,
    layout_hash,
    load_hssd_lookup,
    make_furniture,
    make_opening,
    near_duplicate_distance,
    object_rect,
    render_topdown,
    room_margins,
    signed_separation,
)


PIPELINE_VERSION = "authored_single_scene_validator_v2"
SCENE_SCHEMA_VERSION = "scene_repair_v3_clean_furniture_scene_v1"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_proposal_schema(proposal: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version",
        "scene_id",
        "room_type",
        "coordinate_frame",
        "quaternion_order",
        "yaw_range",
        "room",
        "openings",
        "furniture",
        "use_clearance_zones",
        "path_targets",
        "primary_zone_goal_xy_m",
        "authorship",
    }
    missing = sorted(required - set(proposal))
    if missing:
        errors.append(f"missing proposal fields: {missing}")
    if proposal.get("schema_version") != "authored_furniture_proposal_v1":
        errors.append("unsupported proposal schema")
    if proposal.get("coordinate_frame") != "room_local_z_up":
        errors.append("coordinate_frame must be room_local_z_up")
    if proposal.get("quaternion_order") != "wxyz":
        errors.append("quaternion_order must be wxyz")
    if proposal.get("yaw_range") != "[-pi,pi)":
        errors.append("yaw_range must be [-pi,pi)")
    if "functional_partners" in proposal:
        errors.append("authored proposal must not contain functional_partners")
    furniture = proposal.get("furniture")
    if not isinstance(furniture, list) or not furniture:
        errors.append("furniture must be a non-empty list")
    elif len(furniture) > 17:
        errors.append("furniture count exceeds 17")
    else:
        ids: set[str] = set()
        for index, item in enumerate(furniture):
            if not isinstance(item, dict):
                errors.append(f"furniture[{index}] must be an object")
                continue
            item_required = {
                "object_id",
                "hssd_id",
                "category",
                "x",
                "y",
                "z",
                "yaw_rad",
                "placement_rationale",
            }
            item_missing = sorted(item_required - set(item))
            if item_missing:
                errors.append(f"furniture[{index}] missing {item_missing}")
            object_id = item.get("object_id")
            if object_id in ids:
                errors.append(f"duplicate object_id {object_id!r}")
            ids.add(object_id)
            if not str(item.get("placement_rationale", "")).strip():
                errors.append(f"furniture[{index}] has no placement rationale")
    if proposal.get("room_type") not in {
        "bedroom",
        "living_room",
        "dining_room",
        "home_office",
    }:
        errors.append("room_type is outside the single-room scope")
    for opening in proposal.get("openings", []):
        if opening.get("kind") not in {"door", "window"}:
            errors.append(f"unsupported opening kind {opening.get('kind')!r}")
    return errors


def _door_path_start(door: dict[str, Any]) -> tuple[float, float]:
    cx, cy, _ = door["clearance_center_xyz_m"]
    sx, sy, _ = door["clearance_size_xyz_m"]
    nx, ny = door["interior_normal_xy"]
    return cx + nx * sx * 0.5, cy + ny * sy * 0.5


def _make_scene(
    proposal: dict[str, Any],
    records: dict[str, dict[str, Any]],
    semantic_mapping: dict[str, Any],
    frozen_rules: FrozenFunctionalPartnerRules,
    source_meta: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    errors = _validate_proposal_schema(proposal)
    if errors:
        raise ValueError("; ".join(errors))
    room = proposal["room"]
    length = float(room["length_m"])
    width = float(room["width_m"])
    openings = [
        make_opening(
            item["opening_id"],
            item["kind"],
            item["wall"],
            float(item["along_m"]),
            length,
            width,
            opening_width=float(item["opening_width_m"]),
            opening_height=float(item["opening_height_m"]),
            sill_height=float(item["sill_height_m"]),
            clearance_depth=float(item["clearance_depth_m"]),
        )
        for item in proposal["openings"]
    ]
    furniture: list[dict[str, Any]] = []
    for authored in proposal["furniture"]:
        hssd_id = str(authored["hssd_id"]).lower()
        record = records.get(hssd_id)
        if record is None:
            raise ValueError(f"unknown HSSD id {hssd_id}")
        bbox = hssd_bbox(record)
        if bbox is None:
            raise ValueError(f"HSSD id {hssd_id} has no supported bbox")
        source_category = str(record.get("category", "")).lower()
        frozen_category = semantic_mapping["source_category_to_frozen"].get(
            source_category
        )
        if frozen_category != authored["category"]:
            raise ValueError(
                f"{authored['object_id']} category {authored['category']!r} does "
                f"not match HSSD source category {source_category!r}"
            )
        semantics = semantic_mapping["frozen_category_semantics"][frozen_category]
        obj = make_furniture(
            authored["object_id"],
            record,
            bbox,
            {"category": frozen_category, **semantics},
            float(authored["x"]),
            float(authored["y"]),
            float(authored["yaw_rad"]),
            rationale=authored["placement_rationale"],
        )
        placement_dof = record.get("placement_dof") or {}
        obj["z"] = float(authored["z"])
        obj["movable"] = bool(float(placement_dof.get("dof", 0)) > 0)
        obj["audit"].update(
            {
                "placement_dof": placement_dof,
                "functional_partner_annotation": (
                    record.get("interaction_clearance") or {}
                ).get("functional_partners"),
                "post_replacement": record.get("post_replacement"),
            }
        )
        furniture.append(obj)

    furniture_inputs = tuple(_furniture_input(value) for value in furniture)
    edge_matches = frozen_rules.build_edge_matches(furniture_inputs)
    functional_partners = [
        {
            "source_id": match.source_id,
            "target_id": match.target_id,
            "orientation_mode": "none",
            "target_strategy": "none",
            "desired_gap_range": None,
            "validity": {
                "source_id": True,
                "target_id": True,
                "orientation_mode": True,
                "target_strategy": True,
                "desired_gap_range": False,
                "relation": True,
            },
            "audit": {
                "source": "FrozenFunctionalPartnerRules.build_edge_matches",
                "rules_schema_version": "hssd_functional_partner_rules_v1",
                "rules_sha256": frozen_rules.rules_sha256,
                "source_hssd_id": match.source_hssd_id,
                "matched_original_partner_category": (
                    match.annotated_partner_category
                ),
                "category_match_mode": match.category_match_mode,
                "compatibility_rule_id": match.compatibility_rule_id,
                "compatibility_schema_version": (
                    frozen_rules.compatibility_schema_version
                ),
                "compatibility_rules_sha256": (
                    frozen_rules.compatibility_rules_sha256
                ),
                "asset_keyed_only": True,
                "category_inheritance": False,
                "commonsense_completion": False,
            },
        }
        for match in edge_matches
    ]
    zones = [
        {
            **item,
            "center_xy_m": [float(value) for value in item["center_xy_m"]],
            "size_xy_m": [float(value) for value in item["size_xy_m"]],
            "yaw_rad": float(item["yaw_rad"]),
            "validity": True,
        }
        for item in proposal["use_clearance_zones"]
    ]
    scene = {
        "schema_version": SCENE_SCHEMA_VERSION,
        "scene_id": proposal["scene_id"],
        "stage": "furniture",
        "coordinate_frame": proposal["coordinate_frame"],
        "quaternion_order": proposal["quaternion_order"],
        "yaw_range": proposal["yaw_range"],
        "room_type": proposal["room_type"],
        "length": length,
        "width": width,
        "validity": {
            "scene_id": True,
            "single_room": True,
            "room_type": True,
            "dimensions": True,
            "coordinate_contract": True,
            "stage_scope": True,
        },
        "openings": openings,
        "furniture": furniture,
        "functional_partners": functional_partners,
        "audit": {
            "pipeline_version": PIPELINE_VERSION,
            "proposal_schema_version": proposal["schema_version"],
            "rules_version": source_meta["layout_rules_schema"],
            "semantic_package_hash": semantic_mapping["semantic_package_hash"],
            "design_summary": room["organization"],
            "diversity_design": proposal["authorship"].get("diversity_design"),
            "daily_activities": room["daily_activities"],
            "main_path": room["main_path"],
            "protected_open_space": room["protected_open_space"],
            "provenance": source_meta,
            "use_clearance_zones": zones,
            "path_targets": proposal["path_targets"],
            "primary_zone_goal_xy_m": proposal["primary_zone_goal_xy_m"],
            "functional_partner_builder": {
                "implementation": "FrozenFunctionalPartnerRules",
                "rules_schema_version": "hssd_functional_partner_rules_v1",
                "rules_sha256": frozen_rules.rules_sha256,
                "source_sha256": frozen_rules.source_sha256,
                "compatibility_schema_version": (
                    frozen_rules.compatibility_schema_version
                ),
                "compatibility_rules_sha256": (
                    frozen_rules.compatibility_rules_sha256
                ),
                "no_handwritten_edges": True,
            },
        },
    }
    scene["audit"]["layout_hash"] = layout_hash(scene)
    return scene, errors


def _furniture_input(obj: dict[str, Any]) -> FurnitureInput:
    return FurnitureInput(
        object_id=obj["object_id"],
        x_m=float(obj["x"]),
        y_m=float(obj["y"]),
        yaw_rad=float(obj["yaw_rad"]),
        bbox_width_m=float(obj["bbox"]["width"]),
        bbox_depth_m=float(obj["bbox"]["depth"]),
        bbox_height_m=float(obj["bbox"]["height"]),
        category=obj["category"],
        hssd_id=obj["hssd_id"],
        family=obj["family"],
        functions=tuple(obj["functions"]),
        family_valid=True,
        functions_valid=True,
        movable=bool(obj["movable"]),
        transform_valid=True,
    )


def _scene_input(scene: dict[str, Any]) -> SceneInput:
    return SceneInput(
        room=RoomInput(scene["room_type"], scene["length"], scene["width"]),
        openings=tuple(
            OpeningInput(
                value["opening_id"],
                value["kind"],
                tuple(value["clearance_center_xyz_m"]),
                tuple(value["clearance_size_xyz_m"]),
                tuple(value["interior_normal_xy"]),
            )
            for value in scene["openings"]
        ),
        furniture=tuple(_furniture_input(value) for value in scene["furniture"]),
        functional_partners=tuple(),
        coordinate_frame=scene["coordinate_frame"],
        quaternion_order=scene["quaternion_order"],
        yaw_range=scene["yaw_range"],
    )


def _all_path_checks(
    scene: dict[str, Any], rules: dict[str, Any]
) -> dict[str, Any]:
    door = next(value for value in scene["openings"] if value["kind"] == "door")
    start = _door_path_start(door)
    obstacles = [object_rect(value) for value in scene["furniture"]]
    rows: list[dict[str, Any]] = []
    for target in scene["audit"]["path_targets"]:
        reachable, details = grid_path_exists(
            scene["length"],
            scene["width"],
            obstacles,
            start,
            tuple(target["goal_xy_m"]),
            grid_m=rules["path"]["grid_resolution_m"],
            human_radius_m=rules["path"]["human_radius_m"],
        )
        rows.append({**target, "reachable": reachable, "details": details})
    return {"passed": all(value["reachable"] for value in rows), "targets": rows}


def _geometry_review_tables(
    scene: dict[str, Any], rules: dict[str, Any]
) -> dict[str, Any]:
    epsilon = float(
        rules["geometry"].get("numeric_separation_epsilon_m", 1.0e-9)
    )
    pairwise: list[dict[str, Any]] = []
    for index, source in enumerate(scene["furniture"]):
        for target in scene["furniture"][index + 1 :]:
            separation = signed_separation(object_rect(source), object_rect(target))
            if separation < -epsilon:
                classification = "penetrating"
            elif separation <= epsilon:
                classification = "touching_within_numeric_epsilon"
            else:
                classification = "positive_gap"
            pairwise.append(
                {
                    "source_id": source["object_id"],
                    "target_id": target["object_id"],
                    "signed_separation_m": round(separation, 9),
                    "classification": classification,
                }
            )
    yaw_rows: list[dict[str, Any]] = []
    for obj in scene["furniture"]:
        yaw = float(obj["yaw_rad"])
        fx, fy = front_vector(yaw)
        if abs(fx) >= abs(fy):
            cardinal = "east" if fx >= 0.0 else "west"
        else:
            cardinal = "north" if fy >= 0.0 else "south"
        yaw_rows.append(
            {
                "object_id": obj["object_id"],
                "yaw_rad": round(yaw, 12),
                "yaw_deg": round(math.degrees(yaw), 6),
                "local_minus_y_front_vector_xy": [round(fx, 9), round(fy, 9)],
                "dominant_cardinal_direction": cardinal,
            }
        )
    return {
        "schema_version": "authored_scene_geometry_review_tables_v1",
        "numeric_separation_epsilon_m": epsilon,
        "accepted_min_signed_separation_m": float(
            rules["geometry"].get("accepted_min_signed_separation_m", 0.0)
        ),
        "pairwise_signed_separation_table": pairwise,
        "yaw_direction_table": yaw_rows,
    }


def _model_input_check(
    scene: dict[str, Any],
    frozen_rules: FrozenFunctionalPartnerRules,
    vocabularies: FurnitureVocabularies,
) -> dict[str, Any]:
    try:
        scene_input = frozen_rules.apply(_scene_input(scene))
        graph = FurnitureGraphBuilder(vocabularies).build(scene_input)
        graph.validate(max_furniture=17)
        return {
            "passed": True,
            "num_nodes": graph.num_nodes,
            "num_furniture": int(graph.furniture_indices.numel()),
            "functional_partner_edges": int(
                graph.edges[next(key for key in graph.edges if key.name == "FUNCTIONAL_PARTNER")].edge_index.shape[1]
            ),
            "furniture_function_dim": int(graph.furniture_functions.shape[1]),
        }
    except Exception as error:  # report-only validation boundary
        return {"passed": False, "error": f"{type(error).__name__}: {error}"}


def _functional_partner_completeness(
    scene: dict[str, Any], frozen_rules: FrozenFunctionalPartnerRules
) -> dict[str, Any]:
    inputs = tuple(_furniture_input(value) for value in scene["furniture"])
    expected = {
        (value.source_id, value.target_id): value
        for value in frozen_rules.build_edge_matches(inputs)
    }
    actual = {
        (value["source_id"], value["target_id"]): value
        for value in scene["functional_partners"]
    }
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    compatibility_matches = [
        {
            "source_id": value.source_id,
            "target_id": value.target_id,
            "annotated_partner_category": value.annotated_partner_category,
            "compatibility_rule_id": value.compatibility_rule_id,
        }
        for value in expected.values()
        if value.category_match_mode == "frozen_semantic_compatibility"
    ]
    return {
        "passed": not missing and not unexpected,
        "expected_edge_count": len(expected),
        "actual_edge_count": len(actual),
        "missing_edges": [
            {"source_id": source, "target_id": target}
            for source, target in missing
        ],
        "unexpected_edges": [
            {"source_id": source, "target_id": target}
            for source, target in unexpected
        ],
        "frozen_compatibility_matches": compatibility_matches,
        "compatibility_rule_count": len(compatibility_matches),
        "source_asset_annotation_required": True,
    }


def _wall_attachment(obj: dict[str, Any], scene: dict[str, Any]) -> str:
    margins = room_margins(object_rect(obj), scene["length"], scene["width"])
    labels = ("west", "east", "south", "north")
    index = min(range(4), key=lambda value: margins[value])
    return labels[index] if margins[index] <= 0.22 else "interior"


def _counter_jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = Counter(left), Counter(right)
    union = sum((a | b).values())
    return sum((a & b).values()) / union if union else 1.0


def _edge_signatures(scene: dict[str, Any]) -> list[str]:
    categories = {value["object_id"]: value["category"] for value in scene["furniture"]}
    return [
        f"{categories[value['source_id']]}->{categories[value['target_id']]}"
        for value in scene.get("functional_partners", [])
    ]


def _yaw_bins(scene: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for value in scene["furniture"]:
        yaw = float(value["yaw_rad"])
        index = int(round((yaw % (2 * math.pi)) / (math.pi / 4))) % 8
        result.append(f"{value['category']}:{index}")
    return result


def _open_space_signature(scene: dict[str, Any]) -> list[float]:
    counts = [0.0] * 9
    for value in scene["furniture"]:
        ix = min(2, max(0, int((float(value["x"]) / scene["length"] + 0.5) * 3)))
        iy = min(2, max(0, int((float(value["y"]) / scene["width"] + 0.5) * 3)))
        counts[iy * 3 + ix] += 1.0
    total = max(1.0, sum(counts))
    return [value / total for value in counts]


def _position_rms(left: dict[str, Any], right: dict[str, Any]) -> tuple[float, str]:
    common = sorted(
        set(value["category"] for value in left["furniture"])
        & set(value["category"] for value in right["furniture"])
    )
    if not common:
        return 1.0, "identity"
    left_by = {
        category: [value for value in left["furniture"] if value["category"] == category]
        for category in common
    }
    right_by = {
        category: [value for value in right["furniture"] if value["category"] == category]
        for category in common
    }
    transforms = {
        "identity": lambda x, y: (x, y),
        "mirror_x": lambda x, y: (-x, y),
        "mirror_y": lambda x, y: (x, -y),
        "rotate_180": lambda x, y: (-x, -y),
    }
    best = (math.inf, "identity")
    for name, transform in transforms.items():
        squared: list[float] = []
        for category in common:
            left_points = [
                transform(
                    float(value["x"]) / left["length"],
                    float(value["y"]) / left["width"],
                )
                for value in left_by[category]
            ]
            right_points = [
                (
                    float(value["x"]) / right["length"],
                    float(value["y"]) / right["width"],
                )
                for value in right_by[category]
            ]
            candidates = sorted(
                (
                    (ax - bx) ** 2 + (ay - by) ** 2,
                    left_index,
                    right_index,
                )
                for left_index, (ax, ay) in enumerate(left_points)
                for right_index, (bx, by) in enumerate(right_points)
            )
            used_left: set[int] = set()
            used_right: set[int] = set()
            for distance, left_index, right_index in candidates:
                if left_index in used_left or right_index in used_right:
                    continue
                used_left.add(left_index)
                used_right.add(right_index)
                squared.extend((distance * 0.5, distance * 0.5))
                if len(used_left) == min(len(left_points), len(right_points)):
                    break
        score = math.sqrt(sum(squared) / len(squared))
        if score < best[0]:
            best = score, name
    return best


def _transform_opening(
    wall: str, x: float, y: float, transform: str
) -> tuple[str, float, float]:
    if transform == "identity":
        return wall, x, y
    if transform == "mirror_x":
        return {"east": "west", "west": "east"}.get(wall, wall), -x, y
    if transform == "mirror_y":
        return {"north": "south", "south": "north"}.get(wall, wall), x, -y
    if transform == "rotate_180":
        return {
            "north": "south",
            "south": "north",
            "east": "west",
            "west": "east",
        }[wall], -x, -y
    raise ValueError(f"unknown symmetry transform {transform!r}")


def _opening_signatures(scene: dict[str, Any], transform: str = "identity") -> list[str]:
    values: list[str] = []
    for opening in scene["openings"]:
        x, y, _ = opening["clearance_center_xyz_m"]
        wall, x, y = _transform_opening(
            str(opening["wall"]),
            float(x) / float(scene["length"]),
            float(y) / float(scene["width"]),
            transform,
        )
        along = x if wall in {"north", "south"} else y
        offset_band = "negative" if along < -0.12 else "positive" if along > 0.12 else "center"
        values.append(f"{opening['kind']}:{wall}:{offset_band}")
    return values


def _asset_geometry_signatures(scene: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for item in scene["furniture"]:
        bbox = item["bbox"]
        size_bins = ":".join(
            str(int(round(float(bbox[key]) / 0.25)))
            for key in ("width", "depth", "height")
        )
        values.append(f"{item['category']}:{size_bins}")
    return values


def _room_shape_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_ratio = float(left["length"]) / float(left["width"])
    right_ratio = float(right["length"]) / float(right["width"])
    return math.exp(-2.0 * abs(math.log(left_ratio / right_ratio)))


def _compare_scene(
    candidate: dict[str, Any], existing: dict[str, Any], rules: dict[str, Any]
) -> dict[str, Any]:
    categories = _counter_jaccard(
        (value["category"] for value in candidate["furniture"]),
        (value["category"] for value in existing["furniture"]),
    )
    count_similarity = 1.0 - abs(
        len(candidate["furniture"]) - len(existing["furniture"])
    ) / max(len(candidate["furniture"]), len(existing["furniture"]), 1)
    walls = _counter_jaccard(
        (
            f"{value['category']}:{_wall_attachment(value, candidate)}"
            for value in candidate["furniture"]
        ),
        (
            f"{value['category']}:{_wall_attachment(value, existing)}"
            for value in existing["furniture"]
        ),
    )
    edges = _counter_jaccard(_edge_signatures(candidate), _edge_signatures(existing))
    yaw = _counter_jaccard(_yaw_bins(candidate), _yaw_bins(existing))
    rms, transform = _position_rms(candidate, existing)
    position_similarity = math.exp(-4.0 * rms)
    openings = _counter_jaccard(
        _opening_signatures(candidate, transform), _opening_signatures(existing)
    )
    room_shape = _room_shape_similarity(candidate, existing)
    asset_geometry = _counter_jaccard(
        _asset_geometry_signatures(candidate), _asset_geometry_signatures(existing)
    )
    left_space = _open_space_signature(candidate)
    right_space = _open_space_signature(existing)
    open_space_similarity = 1.0 - 0.5 * sum(
        abs(a - b) for a, b in zip(left_space, right_space)
    )
    candidate_goal = candidate["audit"]["primary_zone_goal_xy_m"]
    existing_goal = existing["audit"]["primary_zone_goal_xy_m"]
    goal_distance = math.hypot(
        candidate_goal[0] / candidate["length"] - existing_goal[0] / existing["length"],
        candidate_goal[1] / candidate["width"] - existing_goal[1] / existing["width"],
    )
    activity_similarity = math.exp(-4.0 * goal_distance)
    score = (
        0.20 * categories
        + 0.05 * count_similarity
        + 0.15 * walls
        + 0.05 * edges
        + 0.20 * position_similarity
        + 0.08 * yaw
        + 0.07 * open_space_similarity
        + 0.05 * activity_similarity
        + 0.10 * openings
        + 0.05 * room_shape
    )
    exact_distance = near_duplicate_distance(candidate, existing)
    same_template = (
        categories >= float(rules["template_category_similarity_threshold"])
        and walls >= float(rules["template_wall_role_similarity_threshold"])
        and openings >= float(rules["template_opening_similarity_threshold"])
        and rms <= float(rules["template_position_rms_threshold"])
        and yaw >= float(rules["template_yaw_similarity_threshold"])
    )
    high_similarity_copy = (
        score >= float(rules["template_similarity_threshold"])
        and openings >= float(rules["template_opening_similarity_threshold"])
        and rms <= float(rules["template_position_rms_threshold"])
    )
    return {
        "scene_id": existing["scene_id"],
        "room_type": existing["room_type"],
        "overall_similarity": round(score, 6),
        "category_multiset_jaccard": round(categories, 6),
        "furniture_count": len(existing["furniture"]),
        "furniture_count_delta": len(candidate["furniture"]) - len(existing["furniture"]),
        "wall_attachment_similarity": round(walls, 6),
        "functional_partner_graph_similarity": round(edges, 6),
        "normalized_position_rms": round(rms, 6),
        "best_symmetry_transform": transform,
        "yaw_combination_similarity": round(yaw, 6),
        "opening_signature_similarity": round(openings, 6),
        "room_shape_similarity": round(room_shape, 6),
        "asset_geometry_similarity": round(asset_geometry, 6),
        "primary_activity_goal_distance": round(goal_distance, 6),
        "open_space_distribution_similarity": round(open_space_similarity, 6),
        "legacy_near_duplicate_distance": (
            None if math.isinf(exact_distance) else round(exact_distance, 6)
        ),
        "template_risk": same_template or high_similarity_copy,
        "template_risk_reason": (
            "same_structure_and_geometry"
            if same_template
            else "high_similarity_copy" if high_similarity_copy else None
        ),
    }


def _diversity_report(
    scene: dict[str, Any], existing_directories: list[Path], rules: dict[str, Any]
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    for directory in existing_directories:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            existing = _load_json(path)
            if existing.get("scene_id") == scene["scene_id"]:
                continue
            comparisons.append(_compare_scene(scene, existing, rules))
    comparisons.sort(key=lambda value: value["overall_similarity"], reverse=True)
    closest_global = comparisons[0] if comparisons else None
    same_room_comparisons = [
        value for value in comparisons if value["room_type"] == scene["room_type"]
    ]
    closest_same_room_type = (
        same_room_comparisons[0] if same_room_comparisons else None
    )
    same_room_template_risk = any(
        value["template_risk"]
        for value in same_room_comparisons
    )
    global_near_duplicate = any(
        value["legacy_near_duplicate_distance"] is not None
        and value["legacy_near_duplicate_distance"]
        <= float(rules["near_duplicate_distance_threshold"])
        for value in comparisons
    )
    pass_novelty = not same_room_template_risk and not global_near_duplicate
    return {
        "schema_version": "authored_layout_diversity_report_v2",
        "scene_id": scene["scene_id"],
        "comparison_count": len(comparisons),
        "same_room_type_comparison_count": len(same_room_comparisons),
        "novelty_pass": pass_novelty,
        "closest_scene": closest_global,
        "closest_global": closest_global,
        "closest_same_room_type": closest_same_room_type,
        "template_risk": same_room_template_risk or global_near_duplicate,
        "template_risk_assessment": {
            "primary_basis": "closest_same_room_type and all same-room comparisons",
            "closest_same_room_type_template_risk": bool(
                closest_same_room_type
                and closest_same_room_type["template_risk"]
            ),
            "closest_same_room_type_similarity_threshold_exceeded": bool(
                closest_same_room_type
                and closest_same_room_type["overall_similarity"] >= 0.86
            ),
            "any_same_room_type_template_risk": same_room_template_risk,
            "global_near_duplicate": global_near_duplicate,
            "similarity_threshold": float(rules["template_similarity_threshold"]),
            "near_duplicate_distance_threshold": float(
                rules["near_duplicate_distance_threshold"]
            ),
        },
        "top_comparisons": comparisons[:10],
        "top_same_room_type_comparisons": same_room_comparisons[:10],
        "candidate_features": {
            "category_multiset": dict(
                Counter(value["category"] for value in scene["furniture"])
            ),
            "furniture_count": len(scene["furniture"]),
            "wall_attachments": {
                value["object_id"]: _wall_attachment(value, scene)
                for value in scene["furniture"]
            },
            "functional_partner_graph": _edge_signatures(scene),
            "yaw_bins": _yaw_bins(scene),
            "opening_signature": _opening_signatures(scene),
            "asset_geometry_signature": _asset_geometry_signatures(scene),
            "room_shape_ratio": round(
                float(scene["length"]) / float(scene["width"]), 6
            ),
            "diversity_design": (scene.get("audit") or {}).get(
                "diversity_design"
            ),
            "primary_activity_goal_xy_m": scene["audit"]["primary_zone_goal_xy_m"],
            "open_space_distribution_3x3": _open_space_signature(scene),
            "functional_zone_composition": [
                {
                    "owner_id": value["owner_id"],
                    "purpose": value["purpose"],
                }
                for value in scene["audit"]["use_clearance_zones"]
            ],
        },
    }


def run(args: argparse.Namespace) -> int:
    proposal_path = Path(args.proposal)
    hssd_path = Path(args.hssd_lookup)
    semantic_path = Path(args.semantic_mapping)
    layout_rules_path = Path(args.layout_rules)
    functional_rules_path = Path(args.functional_rules)
    functional_compatibility_path = Path(args.functional_compatibility)
    vocab_path = Path(args.vocab)
    output_root = Path(args.output_root)
    proposal = _load_json(proposal_path)
    records = load_hssd_lookup(hssd_path)
    semantic_mapping = _load_json(semantic_path)
    layout_rules = _load_json(layout_rules_path)
    frozen_rules = FrozenFunctionalPartnerRules.load(
        functional_rules_path, functional_compatibility_path
    )
    vocabularies = FurnitureVocabularies.load(vocab_path)
    source_meta = {
        "layout_author": "GPT-5.6 Codex primary agent",
        "python_layout_decisions": False,
        "proposal_path": proposal_path.as_posix(),
        "proposal_sha256": canonical_json_hash(proposal),
        "hssd_lookup_path": hssd_path.as_posix(),
        "hssd_lookup_sha256": _sha256_file(hssd_path),
        "semantic_mapping_path": semantic_path.as_posix(),
        "semantic_mapping_sha256": canonical_json_hash(semantic_mapping),
        "layout_rules_path": layout_rules_path.as_posix(),
        "layout_rules_schema": layout_rules["schema_version"],
        "layout_rules_sha256": canonical_json_hash(layout_rules),
        "functional_rules_path": functional_rules_path.as_posix(),
        "functional_rules_sha256": _sha256_file(functional_rules_path),
        "functional_compatibility_path": (
            functional_compatibility_path.as_posix()
        ),
        "functional_compatibility_sha256": _sha256_file(
            functional_compatibility_path
        ),
        "vocabulary_path": vocab_path.as_posix(),
        "vocabulary_sha256": vocabularies.sha256,
    }
    scene, schema_errors = _make_scene(
        proposal, records, semantic_mapping, frozen_rules, source_meta
    )
    verifier = SceneVerifier(records, semantic_mapping, layout_rules)
    geometry = verifier.verify(scene)
    paths = _all_path_checks(scene, layout_rules)
    model_input = _model_input_check(scene, frozen_rules, vocabularies)
    functional_partner_completeness = _functional_partner_completeness(
        scene, frozen_rules
    )
    geometry_review_tables = _geometry_review_tables(scene, layout_rules)
    diversity = _diversity_report(
        scene,
        [Path(value) for value in args.existing_scenes],
        layout_rules["deduplication"],
    )
    hard_pass = (
        not schema_errors
        and geometry["passed"]
        and paths["passed"]
        and model_input["passed"]
        and functional_partner_completeness["passed"]
    )
    checks = {
        "schema_valid": not schema_errors,
        "hssd_metadata_valid": geometry["checks"]["identity_bbox"]["passed"],
        "bbox_valid": geometry["checks"]["identity_bbox"]["passed"],
        "geometry_hard_valid": geometry["passed"],
        "room_bounds_valid": geometry["checks"]["room_bounds"]["passed"],
        "collision_valid": geometry["checks"]["furniture_collision"]["passed"],
        "opening_clearance_valid": geometry["checks"]["opening_clearance"]["passed"],
        "path_reachability_valid": paths["passed"],
        "use_clearance_valid": geometry["checks"]["use_clearance"]["passed"],
        "frozen_functional_partner_rules_used": True,
        "no_handwritten_functional_edges": "functional_partners" not in proposal,
        "all_model_input_fields_available": model_input["passed"],
        "functional_partner_coverage_valid": functional_partner_completeness[
            "passed"
        ],
        "diversity_novelty_pass": diversity["novelty_pass"],
    }
    report = {
        "schema_version": "authored_scene_deterministic_validation_v2",
        "scene_id": scene["scene_id"],
        "pipeline_version": PIPELINE_VERSION,
        "passed": hard_pass,
        "checks": checks,
        "geometry_report": geometry,
        "all_path_targets_report": paths,
        "model_input_report": model_input,
        "functional_partner_completeness": functional_partner_completeness,
        "geometry_review_tables": geometry_review_tables,
        "functional_partner_match_table": [
            {
                "source_id": edge["source_id"],
                "target_id": edge["target_id"],
                **edge["audit"],
            }
            for edge in scene["functional_partners"]
        ],
        "functional_partner_edge_count": len(scene["functional_partners"]),
        "schema_errors": schema_errors,
        "violations": geometry["violations"],
    }
    scene_id = scene["scene_id"]
    paths_by_kind = {
        "scene": output_root / "scenes" / f"{scene_id}.json",
        "validation": output_root / "validation" / f"{scene_id}.json",
        "diversity": output_root / "diversity" / f"{scene_id}.json",
        "legacy_diversity": (
            output_root / "validation" / f"{scene_id}_diversity.json"
        ),
        "visualization": output_root / "visualizations" / f"{scene_id}.png",
        "reviewer_scene": (
            output_root / "reviews" / "reviewer_packet" / f"{scene_id}_scene.json"
        ),
        "reviewer_geometry": (
            output_root
            / "reviews"
            / "reviewer_packet"
            / f"{scene_id}_geometry.json"
        ),
        "functional_partner_build": (
            output_root
            / "validation"
            / f"functional_partner_build_{scene_id}.json"
        ),
    }
    for path in paths_by_kind.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    paths_by_kind["scene"].write_text(
        json.dumps(scene, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paths_by_kind["validation"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paths_by_kind["diversity"].write_text(
        json.dumps(diversity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paths_by_kind["legacy_diversity"].write_text(
        json.dumps(diversity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    reviewer_scene = json.loads(json.dumps(scene))
    for obj in reviewer_scene["furniture"]:
        obj.get("audit", {}).pop("placement_rationale", None)
    reviewer_scene["audit"] = {
        key: reviewer_scene["audit"][key]
        for key in (
            "use_clearance_zones",
            "path_targets",
            "primary_zone_goal_xy_m",
            "functional_partner_builder",
            "layout_hash",
        )
    }
    paths_by_kind["reviewer_scene"].write_text(
        json.dumps(reviewer_scene, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    reviewer_geometry = {
        **geometry_review_tables,
        "scene_id": scene_id,
        "scene_layout_hash": scene["audit"]["layout_hash"],
        "functional_partner_match_table": report[
            "functional_partner_match_table"
        ],
    }
    paths_by_kind["reviewer_geometry"].write_text(
        json.dumps(reviewer_geometry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    functional_build_report = {
        "schema_version": "frozen_functional_partner_build_report_v2",
        "scene_id": scene_id,
        "edge_count": len(scene["functional_partners"]),
        "edges": scene["functional_partners"],
        "asset_audit": [
            {
                "object_id": obj["object_id"],
                "hssd_id": obj["hssd_id"],
                "source_category": obj["hssd_source_category"],
                "functional_partner_annotation": obj["audit"][
                    "functional_partner_annotation"
                ],
                "emitted_edge_count": sum(
                    edge["source_id"] == obj["object_id"]
                    for edge in scene["functional_partners"]
                ),
            }
            for obj in scene["furniture"]
        ],
        "unannotated_assets": [
            {"object_id": obj["object_id"], "hssd_id": obj["hssd_id"]}
            for obj in scene["furniture"]
            if obj["audit"]["functional_partner_annotation"] is None
        ],
        "rules_version": scene["audit"]["functional_partner_builder"],
        "handwritten_edges": False,
        "source": "FrozenFunctionalPartnerRules.build_edge_matches",
    }
    paths_by_kind["functional_partner_build"].write_text(
        json.dumps(functional_build_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    render_topdown(scene, paths_by_kind["visualization"])
    print(
        json.dumps(
            {
                "scene_id": scene_id,
                "hard_pass": hard_pass,
                "diversity_pass": diversity["novelty_pass"],
                "furniture_count": len(scene["furniture"]),
                "functional_partner_edge_count": len(scene["functional_partners"]),
                "violations": report["violations"],
                "closest_global": diversity["closest_global"],
                "closest_same_room_type": diversity["closest_same_room_type"],
                "outputs": {key: str(value) for key, value in paths_by_kind.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if hard_pass else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministically enrich and validate one authored layout proposal."
    )
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--hssd-lookup", required=True)
    parser.add_argument("--semantic-mapping", required=True)
    parser.add_argument("--layout-rules", required=True)
    parser.add_argument("--functional-rules", required=True)
    parser.add_argument("--functional-compatibility", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--existing-scenes", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
