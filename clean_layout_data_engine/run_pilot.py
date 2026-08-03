"""Generate, compile, and validate one API-authored orientation-intent scene."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from openai import OpenAI

from clean_layout_data_engine.api_runtime import (
    create_openai_client,
    parse_structured_response,
    persist_api_failure,
)
from clean_layout_data_engine.orientation_intent_compiler import (
    INTENT_SCHEMA_VERSION,
    compile_intent_proposal,
)
from clean_layout_data_engine.storage import atomic_write_json, load_json, utc_now
from tools.clean_layouts_lib import hssd_bbox, load_hssd_lookup, object_rect, room_margins
from tools.validate_authored_scene import run as run_validator


ENGINE_DIR = Path(__file__).resolve().parent

AUTHOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "room_design",
        "furniture",
        "path_target_intents",
        "primary_target_id",
    ],
    "properties": {
        "room_design": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "organization",
                "daily_activities",
                "main_path",
                "protected_open_space",
            ],
            "properties": {
                "organization": {"type": "string"},
                "daily_activities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 6,
                },
                "main_path": {"type": "string"},
                "protected_open_space": {"type": "string"},
            },
        },
        "furniture": {
            "type": "array",
            "minItems": 1,
            "maxItems": 17,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "object_id",
                    "hssd_id",
                    "category",
                    "x",
                    "y",
                    "z",
                    "orientation_intent",
                    "clearance_intents",
                    "placement_rationale",
                ],
                "properties": {
                    "object_id": {"type": "string"},
                    "hssd_id": {"type": "string"},
                    "category": {"type": "string"},
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "z": {"type": "number"},
                    "placement_rationale": {"type": "string"},
                    "orientation_intent": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "mode",
                            "wall",
                            "target_object_id",
                            "target_xy_m",
                            "direction",
                            "long_axis",
                        ],
                        "properties": {
                            "mode": {
                                "type": "string",
                                "enum": [
                                    "back_to_wall",
                                    "face_object",
                                    "face_point",
                                    "face_direction",
                                    "axis_only",
                                    "orientation_irrelevant",
                                ],
                            },
                            "wall": {
                                "type": ["string", "null"],
                                "enum": ["north", "south", "east", "west", None],
                            },
                            "target_object_id": {"type": ["string", "null"]},
                            "target_xy_m": {
                                "anyOf": [
                                    {"type": "null"},
                                    {
                                        "type": "array",
                                        "items": {"type": "number"},
                                        "minItems": 2,
                                        "maxItems": 2,
                                    },
                                ]
                            },
                            "direction": {
                                "type": ["string", "null"],
                                "enum": ["north", "south", "east", "west", None],
                            },
                            "long_axis": {
                                "type": ["string", "null"],
                                "enum": ["east_west", "north_south", None],
                            },
                        },
                    },
                    "clearance_intents": {
                        "type": "array",
                        "maxItems": 2,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "clearance_id",
                                "type",
                                "purpose",
                                "allowed_overlap_object_ids",
                            ],
                            "properties": {
                                "clearance_id": {"type": "string"},
                                "type": {
                                    "type": "string",
                                    "enum": [
                                        "front_access",
                                        "rear_pullout",
                                        "side_access_left",
                                        "side_access_right",
                                        "none",
                                    ],
                                },
                                "purpose": {"type": "string"},
                                "allowed_overlap_object_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
        },
        "path_target_intents": {
            "type": "array",
            "minItems": 1,
            "maxItems": 6,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["target_id", "clearance_id", "purpose"],
                "properties": {
                    "target_id": {"type": "string"},
                    "clearance_id": {"type": "string"},
                    "purpose": {"type": "string"},
                },
            },
        },
        "primary_target_id": {"type": "string"},
    },
}


ROOM_NORMALS = {
    "north": [0.0, -1.0],
    "south": [0.0, 1.0],
    "east": [-1.0, 0.0],
    "west": [1.0, 0.0],
}


def _load_json(path: Path) -> dict[str, Any]:
    return load_json(path)


def _load_api_config(path: Path) -> dict[str, Any]:
    """Load connection settings without logging or copying secret values."""

    value = _load_json(path)
    api = value.get("api")
    if not isinstance(api, dict):
        raise ValueError(f"{path} must contain an api object")
    required = {"api_key", "base_url", "model", "wire_api", "timeout_seconds"}
    missing = sorted(required - set(api))
    if missing:
        raise ValueError(f"{path} api object is missing fields: {missing}")
    if not isinstance(api["api_key"], str) or not api["api_key"].strip():
        raise ValueError(f"{path} api.api_key is empty")
    if not isinstance(api["base_url"], str) or not api["base_url"].strip():
        raise ValueError(f"{path} api.base_url is empty")
    if not isinstance(api["model"], str) or not api["model"].strip():
        raise ValueError(f"{path} api.model is empty")
    if api["wire_api"] != "responses":
        raise ValueError(f"{path} api.wire_api must be 'responses'")
    timeout = float(api["timeout_seconds"])
    if timeout <= 0:
        raise ValueError(f"{path} api.timeout_seconds must be positive")
    return {
        "api_key": api["api_key"].strip(),
        "base_url": api["base_url"].strip(),
        "model": api["model"].strip(),
        "timeout_seconds": timeout,
    }


def _existing_hssd_ids(scene_directories: list[Path]) -> set[str]:
    values: set[str] = set()
    for scene_directory in scene_directories:
        if not scene_directory.exists():
            continue
        for path in scene_directory.glob("*.json"):
            scene = _load_json(path)
            values.update(
                str(item["hssd_id"]).lower() for item in scene.get("furniture", [])
            )
    return values


def _asset_catalog(
    records: dict[str, dict[str, Any]],
    semantic_mapping: dict[str, Any],
    categories: list[str],
    *,
    per_category: int,
    seed: int,
    prefer_unseen: set[str],
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    source_to_frozen = semantic_mapping["source_category_to_frozen"]
    by_category: dict[str, list[dict[str, Any]]] = {key: [] for key in categories}
    for hssd_id, record in records.items():
        frozen = source_to_frozen.get(str(record.get("category", "")).lower())
        if frozen not in by_category:
            continue
        bbox = hssd_bbox(record)
        if bbox is None:
            continue
        extents = [float(bbox[key]) for key in ("width", "depth", "height")]
        if min(extents) <= 0.05 or max(extents) > 3.5:
            continue
        annotations = (record.get("interaction_clearance") or {}).get(
            "functional_partners"
        )
        by_category[frozen].append(
            {
                "hssd_id": hssd_id,
                "category": frozen,
                "bbox_width_depth_height_m": [round(value, 4) for value in extents],
                "functional_partner_annotation": annotations,
                "unseen_in_current_clean_dataset": hssd_id not in prefer_unseen,
            }
        )
    result: list[dict[str, Any]] = []
    for category in categories:
        candidates = by_category[category]
        if not candidates:
            continue
        rng.shuffle(candidates)
        candidates.sort(
            key=lambda value: (
                not value["unseen_in_current_clean_dataset"],
                value["functional_partner_annotation"] is None,
            )
        )
        result.extend(candidates[:per_category])
    return result


def _opening(
    opening_id: str,
    kind: str,
    wall: str,
    along: float,
    width: float,
) -> dict[str, Any]:
    return {
        "opening_id": opening_id,
        "kind": kind,
        "wall": wall,
        "along_m": round(along, 3),
        "opening_width_m": width,
        "opening_height_m": 2.05 if kind == "door" else 1.2,
        "sill_height_m": 0.0 if kind == "door" else 0.85,
        "clearance_depth_m": 0.8 if kind == "door" else 0.5,
        "interior_normal_xy": ROOM_NORMALS[wall],
    }


_OPPOSITE_WALL = {"north": "south", "south": "north", "east": "west", "west": "east"}
_ADJACENT_WALLS = {
    "north": ("east", "west"),
    "south": ("east", "west"),
    "east": ("north", "south"),
    "west": ("north", "south"),
}
_MARGIN_LABELS = ("west", "east", "south", "north")


def _existing_scene_rows(scene_directories: list[Path], room_type: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for directory in scene_directories:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            scene = _load_json(path)
            if scene.get("room_type") == room_type:
                rows.append(scene)
    return rows


def _choose_undercovered(
    options: list[Any],
    key: str,
    coverage_rows: list[dict[str, Any]],
    rng: random.Random,
) -> Any:
    """Choose among least-used design values; Python never chooses coordinates."""

    counts: Counter[str] = Counter()
    for scene in coverage_rows:
        design = (scene.get("audit") or {}).get("diversity_design") or {}
        value = design.get(key)
        if value is not None:
            counts[str(value)] += 1
    keyed = [
        (value, str(value["id"]) if isinstance(value, dict) else str(value))
        for value in options
    ]
    minimum = min(counts[label] for _, label in keyed)
    candidates = [value for value, label in keyed if counts[label] == minimum]
    return rng.choice(candidates)


def _quantile(range_m: list[float], value: float) -> float:
    low, high = float(range_m[0]), float(range_m[1])
    return low + (high - low) * max(0.0, min(1.0, value))


def _dimension_pair(
    profile: dict[str, Any], room_scale: str, aspect_mode: str, rng: random.Random
) -> tuple[float, float]:
    scale_center = {"small": 0.18, "medium": 0.50, "large": 0.82}[room_scale]
    aspect_shift = 0.04 if aspect_mode == "balanced" else 0.15
    jitter = rng.uniform(-0.07, 0.07)
    length = _quantile(profile["length_range_m"], scale_center + aspect_shift + jitter)
    width = _quantile(profile["width_range_m"], scale_center - aspect_shift + jitter)
    return round(length, 1), round(width, 1)


def _along_from_band(span: float, opening_width: float, band: str, rng: random.Random) -> float:
    available = max(0.0, span * 0.5 - opening_width * 0.5 - 0.08)
    center = {"negative": -0.58, "center": 0.0, "positive": 0.58}[band]
    jitter = rng.uniform(-0.10, 0.10)
    return round(max(-available, min(available, available * (center + jitter))), 3)


def _window_walls(door_wall: str, pattern: str, rng: random.Random) -> list[str]:
    adjacent = list(_ADJACENT_WALLS[door_wall])
    opposite = _OPPOSITE_WALL[door_wall]
    if pattern == "none":
        return []
    if pattern == "single_adjacent":
        return [rng.choice(adjacent)]
    if pattern == "single_opposite":
        return [opposite]
    if pattern == "corner_pair":
        return [opposite, rng.choice(adjacent)]
    if pattern == "opposite_pair":
        return adjacent
    raise ValueError(f"unsupported window pattern {pattern!r}")


def _target_furniture_count(
    profile: dict[str, Any], density: str, topology: dict[str, Any]
) -> int:
    minimum, maximum = (int(value) for value in profile["furniture_count_range"])
    target = {
        "sparse": minimum,
        "medium": round((minimum + maximum) / 2),
        "full": maximum,
    }[density]
    return max(target, int(topology.get("min_furniture", minimum)))


def _brief(
    profile: dict[str, Any],
    room_type: str,
    seed: int,
    diversity_strategy: dict[str, Any],
    coverage_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rng = random.Random(seed)
    room_scale = _choose_undercovered(
        diversity_strategy["room_scales"], "room_scale", coverage_rows, rng
    )
    aspect_mode = _choose_undercovered(
        diversity_strategy["aspect_modes"], "aspect_mode", coverage_rows, rng
    )
    door_wall = _choose_undercovered(
        diversity_strategy["door_walls"], "door_wall", coverage_rows, rng
    )
    door_offset_band = _choose_undercovered(
        diversity_strategy["door_offset_bands"],
        "door_offset_band",
        coverage_rows,
        rng,
    )
    window_pattern = _choose_undercovered(
        diversity_strategy["window_patterns"][room_type],
        "window_pattern",
        coverage_rows,
        rng,
    )
    topology = _choose_undercovered(
        diversity_strategy["topologies"][room_type],
        "primary_layout_topology",
        coverage_rows,
        rng,
    )
    primary_anchor = _choose_undercovered(
        diversity_strategy["primary_anchors"],
        "primary_anchor",
        coverage_rows,
        rng,
    )
    density = _choose_undercovered(
        diversity_strategy["density_levels"],
        "furniture_density",
        coverage_rows,
        rng,
    )
    window_offset_band = _choose_undercovered(
        diversity_strategy["door_offset_bands"],
        "window_offset_band",
        coverage_rows,
        rng,
    )
    length, width = _dimension_pair(profile, room_scale, aspect_mode, rng)
    door_span = length if door_wall in {"north", "south"} else width
    openings = [
        _opening(
            "door_00",
            "door",
            door_wall,
            _along_from_band(door_span, 0.9, door_offset_band, rng),
            0.9,
        )
    ]
    for index, wall in enumerate(_window_walls(door_wall, window_pattern, rng)):
        span = length if wall in {"north", "south"} else width
        openings.append(
            _opening(
                f"window_{index:02d}",
                "window",
                wall,
                _along_from_band(span, 1.4, window_offset_band, rng),
                1.4,
            )
        )
    design = {
        "schema_version": diversity_strategy["schema_version"],
        "room_scale": room_scale,
        "aspect_mode": aspect_mode,
        "door_wall": door_wall,
        "door_offset_band": door_offset_band,
        "window_offset_band": window_offset_band,
        "window_pattern": window_pattern,
        "primary_layout_topology": topology["id"],
        "topology_guidance": topology["guidance"],
        "primary_anchor": primary_anchor,
        "furniture_density": density,
    }
    return {
        "room_type": room_type,
        "length_m": length,
        "width_m": width,
        "openings": openings,
        "required_categories": profile["required_categories"],
        "furniture_count_range": profile["furniture_count_range"],
        "target_furniture_count": _target_furniture_count(profile, density, topology),
        "diversity_design": design,
    }


def _wall_attachment_summary(obj: dict[str, Any], scene: dict[str, Any]) -> str:
    margins = room_margins(object_rect(obj), float(scene["length"]), float(scene["width"]))
    index = min(range(4), key=lambda value: margins[value])
    return _MARGIN_LABELS[index] if margins[index] <= 0.22 else "interior"


def _historical_template_summaries(
    scene_directories: list[Path], room_type: str, limit: int
) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    details: dict[str, dict[str, Any]] = {}
    for scene in _existing_scene_rows(scene_directories, room_type):
        categories = sorted(value["category"] for value in scene.get("furniture", []))
        wall_roles = sorted(
            f"{value['category']}:{_wall_attachment_summary(value, scene)}"
            for value in scene.get("furniture", [])
        )
        openings = sorted(
            f"{value['kind']}:{value['wall']}" for value in scene.get("openings", [])
        )
        signature = json.dumps(
            {"categories": categories, "wall_roles": wall_roles, "openings": openings},
            ensure_ascii=False,
            sort_keys=True,
        )
        counts[signature] += 1
        details[signature] = {
            "count": counts[signature],
            "categories": categories,
            "wall_roles": wall_roles,
            "openings": openings,
        }
    result = []
    for signature, count in counts.most_common(limit):
        value = dict(details[signature])
        value["count"] = count
        result.append(value)
    return result


def _diversity_retry_feedback(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    report = _load_json(path)
    closest = report.get("closest_same_room_type") or {}
    return {
        "closest_scene_id": closest.get("scene_id"),
        "category_similarity": closest.get("category_multiset_jaccard"),
        "wall_role_similarity": closest.get("wall_attachment_similarity"),
        "position_rms": closest.get("normalized_position_rms"),
        "opening_similarity": closest.get("opening_signature_similarity"),
        "reason": "The previous candidate was semantically acceptable but too similar. Reorganize the layout topology; do not merely nudge coordinates.",
    }


def _author_prompt(
    scene_id: str,
    brief: dict[str, Any],
    assets: list[dict[str, Any]],
    previous: dict[str, Any] | None,
    errors: list[dict[str, Any]] | None,
    historical_templates: list[dict[str, Any]],
    diversity_retry_feedback: dict[str, Any] | None,
) -> str:
    bounds = {
        "x": [-brief["length_m"] * 0.5, brief["length_m"] * 0.5],
        "y": [-brief["width_m"] * 0.5, brief["width_m"] * 0.5],
    }
    prompt = f"""
Design exactly one clean, human-plausible {brief['room_type']} furniture layout.
You are the layout author. Code will only compile geometry and validate it.

Coordinate contract:
- room_local_z_up, +X east, +Y north, room center at [0,0]
- HSSD semantic front is local -Y
- room bounds: {json.dumps(bounds)}
- room/opening brief: {json.dumps(brief, ensure_ascii=False)}
- scene_id: {scene_id}

Diversity design brief. This is a design direction, not a request to force an
implausible layout. Preserve all geometric and human-use requirements first:
{json.dumps(brief['diversity_design'], ensure_ascii=False)}
- Select exactly {brief['target_furniture_count']} objects, while including all
  required categories.
- Interpret primary_anchor as the preferred location of the primary activity
  zone. Use interior only when a floating composition is actually justified.
- Follow topology_guidance by changing the functional organization, not merely
  by making small coordinate adjustments.

Use only the supplied HSSD asset cards. Copy hssd_id and category exactly:
{json.dumps(assets, ensure_ascii=False)}

Requirements:
- Include every required category: {brief['required_categories']}.
- Use bbox dimensions to keep every oriented footprint inside the room and separated.
- Keep door/window clearance, walking routes, and use zones unobstructed.
- Do not output yaw, quaternion, clearance coordinates, or functional edges.
- Express orientation only with orientation_intent.
- Use back_to_wall for boundary-backed objects, face_object for functional seating,
  axis_only for rectangular objects without a semantic front, and
  orientation_irrelevant only when rotation truly has no layout meaning.
- A reading/lounge seat must face a real activity target, a room direction, or an
  explicit point. It must never face a side table, nightstand, or floor lamp.
- Every required-use object needs a clearance intent. Desk front_access may allow
  its chair; a chair normally uses rear_pullout; storage uses front_access.
- For a chair whose target is a desk, create a real workstation: align it with
  the desk front and keep the actual bbox edge gap between 0.00 m and 0.35 m.
  A chair merely facing a desk from across the room is invalid.
- If a room has one sofa and a coffee table, keep their actual bbox edge gap
  within 0.35..0.65 m and place the table directly in the sofa's forward
  seating sector, not beside its end or against an unrelated wall. An armchair
  facing a coffee table must be within 0.35..0.60 m of it. A side table must
  be within 0.00..0.35 m of a sofa or armchair, on a lateral/reachable side.
- In a dining_room, choose a table asset whose HSSD bbox height is 0.65..0.90 m.
  Low coffee/side tables are not valid dining surfaces for conventional chairs.
- Do not let one object's required access or pullout space overlap another
  object's independent use-clearance zone. Only the explicitly allowed paired
  object may share a zone, such as a desk with its own chair.
- side_table, coffee_table, ottoman, and floor_lamp must have an empty
  clearance_intents list; they are reached from adjacent seating/circulation.
- Do not use allowed_overlap_object_ids to bypass geometry. The only approved
  pairs are desk->chair/armchair, table->chair/bench, and bed->nightstand.
- Path targets must reference generated clearance_id values.
- Avoid symmetric templates and explain each placement concretely. Reuse of a
  common functional relation is acceptable; copying the same wall roles and
  normalized furniture arrangement is not.
""".strip()
    if historical_templates:
        prompt += (
            "\n\nOverrepresented historical signatures for this room type. "
            "Do not reproduce one of these as the overall organization:\n"
            + json.dumps(historical_templates, ensure_ascii=False)
        )
    if diversity_retry_feedback is not None:
        prompt += (
            "\n\nDiversity retry feedback from the previous candidate:\n"
            + json.dumps(diversity_retry_feedback, ensure_ascii=False)
        )
    if previous is not None:
        prompt += (
            "\n\nThe previous attempt failed. Revise it minimally while preserving its "
            "useful design decisions.\nPrevious attempt:\n"
            + json.dumps(previous, ensure_ascii=False)
            + "\nMachine-readable failures:\n"
            + json.dumps(errors or [], ensure_ascii=False)
        )
    return prompt


def _call_author(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    max_output_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    response = client.responses.create(
        model=model,
        reasoning={"effort": "medium"},
        input=[
            {
                "role": "system",
                "content": (
                    "You author diverse clean indoor layouts. Return only the "
                    "strict JSON object requested by the response schema."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "authored_layout_intent",
                "strict": True,
                "schema": AUTHOR_SCHEMA,
            }
        },
        max_output_tokens=max_output_tokens,
    )
    value = parse_structured_response(response)
    return value, {
        "response_id": response.id,
        "response_model": response.model,
        "status": response.status,
    }


def _assemble_intent(
    scene_id: str,
    brief: dict[str, Any],
    authored: dict[str, Any],
    api_meta: dict[str, Any],
) -> dict[str, Any]:
    room_design = authored["room_design"]
    return {
        "schema_version": INTENT_SCHEMA_VERSION,
        "scene_id": scene_id,
        "room_type": brief["room_type"],
        "coordinate_frame": "room_local_z_up",
        "quaternion_order": "wxyz",
        "yaw_range": "[-pi,pi)",
        "room": {
            "length_m": brief["length_m"],
            "width_m": brief["width_m"],
            **room_design,
        },
        "openings": brief["openings"],
        "furniture": authored["furniture"],
        "path_target_intents": authored["path_target_intents"],
        "primary_target_id": authored["primary_target_id"],
        "authorship": {
            "author": "OpenAI API independent single-scene author",
            "model": api_meta["response_model"],
            "response_id": api_meta["response_id"],
            "layout_decisions_by_python": False,
            "diversity_design": brief["diversity_design"],
        },
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write_json(path, value)


def _status_path(args: argparse.Namespace) -> Path:
    return args.output_root / "status" / f"{args.scene_id}.json"


def _write_status(
    args: argparse.Namespace,
    status: str,
    *,
    author_attempts: int,
    **details: Any,
) -> None:
    existing = (
        _load_json(_status_path(args)) if _status_path(args).exists() else {}
    )
    _write_json(
        _status_path(args),
        {
            "schema_version": "clean_layout_scene_status_v1",
            "scene_id": args.scene_id,
            "room_type": args.room_type,
            "status": status,
            "author_attempts": author_attempts,
            "started_at": existing.get("started_at", utc_now()),
            "updated_at": utc_now(),
            "formal_dataset_modified": False,
            **details,
        },
    )


def _validator_args(args: argparse.Namespace, proposal: Path) -> argparse.Namespace:
    existing_scenes = [str(args.existing_scenes)]
    if args.coverage_scenes is not None:
        existing_scenes.append(str(args.coverage_scenes))
    return argparse.Namespace(
        proposal=str(proposal),
        output_root=str(args.output_root),
        hssd_lookup=str(args.hssd_lookup),
        semantic_mapping=str(args.semantic_mapping),
        layout_rules=str(args.layout_rules),
        functional_rules=str(args.functional_rules),
        functional_compatibility=str(args.functional_compatibility),
        vocab=str(args.vocab),
        existing_scenes=existing_scenes,
    )


def _validation_errors(validation: dict[str, Any]) -> list[dict[str, Any]]:
    errors = [
        {"stage": "validate", **value} for value in validation["violations"]
    ]
    errors.extend(
        {
            "stage": "validate",
            "type": "failed_check",
            "check": key,
        }
        for key, passed in validation["checks"].items()
        if not passed and key != "diversity_novelty_pass"
    )
    if validation.get("schema_errors"):
        errors.append(
            {
                "stage": "validate",
                "type": "schema_errors",
                "details": validation["schema_errors"],
            }
        )
    if not validation["all_path_targets_report"]["passed"]:
        errors.append(
            {
                "stage": "validate",
                "type": "path_reachability",
                "details": validation["all_path_targets_report"],
            }
        )
    if not validation["model_input_report"]["passed"]:
        errors.append(
            {
                "stage": "validate",
                "type": "model_input_contract",
                "details": validation["model_input_report"],
            }
        )
    return errors


def run(args: argparse.Namespace) -> int:
    config = _load_json(args.config)
    api = _load_api_config(args.api_config)
    model = args.model or api["model"]
    base_url = args.base_url or api["base_url"]
    timeout_seconds = args.timeout_seconds or api["timeout_seconds"]
    profile = config["room_profiles"][args.room_type]
    _write_status(args, "authoring", author_attempts=0)
    records = load_hssd_lookup(args.hssd_lookup)
    semantic_mapping = _load_json(args.semantic_mapping)
    layout_rules = _load_json(args.layout_rules)
    diversity_strategy = config["diversity_strategy"]
    scene_directories = [args.existing_scenes]
    if args.coverage_scenes is not None:
        scene_directories.append(args.coverage_scenes)
    coverage_rows = _existing_scene_rows(scene_directories, args.room_type)
    brief = _brief(
        profile,
        args.room_type,
        args.seed,
        diversity_strategy,
        coverage_rows,
    )
    seen_ids = _existing_hssd_ids(scene_directories)
    historical_templates = _historical_template_summaries(
        scene_directories,
        args.room_type,
        int(diversity_strategy["history_summary_limit"]),
    )
    diversity_retry_feedback = _diversity_retry_feedback(args.diversity_feedback)
    assets = _asset_catalog(
        records,
        semantic_mapping,
        profile["candidate_categories"],
        per_category=int(config["candidate_assets_per_category"]),
        seed=args.seed,
        prefer_unseen=seen_ids,
    )
    _write_json(args.output_root / "briefs" / f"{args.scene_id}.json", brief)
    _write_json(
        args.output_root / "asset_catalogs" / f"{args.scene_id}.json",
        {"scene_id": args.scene_id, "assets": assets},
    )
    client = create_openai_client(
        api,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )
    previous: dict[str, Any] | None = None
    errors: list[dict[str, Any]] | None = None
    attempts = int(config["max_author_attempts"])
    for attempt in range(1, attempts + 1):
        _write_status(args, "authoring", author_attempts=attempt)
        prompt = _author_prompt(
            args.scene_id,
            brief,
            assets,
            previous,
            errors,
            historical_templates,
            diversity_retry_feedback,
        )
        try:
            authored, api_meta = _call_author(
                client,
                model=model,
                prompt=prompt,
                max_output_tokens=args.max_output_tokens,
            )
        except Exception as error:
            failure = persist_api_failure(
                error,
                scene_id=args.scene_id,
                stage="author",
                latest_path=(
                    args.output_root / "api_failures" / f"{args.scene_id}.json"
                ),
                history_dir=args.output_root / "api_failures" / "history",
            )
            _write_status(
                args,
                "transient_api_failure",
                author_attempts=attempt,
                error_type=failure["error_type"],
                error_category=failure["error_category"],
                safe_message=failure["safe_message"],
                **(
                    {"status_code": failure["status_code"]}
                    if "status_code" in failure
                    else {}
                ),
            )
            return 3
        intent = _assemble_intent(args.scene_id, brief, authored, api_meta)
        _write_json(
            args.output_root
            / "attempts"
            / args.scene_id
            / f"attempt_{attempt:02d}_intent.json",
            intent,
        )
        try:
            compiled, compile_report = compile_intent_proposal(
                intent, records, semantic_mapping, layout_rules
            )
        except Exception as error:
            previous = authored
            errors = [{"stage": "compile", "message": str(error)}]
            _write_status(
                args,
                "authoring_retry",
                author_attempts=attempt,
                last_errors=errors,
            )
            continue
        intent_path = args.output_root / "intents" / f"{args.scene_id}.json"
        proposal_path = (
            args.output_root / "compiled_proposals" / f"{args.scene_id}.json"
        )
        _write_json(intent_path, intent)
        _write_json(proposal_path, compiled)
        _write_json(
            args.output_root
            / "validation"
            / f"orientation_compile_{args.scene_id}.json",
            compile_report,
        )
        validator_code = run_validator(_validator_args(args, proposal_path))
        validation = _load_json(
            args.output_root / "validation" / f"{args.scene_id}.json"
        )
        if validator_code == 0 and validation["passed"]:
            _write_status(
                args,
                "hard_valid_pending_independent_review",
                author_attempts=attempt,
                model=api_meta["response_model"],
                hard_validation_pass=True,
                diversity_pass=validation["checks"]["diversity_novelty_pass"],
            )
            return 0
        previous = authored
        errors = _validation_errors(validation)
        _write_status(
            args,
            "authoring_retry",
            author_attempts=attempt,
            last_errors=errors,
        )
    _write_status(
        args,
        "rejected_after_author_retries",
        author_attempts=attempts,
        last_errors=errors,
    )
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument(
        "--room-type",
        required=True,
        choices=("bedroom", "living_room", "dining_room", "home_office"),
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--max-output-tokens", type=int, default=6000)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument(
        "--api-config",
        type=Path,
        default=ENGINE_DIR / "config.json",
        help="Local ignored API configuration file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ENGINE_DIR / "pipeline_config.json",
    )
    parser.add_argument(
        "--hssd-lookup",
        type=Path,
        default=Path("../hssd-annotations/data/hssd_annotation_lookup.json.gz"),
    )
    parser.add_argument(
        "--semantic-mapping",
        type=Path,
        default=Path("data/clean_layouts_gpt56_v1/semantic_mapping.json"),
    )
    parser.add_argument(
        "--layout-rules",
        type=Path,
        default=Path("configs/authored_layout_validation_rules_v2.json"),
    )
    parser.add_argument(
        "--functional-rules",
        type=Path,
        default=Path("configs/functional_partner_rules_hssd_v1.json"),
    )
    parser.add_argument(
        "--functional-compatibility",
        type=Path,
        default=Path("configs/functional_partner_category_compatibility_v1.json"),
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path("configs/furniture_vocab_bootstrap_v1.json"),
    )
    parser.add_argument(
        "--existing-scenes",
        type=Path,
        default=Path("data/clean_layouts_gpt56_v2_authored/scenes"),
    )
    parser.add_argument(
        "--coverage-scenes",
        type=Path,
        help="Accepted production scenes used for diversity coverage planning.",
    )
    parser.add_argument(
        "--diversity-feedback",
        type=Path,
        help="Previous diversity report used for a targeted author retry.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        return run(parse_args())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Data engine error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
