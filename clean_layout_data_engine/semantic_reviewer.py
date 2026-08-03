"""Blind semantic reviewer for one hard-valid clean-layout scene."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from clean_layout_data_engine.api_runtime import (
    create_openai_client,
    mark_api_failure_resolved,
    parse_structured_response,
    persist_api_failure,
)
from clean_layout_data_engine.run_pilot import ENGINE_DIR, _load_api_config
from clean_layout_data_engine.storage import atomic_write_json, load_json, utc_now


ISSUE_CODES = [
    "orientation",
    "functional_group",
    "access",
    "circulation",
    "wall_relation",
    "opening_relation",
    "spacing",
    "ergonomics",
    "room_use",
    "over_template",
    "other",
]

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "verdict",
        "scene_summary",
        "global_issues",
        "object_reviews",
        "repair_instructions",
    ],
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "revise", "reject"]},
        "scene_summary": {"type": "string"},
        "global_issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["code", "severity", "affected_object_ids", "explanation"],
                "properties": {
                    "code": {"type": "string", "enum": ISSUE_CODES},
                    "severity": {"type": "string", "enum": ["minor", "major", "fatal"]},
                    "affected_object_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "explanation": {"type": "string"},
                },
            },
        },
        "object_reviews": {
            "type": "array",
            "minItems": 1,
            "maxItems": 17,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["object_id", "verdict", "issue_codes", "explanation"],
                "properties": {
                    "object_id": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["pass", "revise", "reject"],
                    },
                    "issue_codes": {
                        "type": "array",
                        "items": {"type": "string", "enum": ISSUE_CODES},
                    },
                    "explanation": {"type": "string"},
                },
            },
        },
        "repair_instructions": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}


def _select(value: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: value[key] for key in keys if key in value}


def _compact_review_inputs(
    scene: dict[str, Any], geometry: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    scene_packet = _select(
        scene,
        (
            "scene_id",
            "room_type",
            "length",
            "width",
            "coordinate_frame",
            "yaw_range",
        ),
    )
    scene_packet["openings"] = list(scene.get("openings", []))
    furniture_keys = (
        "object_id",
        "hssd_source_category",
        "category",
        "family",
        "functions",
        "bbox",
        "x",
        "y",
        "z",
        "yaw_rad",
        "movable",
        "validity",
    )
    scene_packet["furniture"] = [
        _select(value, furniture_keys) for value in scene.get("furniture", [])
    ]
    partner_keys = (
        "source_id",
        "target_id",
        "orientation_mode",
        "target_strategy",
        "desired_gap_range",
        "validity",
    )
    scene_packet["functional_partners"] = [
        _select(value, partner_keys)
        for value in scene.get("functional_partners", [])
    ]
    audit = scene.get("audit", {})
    scene_packet["use_clearance_zones"] = audit.get("use_clearance_zones", [])
    scene_packet["path_targets"] = audit.get("path_targets", [])

    geometry_packet = _select(
        geometry,
        (
            "numeric_separation_epsilon_m",
            "accepted_min_signed_separation_m",
            "pairwise_signed_separation_table",
            "yaw_direction_table",
        ),
    )
    return scene_packet, geometry_packet


def _review_prompt(scene: dict[str, Any], geometry: dict[str, Any]) -> str:
    scene_packet, geometry_packet = _compact_review_inputs(scene, geometry)
    return f"""
Independently audit this single-room furniture layout for ordinary human use.
You are a blind reviewer: the author's placement rationales are absent. Do not
assume that hard geometry validation proves semantic quality.

Review every furniture object exactly once. Check functional grouping, facing,
wall relationships, reachability, circulation, openings, realistic spacing,
ergonomics, room-type fit, and obvious template-like organization. A side table
should be beside rather than directly in front of its seat. Work/dining seating
must face its activity target. Storage fronts and access areas must face usable
space. Do not invent requirements unsupported by the supplied geometry.

Use verdict pass only when there are no material issues, every object passes,
and global_issues is empty. Use revise for a locally repairable semantic issue;
use reject when the scene organization is fundamentally unsuitable. You only
review and describe issues. Never output replacement coordinates or a modified
scene.

Reviewer scene packet:
{json.dumps(scene_packet, ensure_ascii=False)}

Deterministic geometry packet:
{json.dumps(geometry_packet, ensure_ascii=False)}
""".strip()


def _validate_review(review: dict[str, Any], scene: dict[str, Any]) -> tuple[bool, list[str]]:
    expected = {str(item["object_id"]) for item in scene["furniture"]}
    reviewed = [str(item["object_id"]) for item in review["object_reviews"]]
    errors: list[str] = []
    if len(reviewed) != len(set(reviewed)):
        errors.append("duplicate object reviews")
    if set(reviewed) != expected:
        errors.append("object review coverage does not match scene furniture")
    passed = (
        review["verdict"] == "pass"
        and not review["global_issues"]
        and not review["repair_instructions"]
        and all(value["verdict"] == "pass" for value in review["object_reviews"])
        and not errors
    )
    if review["verdict"] == "pass" and not passed:
        errors.append("pass verdict conflicts with issue/object fields")
    return passed, errors


def run(args: argparse.Namespace) -> int:
    api = _load_api_config(args.api_config)
    scene = load_json(
        args.output_root
        / "reviews"
        / "reviewer_packet"
        / f"{args.scene_id}_scene.json"
    )
    geometry = load_json(
        args.output_root
        / "reviews"
        / "reviewer_packet"
        / f"{args.scene_id}_geometry.json"
    )
    client = create_openai_client(
        api,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
    )
    failure_path = (
        args.output_root / "reviews" / "semantic_failures" / f"{args.scene_id}.json"
    )
    try:
        response = client.responses.create(
            model=args.model or api["model"],
            reasoning={"effort": "medium"},
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are an independent furniture-layout quality auditor. "
                        "Return only the strict JSON requested by the schema."
                    ),
                },
                {"role": "user", "content": _review_prompt(scene, geometry)},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "clean_layout_semantic_review",
                    "strict": True,
                    "schema": REVIEW_SCHEMA,
                }
            },
            max_output_tokens=args.max_output_tokens,
        )
        review = parse_structured_response(response)
    except Exception as error:
        persist_api_failure(
            error,
            scene_id=args.scene_id,
            stage="semantic_review",
            latest_path=failure_path,
            history_dir=(
                args.output_root / "reviews" / "semantic_failures" / "history"
            ),
        )
        return 3
    passed, contract_errors = _validate_review(review, scene)
    result = {
        "schema_version": "clean_layout_semantic_review_v1",
        "scene_id": args.scene_id,
        "passed": passed,
        "review": review,
        "review_contract_errors": contract_errors,
        "reviewer": {
            "model": response.model,
            "response_id": response.id,
            "blind_to_author_rationale": True,
            "scene_modification_allowed": False,
        },
        "reviewed_at": utc_now(),
    }
    atomic_write_json(
        args.output_root / "reviews" / "semantic" / f"{args.scene_id}.json",
        result,
    )
    mark_api_failure_resolved(failure_path, response=response)
    return 0 if passed else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--api-config", type=Path, default=ENGINE_DIR / "config.json")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--max-output-tokens", type=int, default=5000)
    return parser.parse_args()


def main() -> int:
    try:
        return run(parse_args())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Semantic reviewer error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
