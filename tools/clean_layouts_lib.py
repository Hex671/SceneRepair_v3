from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


EPSILON = 1.0e-9


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def wrap_yaw(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_to_quaternion_wxyz(yaw: float) -> list[float]:
    half = wrap_yaw(yaw) * 0.5
    return [round(math.cos(half), 10), 0.0, 0.0, round(math.sin(half), 10)]


def front_vector(yaw: float) -> tuple[float, float]:
    """Return the room-frame direction of the HSSD local semantic front (-Y)."""

    return math.sin(yaw), -math.cos(yaw)


def yaw_facing(source_xy: tuple[float, float], target_xy: tuple[float, float]) -> float:
    dx = target_xy[0] - source_xy[0]
    dy = target_xy[1] - source_xy[1]
    if math.hypot(dx, dy) <= EPSILON:
        raise ValueError("cannot face a coincident target")
    return wrap_yaw(math.atan2(dx, -dy))


def rotate_xy(x: float, y: float, yaw: float) -> tuple[float, float]:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return cosine * x - sine * y, sine * x + cosine * y


@dataclass(frozen=True)
class OrientedRectangle:
    x: float
    y: float
    width: float
    depth: float
    yaw: float

    @property
    def axes(self) -> tuple[tuple[float, float], tuple[float, float]]:
        cosine, sine = math.cos(self.yaw), math.sin(self.yaw)
        return (cosine, sine), (-sine, cosine)

    @property
    def corners(self) -> list[tuple[float, float]]:
        result: list[tuple[float, float]] = []
        for local_x, local_y in (
            (-self.width * 0.5, -self.depth * 0.5),
            (self.width * 0.5, -self.depth * 0.5),
            (self.width * 0.5, self.depth * 0.5),
            (-self.width * 0.5, self.depth * 0.5),
        ):
            dx, dy = rotate_xy(local_x, local_y, self.yaw)
            result.append((self.x + dx, self.y + dy))
        return result

    def radius_on(self, axis: tuple[float, float]) -> float:
        own_x, own_y = self.axes
        return self.width * 0.5 * abs(_dot(own_x, axis)) + self.depth * 0.5 * abs(
            _dot(own_y, axis)
        )

    @property
    def room_aabb_half_extents(self) -> tuple[float, float]:
        cosine, sine = abs(math.cos(self.yaw)), abs(math.sin(self.yaw))
        return (
            cosine * self.width * 0.5 + sine * self.depth * 0.5,
            sine * self.width * 0.5 + cosine * self.depth * 0.5,
        )


def signed_separation(
    source: OrientedRectangle, target: OrientedRectangle
) -> float:
    delta = target.x - source.x, target.y - source.y
    gaps: list[float] = []
    for axis in (*source.axes, *target.axes):
        projection = abs(_dot(delta, axis))
        gaps.append(projection - source.radius_on(axis) - target.radius_on(axis))
    return max(gaps)


def angular_error(a: float, b: float, *, modulo_pi: bool = False) -> float:
    error = abs(wrap_yaw(a - b))
    if modulo_pi:
        error = min(error, abs(math.pi - error))
    return error


def room_margins(rect: OrientedRectangle, length: float, width: float) -> list[float]:
    half_x, half_y = rect.room_aabb_half_extents
    return [
        rect.x - half_x + length * 0.5,
        length * 0.5 - rect.x - half_x,
        rect.y - half_y + width * 0.5,
        width * 0.5 - rect.y - half_y,
    ]


def hssd_bbox(record: dict[str, Any]) -> dict[str, Any] | None:
    clearance = record.get("interaction_clearance") or {}
    nonartic = clearance.get("nonartic_clearance_v2") or {}
    raw = nonartic.get("object_bbox_m")
    if _valid_extent_triplet(raw):
        width, depth, height = (float(value) for value in raw)
        return {
            "width": width,
            "depth": depth,
            "height": height,
            "source_path": (
                "interaction_clearance.nonartic_clearance_v2.object_bbox_m"
            ),
            "source_axis_order": "room_z_up_[x,y,z]",
            "axis_conversion": "identity",
        }

    official = clearance.get("official_combined_clearance") or {}
    aabb = official.get("obj_aabb") or {}
    low, high = aabb.get("min"), aabb.get("max")
    if _finite_triplet(low) and _finite_triplet(high):
        extents = [float(high[i]) - float(low[i]) for i in range(3)]
        if _valid_extent_triplet(extents):
            return {
                "width": extents[0],
                "depth": extents[2],
                "height": extents[1],
                "source_path": (
                    "interaction_clearance.official_combined_clearance.obj_aabb"
                ),
                "source_axis_order": "asset_[x,y_up,z]",
                "axis_conversion": "room_[x,y,z]=asset_[x,z,y]",
            }

    swept = clearance.get("articulated_swept_volume") or {}
    raw = swept.get("bbox")
    if _valid_extent_triplet(raw):
        extents = [float(value) for value in raw]
        return {
            "width": extents[0],
            "depth": extents[2],
            "height": extents[1],
            "source_path": "interaction_clearance.articulated_swept_volume.bbox",
            "source_axis_order": "asset_[x,y_up,z]",
            "axis_conversion": "room_[x,y,z]=asset_[x,z,y]",
        }
    return None


def load_hssd_lookup(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("HSSD lookup must be a JSON object keyed by HSSD id")
    return value


class AssetPool:
    def __init__(
        self,
        records: dict[str, dict[str, Any]],
        source_categories: dict[str, list[str]],
    ) -> None:
        self.records = records
        self.source_categories = source_categories
        self._used: Counter[str] = Counter()
        self._groups: dict[str, str] = {}

    def select(
        self,
        category: str,
        target_bbox: tuple[float, float, float],
        *,
        reuse_group: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if reuse_group and reuse_group in self._groups:
            hssd_id = self._groups[reuse_group]
            self._used[hssd_id] += 1
            return self.records[hssd_id], hssd_bbox(self.records[hssd_id]) or {}

        allowed = set(self.source_categories[category])
        candidates: list[tuple[float, int, str, dict[str, Any]]] = []
        for hssd_id, record in self.records.items():
            if str(record.get("category", "")).lower() not in allowed:
                continue
            bbox = hssd_bbox(record)
            if bbox is None:
                continue
            extents = bbox["width"], bbox["depth"], bbox["height"]
            if not _plausible_furniture_bbox(category, extents):
                continue
            score = sum(
                (math.log(max(actual, 0.01) / expected)) ** 2
                for actual, expected in zip(extents, target_bbox)
            )
            candidates.append((score, self._used[hssd_id], hssd_id, bbox))
        if not candidates:
            raise ValueError(f"no HSSD asset with a valid bbox for {category}")
        candidates.sort(key=lambda item: (item[1], item[0], item[2]))
        _, _, hssd_id, bbox = candidates[0]
        self._used[hssd_id] += 1
        if reuse_group:
            self._groups[reuse_group] = hssd_id
        return self.records[hssd_id], bbox

    @property
    def use_counts(self) -> Counter[str]:
        return self._used.copy()


def make_opening(
    opening_id: str,
    kind: str,
    wall: str,
    along: float,
    room_length: float,
    room_width: float,
    *,
    opening_width: float,
    opening_height: float,
    sill_height: float = 0.0,
    clearance_depth: float,
) -> dict[str, Any]:
    if wall == "north":
        center = [along, room_width * 0.5 - clearance_depth * 0.5]
        size_xy = [opening_width, clearance_depth]
        normal = [0.0, -1.0]
    elif wall == "south":
        center = [along, -room_width * 0.5 + clearance_depth * 0.5]
        size_xy = [opening_width, clearance_depth]
        normal = [0.0, 1.0]
    elif wall == "east":
        center = [room_length * 0.5 - clearance_depth * 0.5, along]
        size_xy = [clearance_depth, opening_width]
        normal = [-1.0, 0.0]
    elif wall == "west":
        center = [-room_length * 0.5 + clearance_depth * 0.5, along]
        size_xy = [clearance_depth, opening_width]
        normal = [1.0, 0.0]
    else:
        raise ValueError(f"unsupported wall {wall!r}")
    center_z = (sill_height + opening_height) * 0.5
    clearance_height = sill_height + opening_height
    return {
        "opening_id": opening_id,
        "kind": kind,
        "wall": wall,
        "opening_width_m": opening_width,
        "opening_height_m": opening_height,
        "sill_height_m": sill_height,
        "clearance_center_xyz_m": [center[0], center[1], center_z],
        "clearance_size_xyz_m": [size_xy[0], size_xy[1], clearance_height],
        "interior_normal_xy": normal,
        "validity": {
            "opening_id": True,
            "kind": True,
            "clearance_center_xyz_m": True,
            "clearance_size_xyz_m": True,
            "interior_normal_xy": True,
        },
    }


def make_furniture(
    object_id: str,
    record: dict[str, Any],
    bbox: dict[str, Any],
    semantics: dict[str, Any],
    x: float,
    y: float,
    yaw: float,
    *,
    rationale: str,
) -> dict[str, Any]:
    yaw = wrap_yaw(yaw)
    if abs(abs(yaw) - math.pi) < 1.0e-10:
        yaw = -math.pi
    front = record.get("canonical_front") or {}
    return {
        "object_id": object_id,
        "hssd_id": record["hssd_id"],
        "hssd_source_category": record.get("category"),
        "category": semantics["category"],
        "family": semantics["family"],
        "functions": list(semantics["functions"]),
        "bbox": {
            "width": round(float(bbox["width"]), 6),
            "depth": round(float(bbox["depth"]), 6),
            "height": round(float(bbox["height"]), 6),
        },
        "x": round(float(x), 6),
        "y": round(float(y), 6),
        "z": 0.0,
        "yaw_rad": yaw,
        "rotation_wxyz": yaw_to_quaternion_wxyz(yaw),
        "movable": True,
        "validity": {
            "object_id": True,
            "hssd_id": True,
            "category": True,
            "family": True,
            "functions": True,
            "bbox": True,
            "transform": True,
            "movable": True,
        },
        "audit": {
            "bbox_source_path": bbox["source_path"],
            "bbox_source_axis_order": bbox["source_axis_order"],
            "bbox_axis_conversion": bbox["axis_conversion"],
            "canonical_orientation_axis": front.get("canonical_orientation_axis"),
            "canonical_orientation_is_semantic_front": bool(
                front.get("canonical_orientation_is_semantic_front", False)
            ),
            "canonical_orientation_confidence": front.get(
                "canonical_orientation_confidence"
            ),
            "placement_rationale": rationale,
        },
    }


def wall_pose(
    wall: str,
    along: float,
    room_length: float,
    room_width: float,
    bbox: dict[str, Any],
    *,
    wall_gap: float = 0.08,
) -> tuple[float, float, float]:
    depth = float(bbox["depth"])
    if wall == "north":
        return along, room_width * 0.5 - depth * 0.5 - wall_gap, 0.0
    if wall == "south":
        return along, -room_width * 0.5 + depth * 0.5 + wall_gap, -math.pi
    if wall == "east":
        return room_length * 0.5 - depth * 0.5 - wall_gap, along, -math.pi * 0.5
    if wall == "west":
        return -room_length * 0.5 + depth * 0.5 + wall_gap, along, math.pi * 0.5
    raise ValueError(f"unsupported wall {wall!r}")


def object_rect(obj: dict[str, Any]) -> OrientedRectangle:
    return OrientedRectangle(
        float(obj["x"]),
        float(obj["y"]),
        float(obj["bbox"]["width"]),
        float(obj["bbox"]["depth"]),
        float(obj["yaw_rad"]),
    )


def offset_from_object(
    obj: dict[str, Any], local_x: float, local_y: float
) -> tuple[float, float]:
    dx, dy = rotate_xy(local_x, local_y, float(obj["yaw_rad"]))
    return float(obj["x"]) + dx, float(obj["y"]) + dy


def place_in_front(
    target: dict[str, Any], source_bbox: dict[str, Any], gap: float
) -> tuple[float, float]:
    direction = front_vector(float(target["yaw_rad"]))
    distance = float(target["bbox"]["depth"]) * 0.5 + float(
        source_bbox["depth"]
    ) * 0.5 + gap
    return (
        float(target["x"]) + direction[0] * distance,
        float(target["y"]) + direction[1] * distance,
    )


def relation(
    source_id: str,
    target_id: str,
    orientation_mode: str,
    target_strategy: str,
    desired_gap_range: tuple[float, float] | None = None,
    *,
    rationale: str,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "target_id": target_id,
        "orientation_mode": orientation_mode,
        "target_strategy": target_strategy,
        "desired_gap_range": (
            [float(desired_gap_range[0]), float(desired_gap_range[1])]
            if desired_gap_range
            else None
        ),
        "validity": {
            "source_id": True,
            "target_id": True,
            "orientation_mode": True,
            "target_strategy": True,
            "desired_gap_range": desired_gap_range is not None,
            "relation": True,
        },
        "audit": {"rationale": rationale},
    }


def make_zone(
    zone_id: str,
    owner_id: str,
    center: tuple[float, float],
    width: float,
    depth: float,
    yaw: float,
    purpose: str,
    *,
    allowed_overlap_object_ids: Iterable[str] = (),
) -> dict[str, Any]:
    return {
        "zone_id": zone_id,
        "owner_id": owner_id,
        "center_xy_m": [round(center[0], 6), round(center[1], 6)],
        "size_xy_m": [round(width, 6), round(depth, 6)],
        "yaw_rad": round(wrap_yaw(yaw), 10),
        "purpose": purpose,
        "allowed_overlap_object_ids": list(allowed_overlap_object_ids),
        "validity": True,
    }


def front_zone(
    obj: dict[str, Any], depth: float, purpose: str, *, allowed: Iterable[str] = ()
) -> dict[str, Any]:
    direction = front_vector(float(obj["yaw_rad"]))
    distance = float(obj["bbox"]["depth"]) * 0.5 + depth * 0.5
    center = (
        float(obj["x"]) + direction[0] * distance,
        float(obj["y"]) + direction[1] * distance,
    )
    return make_zone(
        f"use_{obj['object_id']}_front",
        obj["object_id"],
        center,
        max(0.45, min(float(obj["bbox"]["width"]), 1.2)),
        depth,
        float(obj["yaw_rad"]),
        purpose,
        allowed_overlap_object_ids=allowed,
    )


def behind_zone(
    obj: dict[str, Any], depth: float, purpose: str, *, allowed: Iterable[str] = ()
) -> dict[str, Any]:
    front = front_vector(float(obj["yaw_rad"]))
    distance = float(obj["bbox"]["depth"]) * 0.5 + depth * 0.5
    center = (
        float(obj["x"]) - front[0] * distance,
        float(obj["y"]) - front[1] * distance,
    )
    return make_zone(
        f"use_{obj['object_id']}_behind",
        obj["object_id"],
        center,
        max(0.5, min(float(obj["bbox"]["width"]), 0.9)),
        depth,
        float(obj["yaw_rad"]),
        purpose,
        allowed_overlap_object_ids=allowed,
    )


def bed_side_zone(obj: dict[str, Any], side: str) -> dict[str, Any]:
    sign = -1.0 if side == "left" else 1.0
    zone_width = 0.62
    zone_depth = max(0.9, float(obj["bbox"]["depth"]) * 0.58)
    local_x = sign * (float(obj["bbox"]["width"]) * 0.5 + zone_width * 0.5)
    local_y = -float(obj["bbox"]["depth"]) * 0.12
    center = offset_from_object(obj, local_x, local_y)
    return make_zone(
        f"use_{obj['object_id']}_{side}",
        obj["object_id"],
        center,
        zone_width,
        zone_depth,
        float(obj["yaw_rad"]),
        f"bed_{side}_egress",
    )


class SceneVerifier:
    def __init__(
        self,
        hssd_records: dict[str, dict[str, Any]],
        semantic_mapping: dict[str, Any],
        rules: dict[str, Any],
    ) -> None:
        self.records = hssd_records
        self.semantic_mapping = semantic_mapping
        self.rules = rules

    def _separation_is_invalid(self, value: float) -> bool:
        geometry = self.rules["geometry"]
        minimum = float(geometry.get("accepted_min_signed_separation_m", 0.0))
        epsilon = float(geometry.get("numeric_separation_epsilon_m", EPSILON))
        return float(value) < minimum - epsilon

    def _separation_policy(self) -> dict[str, float]:
        geometry = self.rules["geometry"]
        return {
            "accepted_min_signed_separation_m": float(
                geometry.get("accepted_min_signed_separation_m", 0.0)
            ),
            "numeric_separation_epsilon_m": float(
                geometry.get("numeric_separation_epsilon_m", EPSILON)
            ),
        }

    def verify(self, scene: dict[str, Any]) -> dict[str, Any]:
        checks = {
            "finite_transform": self._check_finite(scene),
            "identity_bbox": self._check_identity_bbox(scene),
            "room_bounds": self._check_room_bounds(scene),
            "furniture_collision": self._check_collisions(scene),
            "opening_clearance": self._check_openings(scene),
            "functional_relations": self._check_relations(scene),
            "use_clearance": self._check_use_clearance(scene),
            "path_reachability": self._check_path(scene),
        }
        violations = [
            violation
            for check in checks.values()
            for violation in check["violations"]
        ]
        return {
            "passed": not violations,
            "checks": checks,
            "violations": violations,
        }

    def _check_finite(self, scene: dict[str, Any]) -> dict[str, Any]:
        violations: list[dict[str, Any]] = []
        for obj in scene["furniture"]:
            values = [
                obj["x"],
                obj["y"],
                obj["z"],
                obj["yaw_rad"],
                *obj["rotation_wxyz"],
                *obj["bbox"].values(),
            ]
            if not all(math.isfinite(float(value)) for value in values):
                violations.append(_violation("non_finite_transform", obj["object_id"]))
                continue
            norm = math.sqrt(sum(float(value) ** 2 for value in obj["rotation_wxyz"]))
            if abs(norm - 1.0) > self.rules["transform"]["quaternion_norm_tolerance"]:
                violations.append(
                    _violation("invalid_quaternion_norm", obj["object_id"], norm=norm)
                )
            expected = yaw_to_quaternion_wxyz(float(obj["yaw_rad"]))
            direct_error = max(
                abs(float(actual) - expected_value)
                for actual, expected_value in zip(obj["rotation_wxyz"], expected)
            )
            negated_error = max(
                abs(float(actual) + expected_value)
                for actual, expected_value in zip(obj["rotation_wxyz"], expected)
            )
            if min(direct_error, negated_error) > self.rules["transform"]["quaternion_component_tolerance"]:
                violations.append(
                    _violation("yaw_quaternion_mismatch", obj["object_id"])
                )
            if not -math.pi <= float(obj["yaw_rad"]) < math.pi:
                violations.append(_violation("yaw_not_wrapped", obj["object_id"]))
        return _check_result(violations)

    def _check_identity_bbox(self, scene: dict[str, Any]) -> dict[str, Any]:
        violations: list[dict[str, Any]] = []
        ids: set[str] = set()
        for obj in scene["furniture"]:
            object_id = obj["object_id"]
            if object_id in ids:
                violations.append(_violation("duplicate_object_id", object_id))
            ids.add(object_id)
            record = self.records.get(obj["hssd_id"])
            if record is None:
                violations.append(_violation("unknown_hssd_id", object_id))
                continue
            expected = hssd_bbox(record)
            if expected is None:
                violations.append(_violation("missing_hssd_bbox", object_id))
                continue
            tolerance = self.rules["identity_bbox"]["absolute_tolerance_m"]
            for key in ("width", "depth", "height"):
                if abs(float(obj["bbox"][key]) - float(expected[key])) > tolerance:
                    violations.append(
                        _violation(
                            "bbox_mismatch",
                            object_id,
                            field=key,
                            expected=expected[key],
                            actual=obj["bbox"][key],
                        )
                    )
            source = str(record.get("category", "")).lower()
            allowed = self.semantic_mapping["source_category_to_frozen"].get(source)
            if allowed != obj["category"]:
                violations.append(
                    _violation(
                        "category_mapping_mismatch",
                        object_id,
                        hssd_source_category=source,
                        frozen_category=obj["category"],
                    )
                )
            semantics = self.semantic_mapping["frozen_category_semantics"].get(
                obj["category"]
            )
            if semantics is None or semantics["family"] != obj["family"] or sorted(
                semantics["functions"]
            ) != sorted(obj["functions"]):
                violations.append(_violation("semantic_package_mismatch", object_id))
        if len(scene["furniture"]) > self.rules["scene"]["max_furniture"]:
            violations.append(_violation("furniture_budget_exceeded", "ROOM"))
        return _check_result(violations)

    def _check_room_bounds(self, scene: dict[str, Any]) -> dict[str, Any]:
        violations: list[dict[str, Any]] = []
        tolerance = self.rules["geometry"]["room_bounds_tolerance_m"]
        for obj in scene["furniture"]:
            margins = room_margins(
                object_rect(obj), float(scene["length"]), float(scene["width"])
            )
            if min(margins) < -tolerance:
                violations.append(
                    _violation(
                        "room_bounds",
                        obj["object_id"],
                        signed_margins_m=[round(value, 6) for value in margins],
                    )
                )
        return _check_result(violations)

    def _check_collisions(self, scene: dict[str, Any]) -> dict[str, Any]:
        violations: list[dict[str, Any]] = []
        furniture = scene["furniture"]
        minimum_separation = math.inf
        for index, left in enumerate(furniture):
            for right in furniture[index + 1 :]:
                separation = signed_separation(object_rect(left), object_rect(right))
                minimum_separation = min(minimum_separation, separation)
                if self._separation_is_invalid(separation):
                    violations.append(
                        _violation(
                            "furniture_collision",
                            left["object_id"],
                            target_id=right["object_id"],
                            signed_separation_m=round(separation, 6),
                        )
                    )
        return _check_result(
            violations,
            **self._separation_policy(),
            minimum_pairwise_signed_separation_m=(
                None
                if math.isinf(minimum_separation)
                else round(minimum_separation, 9)
            ),
        )

    def _check_openings(self, scene: dict[str, Any]) -> dict[str, Any]:
        violations: list[dict[str, Any]] = []
        for opening in scene["openings"]:
            cx, cy, cz = opening["clearance_center_xyz_m"]
            sx, sy, sz = opening["clearance_size_xyz_m"]
            zone = OrientedRectangle(cx, cy, sx, sy, 0.0)
            low_z, high_z = cz - sz * 0.5, cz + sz * 0.5
            for obj in scene["furniture"]:
                if float(obj["bbox"]["height"]) <= low_z + EPSILON or high_z <= 0.0:
                    continue
                separation = signed_separation(zone, object_rect(obj))
                if self._separation_is_invalid(separation):
                    violations.append(
                        _violation(
                            "opening_clearance",
                            obj["object_id"],
                            opening_id=opening["opening_id"],
                            signed_separation_m=round(separation, 6),
                        )
                    )
        return _check_result(violations)

    def _check_relations(self, scene: dict[str, Any]) -> dict[str, Any]:
        violations: list[dict[str, Any]] = []
        objects = {obj["object_id"]: obj for obj in scene["furniture"]}
        tolerance = math.radians(
            self.rules["functional_relations"]["orientation_tolerance_deg"]
        )
        seen: set[tuple[str, str]] = set()
        for edge in scene["functional_partners"]:
            signature = edge["source_id"], edge["target_id"]
            if signature in seen:
                violations.append(
                    _violation("duplicate_functional_relation", edge["source_id"])
                )
                continue
            seen.add(signature)
            source = objects.get(edge["source_id"])
            target = objects.get(edge["target_id"])
            if source is None or target is None or source is target:
                violations.append(
                    _violation(
                        "invalid_functional_endpoint",
                        edge["source_id"],
                        target_id=edge["target_id"],
                    )
                )
                continue
            mode = edge["orientation_mode"]
            error = 0.0
            if mode == "facing":
                expected = yaw_facing(
                    (float(source["x"]), float(source["y"])),
                    (float(target["x"]), float(target["y"])),
                )
                center_error = angular_error(float(source["yaw_rad"]), expected)
                dx = float(target["x"]) - float(source["x"])
                dy = float(target["y"]) - float(source["y"])
                distance = math.hypot(dx, dy)
                direction = (dx / distance, dy / distance)
                perpendicular = (-direction[1], direction[0])
                target_rect = object_rect(target)
                angular_extent = math.atan2(
                    target_rect.radius_on(perpendicular),
                    max(
                        distance - target_rect.radius_on(direction),
                        self.rules["functional_relations"]["gap_tolerance_m"],
                    ),
                )
                error = max(0.0, center_error - angular_extent)
            elif mode in ("aligned", "parallel"):
                error = angular_error(
                    float(source["yaw_rad"]),
                    float(target["yaw_rad"]),
                    modulo_pi=True,
                )
            elif mode == "perpendicular":
                difference = angular_error(
                    float(source["yaw_rad"]),
                    float(target["yaw_rad"]),
                    modulo_pi=True,
                )
                error = abs(difference - math.pi * 0.5)
            elif mode != "none":
                violations.append(
                    _violation("unknown_orientation_mode", edge["source_id"])
                )
            if error > tolerance:
                violations.append(
                    _violation(
                        "functional_orientation",
                        edge["source_id"],
                        target_id=edge["target_id"],
                        angular_error_deg=round(math.degrees(error), 3),
                    )
                )
            gap_range = edge.get("desired_gap_range")
            if gap_range is not None:
                gap = signed_separation(object_rect(source), object_rect(target))
                gap_tolerance = self.rules["functional_relations"][
                    "gap_tolerance_m"
                ]
                if gap < float(gap_range[0]) - gap_tolerance or gap > float(
                    gap_range[1]
                ) + gap_tolerance:
                    violations.append(
                        _violation(
                            "functional_gap",
                            edge["source_id"],
                            target_id=edge["target_id"],
                            actual_gap_m=round(gap, 6),
                            desired_gap_range=gap_range,
                        )
                    )
        return _check_result(violations)

    def _check_use_clearance(self, scene: dict[str, Any]) -> dict[str, Any]:
        violations: list[dict[str, Any]] = []
        objects = {obj["object_id"]: obj for obj in scene["furniture"]}
        room_length, room_width = float(scene["length"]), float(scene["width"])
        declared_owner_categories: Counter[str] = Counter()
        for zone in scene["audit"]["use_clearance_zones"]:
            owner = objects.get(zone["owner_id"])
            if owner is None:
                violations.append(_violation("use_zone_owner_missing", zone["zone_id"]))
                continue
            declared_owner_categories[owner["category"]] += 1
            rect = OrientedRectangle(
                float(zone["center_xy_m"][0]),
                float(zone["center_xy_m"][1]),
                float(zone["size_xy_m"][0]),
                float(zone["size_xy_m"][1]),
                float(zone["yaw_rad"]),
            )
            margins = room_margins(rect, room_length, room_width)
            if min(margins) < -self.rules["geometry"]["room_bounds_tolerance_m"]:
                violations.append(
                    _violation(
                        "use_zone_out_of_room", zone["zone_id"], margins=margins
                    )
                )
            allowed = set(zone.get("allowed_overlap_object_ids", []))
            for obj in scene["furniture"]:
                if obj["object_id"] == owner["object_id"] or obj["object_id"] in allowed:
                    continue
                if self._separation_is_invalid(
                    signed_separation(rect, object_rect(obj))
                ):
                    violations.append(
                        _violation(
                            "use_zone_blocked",
                            zone["zone_id"],
                            target_id=obj["object_id"],
                        )
                    )
            for opening in scene["openings"]:
                if opening["kind"] != "door":
                    continue
                cx, cy, _ = opening["clearance_center_xyz_m"]
                sx, sy, _ = opening["clearance_size_xyz_m"]
                opening_rect = OrientedRectangle(cx, cy, sx, sy, 0.0)
                if self._separation_is_invalid(
                    signed_separation(rect, opening_rect)
                ):
                    violations.append(
                        _violation(
                            "use_zone_blocks_opening",
                            zone["zone_id"],
                            opening_id=opening["opening_id"],
                        )
                    )
        required = self.rules["use_clearance"]["required_zone_categories"]
        for category in required.get(scene["room_type"], []):
            if any(obj["category"] == category for obj in scene["furniture"]):
                if declared_owner_categories[category] == 0:
                    violations.append(
                        _violation("required_use_zone_missing", category)
                    )
        return _check_result(violations)

    def _check_path(self, scene: dict[str, Any]) -> dict[str, Any]:
        violations: list[dict[str, Any]] = []
        door = next((item for item in scene["openings"] if item["kind"] == "door"), None)
        if door is None:
            return _check_result([_violation("door_missing", "ROOM")])
        goal = tuple(float(value) for value in scene["audit"]["primary_zone_goal_xy_m"])
        start = _door_path_start(door)
        reachable, details = grid_path_exists(
            float(scene["length"]),
            float(scene["width"]),
            [object_rect(obj) for obj in scene["furniture"]],
            start,
            goal,
            grid_m=self.rules["path"]["grid_resolution_m"],
            human_radius_m=self.rules["path"]["human_radius_m"],
        )
        if not reachable:
            violations.append(
                _violation(
                    "primary_zone_unreachable",
                    "ROOM",
                    start_xy_m=start,
                    goal_xy_m=goal,
                    **details,
                )
            )
        return _check_result(violations, details=details)


def grid_path_exists(
    length: float,
    width: float,
    obstacles: list[OrientedRectangle],
    start: tuple[float, float],
    goal: tuple[float, float],
    *,
    grid_m: float,
    human_radius_m: float,
) -> tuple[bool, dict[str, Any]]:
    nx = max(2, int(math.floor(length / grid_m)) + 1)
    ny = max(2, int(math.floor(width / grid_m)) + 1)
    origin_x, origin_y = -length * 0.5, -width * 0.5

    def cell(point: tuple[float, float]) -> tuple[int, int]:
        return (
            min(nx - 1, max(0, int(round((point[0] - origin_x) / grid_m)))),
            min(ny - 1, max(0, int(round((point[1] - origin_y) / grid_m)))),
        )

    def point(index: tuple[int, int]) -> tuple[float, float]:
        return origin_x + index[0] * grid_m, origin_y + index[1] * grid_m

    def blocked(index: tuple[int, int]) -> bool:
        px, py = point(index)
        if abs(px) > length * 0.5 - human_radius_m or abs(py) > width * 0.5 - human_radius_m:
            return True
        for obstacle in obstacles:
            expanded = OrientedRectangle(
                obstacle.x,
                obstacle.y,
                obstacle.width + human_radius_m * 2.0,
                obstacle.depth + human_radius_m * 2.0,
                obstacle.yaw,
            )
            if _point_in_rect((px, py), expanded):
                return True
        return False

    start_cell, goal_cell = cell(start), cell(goal)
    if blocked(start_cell):
        start_cell = _nearest_free(start_cell, nx, ny, blocked)
    if blocked(goal_cell):
        goal_cell = _nearest_free(goal_cell, nx, ny, blocked)
    if start_cell is None or goal_cell is None:
        return False, {"visited_cells": 0, "grid_shape": [nx, ny]}
    queue = deque([start_cell])
    visited = {start_cell}
    while queue:
        current = queue.popleft()
        if current == goal_cell:
            return True, {
                "visited_cells": len(visited),
                "grid_shape": [nx, ny],
                "resolved_start_cell": list(start_cell),
                "resolved_goal_cell": list(goal_cell),
            }
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = current[0] + dx, current[1] + dy
            if not (0 <= nxt[0] < nx and 0 <= nxt[1] < ny):
                continue
            if nxt in visited or blocked(nxt):
                continue
            visited.add(nxt)
            queue.append(nxt)
    return False, {"visited_cells": len(visited), "grid_shape": [nx, ny]}


def layout_hash(scene: dict[str, Any]) -> str:
    payload = {
        "room_type": scene["room_type"],
        "length": round(float(scene["length"]), 3),
        "width": round(float(scene["width"]), 3),
        "openings": [
            {
                "kind": value["kind"],
                "wall": value["wall"],
                "center": [round(float(v), 3) for v in value["clearance_center_xyz_m"]],
                "size": [round(float(v), 3) for v in value["clearance_size_xyz_m"]],
            }
            for value in scene["openings"]
        ],
        "furniture": sorted(
            (
                value["category"],
                round(float(value["x"]), 3),
                round(float(value["y"]), 3),
                round(float(value["yaw_rad"]), 3),
                round(float(value["bbox"]["width"]), 3),
                round(float(value["bbox"]["depth"]), 3),
            )
            for value in scene["furniture"]
        ),
    }
    return canonical_json_hash(payload)


def near_duplicate_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    if left["room_type"] != right["room_type"]:
        return math.inf
    left_categories = sorted(obj["category"] for obj in left["furniture"])
    right_categories = sorted(obj["category"] for obj in right["furniture"])
    if left_categories != right_categories:
        return math.inf
    if abs(float(left["length"]) - float(right["length"])) > 0.35 or abs(
        float(left["width"]) - float(right["width"])
    ) > 0.35:
        return math.inf
    left_rows = sorted(left["furniture"], key=lambda item: (item["category"], item["object_id"]))
    right_rows = sorted(right["furniture"], key=lambda item: (item["category"], item["object_id"]))
    squared: list[float] = []
    for first, second in zip(left_rows, right_rows):
        squared.extend(
            [
                ((float(first["x"]) - float(second["x"])) / max(left["length"], right["length"])) ** 2,
                ((float(first["y"]) - float(second["y"])) / max(left["width"], right["width"])) ** 2,
                (angular_error(float(first["yaw_rad"]), float(second["yaw_rad"]), modulo_pi=True) / math.pi) ** 2,
            ]
        )
    return math.sqrt(sum(squared) / len(squared)) if squared else 0.0


def render_topdown(scene: dict[str, Any], path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    scale = 130
    margin = 70
    width_px = int(round(float(scene["length"]) * scale)) + margin * 2
    height_px = int(round(float(scene["width"]) * scale)) + margin * 2
    image = Image.new("RGB", (width_px, height_px), "#f7f7f4")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()

    def px(point: tuple[float, float]) -> tuple[float, float]:
        return (
            margin + (point[0] + float(scene["length"]) * 0.5) * scale,
            margin + (float(scene["width"]) * 0.5 - point[1]) * scale,
        )

    room_box = [
        px((-float(scene["length"]) * 0.5, float(scene["width"]) * 0.5)),
        px((float(scene["length"]) * 0.5, -float(scene["width"]) * 0.5)),
    ]
    draw.rectangle(room_box, fill="#ffffff", outline="#222222", width=5)

    for opening in scene["openings"]:
        cx, cy, _ = opening["clearance_center_xyz_m"]
        sx, sy, _ = opening["clearance_size_xyz_m"]
        rect = OrientedRectangle(cx, cy, sx, sy, 0.0)
        color = "#e85d5d55" if opening["kind"] == "door" else "#4c9ed955"
        draw.polygon([px(value) for value in rect.corners], fill=color)
        draw.line([px(value) for value in rect.corners + [rect.corners[0]]], fill=color[:7], width=3)

    for zone in scene["audit"]["use_clearance_zones"]:
        rect = OrientedRectangle(
            zone["center_xy_m"][0], zone["center_xy_m"][1], zone["size_xy_m"][0], zone["size_xy_m"][1], zone["yaw_rad"]
        )
        draw.polygon([px(value) for value in rect.corners], fill="#f0b44d30")
        draw.line([px(value) for value in rect.corners + [rect.corners[0]]], fill="#b87920aa", width=1)

    palette = {
        "sleep_surface": "#5b8fd1",
        "seating": "#68a96b",
        "support_surface": "#d1a257",
        "storage": "#8d6cab",
        "lighting": "#d9cf4c",
    }
    for obj in scene["furniture"]:
        rect = object_rect(obj)
        color = palette.get(obj["family"], "#777777")
        draw.polygon([px(value) for value in rect.corners], fill=color + "cc")
        draw.line([px(value) for value in rect.corners + [rect.corners[0]]], fill="#222222", width=2)
        center = px((rect.x, rect.y))
        front = front_vector(rect.yaw)
        tip = px((rect.x + front[0] * min(rect.depth * 0.6, 0.55), rect.y + front[1] * min(rect.depth * 0.6, 0.55)))
        draw.line([center, tip], fill="#111111", width=3)
        label = obj["object_id"].replace("_", " ")
        draw.text((center[0] + 4, center[1] + 3), label, fill="#111111", font=font)

    goal = tuple(scene["audit"]["primary_zone_goal_xy_m"])
    gx, gy = px(goal)
    draw.ellipse([gx - 7, gy - 7, gx + 7, gy + 7], fill="#0b7a75", outline="#ffffff", width=2)
    for index, target in enumerate(scene["audit"].get("path_targets", [])):
        tx, ty = px(tuple(target["goal_xy_m"]))
        draw.ellipse(
            [tx - 5, ty - 5, tx + 5, ty + 5],
            fill="#0b7a75",
            outline="#ffffff",
            width=1,
        )
        draw.text(
            (tx + 6, ty - 9 - index % 2 * 9),
            str(target["target_id"]),
            fill="#075e5a",
            font=font,
        )
    title = f"{scene['scene_id']} | {scene['room_type']} | F={len(scene['furniture'])} | {scene['length']:.2f} x {scene['width']:.2f} m"
    draw.text((margin, 22), title, fill="#111111", font=font)
    draw.text((margin, height_px - 38), "Arrow = local -Y front; red = door clearance; blue = window clearance; amber = use clearance; teal = path target", fill="#333333", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _door_path_start(door: dict[str, Any]) -> tuple[float, float]:
    cx, cy, _ = door["clearance_center_xyz_m"]
    sx, sy, _ = door["clearance_size_xyz_m"]
    nx, ny = door["interior_normal_xy"]
    depth = sy if abs(ny) > 0.5 else sx
    return cx + nx * (depth * 0.5 + 0.35), cy + ny * (depth * 0.5 + 0.35)


def _nearest_free(start: tuple[int, int], nx: int, ny: int, blocked: Any) -> tuple[int, int] | None:
    queue = deque([start])
    visited = {start}
    while queue:
        current = queue.popleft()
        if not blocked(current):
            return current
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = current[0] + dx, current[1] + dy
            if 0 <= nxt[0] < nx and 0 <= nxt[1] < ny and nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
    return None


def _point_in_rect(point: tuple[float, float], rect: OrientedRectangle) -> bool:
    dx, dy = point[0] - rect.x, point[1] - rect.y
    local_x = dx * math.cos(rect.yaw) + dy * math.sin(rect.yaw)
    local_y = -dx * math.sin(rect.yaw) + dy * math.cos(rect.yaw)
    return abs(local_x) <= rect.width * 0.5 and abs(local_y) <= rect.depth * 0.5


def _plausible_furniture_bbox(category: str, extents: tuple[float, float, float]) -> bool:
    width, depth, height = extents
    if not (0.08 <= width <= 3.5 and 0.08 <= depth <= 3.0 and 0.12 <= height <= 3.0):
        return False
    if category == "bed":
        return width >= 0.75 and depth >= 0.75
    if category in {"wardrobe", "bookcase", "cabinet"}:
        return height >= 0.65
    if category in {"sofa", "bench", "tv_bench", "console_table"}:
        return width >= 0.7
    return True


def _valid_extent_triplet(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(item, (int, float)) and math.isfinite(item) for item in value)
        and all(float(item) > 0.0 for item in value)
    )


def _finite_triplet(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(item, (int, float)) and math.isfinite(item) for item in value)
    )


def _dot(left: tuple[float, float], right: tuple[float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1]


def _violation(kind: str, object_id: str, **details: Any) -> dict[str, Any]:
    return {"type": kind, "object_id": object_id, **details}


def _check_result(
    violations: list[dict[str, Any]], **details: Any
) -> dict[str, Any]:
    return {"passed": not violations, "violations": violations, **details}
