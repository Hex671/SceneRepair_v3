"""Scene Functional Usability Rate (SFUR) v1 evaluation contract."""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .geometry import (
    OrientedRectangle,
    room_signed_margins,
    signed_separation_and_escape,
    wrap_yaw,
)
from .graph import FurnitureInput, OpeningInput, SceneInput
from .repair import hard_constraint_report


SFUR_RULE_SCHEMA = "scene_repair_v3_sfur_rules_v1"


@dataclass(frozen=True)
class FunctionalPairRule:
    rule_id: str
    categories: frozenset[str]
    subject_categories: frozenset[str]
    minimum_gap_m: float
    maximum_gap_m: float
    facing_tolerance_deg: Mapping[str, float]


@dataclass(frozen=True)
class SFURRules:
    functional_score_threshold: float
    geometry_tolerance_m: float
    grid_resolution_m: float
    human_radius_m: float
    relation_weight: float
    clearance_weight: float
    reachability_weight: float
    pair_rules: tuple[FunctionalPairRule, ...]

    @classmethod
    def load(cls, path: str | Path) -> "SFURRules":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("schema_version") != SFUR_RULE_SCHEMA:
            raise ValueError("unsupported SFUR rule schema")
        weights = raw["weights"]
        weight_values = (
            float(weights["functional_relations"]),
            float(weights["use_clearance"]),
            float(weights["reachability"]),
        )
        if not math.isclose(sum(weight_values), 1.0, abs_tol=1.0e-9):
            raise ValueError("SFUR component weights must sum to one")
        pair_rules: list[FunctionalPairRule] = []
        for item in raw["pair_rules"]:
            categories = frozenset(str(value) for value in item["categories"])
            subjects = frozenset(str(value) for value in item["subjects"])
            gap = tuple(float(value) for value in item["gap_range_m"])
            if (
                len(categories) != 2
                or not subjects
                or not subjects.issubset(categories)
                or len(gap) != 2
                or not 0 <= gap[0] <= gap[1]
            ):
                raise ValueError(f"invalid SFUR pair rule {item.get('rule_id')!r}")
            pair_rules.append(
                FunctionalPairRule(
                    rule_id=str(item["rule_id"]),
                    categories=categories,
                    subject_categories=subjects,
                    minimum_gap_m=gap[0],
                    maximum_gap_m=gap[1],
                    facing_tolerance_deg={
                        str(key): float(value)
                        for key, value in item.get("facing", {}).items()
                    },
                )
            )
        path_config = raw["path"]
        result = cls(
            functional_score_threshold=float(raw["functional_score_threshold"]),
            geometry_tolerance_m=float(raw["geometry_tolerance_m"]),
            grid_resolution_m=float(path_config["grid_resolution_m"]),
            human_radius_m=float(path_config["human_radius_m"]),
            relation_weight=weight_values[0],
            clearance_weight=weight_values[1],
            reachability_weight=weight_values[2],
            pair_rules=tuple(pair_rules),
        )
        if not 0.0 <= result.functional_score_threshold <= 1.0:
            raise ValueError("functional score threshold must be in [0,1]")
        if min(
            result.geometry_tolerance_m,
            result.grid_resolution_m,
            result.human_radius_m,
        ) <= 0.0:
            raise ValueError("SFUR geometric parameters must be positive")
        return result

    def rule_for(self, left_category: str, right_category: str) -> FunctionalPairRule | None:
        signature = frozenset((left_category, right_category))
        return next((rule for rule in self.pair_rules if rule.categories == signature), None)


@dataclass(frozen=True)
class LocalUseZone:
    zone_id: str
    owner_id: str
    local_center_xy_m: tuple[float, float]
    size_xy_m: tuple[float, float]
    relative_yaw_rad: float
    allowed_overlap_object_ids: frozenset[str]


@dataclass(frozen=True)
class LocalPathTarget:
    target_id: str
    local_goal_xy_m: tuple[float, float]


@dataclass(frozen=True)
class FunctionalSceneContract:
    use_zones: tuple[LocalUseZone, ...]
    path_targets: tuple[LocalPathTarget, ...]

    @classmethod
    def from_raw_scene(cls, raw: Mapping[str, Any]) -> "FunctionalSceneContract":
        objects = {str(item["object_id"]): item for item in raw.get("furniture", ())}
        audit = raw.get("audit") or {}
        use_zones: list[LocalUseZone] = []
        for zone in audit.get("use_clearance_zones") or ():
            if not zone.get("validity", True):
                continue
            owner_id = str(zone["owner_id"])
            owner = objects.get(owner_id)
            if owner is None:
                continue
            local_center = _world_offset_to_local(
                float(zone["center_xy_m"][0]) - float(owner["x"]),
                float(zone["center_xy_m"][1]) - float(owner["y"]),
                float(owner.get("yaw_rad", 0.0)),
            )
            use_zones.append(
                LocalUseZone(
                    zone_id=str(zone["zone_id"]),
                    owner_id=owner_id,
                    local_center_xy_m=local_center,
                    size_xy_m=(
                        float(zone["size_xy_m"][0]),
                        float(zone["size_xy_m"][1]),
                    ),
                    relative_yaw_rad=wrap_yaw(
                        float(zone["yaw_rad"]) - float(owner.get("yaw_rad", 0.0))
                    ),
                    allowed_overlap_object_ids=frozenset(
                        str(value) for value in zone.get("allowed_overlap_object_ids", ())
                    ),
                )
            )
        path_targets: list[LocalPathTarget] = []
        for target in audit.get("path_targets") or ():
            target_id = str(target["target_id"])
            owner = objects.get(target_id)
            if owner is None:
                continue
            local_goal = _world_offset_to_local(
                float(target["goal_xy_m"][0]) - float(owner["x"]),
                float(target["goal_xy_m"][1]) - float(owner["y"]),
                float(owner.get("yaw_rad", 0.0)),
            )
            path_targets.append(LocalPathTarget(target_id, local_goal))
        return cls(tuple(use_zones), tuple(path_targets))


@dataclass(frozen=True)
class SFURViolation:
    kind: str
    subject_id: str
    target_id: str | None = None
    rule_id: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class SFURSceneReport:
    scene_pass: bool
    hard_constraints_pass: bool
    functional_score: float
    relation_pass_rate: float
    clearance_pass_rate: float
    reachability_rate: float
    relation_checks: int
    clearance_checks: int
    path_checks: int
    violations: tuple[SFURViolation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_functional_pass": self.scene_pass,
            "hard_constraints_pass": self.hard_constraints_pass,
            "functional_score": self.functional_score,
            "functional_relation_pass_rate": self.relation_pass_rate,
            "interaction_clearance_pass_rate": self.clearance_pass_rate,
            "core_reachability": self.reachability_rate,
            "applicable_rule_count": (
                self.relation_checks + self.clearance_checks + self.path_checks
            ),
            "relation_checks": self.relation_checks,
            "clearance_checks": self.clearance_checks,
            "path_checks": self.path_checks,
            "violations": [vars(value) for value in self.violations],
        }


class FunctionalLayoutEvaluator:
    """Evaluate functionality without reading clean poses or corruption identity."""

    def __init__(self, rules: SFURRules) -> None:
        self.rules = rules

    def evaluate(
        self, scene: SceneInput, contract: FunctionalSceneContract
    ) -> SFURSceneReport:
        hard_pass = hard_constraint_report(scene).total == 0
        relation_passes, relation_violations = self._relations(scene)
        clearance_passes, clearance_violations = self._clearances(scene, contract)
        path_passes, path_violations = self._paths(scene, contract)
        relation_rate = _pass_rate(relation_passes)
        clearance_rate = _pass_rate(clearance_passes)
        path_rate = _pass_rate(path_passes)
        score = (
            self.rules.relation_weight * relation_rate
            + self.rules.clearance_weight * clearance_rate
            + self.rules.reachability_weight * path_rate
        )
        violations = tuple(
            [*relation_violations, *clearance_violations, *path_violations]
        )
        scene_pass = (
            hard_pass
            and score + 1.0e-9 >= self.rules.functional_score_threshold
            and path_rate == 1.0
        )
        return SFURSceneReport(
            scene_pass=scene_pass,
            hard_constraints_pass=hard_pass,
            functional_score=score,
            relation_pass_rate=relation_rate,
            clearance_pass_rate=clearance_rate,
            reachability_rate=path_rate,
            relation_checks=len(relation_passes),
            clearance_checks=len(clearance_passes),
            path_checks=len(path_passes),
            violations=violations,
        )

    def _relations(
        self, scene: SceneInput
    ) -> tuple[list[bool], list[SFURViolation]]:
        objects = {item.object_id: item for item in scene.furniture}
        connected = {
            tuple(sorted((edge.source_id, edge.target_id)))
            for edge in scene.functional_partners
            if edge.valid and edge.source_id != edge.target_id
        }
        passes: list[bool] = []
        violations: list[SFURViolation] = []
        for rule in self.rules.pair_rules:
            for subject in sorted(objects.values(), key=lambda value: value.object_id):
                if subject.category not in rule.subject_categories:
                    continue
                target_category = next(
                    value for value in rule.categories if value != subject.category
                )
                candidates = [
                    candidate
                    for candidate in objects.values()
                    if candidate.category == target_category
                    and tuple(sorted((subject.object_id, candidate.object_id))) in connected
                ]
                if not candidates:
                    continue
                target = min(
                    candidates,
                    key=lambda candidate: math.hypot(
                        candidate.x_m - subject.x_m, candidate.y_m - subject.y_m
                    ),
                )
                self._check_relation(subject, target, rule, passes, violations)
        return passes, violations

    def _check_relation(
        self,
        subject: FurnitureInput,
        target: FurnitureInput,
        rule: FunctionalPairRule,
        passes: list[bool],
        violations: list[SFURViolation],
    ) -> None:
        gap, _, _ = signed_separation_and_escape(
            _rectangle(subject), _rectangle(target)
        )
        failures: list[str] = []
        if not (
            rule.minimum_gap_m - self.rules.geometry_tolerance_m
            <= gap
            <= rule.maximum_gap_m + self.rules.geometry_tolerance_m
        ):
            failures.append(f"gap={gap:.3f}m")
        tolerance = rule.facing_tolerance_deg.get(subject.category)
        if tolerance is not None:
            error = math.degrees(_facing_error(subject, target))
            if error > tolerance + 1.0e-6:
                failures.append(f"{subject.object_id}_facing_error={error:.1f}deg")
        passed = not failures
        passes.append(passed)
        if not passed:
            violations.append(
                SFURViolation(
                    "functional_relation",
                    subject.object_id,
                    target.object_id,
                    rule.rule_id,
                    "; ".join(failures),
                )
            )

    def _clearances(
        self, scene: SceneInput, contract: FunctionalSceneContract
    ) -> tuple[list[bool], list[SFURViolation]]:
        objects = {item.object_id: item for item in scene.furniture}
        passes: list[bool] = []
        violations: list[SFURViolation] = []
        for zone in contract.use_zones:
            owner = objects.get(zone.owner_id)
            if owner is None:
                continue
            rectangle = _zone_rectangle(owner, zone)
            failures: list[str] = []
            if min(
                room_signed_margins(
                    rectangle, scene.room.length_m, scene.room.width_m
                )
            ) < -self.rules.geometry_tolerance_m:
                failures.append("outside_room")
            for other in scene.furniture:
                if (
                    other.object_id == owner.object_id
                    or other.object_id in zone.allowed_overlap_object_ids
                ):
                    continue
                separation, _, _ = signed_separation_and_escape(
                    rectangle, _rectangle(other)
                )
                if separation < -self.rules.geometry_tolerance_m:
                    failures.append(f"blocked_by={other.object_id}")
            for opening in scene.openings:
                if opening.kind != "door" or not opening.constraint_valid:
                    continue
                separation, _, _ = signed_separation_and_escape(
                    rectangle, _opening_rectangle(opening)
                )
                if separation < -self.rules.geometry_tolerance_m:
                    failures.append(f"blocks_door={opening.opening_id}")
            passed = not failures
            passes.append(passed)
            if not passed:
                violations.append(
                    SFURViolation(
                        "use_clearance", zone.zone_id, owner.object_id, detail="; ".join(failures)
                    )
                )
        return passes, violations

    def _paths(
        self, scene: SceneInput, contract: FunctionalSceneContract
    ) -> tuple[list[bool], list[SFURViolation]]:
        objects = {item.object_id: item for item in scene.furniture}
        door = next(
            (
                opening
                for opening in scene.openings
                if opening.kind == "door" and opening.constraint_valid
            ),
            None,
        )
        if door is None:
            return [False], [SFURViolation("reachability", "ROOM", detail="door_missing")]
        start = _door_path_start(door)
        obstacles = [_rectangle(item) for item in scene.furniture]
        passes: list[bool] = []
        violations: list[SFURViolation] = []
        targets: list[tuple[str, tuple[float, float]]] = []
        for target in contract.path_targets:
            owner = objects.get(target.target_id)
            if owner is None:
                continue
            goal_offset = _local_offset_to_world(
                target.local_goal_xy_m[0], target.local_goal_xy_m[1], owner.yaw_rad
            )
            goal = owner.x_m + goal_offset[0], owner.y_m + goal_offset[1]
            targets.append((target.target_id, goal))
        reachability = _grid_path_targets_exist(
            scene.room.length_m,
            scene.room.width_m,
            obstacles,
            start,
            [value[1] for value in targets],
            grid_m=self.rules.grid_resolution_m,
            human_radius_m=self.rules.human_radius_m,
        )
        for (target_id, _goal), reachable in zip(targets, reachability):
            passes.append(reachable)
            if not reachable:
                violations.append(
                    SFURViolation("reachability", target_id, detail="no_path_from_door")
                )
        return passes, violations


def _pass_rate(values: Sequence[bool]) -> float:
    return sum(bool(value) for value in values) / len(values) if values else 1.0


def _rectangle(item: FurnitureInput) -> OrientedRectangle:
    return OrientedRectangle(
        item.x_m,
        item.y_m,
        item.bbox_width_m,
        item.bbox_depth_m,
        item.yaw_rad,
    )


def _opening_rectangle(opening: OpeningInput) -> OrientedRectangle:
    return OrientedRectangle(
        opening.clearance_center_xyz_m[0],
        opening.clearance_center_xyz_m[1],
        opening.clearance_size_xyz_m[0],
        opening.clearance_size_xyz_m[1],
        0.0,
    )


def _world_offset_to_local(dx: float, dy: float, yaw: float) -> tuple[float, float]:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return cosine * dx + sine * dy, -sine * dx + cosine * dy


def _local_offset_to_world(x: float, y: float, yaw: float) -> tuple[float, float]:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return cosine * x - sine * y, sine * x + cosine * y


def _zone_rectangle(owner: FurnitureInput, zone: LocalUseZone) -> OrientedRectangle:
    dx, dy = _local_offset_to_world(
        zone.local_center_xy_m[0], zone.local_center_xy_m[1], owner.yaw_rad
    )
    return OrientedRectangle(
        owner.x_m + dx,
        owner.y_m + dy,
        zone.size_xy_m[0],
        zone.size_xy_m[1],
        wrap_yaw(owner.yaw_rad + zone.relative_yaw_rad),
    )


def _facing_error(source: FurnitureInput, target: FurnitureInput) -> float:
    dx, dy = target.x_m - source.x_m, target.y_m - source.y_m
    if math.hypot(dx, dy) <= 1.0e-9:
        return math.pi
    expected_yaw = math.atan2(dx, -dy)
    return abs(wrap_yaw(source.yaw_rad - expected_yaw))


def _door_path_start(door: OpeningInput) -> tuple[float, float]:
    cx, cy, _ = door.clearance_center_xyz_m
    sx, sy, _ = door.clearance_size_xyz_m
    nx, ny = door.interior_normal_xy
    depth = sy if abs(ny) > 0.5 else sx
    return cx + nx * (depth * 0.5 + 0.35), cy + ny * (depth * 0.5 + 0.35)


def _grid_path_targets_exist(
    length: float,
    width: float,
    obstacles: Sequence[OrientedRectangle],
    start: tuple[float, float],
    goals: Sequence[tuple[float, float]],
    *,
    grid_m: float,
    human_radius_m: float,
) -> list[bool]:
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

    def point_in_rect(value: tuple[float, float], rect: OrientedRectangle) -> bool:
        local = _world_offset_to_local(
            value[0] - rect.center_x, value[1] - rect.center_y, rect.yaw
        )
        return abs(local[0]) <= rect.width * 0.5 and abs(local[1]) <= rect.depth * 0.5

    expanded = tuple(
        OrientedRectangle(
            item.center_x,
            item.center_y,
            item.width + human_radius_m * 2.0,
            item.depth + human_radius_m * 2.0,
            item.yaw,
        )
        for item in obstacles
    )

    def blocked(index: tuple[int, int]) -> bool:
        px, py = point(index)
        if (
            abs(px) > length * 0.5 - human_radius_m
            or abs(py) > width * 0.5 - human_radius_m
        ):
            return True
        return any(point_in_rect((px, py), item) for item in expanded)

    def nearest_free(initial: tuple[int, int]) -> tuple[int, int] | None:
        queue = deque([initial])
        visited = {initial}
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

    start_cell = nearest_free(cell(start))
    goal_cells = [nearest_free(cell(goal)) for goal in goals]
    if start_cell is None:
        return [False] * len(goals)
    queue = deque([start_cell])
    visited = {start_cell}
    while queue:
        current = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = current[0] + dx, current[1] + dy
            if not (0 <= nxt[0] < nx and 0 <= nxt[1] < ny):
                continue
            if nxt in visited or blocked(nxt):
                continue
            visited.add(nxt)
            queue.append(nxt)
    return [goal is not None and goal in visited for goal in goal_cells]
