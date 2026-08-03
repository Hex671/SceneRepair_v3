"""Deterministic whole-scene projection for observable hard constraints."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import torch

from .graph import FurnitureInput, SceneInput
from .geometry import (
    OrientedRectangle,
    room_signed_margins,
    signed_separation_and_escape,
    wrap_yaw,
)


@dataclass(frozen=True)
class HardConstraintReport:
    boundary: int
    collision: int
    opening: int

    @property
    def total(self) -> int:
        return self.boundary + self.collision + self.opening


@dataclass(frozen=True)
class JointRepairResult:
    scene: SceneInput
    delta_m_rad: torch.Tensor
    report: HardConstraintReport
    iterations: int
    converged: bool
    proposal_scale: float = 1.0


@dataclass(frozen=True)
class ConstraintRefinementConfig:
    """Frozen inference contract for the model's hard-constraint layer."""

    clearance_m: float = 0.06
    max_iterations: int = 160

    def __post_init__(self) -> None:
        if self.clearance_m < 0.0:
            raise ValueError("clearance_m must be non-negative")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "furniture_constraint_refinement_v1",
            "clearance_m": self.clearance_m,
            "max_iterations": self.max_iterations,
            "constraints": (
                "room_boundary",
                "furniture_collision",
                "opening_clearance",
            ),
        }


def apply_pose_deltas(scene: SceneInput, delta_m_rad: torch.Tensor) -> SceneInput:
    if delta_m_rad.shape != (len(scene.furniture), 3):
        raise ValueError("delta_m_rad must be [F,3]")
    furniture = []
    for item, delta in zip(scene.furniture, delta_m_rad.detach().cpu().tolist()):
        furniture.append(
            replace(
                item,
                x_m=item.x_m + float(delta[0]),
                y_m=item.y_m + float(delta[1]),
                yaw_rad=wrap_yaw(item.yaw_rad + float(delta[2])),
            )
        )
    return replace(scene, furniture=tuple(furniture))


def hard_constraint_report(
    scene: SceneInput, *, tolerance_m: float = 1.0e-6
) -> HardConstraintReport:
    rectangles = [_rectangle(item) for item in scene.furniture]
    boundary = sum(
        min(room_signed_margins(rect, scene.room.length_m, scene.room.width_m))
        < -tolerance_m
        for rect in rectangles
    )
    collision = 0
    for left in range(len(rectangles)):
        for right in range(left + 1, len(rectangles)):
            separation, _, _ = signed_separation_and_escape(
                rectangles[left], rectangles[right]
            )
            collision += separation < -tolerance_m
    opening = 0
    for item, rectangle in zip(scene.furniture, rectangles):
        for opening_value in scene.openings:
            if _opening_overlap(item, rectangle, opening_value) < -tolerance_m:
                opening += 1
    return HardConstraintReport(boundary, collision, opening)


def joint_translation_repair(
    scene: SceneInput,
    *,
    clearance_m: float = 0.06,
    max_iterations: int = 160,
) -> JointRepairResult:
    """Project all movable Furniture into a jointly feasible hard-constraint set.

    Pair penetrations are split between both movable objects. Boundary and opening
    corrections are then accumulated with pair corrections before each update,
    so the result does not depend on a hidden clean pose or corruption identity.
    """

    if clearance_m < 0.0 or max_iterations <= 0:
        raise ValueError("clearance_m must be non-negative and max_iterations positive")
    original = scene
    working = scene
    for iteration in range(1, max_iterations + 1):
        report = hard_constraint_report(working)
        if report.total == 0:
            return _result(original, working, report, iteration - 1, True)
        furniture = list(working.furniture)

        # Boundary projection is exact for a fixed yaw because the room is axis aligned.
        for index, item in enumerate(furniture):
            if _movable(item):
                furniture[index] = _project_inside_room(
                    item, working, clearance_m
                )

        # Openings are static obstacles. Project immediately so later constraints see
        # the updated pose instead of cancelling corrections in a shared average.
        for index, item in enumerate(furniture):
            if not _movable(item):
                continue
            for opening_value in working.openings:
                rectangle = _rectangle(item)
                separation, escape_x, escape_y = _opening_escape(
                    item, rectangle, opening_value
                )
                if separation < 0.0:
                    magnitude = math.hypot(escape_x, escape_y)
                    if magnitude > 1.0e-12:
                        scale = (clearance_m - separation) / magnitude
                        item = replace(
                            item,
                            x_m=item.x_m + escape_x * scale,
                            y_m=item.y_m + escape_y * scale,
                        )
                        item = _project_inside_room(item, working, clearance_m)
            furniture[index] = item

        # Sequential pair projection is deterministic and converges substantially
        # better than applying all stale pair vectors at once in dense scenes.
        for left in range(len(furniture)):
            for right in range(left + 1, len(furniture)):
                separation, escape_x, escape_y = signed_separation_and_escape(
                    _rectangle(furniture[left]), _rectangle(furniture[right])
                )
                if separation >= 0.0:
                    continue
                magnitude = math.hypot(escape_x, escape_y)
                if magnitude <= 1.0e-12:
                    continue
                scale = (clearance_m - separation) / magnitude
                escape_x *= scale
                escape_y *= scale
                left_movable = _movable(furniture[left])
                right_movable = _movable(furniture[right])
                if left_movable and right_movable:
                    left_share = right_share = 0.5
                elif left_movable:
                    left_share, right_share = 1.0, 0.0
                elif right_movable:
                    left_share, right_share = 0.0, 1.0
                else:
                    continue
                if left_share:
                    furniture[left] = replace(
                        furniture[left],
                        x_m=furniture[left].x_m - escape_x * left_share,
                        y_m=furniture[left].y_m - escape_y * left_share,
                    )
                    furniture[left] = _project_inside_room(
                        furniture[left], working, clearance_m
                    )
                if right_share:
                    furniture[right] = replace(
                        furniture[right],
                        x_m=furniture[right].x_m + escape_x * right_share,
                        y_m=furniture[right].y_m + escape_y * right_share,
                    )
                    furniture[right] = _project_inside_room(
                        furniture[right], working, clearance_m
                    )
        working = replace(working, furniture=tuple(furniture))

    report = hard_constraint_report(working)
    if report.total:
        working = _search_residual_placements(
            original,
            working,
            clearance_m=clearance_m,
            max_translation_m=3.0,
        )
        report = hard_constraint_report(working)
    return _result(original, working, report, max_iterations, report.total == 0)


def project_predicted_repair(
    scene: SceneInput,
    predicted_delta_m_rad: torch.Tensor,
    *,
    clearance_m: float = 0.06,
    max_iterations: int = 160,
) -> JointRepairResult:
    """Apply the learned proposal, then project only its residual violations."""

    proposed = apply_pose_deltas(scene, predicted_delta_m_rad)
    projected = joint_translation_repair(
        proposed,
        clearance_m=clearance_m,
        max_iterations=max_iterations,
    )
    report = hard_constraint_report(projected.scene)
    return _result(
        scene,
        projected.scene,
        report,
        projected.iterations,
        projected.converged,
    )


def _result(
    original: SceneInput,
    repaired: SceneInput,
    report: HardConstraintReport,
    iterations: int,
    converged: bool,
) -> JointRepairResult:
    delta = torch.tensor(
        [
            [
                after.x_m - before.x_m,
                after.y_m - before.y_m,
                wrap_yaw(after.yaw_rad - before.yaw_rad),
            ]
            for before, after in zip(original.furniture, repaired.furniture)
        ],
        dtype=torch.float32,
    )
    return JointRepairResult(repaired, delta, report, iterations, converged)


def _rectangle(item: FurnitureInput) -> OrientedRectangle:
    return OrientedRectangle(
        item.x_m,
        item.y_m,
        item.bbox_width_m,
        item.bbox_depth_m,
        item.yaw_rad,
    )


def _movable(item: FurnitureInput) -> bool:
    return item.movable and item.transform_valid


def _project_inside_room(
    item: FurnitureInput, scene: SceneInput, clearance_m: float
) -> FurnitureInput:
    half_x, half_y = _rectangle(item).room_aabb_half_extents
    low_x = -scene.room.length_m * 0.5 + half_x + clearance_m
    high_x = scene.room.length_m * 0.5 - half_x - clearance_m
    low_y = -scene.room.width_m * 0.5 + half_y + clearance_m
    high_y = scene.room.width_m * 0.5 - half_y - clearance_m
    if low_x > high_x or low_y > high_y:
        return item
    return replace(
        item,
        x_m=min(max(item.x_m, low_x), high_x),
        y_m=min(max(item.y_m, low_y), high_y),
    )


def _search_residual_placements(
    original: SceneInput,
    working: SceneInput,
    *,
    clearance_m: float,
    max_translation_m: float,
) -> SceneInput:
    """Resolve rare projection cycles by nearest deterministic local search."""

    result = working
    angle_count = 48
    radii = [0.05 * step for step in range(1, int(max_translation_m / 0.05) + 1)]
    for _ in range(max(2, len(result.furniture) * 2)):
        current_report = hard_constraint_report(result)
        if current_report.total == 0:
            break
        violating = [
            index
            for index, item in enumerate(result.furniture)
            if _movable(item) and _item_has_violation(result, index)
        ]
        best: tuple[int, float, int, FurnitureInput] | None = None
        for index in violating:
            origin = original.furniture[index]
            item = result.furniture[index]
            for radius in radii:
                for angle_index in range(angle_count):
                    angle = 2.0 * math.pi * angle_index / angle_count
                    candidate = replace(
                        item,
                        x_m=origin.x_m + radius * math.cos(angle),
                        y_m=origin.y_m + radius * math.sin(angle),
                    )
                    candidate = _project_inside_room(candidate, result, clearance_m)
                    if math.hypot(
                        candidate.x_m - origin.x_m,
                        candidate.y_m - origin.y_m,
                    ) > max_translation_m + 1.0e-6:
                        continue
                    candidate_scene = _replace_item(result, index, candidate)
                    item_violations = _item_violation_count(candidate_scene, index)
                    if item_violations:
                        continue
                    total = hard_constraint_report(candidate_scene).total
                    distance = math.hypot(
                        candidate.x_m - origin.x_m,
                        candidate.y_m - origin.y_m,
                    )
                    score = (total, distance, index, candidate)
                    if best is None or score[:3] < best[:3]:
                        best = score
                if best is not None and best[0] < current_report.total:
                    break
        if best is None or best[0] >= current_report.total:
            break
        result = _replace_item(result, best[2], best[3])
    return result


def _replace_item(scene: SceneInput, index: int, item: FurnitureInput) -> SceneInput:
    furniture = list(scene.furniture)
    furniture[index] = item
    return replace(scene, furniture=tuple(furniture))


def _item_has_violation(scene: SceneInput, index: int) -> bool:
    return _item_violation_count(scene, index) > 0


def _item_violation_count(scene: SceneInput, index: int) -> int:
    item = scene.furniture[index]
    rectangle = _rectangle(item)
    count = int(
        min(room_signed_margins(rectangle, scene.room.length_m, scene.room.width_m))
        < -1.0e-6
    )
    for other_index, other in enumerate(scene.furniture):
        if other_index == index:
            continue
        separation, _, _ = signed_separation_and_escape(
            _rectangle(other), rectangle
        )
        count += separation < -1.0e-6
    for opening_value in scene.openings:
        count += _opening_overlap(item, rectangle, opening_value) < -1.0e-6
    return count


def _opening_overlap(item, rectangle, opening) -> float:
    cx, cy, cz = opening.clearance_center_xyz_m
    sx, sy, sz = opening.clearance_size_xyz_m
    opening_rectangle = OrientedRectangle(cx, cy, sx, sy, 0.0)
    xy_separation, _, _ = signed_separation_and_escape(opening_rectangle, rectangle)
    z_separation = max(cz - sz * 0.5 - item.bbox_height_m, -(cz + sz * 0.5))
    return max(xy_separation, z_separation)


def _opening_escape(item, rectangle, opening) -> tuple[float, float, float]:
    separation = _opening_overlap(item, rectangle, opening)
    if separation >= 0.0:
        return separation, 0.0, 0.0
    cx, cy, _ = opening.clearance_center_xyz_m
    sx, sy, _ = opening.clearance_size_xyz_m
    opening_rectangle = OrientedRectangle(cx, cy, sx, sy, 0.0)
    _, escape_x, escape_y = signed_separation_and_escape(opening_rectangle, rectangle)
    return separation, escape_x, escape_y
