"""Model-owned deterministic refinement of residual SFUR constraints."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable

import torch
from torch import nn

from .contracts import ActionRange
from .geometry import OrientedRectangle, room_signed_margins, signed_separation_and_escape, wrap_yaw
from .graph import FurnitureInput, SceneInput
from .repair import hard_constraint_report, joint_translation_repair
from .sfur import (
    FunctionalLayoutEvaluator,
    FunctionalPairRule,
    FunctionalSceneContract,
    SFURRules,
    SFURSceneReport,
    _door_path_start,
    _local_offset_to_world,
    _opening_rectangle,
    _rectangle,
    _zone_rectangle,
)


FUNCTIONAL_REFINEMENT_SCHEMA = "furniture_functional_refinement_v1"


@dataclass(frozen=True)
class FunctionalRefinementConfig:
    max_iterations: int = 24
    hard_clearance_m: float = 0.06
    escape_padding_m: float = 0.08
    corridor_padding_m: float = 0.64

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("functional refinement iterations must be positive")
        if min(
            self.hard_clearance_m,
            self.escape_padding_m,
            self.corridor_padding_m,
        ) <= 0.0:
            raise ValueError("functional refinement distances must be positive")


@dataclass(frozen=True)
class FunctionalRefinementResult:
    scene: SceneInput
    report: SFURSceneReport
    delta_m_rad: torch.Tensor
    iterations: int
    converged: bool


class FunctionalConstraintRefinementLayer(nn.Module):
    """Greedy semantic projection using only the runtime functional contract."""

    def forward(
        self,
        original_scene: SceneInput,
        proposed_scene: SceneInput,
        contract: FunctionalSceneContract,
        rules: SFURRules,
        action_range: ActionRange,
        *,
        config: FunctionalRefinementConfig | None = None,
    ) -> FunctionalRefinementResult:
        active_config = config or FunctionalRefinementConfig()
        evaluator = FunctionalLayoutEvaluator(rules)
        current = proposed_scene
        current_report = evaluator.evaluate(current, contract)
        iterations = 0
        for iterations in range(active_config.max_iterations + 1):
            if current_report.scene_pass or iterations == active_config.max_iterations:
                break
            best_scene = current
            best_report = current_report
            best_rank = _report_rank(current_report)
            for candidate in self._candidates(
                current,
                current_report,
                contract,
                rules,
                active_config,
            ):
                projected = joint_translation_repair(
                    candidate,
                    clearance_m=active_config.hard_clearance_m,
                )
                if not projected.converged or projected.report.total != 0:
                    continue
                if not _within_action_range(
                    original_scene, projected.scene, action_range
                ):
                    continue
                report = evaluator.evaluate(projected.scene, contract)
                rank = _report_rank(report)
                if rank > best_rank:
                    best_scene, best_report, best_rank = projected.scene, report, rank
            if best_scene is current:
                break
            current, current_report = best_scene, best_report
        delta = _pose_delta(original_scene, current)
        return FunctionalRefinementResult(
            scene=current,
            report=current_report,
            delta_m_rad=delta,
            iterations=iterations,
            converged=current_report.scene_pass,
        )

    def _candidates(
        self,
        scene: SceneInput,
        report: SFURSceneReport,
        contract: FunctionalSceneContract,
        rules: SFURRules,
        config: FunctionalRefinementConfig,
    ) -> Iterable[SceneInput]:
        objects = {item.object_id: item for item in scene.furniture}
        pair_rules = {item.rule_id: item for item in rules.pair_rules}
        zones = {item.zone_id: item for item in contract.use_zones}
        for violation in report.violations:
            if violation.kind == "functional_relation" and violation.rule_id:
                subject = objects.get(violation.subject_id)
                target = objects.get(violation.target_id or "")
                rule = pair_rules.get(violation.rule_id)
                if subject is not None and target is not None and rule is not None:
                    yield from _relation_candidates(scene, subject, target, rule)
            elif violation.kind == "use_clearance":
                zone = zones.get(violation.subject_id)
                if zone is not None:
                    yield from _clearance_candidates(scene, zone, config)
            elif violation.kind == "reachability":
                yield from _path_candidates(
                    scene, contract, violation.subject_id, config
                )

    @staticmethod
    def contract(config: FunctionalRefinementConfig | None = None) -> dict[str, object]:
        active = config or FunctionalRefinementConfig()
        return {
            "schema_version": FUNCTIONAL_REFINEMENT_SCHEMA,
            "max_iterations": active.max_iterations,
            "hard_clearance_m": active.hard_clearance_m,
            "escape_padding_m": active.escape_padding_m,
            "corridor_padding_m": active.corridor_padding_m,
            "reads_clean_pose": False,
            "reads_corruption_identity": False,
        }


def _relation_candidates(
    scene: SceneInput,
    subject: FurnitureInput,
    target: FurnitureInput,
    rule: FunctionalPairRule,
) -> Iterable[SceneInput]:
    current_angle = math.atan2(subject.y_m - target.y_m, subject.x_m - target.x_m)
    angles = [current_angle, *(step * math.pi / 4.0 for step in range(8))]
    seen: set[tuple[int, int, int]] = set()
    desired_gap = (rule.minimum_gap_m + rule.maximum_gap_m) * 0.5
    for angle in angles:
        cosine, sine = math.cos(angle), math.sin(angle)
        yaw = subject.yaw_rad
        if subject.category in rule.facing_tolerance_deg:
            yaw = wrap_yaw(math.atan2(target.x_m - subject.x_m, -(target.y_m - subject.y_m)))
            # Facing depends on the candidate center, not the old center.
            yaw = wrap_yaw(math.atan2(-cosine, sine))
        low, high = 0.0, math.hypot(scene.room.length_m, scene.room.width_m)
        for _ in range(32):
            radius = (low + high) * 0.5
            candidate_item = replace(
                subject,
                x_m=target.x_m + cosine * radius,
                y_m=target.y_m + sine * radius,
                yaw_rad=yaw,
            )
            gap, _, _ = signed_separation_and_escape(
                _rectangle(target), _rectangle(candidate_item)
            )
            if gap < desired_gap:
                low = radius
            else:
                high = radius
        candidate_item = replace(
            subject,
            x_m=target.x_m + cosine * high,
            y_m=target.y_m + sine * high,
            yaw_rad=yaw,
        )
        signature = (
            round(candidate_item.x_m * 1000),
            round(candidate_item.y_m * 1000),
            round(candidate_item.yaw_rad * 1000),
        )
        if signature in seen:
            continue
        seen.add(signature)
        yield _replace_item(scene, subject.object_id, candidate_item)


def _clearance_candidates(
    scene: SceneInput,
    zone,
    config: FunctionalRefinementConfig,
) -> Iterable[SceneInput]:
    objects = {item.object_id: item for item in scene.furniture}
    owner = objects.get(zone.owner_id)
    if owner is None:
        return
    zone_rect = _zone_rectangle(owner, zone)
    margins = room_signed_margins(
        zone_rect, scene.room.length_m, scene.room.width_m
    )
    dx = (-margins[0] if margins[0] < 0.0 else 0.0) + (
        margins[1] if margins[1] < 0.0 else 0.0
    )
    dy = (-margins[2] if margins[2] < 0.0 else 0.0) + (
        margins[3] if margins[3] < 0.0 else 0.0
    )
    if math.hypot(dx, dy) > 1.0e-9:
        yield _replace_item(
            scene,
            owner.object_id,
            replace(owner, x_m=owner.x_m + dx, y_m=owner.y_m + dy),
        )
    for other in scene.furniture:
        if other.object_id == owner.object_id or other.object_id in zone.allowed_overlap_object_ids:
            continue
        separation, escape_x, escape_y = signed_separation_and_escape(
            zone_rect, _rectangle(other)
        )
        if separation >= 0.0:
            continue
        escape_x, escape_y = _pad_escape(
            escape_x, escape_y, config.escape_padding_m
        )
        yield _replace_item(
            scene,
            other.object_id,
            replace(other, x_m=other.x_m + escape_x, y_m=other.y_m + escape_y),
        )
        yield _replace_item(
            scene,
            owner.object_id,
            replace(owner, x_m=owner.x_m - escape_x, y_m=owner.y_m - escape_y),
        )
    for opening in scene.openings:
        if opening.kind != "door" or not opening.constraint_valid:
            continue
        separation, escape_x, escape_y = signed_separation_and_escape(
            _opening_rectangle(opening), zone_rect
        )
        if separation < 0.0:
            escape_x, escape_y = _pad_escape(
                escape_x, escape_y, config.escape_padding_m
            )
            yield _replace_item(
                scene,
                owner.object_id,
                replace(owner, x_m=owner.x_m + escape_x, y_m=owner.y_m + escape_y),
            )


def _path_candidates(
    scene: SceneInput,
    contract: FunctionalSceneContract,
    target_id: str,
    config: FunctionalRefinementConfig,
) -> Iterable[SceneInput]:
    owner = next((item for item in scene.furniture if item.object_id == target_id), None)
    target = next((item for item in contract.path_targets if item.target_id == target_id), None)
    door = next(
        (item for item in scene.openings if item.kind == "door" and item.constraint_valid),
        None,
    )
    if owner is None or target is None or door is None:
        return
    offset = _local_offset_to_world(
        target.local_goal_xy_m[0], target.local_goal_xy_m[1], owner.yaw_rad
    )
    goal = owner.x_m + offset[0], owner.y_m + offset[1]
    start = _door_path_start(door)
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 1.0e-6:
        return
    corridor = OrientedRectangle(
        (start[0] + goal[0]) * 0.5,
        (start[1] + goal[1]) * 0.5,
        length,
        config.corridor_padding_m,
        math.atan2(dy, dx),
    )
    for blocker in scene.furniture:
        if blocker.object_id == owner.object_id:
            continue
        separation, escape_x, escape_y = signed_separation_and_escape(
            corridor, _rectangle(blocker)
        )
        if separation >= 0.0:
            continue
        escape_x, escape_y = _pad_escape(
            escape_x, escape_y, config.escape_padding_m
        )
        yield _replace_item(
            scene,
            blocker.object_id,
            replace(
                blocker,
                x_m=blocker.x_m + escape_x,
                y_m=blocker.y_m + escape_y,
            ),
        )


def _replace_item(scene: SceneInput, object_id: str, value: FurnitureInput) -> SceneInput:
    return replace(
        scene,
        furniture=tuple(
            value if item.object_id == object_id else item for item in scene.furniture
        ),
    )


def _pad_escape(dx: float, dy: float, padding: float) -> tuple[float, float]:
    magnitude = math.hypot(dx, dy)
    if magnitude <= 1.0e-9:
        return padding, 0.0
    scale = (magnitude + padding) / magnitude
    return dx * scale, dy * scale


def _report_rank(report: SFURSceneReport) -> tuple[float, ...]:
    return (
        float(report.scene_pass),
        report.reachability_rate,
        report.functional_score,
        report.clearance_pass_rate,
        report.relation_pass_rate,
        -float(len(report.violations)),
    )


def _within_action_range(
    original: SceneInput, candidate: SceneInput, action_range: ActionRange
) -> bool:
    return all(
        math.hypot(right.x_m - left.x_m, right.y_m - left.y_m)
        <= action_range.max_translation_m + 1.0e-6
        for left, right in zip(original.furniture, candidate.furniture)
    )


def _pose_delta(original: SceneInput, candidate: SceneInput) -> torch.Tensor:
    return torch.tensor(
        [
            [
                right.x_m - left.x_m,
                right.y_m - left.y_m,
                wrap_yaw(right.yaw_rad - left.yaw_rad),
            ]
            for left, right in zip(original.furniture, candidate.furniture)
        ],
        dtype=torch.float32,
    )
