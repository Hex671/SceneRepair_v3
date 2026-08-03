"""Scene JSON adaptation and reproducible online corruption for Furniture v1."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import torch

from .contracts import ActionType
from .functional_partners import FrozenFunctionalPartnerRules
from .graph import (
    FurnitureGraphBuilder,
    FurnitureInput,
    OpeningInput,
    RoomInput,
    SceneInput,
    collate_graphs,
)
from .geometry import OrientedRectangle, room_signed_margins, signed_separation_and_escape
from .losses import FurnitureTargets, action_types_from_delta
from .repair import apply_pose_deltas, hard_constraint_report, joint_translation_repair
from .vocabulary import FurnitureVocabularies


@dataclass(frozen=True)
class CorruptionConfig:
    label_mode: str = "semantic_v2"
    max_translation_m: float = 2.0
    translation_noise_bound_m: float = 0.2
    translation_noise_std_m: float = 0.067
    yaw_noise_bound_rad: float = 0.05 * math.pi
    yaw_noise_std_rad: float = 0.05 * math.pi / 3.0
    translation_epsilon_m: float = 0.02
    yaw_epsilon_rad: float = math.radians(1.0)
    joint_repair_clearance_m: float = 0.25

    def __post_init__(self) -> None:
        if self.label_mode not in {
            "observable_joint_v4",
            "observable_v3",
            "semantic_restore_v1",
            "semantic_v2",
            "legacy_random",
        }:
            raise ValueError(
                "label_mode must be observable_joint_v4, observable_v3, "
                "semantic_restore_v1, semantic_v2 or legacy_random"
            )
        if self.joint_repair_clearance_m < 0.0:
            raise ValueError("joint_repair_clearance_m must be non-negative")


@dataclass(frozen=True)
class CorruptedSceneSample:
    graph: Any
    targets: FurnitureTargets
    scene_id: str
    seed: int
    action_types: tuple[int, ...]
    changed_count: int
    sampled_action_types: tuple[int, ...]
    sampled_changed_count: int
    corrupted_scene: SceneInput
    target_converged: bool


def load_scene_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"scene JSON must be an object: {path}")
    return value


def scene_to_input(
    raw: dict[str, Any],
    *,
    rules: FrozenFunctionalPartnerRules | None = None,
    furniture_override: Sequence[dict[str, Any]] | None = None,
) -> SceneInput:
    if raw.get("stage") != "furniture":
        raise ValueError(f"unsupported scene stage: {raw.get('stage')!r}")
    furniture_rows = furniture_override if furniture_override is not None else raw.get("furniture", ())
    furniture: list[FurnitureInput] = []
    for row in furniture_rows:
        validity = row.get("validity") or {}
        bbox = row.get("bbox") or {}
        transform_valid = bool(validity.get("transform", True))
        bbox_valid = bool(validity.get("bbox", True))
        geometry_validity = (transform_valid,) * 4 + (bbox_valid,) * 3
        family = row.get("family")
        functions = tuple(str(value) for value in (row.get("functions") or ()))
        furniture.append(
            FurnitureInput(
                object_id=str(row["object_id"]),
                x_m=float(row["x"]),
                y_m=float(row["y"]),
                yaw_rad=float(row.get("yaw_rad", 0.0)),
                bbox_width_m=float(bbox["width"]),
                bbox_depth_m=float(bbox["depth"]),
                bbox_height_m=float(bbox["height"]),
                category=str(row.get("category") or "UNK"),
                hssd_id=row.get("hssd_id"),
                family=family,
                functions=functions,
                geometry_validity=geometry_validity,
                family_valid=bool(validity.get("family", family is not None)),
                functions_valid=bool(validity.get("functions", bool(functions))),
                movable=bool(row.get("movable", True)),
                transform_valid=transform_valid,
            )
        )

    openings: list[OpeningInput] = []
    for row in raw.get("openings") or ():
        validity = row.get("validity") or {}
        center_valid = bool(validity.get("clearance_center_xyz_m", True))
        size_valid = bool(validity.get("clearance_size_xyz_m", True))
        normal_valid = bool(validity.get("interior_normal_xy", True))
        opening_validity = (center_valid,) * 3 + (size_valid,) * 3 + (normal_valid,) * 2
        openings.append(
            OpeningInput(
                opening_id=str(row["opening_id"]),
                kind=str(row.get("kind") or "UNK"),
                clearance_center_xyz_m=tuple(float(v) for v in row["clearance_center_xyz_m"]),
                clearance_size_xyz_m=tuple(float(v) for v in row["clearance_size_xyz_m"]),
                interior_normal_xy=tuple(float(v) for v in row["interior_normal_xy"]),
                geometry_validity=opening_validity,
                constraint_valid=all(opening_validity),
            )
        )

    scene = SceneInput(
        room=RoomInput(str(raw["room_type"]), float(raw["length"]), float(raw["width"])),
        furniture=tuple(furniture),
        openings=tuple(openings),
        coordinate_frame=str(raw.get("coordinate_frame", "room_local_z_up")),
        quaternion_order=str(raw.get("quaternion_order", "wxyz")),
        yaw_range=str(raw.get("yaw_range", "[-pi,pi)")),
    )
    if rules is not None:
        scene = rules.apply(scene)
    return scene


def _truncated_gaussian(rng: random.Random, std: float, bound: float) -> float:
    for _ in range(64):
        value = rng.gauss(0.0, std)
        if abs(value) <= bound:
            return value
    return max(-bound, min(bound, rng.gauss(0.0, std)))


def _sample_action(rng: random.Random) -> ActionType:
    move = bool(rng.getrandbits(1))
    rotate = bool(rng.getrandbits(1))
    if move and rotate:
        return ActionType.BOTH
    if move:
        return ActionType.TRANSLATE
    if rotate:
        return ActionType.ROTATE
    return ActionType.KEEP


def _sample_yaw_limit(rng: random.Random) -> float:
    value = rng.random()
    if value < 0.40:
        return 0.10 * math.pi
    if value < 0.70:
        return 0.25 * math.pi
    if value < 0.90:
        return 0.50 * math.pi
    return math.pi


def _wrap_yaw(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _object_has_violation(scene: SceneInput, index: int) -> bool:
    """Return whether one Furniture pose violates a visible hard constraint."""
    rectangles = {
        item.object_id: OrientedRectangle(
            item.x_m, item.y_m, item.bbox_width_m, item.bbox_depth_m, item.yaw_rad
        )
        for item in scene.furniture
    }
    item = scene.furniture[index]
    rectangle = rectangles[item.object_id]
    if min(room_signed_margins(rectangle, scene.room.length_m, scene.room.width_m)) < 0.0:
        return True
    for other_index, other in enumerate(scene.furniture):
        if other_index == index:
            continue
        separation, _, _ = signed_separation_and_escape(
            rectangle, rectangles[other.object_id]
        )
        if separation < 0.0:
            return True

    for opening in scene.openings:
        cx, cy, cz = opening.clearance_center_xyz_m
        sx, sy, sz = opening.clearance_size_xyz_m
        opening_rect = OrientedRectangle(cx, cy, sx, sy, 0.0)
        opening_low_z, opening_high_z = cz - sz * 0.5, cz + sz * 0.5
        for candidate_index, candidate in enumerate(scene.furniture):
            if candidate_index != index:
                continue
            xy_separation, _, _ = signed_separation_and_escape(
                opening_rect, rectangle
            )
            z_separation = max(
                opening_low_z - candidate.bbox_height_m,
                0.0 - opening_high_z,
            )
            if max(xy_separation, z_separation) < 0.0:
                return True
    return False


def _replace_furniture(scene: SceneInput, index: int, value: FurnitureInput) -> SceneInput:
    furniture = list(scene.furniture)
    furniture[index] = value
    return replace(scene, furniture=tuple(furniture))


def _violation_energy_and_vectors(
    scene: SceneInput, index: int
) -> tuple[float, list[tuple[float, float]]]:
    item = scene.furniture[index]
    rectangle = OrientedRectangle(
        item.x_m, item.y_m, item.bbox_width_m, item.bbox_depth_m, item.yaw_rad
    )
    energy = 0.0
    vectors: list[tuple[float, float]] = []
    left, right, bottom, top = room_signed_margins(
        rectangle, scene.room.length_m, scene.room.width_m
    )
    room_dx = (-left if left < 0.0 else 0.0) + (right if right < 0.0 else 0.0)
    room_dy = (-bottom if bottom < 0.0 else 0.0) + (top if top < 0.0 else 0.0)
    room_energy = sum(max(-value, 0.0) for value in (left, right, bottom, top))
    if room_energy > 0.0:
        energy += room_energy
        vectors.append((room_dx, room_dy))

    for other_index, other in enumerate(scene.furniture):
        if other_index == index:
            continue
        other_rectangle = OrientedRectangle(
            other.x_m, other.y_m, other.bbox_width_m, other.bbox_depth_m, other.yaw_rad
        )
        separation, escape_x, escape_y = signed_separation_and_escape(
            other_rectangle, rectangle
        )
        if separation < 0.0:
            energy += -separation
            vectors.append((escape_x * 1.01, escape_y * 1.01))

    for opening in scene.openings:
        cx, cy, cz = opening.clearance_center_xyz_m
        sx, sy, sz = opening.clearance_size_xyz_m
        opening_rectangle = OrientedRectangle(cx, cy, sx, sy, 0.0)
        xy_separation, escape_x, escape_y = signed_separation_and_escape(
            opening_rectangle, rectangle
        )
        z_separation = max(
            cz - sz * 0.5 - item.bbox_height_m,
            0.0 - (cz + sz * 0.5),
        )
        separation = max(xy_separation, z_separation)
        if separation < 0.0:
            energy += -separation
            vectors.append((escape_x * 1.01, escape_y * 1.01))
    return energy, vectors


def _translation_repair(
    scene: SceneInput, index: int, *, max_steps: int = 12
) -> tuple[float, float] | None:
    working = scene
    total_x = total_y = 0.0
    for _ in range(max_steps):
        energy, vectors = _violation_energy_and_vectors(working, index)
        if energy <= 1.0e-6:
            return total_x, total_y
        if not vectors:
            return None
        sum_vector = (sum(v[0] for v in vectors), sum(v[1] for v in vectors))
        candidates = [*vectors, sum_vector]
        best: tuple[float, float, float, float, SceneInput] | None = None
        item = working.furniture[index]
        for dx, dy in candidates:
            if math.hypot(dx, dy) <= 1.0e-9:
                continue
            moved = replace(item, x_m=item.x_m + dx, y_m=item.y_m + dy)
            candidate = _replace_furniture(working, index, moved)
            candidate_energy, _ = _violation_energy_and_vectors(candidate, index)
            score = (candidate_energy, math.hypot(dx, dy), dx, dy)
            if best is None or score[:2] < best[:2]:
                best = (score[0], score[1], dx, dy, candidate)
        if best is None:
            return (total_x, total_y) if math.hypot(total_x, total_y) > 1.0e-9 else None
        if best[0] >= energy - 1.0e-7:
            _, _, dx, dy, _ = best
            return total_x + dx, total_y + dy
        _, _, dx, dy, working = best
        total_x += dx
        total_y += dy
    energy, _ = _violation_energy_and_vectors(working, index)
    return (total_x, total_y) if math.hypot(total_x, total_y) > 1.0e-9 else None


def _rotation_repair(scene: SceneInput, index: int) -> float | None:
    item = scene.furniture[index]
    best: tuple[float, float] | None = None
    for step in range(24):
        candidate_yaw = -math.pi + step * math.pi / 12.0
        delta = _wrap_yaw(candidate_yaw - item.yaw_rad)
        if abs(delta) <= 1.0e-6:
            continue
        rotated = replace(item, yaw_rad=candidate_yaw)
        candidate = _replace_furniture(scene, index, rotated)
        energy, _ = _violation_energy_and_vectors(candidate, index)
        if energy <= 1.0e-6:
            score = (abs(delta), delta)
            if best is None or score < best:
                best = score
    return None if best is None else best[1]


def _observable_targets(
    corrupted: SceneInput, config: CorruptionConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    actions = torch.full(
        (len(corrupted.furniture),), int(ActionType.KEEP), dtype=torch.long
    )
    deltas = torch.zeros((len(corrupted.furniture), 3), dtype=torch.float32)
    for index in range(len(corrupted.furniture)):
        energy, _ = _violation_energy_and_vectors(corrupted, index)
        if energy <= 1.0e-6:
            continue
        translation = _translation_repair(corrupted, index)
        rotation = _rotation_repair(corrupted, index)
        translation_cost = (
            math.hypot(*translation)
            / (config.max_translation_m + config.translation_noise_bound_m)
            if translation is not None
            else math.inf
        )
        rotation_cost = abs(rotation) / math.pi if rotation is not None else math.inf
        if rotation_cost < translation_cost:
            actions[index] = int(ActionType.ROTATE)
            deltas[index, 2] = float(rotation)
        elif translation is not None:
            actions[index] = int(ActionType.TRANSLATE)
            deltas[index, 0] = float(translation[0])
            deltas[index, 1] = float(translation[1])
        elif rotation is not None:
            actions[index] = int(ActionType.ROTATE)
            deltas[index, 2] = float(rotation)
    return actions, deltas


def _semantic_targets(
    clean: SceneInput,
    corrupted: SceneInput,
    sampled_actions: Sequence[ActionType],
    full_delta: torch.Tensor,
    config: CorruptionConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_delta = torch.zeros_like(full_delta)
    target_action = torch.full(
        (len(corrupted.furniture),), int(ActionType.KEEP), dtype=torch.long
    )
    for index, sampled_action in enumerate(sampled_actions):
        if sampled_action == ActionType.KEEP or not _object_has_violation(corrupted, index):
            continue
        clean_item = clean.furniture[index]
        corrupt_item = corrupted.furniture[index]
        translation_pose = replace(
            corrupt_item, x_m=clean_item.x_m, y_m=clean_item.y_m
        )
        rotation_pose = replace(corrupt_item, yaw_rad=clean_item.yaw_rad)
        translation_fixes = not _object_has_violation(
            _replace_furniture(corrupted, index, translation_pose), index
        )
        rotation_fixes = not _object_has_violation(
            _replace_furniture(corrupted, index, rotation_pose), index
        )
        has_translation = (
            torch.linalg.vector_norm(full_delta[index, :2]).item()
            > config.translation_epsilon_m
        )
        has_rotation = abs(float(full_delta[index, 2])) > config.yaw_epsilon_rad
        if translation_fixes and has_translation and rotation_fixes and has_rotation:
            translation_cost = float(
                torch.linalg.vector_norm(full_delta[index, :2])
            ) / (config.max_translation_m + config.translation_noise_bound_m)
            rotation_cost = abs(float(full_delta[index, 2])) / math.pi
            translation_fixes = translation_cost <= rotation_cost
            rotation_fixes = not translation_fixes
        if translation_fixes and has_translation:
            target_action[index] = int(ActionType.TRANSLATE)
            target_delta[index, :2] = full_delta[index, :2]
        elif rotation_fixes and has_rotation:
            target_action[index] = int(ActionType.ROTATE)
            target_delta[index, 2] = full_delta[index, 2]
        elif has_translation and has_rotation:
            target_action[index] = int(ActionType.BOTH)
            target_delta[index] = full_delta[index]
        elif has_translation:
            target_action[index] = int(ActionType.TRANSLATE)
            target_delta[index, :2] = full_delta[index, :2]
        elif has_rotation:
            target_action[index] = int(ActionType.ROTATE)
            target_delta[index, 2] = full_delta[index, 2]
    return target_action, target_delta


def corrupt_scene(
    raw: dict[str, Any],
    *,
    seed: int,
    builder: FurnitureGraphBuilder,
    rules: FrozenFunctionalPartnerRules | None = None,
    config: CorruptionConfig | None = None,
) -> CorruptedSceneSample:
    config = config or CorruptionConfig()
    rng = random.Random(int(seed))
    clean_rows = list(raw.get("furniture") or ())
    corrupted_rows: list[dict[str, Any]] = []
    sampled_actions: list[ActionType] = []
    for row in clean_rows:
        changed = dict(row)
        action = _sample_action(rng)
        x, y, yaw = float(row["x"]), float(row["y"]), float(row.get("yaw_rad", 0.0))
        movable = bool(row.get("movable", True)) and bool((row.get("validity") or {}).get("transform", True))
        if not movable:
            action = ActionType.KEEP
        sampled_actions.append(action)
        if action in (ActionType.TRANSLATE, ActionType.BOTH):
            distance = rng.uniform(0.0, config.max_translation_m)
            direction = rng.uniform(-math.pi, math.pi)
            x += distance * math.cos(direction) + _truncated_gaussian(rng, config.translation_noise_std_m, config.translation_noise_bound_m)
            y += distance * math.sin(direction) + _truncated_gaussian(rng, config.translation_noise_std_m, config.translation_noise_bound_m)
        if action in (ActionType.ROTATE, ActionType.BOTH):
            yaw_limit = _sample_yaw_limit(rng)
            yaw += rng.uniform(-yaw_limit, yaw_limit)
            yaw += _truncated_gaussian(rng, config.yaw_noise_std_rad, config.yaw_noise_bound_rad)
        changed["x"], changed["y"], changed["yaw_rad"] = x, y, _wrap_yaw(yaw)
        corrupted_rows.append(changed)

    clean_scene = scene_to_input(raw, rules=rules)
    corrupted_scene = scene_to_input(raw, rules=rules, furniture_override=corrupted_rows)
    graph = builder.build(corrupted_scene)
    deltas: list[list[float]] = []
    for clean, corrupt in zip(clean_scene.furniture, corrupted_scene.furniture):
        deltas.append([
            clean.x_m - corrupt.x_m,
            clean.y_m - corrupt.y_m,
            _wrap_yaw(clean.yaw_rad - corrupt.yaw_rad),
        ])
    delta_tensor = torch.tensor(deltas, dtype=torch.float32)
    action_tensor = action_types_from_delta(
        delta_tensor,
        translation_epsilon_m=config.translation_epsilon_m,
        yaw_epsilon_rad=config.yaw_epsilon_rad,
    )
    target_converged = True
    if config.label_mode == "observable_joint_v4":
        repaired = joint_translation_repair(
            corrupted_scene,
            clearance_m=config.joint_repair_clearance_m,
        )
        delta_tensor = repaired.delta_m_rad
        action_tensor = action_types_from_delta(
            delta_tensor,
            # Joint projection may rely on a sub-centimetre move by one member
            # of a dense collision component. Dropping it would make the
            # decoded supervision geometrically inconsistent.
            translation_epsilon_m=1.0e-6,
            yaw_epsilon_rad=config.yaw_epsilon_rad,
        )
        delta_tensor[action_tensor == int(ActionType.KEEP)] = 0.0
        target_converged = (
            repaired.converged
            and hard_constraint_report(
                apply_pose_deltas(corrupted_scene, delta_tensor)
            ).total == 0
        )
    elif config.label_mode == "observable_v3":
        action_tensor, delta_tensor = _observable_targets(corrupted_scene, config)
    elif config.label_mode == "semantic_v2":
        action_tensor, delta_tensor = _semantic_targets(
            clean_scene,
            corrupted_scene,
            sampled_actions,
            delta_tensor,
            config,
        )
    elif config.label_mode == "semantic_restore_v1":
        # Every independently sampled pose corruption is supervised back to the
        # authored clean arrangement. This teaches relational layout recovery,
        # including non-colliding orientation and usability errors.
        pass
    eligibility = graph.furniture_eligibility.clone()
    targets = FurnitureTargets(
        action_types=action_tensor,
        delta_m_rad=delta_tensor,
        pose_valid=torch.ones(len(clean_scene.furniture), dtype=torch.bool),
        eligibility=eligibility,
    )
    return CorruptedSceneSample(
        graph=graph,
        targets=targets,
        scene_id=str(raw.get("scene_id", "unknown")),
        seed=int(seed),
        action_types=tuple(int(value) for value in action_tensor.tolist()),
        changed_count=int((action_tensor != int(ActionType.KEEP)).sum()),
        sampled_action_types=tuple(int(value) for value in sampled_actions),
        sampled_changed_count=sum(action != ActionType.KEEP for action in sampled_actions),
        corrupted_scene=corrupted_scene,
        target_converged=target_converged,
    )


def collate_corrupted_samples(samples: Sequence[CorruptedSceneSample]) -> tuple[Any, FurnitureTargets, dict[str, Any]]:
    if not samples:
        raise ValueError("cannot collate an empty sample batch")
    graph = collate_graphs([sample.graph for sample in samples])
    targets = FurnitureTargets(
        action_types=torch.cat([sample.targets.action_types for sample in samples]),
        delta_m_rad=torch.cat([sample.targets.delta_m_rad for sample in samples]),
        pose_valid=torch.cat([sample.targets.pose_valid for sample in samples]),
        eligibility=torch.cat([sample.targets.eligibility for sample in samples]),
    )
    metadata = {
        "scene_ids": [sample.scene_id for sample in samples],
        "seeds": [sample.seed for sample in samples],
        "changed_counts": [sample.changed_count for sample in samples],
        "sampled_changed_counts": [sample.sampled_changed_count for sample in samples],
        "sampled_action_types": [sample.sampled_action_types for sample in samples],
        "corrupted_scenes": [sample.corrupted_scene for sample in samples],
        "target_converged": [sample.target_converged for sample in samples],
    }
    return graph, targets, metadata
