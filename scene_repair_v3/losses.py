from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn

from .contracts import ActionRange, ActionType, EdgeType, ModelOutput


@dataclass(frozen=True)
class FurnitureLossConfig:
    action_weight: float = 1.0
    component_weight: float = 0.5
    translation_weight: float = 1.0
    yaw_weight: float = 1.0
    keep_zero_weight: float = 0.02
    dense_pose_weight: float = 0.0
    relation_pose_weight: float = 0.0
    geometry_weight: float = 0.5
    geometry_clearance_m: float = 0.12
    smooth_l1_beta_m: float = 0.05

    def __post_init__(self) -> None:
        values = (
            self.action_weight,
            self.component_weight,
            self.translation_weight,
            self.yaw_weight,
            self.keep_zero_weight,
            self.dense_pose_weight,
            self.relation_pose_weight,
            self.geometry_weight,
            self.geometry_clearance_m,
            self.smooth_l1_beta_m,
        )
        if any(value < 0 for value in values):
            raise ValueError("loss weights and beta must be non-negative")
        if self.smooth_l1_beta_m == 0:
            raise ValueError("smooth_l1_beta_m must be positive")


@dataclass
class FurnitureTargets:
    action_types: torch.Tensor
    delta_m_rad: torch.Tensor
    pose_valid: torch.Tensor
    eligibility: torch.Tensor

    def validate(self, furniture_count: int) -> None:
        if self.action_types.shape != (furniture_count,) or self.action_types.dtype != torch.long:
            raise ValueError("target action_types must be long [F]")
        if self.delta_m_rad.shape != (furniture_count, 3):
            raise ValueError("target delta_m_rad must be [F,3]")
        if self.pose_valid.shape != (furniture_count,) or self.pose_valid.dtype != torch.bool:
            raise ValueError("target pose_valid must be bool [F]")
        if self.eligibility.shape != (furniture_count,) or self.eligibility.dtype != torch.bool:
            raise ValueError("target eligibility must be bool [F]")
        if not torch.isfinite(self.delta_m_rad).all():
            raise ValueError("target deltas must be finite")
        if furniture_count and (
            int(self.action_types.min()) < 0
            or int(self.action_types.max()) >= len(ActionType)
        ):
            raise ValueError("target action type is outside the four-class contract")

    def to(self, device: torch.device | str) -> "FurnitureTargets":
        return FurnitureTargets(
            self.action_types.to(device),
            self.delta_m_rad.to(device),
            self.pose_valid.to(device),
            self.eligibility.to(device),
        )


@dataclass
class FurnitureLossOutput:
    total: torch.Tensor
    action: torch.Tensor
    component: torch.Tensor
    translation: torch.Tensor
    yaw: torch.Tensor
    keep_zero: torch.Tensor
    dense_pose: torch.Tensor
    relation_pose: torch.Tensor
    geometry: torch.Tensor

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            "loss": self.total,
            "loss_action": self.action,
            "loss_component": self.component,
            "loss_translation": self.translation,
            "loss_yaw": self.yaw,
            "loss_keep_zero": self.keep_zero,
            "loss_dense_pose": self.dense_pose,
            "loss_relation_pose": self.relation_pose,
            "loss_geometry": self.geometry,
        }


class FurnitureLoss(nn.Module):
    def __init__(
        self,
        action_range: ActionRange,
        config: FurnitureLossConfig | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.action_range = action_range
        self.config = config or FurnitureLossConfig()
        if class_weights is None:
            class_weights = torch.ones(len(ActionType), dtype=torch.float32)
        if class_weights.shape != (len(ActionType),):
            raise ValueError("class_weights must contain four values")
        self.register_buffer("class_weights", class_weights.float())

    def forward(
        self,
        output: ModelOutput,
        targets: FurnitureTargets,
        graph=None,
    ) -> FurnitureLossOutput:
        furniture_count = output.action_type_logits.shape[0]
        targets.validate(furniture_count)
        if not torch.equal(
            output.furniture_eligibility.to(targets.eligibility.device),
            targets.eligibility,
        ):
            raise ValueError("model input and target eligibility masks disagree")

        if furniture_count == 0:
            zero = (
                output.action_type_logits.sum()
                + output.primary_delta_normalized.sum()
            ) * 0.0
            return FurnitureLossOutput(
                zero, zero, zero, zero, zero, zero, zero, zero, zero
            )

        eligible = targets.eligibility
        action_per_item = functional.cross_entropy(
            output.action_type_logits,
            targets.action_types,
            weight=self.class_weights.to(output.action_type_logits.device),
            reduction="none",
        )
        action_loss = _masked_mean(action_per_item, eligible)
        logits = output.action_type_logits
        move_logits = torch.stack(
            [
                torch.logsumexp(logits[:, [int(ActionType.KEEP), int(ActionType.ROTATE)]], dim=-1),
                torch.logsumexp(logits[:, [int(ActionType.TRANSLATE), int(ActionType.BOTH)]], dim=-1),
            ],
            dim=-1,
        )
        rotate_logits = torch.stack(
            [
                torch.logsumexp(logits[:, [int(ActionType.KEEP), int(ActionType.TRANSLATE)]], dim=-1),
                torch.logsumexp(logits[:, [int(ActionType.ROTATE), int(ActionType.BOTH)]], dim=-1),
            ],
            dim=-1,
        )
        move_target = (
            (targets.action_types == int(ActionType.TRANSLATE))
            | (targets.action_types == int(ActionType.BOTH))
        ).long()
        rotate_target = (
            (targets.action_types == int(ActionType.ROTATE))
            | (targets.action_types == int(ActionType.BOTH))
        ).long()
        component_per_item = 0.5 * (
            functional.cross_entropy(move_logits, move_target, reduction="none")
            + functional.cross_entropy(rotate_logits, rotate_target, reduction="none")
        )
        component_loss = _masked_mean(component_per_item, eligible)

        normalized = output.primary_delta_normalized
        predicted_delta = torch.stack(
            [
                normalized[:, 0] * self.action_range.max_translation_m,
                normalized[:, 1] * self.action_range.max_translation_m,
                normalized[:, 2] * self.action_range.max_yaw_rad,
            ],
            dim=-1,
        )
        translation_action = (
            targets.action_types == int(ActionType.TRANSLATE)
        ) | (targets.action_types == int(ActionType.BOTH))
        rotation_action = (targets.action_types == int(ActionType.ROTATE)) | (
            targets.action_types == int(ActionType.BOTH)
        )
        pose_mask = targets.pose_valid & eligible
        translation_mask = pose_mask & translation_action
        rotation_mask = pose_mask & rotation_action

        translation_per_item = functional.smooth_l1_loss(
            predicted_delta[:, :2],
            targets.delta_m_rad[:, :2],
            beta=self.config.smooth_l1_beta_m,
            reduction="none",
        ).mean(dim=-1)
        translation_loss = _masked_mean(translation_per_item, translation_mask)

        yaw_error = predicted_delta[:, 2] - targets.delta_m_rad[:, 2]
        yaw_per_item = 1.0 - torch.cos(yaw_error)
        yaw_loss = _masked_mean(yaw_per_item, rotation_mask)

        keep_mask = eligible & (targets.action_types == int(ActionType.KEEP))
        keep_per_item = normalized.square().mean(dim=-1)
        keep_zero_loss = _masked_mean(keep_per_item, keep_mask)
        dense_translation = functional.smooth_l1_loss(
            predicted_delta[:, :2],
            targets.delta_m_rad[:, :2],
            beta=self.config.smooth_l1_beta_m,
            reduction="none",
        ).mean(dim=-1)
        dense_yaw = 1.0 - torch.cos(
            predicted_delta[:, 2] - targets.delta_m_rad[:, 2]
        )
        dense_pose_loss = _masked_mean(
            dense_translation + dense_yaw, pose_mask
        )
        relation_pose_loss = (
            _relation_pose_loss(predicted_delta, targets, graph)
            if graph is not None
            else normalized.sum() * 0.0
        )
        geometry_loss = (
            _geometry_feasibility_loss(
                predicted_delta,
                targets,
                graph,
                clearance_m=self.config.geometry_clearance_m,
            )
            if graph is not None
            else normalized.sum() * 0.0
        )

        total = (
            self.config.action_weight * action_loss
            + self.config.component_weight * component_loss
            + self.config.translation_weight * translation_loss
            + self.config.yaw_weight * yaw_loss
            + self.config.keep_zero_weight * keep_zero_loss
            + self.config.dense_pose_weight * dense_pose_loss
            + self.config.relation_pose_weight * relation_pose_loss
            + self.config.geometry_weight * geometry_loss
        )
        return FurnitureLossOutput(
            total=total,
            action=action_loss,
            component=component_loss,
            translation=translation_loss,
            yaw=yaw_loss,
            keep_zero=keep_zero_loss,
            dense_pose=dense_pose_loss,
            relation_pose=relation_pose_loss,
            geometry=geometry_loss,
        )


def action_types_from_delta(
    delta_m_rad: torch.Tensor,
    *,
    translation_epsilon_m: float,
    yaw_epsilon_rad: float,
) -> torch.Tensor:
    if delta_m_rad.ndim != 2 or delta_m_rad.shape[1] != 3:
        raise ValueError("delta_m_rad must be [F,3]")
    if translation_epsilon_m < 0 or yaw_epsilon_rad < 0:
        raise ValueError("action epsilons must be non-negative")
    translation = torch.linalg.vector_norm(delta_m_rad[:, :2], dim=-1) > translation_epsilon_m
    rotation = delta_m_rad[:, 2].abs() > yaw_epsilon_rad
    result = torch.full(
        (delta_m_rad.shape[0],),
        int(ActionType.KEEP),
        dtype=torch.long,
        device=delta_m_rad.device,
    )
    result[translation & ~rotation] = int(ActionType.TRANSLATE)
    result[~translation & rotation] = int(ActionType.ROTATE)
    result[translation & rotation] = int(ActionType.BOTH)
    return result


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _relation_pose_loss(
    predicted_delta: torch.Tensor, targets: FurnitureTargets, graph
) -> torch.Tensor:
    """Translation-invariant supervision of clean pair geometry and relative yaw."""

    geometry = graph.furniture_geometry
    graph_ids = graph.furniture_graph_index
    room = graph.room_features
    lengths, widths = room[graph_ids, 0], room[graph_ids, 1]
    current_xy = torch.stack(
        (geometry[:, 0] * lengths, geometry[:, 1] * widths), dim=-1
    )
    predicted_xy = current_xy + predicted_delta[:, :2]
    target_xy = current_xy + targets.delta_m_rad[:, :2]
    current_yaw = torch.atan2(geometry[:, 2], geometry[:, 3])
    predicted_yaw = current_yaw + predicted_delta[:, 2]
    target_yaw = current_yaw + targets.delta_m_rad[:, 2]

    spatial = graph.edges[EdgeType.FURNITURE_SPATIAL]
    node_to_furniture = torch.full(
        (graph.num_nodes,), -1, dtype=torch.long, device=geometry.device
    )
    node_to_furniture[graph.furniture_indices] = torch.arange(
        geometry.shape[0], device=geometry.device
    )
    source = node_to_furniture[spatial.edge_index[0]]
    target = node_to_furniture[spatial.edge_index[1]]
    mask = spatial.active & (source >= 0) & (target >= 0) & (source < target)
    left, right = source[mask], target[mask]
    if left.numel() == 0:
        return predicted_delta.sum() * 0.0
    pair_valid = (
        targets.pose_valid[left]
        & targets.pose_valid[right]
        & targets.eligibility[left]
        & targets.eligibility[right]
    )
    predicted_relative = predicted_xy[right] - predicted_xy[left]
    target_relative = target_xy[right] - target_xy[left]
    translation = functional.smooth_l1_loss(
        predicted_relative,
        target_relative,
        beta=0.05,
        reduction="none",
    ).mean(dim=-1)
    predicted_relative_yaw = predicted_yaw[right] - predicted_yaw[left]
    target_relative_yaw = target_yaw[right] - target_yaw[left]
    yaw = 1.0 - torch.cos(predicted_relative_yaw - target_relative_yaw)
    return _masked_mean(translation + 0.5 * yaw, pair_valid)


def _geometry_feasibility_loss(
    predicted_delta, targets, graph, *, clearance_m: float
) -> torch.Tensor:
    """Differentiable hard-geometry surrogate on target-action predicted poses."""
    geometry = graph.furniture_geometry
    graph_ids = graph.furniture_graph_index
    room = graph.room_features
    device, dtype = geometry.device, geometry.dtype
    move = ((targets.action_types == int(ActionType.TRANSLATE)) | (
        targets.action_types == int(ActionType.BOTH)
    )).to(dtype)
    lengths, widths = room[graph_ids, 0], room[graph_ids, 1]
    x = geometry[:, 0] * lengths + predicted_delta[:, 0] * move
    y = geometry[:, 1] * widths + predicted_delta[:, 1] * move
    sine, cosine = geometry[:, 2], geometry[:, 3]
    box_w, box_d = geometry[:, 4] * lengths, geometry[:, 5] * widths
    half_x = cosine.abs() * box_w * 0.5 + sine.abs() * box_d * 0.5
    half_y = sine.abs() * box_w * 0.5 + cosine.abs() * box_d * 0.5
    penalties = [
        torch.relu(clearance_m + half_x - lengths * 0.5 + x.abs()),
        torch.relu(clearance_m + half_y - widths * 0.5 + y.abs()),
    ]
    positions = torch.stack((x, y), dim=-1)
    spatial = graph.edges[EdgeType.FURNITURE_SPATIAL]
    node_to_furniture = torch.full(
        (graph.num_nodes,), -1, dtype=torch.long, device=device
    )
    node_to_furniture[graph.furniture_indices] = torch.arange(
        geometry.shape[0], device=device
    )
    # The relation is a complete directed graph. Keeping source < target gives
    # one vectorized row per unordered pair.
    source = node_to_furniture[spatial.edge_index[0]]
    target = node_to_furniture[spatial.edge_index[1]]
    pair_mask = spatial.active & (source >= 0) & (target >= 0) & (source < target)
    left, right = source[pair_mask], target[pair_mask]
    if left.numel():
        delta = positions[right] - positions[left]
        left_x = torch.stack((cosine[left], sine[left]), dim=-1)
        left_y = torch.stack((-sine[left], cosine[left]), dim=-1)
        right_x = torch.stack((cosine[right], sine[right]), dim=-1)
        right_y = torch.stack((-sine[right], cosine[right]), dim=-1)
        axes = torch.stack((left_x, left_y, right_x, right_y), dim=1)
        projection = torch.einsum("pd,pad->pa", delta, axes).abs()
        left_radius = (
            box_w[left, None] * 0.5 * torch.einsum("pd,pad->pa", left_x, axes).abs()
            + box_d[left, None] * 0.5 * torch.einsum("pd,pad->pa", left_y, axes).abs()
        )
        right_radius = (
            box_w[right, None] * 0.5 * torch.einsum("pd,pad->pa", right_x, axes).abs()
            + box_d[right, None] * 0.5 * torch.einsum("pd,pad->pa", right_y, axes).abs()
        )
        signed_separation = (projection - left_radius - right_radius).max(dim=-1).values
        penalties.append(torch.relu(clearance_m - signed_separation))

    opening_edges = graph.edges[EdgeType.OBJECT_OPENING]
    node_to_opening = torch.full(
        (graph.num_nodes,), -1, dtype=torch.long, device=device
    )
    node_to_opening[graph.opening_indices] = torch.arange(
        graph.opening_indices.numel(), device=device
    )
    opening_row = node_to_opening[opening_edges.edge_index[0]]
    furniture_row = node_to_furniture[opening_edges.edge_index[1]]
    opening_mask = (
        opening_edges.active & (opening_row >= 0) & (furniture_row >= 0)
    )
    opening_row = opening_row[opening_mask]
    furniture_row = furniture_row[opening_mask]
    if opening_row.numel():
        opening_graph = graph.graph_index[graph.opening_indices[opening_row]]
        opening_room = room[opening_graph]
        opening_values = graph.opening_geometry[opening_row]
        opening_center = torch.stack((
            opening_values[:, 0] * opening_room[:, 0],
            opening_values[:, 1] * opening_room[:, 1],
        ), dim=-1)
        opening_size = torch.stack((
            opening_values[:, 3] * opening_room[:, 0],
            opening_values[:, 4] * opening_room[:, 1],
        ), dim=-1)
        item_center = positions[furniture_row]
        item_x = torch.stack((
            cosine[furniture_row], sine[furniture_row]
        ), dim=-1)
        item_y = torch.stack((
            -sine[furniture_row], cosine[furniture_row]
        ), dim=-1)
        room_x = torch.tensor([1.0, 0.0], dtype=dtype, device=device).expand_as(item_x)
        room_y = torch.tensor([0.0, 1.0], dtype=dtype, device=device).expand_as(item_y)
        axes = torch.stack((room_x, room_y, item_x, item_y), dim=1)
        center_delta = item_center - opening_center
        projection = torch.einsum("pd,pad->pa", center_delta, axes).abs()
        opening_radius = (
            opening_size[:, 0, None] * 0.5 * axes[:, :, 0].abs()
            + opening_size[:, 1, None] * 0.5 * axes[:, :, 1].abs()
        )
        item_radius = (
            box_w[furniture_row, None] * 0.5
            * torch.einsum("pd,pad->pa", item_x, axes).abs()
            + box_d[furniture_row, None] * 0.5
            * torch.einsum("pd,pad->pa", item_y, axes).abs()
        )
        xy_separation = (
            projection - opening_radius - item_radius
        ).max(dim=-1).values
        room_scale = opening_room.max(dim=-1).values
        opening_z = opening_values[:, 2] * room_scale
        opening_height = opening_values[:, 5] * room_scale
        item_height = geometry[furniture_row, 6] * room_scale
        z_separation = torch.maximum(
            opening_z - opening_height * 0.5 - item_height,
            -(opening_z + opening_height * 0.5),
        )
        penalties.append(torch.relu(
            clearance_m - torch.maximum(xy_separation, z_separation)
        ))
    if not penalties:
        return predicted_delta.sum() * 0.0
    return torch.cat([value.reshape(-1) for value in penalties]).mean()
