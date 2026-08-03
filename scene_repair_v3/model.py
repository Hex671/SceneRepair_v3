from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Sequence

import torch
from torch import nn

from .contracts import (
    COORDINATE_FRAME,
    GRAPH_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION,
    QUATERNION_ORDER,
    YAW_RANGE,
    ActionRange,
    ActionType,
    EdgeType,
    FurnitureGraphBatch,
    ModelConfig,
    ModelOutput,
    NodeType,
    TypedEdgeSet,
)
from .vocabulary import FurnitureVocabularies
from .geometry import wrap_yaw
from .graph import FurnitureGraphBuilder, SceneInput, collate_graphs
from .functional_refinement import (
    FunctionalConstraintRefinementLayer,
    FunctionalRefinementConfig,
)
from .sfur import FunctionalSceneContract, SFURRules, SFURSceneReport
from .repair import (
    ConstraintRefinementConfig,
    HardConstraintReport,
    JointRepairResult,
    apply_pose_deltas,
    hard_constraint_report,
    project_predicted_repair,
)


def _encoder_mlp(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.GELU(),
        nn.Linear(output_dim, output_dim),
        nn.LayerNorm(output_dim),
    )


class TypeSpecificNodeEncoder(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        vocabularies: FurnitureVocabularies,
    ) -> None:
        super().__init__()
        self.config = config
        self.room_type_embedding = nn.Embedding(len(vocabularies.room_types), 32)
        self.room_encoder = _encoder_mlp(32 + 2, config.hidden_dim)

        self.opening_kind_embedding = nn.Embedding(len(vocabularies.opening_kinds), 16)
        self.opening_encoder = _encoder_mlp(16 + 8 + 8, config.hidden_dim)

        self.furniture_geometry_encoder = _encoder_mlp(
            7 + 7, config.geometry_dim
        )
        self.category_embedding = nn.Embedding(
            len(vocabularies.categories), config.category_dim
        )
        self.category_norm = nn.LayerNorm(config.category_dim)
        self.family_embedding = nn.Embedding(len(vocabularies.families), 32)
        self.behavior_encoder = _encoder_mlp(
            32 + len(vocabularies.functions) + 2,
            config.behavior_dim,
        )
        fusion_input = (
            config.geometry_dim
            + config.category_dim
            + config.behavior_dim
            + 7
            + 2
        )
        self.furniture_fusion = _encoder_mlp(fusion_input, config.hidden_dim)

    def forward(self, batch: FurnitureGraphBatch) -> torch.Tensor:
        device = batch.node_types.device
        dtype = self.room_encoder[0].weight.dtype
        states = torch.zeros(
            (batch.num_nodes, self.config.hidden_dim), device=device, dtype=dtype
        )

        room_features = batch.room_features.to(dtype=dtype)
        room_state = self.room_encoder(
            torch.cat(
                [self.room_type_embedding(batch.room_type_ids), room_features], dim=-1
            )
        )
        states.index_copy_(0, batch.room_indices, room_state)

        opening_mask = batch.opening_validity.to(dtype=dtype)
        opening_geometry = batch.opening_geometry.to(dtype=dtype) * opening_mask
        opening_state = self.opening_encoder(
            torch.cat(
                [
                    self.opening_kind_embedding(batch.opening_kind_ids),
                    opening_geometry,
                    opening_mask,
                ],
                dim=-1,
            )
        )
        states.index_copy_(0, batch.opening_indices, opening_state)

        geometry_mask = batch.furniture_geometry_validity.to(dtype=dtype)
        geometry = batch.furniture_geometry.to(dtype=dtype) * geometry_mask
        geometry_state = self.furniture_geometry_encoder(
            torch.cat([geometry, geometry_mask], dim=-1)
        )
        category_state = self.category_norm(
            self.category_embedding(batch.furniture_category_ids)
        )
        behavior_validity = batch.furniture_behavior_validity.to(dtype=dtype)
        family_state = self.family_embedding(batch.furniture_family_ids)
        family_state = family_state * behavior_validity[:, 0:1]
        functions = (
            batch.furniture_functions.to(dtype=dtype) * behavior_validity[:, 1:2]
        )
        behavior_state = self.behavior_encoder(
            torch.cat([family_state, functions, behavior_validity], dim=-1)
        )
        furniture_state = self.furniture_fusion(
            torch.cat(
                [
                    geometry_state,
                    category_state,
                    behavior_state,
                    geometry_mask,
                    behavior_validity,
                ],
                dim=-1,
            )
        )
        states.index_copy_(0, batch.furniture_indices, furniture_state)
        return states


class TypedEdgeEncoder(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        vocabularies: FurnitureVocabularies,
    ) -> None:
        super().__init__()
        self.edge_dim = config.edge_dim
        self.anchor_direction_embedding = nn.Embedding(
            len(vocabularies.anchor_directions), 8
        )
        self.anchor_mode_embedding = nn.Embedding(len(vocabularies.anchor_modes), 8)
        self.orientation_mode_embedding = nn.Embedding(
            len(vocabularies.orientation_modes), 8
        )
        self.target_strategy_embedding = nn.Embedding(
            len(vocabularies.target_strategies), 8
        )
        self.membership_encoder = _encoder_mlp(6 + 8 + 8, config.edge_dim)
        self.spatial_encoder = _encoder_mlp(8, config.edge_dim)
        self.functional_encoder = _encoder_mlp(4 + 8 + 8, config.edge_dim)
        self.opening_encoder = _encoder_mlp(4, config.edge_dim)
        role_count = len(NodeType) * len(EdgeType) * len(NodeType)
        self.direction_role_embedding = nn.Embedding(role_count, config.edge_dim)

    def forward(
        self,
        edge_set: TypedEdgeSet,
        node_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        active = edge_set.active
        edge_index = edge_set.edge_index[:, active]
        continuous = edge_set.continuous[active].to(
            dtype=self.direction_role_embedding.weight.dtype
        )
        categorical = edge_set.categorical[active]
        if edge_set.edge_type == EdgeType.ROOM_MEMBERSHIP:
            continuous = continuous.clone()
            continuous[:, :4] = continuous[:, :4] * continuous[:, 4:5]
            anchor_valid = continuous[:, 5:6]
            direction = self.anchor_direction_embedding(categorical[:, 0]) * anchor_valid
            mode = self.anchor_mode_embedding(categorical[:, 1]) * anchor_valid
            encoded = self.membership_encoder(
                torch.cat([continuous, direction, mode], dim=-1)
            )
        elif edge_set.edge_type == EdgeType.FURNITURE_SPATIAL:
            encoded = self.spatial_encoder(continuous)
        elif edge_set.edge_type == EdgeType.FUNCTIONAL_PARTNER:
            relation_valid = continuous[:, 3:4]
            orientation = (
                self.orientation_mode_embedding(categorical[:, 0]) * relation_valid
            )
            strategy = (
                self.target_strategy_embedding(categorical[:, 1]) * relation_valid
            )
            encoded = self.functional_encoder(
                torch.cat([continuous, orientation, strategy], dim=-1)
            )
        elif edge_set.edge_type == EdgeType.OBJECT_OPENING:
            encoded = self.opening_encoder(continuous)
        else:
            raise ValueError(f"unsupported edge type {edge_set.edge_type}")

        if edge_index.shape[1] == 0:
            return edge_index, encoded
        source_type = node_types[edge_index[0]]
        target_type = node_types[edge_index[1]]
        role_id = (
            source_type * (len(EdgeType) * len(NodeType))
            + int(edge_set.edge_type) * len(NodeType)
            + target_type
        )
        return edge_index, encoded + self.direction_role_embedding(role_id)


def incoming_edge_softmax(
    scores: torch.Tensor,
    target: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Softmax over edges with the same target, independently per head."""

    if scores.ndim != 2:
        raise ValueError("scores must be [E,heads]")
    if scores.shape[0] == 0:
        return torch.zeros_like(scores)
    edges, heads = scores.shape
    target_index = target.reshape(edges, 1).expand(edges, heads)
    maxima = scores.new_full((num_nodes, heads), -torch.inf)
    maxima.scatter_reduce_(0, target_index, scores, reduce="amax", include_self=True)
    unnormalized = torch.exp(scores - maxima[target])
    denominator = scores.new_zeros((num_nodes, heads))
    denominator.scatter_add_(0, target_index, unnormalized)
    return unnormalized / denominator[target].clamp_min(
        torch.finfo(scores.dtype).tiny
    )


class RelationAwareGraphTransformerLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.heads = config.heads
        self.head_dim = config.hidden_dim // config.heads
        self.query = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.key = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.value = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.edge_key = nn.Linear(config.edge_dim, config.hidden_dim)
        self.edge_value = nn.Linear(config.edge_dim, config.hidden_dim)
        self.relation_bias = nn.Parameter(torch.zeros(len(EdgeType), config.heads))
        self.relation_gates = nn.ModuleList(
            [nn.Linear(config.hidden_dim * 2, 1) for _ in EdgeType]
        )
        self.output_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
        )
        self.norm2 = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        node_states: torch.Tensor,
        edge_states: dict[EdgeType, tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nodes = node_states.shape[0]
        query = self.query(node_states).view(nodes, self.heads, self.head_dim)
        key = self.key(node_states).view(nodes, self.heads, self.head_dim)
        value = self.value(node_states).view(nodes, self.heads, self.head_dim)

        relation_messages: list[torch.Tensor] = []
        relation_present: list[torch.Tensor] = []
        gate_logits: list[torch.Tensor] = []
        for edge_type in EdgeType:
            edge_index, encoded_edge = edge_states[edge_type]
            source, target = edge_index
            aggregate = node_states.new_zeros((nodes, self.hidden_dim))
            present = torch.zeros(nodes, dtype=torch.bool, device=node_states.device)
            if source.numel():
                edge_key = self.edge_key(encoded_edge).view(
                    -1, self.heads, self.head_dim
                )
                edge_value = self.edge_value(encoded_edge).view(
                    -1, self.heads, self.head_dim
                )
                scores = (
                    query[target] * (key[source] + edge_key)
                ).sum(dim=-1) / math.sqrt(self.head_dim)
                scores = scores + self.relation_bias[int(edge_type)]
                weights = incoming_edge_softmax(scores, target, nodes)
                messages = (
                    weights.unsqueeze(-1) * (value[source] + edge_value)
                ).reshape(-1, self.hidden_dim)
                aggregate.index_add_(0, target, messages)
                present[target] = True
            relation_messages.append(aggregate)
            relation_present.append(present)
            gate_logits.append(
                self.relation_gates[int(edge_type)](
                    torch.cat([node_states, aggregate], dim=-1)
                ).squeeze(-1)
            )

        message_stack = torch.stack(relation_messages, dim=1)
        present_mask = torch.stack(relation_present, dim=1)
        logits = torch.stack(gate_logits, dim=1)
        masked_logits = logits.masked_fill(~present_mask, -torch.inf)
        any_present = present_mask.any(dim=1, keepdim=True)
        safe_logits = torch.where(any_present, masked_logits, torch.zeros_like(logits))
        gate_weights = torch.softmax(safe_logits, dim=1)
        gate_weights = gate_weights * present_mask.to(gate_weights.dtype)
        gate_weights = torch.where(
            any_present,
            gate_weights
            / gate_weights.sum(dim=1, keepdim=True).clamp_min(
                torch.finfo(gate_weights.dtype).tiny
            ),
            torch.zeros_like(gate_weights),
        )
        fused = (message_stack * gate_weights.unsqueeze(-1)).sum(dim=1)
        attended = self.output_projection(fused)
        node_states = self.norm1(node_states + self.dropout(attended))
        node_states = self.norm2(
            node_states + self.dropout(self.feed_forward(node_states))
        )
        return node_states, gate_weights


class HardConstraintRefinementLayer(nn.Module):
    """Parameter-free feasibility layer owned by the deployed network."""

    def __init__(self, config: ConstraintRefinementConfig | None = None) -> None:
        super().__init__()
        self.config = config or ConstraintRefinementConfig()

    def forward(
        self,
        scene: SceneInput,
        proposal_delta_m_rad: torch.Tensor,
        *,
        config: ConstraintRefinementConfig | None = None,
        action_range: ActionRange | None = None,
    ) -> JointRepairResult:
        active_config = config or self.config
        best: JointRepairResult | None = None
        best_score: tuple[int, float, float] | None = None
        scales = (1.0,) if action_range is None else (1.0, 0.75, 0.5, 0.25, 0.0)
        for scale in scales:
            result = project_predicted_repair(
                scene,
                proposal_delta_m_rad * scale,
                clearance_m=active_config.clearance_m,
                max_iterations=active_config.max_iterations,
            )
            translation = torch.linalg.vector_norm(
                result.delta_m_rad[:, :2], dim=-1
            )
            yaw = result.delta_m_rad[:, 2].abs()
            range_excess = 0.0
            if action_range is not None and translation.numel():
                range_excess = max(
                    float((translation - action_range.max_translation_m).clamp_min(0).max()),
                    float((yaw - action_range.max_yaw_rad).clamp_min(0).max()),
                )
            travel = float(translation.sum()) if translation.numel() else 0.0
            score = (result.report.total, range_excess, travel)
            candidate = replace(result, proposal_scale=scale)
            if best_score is None or score < best_score:
                best, best_score = candidate, score
            if result.report.total == 0 and range_excess <= 1.0e-6:
                return candidate
        assert best is not None
        return best

    def contract(
        self, config: ConstraintRefinementConfig | None = None
    ) -> dict[str, object]:
        return (config or self.config).to_dict()


class FurnitureRepairNetwork(nn.Module):
    """Constraint-aware typed graph encoder with per-Furniture action heads."""

    def __init__(
        self,
        vocabularies: FurnitureVocabularies,
        config: ModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.vocabularies = vocabularies
        self.config = config or ModelConfig()
        self.node_encoder = TypeSpecificNodeEncoder(self.config, vocabularies)
        self.edge_encoder = TypedEdgeEncoder(self.config, vocabularies)
        self.layers = nn.ModuleList(
            [
                RelationAwareGraphTransformerLayer(self.config)
                for _ in range(self.config.layers)
            ]
        )
        self.constraint_encoder = _encoder_mlp(12, self.config.constraint_dim)
        action_input_dim = self.config.hidden_dim + self.config.constraint_dim
        self.action_type_head = nn.Sequential(
            nn.Linear(action_input_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, len(ActionType)),
        )
        self.pose_delta_head = nn.Sequential(
            nn.Linear(action_input_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 3),
            nn.Tanh(),
        )
        self.constraint_refinement = HardConstraintRefinementLayer()
        self.functional_refinement = FunctionalConstraintRefinementLayer()

    def forward(
        self,
        batch: FurnitureGraphBatch,
        *,
        validate: bool = True,
    ) -> ModelOutput:
        if validate:
            batch.validate(max_furniture=self.config.max_furniture)
        states = self.node_encoder(batch)
        edge_states = {
            edge_type: self.edge_encoder(batch.edges[edge_type], batch.node_types)
            for edge_type in EdgeType
        }
        gate_weights: list[torch.Tensor] = []
        for layer in self.layers:
            states, layer_gates = layer(states, edge_states)
            gate_weights.append(layer_gates)
        furniture_states = states[batch.furniture_indices]
        constraint_state = self.constraint_encoder(
            _furniture_constraint_summary(batch).to(furniture_states.dtype)
        )
        furniture_output_state = torch.cat(
            [furniture_states, constraint_state], dim=-1
        )
        return ModelOutput(
            action_type_logits=self.action_type_head(furniture_output_state),
            primary_delta_normalized=self.pose_delta_head(furniture_output_state),
            furniture_indices=batch.furniture_indices,
            furniture_graph_index=batch.furniture_graph_index,
            furniture_eligibility=batch.furniture_eligibility,
            node_states=states,
            relation_gate_weights=tuple(gate_weights),
        )

    @torch.no_grad()
    def repair_batch(
        self,
        batch: FurnitureGraphBatch,
        scenes: Sequence[SceneInput],
        action_range: ActionRange,
        *,
        refinement: ConstraintRefinementConfig | None = None,
        functional_contracts: Sequence[FunctionalSceneContract] | None = None,
        functional_origin_scenes: Sequence[SceneInput] | None = None,
        sfur_rules: SFURRules | None = None,
        functional_refinement: FunctionalRefinementConfig | None = None,
        action_policy: str = "hard_constraint_gated",
        semantic_passes: int = 1,
        validate: bool = True,
    ) -> ConstraintAwareModelOutput:
        """Run the complete deployed model, including its frozen constraint layer.

        ``forward`` remains the measurable neural-backbone output. Deployment uses
        this method so hard-constraint refinement is a versioned part of the model
        contract rather than an optional caller-side post-processing step.
        """

        if semantic_passes <= 0:
            raise ValueError("semantic_passes must be positive")
        origins = tuple(scenes)
        if semantic_passes == 1:
            output = self(batch, validate=validate)
            result = self.refine_output(
                batch,
                scenes,
                output,
                action_range,
                refinement=refinement,
                action_policy=action_policy,
            )
        else:
            current_scenes = origins
            current_batch = batch
            result = None
            for pass_index in range(semantic_passes):
                result = self.repair_batch(
                    current_batch,
                    current_scenes,
                    action_range,
                    refinement=refinement,
                    action_policy=action_policy,
                    semantic_passes=1,
                    validate=validate,
                )
                current_scenes = result.scenes
                if pass_index + 1 < semantic_passes:
                    builder = FurnitureGraphBuilder(
                        self.vocabularies, max_furniture=self.config.max_furniture
                    )
                    current_batch = collate_graphs(
                        builder.build(scene) for scene in current_scenes
                    ).to(batch.node_types.device)
            assert result is not None
            active_refinement = refinement or ConstraintRefinementConfig()
            rebased = tuple(
                self.constraint_refinement(
                    origin,
                    _scene_pose_delta(origin, current),
                    config=active_refinement,
                    action_range=action_range,
                )
                for origin, current in zip(origins, current_scenes)
            )
            current_scenes = tuple(value.scene for value in rebased)
            combined_delta = torch.cat(
                [value.delta_m_rad for value in rebased], dim=0
            ).to(
                device=result.neural_output.action_type_logits.device,
                dtype=result.neural_output.primary_delta_normalized.dtype,
            )
            translation = (
                torch.linalg.vector_norm(combined_delta[:, :2], dim=-1) > 1.0e-6
            )
            rotation = combined_delta[:, 2].abs() > 1.0e-6
            result = replace(
                result,
                final_action_types=(
                    translation.to(torch.long) + 2 * rotation.to(torch.long)
                ),
                final_delta_m_rad=combined_delta,
                scenes=current_scenes,
                reports=tuple(value.report for value in rebased),
                semantic_passes=semantic_passes,
            )
        if functional_contracts is None:
            return result
        if sfur_rules is None:
            raise ValueError("sfur_rules are required with functional_contracts")
        if len(functional_contracts) != len(scenes):
            raise ValueError("functional_contracts must match scenes")
        functional_origins = tuple(functional_origin_scenes or origins)
        if len(functional_origins) != len(scenes):
            raise ValueError("functional_origin_scenes must match scenes")
        functional_results = tuple(
            self.functional_refinement(
                original,
                proposed,
                contract,
                sfur_rules,
                action_range,
                config=functional_refinement,
            )
            for original, proposed, contract in zip(
                functional_origins, result.scenes, functional_contracts
            )
        )
        final_delta = torch.cat(
            [value.delta_m_rad for value in functional_results], dim=0
        ).to(
            device=result.neural_output.action_type_logits.device,
            dtype=result.neural_output.primary_delta_normalized.dtype,
        )
        translation = torch.linalg.vector_norm(final_delta[:, :2], dim=-1) > 1.0e-6
        rotation = final_delta[:, 2].abs() > 1.0e-6
        final_actions = translation.to(torch.long) + 2 * rotation.to(torch.long)
        return replace(
            result,
            final_action_types=final_actions,
            final_delta_m_rad=final_delta,
            scenes=tuple(value.scene for value in functional_results),
            reports=tuple(
                hard_constraint_report(value.scene) for value in functional_results
            ),
            functional_reports=tuple(value.report for value in functional_results),
            functional_refinement_iterations=tuple(
                value.iterations for value in functional_results
            ),
            functional_refinement_converged=tuple(
                value.converged for value in functional_results
            ),
            functional_refinement_contract=self.functional_refinement.contract(
                functional_refinement
            ),
        )

    @torch.no_grad()
    def refine_output(
        self,
        batch: FurnitureGraphBatch,
        scenes: Sequence[SceneInput],
        output: ModelOutput,
        action_range: ActionRange,
        *,
        refinement: ConstraintRefinementConfig | None = None,
        action_policy: str = "hard_constraint_gated",
    ) -> ConstraintAwareModelOutput:
        """Apply the model-owned constraint layer to a computed backbone output."""

        if len(scenes) != batch.num_graphs:
            raise ValueError("scenes must contain one SceneInput per graph")
        config = refinement or ConstraintRefinementConfig()
        if action_policy == "hard_constraint_gated":
            decoded = decode_actions(output, action_range, batch=batch)
        elif action_policy == "semantic":
            decoded = decode_actions(output, action_range)
        elif action_policy == "dense_delta":
            decoded = decode_dense_deltas(output, action_range)
        else:
            raise ValueError(
                "action_policy must be hard_constraint_gated, semantic or dense_delta"
            )
        neural_scenes: list[SceneInput] = []
        refined_scenes: list[SceneInput] = []
        neural_reports: list[HardConstraintReport] = []
        refined_reports: list[HardConstraintReport] = []
        refined_deltas: list[torch.Tensor] = []
        refinement_iterations: list[int] = []
        refinement_converged: list[bool] = []
        refinement_proposal_scales: list[float] = []
        for graph_id, scene in enumerate(scenes):
            mask = output.furniture_graph_index == graph_id
            proposal = decoded.delta_m_rad[mask].detach().cpu()
            if proposal.shape != (len(scene.furniture), 3):
                raise ValueError("Furniture output order does not match SceneInput")
            neural_scene = apply_pose_deltas(scene, proposal)
            result = self.constraint_refinement(
                scene,
                proposal,
                config=config,
                action_range=action_range,
            )
            neural_scenes.append(neural_scene)
            refined_scenes.append(result.scene)
            neural_reports.append(hard_constraint_report(neural_scene))
            refined_reports.append(result.report)
            refined_deltas.append(result.delta_m_rad)
            refinement_iterations.append(result.iterations)
            refinement_converged.append(result.converged)
            refinement_proposal_scales.append(result.proposal_scale)

        final_delta = torch.cat(refined_deltas, dim=0).to(
            device=output.action_type_logits.device,
            dtype=output.primary_delta_normalized.dtype,
        )
        translation = torch.linalg.vector_norm(final_delta[:, :2], dim=-1) > 1.0e-6
        rotation = final_delta[:, 2].abs() > 1.0e-6
        final_actions = translation.to(torch.long) + 2 * rotation.to(torch.long)
        final_actions[~output.furniture_eligibility] = int(ActionType.KEEP)
        final_delta[~output.furniture_eligibility] = 0.0
        return ConstraintAwareModelOutput(
            neural_output=output,
            neural_action_types=decoded.action_types,
            neural_delta_m_rad=decoded.delta_m_rad,
            final_action_types=final_actions,
            final_delta_m_rad=final_delta,
            neural_scenes=tuple(neural_scenes),
            scenes=tuple(refined_scenes),
            neural_reports=tuple(neural_reports),
            reports=tuple(refined_reports),
            refinement_iterations=tuple(refinement_iterations),
            refinement_converged=tuple(refinement_converged),
            refinement_proposal_scales=tuple(refinement_proposal_scales),
            refinement_contract=self.constraint_refinement.contract(config),
        )

    def checkpoint_contract(self) -> dict[str, object]:
        return {
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "graph_schema_version": GRAPH_SCHEMA_VERSION,
            "coordinate_frame": COORDINATE_FRAME,
            "quaternion_order": QUATERNION_ORDER,
            "yaw_range": YAW_RANGE,
            "yaw_positive_axis": "+Z counter-clockwise viewed from above",
            "vocabulary_schema_version": self.vocabularies.schema_version,
            "vocabulary_sha256": self.vocabularies.sha256,
            "config": asdict(self.config),
        }

    def inference_contract(
        self, refinement: ConstraintRefinementConfig | None = None
    ) -> dict[str, object]:
        return {
            "schema_version": "furniture_constraint_aware_model_v1",
            "backbone_contract": self.checkpoint_contract(),
            "refinement": self.constraint_refinement.contract(refinement),
            "output": "per-furniture action_type and delta_x/delta_y/delta_yaw",
        }


def _furniture_constraint_summary(batch: FurnitureGraphBatch) -> torch.Tensor:
    """Aggregate hard-constraint evidence already present on typed edges."""
    furniture_count = int(batch.furniture_indices.numel())
    dtype = batch.furniture_geometry.dtype
    device = batch.node_types.device
    summary = torch.zeros((furniture_count, 12), dtype=dtype, device=device)
    node_to_furniture = torch.full(
        (batch.num_nodes,), -1, dtype=torch.long, device=device
    )
    node_to_furniture[batch.furniture_indices] = torch.arange(
        furniture_count, device=device
    )

    membership = batch.edges[EdgeType.ROOM_MEMBERSHIP]
    membership_mask = membership.active & (
        node_to_furniture[membership.edge_index[1]] >= 0
    )
    membership_target = node_to_furniture[
        membership.edge_index[1, membership_mask]
    ]
    if membership_target.numel():
        summary[membership_target, :4] = membership.continuous[membership_mask, :4]

    spatial = batch.edges[EdgeType.FURNITURE_SPATIAL]
    spatial_mask = spatial.active & (node_to_furniture[spatial.edge_index[1]] >= 0)
    spatial_target = node_to_furniture[spatial.edge_index[1, spatial_mask]]
    if spatial_target.numel():
        values = spatial.continuous[spatial_mask]
        depth = torch.relu(-values[:, 4])
        summary[:, 4].scatter_reduce_(
            0, spatial_target, depth, reduce="amax", include_self=True
        )
        summary[:, 5].index_add_(0, spatial_target, values[:, 5] * (depth > 0))
        summary[:, 6].index_add_(0, spatial_target, values[:, 6] * (depth > 0))
        summary[:, 7].index_add_(
            0, spatial_target, (depth > 0).to(dtype) / 17.0
        )

    opening = batch.edges[EdgeType.OBJECT_OPENING]
    opening_mask = opening.active & (node_to_furniture[opening.edge_index[1]] >= 0)
    opening_target = node_to_furniture[opening.edge_index[1, opening_mask]]
    if opening_target.numel():
        values = opening.continuous[opening_mask]
        depth = torch.relu(-values[:, 0])
        summary[:, 8].scatter_reduce_(
            0, opening_target, depth, reduce="amax", include_self=True
        )
        summary[:, 9].index_add_(0, opening_target, values[:, 1] * (depth > 0))
        summary[:, 10].index_add_(0, opening_target, values[:, 2] * (depth > 0))
        summary[:, 11].index_add_(
            0, opening_target, (depth > 0).to(dtype) / 4.0
        )
    return summary


def furniture_violation_mask(
    batch: FurnitureGraphBatch, *, normalized_epsilon: float = 1.0e-6
) -> torch.Tensor:
    summary = _furniture_constraint_summary(batch)
    return (
        (summary[:, :4].min(dim=-1).values < -normalized_epsilon)
        | (summary[:, 4] > normalized_epsilon)
        | (summary[:, 8] > normalized_epsilon)
    )


def constraint_gated_action_types(
    output: ModelOutput,
    batch: FurnitureGraphBatch,
) -> torch.Tensor:
    """Use exact hard-constraint presence to gate the learned action subtype."""
    violation = furniture_violation_mask(batch)
    result = output.action_type_logits.argmax(dim=-1)
    result = result.clone()
    result[~violation] = int(ActionType.KEEP)
    repair_required = violation & output.furniture_eligibility
    if bool(repair_required.any()):
        repair_logits = output.action_type_logits[repair_required, 1:]
        result[repair_required] = repair_logits.argmax(dim=-1) + 1
    result[~output.furniture_eligibility] = int(ActionType.KEEP)
    return result


@dataclass
class DecodedActions:
    action_types: torch.Tensor
    delta_m_rad: torch.Tensor


def _scene_pose_delta(source: SceneInput, target: SceneInput) -> torch.Tensor:
    if len(source.furniture) != len(target.furniture):
        raise ValueError("scene furniture counts must match")
    return torch.tensor(
        [
            [
                right.x_m - left.x_m,
                right.y_m - left.y_m,
                wrap_yaw(right.yaw_rad - left.yaw_rad),
            ]
            for left, right in zip(source.furniture, target.furniture)
        ],
        dtype=torch.float32,
    )


def decode_dense_deltas(
    output: ModelOutput,
    action_range: ActionRange,
    *,
    translation_epsilon_m: float = 0.02,
    yaw_epsilon_rad: float = math.radians(1.0),
) -> DecodedActions:
    """Decode the densely supervised pose field without a class-logit gate."""

    if translation_epsilon_m < 0.0 or yaw_epsilon_rad < 0.0:
        raise ValueError("dense delta thresholds must be non-negative")
    normalized = output.primary_delta_normalized
    delta = torch.stack(
        (
            normalized[:, 0] * action_range.max_translation_m,
            normalized[:, 1] * action_range.max_translation_m,
            normalized[:, 2] * action_range.max_yaw_rad,
        ),
        dim=-1,
    )
    translation = (
        torch.linalg.vector_norm(delta[:, :2], dim=-1) > translation_epsilon_m
    )
    rotation = delta[:, 2].abs() > yaw_epsilon_rad
    actions = translation.to(torch.long) + 2 * rotation.to(torch.long)
    delta = delta.clone()
    delta[~translation, :2] = 0.0
    delta[~rotation, 2] = 0.0
    actions[~output.furniture_eligibility] = int(ActionType.KEEP)
    delta[~output.furniture_eligibility] = 0.0
    return DecodedActions(actions, delta)


@dataclass
class ConstraintAwareModelOutput:
    neural_output: ModelOutput
    neural_action_types: torch.Tensor
    neural_delta_m_rad: torch.Tensor
    final_action_types: torch.Tensor
    final_delta_m_rad: torch.Tensor
    neural_scenes: tuple[SceneInput, ...]
    scenes: tuple[SceneInput, ...]
    neural_reports: tuple[HardConstraintReport, ...]
    reports: tuple[HardConstraintReport, ...]
    refinement_iterations: tuple[int, ...]
    refinement_converged: tuple[bool, ...]
    refinement_proposal_scales: tuple[float, ...]
    refinement_contract: dict[str, object]
    functional_reports: tuple[SFURSceneReport, ...] = ()
    functional_refinement_iterations: tuple[int, ...] = ()
    functional_refinement_converged: tuple[bool, ...] = ()
    functional_refinement_contract: dict[str, object] | None = None
    semantic_passes: int = 1


def decode_actions(
    output: ModelOutput,
    action_range: ActionRange,
    *,
    batch: FurnitureGraphBatch | None = None,
    action_types: torch.Tensor | None = None,
) -> DecodedActions:
    """Apply deterministic eligibility and action-component gates."""

    if action_types is None:
        action_types = (
            constraint_gated_action_types(output, batch)
            if batch is not None
            else output.action_type_logits.argmax(dim=-1)
        )
    action_types = action_types.to(
        device=output.action_type_logits.device, dtype=torch.long
    ).clone()
    if action_types.shape != (output.action_type_logits.shape[0],):
        raise ValueError("action_types must be [F]")
    action_types[~output.furniture_eligibility] = int(ActionType.KEEP)

    normalized = output.primary_delta_normalized
    delta = torch.stack(
        [
            normalized[:, 0] * action_range.max_translation_m,
            normalized[:, 1] * action_range.max_translation_m,
            normalized[:, 2] * action_range.max_yaw_rad,
        ],
        dim=-1,
    )
    translation_enabled = (action_types == int(ActionType.TRANSLATE)) | (
        action_types == int(ActionType.BOTH)
    )
    rotation_enabled = (action_types == int(ActionType.ROTATE)) | (
        action_types == int(ActionType.BOTH)
    )
    delta[:, :2] = delta[:, :2] * translation_enabled.unsqueeze(-1)
    delta[:, 2] = delta[:, 2] * rotation_enabled
    return DecodedActions(action_types=action_types, delta_m_rad=delta)
