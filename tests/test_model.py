from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from scene_repair_v3 import (
    ActionRange,
    ActionType,
    EdgeType,
    FurnitureGraphBuilder,
    FurnitureInput,
    FurnitureRepairNetwork,
    FunctionalConstraintRefinementLayer,
    HardConstraintRefinementLayer,
    ModelConfig,
    SceneInput,
    apply_pose_deltas,
    constraint_gated_action_types,
    decode_actions,
    decode_dense_deltas,
    hard_constraint_report,
)
from scene_repair_v3.model import incoming_edge_softmax


def _small_config() -> ModelConfig:
    return ModelConfig(
        hidden_dim=32,
        heads=4,
        edge_dim=16,
        geometry_dim=24,
        category_dim=16,
        behavior_dim=24,
        dropout=0.0,
    )


def test_relation_softmax_normalizes_per_target() -> None:
    scores = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    target = torch.tensor([1, 1, 2, 2])
    weights = incoming_edge_softmax(scores, target, num_nodes=3)
    assert weights[target == 1].sum().item() == pytest.approx(1.0)
    assert weights[target == 2].sum().item() == pytest.approx(1.0)


def test_forward_has_only_per_furniture_action_outputs(
    vocabularies, sample_scene
) -> None:
    torch.manual_seed(7)
    graph = FurnitureGraphBuilder(vocabularies).build(sample_scene)
    model = FurnitureRepairNetwork(vocabularies, _small_config()).eval()
    assert isinstance(model.constraint_refinement, HardConstraintRefinementLayer)
    assert isinstance(model.functional_refinement, FunctionalConstraintRefinementLayer)
    output = model(graph)

    assert output.action_type_logits.shape == (2, 4)
    assert output.primary_delta_normalized.shape == (2, 3)
    assert output.node_states.shape == (4, 32)
    assert len(output.relation_gate_weights) == 2
    assert torch.isfinite(output.action_type_logits).all()
    assert bool((output.primary_delta_normalized.abs() <= 1.0).all())
    for gates in output.relation_gate_weights:
        assert gates.shape == (4, 4)
        assert torch.allclose(gates.sum(dim=1), torch.ones(4), atol=1.0e-6)
    assert not hasattr(model, "candidate_queries")
    assert not hasattr(model, "route_head")
    assert not hasattr(model, "violation_head")


def test_default_architecture_is_128_by_4_by_2(vocabularies, sample_scene) -> None:
    graph = FurnitureGraphBuilder(vocabularies).build(sample_scene)
    model = FurnitureRepairNetwork(vocabularies).eval()
    output = model(graph)
    assert model.config.hidden_dim == 128
    assert model.config.heads == 4
    assert len(model.layers) == 2
    assert model.layers[0].head_dim == 32
    assert output.node_states.shape == (4, 128)


def test_missing_relation_is_excluded_from_fusion(vocabularies, sample_scene) -> None:
    scene = replace(sample_scene, functional_partners=())
    graph = FurnitureGraphBuilder(vocabularies).build(scene)
    model = FurnitureRepairNetwork(vocabularies, _small_config()).eval()
    output = model(graph)
    for gates in output.relation_gate_weights:
        assert torch.count_nonzero(gates[:, int(EdgeType.FUNCTIONAL_PARTNER)]) == 0


def test_room_context_reaches_furniture_in_two_layers(vocabularies, sample_scene) -> None:
    torch.manual_seed(11)
    graph = FurnitureGraphBuilder(vocabularies).build(sample_scene)
    altered = FurnitureGraphBuilder(vocabularies).build(
        replace(sample_scene, room=replace(sample_scene.room, length_m=7.0))
    )
    model = FurnitureRepairNetwork(vocabularies, _small_config()).eval()
    with torch.no_grad():
        first = model(graph).action_type_logits
        second = model(altered).action_type_logits
    assert not torch.allclose(first, second)


def test_decode_forces_ineligible_furniture_to_keep(vocabularies, sample_scene) -> None:
    furniture = list(sample_scene.furniture)
    furniture[1] = replace(furniture[1], movable=False)
    scene = replace(sample_scene, furniture=tuple(furniture))
    graph = FurnitureGraphBuilder(vocabularies).build(scene)
    model = FurnitureRepairNetwork(vocabularies, _small_config()).eval()
    output = model(graph)
    selected = torch.tensor([int(ActionType.BOTH), int(ActionType.BOTH)])
    decoded = decode_actions(
        output,
        ActionRange(0.5, math.pi / 3),
        action_types=selected,
    )
    assert decoded.action_types.tolist() == [int(ActionType.BOTH), int(ActionType.KEEP)]
    assert torch.equal(decoded.delta_m_rad[1], torch.zeros(3))


def test_dense_delta_decode_does_not_depend_on_action_logits(
    vocabularies, sample_scene
) -> None:
    graph = FurnitureGraphBuilder(vocabularies).build(sample_scene)
    model = FurnitureRepairNetwork(vocabularies, _small_config()).eval()
    output = model(graph)
    output.action_type_logits = torch.tensor(
        [[20.0, 0.0, 0.0, 0.0], [20.0, 0.0, 0.0, 0.0]]
    )
    output.primary_delta_normalized = torch.tensor(
        [[0.1, 0.0, 0.1], [0.0, 0.0, 0.0]]
    )
    decoded = decode_dense_deltas(output, ActionRange(3.0, math.pi))
    assert decoded.action_types.tolist() == [int(ActionType.BOTH), int(ActionType.KEEP)]
    assert decoded.delta_m_rad[0].tolist() == pytest.approx(
        [0.3, 0.0, 0.1 * math.pi]
    )


def test_constraint_gate_forces_repair_only_for_visible_violation(
    vocabularies, sample_scene
) -> None:
    furniture = list(sample_scene.furniture)
    furniture[0] = replace(furniture[0], x_m=3.0)
    graph = FurnitureGraphBuilder(vocabularies).build(
        replace(sample_scene, furniture=tuple(furniture))
    )
    model = FurnitureRepairNetwork(vocabularies, _small_config()).eval()
    output = model(graph)
    output.action_type_logits = torch.tensor(
        [[10.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 10.0]]
    )
    gated = constraint_gated_action_types(output, graph)
    assert gated.tolist() == [int(ActionType.BOTH), int(ActionType.KEEP)]


def test_complete_model_contract_refines_neural_residuals(
    vocabularies, sample_scene
) -> None:
    furniture = list(sample_scene.furniture)
    furniture[0] = replace(furniture[0], x_m=3.0)
    corrupted = replace(sample_scene, furniture=tuple(furniture))
    graph = FurnitureGraphBuilder(vocabularies).build(corrupted)
    model = FurnitureRepairNetwork(vocabularies, _small_config()).eval()

    result = model.repair_batch(graph, [corrupted], ActionRange(3.0, math.pi))

    assert result.final_delta_m_rad.shape == (2, 3)
    assert result.reports[0].total == 0
    assert hard_constraint_report(result.scenes[0]).total == 0
    reconstructed = apply_pose_deltas(corrupted, result.final_delta_m_rad.cpu())
    for actual, expected in zip(reconstructed.furniture, result.scenes[0].furniture):
        assert actual.x_m == pytest.approx(expected.x_m)
        assert actual.y_m == pytest.approx(expected.y_m)
        assert actual.yaw_rad == pytest.approx(expected.yaw_rad)
    assert result.refinement_contract["schema_version"] == (
        "furniture_constraint_refinement_v1"
    )
    assert model.inference_contract()["schema_version"] == (
        "furniture_constraint_aware_model_v1"
    )


def test_complete_model_owns_iterative_semantic_repair(
    vocabularies, sample_scene
) -> None:
    graph = FurnitureGraphBuilder(vocabularies).build(sample_scene)
    model = FurnitureRepairNetwork(vocabularies, _small_config()).eval()
    result = model.repair_batch(
        graph,
        [sample_scene],
        ActionRange(3.0, math.pi),
        action_policy="dense_delta",
        semantic_passes=2,
    )
    assert result.semantic_passes == 2
    assert result.reports[0].total == 0
    assert float(
        torch.linalg.vector_norm(result.final_delta_m_rad[:, :2], dim=-1).max()
    ) <= 3.0 + 1.0e-6
