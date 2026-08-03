from __future__ import annotations

import math

import pytest
import torch

from scene_repair_v3 import (
    ActionRange,
    ActionType,
    FurnitureGraphBuilder,
    FurnitureLoss,
    FurnitureRepairNetwork,
    FurnitureTargets,
    ModelConfig,
    RoomInput,
    SceneInput,
    action_types_from_delta,
)
from scene_repair_v3.checkpoint import load_checkpoint, save_checkpoint


def _model(vocabularies) -> FurnitureRepairNetwork:
    return FurnitureRepairNetwork(
        vocabularies,
        ModelConfig(
            hidden_dim=32,
            heads=4,
            edge_dim=16,
            geometry_dim=24,
            category_dim=16,
            behavior_dim=24,
            dropout=0.0,
        ),
    )


def test_action_type_labels_use_explicit_epsilons() -> None:
    delta = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 0.0, 0.3],
            [0.2, 0.0, 0.3],
        ]
    )
    labels = action_types_from_delta(
        delta, translation_epsilon_m=0.01, yaw_epsilon_rad=0.01
    )
    assert labels.tolist() == [
        int(ActionType.KEEP),
        int(ActionType.TRANSLATE),
        int(ActionType.ROTATE),
        int(ActionType.BOTH),
    ]


def test_loss_runs_backward_through_typed_graph(vocabularies, sample_scene) -> None:
    torch.manual_seed(13)
    graph = FurnitureGraphBuilder(vocabularies).build(sample_scene)
    model = _model(vocabularies)
    output = model(graph)
    targets = FurnitureTargets(
        action_types=torch.tensor([int(ActionType.BOTH), int(ActionType.KEEP)]),
        delta_m_rad=torch.tensor([[0.1, -0.05, 0.2], [0.0, 0.0, 0.0]]),
        pose_valid=torch.tensor([True, True]),
        eligibility=graph.furniture_eligibility,
    )
    losses = FurnitureLoss(ActionRange(0.5, math.pi / 3))(output, targets)
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert model.action_type_head[-1].weight.grad is not None
    assert model.pose_delta_head[-2].weight.grad is not None
    functional_grads = [
        parameter.grad
        for parameter in model.edge_encoder.functional_encoder.parameters()
        if parameter.grad is not None
    ]
    assert functional_grads
    assert any(torch.count_nonzero(gradient) for gradient in functional_grads)


def test_checkpoint_rejects_action_contract_mismatch(
    tmp_path, vocabularies
) -> None:
    model = _model(vocabularies)
    path = tmp_path / "model.pt"
    action_range = ActionRange(0.5, math.pi / 3)
    save_checkpoint(path, model, action_range, extra={"epoch": 2})
    assert load_checkpoint(path, model, action_range) == {"epoch": 2}
    with pytest.raises(ValueError, match="action range"):
        load_checkpoint(path, model, ActionRange(0.4, math.pi / 4))


def test_empty_furniture_scene_has_well_defined_zero_loss(vocabularies) -> None:
    graph = FurnitureGraphBuilder(vocabularies).build(
        SceneInput(RoomInput("bedroom", 5.0, 4.0), ())
    )
    model = _model(vocabularies)
    output = model(graph)
    targets = FurnitureTargets(
        action_types=torch.empty(0, dtype=torch.long),
        delta_m_rad=torch.empty((0, 3)),
        pose_valid=torch.empty(0, dtype=torch.bool),
        eligibility=torch.empty(0, dtype=torch.bool),
    )
    losses = FurnitureLoss(ActionRange(0.5, math.pi / 3))(output, targets)
    assert losses.total.item() == 0.0
    losses.total.backward()
