from __future__ import annotations

import json
from dataclasses import replace

import torch

from scene_repair_v3 import FurnitureGraphBuilder
from scene_repair_v3 import ActionType
from scene_repair_v3 import (
    apply_pose_deltas,
    hard_constraint_report,
    joint_translation_repair,
    project_predicted_repair,
)
from scene_repair_v3.data import (
    CorruptionConfig,
    _observable_targets,
    _semantic_targets,
    corrupt_scene,
    scene_to_input,
)
from scene_repair_v3.dataset import FurnitureCorruptionDataset, freeze_scene_manifest


def _raw_scene() -> dict:
    furniture = []
    for index, x in enumerate((-0.8, 0.8)):
        furniture.append({
            "object_id": f"chair_{index}", "hssd_id": f"asset_{index}",
            "category": "chair", "family": "seating", "functions": ["sit"],
            "bbox": {"width": 0.5, "depth": 0.5, "height": 0.9},
            "x": x, "y": 0.0, "yaw_rad": 0.0, "movable": True,
            "validity": {"bbox": True, "transform": True, "family": True,
                         "functions": True},
        })
    return {
        "schema_version": "scene_repair_v3_clean_furniture_scene_v1",
        "scene_id": "unit_scene", "stage": "furniture",
        "coordinate_frame": "room_local_z_up", "quaternion_order": "wxyz",
        "yaw_range": "[-pi,pi)", "room_type": "living_room",
        "length": 5.0, "width": 4.0, "openings": [], "furniture": furniture,
    }


def test_corruption_is_reproducible_and_targets_match_final_graph(vocabularies) -> None:
    raw = _raw_scene()
    builder = FurnitureGraphBuilder(vocabularies)
    first = corrupt_scene(raw, seed=123456, builder=builder)
    second = corrupt_scene(raw, seed=123456, builder=builder)
    assert torch.equal(first.targets.action_types, second.targets.action_types)
    assert torch.equal(first.targets.delta_m_rad, second.targets.delta_m_rad)
    assert torch.equal(first.graph.furniture_geometry, second.graph.furniture_geometry)

    clean = scene_to_input(raw)
    reconstructed_x = first.graph.furniture_geometry[:, 0] * clean.room.length_m
    reconstructed_y = first.graph.furniture_geometry[:, 1] * clean.room.width_m
    clean_xy = torch.tensor([[value.x_m, value.y_m] for value in clean.furniture])
    expected_delta = clean_xy - torch.stack((reconstructed_x, reconstructed_y), dim=-1)
    assert torch.allclose(first.targets.delta_m_rad[:, :2], expected_delta, atol=1e-6)


def test_manifest_freezes_scene_split_and_epoch_changes_corruption(
    tmp_path, vocabularies
) -> None:
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    for index in range(10):
        raw = _raw_scene()
        raw["scene_id"] = f"scene_{index}"
        (scenes / f"scene_{index}.json").write_text(json.dumps(raw), encoding="utf-8")
    manifest_path = tmp_path / "prepared" / "manifest.json"
    manifest = freeze_scene_manifest(scenes, manifest_path, seed=7)
    assert manifest["splits"] == {"train": 8, "val": 1, "test": 1}
    dataset = FurnitureCorruptionDataset(
        manifest_path, split="train", vocabularies=vocabularies, rules=None,
        base_seed=99, samples_per_scene=2,
    )
    first = dataset[0]
    dataset.set_epoch(1)
    second = dataset[0]
    assert first.seed != second.seed
    assert first.sampled_action_types != second.sampled_action_types


def test_semantic_targets_drop_unnecessary_rotation_component() -> None:
    clean = scene_to_input(_raw_scene())
    moved = replace(clean.furniture[0], x_m=0.7, yaw_rad=0.6)
    corrupted = replace(clean, furniture=(moved, clean.furniture[1]))
    full_delta = torch.tensor([[-1.5, 0.0, -0.6], [0.0, 0.0, 0.0]])
    actions, delta = _semantic_targets(
        clean, corrupted, (ActionType.BOTH, ActionType.KEEP),
        full_delta, CorruptionConfig(),
    )
    assert actions.tolist() == [int(ActionType.TRANSLATE), int(ActionType.KEEP)]
    assert torch.allclose(delta[0], torch.tensor([-1.5, 0.0, 0.0]))


def test_semantic_targets_keep_perturbation_that_remains_valid() -> None:
    clean = scene_to_input(_raw_scene())
    rotated = replace(clean.furniture[0], yaw_rad=0.3)
    corrupted = replace(clean, furniture=(rotated, clean.furniture[1]))
    actions, delta = _semantic_targets(
        clean, corrupted, (ActionType.ROTATE, ActionType.KEEP),
        torch.tensor([[0.0, 0.0, -0.3], [0.0, 0.0, 0.0]]),
        CorruptionConfig(),
    )
    assert actions.tolist() == [int(ActionType.KEEP), int(ActionType.KEEP)]
    assert torch.count_nonzero(delta) == 0


def test_observable_targets_repair_both_collision_participants() -> None:
    clean = scene_to_input(_raw_scene())
    first = replace(clean.furniture[0], x_m=0.7)
    corrupted = replace(clean, furniture=(first, clean.furniture[1]))
    actions, delta = _observable_targets(
        corrupted, CorruptionConfig(label_mode="observable_v3")
    )
    assert actions.tolist() == [
        int(ActionType.TRANSLATE), int(ActionType.TRANSLATE)
    ]
    assert delta[0, 0] < 0.0
    assert delta[1, 0] > 0.0


def test_joint_observable_targets_form_one_feasible_scene() -> None:
    raw = _raw_scene()
    raw["furniture"].append({
        **raw["furniture"][0],
        "object_id": "chair_2",
        "hssd_id": "asset_2",
        "x": 0.0,
    })
    for row in raw["furniture"]:
        row["x"] = 0.0
    corrupted = scene_to_input(raw)
    result = joint_translation_repair(corrupted)
    repaired = apply_pose_deltas(corrupted, result.delta_m_rad)
    actions = torch.linalg.vector_norm(result.delta_m_rad[:, :2], dim=-1) > 0.02
    assert result.converged
    assert hard_constraint_report(repaired).total == 0
    assert int(actions.sum()) >= 2


def test_project_predicted_repair_clears_residual_error() -> None:
    raw = _raw_scene()
    raw["furniture"][0]["x"] = 0.7
    raw["furniture"][1]["x"] = 0.8
    scene = scene_to_input(raw)
    weak_prediction = torch.tensor([[-0.02, 0.0, 0.0], [0.02, 0.0, 0.0]])
    result = project_predicted_repair(scene, weak_prediction)
    assert result.converged
    assert result.report.total == 0
    assert result.delta_m_rad.shape == (2, 3)
