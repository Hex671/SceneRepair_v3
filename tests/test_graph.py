from __future__ import annotations

import math

import pytest
import torch

from scene_repair_v3 import (
    EdgeType,
    FurnitureGraphBuilder,
    FurnitureInput,
    RoomInput,
    SceneInput,
    collate_graphs,
)
from scene_repair_v3.geometry import OrientedRectangle, signed_separation_and_escape


def test_graph_builder_emits_frozen_topology(vocabularies, sample_scene) -> None:
    graph = FurnitureGraphBuilder(vocabularies).build(sample_scene)

    assert graph.num_nodes == 4
    assert graph.room_indices.tolist() == [0]
    assert graph.opening_indices.tolist() == [1]
    assert graph.furniture_indices.tolist() == [2, 3]
    assert graph.edges[EdgeType.ROOM_MEMBERSHIP].edge_index.shape[1] == 6
    assert graph.edges[EdgeType.FURNITURE_SPATIAL].edge_index.shape[1] == 2
    assert graph.edges[EdgeType.FUNCTIONAL_PARTNER].edge_index.shape[1] == 1
    assert graph.edges[EdgeType.OBJECT_OPENING].edge_index.shape[1] == 2
    graph.validate()


def test_spatial_features_use_room_frame_and_wrapped_yaw(
    vocabularies, sample_scene
) -> None:
    graph = FurnitureGraphBuilder(vocabularies).build(sample_scene)
    edges = graph.edges[EdgeType.FURNITURE_SPATIAL]
    pairs = edges.edge_index.t().tolist()
    bed_to_chair = pairs.index([2, 3])
    chair_to_bed = pairs.index([3, 2])

    forward = edges.continuous[bed_to_chair]
    reverse = edges.continuous[chair_to_bed]
    assert forward[0].item() == pytest.approx(2.0 / 5.0)
    assert forward[1].item() == pytest.approx(0.2 / 4.0)
    assert reverse[0].item() == pytest.approx(-2.0 / 5.0)
    assert reverse[1].item() == pytest.approx(-0.2 / 4.0)
    assert forward[2].item() == pytest.approx(0.0, abs=1.0e-6)
    assert forward[3].item() == pytest.approx(-1.0, abs=1.0e-6)


def test_collision_escape_moves_target_out_of_source() -> None:
    source = OrientedRectangle(0.0, 0.0, 2.0, 1.0, 0.0)
    target = OrientedRectangle(0.8, 0.0, 1.0, 1.0, 0.0)
    separation, escape_x, escape_y = signed_separation_and_escape(source, target)
    assert separation < 0
    assert escape_x > 0
    assert escape_y == pytest.approx(0.0)

    moved = OrientedRectangle(0.8 + escape_x + 1.0e-5, 0.0, 1.0, 1.0, 0.0)
    moved_separation, _, _ = signed_separation_and_escape(source, moved)
    assert moved_separation >= 0


def test_collate_builds_disjoint_batch(vocabularies, sample_scene) -> None:
    builder = FurnitureGraphBuilder(vocabularies)
    first = builder.build(sample_scene)
    second = builder.build(sample_scene)
    batch = collate_graphs([first, second])

    assert batch.num_graphs == 2
    assert batch.num_nodes == 8
    assert batch.furniture_graph_index.tolist() == [0, 0, 1, 1]
    spatial = batch.edges[EdgeType.FURNITURE_SPATIAL].edge_index
    assert spatial.shape[1] == 4
    assert torch.equal(batch.graph_index[spatial[0]], batch.graph_index[spatial[1]])
    batch.validate()


def test_builder_rejects_scenes_beyond_furniture_budget(vocabularies) -> None:
    furniture = tuple(
        FurnitureInput(
            f"chair_{index}",
            float(index),
            0.0,
            0.0,
            0.5,
            0.5,
            0.8,
            "chair",
        )
        for index in range(18)
    )
    with pytest.raises(ValueError, match="exceeds"):
        FurnitureGraphBuilder(vocabularies).build(
            SceneInput(RoomInput("bedroom", 20.0, 10.0), furniture)
        )
