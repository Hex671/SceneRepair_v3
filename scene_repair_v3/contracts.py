from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Mapping

import torch


MODEL_SCHEMA_VERSION = "furniture_network_v2_constraint_summary"
GRAPH_SCHEMA_VERSION = "furniture_graph_v1"
COORDINATE_FRAME = "room_local_z_up"
QUATERNION_ORDER = "wxyz"
YAW_RANGE = "[-pi,pi)"

ROOM_FEATURE_DIM = 2
OPENING_GEOMETRY_DIM = 8
FURNITURE_GEOMETRY_DIM = 7

EDGE_CONTINUOUS_DIMS = {
    0: 6,
    1: 8,
    2: 4,
    3: 4,
}
EDGE_CATEGORICAL_DIMS = {
    0: 2,
    1: 0,
    2: 2,
    3: 0,
}


class NodeType(IntEnum):
    ROOM = 0
    OPENING = 1
    FURNITURE = 2


class EdgeType(IntEnum):
    ROOM_MEMBERSHIP = 0
    FURNITURE_SPATIAL = 1
    FUNCTIONAL_PARTNER = 2
    OBJECT_OPENING = 3


class ActionType(IntEnum):
    KEEP = 0
    TRANSLATE = 1
    ROTATE = 2
    BOTH = 3


@dataclass(frozen=True)
class ActionRange:
    max_translation_m: float
    max_yaw_rad: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_translation_m) or self.max_translation_m <= 0:
            raise ValueError("max_translation_m must be finite and positive")
        if not math.isfinite(self.max_yaw_rad) or not 0 < self.max_yaw_rad <= math.pi:
            raise ValueError("max_yaw_rad must be in (0, pi]")

    def to_dict(self) -> dict[str, float]:
        return {
            "max_translation_m": self.max_translation_m,
            "max_yaw_rad": self.max_yaw_rad,
        }


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 128
    layers: int = 2
    heads: int = 4
    edge_dim: int = 32
    geometry_dim: int = 48
    category_dim: int = 32
    behavior_dim: int = 48
    constraint_dim: int = 32
    dropout: float = 0.1
    max_furniture: int = 17

    def __post_init__(self) -> None:
        if self.layers != 2:
            raise ValueError("Furniture v1 requires exactly two graph layers")
        if self.hidden_dim <= 0 or self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be positive and divisible by heads")
        if self.edge_dim <= 0 or min(
            self.geometry_dim, self.category_dim, self.behavior_dim, self.constraint_dim
        ) <= 0:
            raise ValueError("encoder dimensions must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.max_furniture <= 0:
            raise ValueError("max_furniture must be positive")


@dataclass
class TypedEdgeSet:
    edge_type: EdgeType
    edge_index: torch.Tensor
    continuous: torch.Tensor
    categorical: torch.Tensor
    active: torch.Tensor

    def validate(self, num_nodes: int) -> None:
        if self.edge_index.dtype != torch.long or self.edge_index.ndim != 2:
            raise ValueError("edge_index must be a long tensor shaped [2,E]")
        if self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must be shaped [2,E]")
        edges = self.edge_index.shape[1]
        expected_continuous = EDGE_CONTINUOUS_DIMS[int(self.edge_type)]
        expected_categorical = EDGE_CATEGORICAL_DIMS[int(self.edge_type)]
        if self.continuous.shape != (edges, expected_continuous):
            raise ValueError(
                f"{self.edge_type.name} continuous features must be "
                f"[{edges},{expected_continuous}]"
            )
        if self.categorical.shape != (edges, expected_categorical):
            raise ValueError(
                f"{self.edge_type.name} categorical features must be "
                f"[{edges},{expected_categorical}]"
            )
        if self.categorical.dtype != torch.long:
            raise ValueError("categorical edge features must use torch.long")
        if self.active.shape != (edges,) or self.active.dtype != torch.bool:
            raise ValueError("edge active mask must be bool [E]")
        if edges and (
            int(self.edge_index.min()) < 0 or int(self.edge_index.max()) >= num_nodes
        ):
            raise ValueError("edge endpoint is outside the node array")
        if not torch.isfinite(self.continuous).all():
            raise ValueError("edge features must be finite")

    def to(self, device: torch.device | str) -> "TypedEdgeSet":
        return TypedEdgeSet(
            self.edge_type,
            self.edge_index.to(device),
            self.continuous.to(device),
            self.categorical.to(device),
            self.active.to(device),
        )


@dataclass
class FurnitureGraphBatch:
    node_types: torch.Tensor
    graph_index: torch.Tensor
    room_indices: torch.Tensor
    room_type_ids: torch.Tensor
    room_features: torch.Tensor
    opening_indices: torch.Tensor
    opening_kind_ids: torch.Tensor
    opening_geometry: torch.Tensor
    opening_validity: torch.Tensor
    furniture_indices: torch.Tensor
    furniture_category_ids: torch.Tensor
    furniture_geometry: torch.Tensor
    furniture_geometry_validity: torch.Tensor
    furniture_family_ids: torch.Tensor
    furniture_functions: torch.Tensor
    furniture_behavior_validity: torch.Tensor
    furniture_eligibility: torch.Tensor
    edges: Mapping[EdgeType, TypedEdgeSet]
    object_ids: tuple[str, ...] = field(default_factory=tuple)
    graph_schema_version: str = GRAPH_SCHEMA_VERSION
    coordinate_frame: str = COORDINATE_FRAME

    @property
    def num_nodes(self) -> int:
        return int(self.node_types.numel())

    @property
    def num_graphs(self) -> int:
        return int(self.graph_index.max()) + 1 if self.graph_index.numel() else 0

    @property
    def furniture_graph_index(self) -> torch.Tensor:
        return self.graph_index[self.furniture_indices]

    def validate(self, *, max_furniture: int = 17, strict_topology: bool = True) -> None:
        n = self.num_nodes
        if self.node_types.shape != (n,) or self.node_types.dtype != torch.long:
            raise ValueError("node_types must be long [N]")
        if self.graph_index.shape != (n,) or self.graph_index.dtype != torch.long:
            raise ValueError("graph_index must be long [N]")
        if n == 0 or int(self.graph_index.min()) < 0:
            raise ValueError("graph batch must contain nodes with non-negative graph ids")
        if self.graph_schema_version != GRAPH_SCHEMA_VERSION:
            raise ValueError("graph schema version mismatch")
        if self.coordinate_frame != COORDINATE_FRAME:
            raise ValueError("Furniture v1 requires room_local_z_up")
        self._validate_node_block(
            self.room_indices,
            NodeType.ROOM,
            self.room_type_ids,
            self.room_features,
            feature_dim=ROOM_FEATURE_DIM,
        )
        self._validate_node_block(
            self.opening_indices,
            NodeType.OPENING,
            self.opening_kind_ids,
            self.opening_geometry,
            self.opening_validity,
            feature_dim=OPENING_GEOMETRY_DIM,
        )
        self._validate_node_block(
            self.furniture_indices,
            NodeType.FURNITURE,
            self.furniture_category_ids,
            self.furniture_geometry,
            self.furniture_geometry_validity,
            feature_dim=FURNITURE_GEOMETRY_DIM,
        )
        furniture = self.furniture_indices.numel()
        if self.furniture_family_ids.shape != (furniture,):
            raise ValueError("furniture_family_ids must be [F]")
        if self.furniture_functions.ndim != 2 or self.furniture_functions.shape[0] != furniture:
            raise ValueError("furniture_functions must be [F,num_functions]")
        if self.furniture_behavior_validity.shape != (furniture, 2):
            raise ValueError("furniture_behavior_validity must be [F,2]")
        if self.furniture_eligibility.shape != (furniture,) or self.furniture_eligibility.dtype != torch.bool:
            raise ValueError("furniture_eligibility must be bool [F]")
        for tensor in (
            self.room_features,
            self.opening_geometry,
            self.opening_validity,
            self.furniture_geometry,
            self.furniture_geometry_validity,
            self.furniture_functions,
            self.furniture_behavior_validity,
        ):
            if not torch.isfinite(tensor).all():
                raise ValueError("node features and masks must be finite")
        expected_edges = set(EdgeType)
        if set(self.edges) != expected_edges:
            raise ValueError("all four typed edge sets must be present")
        for edge_type, edge_set in self.edges.items():
            if edge_set.edge_type != edge_type:
                raise ValueError("edge map key and edge type disagree")
            edge_set.validate(n)
            if edge_set.edge_index.numel():
                source_graph = self.graph_index[edge_set.edge_index[0]]
                target_graph = self.graph_index[edge_set.edge_index[1]]
                if not torch.equal(source_graph, target_graph):
                    raise ValueError("edges cannot cross disjoint graphs")
        counts = torch.bincount(self.furniture_graph_index, minlength=self.num_graphs)
        if bool((counts > max_furniture).any()):
            raise ValueError(f"Furniture count exceeds contract F <= {max_furniture}")
        room_counts = torch.bincount(
            self.graph_index[self.room_indices], minlength=self.num_graphs
        )
        if not bool((room_counts == 1).all()):
            raise ValueError("each graph must have exactly one ROOM node")
        if strict_topology:
            self._validate_spatial_topology(counts)

    def _validate_node_block(
        self,
        indices: torch.Tensor,
        node_type: NodeType,
        ids: torch.Tensor,
        features: torch.Tensor,
        validity: torch.Tensor | None = None,
        *,
        feature_dim: int,
    ) -> None:
        count = indices.numel()
        if indices.dtype != torch.long or indices.shape != (count,):
            raise ValueError("typed node indices must be long [count]")
        if ids.dtype != torch.long or ids.shape != (count,):
            raise ValueError("typed vocabulary ids must be long [count]")
        if features.shape != (count, feature_dim):
            raise ValueError(f"typed node features must be [{count},{feature_dim}]")
        if validity is not None and validity.shape != features.shape:
            raise ValueError("node validity masks must match feature shape")
        if count and not bool((self.node_types[indices] == int(node_type)).all()):
            raise ValueError(f"{node_type.name} indices point to another node type")

    def _validate_spatial_topology(self, counts: torch.Tensor) -> None:
        edges = self.edges[EdgeType.FURNITURE_SPATIAL]
        source, target = edges.edge_index
        if source.numel() and bool((source == target).any()):
            raise ValueError("FURNITURE_SPATIAL cannot contain self loops")
        for graph_id, count_tensor in enumerate(counts):
            count = int(count_tensor)
            expected = count * (count - 1)
            if source.numel():
                mask = self.graph_index[source] == graph_id
                actual = int(mask.sum())
            else:
                actual = 0
            if actual != expected:
                raise ValueError(
                    f"graph {graph_id} spatial topology has {actual} edges; expected {expected}"
                )

    def to(self, device: torch.device | str) -> "FurnitureGraphBatch":
        return FurnitureGraphBatch(
            node_types=self.node_types.to(device),
            graph_index=self.graph_index.to(device),
            room_indices=self.room_indices.to(device),
            room_type_ids=self.room_type_ids.to(device),
            room_features=self.room_features.to(device),
            opening_indices=self.opening_indices.to(device),
            opening_kind_ids=self.opening_kind_ids.to(device),
            opening_geometry=self.opening_geometry.to(device),
            opening_validity=self.opening_validity.to(device),
            furniture_indices=self.furniture_indices.to(device),
            furniture_category_ids=self.furniture_category_ids.to(device),
            furniture_geometry=self.furniture_geometry.to(device),
            furniture_geometry_validity=self.furniture_geometry_validity.to(device),
            furniture_family_ids=self.furniture_family_ids.to(device),
            furniture_functions=self.furniture_functions.to(device),
            furniture_behavior_validity=self.furniture_behavior_validity.to(device),
            furniture_eligibility=self.furniture_eligibility.to(device),
            edges={key: value.to(device) for key, value in self.edges.items()},
            object_ids=self.object_ids,
            graph_schema_version=self.graph_schema_version,
            coordinate_frame=self.coordinate_frame,
        )


@dataclass
class ModelOutput:
    action_type_logits: torch.Tensor
    primary_delta_normalized: torch.Tensor
    furniture_indices: torch.Tensor
    furniture_graph_index: torch.Tensor
    furniture_eligibility: torch.Tensor
    node_states: torch.Tensor
    relation_gate_weights: tuple[torch.Tensor, ...]
