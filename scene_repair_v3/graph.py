from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch

from .contracts import (
    COORDINATE_FRAME,
    EDGE_CATEGORICAL_DIMS,
    EDGE_CONTINUOUS_DIMS,
    GRAPH_SCHEMA_VERSION,
    QUATERNION_ORDER,
    YAW_RANGE,
    EdgeType,
    FurnitureGraphBatch,
    NodeType,
    TypedEdgeSet,
)
from .geometry import (
    OrientedRectangle,
    room_signed_margins,
    signed_separation_and_escape,
    wrap_yaw,
)
from .vocabulary import FurnitureVocabularies


@dataclass(frozen=True)
class RoomInput:
    room_type: str
    length_m: float
    width_m: float


@dataclass(frozen=True)
class OpeningInput:
    opening_id: str
    kind: str
    clearance_center_xyz_m: tuple[float, float, float]
    clearance_size_xyz_m: tuple[float, float, float]
    interior_normal_xy: tuple[float, float]
    geometry_validity: tuple[bool, ...] = (True,) * 8
    constraint_valid: bool = True


@dataclass(frozen=True)
class FurnitureInput:
    object_id: str
    x_m: float
    y_m: float
    yaw_rad: float
    bbox_width_m: float
    bbox_depth_m: float
    bbox_height_m: float
    category: str
    hssd_id: str | None = None
    family: str | None = None
    functions: tuple[str, ...] = ()
    geometry_validity: tuple[bool, ...] = (True,) * 7
    family_valid: bool = False
    functions_valid: bool = False
    movable: bool = True
    transform_valid: bool = True
    wall_anchor_direction: str | None = None
    wall_anchor_mode: str | None = None
    wall_anchor_valid: bool = False


@dataclass(frozen=True)
class FunctionalPartnerInput:
    source_id: str
    target_id: str
    orientation_mode: str
    target_strategy: str
    desired_gap_range_m: tuple[float, float] | None = None
    valid: bool = True


@dataclass(frozen=True)
class SceneInput:
    room: RoomInput
    furniture: tuple[FurnitureInput, ...]
    openings: tuple[OpeningInput, ...] = ()
    functional_partners: tuple[FunctionalPartnerInput, ...] = ()
    coordinate_frame: str = COORDINATE_FRAME
    quaternion_order: str = QUATERNION_ORDER
    yaw_range: str = YAW_RANGE


@dataclass
class _EdgeRows:
    index: list[tuple[int, int]] = field(default_factory=list)
    continuous: list[list[float]] = field(default_factory=list)
    categorical: list[list[int]] = field(default_factory=list)
    active: list[bool] = field(default_factory=list)

    def append(
        self,
        source: int,
        target: int,
        continuous: Sequence[float],
        categorical: Sequence[int],
        *,
        active: bool,
    ) -> None:
        self.index.append((source, target))
        self.continuous.append([float(value) for value in continuous])
        self.categorical.append([int(value) for value in categorical])
        self.active.append(bool(active))


class FurnitureGraphBuilder:
    """Build the frozen Furniture v1 graph from source-neutral scene fields."""

    def __init__(
        self,
        vocabularies: FurnitureVocabularies,
        *,
        max_furniture: int = 17,
    ) -> None:
        self.vocabularies = vocabularies
        self.max_furniture = int(max_furniture)
        if self.max_furniture <= 0:
            raise ValueError("max_furniture must be positive")

    def build(self, scene: SceneInput) -> FurnitureGraphBatch:
        self._validate_scene(scene)
        room = scene.room
        length, width = float(room.length_m), float(room.width_m)
        room_scale = max(length, width)

        opening_offset = 1
        furniture_offset = opening_offset + len(scene.openings)
        room_indices = torch.tensor([0], dtype=torch.long)
        opening_indices = torch.arange(
            opening_offset, furniture_offset, dtype=torch.long
        )
        furniture_indices = torch.arange(
            furniture_offset,
            furniture_offset + len(scene.furniture),
            dtype=torch.long,
        )
        node_types = torch.tensor(
            [int(NodeType.ROOM)]
            + [int(NodeType.OPENING)] * len(scene.openings)
            + [int(NodeType.FURNITURE)] * len(scene.furniture),
            dtype=torch.long,
        )
        graph_index = torch.zeros(node_types.numel(), dtype=torch.long)

        room_type_ids = torch.tensor(
            [self.vocabularies.room_types.encode(room.room_type)], dtype=torch.long
        )
        room_features = torch.tensor([[length, width]], dtype=torch.float32)

        opening_kind_ids = torch.tensor(
            [self.vocabularies.opening_kinds.encode(value.kind) for value in scene.openings],
            dtype=torch.long,
        )
        opening_geometry_rows: list[list[float]] = []
        opening_validity_rows: list[list[float]] = []
        for opening in scene.openings:
            cx, cy, cz = opening.clearance_center_xyz_m
            sx, sy, sz = opening.clearance_size_xyz_m
            nx, ny = _normalize_direction(opening.interior_normal_xy)
            values = [
                cx / length,
                cy / width,
                cz / room_scale,
                sx / length,
                sy / width,
                sz / room_scale,
                nx,
                ny,
            ]
            mask = [float(value) for value in opening.geometry_validity]
            opening_geometry_rows.append(
                [value * valid for value, valid in zip(values, mask)]
            )
            opening_validity_rows.append(mask)
        opening_geometry = _float_matrix(opening_geometry_rows, 8)
        opening_validity = _float_matrix(opening_validity_rows, 8)

        furniture_category_ids: list[int] = []
        furniture_geometry_rows: list[list[float]] = []
        furniture_geometry_validity_rows: list[list[float]] = []
        furniture_family_ids: list[int] = []
        furniture_function_rows: list[list[float]] = []
        furniture_behavior_validity_rows: list[list[float]] = []
        furniture_eligibility: list[bool] = []
        rectangles: dict[str, OrientedRectangle] = {}
        furniture_node: dict[str, int] = {}

        for offset, furniture in enumerate(scene.furniture):
            node_index = furniture_offset + offset
            furniture_node[furniture.object_id] = node_index
            yaw = wrap_yaw(furniture.yaw_rad)
            rectangle = OrientedRectangle(
                furniture.x_m,
                furniture.y_m,
                furniture.bbox_width_m,
                furniture.bbox_depth_m,
                yaw,
            )
            rectangles[furniture.object_id] = rectangle
            furniture_category_ids.append(
                self.vocabularies.categories.encode(furniture.category)
            )
            raw_geometry = [
                furniture.x_m / length,
                furniture.y_m / width,
                math.sin(yaw),
                math.cos(yaw),
                furniture.bbox_width_m / length,
                furniture.bbox_depth_m / width,
                furniture.bbox_height_m / room_scale,
            ]
            geometry_mask = [float(value) for value in furniture.geometry_validity]
            furniture_geometry_rows.append(
                [value * valid for value, valid in zip(raw_geometry, geometry_mask)]
            )
            furniture_geometry_validity_rows.append(geometry_mask)
            furniture_family_ids.append(
                self.vocabularies.families.encode(
                    furniture.family,
                    strict=furniture.family is not None and furniture.family_valid,
                )
            )
            furniture_function_rows.append(
                self.vocabularies.functions.multihot(
                    furniture.functions,
                    strict=furniture.functions_valid,
                )
                if furniture.functions_valid
                else [0.0] * len(self.vocabularies.functions)
            )
            furniture_behavior_validity_rows.append(
                [float(furniture.family_valid), float(furniture.functions_valid)]
            )
            furniture_eligibility.append(
                bool(furniture.movable and furniture.transform_valid)
            )

        edges = {edge_type: _EdgeRows() for edge_type in EdgeType}
        for opening_index in opening_indices.tolist():
            continuous = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            categorical = [0, 0]
            edges[EdgeType.ROOM_MEMBERSHIP].append(
                0, opening_index, continuous, categorical, active=True
            )
            edges[EdgeType.ROOM_MEMBERSHIP].append(
                opening_index, 0, continuous, categorical, active=True
            )

        for furniture in scene.furniture:
            node_index = furniture_node[furniture.object_id]
            margins = room_signed_margins(rectangles[furniture.object_id], length, width)
            margins_valid = all(furniture.geometry_validity)
            continuous = [
                margins[0] / length,
                margins[1] / length,
                margins[2] / width,
                margins[3] / width,
                float(margins_valid),
                float(furniture.wall_anchor_valid),
            ]
            anchor_direction = self.vocabularies.anchor_directions.encode(
                furniture.wall_anchor_direction,
                strict=furniture.wall_anchor_valid,
            )
            anchor_mode = self.vocabularies.anchor_modes.encode(
                furniture.wall_anchor_mode,
                strict=furniture.wall_anchor_valid,
            )
            categorical = [anchor_direction, anchor_mode]
            edges[EdgeType.ROOM_MEMBERSHIP].append(
                0, node_index, continuous, categorical, active=True
            )
            edges[EdgeType.ROOM_MEMBERSHIP].append(
                node_index, 0, continuous, categorical, active=True
            )

        for source in scene.furniture:
            for target in scene.furniture:
                if source.object_id == target.object_id:
                    continue
                source_rect = rectangles[source.object_id]
                target_rect = rectangles[target.object_id]
                separation, escape_x, escape_y = signed_separation_and_escape(
                    source_rect, target_rect
                )
                relative_yaw = wrap_yaw(target_rect.yaw - source_rect.yaw)
                pair_valid = all(source.geometry_validity) and all(target.geometry_validity)
                continuous = [
                    (target.x_m - source.x_m) / length,
                    (target.y_m - source.y_m) / width,
                    math.sin(relative_yaw),
                    math.cos(relative_yaw),
                    separation / room_scale,
                    escape_x / length,
                    escape_y / width,
                    float(pair_valid),
                ]
                edges[EdgeType.FURNITURE_SPATIAL].append(
                    furniture_node[source.object_id],
                    furniture_node[target.object_id],
                    continuous,
                    (),
                    active=pair_valid,
                )

        seen_functional: set[tuple[str, str]] = set()
        for relation in scene.functional_partners:
            signature = (relation.source_id, relation.target_id)
            if signature in seen_functional:
                raise ValueError(f"duplicate FUNCTIONAL_PARTNER edge {signature}")
            seen_functional.add(signature)
            if relation.source_id == relation.target_id:
                raise ValueError("FUNCTIONAL_PARTNER cannot be a self loop")
            if relation.source_id not in furniture_node or relation.target_id not in furniture_node:
                raise ValueError("FUNCTIONAL_PARTNER endpoint is not a Furniture node")
            orientation_id = self.vocabularies.orientation_modes.encode(
                relation.orientation_mode, strict=relation.valid
            )
            strategy_id = self.vocabularies.target_strategies.encode(
                relation.target_strategy, strict=relation.valid
            )
            if relation.desired_gap_range_m is None:
                gap_min = gap_max = 0.0
                gap_valid = False
            else:
                gap_min, gap_max = relation.desired_gap_range_m
                if not all(math.isfinite(value) for value in (gap_min, gap_max)):
                    raise ValueError("desired gap range must be finite")
                if gap_min < 0 or gap_max < gap_min:
                    raise ValueError("desired gap range must satisfy 0 <= min <= max")
                gap_valid = True
            continuous = [
                gap_min / room_scale,
                gap_max / room_scale,
                float(gap_valid),
                float(relation.valid),
            ]
            edges[EdgeType.FUNCTIONAL_PARTNER].append(
                furniture_node[relation.source_id],
                furniture_node[relation.target_id],
                continuous,
                [orientation_id, strategy_id],
                active=relation.valid,
            )

        for opening_offset_index, opening in enumerate(scene.openings):
            opening_node = opening_offset + opening_offset_index
            cx, cy, cz = opening.clearance_center_xyz_m
            sx, sy, sz = opening.clearance_size_xyz_m
            opening_rect = OrientedRectangle(cx, cy, sx, sy, 0.0)
            opening_low_z, opening_high_z = cz - sz * 0.5, cz + sz * 0.5
            for furniture in scene.furniture:
                furniture_rect = rectangles[furniture.object_id]
                xy_separation, escape_x, escape_y = signed_separation_and_escape(
                    opening_rect, furniture_rect
                )
                z_separation = max(
                    opening_low_z - furniture.bbox_height_m,
                    0.0 - opening_high_z,
                )
                separation = max(xy_separation, z_separation)
                if separation >= 0:
                    escape_x = escape_y = 0.0
                active = bool(
                    opening.constraint_valid
                    and all(opening.geometry_validity)
                    and all(furniture.geometry_validity)
                )
                continuous = [
                    separation / room_scale,
                    escape_x / length,
                    escape_y / width,
                    float(active),
                ]
                edges[EdgeType.OBJECT_OPENING].append(
                    opening_node,
                    furniture_node[furniture.object_id],
                    continuous,
                    (),
                    active=active,
                )

        typed_edges = {
            edge_type: _to_typed_edges(edge_type, rows) for edge_type, rows in edges.items()
        }
        batch = FurnitureGraphBatch(
            node_types=node_types,
            graph_index=graph_index,
            room_indices=room_indices,
            room_type_ids=room_type_ids,
            room_features=room_features,
            opening_indices=opening_indices,
            opening_kind_ids=opening_kind_ids,
            opening_geometry=opening_geometry,
            opening_validity=opening_validity,
            furniture_indices=furniture_indices,
            furniture_category_ids=torch.tensor(
                furniture_category_ids, dtype=torch.long
            ),
            furniture_geometry=_float_matrix(furniture_geometry_rows, 7),
            furniture_geometry_validity=_float_matrix(
                furniture_geometry_validity_rows, 7
            ),
            furniture_family_ids=torch.tensor(furniture_family_ids, dtype=torch.long),
            furniture_functions=_float_matrix(
                furniture_function_rows, len(self.vocabularies.functions)
            ),
            furniture_behavior_validity=_float_matrix(
                furniture_behavior_validity_rows, 2
            ),
            furniture_eligibility=torch.tensor(
                furniture_eligibility, dtype=torch.bool
            ),
            edges=typed_edges,
            object_ids=(
                "ROOM",
                *(opening.opening_id for opening in scene.openings),
                *(furniture.object_id for furniture in scene.furniture),
            ),
        )
        batch.validate(max_furniture=self.max_furniture)
        return batch

    def _validate_scene(self, scene: SceneInput) -> None:
        if scene.coordinate_frame != COORDINATE_FRAME:
            raise ValueError(f"coordinate frame must be {COORDINATE_FRAME}")
        if scene.quaternion_order != QUATERNION_ORDER:
            raise ValueError(f"quaternion order must be {QUATERNION_ORDER}")
        if scene.yaw_range != YAW_RANGE:
            raise ValueError(f"yaw range must be {YAW_RANGE}")
        if len(scene.furniture) > self.max_furniture:
            raise ValueError(
                f"Furniture count {len(scene.furniture)} exceeds {self.max_furniture}"
            )
        if not all(
            math.isfinite(value) and value > 0
            for value in (scene.room.length_m, scene.room.width_m)
        ):
            raise ValueError("room length and width must be finite and positive")
        object_ids = [value.object_id for value in scene.furniture]
        opening_ids = [value.opening_id for value in scene.openings]
        all_ids = [*object_ids, *opening_ids]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("Furniture and Opening IDs must be unique")
        for furniture in scene.furniture:
            values = (
                furniture.x_m,
                furniture.y_m,
                furniture.yaw_rad,
                furniture.bbox_width_m,
                furniture.bbox_depth_m,
                furniture.bbox_height_m,
            )
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Furniture {furniture.object_id!r} has non-finite geometry")
            if min(
                furniture.bbox_width_m,
                furniture.bbox_depth_m,
                furniture.bbox_height_m,
            ) <= 0:
                raise ValueError(f"Furniture {furniture.object_id!r} has invalid bbox")
            if len(furniture.geometry_validity) != 7:
                raise ValueError("Furniture geometry_validity must have seven entries")
            if furniture.family_valid and furniture.family is None:
                raise ValueError("family_valid requires a family label")
            if furniture.wall_anchor_valid and (
                furniture.wall_anchor_direction is None
                or furniture.wall_anchor_mode is None
            ):
                raise ValueError("valid wall anchor requires direction and mode")
        for opening in scene.openings:
            if len(opening.geometry_validity) != 8:
                raise ValueError("Opening geometry_validity must have eight entries")
            values = (
                *opening.clearance_center_xyz_m,
                *opening.clearance_size_xyz_m,
                *opening.interior_normal_xy,
            )
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Opening {opening.opening_id!r} has non-finite geometry")
            if min(opening.clearance_size_xyz_m) <= 0:
                raise ValueError(f"Opening {opening.opening_id!r} has invalid clearance size")


def collate_graphs(graphs: Iterable[FurnitureGraphBatch]) -> FurnitureGraphBatch:
    values = tuple(graphs)
    if not values:
        raise ValueError("cannot collate an empty graph list")
    for graph in values:
        graph.validate()
    function_dim = values[0].furniture_functions.shape[1]
    if any(graph.furniture_functions.shape[1] != function_dim for graph in values):
        raise ValueError("all graphs must use the same function vocabulary")

    node_offset = 0
    graph_offset = 0
    node_types: list[torch.Tensor] = []
    graph_indices: list[torch.Tensor] = []
    room_indices: list[torch.Tensor] = []
    opening_indices: list[torch.Tensor] = []
    furniture_indices: list[torch.Tensor] = []
    object_ids: list[str] = []
    edge_parts: dict[EdgeType, list[TypedEdgeSet]] = {
        edge_type: [] for edge_type in EdgeType
    }
    for graph in values:
        node_types.append(graph.node_types)
        graph_indices.append(graph.graph_index + graph_offset)
        room_indices.append(graph.room_indices + node_offset)
        opening_indices.append(graph.opening_indices + node_offset)
        furniture_indices.append(graph.furniture_indices + node_offset)
        object_ids.extend(graph.object_ids)
        for edge_type in EdgeType:
            edge_set = graph.edges[edge_type]
            edge_parts[edge_type].append(
                TypedEdgeSet(
                    edge_type,
                    edge_set.edge_index + node_offset,
                    edge_set.continuous,
                    edge_set.categorical,
                    edge_set.active,
                )
            )
        node_offset += graph.num_nodes
        graph_offset += graph.num_graphs

    result = FurnitureGraphBatch(
        node_types=torch.cat(node_types),
        graph_index=torch.cat(graph_indices),
        room_indices=torch.cat(room_indices),
        room_type_ids=torch.cat([graph.room_type_ids for graph in values]),
        room_features=torch.cat([graph.room_features for graph in values]),
        opening_indices=torch.cat(opening_indices),
        opening_kind_ids=torch.cat([graph.opening_kind_ids for graph in values]),
        opening_geometry=torch.cat([graph.opening_geometry for graph in values]),
        opening_validity=torch.cat([graph.opening_validity for graph in values]),
        furniture_indices=torch.cat(furniture_indices),
        furniture_category_ids=torch.cat(
            [graph.furniture_category_ids for graph in values]
        ),
        furniture_geometry=torch.cat([graph.furniture_geometry for graph in values]),
        furniture_geometry_validity=torch.cat(
            [graph.furniture_geometry_validity for graph in values]
        ),
        furniture_family_ids=torch.cat(
            [graph.furniture_family_ids for graph in values]
        ),
        furniture_functions=torch.cat(
            [graph.furniture_functions for graph in values]
        ),
        furniture_behavior_validity=torch.cat(
            [graph.furniture_behavior_validity for graph in values]
        ),
        furniture_eligibility=torch.cat(
            [graph.furniture_eligibility for graph in values]
        ),
        edges={
            edge_type: TypedEdgeSet(
                edge_type,
                torch.cat([part.edge_index for part in parts], dim=1),
                torch.cat([part.continuous for part in parts]),
                torch.cat([part.categorical for part in parts]),
                torch.cat([part.active for part in parts]),
            )
            for edge_type, parts in edge_parts.items()
        },
        object_ids=tuple(object_ids),
        graph_schema_version=GRAPH_SCHEMA_VERSION,
        coordinate_frame=COORDINATE_FRAME,
    )
    result.validate()
    return result


def _to_typed_edges(edge_type: EdgeType, rows: _EdgeRows) -> TypedEdgeSet:
    if rows.index:
        edge_index = torch.tensor(rows.index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return TypedEdgeSet(
        edge_type=edge_type,
        edge_index=edge_index,
        continuous=_float_matrix(
            rows.continuous, EDGE_CONTINUOUS_DIMS[int(edge_type)]
        ),
        categorical=_long_matrix(
            rows.categorical, EDGE_CATEGORICAL_DIMS[int(edge_type)]
        ),
        active=torch.tensor(rows.active, dtype=torch.bool),
    )


def _float_matrix(rows: Sequence[Sequence[float]], columns: int) -> torch.Tensor:
    if not rows:
        return torch.empty((0, columns), dtype=torch.float32)
    result = torch.tensor(rows, dtype=torch.float32)
    if result.shape != (len(rows), columns):
        raise ValueError(f"expected matrix [{len(rows)},{columns}], got {tuple(result.shape)}")
    return result


def _long_matrix(rows: Sequence[Sequence[int]], columns: int) -> torch.Tensor:
    if not rows:
        return torch.empty((0, columns), dtype=torch.long)
    if columns == 0:
        return torch.empty((len(rows), 0), dtype=torch.long)
    result = torch.tensor(rows, dtype=torch.long)
    if result.shape != (len(rows), columns):
        raise ValueError(f"expected matrix [{len(rows)},{columns}], got {tuple(result.shape)}")
    return result


def _normalize_direction(value: tuple[float, float]) -> tuple[float, float]:
    x, y = float(value[0]), float(value[1])
    norm = math.hypot(x, y)
    if norm <= 1.0e-12:
        return 0.0, 0.0
    return x / norm, y / norm
