from __future__ import annotations

import math
from dataclasses import dataclass


def wrap_yaw(value: float) -> float:
    return (float(value) + math.pi) % (2 * math.pi) - math.pi


def quaternion_to_yaw(rotation_wxyz: tuple[float, float, float, float]) -> float:
    w, x, y, z = (float(value) for value in rotation_wxyz)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("rotation_wxyz must be a finite non-zero quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return wrap_yaw(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


@dataclass(frozen=True)
class OrientedRectangle:
    center_x: float
    center_y: float
    width: float
    depth: float
    yaw: float

    def __post_init__(self) -> None:
        values = (self.center_x, self.center_y, self.width, self.depth, self.yaw)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("oriented rectangle values must be finite")
        if self.width <= 0 or self.depth <= 0:
            raise ValueError("oriented rectangle extents must be positive")

    @property
    def axes(self) -> tuple[tuple[float, float], tuple[float, float]]:
        cosine, sine = math.cos(self.yaw), math.sin(self.yaw)
        return ((cosine, sine), (-sine, cosine))

    def radius_on(self, axis: tuple[float, float]) -> float:
        own_x, own_y = self.axes
        return (
            self.width * 0.5 * abs(_dot(own_x, axis))
            + self.depth * 0.5 * abs(_dot(own_y, axis))
        )

    @property
    def room_aabb_half_extents(self) -> tuple[float, float]:
        cosine, sine = abs(math.cos(self.yaw)), abs(math.sin(self.yaw))
        return (
            cosine * self.width * 0.5 + sine * self.depth * 0.5,
            sine * self.width * 0.5 + cosine * self.depth * 0.5,
        )


def signed_separation_and_escape(
    source: OrientedRectangle,
    target: OrientedRectangle,
) -> tuple[float, float, float]:
    """Return SAT signed separation and the minimum vector moving target out.

    Positive separation means a separating axis exists. Negative separation is
    penetration depth on the minimum-translation axis. The escape vector is
    zero for already separated rectangles.
    """

    delta = (target.center_x - source.center_x, target.center_y - source.center_y)
    candidates: list[tuple[float, tuple[float, float], float]] = []
    for axis in (*source.axes, *target.axes):
        projection = _dot(delta, axis)
        gap = abs(projection) - source.radius_on(axis) - target.radius_on(axis)
        candidates.append((gap, axis, projection))
    signed_separation, axis, projection = max(candidates, key=lambda item: item[0])
    if signed_separation >= 0:
        return signed_separation, 0.0, 0.0
    direction = 1.0 if projection >= 0 else -1.0
    magnitude = -signed_separation
    return signed_separation, axis[0] * direction * magnitude, axis[1] * direction * magnitude


def room_signed_margins(
    rectangle: OrientedRectangle,
    length: float,
    width: float,
) -> tuple[float, float, float, float]:
    half_x, half_y = rectangle.room_aabb_half_extents
    left = rectangle.center_x - half_x + length * 0.5
    right = length * 0.5 - rectangle.center_x - half_x
    bottom = rectangle.center_y - half_y + width * 0.5
    top = width * 0.5 - rectangle.center_y - half_y
    return left, right, bottom, top


def _dot(left: tuple[float, float], right: tuple[float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1]
