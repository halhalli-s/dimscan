"""Scene-level geometry dictionary helpers."""

from __future__ import annotations

from typing import Any

from geometry.utils import optional_positive_float, positive_float


def _validate_point_count(point_count: int | None) -> int | None:
    if point_count is None:
        return None
    if not isinstance(point_count, int) or isinstance(point_count, bool) or point_count < 0:
        raise ValueError("point_count must be a non-negative integer.")
    return point_count


def make_scene_geometry(
    *,
    height_in: float,
    width_in: float,
    length_in: float | None = None,
    point_count: int | None = None,
) -> dict[str, Any]:
    """Create a validated scene-level geometry dictionary."""
    return {
        "height_in": positive_float(height_in, name="height_in"),
        "width_in": positive_float(width_in, name="width_in"),
        "length_in": optional_positive_float(length_in, name="length_in"),
        "point_count": _validate_point_count(point_count),
    }
