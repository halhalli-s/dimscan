"""Pot-level geometry dictionary helpers."""

from __future__ import annotations

from typing import Any

from geometry.utils import optional_positive_float


def _validate_point_count(point_count: int | None) -> int | None:
    if point_count is None:
        return None
    if not isinstance(point_count, int) or isinstance(point_count, bool) or point_count < 0:
        raise ValueError("point_count must be a non-negative integer.")
    return point_count


def make_pot_geometry(
    *,
    visible_top_diameter_in: float | None = None,
    visible_height_in: float | None = None,
    point_count: int | None = None,
) -> dict[str, Any]:
    """Create a validated pot-level geometry dictionary."""
    top_diameter = optional_positive_float(
        visible_top_diameter_in,
        name="visible_top_diameter_in",
    )
    visible_height = optional_positive_float(visible_height_in, name="visible_height_in")

    return {
        "visible_top_diameter_in": top_diameter,
        "visible_top_diameter_available": top_diameter is not None,
        "visible_height_in": visible_height,
        "visible_height_available": visible_height is not None,
        "point_count": _validate_point_count(point_count),
    }
