"""Plant-level geometry dictionary helpers."""

from __future__ import annotations

from typing import Any

from geometry.profiles import summarize_profile
from geometry.utils import optional_positive_float


def _validate_point_count(point_count: int | None) -> int | None:
    if point_count is None:
        return None
    if not isinstance(point_count, int) or isinstance(point_count, bool) or point_count < 0:
        raise ValueError("point_count must be a non-negative integer.")
    return point_count


def make_plant_geometry(
    *,
    height_in: float | None = None,
    width_in: float | None = None,
    density_profile: list[float] | None = None,
    height_profile_widths: list[float] | None = None,
    point_count: int | None = None,
) -> dict[str, Any]:
    """Create a validated plant-level geometry dictionary."""
    return {
        "height_in": optional_positive_float(height_in, name="height_in"),
        "width_in": optional_positive_float(width_in, name="width_in"),
        "density_profile": density_profile or [],
        "height_profile_widths": height_profile_widths or [],
        "density_profile_summary": summarize_profile(density_profile),
        "height_profile_widths_summary": summarize_profile(height_profile_widths),
        "point_count": _validate_point_count(point_count),
    }
