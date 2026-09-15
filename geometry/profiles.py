"""Helpers for summarizing simple geometry profile arrays."""

from __future__ import annotations

from typing import Any

import numpy as np


M_TO_IN = 39.3701


def summarize_profile(values: list[float] | None) -> dict[str, Any]:
    """Summarize a numeric profile list."""
    if not values:
        return {
            "available": False,
            "count": 0,
        }

    return {
        "available": True,
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def profile_peak_index(values: list[float] | None) -> int | None:
    """Return the index of the profile peak, or None when unavailable."""
    if not values:
        return None
    return max(range(len(values)), key=values.__getitem__)


def summarize_point_cloud_profile(cloud: Any, *, bin_count: int = 10) -> dict[str, Any]:
    """Summarize a point cloud for model-ready density features."""
    points = np.asarray(cloud.points)
    if points.size == 0:
        return {
            "available": False,
            "point_count": 0,
            "bbox_length_in": None,
            "bbox_width_in": None,
            "bbox_height_in": None,
            "height_percentiles": {},
            "vertical_density_bins": [],
            "occupied_bin_count": 0,
            "max_bin_density": 0.0,
            "density_center_of_mass_z": None,
            "compactness_point_count_per_bbox_volume": None,
        }

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extents_m = maxs - mins
    horizontal_span_m = float(extents_m[0])
    vertical_span_m = float(extents_m[1])
    depth_thickness_m = float(extents_m[2])
    vertical_values_in = points[:, 1] * M_TO_IN
    hist, _ = np.histogram(points[:, 1], bins=max(1, int(bin_count)), range=(float(mins[1]), float(maxs[1])))
    point_count = int(len(points))
    bins = [float(value / point_count) for value in hist] if point_count else []
    bbox_volume = float(np.prod(extents_m)) * (M_TO_IN**3)
    compactness = float(point_count / bbox_volume) if bbox_volume > 0 else None

    profile = {
        "available": True,
        "point_count": point_count,
        "bbox_length_in": horizontal_span_m * M_TO_IN,
        "bbox_width_in": depth_thickness_m * M_TO_IN,
        "bbox_height_in": vertical_span_m * M_TO_IN,
        "height_percentiles": {
            f"z_p{percentile}": float(np.percentile(vertical_values_in, percentile))
            for percentile in (10, 25, 50, 75, 90, 95, 99)
        },
        "vertical_density_bins": bins,
        "occupied_bin_count": int(np.count_nonzero(hist)),
        "max_bin_density": max(bins) if bins else 0.0,
        "density_center_of_mass_z": float(np.mean(vertical_values_in)),
        "compactness_point_count_per_bbox_volume": compactness,
    }

    thirds: dict[str, dict[str, float | None]] = {}
    y_min = float(mins[1])
    y_span = float(maxs[1] - mins[1])
    for name, low, high in (
        ("lower_third", 0.0, 1.0 / 3.0),
        ("middle_third", 1.0 / 3.0, 2.0 / 3.0),
        ("upper_third", 2.0 / 3.0, 1.0),
    ):
        if y_span <= 0:
            band = points
        else:
            band_mask = (points[:, 1] >= y_min + y_span * low) & (points[:, 1] <= y_min + y_span * high)
            band = points[band_mask]
        if band.size == 0:
            thirds[name] = {"width_in": None, "depth_in": None}
            continue
        band_extents = band.max(axis=0) - band.min(axis=0)
        thirds[name] = {
            "width_in": float(band_extents[0]) * M_TO_IN,
            "depth_in": float(band_extents[2]) * M_TO_IN,
        }
    profile["thirds"] = thirds
    return profile
