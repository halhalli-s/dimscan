"""Geometry-primary object extraction for controlled DimScan captures."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.config import DimScanConfig
from capture.pointcloud import write_ascii_ply, write_ascii_ply_with_colors
from pipeline.profiling import add_timing, timed_stage
from utils.io import read_json_if_exists, write_json_atomic


TABLE_DISTANCE_THRESHOLD_M = 0.008
RANSAC_ITERATIONS = 160
CLUSTER_VOXEL_SIZE_M = 0.025
TABLE_PLANE_FIT_MAX_POINTS = 50000
MIN_CLUSTER_POINTS = 30
FRAGMENT_MERGE_GAP_M = 0.08
MAX_FRAGMENT_POINT_RATIO = 0.35
MAX_OBJECT_SPAN_M = 2.0
TABLE_PLANE_CACHE_MIN_INLIERS = 500
TABLE_PLANE_CACHE_MIN_INLIER_RATIO = 0.20
TABLE_PLANE_CACHE_MIN_DOMINANT_SIDE_RATIO = 0.03
TABLE_PLANE_CACHE_MIN_NORMAL_Y_ABS = 0.85
TABLE_PLANE_CACHE_MAX_ABS_OFFSET_M = 2.0


_TABLE_PLANE_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


def reset_table_plane_cache() -> None:
    """Clear the process-local table plane cache."""
    _TABLE_PLANE_CACHE.clear()


def _intrinsic_value(intrinsics: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = intrinsics.get(name)
        if value is not None:
            return float(value)
    return None


def _depth_scale_to_meters(capture_meta: dict[str, Any]) -> tuple[float, str]:
    saved_units = str(capture_meta.get("saved_depth_units") or "meters").lower()
    scale_applied = capture_meta.get("depth_scale_applied_to_saved_depth")
    scale = capture_meta.get("sdk_depth_scale_m_per_unit", capture_meta.get("object_cloud_depth_scale", 1.0))
    scale = float(scale)
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("geometry_primary_depth_scale_invalid")
    if saved_units == "meters":
        if scale_applied is not True:
            raise ValueError("geometry_primary_depth_scale_state_invalid")
        return scale, saved_units
    if saved_units in {"raw_sdk_unit", "raw_sdk_units"}:
        if scale_applied is not False:
            raise ValueError("geometry_primary_depth_scale_state_invalid")
        if scale >= 0.1:
            raise ValueError("geometry_primary_depth_scale_not_meters_per_unit")
        return scale, saved_units
    if saved_units in {"millimeter", "millimeters", "mm"}:
        if scale_applied is not False:
            raise ValueError("geometry_primary_depth_scale_state_invalid")
        return scale * 0.001, saved_units
    raise ValueError(f"geometry_primary_depth_units_unsupported:{saved_units}")


def full_scene_cloud_from_aligned_depth(
    *,
    aligned_depth: np.ndarray,
    rgb_intrinsics: dict[str, Any],
    depth_scale_to_meters: float,
    rgb_image: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Project every valid RGB-grid aligned-depth pixel into RGB-camera meters."""
    depth = np.asarray(aligned_depth, dtype=float)
    if depth.ndim != 2:
        raise ValueError(f"aligned_depth_must_be_2d:{depth.shape}")
    fx = _intrinsic_value(rgb_intrinsics, ("fx", "focal_length_x"))
    fy = _intrinsic_value(rgb_intrinsics, ("fy", "focal_length_y"))
    cx = _intrinsic_value(rgb_intrinsics, ("cx", "principal_point_x", "ppx"))
    cy = _intrinsic_value(rgb_intrinsics, ("cy", "principal_point_y", "ppy"))
    if fx is None or fy is None or cx is None or cy is None or fx == 0 or fy == 0:
        raise ValueError("rgb_intrinsics_invalid_for_geometry_primary_cloud")
    valid = np.isfinite(depth) & (depth > 0)
    rows, cols = np.nonzero(valid)
    z = depth[rows, cols] * float(depth_scale_to_meters)
    x = (cols.astype(float) - float(cx)) * z / float(fx)
    y = (rows.astype(float) - float(cy)) * z / float(fy)
    points = np.column_stack((x, y, z)) if z.size else np.empty((0, 3), dtype=float)
    colors = None
    if rgb_image is not None:
        rgb = np.asarray(rgb_image, dtype=np.uint8)
        if rgb.shape[:2] != depth.shape:
            raise ValueError(f"rgb_aligned_depth_shape_mismatch:rgb={rgb.shape[:2]}:depth={depth.shape}")
        colors = rgb[rows, cols, :3].astype(np.uint8) if rows.size else np.empty((0, 3), dtype=np.uint8)
    return points, colors


def _stats(points: np.ndarray) -> dict[str, Any]:
    if points.size == 0:
        return {"point_count": 0, "xyz_min": None, "xyz_max": None, "xyz_spans": None}
    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    return {
        "point_count": int(len(points)),
        "xyz_min": [float(v) for v in xyz_min],
        "xyz_max": [float(v) for v in xyz_max],
        "xyz_spans": [float(v) for v in (xyz_max - xyz_min)],
    }


def apply_calibrated_roi(
    points: np.ndarray,
    colors: np.ndarray | None,
    cfg: DimScanConfig,
    *,
    keep_rejected: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None, dict[str, Any]]:
    if str(getattr(cfg, "roi_coordinate_frame", "")) != "rgb_camera":
        raise ValueError(f"geometry_primary_roi_frame_unsupported:{getattr(cfg, 'roi_coordinate_frame', None)}")
    if str(getattr(cfg, "roi_units", "")) != "meters":
        raise ValueError(f"geometry_primary_roi_units_unsupported:{getattr(cfg, 'roi_units', None)}")
    margin = float(cfg.roi_margin_m or 0.0)
    bounds = {
        "min_x_m": -float(cfg.roi_half_width_m) - margin,
        "max_x_m": float(cfg.roi_half_width_m) + margin,
        "min_y_m": cfg.roi_min_height_m,
        "max_y_m": cfg.roi_max_height_m,
        "min_z_m": float(cfg.roi_min_depth_m) - margin,
        "max_z_m": float(cfg.roi_max_depth_m) + margin,
        "margin_m": margin,
    }
    inside = (
        (points[:, 0] >= float(bounds["min_x_m"]))
        & (points[:, 0] <= float(bounds["max_x_m"]))
        & (points[:, 2] >= float(bounds["min_z_m"]))
        & (points[:, 2] <= float(bounds["max_z_m"]))
    )
    if bounds["min_y_m"] is not None:
        inside &= points[:, 1] >= float(bounds["min_y_m"]) - margin
    if bounds["max_y_m"] is not None:
        inside &= points[:, 1] <= float(bounds["max_y_m"]) + margin
    kept = points[inside]
    kept_colors = colors[inside] if colors is not None else None
    rejected_count = int(len(points) - len(kept))
    if keep_rejected:
        rejected = points[~inside]
        rejected_colors = colors[~inside] if colors is not None else None
    else:
        rejected = np.empty((0, 3), dtype=points.dtype)
        rejected_colors = np.empty((0, 3), dtype=colors.dtype) if colors is not None else None
    return kept, kept_colors, rejected, rejected_colors, {
        "roi_frame": "rgb_camera",
        "roi_units": "meters",
        "roi_bounds": bounds,
        "point_count_before_roi": int(len(points)),
        "point_count_after_roi": int(len(kept)),
        "point_count_rejected_by_roi": rejected_count,
    }


def _plane_from_points(points: np.ndarray) -> np.ndarray | None:
    p1, p2, p3 = points
    normal = np.cross(p2 - p1, p3 - p1)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-9:
        return None
    normal = normal / norm
    d = -float(np.dot(normal, p1))
    return np.asarray([normal[0], normal[1], normal[2], d], dtype=float)


def _cache_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((str(key), _cache_value(child)) for key, child in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_cache_value(child) for child in value)
    if isinstance(value, float):
        return round(value, 9)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _table_plane_cache_key(
    cfg: DimScanConfig,
    capture_meta: dict[str, Any],
    *,
    distance_threshold_m: float,
    plane_fit_voxel_size_m: float | None,
    plane_fit_max_points: int | None,
) -> tuple[Any, ...]:
    sdk_device = capture_meta.get("sdk_device")
    if isinstance(sdk_device, dict):
        camera_identity = (
            sdk_device.get("serial_number")
            or sdk_device.get("serial")
            or sdk_device.get("device_serial")
            or sdk_device.get("uid")
            or _cache_value(sdk_device)
        )
    else:
        camera_identity = sdk_device or capture_meta.get("camera_serial") or capture_meta.get("camera_type")
    return (
        ("camera_identity", _cache_value(camera_identity)),
        ("camera_type", _cache_value(capture_meta.get("camera_type"))),
        ("color_profile", _cache_value(capture_meta.get("color_profile"))),
        ("depth_profile", _cache_value(capture_meta.get("depth_profile"))),
        ("rgb_size", (capture_meta.get("rgb_width"), capture_meta.get("rgb_height"))),
        ("raw_depth_size", (capture_meta.get("depth_width"), capture_meta.get("depth_height"))),
        ("aligned_depth_size", (capture_meta.get("aligned_depth_width"), capture_meta.get("aligned_depth_height"))),
        ("rgb_intrinsics", _cache_value(capture_meta.get("rgb_intrinsics"))),
        ("roi_frame", getattr(cfg, "roi_coordinate_frame", None)),
        ("roi_units", getattr(cfg, "roi_units", None)),
        ("roi_min_depth_m", _cache_value(getattr(cfg, "roi_min_depth_m", None))),
        ("roi_max_depth_m", _cache_value(getattr(cfg, "roi_max_depth_m", None))),
        ("roi_half_width_m", _cache_value(getattr(cfg, "roi_half_width_m", None))),
        ("roi_min_height_m", _cache_value(getattr(cfg, "roi_min_height_m", None))),
        ("roi_max_height_m", _cache_value(getattr(cfg, "roi_max_height_m", None))),
        ("roi_margin_m", _cache_value(getattr(cfg, "roi_margin_m", None))),
        ("distance_threshold_m", _cache_value(distance_threshold_m)),
        ("plane_fit_voxel_size_m", _cache_value(plane_fit_voxel_size_m)),
        ("plane_fit_max_points", _cache_value(plane_fit_max_points)),
    )


def _normalized_plane(plane: Any) -> np.ndarray | None:
    values = np.asarray(plane, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        return None
    normal_norm = float(np.linalg.norm(values[:3]))
    if abs(normal_norm - 1.0) > 1e-3 or normal_norm <= 1e-9:
        return None
    return values.copy()


def _apply_table_plane(
    points: np.ndarray,
    colors: np.ndarray | None,
    plane: np.ndarray,
    *,
    distance_threshold_m: float,
    keep_table_points: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None, dict[str, Any]]:
    apply_started = time.perf_counter()
    signed = points @ plane[:3] + plane[3]
    full_inliers = np.abs(signed) <= distance_threshold_m
    positive = signed > distance_threshold_m
    negative = signed < -distance_threshold_m
    if np.count_nonzero(positive) >= np.count_nonzero(negative):
        object_mask = positive
        above_sign = 1
    else:
        object_mask = negative
        above_sign = -1
    object_points = points[object_mask]
    object_colors = colors[object_mask] if colors is not None else None
    if keep_table_points:
        table_points = points[full_inliers]
        table_colors = colors[full_inliers] if colors is not None else None
    else:
        table_points = np.empty((0, 3), dtype=points.dtype)
        table_colors = np.empty((0, 3), dtype=colors.dtype) if colors is not None else None
    apply_s = time.perf_counter() - apply_started
    return object_points, object_colors, table_points, table_colors, {
        "plane_coefficients": [float(v) for v in plane],
        "plane_model": [float(v) for v in plane],
        "full_resolution_inlier_count": int(np.count_nonzero(full_inliers)),
        "distance_threshold_m": float(distance_threshold_m),
        "above_table_side_sign": above_sign,
        "point_count_before_table_removal": int(len(points)),
        "point_count_after_table_removal": int(len(object_points)),
        "plane_applied_to_full_resolution_points": True,
        "full_resolution_apply_s": apply_s,
        "positive_side_count": int(np.count_nonzero(positive)),
        "negative_side_count": int(np.count_nonzero(negative)),
    }


def _validate_table_plane(
    plane: Any,
    points: np.ndarray,
    *,
    distance_threshold_m: float,
) -> tuple[bool, dict[str, Any]]:
    normalized = _normalized_plane(plane)
    if normalized is None:
        return False, {"valid": False, "reason": "plane_coefficients_invalid_or_not_normalized"}
    if abs(float(normalized[1])) < TABLE_PLANE_CACHE_MIN_NORMAL_Y_ABS:
        return False, {
            "valid": False,
            "reason": "plane_normal_orientation_invalid",
            "normal_y": float(normalized[1]),
            "min_abs_normal_y": TABLE_PLANE_CACHE_MIN_NORMAL_Y_ABS,
        }
    if abs(float(normalized[3])) > TABLE_PLANE_CACHE_MAX_ABS_OFFSET_M:
        return False, {
            "valid": False,
            "reason": "plane_offset_implausible",
            "offset_m": float(normalized[3]),
            "max_abs_offset_m": TABLE_PLANE_CACHE_MAX_ABS_OFFSET_M,
        }
    if len(points) < 3:
        return False, {"valid": False, "reason": "not_enough_points_for_cache_validation"}
    signed = points @ normalized[:3] + normalized[3]
    if not np.isfinite(signed).all():
        return False, {"valid": False, "reason": "signed_distances_invalid"}
    inliers = np.abs(signed) <= distance_threshold_m
    positive = signed > distance_threshold_m
    negative = signed < -distance_threshold_m
    inlier_count = int(np.count_nonzero(inliers))
    positive_count = int(np.count_nonzero(positive))
    negative_count = int(np.count_nonzero(negative))
    dominant_side_count = max(positive_count, negative_count)
    inlier_ratio = float(inlier_count / len(points))
    dominant_side_ratio = float(dominant_side_count / len(points))
    validation = {
        "valid": True,
        "reason": None,
        "inlier_count": inlier_count,
        "inlier_ratio": inlier_ratio,
        "positive_side_count": positive_count,
        "negative_side_count": negative_count,
        "dominant_side_count": dominant_side_count,
        "dominant_side_ratio": dominant_side_ratio,
        "min_inliers": TABLE_PLANE_CACHE_MIN_INLIERS,
        "min_inlier_ratio": TABLE_PLANE_CACHE_MIN_INLIER_RATIO,
        "min_dominant_side_ratio": TABLE_PLANE_CACHE_MIN_DOMINANT_SIDE_RATIO,
        "min_remaining_non_table_points": MIN_CLUSTER_POINTS,
    }
    if inlier_count < TABLE_PLANE_CACHE_MIN_INLIERS:
        validation.update({"valid": False, "reason": "near_plane_inlier_count_too_low"})
        return False, validation
    if inlier_ratio < TABLE_PLANE_CACHE_MIN_INLIER_RATIO:
        validation.update({"valid": False, "reason": "near_plane_inlier_ratio_too_low"})
        return False, validation
    if dominant_side_count < MIN_CLUSTER_POINTS:
        validation.update({"valid": False, "reason": "remaining_non_table_points_too_low"})
        return False, validation
    if dominant_side_ratio < TABLE_PLANE_CACHE_MIN_DOMINANT_SIDE_RATIO:
        validation.update({"valid": False, "reason": "above_below_distribution_implausible"})
        return False, validation
    return True, validation


def _voxel_downsample_centroids(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    if len(points) == 0:
        return points
    voxels = np.floor(points / float(voxel_size_m)).astype(np.int64)
    unique_voxels, inverse = np.unique(voxels, axis=0, return_inverse=True)
    if len(unique_voxels) == len(points):
        return points
    sums = np.zeros((len(unique_voxels), 3), dtype=np.float64)
    counts = np.bincount(inverse, minlength=len(unique_voxels)).astype(np.float64)
    np.add.at(sums, inverse, points.astype(np.float64, copy=False))
    return (sums / counts[:, None]).astype(points.dtype, copy=False)


def _plane_fit_points(
    points: np.ndarray,
    *,
    voxel_size_m: float | None,
    max_points: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    fit_points = points
    method = "full_resolution"
    if voxel_size_m is not None and voxel_size_m > 0:
        downsampled = _voxel_downsample_centroids(fit_points, voxel_size_m)
        if len(downsampled) >= 3 and len(downsampled) < len(fit_points):
            fit_points = downsampled
            method = "voxel_centroid"
    if max_points is not None and int(max_points) > 0 and len(fit_points) > int(max_points):
        indices = np.linspace(0, len(fit_points) - 1, num=int(max_points), dtype=np.int64)
        fit_points = fit_points[indices]
        method = f"{method}+deterministic_stride_sample" if method != "full_resolution" else "deterministic_stride_sample"
    return fit_points, {
        "plane_fit_method": method,
        "plane_fit_voxel_size_m": float(voxel_size_m) if voxel_size_m is not None else None,
        "plane_fit_max_points": int(max_points) if max_points is not None else None,
        "plane_fit_point_count": int(len(fit_points)),
        "full_point_count": int(len(points)),
        "downsample_ratio": float(len(fit_points) / len(points)) if len(points) else 0.0,
    }


def _fit_table_plane(
    points: np.ndarray,
    *,
    distance_threshold_m: float,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    best_plane = None
    best_inliers: np.ndarray | None = None
    sample_count = min(iterations, max(1, len(points) * 2))
    for _ in range(sample_count):
        sample_indices = rng.choice(len(points), size=3, replace=False)
        plane = _plane_from_points(points[sample_indices])
        if plane is None:
            continue
        distances = np.abs(points @ plane[:3] + plane[3])
        inliers = distances <= distance_threshold_m
        if best_inliers is None or int(np.count_nonzero(inliers)) > int(np.count_nonzero(best_inliers)):
            best_plane = plane
            best_inliers = inliers
    if best_plane is None or best_inliers is None:
        raise ValueError("geometry_primary_table_plane_ransac_failed")
    inlier_count = int(np.count_nonzero(best_inliers))
    if inlier_count < max(30, int(0.03 * len(points))):
        raise ValueError("geometry_primary_table_plane_too_few_inliers")
    return best_plane, best_inliers


def remove_table_plane(
    points: np.ndarray,
    colors: np.ndarray | None,
    *,
    distance_threshold_m: float = TABLE_DISTANCE_THRESHOLD_M,
    iterations: int = RANSAC_ITERATIONS,
    seed: int = 7,
    keep_table_points: bool = True,
    plane_fit_voxel_size_m: float | None = None,
    plane_fit_max_points: int | None = TABLE_PLANE_FIT_MAX_POINTS,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None, dict[str, Any]]:
    if len(points) < 3:
        raise ValueError("geometry_primary_table_plane_not_enough_points")
    fit_started = time.perf_counter()
    fit_points, fit_debug = _plane_fit_points(
        points,
        voxel_size_m=plane_fit_voxel_size_m,
        max_points=plane_fit_max_points,
    )
    fallback_reason = None
    try:
        best_plane, fit_inliers = _fit_table_plane(
            fit_points,
            distance_threshold_m=distance_threshold_m,
            iterations=iterations,
            seed=seed,
        )
    except ValueError as exc:
        fallback_reason = str(exc)
        fit_points, fit_debug = _plane_fit_points(points, voxel_size_m=None, max_points=None)
        best_plane, fit_inliers = _fit_table_plane(
            fit_points,
            distance_threshold_m=distance_threshold_m,
            iterations=iterations,
            seed=seed,
        )
    fit_s = time.perf_counter() - fit_started
    object_points, object_colors, table_points, table_colors, apply_debug = _apply_table_plane(
        points,
        colors,
        best_plane,
        distance_threshold_m=distance_threshold_m,
        keep_table_points=keep_table_points,
    )
    return object_points, object_colors, table_points, table_colors, {
        "ransac_succeeded": True,
        **apply_debug,
        "inlier_count": int(np.count_nonzero(fit_inliers)),
        "full_point_count": fit_debug["full_point_count"],
        "plane_fit_point_count": fit_debug["plane_fit_point_count"],
        "downsample_ratio": fit_debug["downsample_ratio"],
        "plane_fit_method": fit_debug["plane_fit_method"],
        "plane_fit_voxel_size_m": fit_debug["plane_fit_voxel_size_m"],
        "plane_fit_max_points": fit_debug["plane_fit_max_points"],
        "plane_fit_s": fit_s,
        "fallback_to_full_resolution_reason": fallback_reason,
    }


def remove_table_plane_with_cache(
    points: np.ndarray,
    colors: np.ndarray | None,
    cfg: DimScanConfig,
    capture_meta: dict[str, Any],
    *,
    distance_threshold_m: float = TABLE_DISTANCE_THRESHOLD_M,
    iterations: int = RANSAC_ITERATIONS,
    seed: int = 7,
    keep_table_points: bool = True,
    plane_fit_voxel_size_m: float | None = None,
    plane_fit_max_points: int | None = TABLE_PLANE_FIT_MAX_POINTS,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Remove table points, using a strictly validated process-local plane cache when safe."""
    lookup_started = time.perf_counter()
    cache_key = _table_plane_cache_key(
        cfg,
        capture_meta,
        distance_threshold_m=distance_threshold_m,
        plane_fit_voxel_size_m=plane_fit_voxel_size_m,
        plane_fit_max_points=plane_fit_max_points,
    )
    cached = _TABLE_PLANE_CACHE.get(cache_key)
    lookup_s = time.perf_counter() - lookup_started
    add_timing("table_plane_cache_lookup_s", lookup_s)

    cache_validation: dict[str, Any] = {
        "cache_key_found": cached is not None,
        "valid": False,
        "reason": "cache_miss" if cached is None else None,
    }
    if isinstance(cached, dict):
        validation_started = time.perf_counter()
        cache_ok, cache_validation = _validate_table_plane(
            cached.get("plane_model"),
            points,
            distance_threshold_m=distance_threshold_m,
        )
        validation_s = time.perf_counter() - validation_started
        cache_validation["validation_s"] = validation_s
        add_timing("table_plane_cache_validation_s", validation_s)
        if cache_ok:
            apply_started = time.perf_counter()
            plane = np.asarray(cached["plane_model"], dtype=float)
            object_points, object_colors, table_points, table_colors, apply_debug = _apply_table_plane(
                points,
                colors,
                plane,
                distance_threshold_m=distance_threshold_m,
                keep_table_points=keep_table_points,
            )
            apply_s = time.perf_counter() - apply_started
            add_timing("table_plane_cached_apply_s", apply_s)
            add_timing("table_plane_cache_hit", 1.0)
            add_timing("table_plane_cache_replaced", 0.0)
            return object_points, object_colors, table_points, table_colors, {
                "ransac_succeeded": False,
                **apply_debug,
                "inlier_count": int(cache_validation.get("inlier_count") or 0),
                "full_point_count": int(len(points)),
                "plane_fit_point_count": 0,
                "downsample_ratio": 0.0,
                "plane_fit_method": "process_local_cache",
                "plane_fit_voxel_size_m": float(plane_fit_voxel_size_m) if plane_fit_voxel_size_m is not None else None,
                "plane_fit_max_points": int(plane_fit_max_points) if plane_fit_max_points is not None else None,
                "plane_fit_s": 0.0,
                "fallback_to_full_resolution_reason": None,
                "cached_plane_used": True,
                "plane_source": "process_local_cache",
                "cache_validation": cache_validation,
                "cache_lookup_s": lookup_s,
                "cache_validation_s": validation_s,
                "cached_apply_s": apply_s,
                "cache_entry_created_at": cached.get("created_at"),
            }
    else:
        add_timing("table_plane_cache_validation_s", 0.0)

    fallback_started = time.perf_counter()
    object_points, object_colors, table_points, table_colors, table_debug = remove_table_plane(
        points,
        colors,
        distance_threshold_m=distance_threshold_m,
        iterations=iterations,
        seed=seed,
        keep_table_points=keep_table_points,
        plane_fit_voxel_size_m=plane_fit_voxel_size_m,
        plane_fit_max_points=plane_fit_max_points,
    )
    fallback_s = time.perf_counter() - fallback_started
    add_timing("table_plane_ransac_fallback_s", fallback_s)
    add_timing("table_plane_cache_hit", 0.0)

    replacement_started = time.perf_counter()
    replace_ok, replacement_validation = _validate_table_plane(
        table_debug.get("plane_model"),
        points,
        distance_threshold_m=distance_threshold_m,
    )
    replacement_validation_s = time.perf_counter() - replacement_started
    replacement_validation["validation_s"] = replacement_validation_s
    add_timing("table_plane_cache_validation_s", replacement_validation_s)
    replaced = bool(replace_ok and table_debug.get("ransac_succeeded") is True)
    if replaced:
        _TABLE_PLANE_CACHE[cache_key] = {
            "plane_model": list(table_debug["plane_model"]),
            "created_at": time.time(),
            "validation": replacement_validation,
            "key": cache_key,
        }
    add_timing("table_plane_cache_replaced", 1.0 if replaced else 0.0)
    table_debug.update(
        {
            "cached_plane_used": False,
            "plane_source": "ransac",
            "cache_lookup_s": lookup_s,
            "cache_validation": cache_validation,
            "cache_replacement_validation": replacement_validation,
            "cache_replaced": replaced,
            "ransac_fallback_s": fallback_s,
        }
    )
    return object_points, object_colors, table_points, table_colors, table_debug


def _bbox_gap(first: np.ndarray, second: np.ndarray) -> float:
    dx = max(0.0, float(max(first[0, 0] - second[1, 0], second[0, 0] - first[1, 0])))
    dy = max(0.0, float(max(first[0, 1] - second[1, 1], second[0, 1] - first[1, 1])))
    dz = max(0.0, float(max(first[0, 2] - second[1, 2], second[0, 2] - first[1, 2])))
    return float((dx * dx + dy * dy + dz * dz) ** 0.5)


def voxel_cluster_points(
    points: np.ndarray,
    *,
    voxel_size_m: float = CLUSTER_VOXEL_SIZE_M,
    min_points: int = MIN_CLUSTER_POINTS,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    if len(points) == 0:
        return np.zeros((0,), dtype=int), [], {"algorithm": "voxel_connected_components", "cluster_count": 0}
    voxels = np.floor(points / float(voxel_size_m)).astype(np.int64)
    unique_voxels, inverse, voxel_counts = np.unique(voxels, axis=0, return_inverse=True, return_counts=True)
    voxel_lookup = {tuple(voxel.tolist()): index for index, voxel in enumerate(unique_voxels)}
    voxel_labels = np.full(len(unique_voxels), -1, dtype=int)
    clusters: list[list[int]] = []
    label = 0
    for start in range(len(unique_voxels)):
        if voxel_labels[start] >= 0:
            continue
        stack = [start]
        voxel_labels[start] = label
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            base = unique_voxels[current]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        neighbor = voxel_lookup.get((int(base[0] + dx), int(base[1] + dy), int(base[2] + dz)))
                        if neighbor is not None and voxel_labels[neighbor] < 0:
                            voxel_labels[neighbor] = label
                            stack.append(neighbor)
        clusters.append(component)
        label += 1
    point_labels = voxel_labels[inverse]
    records: list[dict[str, Any]] = []
    for cluster_index, component in enumerate(clusters):
        point_indices = np.where(point_labels == cluster_index)[0]
        count = int(len(point_indices))
        if count < min_points:
            continue
        cluster_points = points[point_indices]
        xyz_min = cluster_points.min(axis=0)
        xyz_max = cluster_points.max(axis=0)
        records.append(
            {
                "cluster_index": int(cluster_index),
                "point_count": count,
                "voxel_count": int(len(component)),
                "centroid": [float(v) for v in cluster_points.mean(axis=0)],
                "bbox_min": [float(v) for v in xyz_min],
                "bbox_max": [float(v) for v in xyz_max],
                "bbox_spans": [float(v) for v in (xyz_max - xyz_min)],
            }
        )
    records.sort(key=lambda item: int(item["point_count"]), reverse=True)
    debug = {
        "algorithm": "voxel_connected_components",
        "voxel_size_m": float(voxel_size_m),
        "min_cluster_points": int(min_points),
        "cluster_count": len(records),
        "raw_component_count": len(clusters),
        "cluster_point_counts": [record["point_count"] for record in records],
    }
    return point_labels, records, debug


def select_and_merge_clusters(
    points: np.ndarray,
    point_labels: np.ndarray,
    clusters: list[dict[str, Any]],
    cfg: DimScanConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not clusters:
        raise ValueError("geometry_primary_no_clusters_after_table_removal")
    half_width = float(cfg.roi_half_width_m)
    central_limit = half_width * 0.60
    plausible = []
    rejected: list[dict[str, Any]] = []
    for record in clusters:
        centroid = record["centroid"]
        spans = record["bbox_spans"]
        if abs(float(centroid[0])) > central_limit:
            rejected.append({"cluster_index": record["cluster_index"], "reason": "centroid_outside_expected_object_zone"})
            continue
        if max(float(v) for v in spans) > MAX_OBJECT_SPAN_M:
            rejected.append({"cluster_index": record["cluster_index"], "reason": "physically_implausible_span"})
            continue
        center_penalty = abs(float(centroid[0])) / max(0.01, half_width)
        score = float(record["point_count"]) / (1.0 + center_penalty)
        plausible.append((score, record))
    if not plausible:
        raise ValueError("geometry_primary_no_plausible_central_cluster")
    primary = max(plausible, key=lambda item: item[0])[1]
    primary_bbox = np.asarray([primary["bbox_min"], primary["bbox_max"]], dtype=float)
    primary_count = int(primary["point_count"])
    selected_indices = [int(primary["cluster_index"])]
    merged_indices: list[int] = []
    for _, record in sorted(plausible, key=lambda item: item[1]["point_count"], reverse=True):
        cluster_index = int(record["cluster_index"])
        if cluster_index == int(primary["cluster_index"]):
            continue
        count = int(record["point_count"])
        bbox = np.asarray([record["bbox_min"], record["bbox_max"]], dtype=float)
        gap = _bbox_gap(primary_bbox, bbox)
        union_bbox = np.asarray([np.minimum(primary_bbox[0], bbox[0]), np.maximum(primary_bbox[1], bbox[1])])
        union_span = union_bbox[1] - union_bbox[0]
        if count > primary_count * MAX_FRAGMENT_POINT_RATIO:
            rejected.append({"cluster_index": cluster_index, "reason": "large_separate_cluster", "bbox_gap_m": gap})
            continue
        if gap > FRAGMENT_MERGE_GAP_M:
            rejected.append({"cluster_index": cluster_index, "reason": "fragment_too_far_from_primary", "bbox_gap_m": gap})
            continue
        if float(np.max(union_span)) > MAX_OBJECT_SPAN_M:
            rejected.append({"cluster_index": cluster_index, "reason": "merge_would_make_bounds_implausible", "bbox_gap_m": gap})
            continue
        selected_indices.append(cluster_index)
        merged_indices.append(cluster_index)
        primary_bbox = union_bbox
    selected_mask = np.isin(point_labels, np.asarray(selected_indices, dtype=int))
    debug = {
        "selected_primary_cluster": int(primary["cluster_index"]),
        "selected_cluster_indices": selected_indices,
        "merged_fragment_indices": merged_indices,
        "rejected_clusters": rejected,
        "primary_cluster": primary,
        "selection_policy": {
            "dominant_cluster_score": "point_count_penalized_by_lateral_distance_from_roi_center",
            "max_fragment_point_ratio": MAX_FRAGMENT_POINT_RATIO,
            "fragment_merge_gap_m": FRAGMENT_MERGE_GAP_M,
            "max_object_span_m": MAX_OBJECT_SPAN_M,
            "central_lateral_limit_m": central_limit,
        },
    }
    return selected_mask, debug


def _as_tuples(points: np.ndarray) -> list[tuple[float, float, float]]:
    return [tuple(float(v) for v in row) for row in np.asarray(points, dtype=float).reshape((-1, 3))]


def _color_tuples(colors: np.ndarray | None) -> list[tuple[int, int, int]] | None:
    if colors is None:
        return None
    return [tuple(int(v) for v in row) for row in np.asarray(colors, dtype=np.uint8).reshape((-1, 3))]


def _write_cloud(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    color_rows = _color_tuples(colors)
    if color_rows is not None and len(color_rows) == len(points):
        write_ascii_ply_with_colors(path, _as_tuples(points), color_rows)
    else:
        write_ascii_ply(path, _as_tuples(points))


def _robust_x_width_line_debug(
    points: np.ndarray,
    colors: np.ndarray | None,
    *,
    line_points: int = 512,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build a visual-only line for the same 1/99 robust X span used by geometry."""
    object_points = np.asarray(points, dtype=float).reshape((-1, 3))
    if len(object_points) < 10:
        x_left = float(np.min(object_points[:, 0]))
        x_right = float(np.max(object_points[:, 0]))
        endpoint_source = "min_max_for_under_10_points_matches_geometry_robust_extent"
    else:
        x_left, x_right = (float(v) for v in np.percentile(object_points[:, 0], [1.0, 99.0]))
        endpoint_source = "np.percentile(final_points[:, 0], [1.0, 99.0])"

    def endpoint_yz(x_value: float) -> tuple[float, float, int]:
        x_values = object_points[:, 0]
        tolerance = max((x_right - x_left) * 0.01, 0.001)
        mask = np.abs(x_values - x_value) <= tolerance
        if not np.any(mask):
            distances = np.abs(x_values - x_value)
            nearest_count = min(max(8, len(object_points) // 200), len(object_points))
            nearest_indices = np.argpartition(distances, nearest_count - 1)[:nearest_count]
            mask = np.zeros(len(object_points), dtype=bool)
            mask[nearest_indices] = True
        side_points = object_points[mask]
        return float(np.median(side_points[:, 1])), float(np.median(side_points[:, 2])), int(len(side_points))

    y_left, z_left, left_samples = endpoint_yz(x_left)
    y_right, z_right, right_samples = endpoint_yz(x_right)
    representative_y = float(np.median([y_left, y_right]))
    representative_z = float(np.median([z_left, z_right]))
    count = max(2, int(line_points))
    line = np.column_stack(
        [
            np.linspace(x_left, x_right, count),
            np.full(count, representative_y),
            np.full(count, representative_z),
        ]
    )
    line_colors = np.tile(np.asarray([[255, 0, 255]], dtype=np.uint8), (count, 1))
    object_colors = colors
    if object_colors is None or len(object_colors) != len(object_points):
        object_colors = np.full((len(object_points), 3), 160, dtype=np.uint8)
    else:
        object_colors = np.asarray(object_colors, dtype=np.uint8).reshape((-1, 3))
    combined_points = np.vstack([object_points, line])
    combined_colors = np.vstack([object_colors, line_colors])
    debug = {
        "purpose": "debug_visualization_only_not_authoritative_geometry",
        "endpoint_source": endpoint_source,
        "placement": (
            "Y and Z use the median of the left-endpoint and right-endpoint neighborhood medians; "
            "the debug line holds those values constant while X spans the robust endpoints"
        ),
        "color_rgb": [255, 0, 255],
        "line_point_count": int(count),
        "left_endpoint_m": [x_left, representative_y, representative_z],
        "right_endpoint_m": [x_right, representative_y, representative_z],
        "represented_width_m": float(max(x_right - x_left, 0.0)),
        "represented_width_in": float(max(x_right - x_left, 0.0) * 39.3701),
        "left_endpoint_neighborhood_median_yz_m": [y_left, z_left],
        "right_endpoint_neighborhood_median_yz_m": [y_right, z_right],
        "left_endpoint_sample_count": left_samples,
        "right_endpoint_sample_count": right_samples,
    }
    return combined_points, combined_colors, debug


def extract_geometry_primary_object_cloud(
    cfg: DimScanConfig,
    view_dir: str | Path,
    *,
    debug_mode: bool = True,
    save_object_cloud: bool = True,
) -> dict[str, Any]:
    """Write final object_cloud.ply from ROI, table removal, and dominant 3D clustering."""
    started = time.perf_counter()
    view_path = Path(view_dir)
    debug_dir = view_path / "debug"
    rejected_dir = debug_dir / "rejected_clusters"
    if debug_mode:
        debug_dir.mkdir(parents=True, exist_ok=True)
        rejected_dir.mkdir(parents=True, exist_ok=True)
    timing: dict[str, float] = {}

    stage_started = time.perf_counter()
    capture_meta = read_json_if_exists(view_path / "capture_meta.json", default={})
    if not isinstance(capture_meta, dict):
        capture_meta = {}
    rgb_intrinsics = capture_meta.get("rgb_intrinsics")
    if not isinstance(rgb_intrinsics, dict):
        raise ValueError("geometry_primary_missing_rgb_intrinsics")
    aligned_path = view_path / getattr(cfg, "depth_aligned_to_rgb_filename", "depth_aligned_to_rgb.npy")
    if not aligned_path.is_file():
        raise ValueError("geometry_primary_missing_depth_aligned_to_rgb")
    timing["setup_s"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    aligned_depth = np.load(aligned_path)
    timing["aligned_depth_load_s"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    rgb_image = np.asarray(Image.open(view_path / getattr(cfg, "rgb_filename", "rgb.png")).convert("RGB"), dtype=np.uint8)
    timing["rgb_image_decode_s"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    scale_to_meters, saved_units = _depth_scale_to_meters(capture_meta)
    points, colors = full_scene_cloud_from_aligned_depth(
        aligned_depth=aligned_depth,
        rgb_intrinsics=rgb_intrinsics,
        depth_scale_to_meters=scale_to_meters,
        rgb_image=rgb_image,
    )
    timing["cloud_creation_s"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    roi_points, roi_colors, rejected_roi_points, rejected_roi_colors, roi_debug = apply_calibrated_roi(
        points,
        colors,
        cfg,
        keep_rejected=debug_mode,
    )
    if len(roi_points) < MIN_CLUSTER_POINTS:
        raise ValueError(f"geometry_primary_roi_too_few_points:{len(roi_points)}")
    if debug_mode:
        _write_cloud(debug_dir / "object_cloud_roi_all_points.ply", roi_points, roi_colors)
    if debug_mode and len(rejected_roi_points):
        _write_cloud(debug_dir / "object_cloud_rejected_by_roi.ply", rejected_roi_points, rejected_roi_colors)
    timing["roi_crop_s"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    non_table_points, non_table_colors, table_points, table_colors, table_debug = remove_table_plane_with_cache(
        roi_points,
        roi_colors,
        cfg,
        capture_meta,
        keep_table_points=debug_mode,
        plane_fit_voxel_size_m=getattr(cfg, "table_plane_fit_voxel_size_m", None),
        plane_fit_max_points=getattr(cfg, "table_plane_fit_max_points", TABLE_PLANE_FIT_MAX_POINTS),
    )
    if len(non_table_points) < MIN_CLUSTER_POINTS:
        raise ValueError(f"geometry_primary_table_removal_too_few_points:{len(non_table_points)}")
    if debug_mode:
        _write_cloud(debug_dir / "object_cloud_after_table_removal.ply", non_table_points, non_table_colors)
        _write_cloud(debug_dir / "table_plane.ply", table_points, table_colors)
    timing["table_removal_s"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    labels, clusters, clustering_debug = voxel_cluster_points(non_table_points)
    timing["clustering_s"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    selected_mask, selection_debug = select_and_merge_clusters(non_table_points, labels, clusters, cfg)
    final_points = non_table_points[selected_mask]
    final_colors = non_table_colors[selected_mask] if non_table_colors is not None else None
    if len(final_points) < MIN_CLUSTER_POINTS:
        raise ValueError(f"geometry_primary_final_object_too_few_points:{len(final_points)}")
    output_path = view_path / "object_cloud.ply"
    object_write_started = time.perf_counter()
    if save_object_cloud:
        with timed_stage("object_cloud_ply_write_s"):
            _write_cloud(output_path, final_points, final_colors)
        object_cloud_write_s = time.perf_counter() - object_write_started
    else:
        add_timing("object_cloud_ply_write_s", 0.0)
        object_cloud_write_s = 0.0
    timing["object_cloud_ply_write_s"] = object_cloud_write_s
    width_line_path = debug_dir / "object_cloud_width_line.ply"
    width_line_debug: dict[str, Any] | None = None
    width_line_write_started = time.perf_counter()
    if debug_mode and save_object_cloud:
        width_line_points, width_line_colors, width_line_debug = _robust_x_width_line_debug(final_points, final_colors)
        _write_cloud(width_line_path, width_line_points, width_line_colors)
        width_line_write_s = time.perf_counter() - width_line_write_started
    else:
        width_line_write_s = 0.0
    timing["object_cloud_width_line_debug_write_s"] = width_line_write_s
    if debug_mode:
        rejected_points = non_table_points[~selected_mask]
        rejected_colors = non_table_colors[~selected_mask] if non_table_colors is not None else None
    else:
        rejected_points = np.empty((0, 3), dtype=non_table_points.dtype)
        rejected_colors = np.empty((0, 3), dtype=non_table_colors.dtype) if non_table_colors is not None else None
    if debug_mode and len(rejected_points):
        _write_cloud(debug_dir / "object_cloud_rejected_points.ply", rejected_points, rejected_colors)
    if debug_mode:
        for record in clusters:
            cluster_index = int(record["cluster_index"])
            if cluster_index in selection_debug["selected_cluster_indices"]:
                continue
            cluster_mask = labels == cluster_index
            if np.count_nonzero(cluster_mask):
                cluster_colors = non_table_colors[cluster_mask] if non_table_colors is not None else None
                _write_cloud(rejected_dir / f"cluster_{cluster_index:03d}.ply", non_table_points[cluster_mask], cluster_colors)
    timing["fragment_merging_s"] = time.perf_counter() - stage_started
    timing["total_object_extraction_s"] = time.perf_counter() - started

    stage_started = time.perf_counter()
    debug_payload = {
        "extraction_method": "roi_table_cluster",
        "authoritative_object_source": "geometry_cluster",
        "point_cloud_frame": "rgb_camera",
        "point_cloud_units": "meters",
        "saved_depth_units": saved_units,
        "sdk_depth_scale_m_per_unit": scale_to_meters,
        "roi_frame": roi_debug["roi_frame"],
        "roi_units": roi_debug["roi_units"],
        "roi": roi_debug,
        "table_plane": table_debug,
        "clustering": clustering_debug,
        "clusters": clusters,
        "selection": selection_debug,
        "pre_selection_bounds": _stats(non_table_points),
        "post_selection_bounds": _stats(final_points),
        "full_scene_point_count": int(len(points)),
        "final_point_count": int(len(final_points)),
        "ai1_enabled": bool(getattr(cfg, "enable_ai1_validation", True)),
        "ai1_role": "validation_only",
        "ai1_used_for_object_extraction": False,
        "ai1_validation_overlap": None,
        "timing": timing,
        "artifacts": {
            **({"object_cloud": str(output_path)} if save_object_cloud else {}),
            "object_cloud_status": "ready" if save_object_cloud else "not_saved",
            "debug_ply_artifacts_enabled": bool(debug_mode),
            "object_cloud_roi_all_points": str(debug_dir / "object_cloud_roi_all_points.ply") if debug_mode else None,
            "object_cloud_after_table_removal": str(debug_dir / "object_cloud_after_table_removal.ply") if debug_mode else None,
            "table_plane": str(debug_dir / "table_plane.ply") if debug_mode else None,
            "object_cloud_rejected_points": str(debug_dir / "object_cloud_rejected_points.ply") if debug_mode else None,
            "object_cloud_width_line_debug": str(width_line_path) if width_line_debug is not None else None,
            "rejected_clusters_dir": str(rejected_dir) if debug_mode else None,
        },
    }
    if width_line_debug is not None:
        debug_payload["width_line_debug"] = width_line_debug
    if debug_mode:
        write_json_atomic(debug_dir / "object_extraction_debug.json", debug_payload)
    object_cloud_debug = {
        "generation_method": "roi_table_cluster",
        "extraction_method": "roi_table_cluster",
        "authoritative_object_source": "geometry_cluster",
        "object_cloud_coordinate_frame": "rgb_camera",
        "point_cloud_frame": "rgb_camera",
        "point_cloud_units": "meters",
        "roi_coordinate_frame": roi_debug["roi_frame"],
        "roi_units": roi_debug["roi_units"],
        "final_object_cloud_path": str(output_path) if save_object_cloud else None,
        "object_cloud_saved": bool(save_object_cloud),
        "final_point_count": int(len(final_points)),
        "xyz_min": debug_payload["post_selection_bounds"]["xyz_min"],
        "xyz_max": debug_payload["post_selection_bounds"]["xyz_max"],
        "xyz_spans": debug_payload["post_selection_bounds"]["xyz_spans"],
        "post_roi_xyz_spans": debug_payload["post_selection_bounds"]["xyz_spans"],
        "geometry_primary_extraction": True,
        "ai1_role": "validation_only",
        "ai1_used_for_object_extraction": False,
    }
    if width_line_debug is not None:
        object_cloud_debug["width_line_debug"] = width_line_debug
    if debug_mode:
        write_json_atomic(debug_dir / "object_cloud_debug.json", object_cloud_debug)
    timing["debug_json_write_s"] = time.perf_counter() - stage_started
    debug_payload["_runtime_final_object_points"] = final_points
    debug_payload["_runtime_final_object_colors"] = final_colors
    add_timing("object_extraction_setup_s", timing.get("setup_s", 0.0))
    add_timing("object_extraction_depth_load_s", timing.get("aligned_depth_load_s", 0.0))
    add_timing("object_extraction_rgb_decode_s", timing.get("rgb_image_decode_s", 0.0))
    add_timing("metric_cloud_creation_s", timing.get("cloud_creation_s", 0.0))
    add_timing("roi_filtering_s", timing.get("roi_crop_s", 0.0))
    add_timing("table_removal_s", timing.get("table_removal_s", 0.0))
    add_timing("voxelization_clustering_s", timing.get("clustering_s", 0.0))
    add_timing("fragment_recovery_s", max(0.0, timing.get("fragment_merging_s", 0.0) - object_cloud_write_s))
    add_timing("object_cloud_width_line_debug_write_s", timing.get("object_cloud_width_line_debug_write_s", 0.0))
    add_timing("object_extraction_debug_write_s", timing.get("debug_json_write_s", 0.0))
    return debug_payload
