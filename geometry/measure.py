"""High-level assembly helpers and Open3D geometry measurement for DimScan."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import math
import numpy as np

from app.config import DimScanConfig
from geometry.plant import make_plant_geometry
from geometry.pot import make_pot_geometry
from geometry.profiles import summarize_point_cloud_profile
from geometry.scene import make_scene_geometry
from utils.io import read_json_if_exists, utc_now_iso, write_json_atomic


def make_geometry_result(
    *,
    scene: dict[str, Any],
    plant: dict[str, Any] | None = None,
    pot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine scene, plant, and pot dictionaries into a standard geometry result."""
    return {
        "scene": scene,
        "plant": plant or make_plant_geometry(),
        "pot": pot or make_pot_geometry(),
    }


def make_placeholder_geometry(
    *,
    height_in: float,
    width_in: float,
    length_in: float | None = None,
    point_count: int | None = None,
) -> dict[str, Any]:
    """Create placeholder geometry for smoke tests and fixture generation."""
    scene = make_scene_geometry(
        height_in=height_in,
        width_in=width_in,
        length_in=length_in,
        point_count=point_count,
    )
    return make_geometry_result(scene=scene)


M_TO_IN = 39.3701
DEFAULT_MAX_POINTS = 120_000
DEFAULT_VOXEL_SIZE_M = 0.005
TABLE_DISTANCE_THRESHOLD_M = 0.015
CLUSTER_EPS_M = 0.04
MIN_CLUSTER_POINTS = 30
CLOUD_QUALITY_MIN_OBJECT_POINTS = 500
CLOUD_QUALITY_MIN_LEAF_POINTS = 300
CLOUD_QUALITY_OUTLIER_RATIO_REVIEW = 1.75
CLOUD_QUALITY_OUTLIER_RATIO_FAIL = 2.5
CLOUD_QUALITY_FRAGMENTED_REVIEW = 0.70
CLOUD_QUALITY_FRAGMENTED_FAIL = 0.40
CLOUD_QUALITY_MAX_DIMENSION_IN = 120.0


def _debug_paths(view_dir: Path) -> dict[str, str]:
    debug_dir = view_dir / "debug"
    return {
        "input_cloud_used": str(debug_dir / "input_cloud_used.ply"),
        "cropped_cloud": str(debug_dir / "cropped_cloud.ply"),
        "object_above_table": str(debug_dir / "object_above_table.ply"),
        "table_plane": str(debug_dir / "table_plane.json"),
        "geometry_debug": str(debug_dir / "geometry_debug.json"),
        "cloud_quality": str(debug_dir / "cloud_quality.json"),
    }


def _failed_geometry(
    *,
    reason: str,
    warnings: list[str],
    debug_paths: dict[str, str],
    metadata: dict[str, Any],
    source_cloud: str | None = None,
    cloud_type: str | None = None,
    segmentation_used: bool | None = None,
    segmentation_status: str | None = None,
    counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    active_source_cloud = source_cloud or str(metadata.get("source_cloud") or "unknown")
    active_cloud_type = cloud_type or str(metadata.get("cloud_type") or "unknown")
    active_segmentation_used = (
        bool(metadata.get("segmentation_used")) if segmentation_used is None else segmentation_used
    )
    active_segmentation_status = segmentation_status or str(
        metadata.get("segmentation_status") or "none"
    )
    roi_metadata = metadata.get("roi") if isinstance(metadata.get("roi"), dict) else {}
    return {
        "status": "failed",
        "reason": reason,
        "source_cloud": active_source_cloud,
        "cloud_type": active_cloud_type,
        "segmentation_used": active_segmentation_used,
        "segmentation_status": active_segmentation_status,
        "object_dimensions_in": None,
        "pot_dimensions_in": None,
        "leaf_canopy_dimensions_in": None,
        "table_plane": None,
        "counts": counts or {},
        "confidence": {},
        "roi_enabled": bool(roi_metadata.get("roi_enabled", False)),
        "roi_bounds": roi_metadata.get("roi_bounds", {}),
        "point_count_before_roi": roi_metadata.get("point_count_before_roi"),
        "point_count_after_roi": roi_metadata.get("point_count_after_roi"),
        "dimensions_in": {},
        "raw_dimensions_in": {},
        "warnings": sorted(set(warnings)),
        "debug_paths": debug_paths,
        "metadata": metadata,
        "scene": {
            "height_in": None,
            "width_in": None,
            "length_in": None,
            "point_count": 0,
        },
        "plant": make_plant_geometry(),
        "pot": make_pot_geometry(),
        "pot_quality": {},
        "profiles": {},
        "cloud_quality": metadata.get("cloud_quality", {}),
    }


def _load_open3d() -> Any:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Open3D is required for geometry processing.") from exc
    return o3d


def _read_segmentation_status(view_dir: Path) -> tuple[str, dict[str, Any]]:
    segmentation = read_json_if_exists(view_dir / "segmentation.json", default=None)
    if not isinstance(segmentation, dict):
        return "none", {}
    status = str(segmentation.get("status") or segmentation.get("segmentation_status") or "").lower()
    if status in {"skipped", "failed", "partial", "ok"}:
        return status, segmentation
    return "ok", segmentation


def _segment_paths(view_dir: Path) -> dict[str, Path]:
    return {
        "raw_cloud": view_dir / "cloud.ply",
        "segmented_cloud": view_dir / "segmented_cloud.ply",
        "object_cloud": view_dir / "object_cloud.ply",
        "pot_cloud": view_dir / "pot_cloud.ply",
        "leaf_cloud": view_dir / "leaf_cloud.ply",
        "table_cloud": view_dir / "table_cloud.ply",
    }


def _object_cloud_contract(
    view_dir: Path,
    name: str,
    path: Path,
    *,
    allow_missing_file: bool = False,
) -> dict[str, Any]:
    if name != "object_cloud" or (not path.is_file() and not allow_missing_file):
        return {}
    debug = read_json_if_exists(view_dir / "debug" / "object_cloud_debug.json", default={})
    if not isinstance(debug, dict):
        debug = {}
    frame = debug.get("object_cloud_coordinate_frame") or debug.get("point_cloud_frame") or "rgb_camera"
    units = debug.get("point_cloud_units") or debug.get("roi_units") or "meters"
    if frame != "rgb_camera" or units != "meters":
        raise RuntimeError(
            "object_cloud_unit_contract_invalid:"
            f"frame={frame or 'missing'}:units={units or 'missing'}"
        )
    return {
        "point_cloud_frame": "rgb_camera",
        "point_cloud_units": "meters",
        "unit_contract_source": (
            str(view_dir / "debug" / "object_cloud_debug.json")
            if debug
            else str(path)
        ),
        "trusted_metric_units": True,
    }


def _load_cloud(
    o3d: Any,
    path: Path,
    warnings: list[str],
    *,
    unit_contract: dict[str, Any] | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    if not path.is_file():
        return None, {"path": str(path), "point_count_input": 0, "removed_nonfinite_points": 0}

    cloud = o3d.io.read_point_cloud(str(path))
    cloud, removed_nonfinite = _finite_cloud(o3d, cloud)
    point_count_input = int(len(cloud.points))
    info: dict[str, Any] = {
        "path": str(path),
        "point_count_input": point_count_input,
        "removed_nonfinite_points": removed_nonfinite,
    }
    if point_count_input == 0:
        return None, info

    unit_contract = unit_contract if isinstance(unit_contract, dict) else {}
    if unit_contract.get("trusted_metric_units") is True:
        unit_mode = "meters"
    else:
        cloud, unit_mode = _normalize_metric_units(o3d, cloud, warnings)
    info["unit_mode"] = unit_mode
    info.update(unit_contract)
    return cloud, info


def _cloud_from_in_memory_points(
    o3d: Any,
    points: Any,
    warnings: list[str],
) -> tuple[Any | None, dict[str, Any]]:
    point_array = np.asarray(points, dtype=float)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        warnings.append("in_memory_object_cloud_invalid_shape")
        return None, {"point_count_input": 0, "removed_nonfinite_points": 0}

    finite_mask = np.isfinite(point_array).all(axis=1)
    removed_nonfinite = int(len(point_array) - np.count_nonzero(finite_mask))
    if removed_nonfinite:
        point_array = point_array[finite_mask]
    if len(point_array) == 0:
        warnings.append("in_memory_object_cloud_empty")
        return None, {
            "point_count_input": 0,
            "removed_nonfinite_points": removed_nonfinite,
        }

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(point_array.copy())
    return cloud, {
        "point_count_input": int(len(point_array)),
        "removed_nonfinite_points": removed_nonfinite,
        "unit_mode": "meters",
        "source": "in_memory_geometry_primary",
    }


def _merge_clouds(o3d: Any, clouds: list[Any]) -> Any:
    merged = o3d.geometry.PointCloud()
    point_arrays = [np.asarray(cloud.points) for cloud in clouds if len(cloud.points)]
    if not point_arrays:
        return merged
    merged.points = o3d.utility.Vector3dVector(np.vstack(point_arrays))

    color_arrays = [
        np.asarray(cloud.colors)
        for cloud in clouds
        if cloud.has_colors() and len(cloud.colors) == len(cloud.points)
    ]
    if len(color_arrays) == len(point_arrays):
        merged.colors = o3d.utility.Vector3dVector(np.vstack(color_arrays))
    return merged


def _finite_cloud(o3d: Any, cloud: Any) -> tuple[Any, int]:
    points = np.asarray(cloud.points)
    if points.size == 0:
        return cloud, 0
    mask = np.isfinite(points).all(axis=1)
    removed_count = int(len(points) - np.count_nonzero(mask))
    if removed_count == 0:
        return cloud, 0
    filtered = cloud.select_by_index(np.where(mask)[0].tolist())
    return filtered, removed_count


def _copy_cloud(o3d: Any, cloud: Any) -> Any:
    copied = o3d.geometry.PointCloud()
    copied.points = o3d.utility.Vector3dVector(np.asarray(cloud.points).copy())
    if cloud.has_colors():
        copied.colors = o3d.utility.Vector3dVector(np.asarray(cloud.colors).copy())
    return copied


def _normalize_metric_units(o3d: Any, cloud: Any, warnings: list[str]) -> tuple[Any, str]:
    points = np.asarray(cloud.points)
    if len(points) == 0:
        return cloud, "unknown"

    spans = points.max(axis=0) - points.min(axis=0)
    max_span = float(np.max(spans))
    if max_span > 20.0:
        scaled = _copy_cloud(o3d, cloud)
        scaled.points = o3d.utility.Vector3dVector(points / 1000.0)
        warnings.append("metric_unit_assumed_millimeters")
        return scaled, "meters_from_millimeters"
    return cloud, "meters"


def _write_cloud(path: str, cloud: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d = _load_open3d()
    o3d.io.write_point_cloud(str(output_path), cloud, write_ascii=True)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _central_crop(o3d: Any, cloud: Any, *, keep_ratio: float = 0.92) -> Any:
    points = np.asarray(cloud.points)
    if len(points) < 10:
        return cloud
    low = np.quantile(points[:, :2], (1.0 - keep_ratio) / 2.0, axis=0)
    high = np.quantile(points[:, :2], 1.0 - ((1.0 - keep_ratio) / 2.0), axis=0)
    mask = (
        (points[:, 0] >= low[0])
        & (points[:, 0] <= high[0])
        & (points[:, 1] >= low[1])
        & (points[:, 1] <= high[1])
    )
    indices = np.where(mask)[0].tolist()
    if not indices:
        return cloud
    return cloud.select_by_index(indices)


def _roi_bounds(cfg: DimScanConfig) -> dict[str, float | None]:
    margin = float(cfg.roi_margin_m or 0.0)
    return {
        "min_depth_m": float(cfg.roi_min_depth_m) - margin,
        "max_depth_m": float(cfg.roi_max_depth_m) + margin,
        "min_x_m": -float(cfg.roi_half_width_m) - margin,
        "max_x_m": float(cfg.roi_half_width_m) + margin,
        "table_min_height_m": cfg.roi_min_height_m,
        "table_max_height_m": cfg.roi_max_height_m,
        "margin_m": cfg.roi_margin_m,
    }


def _apply_workspace_roi(
    o3d: Any,
    cloud: Any,
    cfg: DimScanConfig,
    warnings: list[str],
) -> tuple[Any, dict[str, Any]]:
    point_count_before = int(len(cloud.points))
    bounds = _roi_bounds(cfg)
    metadata: dict[str, Any] = {
        "roi_enabled": bool(cfg.roi_enabled),
        "roi_bounds": bounds,
        "point_count_before_roi": point_count_before,
        "point_count_after_roi": point_count_before,
    }
    if not cfg.roi_enabled or point_count_before == 0:
        return cloud, metadata

    points = np.asarray(cloud.points)
    mask = (
        (points[:, 2] >= float(bounds["min_depth_m"]))
        & (points[:, 2] <= float(bounds["max_depth_m"]))
        & (points[:, 0] >= float(bounds["min_x_m"]))
        & (points[:, 0] <= float(bounds["max_x_m"]))
    )

    indices = np.where(mask)[0].tolist()
    metadata["point_count_after_roi"] = len(indices)
    if not indices:
        warnings.append("roi_empty")
        return cloud.select_by_index([]), metadata

    kept_ratio = len(indices) / point_count_before
    if kept_ratio < 0.05:
        warnings.append("roi_removed_most_points")
    return cloud.select_by_index(indices), metadata


def _merge_roi_metadata(target: dict[str, Any], roi: dict[str, Any], *, prefix: str | None = None) -> None:
    if prefix:
        target[f"{prefix}_roi"] = roi
        return
    target["roi"] = roi


def _largest_cluster(o3d: Any, cloud: Any) -> tuple[Any, float, int]:
    points = np.asarray(cloud.points)
    if len(points) < MIN_CLUSTER_POINTS:
        return cloud, 1.0 if len(points) else 0.0, 0

    labels = np.asarray(
        cloud.cluster_dbscan(eps=CLUSTER_EPS_M, min_points=MIN_CLUSTER_POINTS, print_progress=False)
    )
    valid_labels = labels[labels >= 0]
    if valid_labels.size == 0:
        return cloud, 0.0, 0

    counts = np.bincount(valid_labels)
    largest_label = int(np.argmax(counts))
    largest_indices = np.where(labels == largest_label)[0].tolist()
    return cloud.select_by_index(largest_indices), _safe_ratio(len(largest_indices), len(points)), int(len(counts))


def _dimensions_from_cloud(cloud: Any) -> dict[str, float]:
    points = np.asarray(cloud.points)
    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    extents_m = cloud.get_axis_aligned_bounding_box().get_extent()
    robust_extents_m = _robust_extents(points)
    raw_spans_in = np.asarray(extents_m, dtype=float) * M_TO_IN
    robust_spans_in = robust_extents_m * M_TO_IN
    horizontal_span_m = float(extents_m[0])
    vertical_span_m = float(extents_m[1])
    depth_thickness_m = float(extents_m[2])
    robust_horizontal_span_m = float(robust_extents_m[0])
    robust_vertical_span_m = float(robust_extents_m[1])
    robust_depth_thickness_m = float(robust_extents_m[2])
    return {
        "point_cloud_frame": "rgb_camera",
        "point_cloud_units": "meters",
        "x_min_m": float(xyz_min[0]),
        "y_min_m": float(xyz_min[1]),
        "z_min_m": float(xyz_min[2]),
        "x_max_m": float(xyz_max[0]),
        "y_max_m": float(xyz_max[1]),
        "z_max_m": float(xyz_max[2]),
        "x_span_m": horizontal_span_m,
        "y_span_m": vertical_span_m,
        "z_span_m": depth_thickness_m,
        "x_span_in": horizontal_span_m * M_TO_IN,
        "y_span_in": vertical_span_m * M_TO_IN,
        "z_span_in": depth_thickness_m * M_TO_IN,
        "horizontal_span_m": horizontal_span_m,
        "vertical_span_m": vertical_span_m,
        "depth_thickness_m": depth_thickness_m,
        "horizontal_span_in": horizontal_span_m * M_TO_IN,
        "vertical_span_in": vertical_span_m * M_TO_IN,
        "depth_thickness_in": depth_thickness_m * M_TO_IN,
        "robust_horizontal_span_m": robust_horizontal_span_m,
        "robust_vertical_span_m": robust_vertical_span_m,
        "robust_depth_thickness_m": robust_depth_thickness_m,
        "robust_horizontal_span_in": robust_horizontal_span_m * M_TO_IN,
        "robust_vertical_span_in": robust_vertical_span_m * M_TO_IN,
        "robust_depth_thickness_in": robust_depth_thickness_m * M_TO_IN,
        "length_in": horizontal_span_m * M_TO_IN,
        "width_in": depth_thickness_m * M_TO_IN,
        "height_in": vertical_span_m * M_TO_IN,
        "length_robust_in": robust_horizontal_span_m * M_TO_IN,
        "width_robust_in": robust_depth_thickness_m * M_TO_IN,
        "height_robust_in": robust_vertical_span_m * M_TO_IN,
        "raw_x_span_in": float(extents_m[0]) * M_TO_IN,
        "raw_y_span_in": float(extents_m[1]) * M_TO_IN,
        "raw_z_span_in": float(extents_m[2]) * M_TO_IN,
        "raw_x_span_m": horizontal_span_m,
        "raw_y_span_m": vertical_span_m,
        "raw_z_span_m": depth_thickness_m,
        "robust_x_span_in": float(robust_extents_m[0]) * M_TO_IN,
        "robust_y_span_in": float(robust_extents_m[1]) * M_TO_IN,
        "robust_z_span_in": float(robust_extents_m[2]) * M_TO_IN,
        "max_raw_to_robust_span_ratio": _max_span_ratio(raw_spans_in, robust_spans_in),
        "point_count": int(len(points)),
    }


def _robust_extents(points: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0) -> np.ndarray:
    if len(points) < 10:
        return points.max(axis=0) - points.min(axis=0)
    low = np.percentile(points, low_percentile, axis=0)
    high = np.percentile(points, high_percentile, axis=0)
    robust_extents = np.maximum(high - low, 0.0)
    robust_extents[1] = points[:, 1].max() - points[:, 1].min()
    return robust_extents


def _max_span_ratio(raw_spans: np.ndarray, robust_spans: np.ndarray) -> float:
    ratios = [
        float(raw / robust)
        for raw, robust in zip(raw_spans, robust_spans)
        if math.isfinite(float(raw)) and math.isfinite(float(robust)) and robust > 0
    ]
    return max(ratios) if ratios else 1.0


def _dimension_warnings(dimensions: dict[str, float]) -> list[str]:
    warnings: list[str] = []
    values = [
        dimensions.get("length_in", 0),
        dimensions.get("width_in", 0),
        dimensions.get("height_in", 0),
    ]
    if any(value <= 0 for value in values):
        warnings.append("dimension_suspicious")
    if any(value > 120 for value in values):
        warnings.append("dimension_suspicious")
    if dimensions.get("height_in", 0) < 1:
        warnings.append("dimension_suspicious")
    return warnings


NON_BLOCKING_GEOMETRY_WARNINGS = {
    "leaf_from_fallback_mask",
    "leaf_geometry_from_fallback_mask",
    "leaf_features_suppressed_due_to_fallback",
    "leaf_cloud_missing",
    "raw_bbox_much_larger_than_robust_bbox",
    "object_raw_bbox_much_larger_than_robust_bbox",
    "pot_mask_rejected_not_used_for_geometry",
    "segmentation_failed_manifest_but_object_cloud_available",
    "table_segment_missing_using_ransac",
}


def _geometry_status_from_warnings(reason: str | None, warnings: list[str]) -> str:
    if reason is not None:
        return "degraded"
    blocking_warnings = [warning for warning in warnings if warning not in NON_BLOCKING_GEOMETRY_WARNINGS]
    return "ok" if not blocking_warnings else "degraded"


def _write_table_debug(path: str, table: dict[str, Any]) -> None:
    write_json_atomic(path, table)


def _write_geometry_debug(path: str, debug: dict[str, Any]) -> None:
    write_json_atomic(path, debug)


def _write_cloud_quality(path: str, cloud_quality: dict[str, Any]) -> None:
    write_json_atomic(path, {"cloud_quality": cloud_quality})


def _bbox_payload(dimensions: dict[str, Any], *, robust: bool = False) -> dict[str, Any]:
    prefix = "robust_" if robust else "raw_"
    if robust:
        return {
            "x_span_in": dimensions.get("robust_x_span_in"),
            "y_span_in": dimensions.get("robust_y_span_in"),
            "z_span_in": dimensions.get("robust_z_span_in"),
            "length_in": dimensions.get("length_robust_in"),
            "width_in": dimensions.get("width_robust_in"),
            "height_in": dimensions.get("height_robust_in"),
        }
    return {
        "x_span_in": dimensions.get(f"{prefix}x_span_in"),
        "y_span_in": dimensions.get(f"{prefix}y_span_in"),
        "z_span_in": dimensions.get(f"{prefix}z_span_in"),
        "length_in": dimensions.get("length_in"),
        "width_in": dimensions.get("width_in"),
        "height_in": dimensions.get("height_in"),
    }


def _cloud_quality_record(
    *,
    label: str,
    dimensions: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
    source: str,
    fallback_mask_used: bool,
) -> dict[str, Any]:
    metadata = metadata if isinstance(metadata, dict) else {}
    if not dimensions:
        return {
            "available": False,
            "source": source,
            "fallback_mask_used": fallback_mask_used,
            "point_count": 0,
            "raw_bbox_in": None,
            "robust_bbox_in": None,
            "largest_cluster_ratio": metadata.get("largest_cluster_ratio"),
            "cluster_count": metadata.get("cluster_count"),
            "outlier_ratio": None,
            "outlier_sensitive": False,
            "quality": "fail",
            "warnings": [f"{label}_cloud_missing"],
        }

    point_count = int(dimensions.get("point_count") or 0)
    min_points = CLOUD_QUALITY_MIN_LEAF_POINTS if label == "leaf" else CLOUD_QUALITY_MIN_OBJECT_POINTS
    largest_cluster_ratio = metadata.get("largest_cluster_ratio")
    outlier_ratio = float(dimensions.get("max_raw_to_robust_span_ratio") or 1.0)
    dimension_values = [
        dimensions.get("length_in"),
        dimensions.get("width_in"),
        dimensions.get("height_in"),
    ]
    warnings: list[str] = []

    if point_count <= 0:
        warnings.append(f"{label}_cloud_empty")
    elif point_count < min_points:
        warnings.append(f"{label}_cloud_too_sparse")
    if fallback_mask_used:
        warnings.append(f"{label}_from_fallback_mask")
    if outlier_ratio >= CLOUD_QUALITY_OUTLIER_RATIO_REVIEW:
        warnings.append("raw_bbox_much_larger_than_robust_bbox")
    if isinstance(largest_cluster_ratio, (int, float)) and largest_cluster_ratio < CLOUD_QUALITY_FRAGMENTED_REVIEW:
        warnings.append(f"{label}_cloud_fragmented")
    if any(
        not isinstance(value, (int, float)) or value <= 0 or value > CLOUD_QUALITY_MAX_DIMENSION_IN
        for value in dimension_values
    ):
        warnings.append(f"{label}_cloud_dimensions_suspicious")

    blocking_warnings = [
        warning for warning in warnings if warning != "raw_bbox_much_larger_than_robust_bbox"
    ]
    fail = (
        point_count <= 0
        or point_count < min_points / 2
        or (
            isinstance(largest_cluster_ratio, (int, float))
            and largest_cluster_ratio < CLOUD_QUALITY_FRAGMENTED_FAIL
        )
    )
    review = bool(blocking_warnings)
    quality = "fail" if fail else ("review" if review else "ok")
    return {
        "available": point_count > 0,
        "source": source,
        "fallback_mask_used": fallback_mask_used,
        "point_count": point_count,
        "raw_bbox_in": _bbox_payload(dimensions),
        "robust_bbox_in": _bbox_payload(dimensions, robust=True),
        "largest_cluster_ratio": largest_cluster_ratio,
        "cluster_count": metadata.get("cluster_count"),
        "outlier_ratio": outlier_ratio,
        "outlier_sensitive": outlier_ratio >= CLOUD_QUALITY_OUTLIER_RATIO_REVIEW,
        "quality": quality,
        "warnings": warnings,
        "thresholds": {
            "min_point_count": min_points,
            "outlier_ratio_review": CLOUD_QUALITY_OUTLIER_RATIO_REVIEW,
            "outlier_ratio_fail": CLOUD_QUALITY_OUTLIER_RATIO_FAIL,
            "largest_cluster_ratio_review": CLOUD_QUALITY_FRAGMENTED_REVIEW,
            "largest_cluster_ratio_fail": CLOUD_QUALITY_FRAGMENTED_FAIL,
            "max_dimension_in": CLOUD_QUALITY_MAX_DIMENSION_IN,
        },
    }


def _measure_object_cloud(
    *,
    o3d: Any,
    cloud: Any,
    cfg: DimScanConfig,
    warnings: list[str],
    debug_paths: dict[str, str],
    write_debug: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any] | None, str | None, dict[str, Any] | None]:
    metadata: dict[str, Any] = {}
    table_record: dict[str, Any] | None = None
    working, roi_metadata = _apply_workspace_roi(o3d, cloud, cfg, warnings)
    _merge_roi_metadata(metadata, roi_metadata)
    if len(working.points) == 0:
        return None, metadata, table_record, "No points remain after workspace ROI crop.", None

    if len(working.points) > DEFAULT_MAX_POINTS:
        working = working.voxel_down_sample(DEFAULT_VOXEL_SIZE_M)
        metadata["downsampled"] = True
        metadata["point_count_downsampled"] = int(len(working.points))
    else:
        metadata["downsampled"] = False
        metadata["point_count_downsampled"] = int(len(working.points))

    cropped = _central_crop(o3d, working)
    if write_debug:
        _write_cloud(debug_paths["cropped_cloud"], cropped)
    if len(cropped.points) < MIN_CLUSTER_POINTS:
        warnings.append("low_point_count")
        return None, metadata, table_record, "Not enough points after ROI crop.", None

    table_found = False
    table_record = {"table_found": False}
    object_cloud = cropped
    try:
        plane_model, inliers = cropped.segment_plane(
            distance_threshold=TABLE_DISTANCE_THRESHOLD_M,
            ransac_n=3,
            num_iterations=1000,
        )
        table_found = len(inliers) >= max(100, int(0.05 * len(cropped.points)))
        table_record = {
            "table_found": table_found,
            "plane_model": [float(value) for value in plane_model],
            "inlier_count": int(len(inliers)),
            "inlier_ratio": _safe_ratio(len(inliers), len(cropped.points)),
            "distance_threshold_m": TABLE_DISTANCE_THRESHOLD_M,
        }
        if table_found:
            points = np.asarray(cropped.points)
            a, b, c, d = [float(value) for value in plane_model]
            signed = points @ np.asarray([a, b, c], dtype=float) + d
            positive_mask = signed > TABLE_DISTANCE_THRESHOLD_M
            negative_mask = signed < -TABLE_DISTANCE_THRESHOLD_M
            if np.count_nonzero(positive_mask) >= np.count_nonzero(negative_mask):
                height_above_table = signed
                side_sign = 1
            else:
                height_above_table = -signed
                side_sign = -1

            object_mask = height_above_table > TABLE_DISTANCE_THRESHOLD_M
            margin = float(cfg.roi_margin_m or 0.0)
            if cfg.roi_min_height_m is not None:
                object_mask &= height_above_table >= max(0.0, float(cfg.roi_min_height_m) - margin)
            if cfg.roi_max_height_m is not None:
                object_mask &= height_above_table <= float(cfg.roi_max_height_m) + margin
            chosen = np.where(object_mask)[0]
            table_record["above_table_side_sign"] = side_sign
            table_record["max_height_above_table_m"] = cfg.roi_max_height_m
            if len(chosen) > 0:
                object_cloud = cropped.select_by_index(chosen.tolist())
            else:
                warnings.append("roi_empty")
        else:
            warnings.append("no_table_found")
    except Exception as exc:
        table_record = {"table_found": False, "error": str(exc)}
        warnings.append("no_table_found")

    if write_debug:
        _write_table_debug(debug_paths["table_plane"], table_record)
    metadata["table_found"] = table_found
    metadata["point_count_after_table_removal"] = int(len(object_cloud.points))
    if write_debug:
        _write_cloud(debug_paths["object_above_table"], object_cloud)

    if len(object_cloud.points) < MIN_CLUSTER_POINTS:
        warnings.append("low_point_count")
        return None, metadata, table_record, "Not enough object points after table removal.", None

    clustered, largest_cluster_ratio, cluster_count = _largest_cluster(o3d, object_cloud)
    metadata["largest_cluster_ratio"] = largest_cluster_ratio
    metadata["cluster_count"] = cluster_count
    metadata["point_count_object"] = int(len(clustered.points))
    if len(clustered.points) < MIN_CLUSTER_POINTS:
        warnings.append("low_point_count")
        return None, metadata, table_record, "No usable object cluster found.", None

    dimensions = _dimensions_from_cloud(clustered)
    warnings.extend(_dimension_warnings(dimensions))
    if dimensions["point_count"] < 500:
        warnings.append("low_point_count")
    return dimensions, metadata, table_record, None, summarize_point_cloud_profile(clustered)


def _measure_trusted_object_cloud_direct(
    *,
    cloud: Any,
    warnings: list[str],
    debug_paths: dict[str, str],
    write_debug: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any] | None, str | None, dict[str, Any] | None]:
    metadata: dict[str, Any] = {
        "geometry_method": "trusted_final_object_cloud_direct_extents",
        "point_count_object": int(len(cloud.points)),
        "direct_final_object_cloud_used": True,
        "additional_table_or_cluster_filtering": False,
    }
    if len(cloud.points) < MIN_CLUSTER_POINTS:
        return None, metadata, {"table_found": False, "skipped": True}, "Not enough points in final object_cloud.ply.", None
    if write_debug:
        _write_cloud(debug_paths["cropped_cloud"], cloud)
        _write_cloud(debug_paths["object_above_table"], cloud)
        _write_table_debug(
            debug_paths["table_plane"],
            {
                "table_found": False,
                "skipped": True,
                "reason": "trusted_final_object_cloud_already_object_only",
            },
        )
    dimensions = _dimensions_from_cloud(cloud)
    dimension_warnings = _dimension_warnings(dimensions)
    warnings.extend(dimension_warnings)
    metadata["dimension_warnings"] = dimension_warnings
    if dimensions["point_count"] < 500:
        warnings.append("low_point_count")
    return (
        dimensions,
        metadata,
        {
            "table_found": False,
            "skipped": True,
            "reason": "trusted_final_object_cloud_already_object_only",
        },
        None,
        summarize_point_cloud_profile(cloud),
    )


def _measure_segment_dimensions(
    *,
    o3d: Any,
    cloud: Any,
    cfg: DimScanConfig,
    warnings: list[str],
    warning_prefix: str,
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any] | None]:
    metadata: dict[str, Any] = {"point_count_input": int(len(cloud.points))}
    working, roi_metadata = _apply_workspace_roi(o3d, cloud, cfg, warnings)
    _merge_roi_metadata(metadata, roi_metadata)
    if len(working.points) == 0:
        warnings.append(f"{warning_prefix}_roi_empty")
        return None, metadata, None

    if len(working.points) > DEFAULT_MAX_POINTS:
        working = working.voxel_down_sample(DEFAULT_VOXEL_SIZE_M)
        metadata["downsampled"] = True
    else:
        metadata["downsampled"] = False
    metadata["point_count_downsampled"] = int(len(working.points))

    if len(working.points) < MIN_CLUSTER_POINTS:
        warnings.append(f"{warning_prefix}_low_point_count")
        return None, metadata, None

    clustered, largest_cluster_ratio, cluster_count = _largest_cluster(o3d, working)
    metadata["largest_cluster_ratio"] = largest_cluster_ratio
    metadata["cluster_count"] = cluster_count
    metadata["point_count_object"] = int(len(clustered.points))
    if len(clustered.points) < MIN_CLUSTER_POINTS:
        warnings.append(f"{warning_prefix}_low_point_count")
        return None, metadata, None
    return _dimensions_from_cloud(clustered), metadata, summarize_point_cloud_profile(clustered)


def _public_dimensions(dimensions: dict[str, Any] | None) -> dict[str, Any] | None:
    if not dimensions:
        return None
    public = {
        "length_in": dimensions["length_in"],
        "width_in": dimensions["width_in"],
        "height_in": dimensions["height_in"],
        "point_count": dimensions["point_count"],
    }
    for key in (
        "length_robust_in",
        "width_robust_in",
        "height_robust_in",
        "max_raw_to_robust_span_ratio",
        "point_cloud_frame",
        "point_cloud_units",
        "x_min_m",
        "y_min_m",
        "z_min_m",
        "x_max_m",
        "y_max_m",
        "z_max_m",
        "x_span_m",
        "y_span_m",
        "z_span_m",
        "x_span_in",
        "y_span_in",
        "z_span_in",
        "horizontal_span_m",
        "vertical_span_m",
        "depth_thickness_m",
        "horizontal_span_in",
        "vertical_span_in",
        "depth_thickness_in",
        "robust_horizontal_span_m",
        "robust_vertical_span_m",
        "robust_depth_thickness_m",
        "robust_horizontal_span_in",
        "robust_vertical_span_in",
        "robust_depth_thickness_in",
        "raw_x_span_m",
        "raw_y_span_m",
        "raw_z_span_m",
        "raw_x_span_in",
        "raw_y_span_in",
        "raw_z_span_in",
    ):
        if key in dimensions:
            public[key] = dimensions[key]
    return public


def _positive_or_none(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        return None
    return numeric


def _legacy_raw_dimensions(dimensions: dict[str, Any] | None) -> dict[str, Any]:
    if not dimensions:
        return {}
    return {
        "x_span_in": dimensions["raw_x_span_in"],
        "y_span_in": dimensions["raw_y_span_in"],
        "z_span_in": dimensions["raw_z_span_in"],
    }


def _make_geometry(
    *,
    status: str,
    reason: str | None,
    source_cloud: str,
    cloud_type: str,
    segmentation_used: bool,
    segmentation_status: str,
    object_dimensions: dict[str, Any] | None,
    pot_dimensions: dict[str, Any] | None,
    leaf_dimensions: dict[str, Any] | None,
    table_plane: dict[str, Any] | None,
    counts: dict[str, int],
    confidence: dict[str, Any],
    warnings: list[str],
    debug_paths: dict[str, str],
    metadata: dict[str, Any],
    pot_quality: dict[str, Any] | None = None,
    profiles: dict[str, Any] | None = None,
    cloud_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    object_public = _public_dimensions(object_dimensions)
    scene = {
        "height_in": object_public.get("height_in") if object_public else None,
        "width_in": object_public.get("width_in") if object_public else None,
        "length_in": object_public.get("length_in") if object_public else None,
        "point_count": object_public.get("point_count") if object_public else None,
    }
    pot_public = _public_dimensions(pot_dimensions)
    leaf_public = _public_dimensions(leaf_dimensions)
    metadata.update(
        {
            "segmentation_used": segmentation_used,
            "segmentation_status": segmentation_status,
            "source_cloud": source_cloud,
            "cloud_type": cloud_type,
        }
    )
    roi_metadata = metadata.get("roi") if isinstance(metadata.get("roi"), dict) else {}
    return {
        "status": status,
        "reason": reason,
        "source_cloud": source_cloud,
        "cloud_type": cloud_type,
        "segmentation_used": segmentation_used,
        "segmentation_status": segmentation_status,
        "object_dimensions_in": object_public,
        "pot_dimensions_in": pot_public,
        "leaf_canopy_dimensions_in": leaf_public,
        "table_plane": table_plane,
        "counts": counts,
        "confidence": confidence,
        "roi_enabled": bool(roi_metadata.get("roi_enabled", False)),
        "roi_bounds": roi_metadata.get("roi_bounds", {}),
        "point_count_before_roi": roi_metadata.get("point_count_before_roi"),
        "point_count_after_roi": roi_metadata.get("point_count_after_roi"),
        "dimensions_in": object_public or {},
        "raw_dimensions_in": _legacy_raw_dimensions(object_dimensions),
        "warnings": sorted(set(warnings)),
        "debug_paths": debug_paths,
        "metadata": metadata,
        "pot_quality": pot_quality or {},
        "cloud_quality": cloud_quality or {},
        "profiles": profiles or {},
        "scene": scene,
        "plant": make_plant_geometry(
            height_in=_positive_or_none(
                leaf_public.get("height_in") if leaf_public else scene["height_in"]
            ),
            width_in=_positive_or_none(
                leaf_public.get("width_in") if leaf_public else scene["width_in"]
            ),
            point_count=leaf_public.get("point_count") if leaf_public else scene["point_count"],
        ),
        "pot": make_pot_geometry(
            visible_top_diameter_in=_positive_or_none(
                pot_public.get("width_in") if pot_public else None
            ),
            visible_height_in=_positive_or_none(
                pot_public.get("height_in") if pot_public else None
            ),
            point_count=pot_public.get("point_count") if pot_public else None,
        ),
    }


def measure_view_geometry(
    view_dir: str | Path,
    cfg: DimScanConfig | None = None,
    *,
    debug_mode: bool = True,
    geometry_primary_object_points: Any | None = None,
) -> dict[str, Any]:
    """Measure one view's geometry with optional segmentation artifacts."""
    active_cfg = cfg or DimScanConfig()
    o3d = _load_open3d()
    view_path = Path(view_dir)
    debug_paths = _debug_paths(view_path)
    warnings: list[str] = []
    capture_meta = read_json_if_exists(view_path / "capture_meta.json", default={})
    if not isinstance(capture_meta, dict):
        capture_meta = {}
    cloud_type = str(capture_meta.get("cloud_type") or "unknown")
    if cloud_type != "metric_xyz":
        warnings.append("cloud_not_metric")

    segment_paths = _segment_paths(view_path)
    manifest_status, segmentation_manifest = _read_segmentation_status(view_path)
    segment_statuses = segmentation_manifest.get("segments", {}) if isinstance(segmentation_manifest, dict) else {}
    if not isinstance(segment_statuses, dict):
        segment_statuses = {}
    pot_quality = segmentation_manifest.get("pot_quality") if isinstance(segmentation_manifest, dict) else {}
    if not isinstance(pot_quality, dict):
        pot_quality = {}
    metadata: dict[str, Any] = {
        "processed_at": utc_now_iso(),
        "cloud_type": cloud_type,
        "segmentation_manifest": segmentation_manifest,
    }
    counts: dict[str, int] = {}
    confidence: dict[str, Any] = {}
    loaded: dict[str, Any] = {}
    load_metadata: dict[str, Any] = {}

    has_in_memory_object_cloud = False
    if geometry_primary_object_points is not None:
        in_memory_warnings: list[str] = []
        object_path = segment_paths["object_cloud"]
        object_contract = _object_cloud_contract(
            view_path,
            "object_cloud",
            object_path,
            allow_missing_file=True,
        )
        cloud, info = _cloud_from_in_memory_points(
            o3d,
            geometry_primary_object_points,
            in_memory_warnings,
        )
        info["path"] = str(object_path)
        info.update(object_contract)
        info["in_memory"] = cloud is not None and object_contract.get("trusted_metric_units") is True
        if in_memory_warnings:
            info["warnings"] = in_memory_warnings
        if cloud is not None and object_contract.get("trusted_metric_units") is True:
            loaded["object_cloud"] = cloud
            has_in_memory_object_cloud = True
        load_metadata["object_cloud"] = info
        counts["object_cloud_points"] = int(info.get("point_count_input") or 0)

    for name, path in segment_paths.items():
        if has_in_memory_object_cloud and name in {"raw_cloud", "object_cloud"}:
            if name == "raw_cloud":
                load_metadata[name] = {
                    "path": str(path),
                    "point_count_input": 0,
                    "removed_nonfinite_points": 0,
                    "skipped": True,
                    "reason": "trusted_in_memory_object_cloud_available",
                }
                counts[f"{name}_points"] = 0
            continue
        if name in load_metadata and name in loaded:
            continue
        load_warnings: list[str] = []
        cloud, info = _load_cloud(
            o3d,
            path,
            load_warnings,
            unit_contract=_object_cloud_contract(view_path, name, path),
        )
        if load_warnings:
            info["warnings"] = load_warnings
        load_metadata[name] = info
        counts[f"{name}_points"] = int(info.get("point_count_input") or 0)
        if cloud is not None:
            loaded[name] = cloud
    metadata["clouds"] = load_metadata

    object_cloud_required_missing = (
        manifest_status in {"ok", "partial"}
        and segment_statuses.get("object") == "failed"
        and "object_cloud" not in loaded
    )
    if object_cloud_required_missing:
        warnings.append("object_cloud_unavailable_geometry_blocked")
        metadata["object_cloud_required_for_geometry"] = True
        metadata["object_cloud_required_reason"] = "segmentation_object_cloud_failed"
        cloud_quality = {
            "object": _cloud_quality_record(
                label="object",
                dimensions=None,
                metadata={},
                source="object_cloud",
                fallback_mask_used=False,
            )
        }
        metadata["cloud_quality"] = cloud_quality
        geometry = _failed_geometry(
            reason="object_cloud_unavailable_geometry_blocked",
            warnings=warnings,
            debug_paths=debug_paths,
            metadata=metadata,
            source_cloud="object_cloud",
            cloud_type=cloud_type,
            segmentation_used=True,
            segmentation_status="failed",
            counts=counts,
        )
        if debug_mode:
            _write_cloud_quality(debug_paths["cloud_quality"], cloud_quality)
            _write_geometry_debug(
                debug_paths["geometry_debug"],
                {
                    "status": geometry["status"],
                    "reason": geometry["reason"],
                    "warnings": geometry["warnings"],
                    "metadata": metadata,
                    "cloud_quality": cloud_quality,
                },
            )
        return geometry

    object_source_name = "raw_cloud"
    object_cloud = loaded.get("raw_cloud")
    segmentation_used = False
    segmentation_status = manifest_status

    if manifest_status == "skipped":
        warnings.append("segmentation_skipped_using_raw_cloud")
    elif manifest_status == "failed":
        if "object_cloud" in loaded:
            warnings.append("segmentation_failed_manifest_but_object_cloud_available")
            object_source_name = "object_cloud"
            object_cloud = loaded["object_cloud"]
            segmentation_used = True
            segmentation_status = "partial"
        else:
            warnings.append("segmentation_failed_using_raw_cloud")
    else:
        if "object_cloud" in loaded:
            object_source_name = "object_cloud"
            object_cloud = loaded["object_cloud"]
            segmentation_used = True
        elif "segmented_cloud" in loaded:
            object_source_name = "segmented_cloud"
            object_cloud = loaded["segmented_cloud"]
            segmentation_used = True
        elif "pot_cloud" in loaded and "leaf_cloud" in loaded:
            object_source_name = "combined_pot_leaf_cloud"
            object_cloud = _merge_clouds(o3d, [loaded["pot_cloud"], loaded["leaf_cloud"]])
            segmentation_used = True
        elif "pot_cloud" in loaded or "leaf_cloud" in loaded:
            segmentation_used = True
            segmentation_status = "partial"
            object_source_name = "raw_cloud"
            object_cloud = loaded.get("raw_cloud")
        else:
            warnings.append("segmentation_not_available")

    if manifest_status in {"skipped", "failed"} and not segmentation_used:
        segmentation_used = False
        object_source_name = "raw_cloud"
        object_cloud = loaded.get("raw_cloud")

    if segmentation_used and segmentation_status not in {"partial", "ok"}:
        segmentation_status = "ok"

    pot_dimensions = None
    leaf_dimensions = None
    leaf_meta: dict[str, Any] = {}
    profiles: dict[str, Any] = {}
    if segmentation_used:
        if "pot_cloud" in loaded and pot_quality.get("usable_for_geometry") is True:
            if segment_statuses.get("pot") == "fallback":
                warnings.append("pot_geometry_from_fallback_mask")
            pot_dimensions, pot_meta, pot_profile = _measure_segment_dimensions(
                o3d=o3d,
                cloud=loaded["pot_cloud"],
                cfg=active_cfg,
                warnings=warnings,
                warning_prefix="pot_segment",
            )
            metadata["pot_segment"] = pot_meta
            if pot_profile is not None:
                profiles["pot_cloud"] = pot_profile
        elif segment_statuses.get("pot") in {"rejected", "fallback"} or pot_quality.get("usable_for_geometry") is False:
            warnings.append("pot_mask_rejected_not_used_for_geometry")
            metadata["pot_segment"] = {
                "skipped": True,
                "reason": "pot_quality_not_usable_for_geometry",
                "pot_quality_status": pot_quality.get("status"),
            }
        elif "leaf_cloud" in loaded:
            warnings.append("pot_segment_missing")

        if "leaf_cloud" in loaded:
            if segment_statuses.get("leaf") == "fallback":
                warnings.append("leaf_geometry_from_fallback_mask")
            leaf_dimensions, leaf_meta, leaf_profile = _measure_segment_dimensions(
                o3d=o3d,
                cloud=loaded["leaf_cloud"],
                cfg=active_cfg,
                warnings=warnings,
                warning_prefix="leaf_segment",
            )
            metadata["leaf_segment"] = leaf_meta
            if leaf_profile is not None:
                profiles["leaf_cloud"] = leaf_profile
        elif "pot_cloud" in loaded:
            warnings.append("leaf_segment_missing")

        if "table_cloud" not in loaded and any(
            name in loaded for name in ("object_cloud", "segmented_cloud", "pot_cloud", "leaf_cloud")
        ):
            warnings.append("table_segment_missing_using_ransac")

    if object_source_name == "raw_cloud" and segmentation_used:
        warnings.append("segmentation_partial_using_raw_cloud")
    warnings.extend(
        str(warning)
        for warning in load_metadata.get(object_source_name, {}).get("warnings", [])
    )

    object_dimensions = None
    object_meta: dict[str, Any] = {}
    table_record = None
    object_reason = None
    if object_cloud is not None and len(object_cloud.points):
        if debug_mode:
            _write_cloud(debug_paths["input_cloud_used"], object_cloud)
        trusted_direct_object_cloud = (
            object_source_name == "object_cloud"
            and load_metadata.get("object_cloud", {}).get("trusted_metric_units") is True
        )
        if trusted_direct_object_cloud:
            object_dimensions, object_meta, table_record, object_reason, object_profile = _measure_trusted_object_cloud_direct(
                cloud=object_cloud,
                warnings=warnings,
                debug_paths=debug_paths,
                write_debug=debug_mode,
            )
        else:
            object_dimensions, object_meta, table_record, object_reason, object_profile = _measure_object_cloud(
                o3d=o3d,
                cloud=object_cloud,
                cfg=active_cfg,
                warnings=warnings,
                debug_paths=debug_paths,
                write_debug=debug_mode,
            )
        metadata.update(object_meta)
        if object_profile is not None:
            profiles["object_cloud"] = object_profile
    else:
        object_reason = f"Missing source cloud: {segment_paths[object_source_name]}"
        warnings.append("missing_source_cloud")

    if object_dimensions is None and pot_dimensions is None and leaf_dimensions is None:
        cloud_quality = {
            "object": _cloud_quality_record(
                label="object",
                dimensions=object_dimensions,
                metadata=object_meta,
                source=object_source_name,
                fallback_mask_used=segment_statuses.get("object") == "fallback",
            ),
            "leaf": _cloud_quality_record(
                label="leaf",
                dimensions=leaf_dimensions,
                metadata=leaf_meta,
                source="fallback_mask" if segment_statuses.get("leaf") == "fallback" else "segmented_mask",
                fallback_mask_used=segment_statuses.get("leaf") == "fallback",
            ),
        }
        metadata["cloud_quality"] = cloud_quality
        geometry = _failed_geometry(
            reason=object_reason or "No usable geometry cloud found.",
            warnings=warnings,
            debug_paths=debug_paths,
            metadata=metadata,
            source_cloud=object_source_name,
            cloud_type=cloud_type,
            segmentation_used=segmentation_used,
            segmentation_status=segmentation_status,
            counts=counts,
        )
        if debug_mode:
            _write_cloud_quality(debug_paths["cloud_quality"], cloud_quality)
            _write_geometry_debug(
                debug_paths["geometry_debug"],
                {
                    "status": geometry["status"],
                    "reason": geometry["reason"],
                    "warnings": geometry["warnings"],
                    "metadata": metadata,
                    "cloud_quality": cloud_quality,
                    "table": table_record,
                },
            )
        return geometry

    if isinstance(profiles.get("leaf_cloud"), dict):
        object_profile = profiles.get("object_cloud", {})
        leaf_profile = profiles["leaf_cloud"]
        object_points = object_profile.get("point_count") or 0
        leaf_points = leaf_profile.get("point_count") or 0
        object_volume = (
            (object_profile.get("bbox_length_in") or 0)
            * (object_profile.get("bbox_width_in") or 0)
            * (object_profile.get("bbox_height_in") or 0)
        )
        leaf_volume = (
            (leaf_profile.get("bbox_length_in") or 0)
            * (leaf_profile.get("bbox_width_in") or 0)
            * (leaf_profile.get("bbox_height_in") or 0)
        )
        leaf_profile["canopy_width_in"] = leaf_profile.get("bbox_length_in")
        leaf_profile["canopy_depth_in"] = leaf_profile.get("bbox_width_in")
        leaf_profile["canopy_height_in"] = leaf_profile.get("bbox_height_in")
        leaf_profile["canopy_center_z_in"] = leaf_profile.get("density_center_of_mass_z")
        leaf_profile["leaf_to_object_point_ratio"] = (
            float(leaf_points / object_points) if object_points else None
        )
        leaf_profile["leaf_to_object_volume_ratio"] = (
            float(leaf_volume / object_volume) if object_volume else None
        )

    if segmentation_used:
        has_partial_segments = ("pot_cloud" in loaded) != ("leaf_cloud" in loaded)
        if has_partial_segments or object_source_name == "raw_cloud":
            segmentation_status = "partial"
    elif segmentation_status == "none":
        warnings.append("segmentation_not_available")

    reason = None if object_dimensions is not None else object_reason
    cloud_quality = {
        "object": _cloud_quality_record(
            label="object",
            dimensions=object_dimensions,
            metadata=object_meta,
            source=object_source_name,
            fallback_mask_used=segment_statuses.get("object") == "fallback",
        ),
        "leaf": _cloud_quality_record(
            label="leaf",
            dimensions=leaf_dimensions,
            metadata=leaf_meta,
            source="fallback_mask" if segment_statuses.get("leaf") == "fallback" else "segmented_mask",
            fallback_mask_used=segment_statuses.get("leaf") == "fallback",
        ),
    }
    metadata["cloud_quality"] = cloud_quality
    for cloud_name, record in cloud_quality.items():
        if not isinstance(record, dict):
            continue
        for warning in record.get("warnings", []):
            warning_text = str(warning)
            warnings.append(
                warning_text if warning_text.startswith(f"{cloud_name}_") else f"{cloud_name}_{warning_text}"
            )
    status = _geometry_status_from_warnings(reason, warnings)
    geometry = _make_geometry(
        status=status,
        reason=reason,
        source_cloud=object_source_name,
        cloud_type=cloud_type,
        segmentation_used=segmentation_used,
        segmentation_status=segmentation_status,
        object_dimensions=object_dimensions,
        pot_dimensions=pot_dimensions,
        leaf_dimensions=leaf_dimensions,
        table_plane=table_record,
        counts=counts,
        confidence=confidence,
        warnings=warnings,
        debug_paths=debug_paths,
        metadata=metadata,
        pot_quality=pot_quality,
        profiles=profiles,
        cloud_quality=cloud_quality,
    )
    if debug_mode:
        _write_cloud_quality(debug_paths["cloud_quality"], cloud_quality)

        _write_geometry_debug(
            debug_paths["geometry_debug"],
            {
                "status": status,
                "warnings": geometry["warnings"],
                "metadata": metadata,
                "object_dimensions": object_dimensions,
                "pot_dimensions": pot_dimensions,
                "leaf_dimensions": leaf_dimensions,
                "pot_quality": pot_quality,
                "cloud_quality": cloud_quality,
                "profiles": profiles,
                "table": table_record,
            },
        )
    return geometry
