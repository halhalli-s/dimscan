"""Extract dictionary-based ML features from DimScan geometry and job files."""

from __future__ import annotations

import math
import re
from typing import Any

from app.config import DimScanConfig
from features.schema import make_view_features
from features.validators import as_bool_available, as_float_or_none, get_nested
from metadata.parser import is_square_arrangement, normalize_arrangement_type, parse_arrangement_dims, parse_pot_prior
from metadata.sku_catalogue import pot_prior_available, pot_prior_model_confidence
from geometry.utils import max_or_none
from pipeline.session import mark_step, set_status
from utils.io import read_json_if_exists, write_json_atomic
from utils.paths import get_job_file_path, get_view_file_path


POT_DEBUG_FILENAME = "pot_debug.json"
COMBINED_DEBUG_FILENAME = "combined_features_debug.json"
GEOMETRY_SEMANTIC_DEBUG_FILENAME = "geometry_semantic_debug.json"
M_TO_IN = 39.3701
GROUP_COMPOSITION_VERSION = "v1"


LEAF_AI2_SUPPRESSED_FIELDS = (
    "leaf_canopy_length_in",
    "leaf_canopy_width_in",
    "leaf_canopy_height_in",
    "leaf_canopy_point_count",
    "leaf_canopy_length_robust_in",
    "leaf_canopy_width_robust_in",
    "leaf_canopy_depth_robust_in",
    "leaf_canopy_height_robust_in",
    "leaf_point_count",
    "leaf_bbox_length_in",
    "leaf_bbox_width_in",
    "leaf_bbox_height_in",
    "leaf_vertical_density_bins",
    "leaf_occupied_bin_count",
    "leaf_max_bin_density",
    "leaf_density_center_of_mass_y",
    "leaf_compactness_point_count_per_bbox_volume",
    "leaf_y_p10",
    "leaf_y_p25",
    "leaf_y_p50",
    "leaf_y_p75",
    "leaf_y_p90",
    "leaf_y_p95",
    "leaf_y_p99",
    "leaf_lower_third_width_in",
    "leaf_lower_third_depth_in",
    "leaf_middle_third_width_in",
    "leaf_middle_third_depth_in",
    "leaf_upper_third_width_in",
    "leaf_upper_third_depth_in",
    "leaf_canopy_depth_in",
    "leaf_canopy_center_y_in",
    "leaf_leaf_to_object_point_ratio",
    "leaf_leaf_to_object_volume_ratio",
)


def _as_warning_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _leaf_from_fallback(geometry: dict[str, Any]) -> bool:
    segments = get_nested(geometry, ["metadata", "segmentation_manifest", "segments"]) or {}
    warnings = _as_warning_list(geometry.get("warnings"))
    segmentation_warnings = _as_warning_list(get_nested(geometry, ["metadata", "segmentation_manifest", "warnings"]))
    leaf_quality = get_nested(geometry, ["cloud_quality", "leaf"]) or {}
    return (
        (isinstance(segments, dict) and segments.get("leaf") == "fallback")
        or "leaf_from_fallback_mask" in warnings
        or "leaf_geometry_from_fallback_mask" in warnings
        or "leaf_mask_fallback_from_object" in segmentation_warnings
        or (isinstance(leaf_quality, dict) and bool(leaf_quality.get("fallback_mask_used")))
        or (isinstance(leaf_quality, dict) and leaf_quality.get("source") == "fallback_mask")
    )


def _suppress_leaf_ai2_features(ai2_features: dict[str, Any]) -> None:
    ai2_features["leaf_available"] = False
    ai2_features["leaf_profile_available"] = False
    for key in LEAF_AI2_SUPPRESSED_FIELDS:
        ai2_features[key] = None


def _profile_features(prefix: str, profile: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {
        f"{prefix}_profile_available": bool(profile.get("available")),
        f"{prefix}_point_count": profile.get("point_count"),
        f"{prefix}_vertical_density_bins": profile.get("vertical_density_bins", []),
        f"{prefix}_occupied_bin_count": profile.get("occupied_bin_count"),
        f"{prefix}_max_bin_density": profile.get("max_bin_density"),
        f"{prefix}_density_center_of_mass_y": as_float_or_none(profile.get("density_center_of_mass_z")),
        f"{prefix}_compactness_point_count_per_bbox_volume": profile.get(
            "compactness_point_count_per_bbox_volume"
        ),
    }
    if prefix != "object":
        values.update(
            {
                f"{prefix}_bbox_length_in": as_float_or_none(profile.get("bbox_length_in")),
                f"{prefix}_bbox_width_in": as_float_or_none(profile.get("bbox_width_in")),
                f"{prefix}_bbox_height_in": as_float_or_none(profile.get("bbox_height_in")),
            }
        )
    percentiles = profile.get("height_percentiles")
    if isinstance(percentiles, dict):
        for key, value in percentiles.items():
            output_key = key.replace("z_p", "y_p", 1) if key.startswith("z_p") else key
            values[f"{prefix}_{output_key}"] = as_float_or_none(value)
    thirds = profile.get("thirds")
    if isinstance(thirds, dict):
        for third_name, third_values in thirds.items():
            if not isinstance(third_values, dict):
                continue
            values[f"{prefix}_{third_name}_width_in"] = as_float_or_none(third_values.get("width_in"))
            values[f"{prefix}_{third_name}_depth_in"] = as_float_or_none(third_values.get("depth_in"))
    for key in (
        "canopy_width_in",
        "canopy_depth_in",
        "canopy_height_in",
        "leaf_to_object_point_ratio",
        "leaf_to_object_volume_ratio",
    ):
        if key in profile:
            values[f"{prefix}_{key}"] = as_float_or_none(profile.get(key))
    if "canopy_center_z_in" in profile:
        values[f"{prefix}_canopy_center_y_in"] = as_float_or_none(profile.get("canopy_center_z_in"))
    return values


def _quality_flags(geometry: dict[str, Any]) -> dict[str, Any]:
    pot_quality = geometry.get("pot_quality") if isinstance(geometry.get("pot_quality"), dict) else {}
    pot_trusted = pot_quality.get("status") == "trusted" and bool(pot_quality.get("usable_for_model"))
    segments = get_nested(geometry, ["metadata", "segmentation_manifest", "segments"]) or {}
    leaf_fallback = _leaf_from_fallback(geometry)
    warnings = _as_warning_list(geometry.get("warnings"))
    object_dimensions = _object_dimensions(geometry)
    trusted_rgb_meter_object = (
        geometry.get("source_cloud") == "object_cloud"
        and object_dimensions.get("point_cloud_frame") == "rgb_camera"
        and object_dimensions.get("point_cloud_units") == "meters"
    )
    if trusted_rgb_meter_object:
        warnings = [warning for warning in warnings if warning != "metric_unit_assumed_millimeters"]
    if leaf_fallback and "leaf_features_suppressed_due_to_fallback" not in warnings:
        warnings = [*warnings, "leaf_features_suppressed_due_to_fallback"]
    return {
        "object_available": bool(object_dimensions),
        "leaf_available": isinstance(geometry.get("leaf_canopy_dimensions_in"), dict) and not leaf_fallback,
        "table_available": bool(geometry.get("table_plane")),
        "pot_available": pot_trusted,
        "pot_source": "segmentation_trusted" if pot_trusted else "missing_or_rejected",
        "pot_quality_status": pot_quality.get("status", "missing"),
        "pot_trust_score": pot_quality.get("trust_score", 0.0),
        "pot_usable_for_model": pot_trusted,
        "pot_prior_available": False,
        "pot_prior_source": "missing",
        "pot_prior_confidence": "none",
        "segmentation_status": geometry.get("segmentation_status"),
        "geometry_status": geometry.get("status"),
        "segments": segments,
        "warnings": warnings,
    }


def _object_dimensions(geometry: dict[str, Any]) -> dict[str, Any]:
    for key in ("object_dimensions_in", "dimensions_in", "scene"):
        dimensions = geometry.get(key)
        if isinstance(dimensions, dict) and dimensions:
            return dimensions
    return {}


def _ai2_and_debug_features(geometry: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    object_dimensions = _object_dimensions(geometry)
    leaf_dimensions = (
        geometry.get("leaf_canopy_dimensions_in")
        if isinstance(geometry.get("leaf_canopy_dimensions_in"), dict)
        else {}
    )
    pot_dimensions = geometry.get("pot_dimensions_in") if isinstance(geometry.get("pot_dimensions_in"), dict) else {}
    pot_quality = geometry.get("pot_quality") if isinstance(geometry.get("pot_quality"), dict) else {}
    profiles = geometry.get("profiles") if isinstance(geometry.get("profiles"), dict) else {}
    pot_usable = bool(pot_quality.get("usable_for_model"))
    leaf_fallback = _leaf_from_fallback(geometry)
    pot_diameter = max_or_none(
        [
            as_float_or_none(pot_dimensions.get("length_in")),
            as_float_or_none(pot_dimensions.get("width_in")),
        ]
    )

    ai2_features: dict[str, Any] = {
        "object_available": bool(object_dimensions),
        "object_length_in": as_float_or_none(object_dimensions.get("length_in")),
        "object_width_in": as_float_or_none(object_dimensions.get("width_in")),
        "object_height_in": as_float_or_none(object_dimensions.get("height_in")),
        "object_point_count": object_dimensions.get("point_count"),
        "leaf_available": bool(leaf_dimensions) and not leaf_fallback,
        "leaf_canopy_length_in": as_float_or_none(leaf_dimensions.get("length_in")),
        "leaf_canopy_width_in": as_float_or_none(leaf_dimensions.get("width_in")),
        "leaf_canopy_height_in": as_float_or_none(leaf_dimensions.get("height_in")),
        "leaf_canopy_point_count": leaf_dimensions.get("point_count"),
        "leaf_canopy_length_robust_in": as_float_or_none(leaf_dimensions.get("length_robust_in")),
        "leaf_canopy_width_robust_in": as_float_or_none(leaf_dimensions.get("width_robust_in")),
        "leaf_canopy_depth_robust_in": as_float_or_none(leaf_dimensions.get("width_robust_in")),
        "leaf_canopy_height_robust_in": as_float_or_none(leaf_dimensions.get("height_robust_in")),
        "table_available": bool(geometry.get("table_plane")),
        "pot_diameter_in": pot_diameter if pot_usable else None,
        "pot_height_in": as_float_or_none(pot_dimensions.get("height_in")) if pot_usable else None,
    }
    if isinstance(profiles.get("object_cloud"), dict):
        ai2_features.update(_profile_features("object", profiles["object_cloud"]))
    if isinstance(profiles.get("leaf_cloud"), dict) and not leaf_fallback:
        ai2_features.update(_profile_features("leaf", profiles["leaf_cloud"]))
    if leaf_fallback:
        _suppress_leaf_ai2_features(ai2_features)

    debug_features = {
        "pot_quality": pot_quality,
        "leaf_features_suppressed_due_to_fallback": leaf_fallback,
        "rejected_pot_dimensions_in": pot_dimensions if not pot_usable else None,
        "raw_pot_candidate_measurements": get_nested(geometry, ["metadata", "pot_segment"]),
        "segmentation_confidence": geometry.get("confidence", {}),
        "geometry_metadata_clouds": get_nested(geometry, ["metadata", "clouds"]),
    }
    return ai2_features, _quality_flags(geometry), debug_features


def _has_explicit_rgb_spans(values: dict[str, Any]) -> bool:
    return any(
        values.get(key) is not None
        for key in (
            "object_horizontal_span_m",
            "object_vertical_span_m",
            "object_depth_thickness_m",
            "object_horizontal_span_in",
            "object_vertical_span_in",
            "object_depth_thickness_in",
        )
    )


def _require_rgb_meter_spans(values: dict[str, Any], *, view_name: str) -> None:
    if not _has_explicit_rgb_spans(values):
        return
    frame = values.get("point_cloud_frame")
    units = values.get("point_cloud_units")
    if frame != "rgb_camera" or units != "meters":
        raise ValueError(
            "object_dimension_unit_contract_invalid:"
            f"view={view_name}:frame={frame or 'missing'}:units={units or 'missing'}"
        )


def _span_in(values: dict[str, Any], key: str, fallback_key: str | None = None) -> float | None:
    explicit_in = as_float_or_none(values.get(f"object_{key}_in"))
    if explicit_in is not None:
        return explicit_in
    explicit_m = as_float_or_none(values.get(f"object_{key}_m"))
    if explicit_m is not None:
        return explicit_m * M_TO_IN
    if fallback_key:
        return as_float_or_none(values.get(fallback_key))
    return None


def _view_span_record(view_feature: dict[str, Any]) -> dict[str, Any]:
    values = view_feature.get("features") if isinstance(view_feature.get("features"), dict) else {}
    view_name = str(view_feature.get("view_name") or "")
    _require_rgb_meter_spans(values, view_name=view_name)
    return {
        "view_name": view_name,
        "view_role": view_feature.get("view_role"),
        "point_cloud_path": values.get("point_cloud_path"),
        "point_cloud_frame": values.get("point_cloud_frame"),
        "point_cloud_units": values.get("point_cloud_units"),
        "x_min_m": as_float_or_none(values.get("object_x_min_m")),
        "y_min_m": as_float_or_none(values.get("object_y_min_m")),
        "z_min_m": as_float_or_none(values.get("object_z_min_m")),
        "x_max_m": as_float_or_none(values.get("object_x_max_m")),
        "y_max_m": as_float_or_none(values.get("object_y_max_m")),
        "z_max_m": as_float_or_none(values.get("object_z_max_m")),
        "x_span_m": as_float_or_none(values.get("object_horizontal_span_m")),
        "y_span_m": as_float_or_none(values.get("object_vertical_span_m")),
        "z_span_m": as_float_or_none(values.get("object_depth_thickness_m")),
        "x_span_in": as_float_or_none(values.get("object_x_span_in")),
        "y_span_in": as_float_or_none(values.get("object_y_span_in")),
        "z_span_in": as_float_or_none(values.get("object_z_span_in")),
        "horizontal_span_in": _span_in(values, "horizontal_span", "object_length_in"),
        "vertical_span_in": _span_in(values, "vertical_span", "object_height_in"),
        "depth_thickness_in": _span_in(values, "depth_thickness", "object_width_in"),
        "horizontal_span_robust_in": _span_in(values, "robust_horizontal_span", "object_length_robust_in"),
        "vertical_span_robust_in": _span_in(values, "robust_vertical_span", "object_height_robust_in"),
        "depth_thickness_robust_in": _span_in(values, "robust_depth_thickness", "object_width_robust_in"),
        "point_count": values.get("object_point_count"),
    }


def _arrangement_kind(job_metadata: dict[str, Any]) -> tuple[str, str, bool]:
    arrangement_type = normalize_arrangement_type(str(job_metadata.get("arrangement_type") or "1x1"))
    view_mode = str(job_metadata.get("view_mode") or "")
    square = False if view_mode == "two_view_rectangle" else (
        view_mode == "single_view" or is_square_arrangement(arrangement_type)
    )
    return arrangement_type, view_mode, square


def combine_object_dimensions_for_arrangement(
    *,
    job_metadata: dict[str, Any],
    view_features: list[dict[str, Any]],
) -> dict[str, Any]:
    """Map RGB-camera raw spans into semantic object length/width/height."""
    arrangement_type, view_mode, square = _arrangement_kind(job_metadata)
    per_view = [_view_span_record(view_feature) for view_feature in view_features]
    if not per_view:
        return {
            "arrangement_type": arrangement_type,
            "view_mode": view_mode,
            "view_count": 0,
            "per_view": [],
            "semantic_dimensions_in": {},
            "semantic_mapping": {},
            "square_width_equals_length": False,
            "z_used_as_physical_width": False,
            "units_conversion_factor": M_TO_IN,
        }

    if square:
        source = per_view[0]
        source_uses_robust = source["horizontal_span_robust_in"] is not None
        length = source["horizontal_span_robust_in"] if source_uses_robust else source["horizontal_span_in"]
        width = length
        height = source["vertical_span_in"]
        length_robust = source["horizontal_span_robust_in"]
        width_robust = length_robust
        height_robust = source["vertical_span_robust_in"]
        semantic_mapping = {
            "policy": "square_single_view",
            "length_source": (
                f"{source['view_name']}.horizontal_span_robust_1_99"
                if source_uses_robust
                else f"{source['view_name']}.horizontal_span"
            ),
            "width_source": (
                f"{source['view_name']}.horizontal_span_robust_1_99"
                if source_uses_robust
                else f"{source['view_name']}.horizontal_span"
            ),
            "height_source": f"{source['view_name']}.vertical_span",
            "raw_length_debug_source": f"{source['view_name']}.horizontal_span",
            "raw_width_debug_source": f"{source['view_name']}.horizontal_span",
            "depth_thickness_source": f"{source['view_name']}.depth_thickness_debug_only",
            "horizontal_robust_percentiles": [1.0, 99.0],
        }
        if length is not None and width is not None:
            assert abs(length - width) < 1e-9
        if height is not None and source["vertical_span_in"] is not None:
            assert abs(height - source["vertical_span_in"]) < 1e-9
        if width is not None and source["depth_thickness_in"] is not None:
            assert abs(width - source["depth_thickness_in"]) > 1e-9 or abs(width - length) < 1e-9
    else:
        by_role = {record["view_role"]: record for record in per_view}
        length_source = by_role.get("length_view") or per_view[0]
        width_source = by_role.get("width_view")
        if width_source is None:
            raise ValueError("rectangular_arrangement_requires_width_view")
        length_uses_robust = length_source["horizontal_span_robust_in"] is not None
        width_uses_robust = width_source["horizontal_span_robust_in"] is not None
        length = (
            length_source["horizontal_span_robust_in"]
            if length_uses_robust
            else length_source["horizontal_span_in"]
        )
        width = (
            width_source["horizontal_span_robust_in"]
            if width_uses_robust
            else width_source["horizontal_span_in"]
        )
        height = max_or_none([record["vertical_span_in"] for record in per_view])
        length_robust = length_source["horizontal_span_robust_in"]
        width_robust = width_source["horizontal_span_robust_in"]
        height_robust = max_or_none([record["vertical_span_robust_in"] for record in per_view])
        semantic_mapping = {
            "policy": "rectangular_two_view",
            "length_source": (
                f"{length_source['view_name']}.horizontal_span_robust_1_99"
                if length_uses_robust
                else f"{length_source['view_name']}.horizontal_span"
            ),
            "width_source": (
                f"{width_source['view_name']}.horizontal_span_robust_1_99"
                if width_uses_robust
                else f"{width_source['view_name']}.horizontal_span"
            ),
            "height_source": "max(per_view.vertical_span)",
            "raw_length_debug_source": f"{length_source['view_name']}.horizontal_span",
            "raw_width_debug_source": f"{width_source['view_name']}.horizontal_span",
            "depth_thickness_source": "debug_only_not_physical_width",
            "horizontal_robust_percentiles": [1.0, 99.0],
        }
        assert length_source["view_name"] == "view_01" or length_source["view_role"] == "length_view"
        assert width_source["view_name"] == "view_02" or width_source["view_role"] == "width_view"

    dimensions = {
        "length_in": length,
        "width_in": width,
        "height_in": height,
        "length_robust_in": length_robust,
        "width_robust_in": width_robust,
        "height_robust_in": height_robust,
        "point_count": per_view[0].get("point_count"),
    }
    return {
        "arrangement_type": arrangement_type,
        "view_mode": view_mode,
        "view_count": len(view_features),
        "per_view": per_view,
        "semantic_dimensions_in": dimensions,
        "semantic_mapping": semantic_mapping,
        "square_width_equals_length": (
            bool(square)
            and length is not None
            and width is not None
            and abs(float(length) - float(width)) < 1e-9
        ),
        "z_used_as_physical_width": False,
        "units_conversion_factor": M_TO_IN,
    }


def _apply_object_semantics(
    feature_values: dict[str, Any],
    semantic_dimensions: dict[str, Any],
) -> None:
    for target, source in (
        ("object_length_in", "length_in"),
        ("object_width_in", "width_in"),
        ("object_height_in", "height_in"),
        ("object_length_robust_in", "length_robust_in"),
        ("object_width_robust_in", "width_robust_in"),
        ("object_height_robust_in", "height_robust_in"),
        ("length_in", "length_in"),
        ("width_in", "width_in"),
        ("height_in", "height_in"),
        ("view_width_in", "width_in"),
        ("view_height_in", "height_in"),
        ("view_depth_span_in", "length_in"),
    ):
        value = semantic_dimensions.get(source)
        if value is not None:
            feature_values[target] = value
    ai2 = feature_values.get("ai2_features")
    if isinstance(ai2, dict):
        for target, source in (
            ("object_length_in", "length_in"),
            ("object_width_in", "width_in"),
            ("object_height_in", "height_in"),
        ):
            value = semantic_dimensions.get(source)
            if value is not None:
                ai2[target] = value


def extract_view_feature_values(geometry: dict[str, Any]) -> dict[str, Any]:
    """Extract flat per-view feature values from a geometry dictionary."""
    object_dimensions = _object_dimensions(geometry)
    pot_dimensions = geometry.get("pot_dimensions_in")
    if not isinstance(pot_dimensions, dict):
        pot_dimensions = {}
    leaf_dimensions = geometry.get("leaf_canopy_dimensions_in")
    if not isinstance(leaf_dimensions, dict):
        leaf_dimensions = {}

    visible_pot_top_diameter = as_float_or_none(
        get_nested(geometry, ["pot", "visible_top_diameter_in"])
    )
    visible_pot_height = as_float_or_none(get_nested(geometry, ["pot", "visible_height_in"]))

    ai2_features, quality_flags, debug_features = _ai2_and_debug_features(geometry)
    values = {
        "geometry_status": geometry.get("status"),
        "geometry_reason": geometry.get("reason"),
        "geometry_warnings": quality_flags.get("warnings", []),
        "geometry_source_cloud": geometry.get("source_cloud")
        or get_nested(geometry, ["metadata", "source_cloud"]),
        "geometry_cloud_type": geometry.get("cloud_type") or get_nested(geometry, ["metadata", "cloud_type"]),
        "point_cloud_path": get_nested(
            geometry,
            ["metadata", "clouds", geometry.get("source_cloud") or get_nested(geometry, ["metadata", "source_cloud"]), "path"],
        ),
        "point_cloud_frame": object_dimensions.get("point_cloud_frame"),
        "point_cloud_units": object_dimensions.get("point_cloud_units"),
        "segmentation_used": geometry.get("segmentation_used"),
        "segmentation_status": geometry.get("segmentation_status"),
        "object_x_min_m": as_float_or_none(object_dimensions.get("x_min_m")),
        "object_y_min_m": as_float_or_none(object_dimensions.get("y_min_m")),
        "object_z_min_m": as_float_or_none(object_dimensions.get("z_min_m")),
        "object_x_max_m": as_float_or_none(object_dimensions.get("x_max_m")),
        "object_y_max_m": as_float_or_none(object_dimensions.get("y_max_m")),
        "object_z_max_m": as_float_or_none(object_dimensions.get("z_max_m")),
        "object_x_span_m": as_float_or_none(object_dimensions.get("x_span_m")),
        "object_y_span_m": as_float_or_none(object_dimensions.get("y_span_m")),
        "object_z_span_m": as_float_or_none(object_dimensions.get("z_span_m")),
        "object_x_span_in": as_float_or_none(object_dimensions.get("x_span_in")),
        "object_y_span_in": as_float_or_none(object_dimensions.get("y_span_in")),
        "object_z_span_in": as_float_or_none(object_dimensions.get("z_span_in")),
        "object_horizontal_span_m": as_float_or_none(object_dimensions.get("horizontal_span_m")),
        "object_vertical_span_m": as_float_or_none(object_dimensions.get("vertical_span_m")),
        "object_depth_thickness_m": as_float_or_none(object_dimensions.get("depth_thickness_m")),
        "object_horizontal_span_in": as_float_or_none(object_dimensions.get("horizontal_span_in")),
        "object_vertical_span_in": as_float_or_none(object_dimensions.get("vertical_span_in")),
        "object_depth_thickness_in": as_float_or_none(object_dimensions.get("depth_thickness_in")),
        "object_robust_horizontal_span_m": as_float_or_none(object_dimensions.get("robust_horizontal_span_m")),
        "object_robust_vertical_span_m": as_float_or_none(object_dimensions.get("robust_vertical_span_m")),
        "object_robust_depth_thickness_m": as_float_or_none(object_dimensions.get("robust_depth_thickness_m")),
        "object_robust_horizontal_span_in": as_float_or_none(object_dimensions.get("robust_horizontal_span_in")),
        "object_robust_vertical_span_in": as_float_or_none(object_dimensions.get("robust_vertical_span_in")),
        "object_robust_depth_thickness_in": as_float_or_none(object_dimensions.get("robust_depth_thickness_in")),
        "object_length_in": as_float_or_none(object_dimensions.get("length_in")),
        "object_width_in": as_float_or_none(object_dimensions.get("width_in")),
        "object_height_in": as_float_or_none(object_dimensions.get("height_in")),
        "object_point_count": object_dimensions.get("point_count"),
        "pot_length_in": as_float_or_none(pot_dimensions.get("length_in")),
        "pot_width_in": as_float_or_none(pot_dimensions.get("width_in")),
        "pot_height_in": as_float_or_none(pot_dimensions.get("height_in")),
        "pot_segment_point_count": pot_dimensions.get("point_count"),
        "leaf_canopy_length_in": as_float_or_none(leaf_dimensions.get("length_in")),
        "leaf_canopy_width_in": as_float_or_none(leaf_dimensions.get("width_in")),
        "leaf_canopy_height_in": as_float_or_none(leaf_dimensions.get("height_in")),
        "leaf_canopy_point_count": leaf_dimensions.get("point_count"),
        "leaf_canopy_length_robust_in": as_float_or_none(leaf_dimensions.get("length_robust_in")),
        "leaf_canopy_width_robust_in": as_float_or_none(leaf_dimensions.get("width_robust_in")),
        "leaf_canopy_depth_robust_in": as_float_or_none(leaf_dimensions.get("width_robust_in")),
        "leaf_canopy_height_robust_in": as_float_or_none(leaf_dimensions.get("height_robust_in")),
        "view_height_in": as_float_or_none(get_nested(geometry, ["scene", "height_in"])),
        "view_width_in": as_float_or_none(get_nested(geometry, ["scene", "width_in"])),
        "view_depth_span_in": as_float_or_none(get_nested(geometry, ["scene", "length_in"])),
        "length_in": as_float_or_none(get_nested(geometry, ["dimensions_in", "length_in"])),
        "width_in": as_float_or_none(get_nested(geometry, ["dimensions_in", "width_in"])),
        "height_in": as_float_or_none(get_nested(geometry, ["dimensions_in", "height_in"])),
        "scene_point_count": get_nested(geometry, ["scene", "point_count"]),
        "plant_height_in": as_float_or_none(get_nested(geometry, ["plant", "height_in"])),
        "plant_width_in": as_float_or_none(get_nested(geometry, ["plant", "width_in"])),
        "plant_point_count": get_nested(geometry, ["plant", "point_count"]),
        "pot_point_count": get_nested(geometry, ["pot", "point_count"]),
        "visible_pot_top_diameter_in": visible_pot_top_diameter,
        "visible_pot_top_diameter_available": as_bool_available(visible_pot_top_diameter),
        "visible_pot_height_in": visible_pot_height,
        "visible_pot_height_available": as_bool_available(visible_pot_height),
    }
    values["ai2_features"] = ai2_features
    values["quality_flags"] = quality_flags
    values["debug_features"] = debug_features
    return values


def _model_feature_payload(
    *,
    cfg: DimScanConfig,
    ai2_features: dict[str, Any],
    quality_flags: dict[str, Any],
    feature_schema_version: str | None = None,
) -> dict[str, Any]:
    return {
        "feature_schema_version": feature_schema_version or cfg.feature_schema_version,
        "ai2_features": ai2_features,
        "quality_flags": quality_flags,
    }


def _pot_debug_payload(debug_features: dict[str, Any]) -> dict[str, Any]:
    pot_quality = debug_features.get("pot_quality")
    if not isinstance(pot_quality, dict):
        pot_quality = {}
    metrics = pot_quality.get("metrics") if isinstance(pot_quality.get("metrics"), dict) else {}
    usable = bool(pot_quality.get("usable_for_model"))
    return {
        "pot_quality": pot_quality,
        "rejected_pot_dimensions_in": debug_features.get("rejected_pot_dimensions_in"),
        "raw_pot_candidate_measurements": debug_features.get("raw_pot_candidate_measurements"),
        "pot_rejection_reason": None if usable else pot_quality.get("reason"),
        "pot_validation_metrics": metrics,
        "yoloe_pot_candidate_summary": {
            "confidence": metrics.get("pot_confidence"),
            "mask_pixel_count": metrics.get("pot_mask_pixel_count"),
            "bbox_xyxy": metrics.get("pot_bbox_xyxy"),
            "visible_width_px": metrics.get("visible_pot_width_px"),
            "visible_height_px": metrics.get("visible_pot_height_px"),
            "point_count_in_pot_cloud": metrics.get("point_count_in_pot_cloud"),
        },
        "segmentation_confidence": debug_features.get("segmentation_confidence", {}),
        "geometry_metadata_clouds": debug_features.get("geometry_metadata_clouds", {}),
    }


def _write_pot_debug(cfg: DimScanConfig, job_type: str, job_id: str, view_name: str, debug_features: dict[str, Any]) -> None:
    debug_path = get_view_file_path(cfg, job_type, job_id, view_name, "debug") / POT_DEBUG_FILENAME
    write_json_atomic(debug_path, _pot_debug_payload(debug_features))


def write_view_features(
    cfg: DimScanConfig,
    job_type: str,
    job_id: str,
    *,
    view_name: str,
    view_role: str,
    debug_mode: bool = True,
) -> dict[str, Any]:
    """Read view geometry, write view features, and mark the session step."""
    geometry_path = get_view_file_path(
        cfg,
        job_type,
        job_id,
        view_name,
        cfg.geometry_filename,
    )
    geometry = read_json_if_exists(geometry_path)
    if geometry is None:
        raise FileNotFoundError(f"Missing geometry file: {geometry_path}")
    if not isinstance(geometry, dict):
        raise ValueError(f"Geometry file must contain an object: {geometry_path}")

    feature_values = extract_view_feature_values(geometry)
    metadata_path = get_job_file_path(cfg, job_type, job_id, cfg.job_metadata_filename)
    job_metadata = read_json_if_exists(metadata_path, default={})
    if isinstance(job_metadata, dict):
        arrangement_type, view_mode, square = _arrangement_kind(job_metadata)
        if square and _has_explicit_rgb_spans(feature_values):
            semantic = combine_object_dimensions_for_arrangement(
                job_metadata={**job_metadata, "arrangement_type": arrangement_type, "view_mode": view_mode},
                view_features=[
                    {
                        "view_name": view_name,
                        "view_role": view_role,
                        "features": feature_values,
                    }
                ],
            )
            _apply_object_semantics(feature_values, semantic["semantic_dimensions_in"])
    internal_view_features = make_view_features(
        view_name=view_name,
        view_role=view_role,
        features=feature_values,
    )
    features_path = get_view_file_path(
        cfg,
        job_type,
        job_id,
        view_name,
        cfg.features_filename,
    )
    write_json_atomic(
        features_path,
        _model_feature_payload(
            cfg=cfg,
            ai2_features=feature_values["ai2_features"],
            quality_flags=feature_values["quality_flags"],
        ),
    )
    if debug_mode:
        _write_pot_debug(cfg, job_type, job_id, view_name, feature_values["debug_features"])
    mark_step(cfg, job_type, job_id, f"features_written_{view_name}")
    return internal_view_features


def infer_view_role(view_name: str, view_mode: str) -> str:
    """Infer a simple view role from the view name and view mode."""
    normalized_name = view_name.lower()
    if "length" in normalized_name:
        return "length_view"
    if "width" in normalized_name:
        return "width_view"
    if view_mode == "two_view_rectangle" and normalized_name == "view_01":
        return "length_view"
    if view_mode == "two_view_rectangle" and normalized_name == "view_02":
        return "width_view"
    if view_mode == "single_view":
        return "single_or_length"
    return "unknown"


def read_view_features_for_combine(
    cfg: DimScanConfig,
    job_type: str,
    job_id: str,
    *,
    view_name: str,
    view_role: str,
) -> dict[str, Any]:
    """Build combine-ready view features without rewriting the feature artifact."""
    features_path = get_view_file_path(
        cfg,
        job_type,
        job_id,
        view_name,
        cfg.features_filename,
    )
    payload = read_json_if_exists(features_path)
    if payload is None:
        raise FileNotFoundError(f"Missing features file: {features_path}")
    if not isinstance(payload, dict):
        raise ValueError(f"Features file must contain an object: {features_path}")

    geometry_path = get_view_file_path(
        cfg,
        job_type,
        job_id,
        view_name,
        cfg.geometry_filename,
    )
    geometry = read_json_if_exists(geometry_path)
    if geometry is None:
        raise FileNotFoundError(f"Missing geometry file: {geometry_path}")
    if not isinstance(geometry, dict):
        raise ValueError(f"Geometry file must contain an object: {geometry_path}")

    features = extract_view_feature_values(geometry)
    metadata_path = get_job_file_path(cfg, job_type, job_id, cfg.job_metadata_filename)
    job_metadata = read_json_if_exists(metadata_path, default={})
    if isinstance(job_metadata, dict):
        arrangement_type, view_mode, square = _arrangement_kind(job_metadata)
        if square and _has_explicit_rgb_spans(features):
            semantic = combine_object_dimensions_for_arrangement(
                job_metadata={**job_metadata, "arrangement_type": arrangement_type, "view_mode": view_mode},
                view_features=[
                    {
                        "view_name": view_name,
                        "view_role": view_role,
                        "features": features,
                    }
                ],
            )
            _apply_object_semantics(features, semantic["semantic_dimensions_in"])
    return make_view_features(
        view_name=view_name,
        view_role=view_role,
        features=features,
    )


def _first_view_feature_values(view_features: list[dict[str, Any]]) -> dict[str, Any]:
    if not view_features:
        return {}
    features = view_features[0].get("features")
    if isinstance(features, dict):
        return features
    return {}


def _first_item_pot_prior(item_list: dict[str, Any]) -> dict[str, Any]:
    items = item_list.get("items")
    if not isinstance(items, list) or not items:
        return parse_pot_prior(None)
    first = items[0]
    metadata = first.get("metadata") if isinstance(first, dict) else None
    if isinstance(metadata, dict) and isinstance(metadata.get("pot_prior"), dict):
        return metadata["pot_prior"]
    return parse_pot_prior(metadata, container_lookup=DimScanConfig.nursery_container_lookup)


def _clean_optional_sku_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _composition_token(value: Any, *, fallback: str = "unknown") -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return cleaned or fallback


def group_composition_features(item_list: dict[str, Any]) -> dict[str, Any] | None:
    """Return stable, order-independent model features for every item in a group."""
    items = item_list.get("items")
    if not isinstance(items, list) or not items:
        return None

    total_quantity = 0
    signatures: list[tuple[str, str, int]] = []
    unique_skus: set[str] = set()
    category_quantities: dict[str, int] = {}
    spec_quantities: dict[str, int] = {}
    pot_available_quantity = 0
    pot_available_item_count = 0
    weighted_pot: dict[str, float] = {"diameter_in": 0.0, "width_in": 0.0, "depth_in": 0.0, "height_in": 0.0}
    weighted_pot_quantities: dict[str, int] = {key: 0 for key in weighted_pot}

    for item in items:
        if not isinstance(item, dict):
            return None
        quantity = item.get("quantity")
        if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
            return None
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        sku = str(item.get("sku") or "").strip().upper()
        if sku:
            unique_skus.add(sku)
        category = _composition_token(metadata.get("category"))
        spec = _composition_token(metadata.get("spec"))
        signatures.append((category, spec, quantity))
        total_quantity += quantity
        category_quantities[category] = category_quantities.get(category, 0) + quantity
        spec_quantities[spec] = spec_quantities.get(spec, 0) + quantity

        pot_prior = metadata.get("pot_prior") if isinstance(metadata.get("pot_prior"), dict) else {}
        confidence = pot_prior_model_confidence(pot_prior)
        trusted_prior = pot_prior_available(pot_prior) and confidence in {"high", "medium"}
        if trusted_prior:
            pot_available_item_count += 1
            pot_available_quantity += quantity
            for key in weighted_pot:
                value = as_float_or_none(pot_prior.get(key))
                if value is not None and value > 0:
                    weighted_pot[key] += value * quantity
                    weighted_pot_quantities[key] += quantity

    if total_quantity <= 0:
        return None
    signature_counts: dict[tuple[str, str], int] = {}
    for category, spec, quantity in signatures:
        key = (category, spec)
        signature_counts[key] = signature_counts.get(key, 0) + quantity
    ordered_signatures = sorted(signature_counts.items())
    features: dict[str, Any] = {
        "group_composition_version": GROUP_COMPOSITION_VERSION,
        "group_composition_summary": "|".join(
            f"{category}:{spec}:{quantity}"
            for (category, spec), quantity in ordered_signatures
        ),
        "homogeneous_group": 1.0 if len(unique_skus) == 1 else 0.0,
        "group_composition_type_count": float(len(ordered_signatures)),
        "group_largest_type_quantity": float(max(signature_counts.values())),
        "group_largest_type_ratio": max(signature_counts.values()) / total_quantity,
        "group_pot_prior_item_count": float(pot_available_item_count),
        "group_pot_prior_quantity": float(pot_available_quantity),
        "group_pot_prior_quantity_ratio": pot_available_quantity / total_quantity,
    }
    for token, quantity in sorted(category_quantities.items()):
        features[f"group_category_qty__{token}"] = float(quantity)
        features[f"group_category_ratio__{token}"] = quantity / total_quantity
    for token, quantity in sorted(spec_quantities.items()):
        features[f"group_spec_qty__{token}"] = float(quantity)
        features[f"group_spec_ratio__{token}"] = quantity / total_quantity
    for key, weighted_total in weighted_pot.items():
        quantity = weighted_pot_quantities[key]
        features[f"group_pot_prior_{key}_weighted_mean"] = weighted_total / quantity if quantity else None
    return features


def _first_item_sku_context(item_list: dict[str, Any]) -> dict[str, str | None]:
    items = item_list.get("items")
    metadata: dict[str, Any] = {}
    if isinstance(items, list) and items and isinstance(items[0], dict):
        raw_metadata = items[0].get("metadata")
        if isinstance(raw_metadata, dict):
            metadata = raw_metadata
    return {
        "sku_category": _clean_optional_sku_text(metadata.get("category")),
        "sku_common_name": _clean_optional_sku_text(metadata.get("common_name")),
        "sku_spec": _clean_optional_sku_text(metadata.get("spec")),
    }


def _pot_prior_value(pot_prior: dict[str, Any], key: str) -> Any:
    if key == "height_in":
        return pot_prior.get("height_in")
    return pot_prior.get(key)


def _finite_positive_values(values: list[Any]) -> bool:
    for value in values:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
            return False
    return True


def _combined_geometry_trusted(
    *,
    ai2_features: dict[str, Any],
    view_features: list[dict[str, Any]],
    combined_values: dict[str, Any],
) -> bool:
    semantic = combined_values.get("object_dimension_semantics")
    if not isinstance(semantic, dict):
        return False
    dimensions = semantic.get("semantic_dimensions_in")
    mapping = semantic.get("semantic_mapping")
    per_view = semantic.get("per_view")
    if not isinstance(dimensions, dict) or not isinstance(mapping, dict) or not isinstance(per_view, list):
        return False
    if not combined_values.get("uses_explicit_rgb_spans"):
        return False
    if ai2_features.get("object_available") is not True:
        return False
    if not _finite_positive_values(
        [
            ai2_features.get("object_length_in"),
            ai2_features.get("object_width_in"),
            ai2_features.get("object_height_in"),
            dimensions.get("length_in"),
            dimensions.get("width_in"),
            dimensions.get("height_in"),
        ]
    ):
        return False
    if any(get_nested(view_feature, ["features", "geometry_status"]) != "ok" for view_feature in view_features):
        return False
    if any(get_nested(view_feature, ["features", "geometry_source_cloud"]) != "object_cloud" for view_feature in view_features):
        return False
    if any(
        not str(get_nested(view_feature, ["features", "point_cloud_path"]) or "").endswith("object_cloud.ply")
        for view_feature in view_features
    ):
        return False
    if any(get_nested(view_feature, ["features", "point_cloud_frame"]) != "rgb_camera" for view_feature in view_features):
        return False
    if any(get_nested(view_feature, ["features", "point_cloud_units"]) != "meters" for view_feature in view_features):
        return False
    if any(not _finite_positive_values([get_nested(view_feature, ["features", "object_point_count"])]) for view_feature in view_features):
        return False
    if semantic.get("z_used_as_physical_width") is not False:
        return False
    height_source = str(mapping.get("height_source") or "")
    if "vertical_span" not in height_source:
        return False
    width_source = str(mapping.get("width_source") or "")
    if "depth" in width_source:
        return False
    if semantic.get("square_width_equals_length") is True:
        length = as_float_or_none(dimensions.get("length_in"))
        width = as_float_or_none(dimensions.get("width_in"))
        if length is None or width is None or abs(length - width) >= 1e-9:
            return False
    return True


def _combined_ai2_contract(
    *,
    item_list: dict[str, Any],
    view_features: list[dict[str, Any]],
    combined_values: dict[str, Any],
    job_metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    first_values = _first_view_feature_values(view_features)
    first_ai2 = first_values.get("ai2_features") if isinstance(first_values.get("ai2_features"), dict) else {}
    first_flags = first_values.get("quality_flags") if isinstance(first_values.get("quality_flags"), dict) else {}
    first_debug = first_values.get("debug_features") if isinstance(first_values.get("debug_features"), dict) else {}
    is_group = job_metadata.get("job_type") == "group"
    pot_prior = parse_pot_prior(None) if is_group else _first_item_pot_prior(item_list)
    sku_context = {
        "sku_category": None,
        "sku_common_name": None,
        "sku_spec": None,
    } if is_group else _first_item_sku_context(item_list)
    prior_confidence = pot_prior_model_confidence(pot_prior)
    prior_allowed = pot_prior_available(pot_prior) and prior_confidence in {"high", "medium"}

    ai2_features = dict(first_ai2)
    semantic_dimensions = combined_values.get("object_dimension_semantics", {}).get("semantic_dimensions_in")
    if combined_values.get("uses_explicit_rgb_spans") and isinstance(semantic_dimensions, dict):
        for target, source in (
            ("object_length_in", "length_in"),
            ("object_width_in", "width_in"),
            ("object_height_in", "height_in"),
        ):
            value = semantic_dimensions.get(source)
            if value is not None:
                ai2_features[target] = value
    if prior_allowed and ai2_features.get("pot_diameter_in") is None:
        ai2_features["pot_diameter_in"] = as_float_or_none(_pot_prior_value(pot_prior, "diameter_in"))
        ai2_features["pot_height_in"] = as_float_or_none(_pot_prior_value(pot_prior, "height_in"))
        ai2_features["pot_width_in"] = as_float_or_none(_pot_prior_value(pot_prior, "width_in"))
        ai2_features["pot_depth_in"] = as_float_or_none(_pot_prior_value(pot_prior, "depth_in"))
        ai2_features["pot_prior_shape"] = pot_prior.get("shape")
        ai2_features["pot_prior_source"] = pot_prior.get("source")
    ai2_features.update(sku_context)
    if is_group:
        composition = group_composition_features(item_list)
        if composition is not None:
            ai2_features.update(composition)
    semantic = combined_values.get("object_dimension_semantics")
    arrangement = semantic.get("arrangement_type") if isinstance(semantic, dict) else None
    arrangement_rows, arrangement_columns = parse_arrangement_dims(arrangement or "1x1")
    ai2_features.update(
        {
            "arrangement": arrangement,
            "arrangement_rows": arrangement_rows,
            "arrangement_columns": arrangement_columns,
            "geometry_trusted": _combined_geometry_trusted(
                ai2_features=ai2_features,
                view_features=view_features,
                combined_values=combined_values,
            ),
            "authoritative_object_source": "geometry_primary",
            "total_quantity": item_list.get("total_quantity", 0),
            "unique_sku_count": item_list.get("unique_sku_count", 0),
            "view_count": combined_values.get("view_count"),
        }
    )

    quality_flags = dict(first_flags)
    if prior_allowed and first_ai2.get("pot_diameter_in") is None:
        quality_flags["pot_source"] = "sku_prior"
    quality_flags.update(
        {
            "pot_prior_available": pot_prior_available(pot_prior),
            "pot_prior_source": pot_prior.get("source"),
            "pot_prior_confidence": prior_confidence,
            "segmentation_statuses": combined_values.get("segmentation_statuses"),
            "geometry_statuses": combined_values.get("geometry_statuses"),
            "warnings": combined_values.get("geometry_warnings", []),
        }
    )
    debug_features = dict(first_debug)
    debug_features["pot_prior"] = pot_prior
    debug_features["views"] = [
        {
            "view_name": view_feature.get("view_name"),
            "quality_flags": view_feature.get("quality_flags")
            or get_nested(view_feature, ["features", "quality_flags"]),
            "debug_features": view_feature.get("debug_features")
            or get_nested(view_feature, ["features", "debug_features"]),
        }
        for view_feature in view_features
    ]
    return ai2_features, quality_flags, debug_features


def _combined_debug_payload(
    *,
    combined_values: dict[str, Any],
    debug_features: dict[str, Any],
) -> dict[str, Any]:
    return {
        "combined_candidates": {
            "combined_length_candidate_in": combined_values.get("combined_length_candidate_in"),
            "combined_width_candidate_in": combined_values.get("combined_width_candidate_in"),
            "combined_height_candidate_in": combined_values.get("combined_height_candidate_in"),
            "object_length_candidate_in": combined_values.get("object_length_candidate_in"),
            "object_width_candidate_in": combined_values.get("object_width_candidate_in"),
            "object_height_candidate_in": combined_values.get("object_height_candidate_in"),
            "pot_length_candidate_in": combined_values.get("pot_length_candidate_in"),
            "pot_width_candidate_in": combined_values.get("pot_width_candidate_in"),
            "pot_height_candidate_in": combined_values.get("pot_height_candidate_in"),
        },
        "pot_debug": debug_features,
        "view_roles": combined_values.get("view_roles", {}),
        "object_dimension_semantics": combined_values.get("object_dimension_semantics", {}),
        "views": debug_features.get("views", []),
    }


def _geometry_semantic_debug_payload(combined_values: dict[str, Any]) -> dict[str, Any]:
    semantic = combined_values.get("object_dimension_semantics")
    if not isinstance(semantic, dict):
        semantic = {}
    dimensions = semantic.get("semantic_dimensions_in") if isinstance(semantic.get("semantic_dimensions_in"), dict) else {}
    per_view = [view for view in semantic.get("per_view", []) if isinstance(view, dict)]
    cloud_paths = [view.get("point_cloud_path") for view in per_view]
    return {
        "cloud_path": cloud_paths[0] if len(cloud_paths) == 1 else cloud_paths,
        "point_cloud_path": cloud_paths,
        "point_cloud_frame": "rgb_camera",
        "point_cloud_units": "meters",
        "arrangement": semantic.get("arrangement_type"),
        "view_count": semantic.get("view_count"),
        "per_view": per_view,
        "raw_axis_extents": [
            {
                "view_name": view.get("view_name"),
                "x_min_m": view.get("x_min_m"),
                "x_max_m": view.get("x_max_m"),
                "x_span_m": view.get("x_span_m"),
                "x_span_in": view.get("x_span_in"),
                "y_min_m": view.get("y_min_m"),
                "y_max_m": view.get("y_max_m"),
                "y_span_m": view.get("y_span_m"),
                "y_span_in": view.get("y_span_in"),
                "z_min_m": view.get("z_min_m"),
                "z_max_m": view.get("z_max_m"),
                "z_span_m": view.get("z_span_m"),
                "z_span_in": view.get("z_span_in"),
                "horizontal_span_m": view.get("x_span_m"),
                "vertical_span_m": view.get("y_span_m"),
                "depth_thickness_m": view.get("z_span_m"),
                "horizontal_span_in": view.get("horizontal_span_in"),
                "vertical_span_in": view.get("vertical_span_in"),
                "depth_thickness_in": view.get("depth_thickness_in"),
            }
            for view in per_view
        ],
        "final_semantic_mapping": semantic.get("semantic_mapping", {}),
        "semantic_source_used_for_length": semantic.get("semantic_mapping", {}).get("length_source"),
        "semantic_source_used_for_width": semantic.get("semantic_mapping", {}).get("width_source"),
        "semantic_source_used_for_height": semantic.get("semantic_mapping", {}).get("height_source"),
        "square_width_equals_length": semantic.get("square_width_equals_length"),
        "final_length_in": dimensions.get("length_in"),
        "final_width_in": dimensions.get("width_in"),
        "final_height_in": dimensions.get("height_in"),
        "units_conversion_factor": semantic.get("units_conversion_factor", M_TO_IN),
        "z_used_as_physical_width": semantic.get("z_used_as_physical_width", False),
        "confirmation_z_not_used_as_physical_width": semantic.get("z_used_as_physical_width") is False,
        "final_ai2_dimension_writers": {
            "features_json": "features.extractor.write_view_features->_apply_object_semantics",
            "combined_features_json": "features.extractor.write_combined_features->_combined_ai2_contract",
            "object_length_in": "features.extractor._combined_ai2_contract",
            "object_width_in": "features.extractor._combined_ai2_contract",
            "object_height_in": "features.extractor._combined_ai2_contract",
            "combined_length_candidate_in": "features.extractor.combine_view_features",
            "combined_width_candidate_in": "features.extractor.combine_view_features",
            "combined_height_candidate_in": "features.extractor.combine_view_features",
        },
    }


def _features_for_role(
    view_features: list[dict[str, Any]],
    role: str,
) -> dict[str, Any] | None:
    for view_feature in view_features:
        if view_feature.get("view_role") == role and isinstance(view_feature.get("features"), dict):
            return view_feature["features"]
    return None


def combine_view_features(
    *,
    job_metadata: dict[str, Any],
    item_list: dict[str, Any],
    view_features: list[dict[str, Any]],
) -> dict[str, Any]:
    """Combine one or more view feature dictionaries into candidate job features."""
    view_mode = job_metadata.get("view_mode")
    view_roles = {
        str(view_feature.get("view_name")): view_feature.get("view_role")
        for view_feature in view_features
    }
    view_heights = [
        as_float_or_none(get_nested(view_feature, ["features", "view_height_in"]))
        for view_feature in view_features
    ]

    uses_explicit_rgb_spans = any(
        isinstance(view_feature.get("features"), dict)
        and _has_explicit_rgb_spans(view_feature["features"])
        for view_feature in view_features
    )
    semantic = combine_object_dimensions_for_arrangement(
        job_metadata=job_metadata,
        view_features=view_features,
    )
    if uses_explicit_rgb_spans:
        semantic_dimensions = semantic["semantic_dimensions_in"]
        combined_length = as_float_or_none(semantic_dimensions.get("length_in"))
        combined_width = as_float_or_none(semantic_dimensions.get("width_in"))
        combined_height = as_float_or_none(semantic_dimensions.get("height_in"))
    elif view_mode == "single_view":
        values = _first_view_feature_values(view_features)
        combined_length = as_float_or_none(values.get("length_in")) or as_float_or_none(
            values.get("view_width_in")
        )
        combined_width = as_float_or_none(values.get("width_in")) or as_float_or_none(
            values.get("view_depth_span_in")
        )
        combined_height = as_float_or_none(values.get("height_in")) or as_float_or_none(
            values.get("view_height_in")
        )
    else:
        length_values = _features_for_role(view_features, "length_view") or {}
        width_values = _features_for_role(view_features, "width_view") or {}
        combined_length = as_float_or_none(length_values.get("view_width_in"))
        combined_width = as_float_or_none(width_values.get("view_width_in"))
        combined_height = max_or_none(view_heights)

    return {
        "total_quantity": item_list.get("total_quantity", 0),
        "unique_sku_count": item_list.get("unique_sku_count", 0),
        "view_count": len(view_features),
        "geometry_statuses": {
            str(view_feature.get("view_name")): get_nested(view_feature, ["features", "geometry_status"])
            for view_feature in view_features
        },
        "segmentation_statuses": {
            str(view_feature.get("view_name")): get_nested(
                view_feature,
                ["features", "segmentation_status"],
            )
            for view_feature in view_features
        },
        "geometry_warnings": sorted(
            {
                str(warning)
                for view_feature in view_features
                for warning in (get_nested(view_feature, ["features", "geometry_warnings"]) or [])
            }
        ),
        "view_roles": view_roles,
        "uses_explicit_rgb_spans": uses_explicit_rgb_spans,
        "object_dimension_semantics": semantic,
        "combined_length_candidate_in": combined_length,
        "combined_width_candidate_in": combined_width,
        "combined_height_candidate_in": combined_height,
        "object_length_candidate_in": combined_length,
        "object_width_candidate_in": combined_width,
        "object_height_candidate_in": combined_height,
        "pot_length_candidate_in": as_float_or_none(
            _first_view_feature_values(view_features).get("pot_length_in")
        ),
        "pot_width_candidate_in": as_float_or_none(
            _first_view_feature_values(view_features).get("pot_width_in")
        ),
        "pot_height_candidate_in": as_float_or_none(
            _first_view_feature_values(view_features).get("pot_height_in")
        ),
        "leaf_canopy_length_candidate_in": as_float_or_none(
            _first_view_feature_values(view_features).get("leaf_canopy_length_in")
        ),
        "leaf_canopy_width_candidate_in": as_float_or_none(
            _first_view_feature_values(view_features).get("leaf_canopy_width_in")
        ),
        "leaf_canopy_height_candidate_in": as_float_or_none(
            _first_view_feature_values(view_features).get("leaf_canopy_height_in")
        ),
        "views": view_features,
    }


def write_combined_features(
    cfg: DimScanConfig,
    job_type: str,
    job_id: str,
    *,
    view_names: list[str],
    rewrite_view_names: list[str] | None = None,
    debug_mode: bool = True,
) -> dict[str, Any]:
    """Write per-view features and a combined feature file for a job."""
    metadata_path = get_job_file_path(cfg, job_type, job_id, cfg.job_metadata_filename)
    item_list_path = get_job_file_path(cfg, job_type, job_id, cfg.item_list_filename)

    job_metadata = read_json_if_exists(metadata_path)
    if job_metadata is None:
        raise FileNotFoundError(f"Missing job metadata file: {metadata_path}")
    if not isinstance(job_metadata, dict):
        raise ValueError(f"Job metadata file must contain an object: {metadata_path}")

    item_list = read_json_if_exists(item_list_path)
    if item_list is None:
        raise FileNotFoundError(f"Missing item list file: {item_list_path}")
    if not isinstance(item_list, dict):
        raise ValueError(f"Item list file must contain an object: {item_list_path}")

    view_mode = str(job_metadata.get("view_mode", ""))
    rewrite_set = set(view_names if rewrite_view_names is None else rewrite_view_names)
    view_features = [
        (
            write_view_features(
                cfg,
                job_type,
                job_id,
                view_name=view_name,
                view_role=infer_view_role(view_name, view_mode),
                debug_mode=debug_mode,
            )
            if view_name in rewrite_set
            else read_view_features_for_combine(
                cfg,
                job_type,
                job_id,
                view_name=view_name,
                view_role=infer_view_role(view_name, view_mode),
            )
        )
        for view_name in view_names
    ]
    combined_values = combine_view_features(
        job_metadata=job_metadata,
        item_list=item_list,
        view_features=view_features,
    )
    ai2_features, quality_flags, debug_features = _combined_ai2_contract(
        item_list=item_list,
        view_features=view_features,
        combined_values=combined_values,
        job_metadata=job_metadata,
    )
    combined_features = _model_feature_payload(
        cfg=cfg,
        ai2_features=ai2_features,
        quality_flags=quality_flags,
        feature_schema_version=(
            cfg.group_feature_schema_version
            if job_type == cfg.job_type_group
            else cfg.feature_schema_version
        ),
    )

    combined_path = get_job_file_path(cfg, job_type, job_id, cfg.combined_features_filename)
    write_json_atomic(combined_path, combined_features)
    if debug_mode:
        combined_debug_path = get_job_file_path(cfg, job_type, job_id, "debug") / COMBINED_DEBUG_FILENAME
        write_json_atomic(
            combined_debug_path,
            _combined_debug_payload(combined_values=combined_values, debug_features=debug_features),
        )
        semantic_debug_path = get_job_file_path(cfg, job_type, job_id, "debug") / GEOMETRY_SEMANTIC_DEBUG_FILENAME
        write_json_atomic(semantic_debug_path, _geometry_semantic_debug_payload(combined_values))
    mark_step(cfg, job_type, job_id, "combined_features_written")
    set_status(cfg, job_type, job_id, "features_completed")
    return combined_features
