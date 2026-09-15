"""Small schema helpers for consistent DimScan feature dictionaries."""

from __future__ import annotations

from typing import Any


def _require_non_empty_string(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")


def _require_non_negative_int(value: int, *, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def make_view_features(
    *,
    view_name: str,
    view_role: str,
    features: dict[str, Any],
) -> dict[str, Any]:
    """Create a standard per-view feature dictionary."""
    _require_non_empty_string(view_name, name="view_name")
    _require_non_empty_string(view_role, name="view_role")
    if not isinstance(features, dict):
        raise ValueError("features must be a dictionary.")

    payload = {
        "view_name": view_name,
        "view_role": view_role,
        "features": features,
    }
    for key in ("ai2_features", "quality_flags", "debug_features"):
        if isinstance(features.get(key), dict):
            payload[key] = features[key]
    return payload


def make_combined_features(
    *,
    job_id: str,
    job_type: str,
    arrangement_type: str,
    view_mode: str,
    total_quantity: int,
    unique_sku_count: int,
    features: dict[str, Any],
) -> dict[str, Any]:
    """Create a standard combined feature dictionary for a job."""
    _require_non_empty_string(job_id, name="job_id")
    _require_non_empty_string(job_type, name="job_type")
    _require_non_empty_string(arrangement_type, name="arrangement_type")
    _require_non_empty_string(view_mode, name="view_mode")
    _require_non_negative_int(total_quantity, name="total_quantity")
    _require_non_negative_int(unique_sku_count, name="unique_sku_count")
    if not isinstance(features, dict):
        raise ValueError("features must be a dictionary.")

    payload = {
        "job_id": job_id,
        "job_type": job_type,
        "arrangement_type": arrangement_type,
        "view_mode": view_mode,
        "total_quantity": total_quantity,
        "unique_sku_count": unique_sku_count,
        "features": features,
    }
    for key in ("ai2_features", "quality_flags", "debug_features"):
        if isinstance(features.get(key), dict):
            payload[key] = features[key]
    return payload
