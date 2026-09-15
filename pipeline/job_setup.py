"""Helpers for operator job setup before capture starts."""

from __future__ import annotations

from typing import Any

from app.config import DimScanConfig
from pipeline.arrangement import default_view_names, infer_view_mode
from utils.paths import generate_collection_job_id, get_job_dir, sanitize_job_id


def _unique_generated_job_id(
    cfg: DimScanConfig,
    job_type: str,
    items: list[dict[str, Any]] | None,
) -> str:
    candidate = generate_collection_job_id(job_type, items)
    if get_job_dir(cfg, job_type, candidate).exists():
        raise ValueError(f"job_id already exists: {candidate}")
    return candidate


def prepare_job_setup(
    cfg: DimScanConfig,
    *,
    job_type: str,
    job_id: Any = None,
    arrangement_type: str = "1x1",
    shape_mode: str | None = None,
    items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve a collision-safe job ID and view plan for the operator UI."""
    manual_job_id = str(job_id or "").strip()
    if manual_job_id:
        resolved_job_id = sanitize_job_id(manual_job_id)
        if not resolved_job_id:
            raise ValueError("job_id must contain letters, numbers, dashes, or underscores")
        if resolved_job_id != manual_job_id.lower().replace(" ", "_"):
            raise ValueError("job_id may only use letters, numbers, spaces, dashes, or underscores")
        if get_job_dir(cfg, job_type, resolved_job_id).exists():
            raise ValueError(f"job_id already exists: {resolved_job_id}")
    else:
        resolved_job_id = _unique_generated_job_id(cfg, job_type, items)

    normalized_shape = str(shape_mode or "").strip().lower()
    if job_type == cfg.job_type_single and normalized_shape == "rectangular":
        view_mode = cfg.view_mode_two_view_rectangle
        required_views = ["view_01", "view_02"]
    elif job_type == cfg.job_type_single:
        view_mode = cfg.view_mode_single
        required_views = ["view_01"]
    else:
        view_mode = infer_view_mode(arrangement_type)
        required_views = default_view_names(arrangement_type, view_mode)
    next_action = "Capture View 01 / Front Side." if len(required_views) > 1 else "Capture View 01."
    return {
        "job_id": resolved_job_id,
        "auto_generated_job_id": not bool(manual_job_id),
        "job_type": job_type,
        "arrangement_type": arrangement_type,
        "view_mode": view_mode,
        "required_views": required_views,
        "remaining_views": required_views,
        "captured_views": [],
        "next_action": next_action,
    }
