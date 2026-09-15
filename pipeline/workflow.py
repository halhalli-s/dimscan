"""Workflow constants and request validation helpers for DimScan."""

from __future__ import annotations

from typing import Any


MODE_DATA_COLLECTION = "data_collection"
MODE_PREDICTION = "prediction"

JOB_TYPE_SINGLE = "single"
JOB_TYPE_GROUP = "group"


def validate_mode(mode: str) -> str:
    """Validate and return a supported workflow mode."""
    if mode not in {MODE_DATA_COLLECTION, MODE_PREDICTION}:
        raise ValueError(f"Unsupported mode: {mode!r}")
    return mode


def validate_job_type(job_type: str) -> str:
    """Validate and return a supported job type."""
    if job_type not in {JOB_TYPE_SINGLE, JOB_TYPE_GROUP}:
        raise ValueError(f"Unsupported job type: {job_type!r}")
    return job_type


def make_workflow_request(
    *,
    mode: str,
    job_type: str,
    arrangement_type: str,
    items: list[dict[str, Any]],
    operator_id: str | None = None,
) -> dict[str, Any]:
    """Validate and normalize a workflow request dictionary."""
    if not arrangement_type:
        raise ValueError("arrangement_type is required.")
    if not isinstance(items, list) or not items:
        raise ValueError("items must be a non-empty list.")

    return {
        "mode": validate_mode(mode),
        "job_type": validate_job_type(job_type),
        "arrangement_type": arrangement_type,
        "items": items,
        "operator_id": operator_id,
    }
