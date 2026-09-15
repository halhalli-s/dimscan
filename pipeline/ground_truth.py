"""Ground truth writers for actual box labels and prediction feedback."""

from __future__ import annotations

from typing import Any

from app.config import DimScanConfig
from pipeline.session import mark_step, set_status
from utils.io import read_json_if_exists, utc_now_iso, write_json_atomic
from utils.paths import get_job_file_path


FEEDBACK_TYPES = {"prediction_confirmed", "prediction_mismatch", "skipped"}


def make_box(length_in: float, width_in: float, height_in: float) -> dict[str, float]:
    """Create a validated box dimensions record."""
    length = float(length_in)
    width = float(width_in)
    height = float(height_in)
    if length <= 0 or width <= 0 or height <= 0:
        raise ValueError("Box dimensions must be positive.")

    return {
        "length_in": length,
        "width_in": width,
        "height_in": height,
    }


def record_ground_truth(
    cfg: DimScanConfig,
    job_type: str,
    job_id: str,
    *,
    length_in: float,
    width_in: float,
    height_in: float,
    source: str = "manual",
    fit: str | None = None,
    damage: bool | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Record completed ground truth box dimensions for a job."""
    ground_truth = {
        "completed": True,
        "source": source,
        "actual_box": make_box(length_in, width_in, height_in),
        "fit": fit,
        "damage": damage,
        "notes": notes,
        "recorded_at": utc_now_iso(),
    }

    path = get_job_file_path(cfg, job_type, job_id, cfg.ground_truth_filename)
    write_json_atomic(path, ground_truth)
    mark_step(cfg, job_type, job_id, "ground_truth_completed")
    set_status(cfg, job_type, job_id, "ground_truth_recorded")
    return ground_truth


def record_prediction_feedback(
    cfg: DimScanConfig,
    job_type: str,
    job_id: str,
    *,
    feedback_type: str,
    predicted_box: dict[str, Any] | None = None,
    actual_box: dict[str, Any] | None = None,
    fit: str | None = None,
    damage: bool | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Record prediction feedback as ground truth for a job."""
    if feedback_type not in FEEDBACK_TYPES:
        raise ValueError(f"Unsupported feedback type: {feedback_type!r}")

    ground_truth = {
        "completed": feedback_type != "skipped",
        "feedback_type": feedback_type,
        "predicted_box": predicted_box,
        "actual_box": actual_box,
        "fit": fit,
        "damage": damage,
        "notes": notes,
        "recorded_at": utc_now_iso(),
    }

    path = get_job_file_path(cfg, job_type, job_id, cfg.ground_truth_filename)
    write_json_atomic(path, ground_truth)
    mark_step(cfg, job_type, job_id, feedback_type)
    set_status(cfg, job_type, job_id, feedback_type)
    return ground_truth


def load_ground_truth(cfg: DimScanConfig, job_type: str, job_id: str) -> dict[str, Any] | None:
    """Load ground truth for a job, returning None when it is missing."""
    path = get_job_file_path(cfg, job_type, job_id, cfg.ground_truth_filename)
    return read_json_if_exists(path)
