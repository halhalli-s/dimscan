"""Rule-based placeholder prediction flow for DimScan jobs."""

from __future__ import annotations

from typing import Any

from app.config import DimScanConfig
from packing.rules import suggest_box_from_combined_features
from utils.io import read_json, write_json_atomic
from utils.paths import get_job_file_path


def predict_box_for_job(
    cfg: DimScanConfig,
    job_type: str,
    job_id: str,
    *,
    padding_in: float = 1.0,
    round_increment_in: float = 1.0,
) -> dict[str, Any]:
    """Predict a box for a job using the current rule-based placeholder."""
    combined_features_path = get_job_file_path(
        cfg,
        job_type,
        job_id,
        cfg.combined_features_filename,
    )
    combined_features = read_json(combined_features_path)
    box = suggest_box_from_combined_features(
        combined_features,
        padding_in=padding_in,
        round_increment_in=round_increment_in,
    )
    prediction = {
        "job_id": job_id,
        "job_type": job_type,
        "method": "rules_placeholder",
        "box": box,
        "status": "predicted",
    }

    prediction_filename = getattr(cfg, "prediction_filename", "prediction.json")
    prediction_path = get_job_file_path(cfg, job_type, job_id, prediction_filename)
    write_json_atomic(prediction_path, prediction)
    return prediction
