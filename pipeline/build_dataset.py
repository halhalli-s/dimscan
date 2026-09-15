"""Build simple JSON-backed ML dataset rows from completed DimScan jobs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from app.config import DimScanConfig
from utils.io import read_json_if_exists
from utils.paths import get_exports_root, get_job_file_path, list_job_dirs


def flatten_dict(data: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dictionaries using underscore-separated keys."""
    flattened: dict[str, Any] = {}

    for key, value in data.items():
        flat_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, prefix=flat_key))
        else:
            flattened[flat_key] = value

    return flattened


def build_job_row(cfg: DimScanConfig, job_type: str, job_id: str) -> dict[str, Any] | None:
    """Build one flat dataset row from a completed job folder."""
    job_metadata = read_json_if_exists(
        get_job_file_path(cfg, job_type, job_id, cfg.job_metadata_filename)
    )
    item_list = read_json_if_exists(get_job_file_path(cfg, job_type, job_id, cfg.item_list_filename))
    combined_features = read_json_if_exists(
        get_job_file_path(cfg, job_type, job_id, cfg.combined_features_filename)
    )
    ground_truth = read_json_if_exists(
        get_job_file_path(cfg, job_type, job_id, cfg.ground_truth_filename)
    )

    if not job_metadata or not item_list or not combined_features or not ground_truth:
        return None
    if ground_truth.get("completed") is not True:
        return None

    row: dict[str, Any] = {
        "job_id": job_id,
    }
    row.update(flatten_dict(job_metadata, prefix="metadata"))
    row.update(flatten_dict(item_list, prefix="items"))
    row.update(flatten_dict(combined_features, prefix="features"))
    row.update(flatten_dict(ground_truth, prefix="ground_truth"))
    return row


def build_rows(cfg: DimScanConfig, job_type: str) -> list[dict[str, Any]]:
    """Build dataset rows for all completed jobs of a job type."""
    rows: list[dict[str, Any]] = []
    for job_dir in list_job_dirs(cfg, job_type):
        row = build_job_row(cfg, job_type, job_dir.name)
        if row is not None:
            rows.append(row)
    return rows


def _csv_value(value: Any) -> Any:
    if isinstance(value, dict | list):
        return json.dumps(value, sort_keys=False)
    return value


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Write dataset rows to CSV."""
    export_path = Path(path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        export_path.write_text("", encoding="utf-8")
        return export_path

    fieldnames = sorted({key for row in rows for key in row})
    with export_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})

    return export_path


def export_dataset(
    cfg: DimScanConfig,
    job_type: str,
    *,
    export_name: str | None = None,
) -> Path:
    """Export completed job rows for a job type to CSV."""
    rows = build_rows(cfg, job_type)
    filename = export_name or f"{job_type}_training_export.csv"
    export_path = get_exports_root(cfg, job_type) / filename
    return write_csv(export_path, rows)
