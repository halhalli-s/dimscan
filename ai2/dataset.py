"""Build the AI2 v1 training table from completed DimScan jobs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from utils.io import read_json_if_exists
from features.extractor import group_composition_features
from metadata.parser import parse_arrangement_dims


SAFE_SKU_FIELDS = (
    "sku_category",
    "sku_common_name",
    "sku_spec",
    "group_composition_version",
    "group_composition_summary",
)
LABEL_FIELDS = ("length_in", "width_in", "height_in")
DENSITY_ARRAY_FIELDS = (
    "object_vertical_density_bins",
    "leaf_vertical_density_bins",
)
DENSITY_BIN_COUNT = 10
APPROVED_BOOLEAN_FIELDS = (
    "leaf_available",
    "table_available",
    "object_profile_available",
    "leaf_profile_available",
)
ROW_ADMISSION_FIELDS = ("geometry_trusted", "object_available")
NON_MODEL_SOURCE_FIELDS = (
    *DENSITY_ARRAY_FIELDS,
    *APPROVED_BOOLEAN_FIELDS,
    *ROW_ADMISSION_FIELDS,
    "arrangement",
    "authoritative_object_source",
)
EXCLUDED_KEY_PARTS = (
    "ground_truth",
    "actual_box",
    "predicted",
    "prediction",
    "label",
    "job_id",
    "path",
    "timestamp",
    "recorded_at",
    "created_at",
    "updated_at",
    "source_file",
    "unit_price",
    "uom",
    "raw_sku",
    "spreadsheet",
)


def _skip(report: dict[str, Any], job_id: str, reason: str) -> None:
    report["skipped"] += 1
    report["skip_reasons"][reason] = report["skip_reasons"].get(reason, 0) + 1
    report["jobs"].append({"job_id": job_id, "status": "skipped", "reason": reason})


def _is_safe_feature(name: str, value: Any) -> bool:
    lowered = name.lower()
    if name in SAFE_SKU_FIELDS:
        return value is None or isinstance(value, str)
    if any(part in lowered for part in EXCLUDED_KEY_PARTS):
        return False
    return value is None or (isinstance(value, (int, float)) and not isinstance(value, bool))


def _feature_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _density_bin_values(value: Any) -> dict[str, float | None]:
    values = value if isinstance(value, list) else []
    return {
        f"bin_{index:02d}": _feature_value(values[index]) if index < len(values) else None
        for index in range(DENSITY_BIN_COUNT)
    }


def _boolean_feature_value(value: Any) -> float | None:
    if value is True:
        return 1.0
    if value is False:
        return 0.0
    return None


def model_features_from_ai2(ai2_features: dict[str, Any]) -> dict[str, Any]:
    """Return only model-facing AI2 feature values."""
    features: dict[str, Any] = {}
    for name, value in ai2_features.items():
        if name in NON_MODEL_SOURCE_FIELDS:
            continue
        if _is_safe_feature(name, value):
            features[name] = _feature_value(value)
    for source_name in DENSITY_ARRAY_FIELDS:
        prefix = source_name.removesuffix("_bins")
        for suffix, value in _density_bin_values(ai2_features.get(source_name)).items():
            features[f"{prefix}_{suffix}"] = value
    for name in APPROVED_BOOLEAN_FIELDS:
        features[name] = _boolean_feature_value(ai2_features.get(name))
    return {name: features[name] for name in sorted(features)}


def augment_group_ai2_features(
    ai2_features: dict[str, Any],
    item_list: dict[str, Any],
) -> dict[str, Any] | None:
    """Derive the v1.2 group fields for an older job when source metadata exists."""
    composition = group_composition_features(item_list)
    arrangement = ai2_features.get("arrangement")
    if composition is None or not isinstance(arrangement, str):
        return None
    try:
        rows, columns = parse_arrangement_dims(arrangement)
    except ValueError:
        return None
    return {
        **ai2_features,
        **composition,
        "arrangement_rows": rows,
        "arrangement_columns": columns,
    }


def _valid_label(actual_box: Any) -> dict[str, float] | None:
    if not isinstance(actual_box, dict):
        return None
    label: dict[str, float] = {}
    for field in LABEL_FIELDS:
        value = actual_box.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            return None
        label[field] = number
    return label


def build_training_table(
    jobs_root: str | Path,
    *,
    job_type: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Scan a jobs root and return AI2 v1 training rows plus an audit report."""
    root = Path(jobs_root)
    report: dict[str, Any] = {
        "jobs_root": str(root),
        "included": 0,
        "skipped": 0,
        "skip_reasons": {},
        "jobs": [],
        "feature_names": [],
        "label_fields": list(LABEL_FIELDS),
        "job_type": job_type,
    }
    rows: list[dict[str, Any]] = []
    feature_names: set[str] = set()

    if job_type not in {None, "single", "group"}:
        raise ValueError(f"Unsupported job type: {job_type!r}")
    if "prediction_data" in root.parts:
        report["skip_reasons"]["prediction_data_not_trainable"] = 1
        return rows, report
    if not root.exists():
        report["skip_reasons"]["jobs_root_missing"] = 1
        return rows, report

    for job_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        job_id = job_dir.name
        combined = read_json_if_exists(job_dir / "combined_features.json")
        ground_truth = read_json_if_exists(job_dir / "ground_truth.json")
        metadata = read_json_if_exists(job_dir / "job_metadata.json")
        if job_type is not None:
            if not isinstance(metadata, dict) or metadata.get("job_type") != job_type:
                _skip(report, job_id, "job_type_mismatch")
                continue
            if metadata.get("mode") == "prediction":
                _skip(report, job_id, "prediction_job_not_trainable")
                continue
        if not isinstance(combined, dict):
            _skip(report, job_id, "missing_combined_features")
            continue
        if not isinstance(ground_truth, dict):
            _skip(report, job_id, "missing_ground_truth")
            continue
        if ground_truth.get("completed") is not True:
            _skip(report, job_id, "ground_truth_incomplete")
            continue

        ai2_features = combined.get("ai2_features")
        if not isinstance(ai2_features, dict):
            _skip(report, job_id, "missing_ai2_features")
            continue
        if job_type == "group" and not ai2_features.get("group_composition_version"):
            item_list = read_json_if_exists(job_dir / "item_list.json")
            augmented = augment_group_ai2_features(ai2_features, item_list) if isinstance(item_list, dict) else None
            if augmented is None:
                _skip(report, job_id, "missing_group_composition")
                continue
            ai2_features = augmented
        if ai2_features.get("object_available") is not True:
            _skip(report, job_id, "object_unavailable")
            continue
        if ai2_features.get("geometry_trusted") is not True:
            _skip(report, job_id, "geometry_untrusted")
            continue

        label = _valid_label(ground_truth.get("actual_box"))
        if label is None:
            _skip(report, job_id, "invalid_actual_box")
            continue

        features = model_features_from_ai2(ai2_features)
        feature_names.update(features)
        rows.append({"job_id": job_id, "features": features, "label": label})
        report["included"] += 1
        report["jobs"].append({"job_id": job_id, "status": "included"})

    report["feature_names"] = sorted(feature_names)
    return rows, report


def matrix_from_rows(
    rows: list[dict[str, Any]],
    feature_names: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[list[float]], list[str]]:
    """Return sklearn-ready X/y records while preserving null feature values."""
    names = feature_names or sorted({name for row in rows for name in row["features"]})
    x_rows = [{name: row["features"].get(name) for name in names} for row in rows]
    y_rows = [[row["label"][field] for field in LABEL_FIELDS] for row in rows]
    return x_rows, y_rows, names
