"""Read-only AI2 v1 prediction helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from app.config import DimScanConfig
from ai2.dataset import LABEL_FIELDS, augment_group_ai2_features, model_features_from_ai2
from ai2.train import MODEL_VERSION
from utils.io import read_json_if_exists, utc_now_iso, write_json_atomic
from utils.paths import get_job_dir


MATCH_TOLERANCE_IN = 1.0
AI2_PREDICTION_FILENAME = "ai2_prediction.json"


def default_model_dir(cfg: DimScanConfig, job_type: str) -> Path:
    """Return the default AI2 v1 model directory for a job type."""
    if job_type == cfg.job_type_single:
        return cfg.single_packing_models_dir / "ai2_v1"
    if job_type == cfg.job_type_group:
        return cfg.group_packing_models_dir / "ai2_v1"
    raise ValueError(f"Unsupported job type: {job_type!r}")


def _box_from_values(values: Any, label_fields: list[str]) -> dict[str, float]:
    row = values[0] if hasattr(values, "__getitem__") else values
    return {
        field: float(row[index])
        for index, field in enumerate(label_fields)
    }


def _load_ground_truth(job_dir: Path) -> dict[str, Any]:
    gt = read_json_if_exists(job_dir / "ground_truth.json")
    if not isinstance(gt, dict) or gt.get("completed") is not True:
        return {"present": False}
    actual_box = gt.get("actual_box") if isinstance(gt.get("actual_box"), dict) else None
    return {
        "present": actual_box is not None,
        "actual_box": actual_box,
        "fit": gt.get("fit"),
        "damage": gt.get("damage"),
        "recorded_at": gt.get("recorded_at"),
    }


def _comparison(predicted_box: dict[str, float], ground_truth: dict[str, Any]) -> dict[str, Any] | None:
    actual_box = ground_truth.get("actual_box")
    if not isinstance(actual_box, dict):
        return None
    diff: dict[str, float] = {}
    for field in LABEL_FIELDS:
        actual = actual_box.get(field)
        predicted = predicted_box.get(field)
        if not isinstance(actual, (int, float)) or not isinstance(predicted, (int, float)):
            return None
        diff[field] = float(predicted) - float(actual)
    return {
        "tolerance_in": MATCH_TOLERANCE_IN,
        "diff_in": diff,
        "match": all(abs(value) <= MATCH_TOLERANCE_IN for value in diff.values()),
    }


def predict_ai2_for_job(
    cfg: DimScanConfig,
    job_type: str,
    job_id: str,
    *,
    model_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Predict a box for one job without writing job artifacts."""
    active_model_dir = Path(model_dir) if model_dir is not None else default_model_dir(cfg, job_type)
    model_path = active_model_dir / "model.joblib"
    report = read_json_if_exists(active_model_dir / "report.json", default={})
    warnings: list[str] = []
    if not model_path.is_file():
        warning = f"ai2_model_missing: {model_path}"
        return {
            "ok": True,
            "model_available": False,
            "prediction": None,
            "ground_truth": _load_ground_truth(get_job_dir(cfg, job_type, job_id)),
            "comparison": None,
            "warnings": [warning],
            "model": {
                "path": str(model_path),
                "version": MODEL_VERSION,
                "status": report.get("status") if isinstance(report, dict) else None,
            },
        }

    job_dir = get_job_dir(cfg, job_type, job_id)
    combined = read_json_if_exists(job_dir / "combined_features.json")
    if not isinstance(combined, dict):
        raise FileNotFoundError(f"Missing combined features: {job_dir / 'combined_features.json'}")
    ai2_features = combined.get("ai2_features")
    if not isinstance(ai2_features, dict):
        raise ValueError("combined_features.json missing ai2_features")
    if job_type == cfg.job_type_group and not ai2_features.get("group_composition_version"):
        item_list = read_json_if_exists(job_dir / cfg.item_list_filename)
        augmented = augment_group_ai2_features(ai2_features, item_list) if isinstance(item_list, dict) else None
        if augmented is None:
            raise ValueError("group job missing deterministic composition features")
        ai2_features = augmented
    if ai2_features.get("object_available") is not True:
        warnings.append("object_unavailable")
    if ai2_features.get("geometry_trusted") is not True:
        warnings.append("geometry_untrusted")

    try:
        import joblib
        import pandas as pd
    except ImportError as exc:
        return {
            "ok": True,
            "model_available": False,
            "prediction": None,
            "ground_truth": _load_ground_truth(job_dir),
            "comparison": None,
            "warnings": [f"sklearn_stack_unavailable: {exc}"],
            "model": {"path": str(model_path), "version": MODEL_VERSION},
        }

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_names = list(bundle["feature_names"])
    label_fields = list(bundle.get("label_fields") or LABEL_FIELDS)
    feature_values = model_features_from_ai2(ai2_features)
    feature_snapshot = {name: feature_values.get(name) for name in feature_names}
    values = model.predict(pd.DataFrame([feature_snapshot], columns=feature_names))
    predicted_box = _box_from_values(values, label_fields)
    for key, value in list(predicted_box.items()):
        if not math.isfinite(value):
            predicted_box[key] = 0.0

    ground_truth = _load_ground_truth(job_dir)
    comparison = _comparison(predicted_box, ground_truth)
    return {
        "ok": True,
        "model_available": True,
        "prediction": {
            "method": MODEL_VERSION,
            "predicted_box": predicted_box,
            "confidence": None,
            "quality": "unvalidated_small_data_baseline",
            "dimensions": {
                "object_length_in": ai2_features.get("object_length_in"),
                "object_width_in": ai2_features.get("object_width_in"),
                "object_height_in": ai2_features.get("object_height_in"),
            },
            "warnings": warnings,
            "feature_snapshot": feature_snapshot,
        },
        "ground_truth": ground_truth,
        "comparison": comparison,
        "warnings": warnings,
        "model": {
            "path": str(model_path),
            "version": bundle.get("model_version", MODEL_VERSION),
            "status": report.get("status") if isinstance(report, dict) else None,
        },
    }


def write_ai2_prediction(
    cfg: DimScanConfig,
    job_type: str,
    job_id: str,
    result: dict[str, Any],
) -> Path | None:
    """Persist an exact successful AI2 box prediction for later integrations."""
    if result.get("model_available") is not True:
        return None
    prediction = result.get("prediction")
    box = prediction.get("predicted_box") if isinstance(prediction, dict) else None
    if not isinstance(box, dict):
        return None
    model = result.get("model") if isinstance(result.get("model"), dict) else {}
    payload = {
        "job_id": job_id,
        "job_type": job_type,
        "length_in": box.get("length_in"),
        "width_in": box.get("width_in"),
        "height_in": box.get("height_in"),
        "units": "inches",
        "model": {
            "version": model.get("version"),
            "path": model.get("path"),
        },
        "created_at": utc_now_iso(),
    }
    path = get_job_dir(cfg, job_type, job_id) / AI2_PREDICTION_FILENAME
    write_json_atomic(path, payload)
    return path
