"""Operator-facing capture quality summaries built from processed job JSON."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import DimScanConfig
from utils.io import read_json, read_json_if_exists, write_json_atomic
from utils.paths import get_job_dir, get_job_file_path, get_view_file_path


PASS = "pass"
WARN = "warn"
FAIL = "fail"


def _empty_check(message: str, status: str = FAIL) -> dict[str, str]:
    return {"status": status, "message": message}


def _nested(data: dict[str, Any] | None, keys: list[str]) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _load_json(path: Path, label: str, failures: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        failures.append(f"{label} missing")
        return None
    try:
        data = read_json(path)
    except Exception as exc:
        failures.append(f"{label} malformed: {exc}")
        return None
    if not isinstance(data, dict):
        failures.append(f"{label} malformed: expected JSON object")
        return None
    return data


def _positive_dimensions(dimensions: Any, keys: tuple[str, ...]) -> bool:
    if not isinstance(dimensions, dict):
        return False
    for key in keys:
        value = dimensions.get(key)
        if not isinstance(value, (int, float)) or value <= 0:
            return False
    return True


def _dimension_message(dimensions: Any, label: str) -> str:
    if not isinstance(dimensions, dict):
        return f"{label} dimensions missing."
    return (
        f"{label} dimensions available: "
        f"L {dimensions.get('length_in')}, W {dimensions.get('width_in')}, H {dimensions.get('height_in')}."
    )


def _cloud_bbox_message(record: Any, label: str) -> str:
    if not isinstance(record, dict):
        return f"{label} cloud diagnostics missing."
    quality = str(record.get("quality") or "unknown").upper()
    source = record.get("source") or "unknown"
    point_count = record.get("point_count")
    raw_bbox = record.get("raw_bbox_in") if isinstance(record.get("raw_bbox_in"), dict) else {}
    robust_bbox = record.get("robust_bbox_in") if isinstance(record.get("robust_bbox_in"), dict) else {}
    return (
        f"{label} cloud {quality}; source {source}; points {point_count}; "
        f"raw L/W/H {raw_bbox.get('length_in')}/{raw_bbox.get('width_in')}/{raw_bbox.get('height_in')}; "
        f"robust L/W/H {robust_bbox.get('length_in')}/{robust_bbox.get('width_in')}/{robust_bbox.get('height_in')}."
    )


def _cloud_quality_status(record: Any) -> str:
    if not isinstance(record, dict):
        return FAIL
    quality = str(record.get("quality") or "").lower()
    if quality == "ok":
        return PASS
    if quality == "review":
        return WARN
    return FAIL


def _segment_dimension_check_message(
    *,
    label: str,
    segment_status: Any,
    available: bool,
    dimensions: Any,
) -> str:
    if segment_status == "missing":
        return f"{label} segment missing."
    if not available:
        return f"{label} unavailable in ai2_features."
    return _dimension_message(dimensions, label)


def _contains_blocked_debug_fields(data: Any) -> bool:
    blocked = {
        "debug_features",
        "rejected_pot_dimensions_in",
        "raw_pot_candidate_measurements",
        "pot_debug",
    }
    if isinstance(data, dict):
        for key, value in data.items():
            if key in blocked:
                return True
            if _contains_blocked_debug_fields(value):
                return True
    elif isinstance(data, list):
        return any(_contains_blocked_debug_fields(value) for value in data)
    return False


def _pot_model_facing_leak(
    *,
    segmentation: dict[str, Any] | None,
    features: dict[str, Any] | None,
    combined_features: dict[str, Any] | None,
) -> str | None:
    if _contains_blocked_debug_fields(features) or _contains_blocked_debug_fields(combined_features):
        return "Rejected/debug pot fields appear in model-facing features."

    pot_quality = segmentation.get("pot_quality") if isinstance(segmentation, dict) else {}
    pot_usable = bool(pot_quality.get("usable_for_model")) if isinstance(pot_quality, dict) else False
    view_ai2 = features.get("ai2_features") if isinstance(features, dict) else {}
    combined_ai2 = combined_features.get("ai2_features") if isinstance(combined_features, dict) else {}
    combined_flags = combined_features.get("quality_flags") if isinstance(combined_features, dict) else {}

    view_has_pot = any(
        _nested(view_ai2, [key]) is not None
        for key in ("pot_diameter_in", "pot_height_in")
    )
    if not pot_usable and view_has_pot:
        return "Rejected/fallback pot dimensions appear in view ai2_features."

    combined_has_pot = any(
        _nested(combined_ai2, [key]) is not None
        for key in ("pot_diameter_in", "pot_height_in")
    )
    if not pot_usable and combined_has_pot:
        source = combined_flags.get("pot_source") if isinstance(combined_flags, dict) else None
        confidence = combined_flags.get("pot_prior_confidence") if isinstance(combined_flags, dict) else None
        if source != "sku_prior" or confidence not in {"high", "medium"}:
            return "Combined pot dimensions are not backed by a high/medium SKU prior."
    return None


def _ground_truth_check(job_dir: Path, cfg: DimScanConfig) -> dict[str, str]:
    path = job_dir / cfg.ground_truth_filename
    if not path.is_file():
        return _empty_check("GT pending.", WARN)
    try:
        data = read_json(path)
    except Exception as exc:
        return _empty_check(f"Ground truth malformed: {exc}", WARN)
    if not isinstance(data, dict):
        return _empty_check("Ground truth malformed.", WARN)
    return _empty_check("Ground truth present.", PASS)


def build_quality_summary(
    cfg: DimScanConfig,
    *,
    job_type: str,
    job_id: str,
    view_name: str = "view_01",
    write: bool = True,
) -> dict[str, Any]:
    """Build and optionally persist the operator capture quality summary."""
    job_dir = get_job_dir(cfg, job_type, job_id)
    failures: list[str] = []
    warnings: list[str] = []

    segmentation = _load_json(
        get_view_file_path(cfg, job_type, job_id, view_name, "segmentation.json"),
        "segmentation.json",
        failures,
    )
    geometry = _load_json(
        get_view_file_path(cfg, job_type, job_id, view_name, cfg.geometry_filename),
        "geometry.json",
        failures,
    )
    features = _load_json(
        get_view_file_path(cfg, job_type, job_id, view_name, cfg.features_filename),
        "features.json",
        failures,
    )
    combined_features = _load_json(
        get_job_file_path(cfg, job_type, job_id, cfg.combined_features_filename),
        "combined_features.json",
        failures,
    )
    session = read_json_if_exists(get_job_file_path(cfg, job_type, job_id, cfg.session_filename), default={})
    if not isinstance(session, dict):
        session = {}

    segments = segmentation.get("segments") if isinstance(segmentation, dict) else {}
    pot_quality = segmentation.get("pot_quality") if isinstance(segmentation, dict) else {}
    view_ai2 = features.get("ai2_features") if isinstance(features, dict) else {}
    if not isinstance(view_ai2, dict):
        view_ai2 = {}
    view_flags = features.get("quality_flags") if isinstance(features, dict) else {}
    if not isinstance(view_flags, dict):
        view_flags = {}
    combined_ai2 = combined_features.get("ai2_features") if isinstance(combined_features, dict) else {}
    if not isinstance(combined_ai2, dict):
        combined_ai2 = {}
    combined_flags = combined_features.get("quality_flags") if isinstance(combined_features, dict) else {}
    if not isinstance(combined_flags, dict):
        combined_flags = {}

    object_dimensions = geometry.get("object_dimensions_in") if isinstance(geometry, dict) else None
    leaf_dimensions = geometry.get("leaf_canopy_dimensions_in") if isinstance(geometry, dict) else None
    dimensions = geometry.get("dimensions_in") if isinstance(geometry, dict) else None
    geometry_status = geometry.get("status") if isinstance(geometry, dict) else None
    geometry_warnings = geometry.get("warnings") if isinstance(geometry, dict) else []
    cloud_quality = geometry.get("cloud_quality") if isinstance(geometry, dict) else {}
    if not isinstance(cloud_quality, dict):
        cloud_quality = {}
    object_cloud_quality = cloud_quality.get("object")
    leaf_cloud_quality = cloud_quality.get("leaf")
    if isinstance(geometry_warnings, list):
        warnings.extend(str(warning) for warning in geometry_warnings[:5])

    geometry_ok = (
        geometry_status not in {"failed", "missing"}
        and _positive_dimensions(object_dimensions, ("length_in", "width_in", "height_in"))
        and _positive_dimensions(dimensions, ("length_in", "width_in", "height_in"))
    )
    object_ok = (
        combined_ai2.get("geometry_trusted") is True
        or (
            view_ai2.get("object_available") is True
            and geometry_ok
            and _cloud_quality_status(object_cloud_quality) == PASS
        )
    )
    leaf_ok = (
        segments.get("leaf") != "missing"
        and view_ai2.get("leaf_available") is True
        and _positive_dimensions(leaf_dimensions, ("length_in", "width_in", "height_in"))
    )
    object_message = (
        "Authoritative geometry-primary object is available; AI1 object segmentation missing is non-blocking."
        if object_ok and segments.get("object") == "missing"
        else _segment_dimension_check_message(
            label="Object",
            segment_status=segments.get("object"),
            available=view_ai2.get("object_available") is True,
            dimensions=object_dimensions,
        )
    )

    checks: dict[str, dict[str, str]] = {
        "object": _empty_check(
            object_message,
            PASS if object_ok else FAIL,
        ),
        "leaf": _empty_check(
            _segment_dimension_check_message(
                label="Leaf/canopy",
                segment_status=segments.get("leaf"),
                available=view_ai2.get("leaf_available") is True,
                dimensions=leaf_dimensions,
            ),
            PASS if leaf_ok else FAIL,
        ),
        "geometry": _empty_check(
            "Object geometry available." if geometry_ok else "Geometry failed or required dimensions are missing.",
            PASS if geometry_ok else FAIL,
        ),
        "object_cloud": _empty_check(
            _cloud_bbox_message(object_cloud_quality, "Object"),
            _cloud_quality_status(object_cloud_quality),
        ),
        "leaf_cloud": _empty_check(
            _cloud_bbox_message(leaf_cloud_quality, "Leaf"),
            _cloud_quality_status(leaf_cloud_quality),
        ),
        "features": _empty_check("Model-facing feature files are present.", PASS),
        "gt": _ground_truth_check(job_dir, cfg),
    }

    if failures:
        checks["features"] = _empty_check("; ".join(failures), FAIL)

    table_status = segments.get("table") if isinstance(segments, dict) else None
    table_available = view_ai2.get("table_available") is True
    if table_status == "ok" or table_available:
        checks["table"] = _empty_check("Table support available.", PASS)
    elif geometry_ok and geometry_status in {"ok", "degraded"}:
        checks["table"] = _empty_check("Table mask missing, but geometry still produced usable dimensions.", WARN)
    else:
        checks["table"] = _empty_check("Table missing and geometry is degraded or suspicious.", FAIL)

    leak_reason = _pot_model_facing_leak(
        segmentation=segmentation,
        features=features,
        combined_features=combined_features,
    )
    pot_status = pot_quality.get("status") if isinstance(pot_quality, dict) else "missing"
    combined_pot_source = combined_flags.get("pot_source") if isinstance(combined_flags, dict) else None
    prior_confidence = combined_flags.get("pot_prior_confidence") if isinstance(combined_flags, dict) else None
    if leak_reason:
        checks["pot"] = _empty_check(leak_reason, FAIL)
    elif pot_status == "trusted" and view_flags.get("pot_source") == "segmentation_trusted":
        checks["pot"] = _empty_check("Trusted pot segmentation is available.", PASS)
    elif combined_pot_source == "sku_prior" and prior_confidence in {"high", "medium"}:
        checks["pot"] = _empty_check("Pot segmentation excluded; high/medium SKU prior is available.", PASS)
    elif pot_status in {"fallback", "missing"} or str(pot_status).startswith("rejected"):
        checks["pot"] = _empty_check("Pot segmentation safely excluded from model features.", WARN)
    else:
        checks["pot"] = _empty_check("Pot status needs review.", WARN)

    if checks["features"]["status"] == PASS and leak_reason is None:
        checks["features"] = _empty_check("Model-facing features are clean.", PASS)

    fail_keys = [key for key, check in checks.items() if check["status"] == FAIL]
    warn_keys = [key for key, check in checks.items() if check["status"] == WARN]

    recapture_failures = [
        key
        for key in fail_keys
        if key in {"object", "object_cloud", "geometry", "features"}
    ]
    if recapture_failures:
        decision = "recapture"
        status_label = "RECAPTURE RECOMMENDED"
    elif any(key in warn_keys for key in {"object_cloud", "geometry"}) or "table" in fail_keys:
        decision = "review"
        status_label = "REVIEW"
    elif warn_keys:
        decision = "review" if "table" in fail_keys else "proceed"
        status_label = "REVIEW" if decision == "review" else "GOOD TO PROCEED"
    else:
        decision = "proceed"
        status_label = "GOOD TO PROCEED"

    reasons = [
        check["message"]
        for key, check in checks.items()
        if check["status"] == FAIL and key != "gt"
    ]
    if not reasons and decision == "proceed":
        reasons = ["Object, leaf/canopy, geometry, and model-facing features are usable."]
    if not reasons and decision == "review":
        reasons = ["Capture is usable, but one or more checks needs operator review."]

    summary = {
        "decision": decision,
        "status_label": status_label,
        "reasons": reasons,
        "warnings": sorted({str(warning) for warning in warnings if warning}),
        "checks": checks,
    }

    payload = {"quality_summary": summary}
    if write:
        write_json_atomic(get_view_file_path(cfg, job_type, job_id, view_name, cfg.quality_summary_filename), payload)
        write_json_atomic(get_job_file_path(cfg, job_type, job_id, cfg.quality_summary_filename), payload)
    return summary
