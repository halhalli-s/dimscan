"""Reprocess existing DimScan job folders through geometry and feature extraction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from app.config import DimScanConfig
from features.extractor import infer_view_role, read_view_features_for_combine, write_combined_features, write_view_features
from geometry.measure import measure_view_geometry
from pipeline.object_extraction import extract_geometry_primary_object_cloud
from pipeline.profiling import timed_stage
from pipeline.quality import build_quality_summary
from pipeline.session import mark_step, set_status
from pipeline.workflow import validate_job_type
from segmentation.schema import make_segmentation_record
from segmentation.segmenter import segment_view
from utils.io import read_json_if_exists, write_json_atomic
from utils.paths import get_job_dir, get_view_file_path


def _public_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _public_json_value(child)
            for key, child in value.items()
            if not str(key).startswith("_runtime_")
        }
    if isinstance(value, list):
        return [_public_json_value(child) for child in value]
    if hasattr(value, "item") and value.__class__.__module__.startswith("numpy"):
        shape = getattr(value, "shape", ())
        if shape not in ((), None):
            return value
        return value.item()
    return value


def _positive_dimensions(values: Any) -> bool:
    if not isinstance(values, dict):
        return False
    for key in ("length_in", "width_in", "height_in"):
        value = values.get(key)
        if not isinstance(value, (int, float)) or value <= 0:
            return False
    return True


def _positive_ai2_object_dimensions(values: Any) -> bool:
    if not isinstance(values, dict):
        return False
    for key in ("object_length_in", "object_width_in", "object_height_in"):
        value = values.get(key)
        if not isinstance(value, (int, float)) or value <= 0:
            return False
    return True


def _read_existing_view_result(
    cfg: DimScanConfig,
    job_type: str,
    job_id: str,
    *,
    view_name: str,
) -> dict[str, Any]:
    view_dir = get_job_dir(cfg, job_type, job_id) / view_name
    segmentation = read_json_if_exists(view_dir / "segmentation.json")
    geometry = read_json_if_exists(view_dir / cfg.geometry_filename)
    if not isinstance(segmentation, dict):
        raise FileNotFoundError(f"Missing or invalid segmentation.json for {view_name}")
    if not isinstance(geometry, dict):
        raise FileNotFoundError(f"Missing or invalid {cfg.geometry_filename} for {view_name}")
    return {
        "segmentation": segmentation,
        "geometry": geometry,
    }


def _capture_artifacts_available(cfg: DimScanConfig, view_dir: Path) -> bool:
    required = (
        getattr(cfg, "capture_meta_filename", "capture_meta.json"),
        getattr(cfg, "rgb_filename", "rgb.png"),
        getattr(cfg, "depth_raw_filename", "depth_raw.npy"),
    )
    return all((view_dir / filename).is_file() for filename in required)


def _ensure_in_memory_object_cloud_contract(view_dir: Path, *, debug_mode: bool = True) -> None:
    if not debug_mode:
        return
    debug_dir = view_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / "object_cloud_debug.json"
    debug = read_json_if_exists(debug_path, default={})
    if not isinstance(debug, dict):
        debug = {}
    debug.setdefault("generation_method", "roi_table_cluster")
    debug.setdefault("extraction_method", "roi_table_cluster")
    debug.setdefault("authoritative_object_source", "geometry_cluster")
    debug.setdefault("object_cloud_coordinate_frame", "rgb_camera")
    debug.setdefault("point_cloud_frame", "rgb_camera")
    debug.setdefault("point_cloud_units", "meters")
    debug.setdefault("roi_units", "meters")
    if (view_dir / "object_cloud.ply").is_file():
        debug.setdefault("final_object_cloud_path", str(view_dir / "object_cloud.ply"))
    debug.setdefault("geometry_primary_extraction", True)
    debug.setdefault("ai1_role", "validation_only")
    debug.setdefault("ai1_used_for_object_extraction", False)
    write_json_atomic(debug_path, debug)


def _view_outputs_valid(
    cfg: DimScanConfig,
    job_type: str,
    job_id: str,
    *,
    view_name: str,
) -> bool:
    view_dir = get_job_dir(cfg, job_type, job_id) / view_name
    if not _capture_artifacts_available(cfg, view_dir):
        return False

    segmentation = read_json_if_exists(view_dir / "segmentation.json")
    if not isinstance(segmentation, dict) or segmentation.get("status") not in {"ok", "partial", "failed"}:
        return False

    geometry = read_json_if_exists(view_dir / cfg.geometry_filename)
    if not isinstance(geometry, dict) or geometry.get("status") in {"failed", "missing"}:
        return False
    if geometry.get("source_cloud") != "object_cloud":
        return False
    object_dimensions = geometry.get("object_dimensions_in") or geometry.get("dimensions_in")
    if not _positive_dimensions(object_dimensions):
        return False

    feature_payload = read_json_if_exists(view_dir / cfg.features_filename)
    if not isinstance(feature_payload, dict):
        return False
    ai2_features = feature_payload.get("ai2_features")
    if not isinstance(ai2_features, dict) or ai2_features.get("object_available") is not True:
        return False
    if not _positive_ai2_object_dimensions(ai2_features):
        return False

    return True


def _view_names_for_job(cfg: DimScanConfig, job_type: str, job_id: str) -> list[str]:
    job_dir = get_job_dir(cfg, job_type, job_id)
    session = read_json_if_exists(job_dir / getattr(cfg, "session_filename", "session.json"))
    if isinstance(session, dict):
        captured_views = session.get("captured_views")
        if isinstance(captured_views, list) and all(isinstance(name, str) for name in captured_views):
            existing_captured = [
                name
                for name in captured_views
                if _capture_artifacts_available(cfg, job_dir / name)
            ]
            if existing_captured:
                return existing_captured

    if job_dir.is_dir():
        available_views = [
            path.name
            for path in sorted(job_dir.iterdir())
            if path.is_dir() and path.name.startswith("view_") and _capture_artifacts_available(cfg, path)
        ]
        if available_views:
            return available_views

    arrangement = read_json_if_exists(job_dir / getattr(cfg, "arrangement_filename", "arrangement.json"))
    if isinstance(arrangement, dict):
        view_names = arrangement.get("default_view_names")
        if isinstance(view_names, list) and all(isinstance(name, str) for name in view_names):
            return view_names

    if not job_dir.is_dir():
        raise FileNotFoundError(f"Missing job directory: {job_dir}")
    return sorted(path.name for path in job_dir.iterdir() if path.is_dir() and path.name.startswith("view_"))


def process_view(
    cfg: DimScanConfig,
    job_type: str,
    job_id: str,
    *,
    view_name: str,
    force_segmentation: bool = False,
    debug_mode: bool = True,
    save_object_cloud: bool = True,
    geometry_primary_fast: bool = False,
) -> dict[str, Any]:
    """Process one existing view folder and write geometry.json."""
    with timed_stage("process_view_setup_s"):
        view_dir = get_job_dir(cfg, job_type, job_id) / view_name
    if geometry_primary_fast:
        object_extraction = extract_geometry_primary_object_cloud(
            cfg,
            view_dir,
            debug_mode=debug_mode,
            save_object_cloud=save_object_cloud,
        )
        with timed_stage("process_view_fast_segmentation_record_s"):
            segmentation = make_segmentation_record(
                status="partial",
                model_backend=getattr(cfg, "segmentation_backend", "unknown"),
                model_name=getattr(cfg, "yoloe_model_name", None) or getattr(cfg, "yoloe_model_path", None),
                prompts=list(getattr(cfg, "yoloe_prompts", [])),
                segment_statuses={
                    "object": "missing",
                    "pot": "missing",
                    "leaf": "missing",
                    "table": "missing",
                },
                pot_quality={
                    "status": "missing",
                    "usable_for_geometry": False,
                    "usable_for_model": False,
                    "trust_score": 0.0,
                    "reason": "ai1_validation_pending",
                    "metrics": {},
                },
                warnings=[
                    "ai1_validation_pending",
                    "geometry_primary_object_available",
                    "pot_segment_missing",
                    "leaf_segment_missing",
                    "table_segment_missing",
                ],
                artifacts={
                    "object_cloud_status": str((object_extraction.get("artifacts") or {}).get("object_cloud_status") or "ready"),
                    **(
                        {"object_cloud": str(view_dir / "object_cloud.ply")}
                        if (view_dir / "object_cloud.ply").is_file()
                        else {}
                    ),
                    **(
                        {"object_extraction_debug": str(view_dir / "debug" / "object_extraction_debug.json")}
                        if debug_mode
                        else {}
                    ),
                },
            )
            segmentation["_runtime_final_object_points"] = object_extraction.get("_runtime_final_object_points")
            write_json_atomic(view_dir / "segmentation.json", _public_json_value(segmentation))
    else:
        try:
            object_extraction = extract_geometry_primary_object_cloud(
                cfg,
                view_dir,
                debug_mode=debug_mode,
                save_object_cloud=save_object_cloud,
            )
        except Exception as exc:
            object_extraction = {
                "extraction_method": "roi_table_cluster",
                "authoritative_object_source": "geometry_cluster",
                "status": "failed",
                "error": str(exc),
                "final_point_count": 0,
                "ai1_role": "validation_only",
                "ai1_used_for_object_extraction": False,
            }
        try:
            segmentation = segment_view(
                cfg,
                view_dir,
                force=force_segmentation,
                debug_mode=debug_mode,
                object_extraction_result=object_extraction,
            )
        except Exception as exc:
            segmentation = make_segmentation_record(
                status="failed",
                model_backend=getattr(cfg, "segmentation_backend", "unknown"),
                model_name=getattr(cfg, "yoloe_model_name", None) or getattr(cfg, "yoloe_model_path", None),
                prompts=list(getattr(cfg, "yoloe_prompts", [])),
                reason=f"segmentation_exception: {exc}",
                warnings=["segmentation_exception_raw_cloud_fallback"],
            )
            write_json_atomic(view_dir / "segmentation.json", segmentation)
    with timed_stage("process_view_segmentation_mark_step_s"):
        mark_step(cfg, job_type, job_id, f"segmentation_processed_{view_name}")
    with timed_stage("process_view_runtime_object_contract_s"):
        runtime_object_points = (
            segmentation.get("_runtime_final_object_points")
            if isinstance(segmentation, dict)
            else None
        )
        if runtime_object_points is None and not geometry_primary_fast:
            runtime_object_points = object_extraction.get("_runtime_final_object_points")
        if runtime_object_points is not None:
            _ensure_in_memory_object_cloud_contract(view_dir, debug_mode=debug_mode)
    with timed_stage("geometry_s"):
        geometry = measure_view_geometry(
            view_dir,
            cfg=cfg,
            debug_mode=debug_mode,
            geometry_primary_object_points=runtime_object_points,
        )
    with timed_stage("process_view_geometry_write_s"):
        geometry_path = get_view_file_path(cfg, job_type, job_id, view_name, cfg.geometry_filename)
        write_json_atomic(geometry_path, geometry)
    with timed_stage("process_view_geometry_mark_step_s"):
        mark_step(cfg, job_type, job_id, f"geometry_processed_{view_name}")
    with timed_stage("process_view_result_public_json_s"):
        return {
            "segmentation": _public_json_value(segmentation),
            "geometry": geometry,
        }


def process_job(
    cfg: DimScanConfig,
    *,
    job_type: str,
    job_id: str,
    view_names: list[str] | None = None,
    process_view_names: list[str] | None = None,
    force_segmentation: bool = False,
    debug_mode: bool = True,
    save_object_cloud: bool = True,
    skip_valid_views: bool = False,
    write_combined: bool = True,
    quality_view_name: str | None = None,
    geometry_primary_fast: bool = False,
) -> dict[str, Any]:
    """Process all views for an existing job and rewrite features."""
    with timed_stage("process_job_setup_s"):
        active_job_type = validate_job_type(job_type)
        job_dir = get_job_dir(cfg, active_job_type, job_id)
        if not job_dir.is_dir():
            raise FileNotFoundError(f"Missing job directory: {job_dir}")

    with timed_stage("process_job_view_names_s"):
        active_view_names = view_names or _view_names_for_job(cfg, active_job_type, job_id)
        if not active_view_names:
            raise ValueError(f"No view directories found for job: {job_dir}")

    with timed_stage("process_job_metadata_read_s"):
        explicit_process_view_names = process_view_names is not None
        requested_process_views = set(active_view_names if process_view_names is None else process_view_names)
        view_mode = ""
        metadata = read_json_if_exists(job_dir / getattr(cfg, "job_metadata_filename", "job_metadata.json"), default={})
        if isinstance(metadata, dict):
            view_mode = str(metadata.get("view_mode", ""))

    view_geometry: dict[str, dict[str, Any]] = {}
    rewritten_view_names: list[str] = []
    for view_name in active_view_names:
        should_process = view_name in requested_process_views
        if force_segmentation and not explicit_process_view_names:
            should_process = True
        if skip_valid_views:
            with timed_stage("process_job_view_validity_check_s"):
                view_valid = _view_outputs_valid(
                    cfg,
                    active_job_type,
                    job_id,
                    view_name=view_name,
                )
            should_process = should_process or not view_valid
        if should_process:
            save_object_cloud_kwargs = {} if save_object_cloud else {"save_object_cloud": False}
            result = process_view(
                cfg,
                active_job_type,
                job_id,
                view_name=view_name,
                force_segmentation=force_segmentation,
                debug_mode=debug_mode,
                geometry_primary_fast=geometry_primary_fast,
                **save_object_cloud_kwargs,
            )
            with timed_stage("per_view_features_s"):
                write_view_features(
                    cfg,
                    active_job_type,
                    job_id,
                    view_name=view_name,
                    view_role=infer_view_role(view_name, view_mode),
                    debug_mode=debug_mode,
                )
            rewritten_view_names.append(view_name)
        else:
            with timed_stage("process_job_existing_view_read_s"):
                result = _read_existing_view_result(
                    cfg,
                    active_job_type,
                    job_id,
                    view_name=view_name,
                )
                read_view_features_for_combine(
                    cfg,
                    active_job_type,
                    job_id,
                    view_name=view_name,
                    view_role=infer_view_role(view_name, view_mode),
                )
        view_geometry[view_name] = result

    if write_combined:
        with timed_stage("combined_features_s"):
            combined_features = write_combined_features(
                cfg,
                active_job_type,
                job_id,
                view_names=active_view_names,
                rewrite_view_names=[],
                debug_mode=debug_mode,
            )
        with timed_stage("quality_summary_s"):
            quality_summary = build_quality_summary(
                cfg,
                job_type=active_job_type,
                job_id=job_id,
                view_name=quality_view_name or active_view_names[0],
            )
        with timed_stage("process_job_status_write_s"):
            set_status(cfg, active_job_type, job_id, "geometry_processed")
    else:
        combined_features = None
        quality_summary = None
    with timed_stage("process_job_result_assembly_s"):
        return {
            "job_id": job_id,
            "job_type": active_job_type,
            "job_dir": str(job_dir),
            "view_names": active_view_names,
            "processed_view_names": rewritten_view_names,
            "view_geometry": {
                view_name: result["geometry"] for view_name, result in view_geometry.items()
            },
            "view_segmentation": {
                view_name: result["segmentation"] for view_name, result in view_geometry.items()
            },
            "combined_features": combined_features,
            "quality_summary": quality_summary,
        }


def build_parser() -> argparse.ArgumentParser:
    """Build the process-job CLI parser."""
    parser = argparse.ArgumentParser(description="Process an existing DimScan job.")
    parser.add_argument("--job_type", required=True, choices=("single", "group"))
    parser.add_argument("--job_id", required=True)
    parser.add_argument("--force_segmentation", action="store_true")
    return parser


def main() -> None:
    """Run geometry processing from the command line."""
    args = build_parser().parse_args()
    result = process_job(
        DimScanConfig(),
        job_type=args.job_type,
        job_id=args.job_id,
        force_segmentation=args.force_segmentation,
    )
    print(f"processed {result['job_type']} job {result['job_id']}")
    for view_name, geometry in result["view_geometry"].items():
        print(f"{view_name}: {geometry.get('status')} {geometry.get('dimensions_in')}")
    print(f"quality: {result['quality_summary'].get('status_label')}")


if __name__ == "__main__":
    main()
