"""Real data collection orchestration for DimScan camera captures."""

from __future__ import annotations

import shutil
from typing import Any

from app.config import DimScanConfig
from capture.orbbec_camera import create_orbbec_camera
from pipeline.arrangement import default_view_names
from pipeline.process_job import _public_json_value, process_job
from pipeline.profiling import timed_stage
from pipeline.scan_writer import capture_view, initialize_arranged_job
from pipeline.session import load_session, mark_step, set_status
from utils.io import read_json_if_exists
from utils.paths import get_job_dir, get_job_file_path, resolve_collection_job_id


def _view_id_from_index(view_index: Any) -> str:
    index = int(view_index)
    if index <= 0:
        raise ValueError("view_index must be positive")
    return f"view_{index:02d}"


def _captured_views(cfg: DimScanConfig, job_type: str, job_id: str) -> list[str]:
    session = load_session(cfg, job_type, job_id)
    captured = session.get("captured_views")
    if isinstance(captured, list):
        return [view_name for view_name in captured if isinstance(view_name, str)]
    return []


def _view_capture_exists(cfg: DimScanConfig, job_type: str, job_id: str, view_id: str) -> bool:
    view_dir = get_job_dir(cfg, job_type, job_id) / view_id
    required = (
        getattr(cfg, "capture_meta_filename", "capture_meta.json"),
        getattr(cfg, "rgb_filename", "rgb.png"),
        getattr(cfg, "depth_raw_filename", "depth_raw.npy"),
    )
    return all((view_dir / filename).is_file() for filename in required)


def _select_view_id(
    cfg: DimScanConfig,
    job_type: str,
    job_id: str,
    *,
    required_views: list[str],
    view_id: str | None,
    view_index: Any = None,
    overwrite: bool = False,
) -> str:
    requested_view = view_id or (_view_id_from_index(view_index) if view_index is not None else None)
    if requested_view is not None:
        if requested_view not in required_views:
            raise ValueError(f"Unsupported view_id {requested_view!r}; expected one of {required_views!r}")
        if _view_capture_exists(cfg, job_type, job_id, requested_view) and not overwrite:
            raise ValueError(f"{requested_view} already exists; pass overwrite=true to recapture it")
        return requested_view

    captured = set(_captured_views(cfg, job_type, job_id))
    for candidate in required_views:
        if candidate not in captured and not _view_capture_exists(cfg, job_type, job_id, candidate):
            return candidate

    if overwrite:
        return required_views[-1]
    raise ValueError("All required views are already captured; pass overwrite=true to recapture a view")


def _load_existing_arranged_job(
    cfg: DimScanConfig,
    *,
    job_type: str,
    job_id: str,
) -> dict[str, Any] | None:
    job_dir = get_job_dir(cfg, job_type, job_id)
    if not job_dir.is_dir():
        return None

    metadata = read_json_if_exists(get_job_file_path(cfg, job_type, job_id, cfg.job_metadata_filename))
    arrangement = read_json_if_exists(get_job_file_path(cfg, job_type, job_id, cfg.arrangement_filename))
    if not isinstance(metadata, dict) or not isinstance(arrangement, dict):
        return None

    if job_type == cfg.job_type_single and metadata.get("view_mode") != cfg.view_mode_two_view_rectangle:
        view_names = ["view_01"]
    else:
        view_names = arrangement.get("default_view_names")
    if not isinstance(view_names, list) or not all(isinstance(name, str) for name in view_names):
        view_names = default_view_names(
            str(metadata.get("arrangement_type", "1x1")),
            str(metadata.get("view_mode", "")) or None,
        )

    return {
        "job_id": job_id,
        "job_dir": job_dir,
        "job_metadata": metadata,
        "arrangement": arrangement,
        "view_names": view_names,
        "view_mode": metadata.get("view_mode"),
    }


def _load_committed_arranged_job(
    cfg: DimScanConfig,
    *,
    job_type: str,
    job_id: str,
) -> dict[str, Any]:
    """Load and validate an explicitly committed collection job."""
    job_dir = get_job_dir(cfg, job_type, job_id)
    if not job_dir.is_dir():
        raise FileNotFoundError(f"Committed collection job directory does not exist: {job_dir}")

    metadata_path = get_job_file_path(cfg, job_type, job_id, cfg.job_metadata_filename)
    metadata = read_json_if_exists(metadata_path)
    if not isinstance(metadata, dict):
        raise ValueError(f"Committed collection job metadata is missing or invalid: {metadata_path}")
    if metadata.get("job_id") != job_id:
        raise ValueError(
            "Committed collection job metadata job_id mismatch: "
            f"expected {job_id!r}, found {metadata.get('job_id')!r}"
        )
    if metadata.get("job_type") != job_type:
        raise ValueError(
            "Committed collection job metadata job_type mismatch: "
            f"expected {job_type!r}, found {metadata.get('job_type')!r}"
        )
    expected_mode = getattr(cfg, "mode_data_collection", "data_collection")
    if metadata.get("mode") != expected_mode:
        raise ValueError(
            "Committed collection job metadata mode mismatch: "
            f"expected {expected_mode!r}, found {metadata.get('mode')!r}"
        )

    arranged_job = _load_existing_arranged_job(cfg, job_type=job_type, job_id=job_id)
    if arranged_job is None:
        arrangement_path = get_job_file_path(cfg, job_type, job_id, cfg.arrangement_filename)
        raise ValueError(f"Committed collection job arrangement is missing or invalid: {arrangement_path}")
    return arranged_job


def _view_progress(captured_views: list[str], required_views: list[str]) -> dict[str, Any]:
    remaining_views = [view_name for view_name in required_views if view_name not in captured_views]
    if not remaining_views:
        next_action = "All captures completed. Ground Truth is ready."
    elif len(required_views) > 1 and captured_views:
        next_action = "Rotate arrangement 90 degrees and capture View 02 / Width Side."
    else:
        next_action = "Capture View 01 / Front Side." if len(required_views) > 1 else "Capture View 01."
    return {
        "captured_views": captured_views,
        "required_views": required_views,
        "required_view_count": len(required_views),
        "remaining_views": remaining_views,
        "next_action": next_action,
    }


def _delete_view_capture_artifacts(
    cfg: DimScanConfig,
    job_type: str,
    job_id: str,
    view_name: str,
) -> None:
    view_dir = get_job_dir(cfg, job_type, job_id) / view_name
    if view_dir.is_dir():
        shutil.rmtree(view_dir)


def run_real_data_collection_scan(
    cfg: DimScanConfig,
    *,
    job_id: str | None,
    job_type: str,
    arrangement_type: str,
    items: list[dict[str, Any]],
    operator_id: str | None = None,
    camera: Any = None,
    strict_quantity: bool = False,
    write_placeholder_geometry: bool = False,
    write_features: bool = True,
    prompt_for_views: bool = True,
    view_id: str | None = None,
    view_index: Any = None,
    overwrite: bool = False,
    debug_mode: bool = True,
    save_object_cloud: bool = True,
    shape_mode: str | None = None,
) -> dict[str, Any]:
    """Run an operator-assisted real camera data collection scan."""
    with timed_stage("real_capture_camera_select_s"):
        active_camera = camera or create_orbbec_camera()
    with timed_stage("real_capture_job_resolve_load_s"):
        explicit_job_id = job_id is not None and bool(str(job_id).strip())
        resolved_job_id = resolve_collection_job_id(job_type, job_id) if explicit_job_id else None
        arranged_job = (
            _load_committed_arranged_job(cfg, job_type=job_type, job_id=resolved_job_id)
            if explicit_job_id
            else None
        )
    if arranged_job is None:
        with timed_stage("real_capture_job_initialize_s"):
            arranged_job = initialize_arranged_job(
                cfg,
                job_id=resolved_job_id,
                mode=getattr(cfg, "mode_data_collection", "data_collection"),
                job_type=job_type,
                arrangement_type=arrangement_type,
                items=items,
                operator_id=operator_id,
                strict_quantity=strict_quantity,
                shape_mode=shape_mode,
            )
    job_id = arranged_job["job_id"]
    view_names = arranged_job["view_names"]
    with timed_stage("real_capture_view_select_s"):
        current_view_id = _select_view_id(
            cfg,
            job_type,
            job_id,
            required_views=view_names,
            view_id=view_id,
            view_index=view_index,
            overwrite=overwrite,
        )
    capture_artifacts_by_view: dict[str, dict[str, str]] = {}
    current_view_dir = get_job_dir(cfg, job_type, job_id) / current_view_id
    with timed_stage("real_capture_existing_view_check_s"):
        replacing_existing_view = overwrite and (
            current_view_id in _captured_views(cfg, job_type, job_id) or current_view_dir.exists()
        )

    if prompt_for_views:
        input(f"Position view {current_view_id}, then press Enter to capture...")
    if replacing_existing_view:
        with timed_stage("real_capture_delete_existing_view_s"):
            _delete_view_capture_artifacts(cfg, job_type, job_id, current_view_id)
    capture_artifacts_by_view[current_view_id] = capture_view(
        cfg,
        job_type,
        job_id,
        view_name=current_view_id,
        camera=active_camera,
        overwrite=overwrite,
        debug_mode=debug_mode,
    )
    with timed_stage("real_capture_progress_s"):
        captured_views = _captured_views(cfg, job_type, job_id)
        progress = _view_progress(captured_views, view_names)

    result: dict[str, Any] = {
        "job_id": job_id,
        "job_type": job_type,
        "arrangement": arranged_job["arrangement"],
        "view_names": view_names,
        "current_view_id": current_view_id,
        "capture_artifacts_by_view": capture_artifacts_by_view,
    }

    if write_features:
        with timed_stage("real_capture_all_views_read_s"):
            all_captured_views = _captured_views(cfg, job_type, job_id)
        save_object_cloud_kwargs = {} if save_object_cloud else {"save_object_cloud": False}
        with timed_stage("real_capture_process_job_call_s"):
            if progress["remaining_views"]:
                processing_result = process_job(
                    cfg,
                    job_type=job_type,
                    job_id=job_id,
                    view_names=[current_view_id],
                    process_view_names=[current_view_id],
                    force_segmentation=True,
                    debug_mode=debug_mode,
                    write_combined=False,
                    geometry_primary_fast=False,
                    **save_object_cloud_kwargs,
                )
            else:
                processing_result = process_job(
                    cfg,
                    job_type=job_type,
                    job_id=job_id,
                    view_names=all_captured_views,
                    process_view_names=[current_view_id],
                    force_segmentation=True,
                    debug_mode=debug_mode,
                    skip_valid_views=True,
                    write_combined=True,
                    quality_view_name=current_view_id if replacing_existing_view else None,
                    geometry_primary_fast=False,
                    **save_object_cloud_kwargs,
                )
        with timed_stage("real_capture_public_json_conversion_s"):
            result["geometry_by_view"] = _public_json_value(processing_result["view_geometry"])
            result["segmentation_by_view"] = _public_json_value(processing_result["view_segmentation"])
            if processing_result.get("combined_features") is not None:
                result["combined_features"] = _public_json_value(processing_result["combined_features"])
            if processing_result.get("quality_summary") is not None:
                result["quality_summary"] = _public_json_value(processing_result["quality_summary"])

    result.update(progress)

    with timed_stage("real_capture_session_finalize_s"):
        mark_step(cfg, job_type, job_id, "real_capture_completed")
        if result["remaining_views"]:
            set_status(cfg, job_type, job_id, "capture_in_progress")
        else:
            set_status(cfg, job_type, job_id, "real_capture_completed")
    return result
