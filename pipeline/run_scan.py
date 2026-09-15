"""High-level fake scan orchestration for DimScan development workflows."""

from __future__ import annotations

from typing import Any

from app.config import DimScanConfig
from capture.camera import CameraInterface, create_camera
from features.extractor import write_combined_features
from geometry.measure import make_placeholder_geometry
from packing.rules import suggest_box_from_combined_features
from pipeline.scan_writer import capture_view, initialize_arranged_job
from pipeline.session import mark_step, set_status
from utils.io import write_json_atomic
from utils.paths import get_job_file_path, get_view_file_path


def write_placeholder_geometry_for_view(
    cfg: DimScanConfig,
    job_type: str,
    job_id: str | None,
    *,
    view_name: str,
    height_in: float = 20.0,
    width_in: float = 12.0,
    length_in: float = 12.0,
    point_count: int | None = None,
) -> dict[str, Any]:
    """Write development-only placeholder geometry for one view."""
    geometry = make_placeholder_geometry(
        height_in=height_in,
        width_in=width_in,
        length_in=length_in,
        point_count=point_count,
    )
    geometry_filename = getattr(cfg, "geometry_filename", "geometry.json")
    geometry_path = get_view_file_path(cfg, job_type, job_id, view_name, geometry_filename)
    write_json_atomic(geometry_path, geometry)
    mark_step(cfg, job_type, job_id, f"{view_name}_geometry_written")
    return geometry


def run_fake_data_collection_scan(
    cfg: DimScanConfig,
    *,
    job_id: str | None,
    job_type: str,
    arrangement_type: str,
    items: list[dict[str, Any]],
    operator_id: str | None = None,
    camera: CameraInterface | None = None,
    strict_quantity: bool = True,
    make_box_suggestion: bool = True,
) -> dict[str, Any]:
    """Run a development fake scan from arranged job creation through features."""
    arranged_job = initialize_arranged_job(
        cfg,
        job_id=job_id,
        mode=getattr(cfg, "mode_data_collection", "data_collection"),
        job_type=job_type,
        arrangement_type=arrangement_type,
        items=items,
        operator_id=operator_id,
        strict_quantity=strict_quantity,
        allow_existing=True,
    )
    job_id = arranged_job["job_id"]
    active_camera = camera or create_camera("fake")
    view_names = arranged_job["view_names"]

    capture_artifacts_by_view: dict[str, dict[str, str]] = {}
    for view_name in view_names:
        capture_artifacts_by_view[view_name] = capture_view(
            cfg,
            job_type,
            job_id,
            view_name=view_name,
            camera=active_camera,
            overwrite=True,
        )
        write_placeholder_geometry_for_view(
            cfg,
            job_type,
            job_id,
            view_name=view_name,
        )

    combined_features = write_combined_features(
        cfg,
        job_type,
        job_id,
        view_names=view_names,
    )

    result: dict[str, Any] = {
        "job_id": job_id,
        "job_type": job_type,
        "arrangement": arranged_job["arrangement"],
        "view_names": view_names,
        "capture_artifacts_by_view": capture_artifacts_by_view,
        "combined_features": combined_features,
    }

    if make_box_suggestion:
        box_suggestion = suggest_box_from_combined_features(combined_features)
        box_suggestion_filename = getattr(cfg, "box_suggestion_filename", "box_suggestion.json")
        write_json_atomic(
            get_job_file_path(cfg, job_type, job_id, box_suggestion_filename),
            box_suggestion,
        )
        result["box_suggestion"] = box_suggestion

    mark_step(cfg, job_type, job_id, "fake_scan_completed")
    set_status(cfg, job_type, job_id, "fake_scan_completed")
    return result
