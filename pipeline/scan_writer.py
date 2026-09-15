"""Job folder and metadata writers for DimScan data collection jobs."""

from __future__ import annotations

import shutil
from typing import Any

import numpy as np

from app.config import DimScanConfig
from capture.camera import CameraInterface
from capture.pointcloud import (
    colors_from_rgb_bytes,
    metric_points_from_depth,
    valid_depth_pixel_indices_from_depth,
    write_ascii_ply,
    write_ascii_ply_with_colors,
    write_capture_meta,
    write_depth_aligned_to_rgb_npy,
    write_cloud_pixel_indices_npy,
    write_depth_preview_png,
    write_depth_raw_npy,
    write_pointcloud_from_depth,
    write_rgb_png,
)
from pipeline.arrangement import (
    default_view_names,
    infer_view_mode,
    make_arrangement_record,
)
from pipeline.profiling import add_timing, timed_stage
from pipeline.session import create_session, mark_step, record_captured_view, set_status
from utils.io import read_json_if_exists, utc_now_iso, write_json_atomic
from utils.paths import create_job_folder, generate_collection_job_id, generate_job_id, get_job_dir, get_job_file_path, get_view_file_path, resolve_collection_job_id, resolve_job_id


def create_job_metadata(
    *,
    job_id: str,
    mode: str,
    job_type: str,
    arrangement_type: str,
    view_mode: str,
    operator_id: str | None = None,
    shape_mode: str | None = None,
) -> dict[str, Any]:
    """Create job-level metadata for a DimScan job."""
    payload = {
        "job_id": job_id,
        "mode": mode,
        "job_type": job_type,
        "arrangement_type": arrangement_type,
        "view_mode": view_mode,
        "created_at": utc_now_iso(),
        "operator_id": operator_id,
    }
    if shape_mode is not None:
        payload["shape_mode"] = shape_mode
    return payload


def create_item_list(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Create an item list summary and validate positive quantities."""
    total_quantity = 0
    unique_skus: set[str] = set()

    for item in items:
        quantity = item.get("quantity")
        if not isinstance(quantity, int) or quantity <= 0:
            raise ValueError(f"Invalid item quantity: {quantity!r}")

        total_quantity += quantity
        sku = item.get("sku")
        if isinstance(sku, str) and sku:
            unique_skus.add(sku)

    return {
        "items": items,
        "total_quantity": total_quantity,
        "unique_sku_count": len(unique_skus),
    }


def create_item_metadata(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a job-level SKU metadata audit artifact."""
    records = []
    for item in items:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        records.append(
            {
                "sku": item.get("sku"),
                "quantity": item.get("quantity"),
                "lookup_status": metadata.get("lookup_status") or ("found" if metadata.get("known") else "not_found"),
                "operator_item": {
                    "sku": item.get("sku"),
                    "quantity": item.get("quantity"),
                },
                "catalogue_entry": metadata if metadata.get("known") else None,
                "pot_prior": metadata.get("pot_prior"),
                "catalogue": metadata.get("catalogue"),
            }
        )
    return {
        "items": records,
    }


def initialize_job(
    cfg: DimScanConfig,
    *,
    job_id: str | None,
    mode: str,
    job_type: str,
    arrangement_type: str,
    view_mode: str,
    items: list[dict[str, Any]],
    view_names: list[str] | None = None,
    operator_id: str | None = None,
    shape_mode: str | None = None,
    allow_existing: bool = False,
) -> dict[str, Any]:
    """Create a job folder and write initial job metadata files."""
    if job_id is None or not str(job_id).strip():
        if mode == getattr(cfg, "mode_data_collection", "data_collection"):
            job_id = generate_collection_job_id(job_type, items)
        else:
            job_id = generate_job_id(job_type)
    job_id = (
        resolve_collection_job_id(job_type, job_id)
        if mode == getattr(cfg, "mode_data_collection", "data_collection")
        else resolve_job_id(job_type, job_id)
    )
    if get_job_dir(cfg, job_type, job_id).exists() and not allow_existing:
        raise FileExistsError(f"Job already exists: {job_id}")
    create_view_dirs = mode != getattr(cfg, "mode_data_collection", "data_collection")
    job_dir = create_job_folder(cfg, job_type, job_id, view_names if create_view_dirs else None)
    job_metadata = create_job_metadata(
        job_id=job_id,
        mode=mode,
        job_type=job_type,
        arrangement_type=arrangement_type,
        view_mode=view_mode,
        operator_id=operator_id,
        shape_mode=shape_mode,
    )
    item_list = create_item_list(items)

    write_json_atomic(
        get_job_file_path(cfg, job_type, job_id, cfg.job_metadata_filename),
        job_metadata,
    )
    write_json_atomic(
        get_job_file_path(cfg, job_type, job_id, cfg.item_list_filename),
        item_list,
    )
    write_json_atomic(
        get_job_file_path(cfg, job_type, job_id, "item_metadata.json"),
        create_item_metadata(items),
    )
    create_session(cfg, job_type, job_id)
    mark_step(cfg, job_type, job_id, "metadata_written")
    mark_step(cfg, job_type, job_id, "item_list_written")
    mark_step(cfg, job_type, job_id, "item_metadata_written")
    set_status(cfg, job_type, job_id, "initialized")

    return {
        "job_id": job_id,
        "job_dir": job_dir,
        "job_metadata": job_metadata,
        "item_list": item_list,
    }


def initialize_arranged_job(
    cfg: DimScanConfig,
    *,
    job_id: str | None,
    mode: str,
    job_type: str,
    arrangement_type: str,
    items: list[dict[str, Any]],
    operator_id: str | None = None,
    strict_quantity: bool = True,
    allow_existing: bool = False,
    shape_mode: str | None = None,
) -> dict[str, Any]:
    """Initialize a job using arrangement-derived view mode and view names."""
    arrangement = make_arrangement_record(
        arrangement_type,
        items=items,
        strict_quantity=strict_quantity,
    )
    normalized_shape = str(shape_mode or "").strip().lower()
    if job_type == cfg.job_type_single and normalized_shape == "rectangular":
        view_mode = cfg.view_mode_two_view_rectangle
        view_names = ["view_01", "view_02"]
        arrangement["view_mode"] = view_mode
        arrangement["default_view_names"] = view_names
    elif job_type == cfg.job_type_single:
        view_mode = cfg.view_mode_single
        view_names = ["view_01"]
        arrangement["view_mode"] = view_mode
        arrangement["default_view_names"] = view_names
    else:
        view_mode = infer_view_mode(arrangement["arrangement_type"])
        view_names = default_view_names(arrangement["arrangement_type"], view_mode)

    result = initialize_job(
        cfg,
        job_id=job_id,
        mode=mode,
        job_type=job_type,
        arrangement_type=arrangement["arrangement_type"],
        view_mode=view_mode,
        items=items,
        view_names=view_names,
        operator_id=operator_id,
        shape_mode=normalized_shape or None,
        allow_existing=allow_existing,
    )
    job_id = result["job_id"]

    arrangement_filename = getattr(cfg, "arrangement_filename", "arrangement.json")
    write_json_atomic(
        get_job_file_path(cfg, job_type, job_id, arrangement_filename),
        arrangement,
    )
    mark_step(cfg, job_type, job_id, "arrangement_written")

    result["arrangement"] = arrangement
    result["view_names"] = view_names
    result["view_mode"] = view_mode
    return result


def update_collection_job_items(
    cfg: DimScanConfig,
    *,
    job_type: str,
    job_id: str,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Persist the complete current item list without recreating an existing job."""
    job_dir = get_job_dir(cfg, job_type, job_id)
    metadata = read_json_if_exists(job_dir / cfg.job_metadata_filename)
    arrangement = read_json_if_exists(job_dir / cfg.arrangement_filename)
    if not isinstance(metadata, dict) or metadata.get("job_type") != job_type:
        raise ValueError(f"Existing job metadata does not match {job_type}: {job_id}")
    if not isinstance(arrangement, dict):
        raise ValueError(f"Existing job arrangement is missing: {job_id}")

    item_list = create_item_list(items)
    updated_arrangement = make_arrangement_record(
        str(metadata.get("arrangement_type") or "1x1"),
        items=items,
        strict_quantity=False,
    )
    updated_arrangement["view_mode"] = metadata.get("view_mode")
    updated_arrangement["default_view_names"] = arrangement.get("default_view_names")
    write_json_atomic(job_dir / cfg.item_list_filename, item_list)
    write_json_atomic(job_dir / "item_metadata.json", create_item_metadata(items))
    write_json_atomic(job_dir / cfg.arrangement_filename, updated_arrangement)
    mark_step(cfg, job_type, job_id, "item_list_written")
    mark_step(cfg, job_type, job_id, "item_metadata_written")
    return {
        "job_id": job_id,
        "job_dir": job_dir,
        "job_metadata": metadata,
        "item_list": item_list,
        "arrangement": updated_arrangement,
        "view_names": updated_arrangement.get("default_view_names") or [],
        "view_mode": metadata.get("view_mode"),
    }


def delete_setup_only_collection_job(
    cfg: DimScanConfig,
    *,
    job_type: str,
    job_id: str,
) -> None:
    """Delete an exact committed collection job only while it contains setup metadata."""
    job_dir = get_job_dir(cfg, job_type, job_id)
    if not job_dir.is_dir():
        raise FileNotFoundError(f"Committed collection job directory does not exist: {job_dir}")

    metadata = read_json_if_exists(job_dir / cfg.job_metadata_filename)
    if not isinstance(metadata, dict):
        raise ValueError(f"Cannot remove job with missing or invalid metadata: {job_id}")
    if metadata.get("job_id") != job_id:
        raise ValueError(f"Cannot remove job whose metadata job_id does not match: {job_id}")
    if metadata.get("job_type") != job_type:
        raise ValueError(f"Cannot remove job whose metadata job_type does not match: {job_id}")
    if metadata.get("mode") != getattr(cfg, "mode_data_collection", "data_collection"):
        raise ValueError(f"Cannot remove non-collection job: {job_id}")

    setup_files = {
        cfg.job_metadata_filename,
        cfg.item_list_filename,
        "item_metadata.json",
        cfg.session_filename,
        cfg.arrangement_filename,
    }
    unexpected = sorted(path.name for path in job_dir.iterdir() if path.name not in setup_files)
    if unexpected:
        raise ValueError(
            "Cannot remove item because the job contains capture or processing data: "
            + ", ".join(unexpected)
        )
    shutil.rmtree(job_dir)


def write_capture_artifacts(
    cfg: DimScanConfig,
    job_type: str,
    job_id: str,
    *,
    view_name: str,
    frame: Any,
    debug_mode: bool = True,
) -> dict[str, str]:
    """Write RGB, depth, point cloud, and capture metadata artifacts."""
    rgb_filename = getattr(cfg, "rgb_filename", "rgb.png")
    depth_filename = getattr(cfg, "depth_filename", "depth.json")
    cloud_filename = getattr(cfg, "cloud_filename", "cloud.ply")
    depth_raw_filename = getattr(cfg, "depth_raw_filename", "depth_raw.npy")
    depth_aligned_to_rgb_filename = getattr(cfg, "depth_aligned_to_rgb_filename", "depth_aligned_to_rgb.npy")
    capture_meta_filename = getattr(cfg, "capture_meta_filename", "capture_meta.json")
    cloud_pixel_indices_filename = getattr(cfg, "cloud_pixel_indices_filename", "cloud_pixel_indices.npy")

    rgb_path = get_view_file_path(cfg, job_type, job_id, view_name, rgb_filename)
    depth_path = get_view_file_path(cfg, job_type, job_id, view_name, depth_filename)
    depth_raw_path = get_view_file_path(cfg, job_type, job_id, view_name, depth_raw_filename)
    depth_aligned_to_rgb_path = get_view_file_path(
        cfg,
        job_type,
        job_id,
        view_name,
        depth_aligned_to_rgb_filename,
    )
    cloud_path = get_view_file_path(cfg, job_type, job_id, view_name, cloud_filename)
    capture_meta_path = get_view_file_path(cfg, job_type, job_id, view_name, capture_meta_filename)
    cloud_pixel_indices_path = get_view_file_path(cfg, job_type, job_id, view_name, cloud_pixel_indices_filename)
    write_dense_cloud = bool(debug_mode)

    with timed_stage("capture_artifact_prepare_arrays_s"):
        metadata = getattr(frame, "metadata", {}) or {}
        depth_aligned_to_rgb_values = getattr(frame, "depth_aligned_to_rgb_values", None)
        with timed_stage("raw_depth_array_prepare_s"):
            raw_depth_array = np.asarray(frame.depth_values, dtype=np.float32)
        rgb_width = int(metadata.get("rgb_width") or metadata.get("width") or 0)
        rgb_height = int(metadata.get("rgb_height") or metadata.get("height") or 0)
        aligned_depth_path_meta: str | None = None
    with timed_stage("rgb_depth_npy_writes_s"):
        with timed_stage("rgb_png_write_s"):
            write_rgb_png(rgb_path, frame.rgb_bytes, metadata)
        if depth_aligned_to_rgb_values is frame.depth_values:
            raise ValueError("Raw depth and aligned depth must be stored in separate frame fields.")
        if depth_aligned_to_rgb_values is not None:
            with timed_stage("aligned_depth_array_prepare_s"):
                aligned_depth_array = np.asarray(depth_aligned_to_rgb_values, dtype=np.float32)
            if aligned_depth_array.ndim != 2:
                raise ValueError(f"Aligned depth must be a 2D array, got shape {aligned_depth_array.shape}.")
            if rgb_width > 0 and rgb_height > 0 and aligned_depth_array.shape != (rgb_height, rgb_width):
                raise ValueError(
                    "Aligned depth must match RGB shape; "
                    f"rgb={(rgb_height, rgb_width)}, aligned_depth={aligned_depth_array.shape}."
                )
            if raw_depth_array.shape == aligned_depth_array.shape and np.array_equal(raw_depth_array, aligned_depth_array):
                raise ValueError("Raw depth and aligned depth arrays are identical; true raw depth was not preserved.")
            with timed_stage("aligned_depth_npy_write_s"):
                write_depth_aligned_to_rgb_npy(depth_aligned_to_rgb_path, depth_aligned_to_rgb_values)
            aligned_depth_path_meta = str(depth_aligned_to_rgb_path)
        with timed_stage("raw_depth_npy_write_s"):
            write_depth_raw_npy(depth_raw_path, frame.depth_values)
        write_depth_preview_png(depth_path, frame.depth_values)

    point_cloud_points = getattr(frame, "point_cloud_points", None)
    depth_intrinsics = metadata.get("depth_intrinsics")
    cloud_type = "metric_xyz" if isinstance(depth_intrinsics, dict) or point_cloud_points else "pixel_grid_depth_fallback"
    cloud_pixel_indices_meta_path: str | None = None
    cloud_pixel_indices_count: int | None = None
    cloud_written = False

    if write_dense_cloud:
        with timed_stage("dense_cloud_generation_write_s"):
            metric_points = []
            cloud_pixel_indices = None
            if isinstance(depth_intrinsics, dict):
                metric_points = metric_points_from_depth(frame.depth_values, depth_intrinsics)
                cloud_pixel_indices = valid_depth_pixel_indices_from_depth(frame.depth_values)

            if metric_points:
                if cloud_pixel_indices is None or len(cloud_pixel_indices) != len(metric_points):
                    raise ValueError("Cloud pixel index map count does not match metric point count")
                depth_width = len(frame.depth_values[0]) if frame.depth_values else 0
                depth_height = len(frame.depth_values)
                colors = colors_from_rgb_bytes(
                    frame.rgb_bytes,
                    metadata,
                    target_width=depth_width,
                    target_height=depth_height,
                )
                if colors and len(colors) == depth_width * depth_height:
                    filtered_colors = [
                        colors[row_index * depth_width + col_index]
                        for row_index, row in enumerate(frame.depth_values)
                        for col_index, depth_value in enumerate(row)
                        if depth_value and depth_value > 0
                    ]
                else:
                    filtered_colors = None
                write_ascii_ply_with_colors(cloud_path, metric_points, filtered_colors)
                write_cloud_pixel_indices_npy(cloud_pixel_indices_path, cloud_pixel_indices)
                cloud_pixel_indices_meta_path = str(cloud_pixel_indices_path)
                cloud_pixel_indices_count = int(len(cloud_pixel_indices))
                cloud_written = True
            elif point_cloud_points:
                write_ascii_ply(cloud_path, point_cloud_points)
                cloud_written = True
            else:
                write_pointcloud_from_depth(cloud_path, frame.depth_values)
                cloud_written = True
    else:
        add_timing("dense_cloud_generation_write_s", 0.0)

    with timed_stage("capture_meta_write_s"):
        write_capture_meta(
            capture_meta_path,
            metadata,
            frame.depth_values,
            depth_aligned_to_rgb_values=depth_aligned_to_rgb_values,
            cloud_type=cloud_type,
            cloud_pixel_indices_path=cloud_pixel_indices_meta_path,
            cloud_pixel_indices_count=cloud_pixel_indices_count,
            depth_aligned_to_rgb_path=aligned_depth_path_meta,
        )

    return {
        "rgb": str(rgb_path),
        "depth": str(depth_path),
        "depth_raw": str(depth_raw_path),
        "depth_aligned_to_rgb": str(depth_aligned_to_rgb_path) if aligned_depth_path_meta else "",
        "cloud": str(cloud_path) if cloud_written else "",
        "cloud_pixel_indices": str(cloud_pixel_indices_path) if cloud_pixel_indices_meta_path else "",
        "capture_meta": str(capture_meta_path),
    }

def _view_capture_exists(cfg: DimScanConfig, job_type: str, job_id: str, view_name: str) -> bool:
    view_dir = get_job_dir(cfg, job_type, job_id) / view_name
    required = (
        getattr(cfg, "capture_meta_filename", "capture_meta.json"),
        getattr(cfg, "rgb_filename", "rgb.png"),
        getattr(cfg, "depth_raw_filename", "depth_raw.npy"),
    )
    return all((view_dir / filename).is_file() for filename in required)


def capture_view(
    cfg: DimScanConfig,
    job_type: str,
    job_id: str,
    *,
    view_name: str,
    camera: CameraInterface,
    overwrite: bool = False,
    debug_mode: bool = True,
) -> dict[str, str]:
    """Capture one view with a camera interface and write artifacts."""
    if _view_capture_exists(cfg, job_type, job_id, view_name) and not overwrite:
        raise FileExistsError(f"View already captured and overwrite is false: {view_name}")

    with timed_stage("camera_acquisition_s"):
        frame = camera.capture_frame()
    artifacts = write_capture_artifacts(
        cfg,
        job_type,
        job_id,
        view_name=view_name,
        frame=frame,
        debug_mode=debug_mode,
    )
    with timed_stage("capture_session_mark_s"):
        mark_step(cfg, job_type, job_id, f"{view_name}_captured")
        record_captured_view(cfg, job_type, job_id, view_name)
    return artifacts
