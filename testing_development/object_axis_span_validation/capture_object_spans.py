"""Isolated RGB-camera axis/span validation capture tool.

This script is intentionally outside the production pipeline. It captures one
Orbbec RGB-D frameset, builds an object cloud from SDK-aligned depth plus an RGB
mask, applies the configured ROI, and reports raw X/Y/Z spans only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from app.config import DimScanConfig
from capture.orbbec_camera import OrbbecCamera
from capture.pointcloud import write_ascii_ply, write_ascii_ply_with_colors, write_rgb_png
from segmentation.yoloe_segmenter import (
    _apply_object_cloud_roi,
    _configure_yoloe_text_prompts,
    _depth_preview_array,
    _depth_scale_to_meters,
    _overlay_mask_on_image,
    _restore_masks_to_rgb_shape,
    _select_masks,
    build_object_cloud_from_aligned_depth,
    configured_model_source,
)


M_TO_IN = 39.3701
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
LAST_RUN_DIR: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture and report RGB-camera X/Y/Z object spans.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--model", type=str, default=None, help="YOLOE model path/name override")
    parser.add_argument("--save-unfiltered", action="store_true", help="Retain object_cloud_before_roi.ply")
    parser.add_argument("--no-segmentation", action="store_true", help="Use --mask or a central rectangle mask")
    parser.add_argument("--mask", type=Path, default=None, help="RGB-shape object mask PNG")
    return parser.parse_args()


def timestamped_run_dir(base_dir: Path) -> Path:
    run_dir = base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def failure_report(exc: Exception, run_dir: Path | None) -> None:
    target_dir = run_dir or timestamped_run_dir(DEFAULT_OUTPUT_DIR)
    payload = {
        "status": "failed",
        "error": str(exc),
        "coordinate_frame": "rgb_camera",
        "units": "meters",
        "warnings": [type(exc).__name__],
    }
    write_json(target_dir / "axis_span_report.json", payload)
    (target_dir / "axis_span_report.txt").write_text(
        f"DimScan Object Axis Span Validation\n\nStatus: failed\nError: {exc}\n",
        encoding="utf-8",
    )
    print(f"Axis span validation failed: {exc}")
    print(f"Failure report: {target_dir / 'axis_span_report.json'}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def rgb_array_from_frame(frame: Any, rgb_path: Path) -> np.ndarray:
    write_rgb_png(rgb_path, frame.rgb_bytes, frame.metadata)
    return np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)


def load_supplied_mask(mask_path: Path, rgb_shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    if mask.shape != rgb_shape:
        raise ValueError(f"supplied_mask_shape_mismatch:mask={mask.shape}:rgb={rgb_shape}")
    return mask


def central_rect_mask(rgb_shape: tuple[int, int]) -> np.ndarray:
    height, width = rgb_shape
    mask = np.zeros((height, width), dtype=bool)
    y0 = int(height * 0.18)
    y1 = int(height * 0.88)
    x0 = int(width * 0.25)
    x1 = int(width * 0.75)
    mask[y0:y1, x0:x1] = True
    return mask


def yoloe_object_mask(cfg: DimScanConfig, rgb_path: Path, rgb_size: tuple[int, int], model_override: str | None) -> tuple[np.ndarray, dict[str, Any]]:
    model_source = model_override or configured_model_source(cfg)
    if model_source is None:
        raise RuntimeError("yoloe_model_not_configured: pass --model or set DIMSCAN_YOLOE_MODEL_PATH")
    if (os.sep in model_source or model_source.endswith(".pt")) and not Path(model_source).is_file():
        raise FileNotFoundError(f"yoloe_model_not_found:{model_source}")

    try:
        from ultralytics import YOLOE
    except ImportError as exc:
        raise RuntimeError("ultralytics_not_installed") from exc

    model = YOLOE(model_source, verbose=False)
    prompt_error = _configure_yoloe_text_prompts(model, list(cfg.yoloe_prompts))
    if prompt_error is not None:
        raise RuntimeError(prompt_error)
    results = model.predict(source=str(rgb_path), conf=float(cfg.yoloe_confidence_threshold), verbose=False)
    if not results:
        raise RuntimeError("yoloe_returned_no_results")

    masks, confidences, prediction_debug = _select_masks(results[0])
    masks, restore_debug = _restore_masks_to_rgb_shape(masks, rgb_size=rgb_size)
    if "object" not in masks:
        raise RuntimeError("object_mask_missing_from_yoloe")
    return masks["object"], {
        "method": "production_yoloe_object_mask",
        "model_source": model_source,
        "object_confidence": confidences.get("object"),
        "prediction": prediction_debug,
        "mask_restoration": restore_debug.get("object"),
    }


def acquire_mask(
    *,
    cfg: DimScanConfig,
    rgb_path: Path,
    rgb_shape: tuple[int, int],
    rgb_size: tuple[int, int],
    mask_path: Path | None,
    no_segmentation: bool,
    model_override: str | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if mask_path is not None:
        return load_supplied_mask(mask_path, rgb_shape), {
            "method": "supplied_mask_no_resize",
            "path": str(mask_path),
        }
    if no_segmentation:
        return central_rect_mask(rgb_shape), {
            "method": "central_rect_debug_mask",
            "warning": "no_segmentation_requested_not_default_behavior",
        }
    return yoloe_object_mask(cfg, rgb_path, rgb_size, model_override)


def point_array(points: list[tuple[float, float, float]]) -> np.ndarray:
    return np.asarray(points, dtype=float).reshape((-1, 3))


def span_stats(points: list[tuple[float, float, float]]) -> dict[str, Any]:
    array = point_array(points)
    if array.size == 0:
        raise ValueError("cannot_calculate_spans_for_empty_cloud")
    mins = array.min(axis=0)
    maxs = array.max(axis=0)
    spans = maxs - mins
    return {
        "point_count": int(len(array)),
        "x_min_m": float(mins[0]),
        "x_max_m": float(maxs[0]),
        "x_span_m": float(spans[0]),
        "y_min_m": float(mins[1]),
        "y_max_m": float(maxs[1]),
        "y_span_m": float(spans[1]),
        "z_min_m": float(mins[2]),
        "z_max_m": float(maxs[2]),
        "z_span_m": float(spans[2]),
        "x_span_in": float(spans[0] * M_TO_IN),
        "y_span_in": float(spans[1] * M_TO_IN),
        "z_span_in": float(spans[2] * M_TO_IN),
    }


def axis_convention_from_deprojection() -> dict[str, str]:
    return {
        "x": "x = (col - cx) * z / fx; positive X is toward increasing image columns, normally image right",
        "y": "y = (row - cy) * z / fy; positive Y is toward increasing image rows, normally image down",
        "z": "z = aligned_depth_value * sdk_depth_scale_m_per_unit; positive Z is forward away from the RGB camera",
    }


def write_text_report(path: Path, report: dict[str, Any]) -> None:
    spans = report["spans"]
    roi = report["roi"]
    lines = [
        "DimScan Object Axis Span Validation",
        "",
        f"Status: {report['status']}",
        f"Coordinate frame: {report['coordinate_frame']}",
        f"Units: {report['units']}",
        f"Origin: {report['origin']}",
        "",
        "Axis convention:",
        f"  X: {report['axis_convention']['x']}",
        f"  Y: {report['axis_convention']['y']}",
        f"  Z: {report['axis_convention']['z']}",
        "",
        "ROI:",
        f"  source: {roi['source']}",
        f"  frame: {roi['frame']}",
        f"  units: {roi['units']}",
        f"  points before ROI: {roi['points_before_roi']}",
        f"  points after ROI: {roi['points_after_roi']}",
        f"  rejected count: {roi['rejected_count']}",
        "",
        "Spans:",
        f"  X: {spans['x_span_m']:.6f} m  ({spans['x_span_in']:.4f} in)",
        f"  Y: {spans['y_span_m']:.6f} m  ({spans['y_span_in']:.4f} in)",
        f"  Z: {spans['z_span_m']:.6f} m  ({spans['z_span_in']:.4f} in)",
        "",
        f"Report JSON: {report['artifacts']['axis_span_report_json']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> int:
    global LAST_RUN_DIR
    args = parse_args()
    cfg = DimScanConfig()
    run_dir = timestamped_run_dir(args.output_dir)
    LAST_RUN_DIR = run_dir

    rgb_path = run_dir / "rgb.png"
    raw_depth_path = run_dir / "depth_raw.npy"
    aligned_depth_path = run_dir / "depth_aligned_to_rgb.npy"
    mask_path = run_dir / "object_mask.png"
    overlay_path = run_dir / "object_mask_overlay_rgb.png"
    aligned_vis_path = run_dir / "depth_aligned_visualization.png"
    valid_overlay_path = run_dir / "valid_aligned_depth_overlay_rgb.png"
    before_roi_path = run_dir / "object_cloud_before_roi.ply"
    filtered_path = run_dir / "object_cloud_roi_filtered.ply"
    json_path = run_dir / "axis_span_report.json"
    text_path = run_dir / "axis_span_report.txt"

    camera = OrbbecCamera()
    try:
        frame = None
        for _ in range(max(0, int(args.warmup_frames))):
            frame = camera.capture_frame()
        frame = camera.capture_frame()
    finally:
        camera.close()

    metadata = dict(frame.metadata)
    rgb = rgb_array_from_frame(frame, rgb_path)
    raw_depth = np.asarray(frame.depth_values, dtype=np.float32)
    aligned_depth = np.asarray(frame.depth_aligned_to_rgb_values, dtype=np.float32)
    np.save(raw_depth_path, raw_depth)
    np.save(aligned_depth_path, aligned_depth)

    rgb_shape = tuple(rgb.shape[:2])
    if aligned_depth.shape != rgb_shape:
        raise ValueError(f"aligned_depth_shape_mismatch:aligned={aligned_depth.shape}:rgb={rgb_shape}")
    rgb_size = (int(rgb.shape[1]), int(rgb.shape[0]))
    object_mask, mask_debug = acquire_mask(
        cfg=cfg,
        rgb_path=rgb_path,
        rgb_shape=rgb_shape,
        rgb_size=rgb_size,
        mask_path=args.mask,
        no_segmentation=bool(args.no_segmentation),
        model_override=args.model,
    )
    if object_mask.shape != rgb_shape:
        raise ValueError(f"final_mask_shape_mismatch:mask={object_mask.shape}:rgb={rgb_shape}")

    Image.fromarray(object_mask.astype(np.uint8) * 255, mode="L").save(mask_path, format="PNG")
    _overlay_mask_on_image(Image.fromarray(rgb, mode="RGB"), object_mask, (255, 64, 64)).save(overlay_path, format="PNG")
    Image.fromarray(_depth_preview_array(aligned_depth), mode="L").save(aligned_vis_path, format="PNG")
    valid_aligned = np.isfinite(aligned_depth) & (aligned_depth > 0)
    _overlay_mask_on_image(Image.fromarray(rgb, mode="RGB"), valid_aligned, (64, 200, 80)).save(valid_overlay_path, format="PNG")

    rgb_intrinsics = metadata.get("rgb_intrinsics")
    if not isinstance(rgb_intrinsics, dict):
        raise ValueError("missing_rgb_intrinsics")
    depth_scale_to_meters, saved_units = _depth_scale_to_meters(metadata)
    before_points, before_colors, cloud_debug = build_object_cloud_from_aligned_depth(
        object_mask=object_mask,
        depth_aligned_to_rgb=aligned_depth,
        rgb_intrinsics=rgb_intrinsics,
        depth_scale_to_meters=depth_scale_to_meters,
        saved_depth_units=saved_units,
        rgb_image=rgb,
    )
    if before_colors:
        write_ascii_ply_with_colors(before_roi_path, before_points, before_colors)
    else:
        write_ascii_ply(before_roi_path, before_points)

    filtered_points, filtered_colors, rejected_points, _rejected_colors, roi_debug = _apply_object_cloud_roi(
        points=before_points,
        colors=before_colors,
        cfg=cfg,
    )
    if filtered_colors:
        write_ascii_ply_with_colors(filtered_path, filtered_points, filtered_colors)
    else:
        write_ascii_ply(filtered_path, filtered_points)

    spans = span_stats(filtered_points)
    report = {
        "status": "success",
        "device": metadata.get("sdk_device") or {},
        "capture": {
            "rgb_shape": list(rgb.shape),
            "raw_depth_shape": list(raw_depth.shape),
            "aligned_depth_shape": list(aligned_depth.shape),
            "d2c_mode": metadata.get("d2c_mode") or metadata.get("alignment_method"),
            "selected_sdk_alignment_api": metadata.get("selected_sdk_alignment_api"),
        },
        "coordinate_frame": "rgb_camera",
        "units": "meters",
        "origin": "rgb_camera_optical_center",
        "axis_convention": axis_convention_from_deprojection(),
        "intrinsics": {
            "rgb_intrinsics": rgb_intrinsics,
        },
        "depth_scale": {
            "saved_depth_units": metadata.get("saved_depth_units"),
            "sdk_reported_depth_scale": metadata.get("aligned_sdk_reported_depth_scale", metadata.get("sdk_reported_depth_scale")),
            "sdk_reported_depth_scale_units": metadata.get(
                "aligned_sdk_reported_depth_scale_units",
                metadata.get("sdk_reported_depth_scale_units"),
            ),
            "sdk_depth_scale_m_per_unit": depth_scale_to_meters,
            "scale_applied_during_cloud_generation": cloud_debug.get("scale_applied_during_cloud_generation"),
        },
        "roi": {
            "source": "app.config.DimScanConfig roi_* fields via segmentation.yoloe_segmenter._apply_object_cloud_roi",
            "frame": roi_debug.get("roi_coordinate_frame"),
            "units": roi_debug.get("roi_units"),
            "bounds": roi_debug.get("roi_bounds"),
            "points_before_roi": roi_debug.get("unfiltered_masked_point_count"),
            "points_after_roi": roi_debug.get("points_kept_by_roi"),
            "rejected_count": roi_debug.get("points_rejected_by_roi"),
            "points_before": roi_debug.get("unfiltered_masked_point_count"),
            "points_after": roi_debug.get("points_kept_by_roi"),
            "points_rejected": roi_debug.get("points_rejected_by_roi"),
        },
        "mask": mask_debug,
        "spans": spans,
        "point_count": spans["point_count"],
        "warnings": [],
        "cloud_debug": {
            "before_roi": cloud_debug,
            "roi": roi_debug,
        },
        "artifacts": {
            "rgb": str(rgb_path),
            "depth_raw": str(raw_depth_path),
            "depth_aligned_to_rgb": str(aligned_depth_path),
            "object_mask": str(mask_path),
            "object_mask_overlay_rgb": str(overlay_path),
            "depth_aligned_visualization": str(aligned_vis_path),
            "valid_aligned_depth_overlay_rgb": str(valid_overlay_path),
            "object_cloud_before_roi": str(before_roi_path),
            "object_cloud_roi_filtered": str(filtered_path),
            "axis_span_report_json": str(json_path),
            "axis_span_report_txt": str(text_path),
            "output_dir": str(run_dir),
        },
        "notes": [
            "This isolated tool reports raw X/Y/Z spans only.",
            "It does not assign semantic length, width, or height.",
            "Z camera-depth thickness is not interpreted as physical package width.",
        ],
    }
    write_json(json_path, report)
    write_text_report(text_path, report)
    if not args.save_unfiltered and before_roi_path.is_file():
        # Keep the required path by default; --save-unfiltered is retained for CLI compatibility.
        pass

    print(text_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except Exception as exc:
        failure_report(exc, LAST_RUN_DIR)
        raise SystemExit(1)
