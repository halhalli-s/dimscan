"""Optional YOLOE-style segmentation adapter for DimScan views."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import threading
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.config import DimScanConfig
from capture.pointcloud import write_ascii_ply, write_ascii_ply_with_colors
from pipeline.object_extraction import extract_geometry_primary_object_cloud
from pipeline.profiling import add_note, timed_stage
from segmentation.schema import make_segmentation_record
from utils.io import read_json_if_exists, write_json_atomic


logger = logging.getLogger(__name__)
_YOLOE_MODEL_CACHE_LOCK = threading.RLock()
_YOLOE_MODEL_CACHE: dict[tuple[str, tuple[str, ...]], Any] = {}
PROMPT_FREE_MODEL_MARKERS = ("-pf.", "-pf-", "_pf.")


def reset_yoloe_model_cache() -> None:
    """Clear the process-local YOLOE cache for tests."""
    with _YOLOE_MODEL_CACHE_LOCK:
        _YOLOE_MODEL_CACHE.clear()


def _cached_yoloe_model(model_source: str, prompts: list[str]) -> Any:
    resolved_source = resolve_yoloe_model_source(model_source)
    if _looks_prompt_free_model_source(resolved_source):
        raise ValueError(f"prompt_free_yoloe_checkpoint_rejected:{resolved_source}")
    cache_key = (resolved_source, tuple(prompts))
    with _YOLOE_MODEL_CACHE_LOCK:
        model = _YOLOE_MODEL_CACHE.get(cache_key)
        if model is None:
            from ultralytics import YOLOE

            model = YOLOE(resolved_source, verbose=False)
            setattr(model, "_dimscan_requested_model_source", model_source)
            setattr(model, "_dimscan_resolved_model_source", resolved_source)
            _YOLOE_MODEL_CACHE[cache_key] = model
        return model


SEGMENT_CLASS_HINTS = {
    "pot": ("plant pot", "flower pot", "planter", "pot", "container", "vase"),
    "leaf": ("leaves", "foliage", "leaf", "canopy", "greenery"),
    "table": ("table",),
    "object": ("potted plant", "whole potted plant", "plant", "object"),
}

EXACT_SEGMENT_LABELS = {
    "potted plant": "object",
    "whole potted plant": "object",
    "plant": "object",
    "object": "object",
    "plant pot": "pot",
    "flower pot": "pot",
    "planter": "pot",
    "pot": "pot",
    "container": "pot",
    "vase": "pot",
    "leaves": "leaf",
    "foliage": "leaf",
    "leaf": "leaf",
    "plant canopy": "leaf",
    "canopy": "leaf",
    "greenery": "leaf",
    "table": "table",
}

FOCUSED_CROP_PROMPTS = [
    "plant pot",
    "flower pot",
    "planter",
    "pot",
    "container",
    "vase",
    "leaves",
    "leaf",
    "foliage",
    "plant canopy",
    "greenery",
]

MIN_MASK_PIXEL_COUNT = 1_000
POT_MIN_CONFIDENCE = 0.40
POT_MIN_OBJECT_MASK_OVERLAP = 0.03
POT_MIN_OBJECT_BBOX_OVERLAP = 0.05
POT_MAX_CENTER_DISTANCE_RATIO = 0.35
TINY_EDGE_AREA_RATIO = 0.01
EDGE_MARGIN_RATIO = 0.03
TABLE_MIN_AREA_RATIO = 0.05
TABLE_MIN_CENTER_Y_RATIO = 0.45
OBJECT_CLOUD_ROI_MIN_POINTS = 30
AI1_OBJECT_UNION_MIN_PIXEL_COUNT = 50
AI1_OBJECT_UNION_MIN_MASK_OVERLAP = 0.01
AI1_OBJECT_UNION_MIN_BBOX_OVERLAP = 0.01
AI1_OBJECT_UNION_MAX_BBOX_GAP_RATIO = 0.07
AI1_OBJECT_UNION_MAX_CENTER_DISTANCE_RATIO = 0.55
AI1_OBJECT_UNION_SEGMENTS = {"object", "pot", "leaf"}


def configured_model_source(cfg: DimScanConfig) -> str | None:
    """Return a configured YOLOE model path or name."""
    env_path = os.environ.get("DIMSCAN_YOLOE_MODEL_PATH")
    env_name = os.environ.get("DIMSCAN_YOLOE_MODEL_NAME")
    source = env_path or cfg.yoloe_model_path or env_name or cfg.yoloe_model_name
    if source is None:
        return None
    source = str(source).strip()
    return source or None


def resolve_yoloe_model_source(model_source: str) -> str:
    """Resolve a configured YOLOE source to the exact local checkpoint when present."""
    source = str(model_source).strip()
    path = Path(source)
    if path.is_file():
        return str(path.resolve())
    if path.name == source:
        local_path = Path.cwd() / source
        if local_path.is_file():
            return str(local_path.resolve())
    return source


def _looks_prompt_free_model_source(model_source: str) -> bool:
    filename = Path(str(model_source)).name.lower()
    return any(marker in filename for marker in PROMPT_FREE_MODEL_MARKERS)


def _model_type_name(model: Any) -> str:
    inner = getattr(model, "model", None)
    target = inner if inner is not None else model
    return f"{target.__class__.__module__}.{target.__class__.__name__}"


def _model_trace(model: Any, requested_source: str) -> dict[str, Any]:
    resolved_source = str(
        getattr(model, "ckpt_path", None)
        or getattr(model, "model_name", None)
        or getattr(model, "_dimscan_resolved_model_source", None)
        or resolve_yoloe_model_source(requested_source)
    )
    return {
        "requested_source": requested_source,
        "resolved_source": resolved_source,
        "filename": Path(resolved_source).name,
        "model_type": _model_type_name(model),
        "prompt_free": _looks_prompt_free_model_source(resolved_source),
        "supports_set_classes": bool(hasattr(model, "set_classes")),
    }


def _without_runtime_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_runtime_keys(child)
            for key, child in value.items()
            if not str(key).startswith("_runtime_")
        }
    if isinstance(value, list):
        return [_without_runtime_keys(child) for child in value]
    return value


def _write_record(view_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    write_json_atomic(view_dir / "segmentation.json", _without_runtime_keys(record))
    return record


def _write_yoloe_debug(view_dir: Path, debug: dict[str, Any], *, debug_mode: bool = True) -> None:
    if not debug_mode:
        return
    debug_dir = view_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(debug_dir / "yoloe_predictions.json", debug)
    write_json_atomic(view_dir / "segmentation_debug.json", debug)


def _clear_stale_outputs(view_dir: Path, *, preserve_object_cloud: bool = False) -> None:
    for segment_name in ("object", "pot", "leaf", "table"):
        for path in (
            view_dir / f"{segment_name}_cloud.ply",
            view_dir / "masks" / f"{segment_name}_mask.png",
        ):
            if preserve_object_cloud and path == view_dir / "object_cloud.ply":
                continue
            if path.is_file():
                path.unlink()
    explicit_object_mask = view_dir / "object_mask.png"
    if explicit_object_mask.is_file():
        explicit_object_mask.unlink()
    rejected_pot_cloud = view_dir / "debug" / "pot_cloud_rejected.ply"
    if rejected_pot_cloud.is_file():
        rejected_pot_cloud.unlink()


def _skipped_record(
    cfg: DimScanConfig,
    view_dir: Path,
    *,
    reason: str,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return _write_record(
        view_dir,
        make_segmentation_record(
            status="skipped",
            model_backend=cfg.segmentation_backend,
            model_name=configured_model_source(cfg),
            prompts=list(cfg.yoloe_prompts),
            reason=reason,
            warnings=warnings or [reason],
        ),
    )


def _failed_record(
    cfg: DimScanConfig,
    view_dir: Path,
    *,
    reason: str,
    warnings: list[str] | None = None,
    confidence_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _write_record(
        view_dir,
        make_segmentation_record(
            status="failed",
            model_backend=cfg.segmentation_backend,
            model_name=configured_model_source(cfg),
            prompts=list(cfg.yoloe_prompts),
            reason=reason,
            warnings=warnings or [reason],
            confidence_summary=confidence_summary,
        ),
    )


def _empty_pot_quality() -> dict[str, Any]:
    return {
        "status": "missing",
        "usable_for_geometry": False,
        "usable_for_model": False,
        "trust_score": 0.0,
        "reason": "ai1_validation_not_available",
        "metrics": {},
    }


def _geometry_primary_record(
    cfg: DimScanConfig,
    view_dir: Path,
    *,
    model_source: str | None,
    object_extraction: dict[str, Any] | None,
    ai1_reason: str,
    warnings: list[str] | None = None,
    debug_mode: bool = True,
) -> dict[str, Any]:
    object_available = isinstance(object_extraction, dict) and object_extraction.get("final_point_count", 0) > 0
    artifacts = {}
    if object_available:
        object_cloud_path = view_dir / "object_cloud.ply"
        if object_cloud_path.is_file():
            artifacts["object_cloud"] = str(object_cloud_path)
        if debug_mode:
            artifacts["object_extraction_debug"] = str(view_dir / "debug" / "object_extraction_debug.json")
    segments = {
        "object": "ok" if object_available else "failed",
        "pot": "missing",
        "leaf": "missing",
        "table": "missing",
    }
    record_warnings = list(warnings or [])
    record_warnings.append(ai1_reason)
    if object_available:
        record_warnings.extend(["pot_segment_missing", "leaf_segment_missing", "table_segment_missing"])
    else:
        record_warnings.append("object_cloud_unavailable_geometry_blocked")
    record = make_segmentation_record(
        status="partial" if object_available else "failed",
        model_backend=cfg.segmentation_backend,
        model_name=model_source,
        prompts=list(cfg.yoloe_prompts),
        segment_statuses=segments,
        confidence_summary={},
        pot_quality=_empty_pot_quality(),
        warnings=record_warnings,
        artifacts=artifacts,
    )
    if object_available:
        runtime_points = object_extraction.get("_runtime_final_object_points")
        if runtime_points is not None:
            record["_runtime_final_object_points"] = runtime_points
    return _write_record(view_dir, record)


def _mask_array_from_result(result: Any, index: int) -> np.ndarray | None:
    masks = getattr(result, "masks", None)
    if masks is None:
        return None
    data = getattr(masks, "data", None)
    if data is None or index >= len(data):
        return None
    mask = data[index]
    if hasattr(mask, "cpu"):
        mask = mask.cpu()
    if hasattr(mask, "numpy"):
        mask = mask.numpy()
    return np.asarray(mask, dtype=float)


def _mask_from_result(result: Any, index: int) -> np.ndarray | None:
    mask = _mask_array_from_result(result, index)
    return None if mask is None else np.asarray(mask) > 0.5


def _result_names(result: Any) -> dict[int, str]:
    names = getattr(result, "names", {}) or {}
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def _boxes(result: Any) -> tuple[list[int], list[float]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return [], []
    cls_values = getattr(boxes, "cls", [])
    conf_values = getattr(boxes, "conf", [])
    if hasattr(cls_values, "cpu"):
        cls_values = cls_values.cpu()
    if hasattr(conf_values, "cpu"):
        conf_values = conf_values.cpu()
    if hasattr(cls_values, "numpy"):
        cls_values = cls_values.numpy()
    if hasattr(conf_values, "numpy"):
        conf_values = conf_values.numpy()
    return [int(value) for value in cls_values], [float(value) for value in conf_values]


def _box_coordinates(result: Any) -> list[list[float]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy_values = getattr(boxes, "xyxy", [])
    if hasattr(xyxy_values, "cpu"):
        xyxy_values = xyxy_values.cpu()
    if hasattr(xyxy_values, "numpy"):
        xyxy_values = xyxy_values.numpy()
    return [[float(coord) for coord in row] for row in xyxy_values]


def _box_area(box: list[float] | None) -> float:
    if not box:
        return 0.0
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _box_intersection_area(first: list[float] | None, second: list[float] | None) -> float:
    if not first or not second:
        return 0.0
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_center(box: list[float] | None) -> tuple[float, float] | None:
    if not box:
        return None
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _mask_bbox(mask: np.ndarray | None) -> list[float] | None:
    if mask is None:
        return None
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _mask_sha256(mask: np.ndarray) -> str:
    mask_uint8 = np.ascontiguousarray(np.asarray(mask, dtype=bool).astype(np.uint8))
    return hashlib.sha256(mask_uint8.tobytes()).hexdigest()


def _component_count(mask: np.ndarray) -> int:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 2 or not mask_bool.any():
        return 0
    visited = np.zeros(mask_bool.shape, dtype=bool)
    components = 0
    rows, cols = np.nonzero(mask_bool)
    for row, col in zip(rows.tolist(), cols.tolist()):
        if visited[row, col]:
            continue
        components += 1
        queue: deque[tuple[int, int]] = deque([(row, col)])
        visited[row, col] = True
        while queue:
            current_row, current_col = queue.popleft()
            for next_row, next_col in (
                (current_row - 1, current_col),
                (current_row + 1, current_col),
                (current_row, current_col - 1),
                (current_row, current_col + 1),
            ):
                if (
                    0 <= next_row < mask_bool.shape[0]
                    and 0 <= next_col < mask_bool.shape[1]
                    and mask_bool[next_row, next_col]
                    and not visited[next_row, next_col]
                ):
                    visited[next_row, next_col] = True
                    queue.append((next_row, next_col))
    return components


def _mask_metrics(mask: np.ndarray | None) -> dict[str, Any]:
    if mask is None:
        return {
            "shape": None,
            "foreground_pixel_count": 0,
            "bbox_xyxy": None,
            "connected_component_count": 0,
            "sha256": None,
        }
    mask_bool = np.asarray(mask, dtype=bool)
    return {
        "shape": [int(mask_bool.shape[0]), int(mask_bool.shape[1])] if mask_bool.ndim == 2 else list(mask_bool.shape),
        "foreground_pixel_count": int(np.count_nonzero(mask_bool)),
        "bbox_xyxy": _mask_bbox(mask_bool),
        "connected_component_count": _component_count(mask_bool),
        "sha256": _mask_sha256(mask_bool),
    }


def _bbox_gap(first: list[float] | None, second: list[float] | None) -> float | None:
    if not first or not second:
        return None
    dx = max(0.0, max(first[0] - second[2], second[0] - first[2]))
    dy = max(0.0, max(first[1] - second[3], second[1] - first[3]))
    return float((dx * dx + dy * dy) ** 0.5)


def _center_distance_ratio(first: list[float] | None, second: list[float] | None, image_diag: float) -> float | None:
    first_center = _box_center(first)
    second_center = _box_center(second)
    if first_center is None or second_center is None or image_diag <= 0:
        return None
    return float(
        ((first_center[0] - second_center[0]) ** 2 + (first_center[1] - second_center[1]) ** 2) ** 0.5
        / image_diag
    )


def _mask_overlap_ratio(mask: np.ndarray | None, reference: np.ndarray | None) -> float:
    if mask is None or reference is None:
        return 0.0
    candidate = np.asarray(mask, dtype=bool)
    ref = np.asarray(reference, dtype=bool)
    if candidate.shape != ref.shape:
        ref = _resize_mask(ref, (candidate.shape[1], candidate.shape[0]))
    candidate_count = int(np.count_nonzero(candidate))
    if candidate_count <= 0:
        return 0.0
    return float(np.count_nonzero(candidate & ref) / candidate_count)


def _mask_overlap_over_reference(mask: np.ndarray | None, reference: np.ndarray | None) -> float:
    if mask is None or reference is None:
        return 0.0
    candidate = np.asarray(mask, dtype=bool)
    ref = np.asarray(reference, dtype=bool)
    if candidate.shape != ref.shape:
        ref = _resize_mask(ref, (candidate.shape[1], candidate.shape[0]))
    reference_count = int(np.count_nonzero(ref))
    if reference_count <= 0:
        return 0.0
    return float(np.count_nonzero(candidate & ref) / reference_count)


def _image_shape_from_result(result: Any, masks: list[np.ndarray | None]) -> tuple[int, int]:
    orig_shape = getattr(result, "orig_shape", None)
    if isinstance(orig_shape, tuple) and len(orig_shape) >= 2:
        return int(orig_shape[0]), int(orig_shape[1])
    for mask in masks:
        if mask is not None:
            return int(mask.shape[0]), int(mask.shape[1])
    return 1, 1


def _segment_for_label(label: str) -> str | None:
    normalized = label.strip().lower()
    if normalized in EXACT_SEGMENT_LABELS:
        return EXACT_SEGMENT_LABELS[normalized]
    for segment_name, hints in SEGMENT_CLASS_HINTS.items():
        if any(hint in normalized for hint in hints):
            return segment_name
    return None


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")
    return np.asarray(image.resize(size, Image.Resampling.NEAREST)) > 0


def _restore_masks_to_rgb_shape(
    masks: dict[str, np.ndarray],
    *,
    rgb_size: tuple[int, int],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Restore model-resolution binary masks to the original RGB image grid."""
    rgb_width, rgb_height = rgb_size
    target_shape = (int(rgb_height), int(rgb_width))
    restored: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for segment_name, mask in masks.items():
        mask_bool = np.asarray(mask, dtype=bool)
        original_shape = (int(mask_bool.shape[0]), int(mask_bool.shape[1])) if mask_bool.ndim == 2 else None
        if mask_bool.ndim != 2:
            restored[segment_name] = mask_bool
            metadata[segment_name] = {
                "model_mask_shape": list(original_shape) if original_shape else None,
                "original_rgb_shape": [target_shape[0], target_shape[1]],
                "final_mask_shape": list(mask_bool.shape),
                "resize_method": None,
                "threshold": 0.5,
                "restoration_required": False,
                "foreground_pixel_count": int(np.count_nonzero(mask_bool)),
                "error": "mask_not_2d",
            }
            continue

        restoration_required = mask_bool.shape != target_shape
        final_mask = _resize_mask(mask_bool, (rgb_width, rgb_height)) if restoration_required else mask_bool
        restored[segment_name] = final_mask
        metadata[segment_name] = {
            "model_mask_shape": [int(mask_bool.shape[0]), int(mask_bool.shape[1])],
            "original_rgb_shape": [target_shape[0], target_shape[1]],
            "final_mask_shape": [int(final_mask.shape[0]), int(final_mask.shape[1])],
            "resize_method": "nearest" if restoration_required else "none",
            "threshold": 0.5,
            "restoration_required": restoration_required,
            "foreground_pixel_count": int(np.count_nonzero(final_mask)),
        }
    return restored, metadata


def _read_ascii_ply(path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]] | None]:
    lines = path.read_text(encoding="utf-8").splitlines()
    vertex_count = 0
    has_color = False
    header_end = None
    for index, line in enumerate(lines):
        if line.startswith("element vertex "):
            vertex_count = int(line.split()[-1])
        elif line == "property uchar red":
            has_color = True
        elif line == "end_header":
            header_end = index + 1
            break
    if header_end is None:
        raise ValueError(f"Invalid PLY header: {path}")

    points: list[tuple[float, float, float]] = []
    colors: list[tuple[int, int, int]] = []
    for line in lines[header_end : header_end + vertex_count]:
        parts = line.split()
        if len(parts) < 3:
            continue
        points.append((float(parts[0]), float(parts[1]), float(parts[2])))
        if has_color and len(parts) >= 6:
            colors.append((int(parts[3]), int(parts[4]), int(parts[5])))
    return points, colors if has_color and len(colors) == len(points) else None


def _valid_depth_mask(depth: np.ndarray) -> np.ndarray:
    return np.isfinite(depth) & (depth > 0)


def _intrinsic_value(intrinsics: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = intrinsics.get(name)
        if value is not None:
            return float(value)
    return None


def _project_valid_depth_indices_to_mask(
    *,
    mask: np.ndarray,
    depth: np.ndarray,
    capture_meta: dict[str, Any],
    point_count: int,
) -> tuple[list[int] | None, dict[str, Any]]:
    depth_intrinsics = capture_meta.get("depth_intrinsics")
    rgb_intrinsics = capture_meta.get("rgb_intrinsics")
    debug: dict[str, Any] = {"method": "intrinsics_depth_to_rgb_projection"}
    if not isinstance(depth_intrinsics, dict) or not isinstance(rgb_intrinsics, dict):
        debug["error"] = "missing_depth_or_rgb_intrinsics"
        return None, debug

    fx_d = _intrinsic_value(depth_intrinsics, ("fx", "focal_length_x"))
    fy_d = _intrinsic_value(depth_intrinsics, ("fy", "focal_length_y"))
    cx_d = _intrinsic_value(depth_intrinsics, ("cx", "principal_point_x", "ppx"))
    cy_d = _intrinsic_value(depth_intrinsics, ("cy", "principal_point_y", "ppy"))
    fx_rgb = _intrinsic_value(rgb_intrinsics, ("fx", "focal_length_x"))
    fy_rgb = _intrinsic_value(rgb_intrinsics, ("fy", "focal_length_y"))
    cx_rgb = _intrinsic_value(rgb_intrinsics, ("cx", "principal_point_x", "ppx"))
    cy_rgb = _intrinsic_value(rgb_intrinsics, ("cy", "principal_point_y", "ppy"))
    values = (fx_d, fy_d, cx_d, cy_d, fx_rgb, fy_rgb, cx_rgb, cy_rgb)
    if any(value is None or value == 0 for value in values):
        debug["error"] = "invalid_depth_or_rgb_intrinsics"
        return None, debug

    valid_mask = _valid_depth_mask(depth)
    valid_rows, valid_cols = np.where(valid_mask)
    if point_count != len(valid_rows):
        debug["error"] = "point_count_does_not_match_valid_depth_count"
        debug["valid_depth_count"] = int(len(valid_rows))
        debug["point_count"] = int(point_count)
        return None, debug

    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape != (int(capture_meta.get("rgb_height") or mask_bool.shape[0]), int(capture_meta.get("rgb_width") or mask_bool.shape[1])):
        mask_bool = _resize_mask(
            mask_bool,
            (
                int(capture_meta.get("rgb_width") or mask_bool.shape[1]),
                int(capture_meta.get("rgb_height") or mask_bool.shape[0]),
            ),
        )

    # Minimal alignment: project depth pixels into RGB using both intrinsics. No extrinsics
    # are available in current capture metadata, so this assumes SDK frame sync already
    # provides a shared camera frame and corrects only the depth/RGB intrinsics mismatch.
    z = depth[valid_rows, valid_cols].astype(float)
    x = (valid_cols.astype(float) - float(cx_d)) * z / float(fx_d)
    y = (valid_rows.astype(float) - float(cy_d)) * z / float(fy_d)
    rgb_cols = np.rint((float(fx_rgb) * x / z) + float(cx_rgb)).astype(int)
    rgb_rows = np.rint((float(fy_rgb) * y / z) + float(cy_rgb)).astype(int)
    in_bounds = (
        (rgb_rows >= 0)
        & (rgb_rows < mask_bool.shape[0])
        & (rgb_cols >= 0)
        & (rgb_cols < mask_bool.shape[1])
    )
    selected_valid_positions = np.where(in_bounds & mask_bool[np.clip(rgb_rows, 0, mask_bool.shape[0] - 1), np.clip(rgb_cols, 0, mask_bool.shape[1] - 1)])[0]
    debug.update(
        {
            "valid_depth_count": int(len(valid_rows)),
            "projected_in_bounds_count": int(np.count_nonzero(in_bounds)),
            "selected_point_count": int(len(selected_valid_positions)),
            "mask_shape_used": [int(mask_bool.shape[0]), int(mask_bool.shape[1])],
            "depth_shape": [int(depth.shape[0]), int(depth.shape[1])],
        }
    )
    return selected_valid_positions.astype(int).tolist(), debug


def _extrinsics_from_metadata(capture_meta: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    for key in ("depth_to_rgb_extrinsics", "depth_to_color_extrinsics", "extrinsics_depth_to_rgb"):
        value = capture_meta.get(key)
        if not isinstance(value, dict):
            continue
        rotation = value.get("rotation") or value.get("r")
        translation = value.get("translation") or value.get("t")
        if rotation is None or translation is None:
            continue
        rotation_array = np.asarray(rotation, dtype=float).reshape((3, 3))
        translation_array = np.asarray(translation, dtype=float).reshape((3,))
        return rotation_array, translation_array, key
    return None, None, None


def build_segment_cloud_from_depth(
    *,
    depth_raw: np.ndarray,
    rgb_mask: np.ndarray,
    depth_intrinsics: dict[str, Any],
    rgb_intrinsics: dict[str, Any],
    depth_to_rgb_extrinsics: tuple[np.ndarray | None, np.ndarray | None, str | None] | None = None,
) -> tuple[list[tuple[float, float, float]], np.ndarray, np.ndarray, dict[str, Any]]:
    """Build a segmented cloud directly from depth pixels.

    Input depth values are already scaled by capture/orbbec_camera.py when available.
    Output XYZ points intentionally use the same depth-camera coordinate frame and units
    as pipeline/scan_writer.py's cloud.ply generation, so downstream geometry keeps its
    existing unit normalization behavior.
    """
    depth = np.asarray(depth_raw, dtype=float)
    mask = np.asarray(rgb_mask, dtype=bool)
    fx_d = _intrinsic_value(depth_intrinsics, ("fx", "focal_length_x"))
    fy_d = _intrinsic_value(depth_intrinsics, ("fy", "focal_length_y"))
    cx_d = _intrinsic_value(depth_intrinsics, ("cx", "principal_point_x", "ppx"))
    cy_d = _intrinsic_value(depth_intrinsics, ("cy", "principal_point_y", "ppy"))
    fx_rgb = _intrinsic_value(rgb_intrinsics, ("fx", "focal_length_x"))
    fy_rgb = _intrinsic_value(rgb_intrinsics, ("fy", "focal_length_y"))
    cx_rgb = _intrinsic_value(rgb_intrinsics, ("cx", "principal_point_x", "ppx"))
    cy_rgb = _intrinsic_value(rgb_intrinsics, ("cy", "principal_point_y", "ppy"))
    values = (fx_d, fy_d, cx_d, cy_d, fx_rgb, fy_rgb, cx_rgb, cy_rgb)
    if any(value is None or value == 0 for value in values):
        return [], np.zeros(depth.shape, dtype=bool), np.zeros(mask.shape, dtype=bool), {
            "method": "direct_depth_deprojection",
            "error": "invalid_depth_or_rgb_intrinsics",
        }

    valid_depth = _valid_depth_mask(depth)
    rows, cols = np.where(valid_depth)
    if len(rows) == 0:
        return [], np.zeros(depth.shape, dtype=bool), np.zeros(mask.shape, dtype=bool), {
            "method": "direct_depth_deprojection",
            "valid_depth_pixel_count": 0,
            "selected_object_pixel_count": 0,
        }

    z = depth[rows, cols]
    x = (cols.astype(float) - float(cx_d)) * z / float(fx_d)
    y = (rows.astype(float) - float(cy_d)) * z / float(fy_d)
    depth_points = np.column_stack([x, y, z])

    rotation = translation = source = None
    if depth_to_rgb_extrinsics is not None:
        rotation, translation, source = depth_to_rgb_extrinsics
    if rotation is not None and translation is not None:
        rgb_points = (depth_points @ rotation.T) + translation
        extrinsics_available = True
        extrinsics_source = source
    else:
        rgb_points = depth_points
        extrinsics_available = False
        extrinsics_source = None

    z_rgb = rgb_points[:, 2]
    valid_projection = np.isfinite(z_rgb) & (z_rgb > 0)
    rgb_cols = np.zeros_like(cols)
    rgb_rows = np.zeros_like(rows)
    rgb_cols[valid_projection] = np.rint(
        (float(fx_rgb) * rgb_points[valid_projection, 0] / z_rgb[valid_projection]) + float(cx_rgb)
    ).astype(int)
    rgb_rows[valid_projection] = np.rint(
        (float(fy_rgb) * rgb_points[valid_projection, 1] / z_rgb[valid_projection]) + float(cy_rgb)
    ).astype(int)

    in_bounds = (
        valid_projection
        & (rgb_rows >= 0)
        & (rgb_rows < mask.shape[0])
        & (rgb_cols >= 0)
        & (rgb_cols < mask.shape[1])
    )
    projected_rgb_mask = np.zeros(mask.shape, dtype=bool)
    projected_rgb_mask[rgb_rows[in_bounds], rgb_cols[in_bounds]] = True
    selected = in_bounds & mask[rgb_rows.clip(0, mask.shape[0] - 1), rgb_cols.clip(0, mask.shape[1] - 1)]
    selected_depth_mask = np.zeros(depth.shape, dtype=bool)
    selected_depth_mask[rows[selected], cols[selected]] = True
    selected_points = [tuple(float(value) for value in point) for point in depth_points[selected]]
    debug = {
        "method": "direct_depth_deprojection",
        "valid_depth_pixel_count": int(len(rows)),
        "projected_in_bounds_count": int(np.count_nonzero(in_bounds)),
        "selected_object_pixel_count": int(np.count_nonzero(selected)),
        "final_object_point_count": int(len(selected_points)),
        "extrinsics_available": extrinsics_available,
        "extrinsics_source": extrinsics_source,
        "depth_aligned_to_rgb": depth.shape == mask.shape,
    }
    return selected_points, selected_depth_mask, projected_rgb_mask, debug


def select_cloud_points_from_pixel_map(
    *,
    cloud_points: list[tuple[float, float, float]],
    cloud_colors: list[tuple[int, int, int]] | None,
    pixel_indices: np.ndarray,
    depth_raw: np.ndarray,
    rgb_mask: np.ndarray,
    depth_intrinsics: dict[str, Any],
    rgb_intrinsics: dict[str, Any],
    depth_to_rgb_extrinsics: tuple[np.ndarray | None, np.ndarray | None, str | None] | None = None,
) -> tuple[
    list[tuple[float, float, float]],
    list[tuple[int, int, int]] | None,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Select rows from cloud.ply using an explicit point-row to depth-pixel map."""
    depth = np.asarray(depth_raw, dtype=float)
    mask = np.asarray(rgb_mask, dtype=bool)
    map_array = np.asarray(pixel_indices)
    debug: dict[str, Any] = {
        "method": "raw_cloud_with_explicit_pixel_map",
        "raw_cloud_point_count": int(len(cloud_points)),
        "pixel_map_count": int(len(map_array)) if map_array.ndim >= 1 else 0,
        "point_count_matches": False,
        "depth_shape": [int(depth.shape[0]), int(depth.shape[1])],
        "rgb_shape": [int(mask.shape[0]), int(mask.shape[1])],
    }
    if map_array.ndim != 2 or map_array.shape[1] != 2:
        debug["error"] = "invalid_pixel_map_shape"
        return [], None, np.zeros(depth.shape, dtype=bool), np.zeros(mask.shape, dtype=bool), debug
    if len(cloud_points) != len(map_array):
        debug["error"] = "raw_cloud_pixel_map_count_mismatch"
        return [], None, np.zeros(depth.shape, dtype=bool), np.zeros(mask.shape, dtype=bool), debug
    debug["point_count_matches"] = True

    rows = map_array[:, 0].astype(int)
    cols = map_array[:, 1].astype(int)
    in_depth_bounds = (rows >= 0) & (rows < depth.shape[0]) & (cols >= 0) & (cols < depth.shape[1])
    if not bool(np.all(in_depth_bounds)):
        debug["error"] = "pixel_map_coordinates_out_of_depth_bounds"
        debug["out_of_bounds_pixel_count"] = int(np.count_nonzero(~in_depth_bounds))
        return [], None, np.zeros(depth.shape, dtype=bool), np.zeros(mask.shape, dtype=bool), debug

    unique_pixels = np.unique(map_array.astype(np.int64), axis=0)
    duplicate_pixel_count = int(len(map_array) - len(unique_pixels))
    z = depth[rows, cols]
    valid_depth = np.isfinite(z) & (z > 0)

    fx_d = _intrinsic_value(depth_intrinsics, ("fx", "focal_length_x"))
    fy_d = _intrinsic_value(depth_intrinsics, ("fy", "focal_length_y"))
    cx_d = _intrinsic_value(depth_intrinsics, ("cx", "principal_point_x", "ppx"))
    cy_d = _intrinsic_value(depth_intrinsics, ("cy", "principal_point_y", "ppy"))
    fx_rgb = _intrinsic_value(rgb_intrinsics, ("fx", "focal_length_x"))
    fy_rgb = _intrinsic_value(rgb_intrinsics, ("fy", "focal_length_y"))
    cx_rgb = _intrinsic_value(rgb_intrinsics, ("cx", "principal_point_x", "ppx"))
    cy_rgb = _intrinsic_value(rgb_intrinsics, ("cy", "principal_point_y", "ppy"))
    values = (fx_d, fy_d, cx_d, cy_d, fx_rgb, fy_rgb, cx_rgb, cy_rgb)
    if any(value is None or value == 0 for value in values):
        debug["error"] = "invalid_depth_or_rgb_intrinsics"
        return [], None, np.zeros(depth.shape, dtype=bool), np.zeros(mask.shape, dtype=bool), debug

    x = (cols.astype(float) - float(cx_d)) * z / float(fx_d)
    y = (rows.astype(float) - float(cy_d)) * z / float(fy_d)
    projection_points = np.column_stack([x, y, z])

    rotation = translation = source = None
    if depth_to_rgb_extrinsics is not None:
        rotation, translation, source = depth_to_rgb_extrinsics
    if rotation is not None and translation is not None:
        rgb_points = (projection_points @ rotation.T) + translation
        extrinsics_available = True
        extrinsics_source = source
    else:
        rgb_points = projection_points
        extrinsics_available = False
        extrinsics_source = None

    z_rgb = rgb_points[:, 2]
    valid_projection = valid_depth & np.isfinite(z_rgb) & (z_rgb > 0)
    rgb_cols = np.zeros_like(cols)
    rgb_rows = np.zeros_like(rows)
    rgb_cols[valid_projection] = np.rint(
        (float(fx_rgb) * rgb_points[valid_projection, 0] / z_rgb[valid_projection]) + float(cx_rgb)
    ).astype(int)
    rgb_rows[valid_projection] = np.rint(
        (float(fy_rgb) * rgb_points[valid_projection, 1] / z_rgb[valid_projection]) + float(cy_rgb)
    ).astype(int)

    in_bounds = (
        valid_projection
        & (rgb_rows >= 0)
        & (rgb_rows < mask.shape[0])
        & (rgb_cols >= 0)
        & (rgb_cols < mask.shape[1])
    )
    projected_rgb_mask = np.zeros(mask.shape, dtype=bool)
    projected_rgb_mask[rgb_rows[in_bounds], rgb_cols[in_bounds]] = True
    selected = in_bounds & mask[rgb_rows.clip(0, mask.shape[0] - 1), rgb_cols.clip(0, mask.shape[1] - 1)]
    selected_depth_mask = np.zeros(depth.shape, dtype=bool)
    selected_depth_mask[rows[selected], cols[selected]] = True

    selected_indices = np.where(selected)[0].astype(int).tolist()
    selected_points = [cloud_points[index] for index in selected_indices]
    selected_colors = [cloud_colors[index] for index in selected_indices] if cloud_colors else None
    debug.update(
        {
            "valid_pixel_count": int(np.count_nonzero(valid_depth)),
            "duplicate_pixel_count": duplicate_pixel_count,
            "missing_or_invalid_depth_count": int(np.count_nonzero(~valid_depth)),
            "projected_in_bounds_count": int(np.count_nonzero(in_bounds)),
            "selected_object_point_count": int(len(selected_points)),
            "extrinsics_available": extrinsics_available,
            "extrinsics_source": extrinsics_source,
            "depth_aligned_to_rgb": depth.shape == mask.shape,
        }
    )
    return selected_points, selected_colors, selected_depth_mask, projected_rgb_mask, debug


def _indices_for_mask(
    mask: np.ndarray,
    depth: np.ndarray,
    point_count: int,
    capture_meta: dict[str, Any] | None = None,
) -> tuple[list[int] | None, dict[str, Any]]:
    capture_meta = capture_meta if isinstance(capture_meta, dict) else {}
    valid_flat = _valid_depth_mask(depth).reshape(-1)
    rgb_width = int(capture_meta.get("rgb_width") or 0)
    rgb_height = int(capture_meta.get("rgb_height") or 0)
    depth_shape = (int(depth.shape[0]), int(depth.shape[1]))
    mask_shape = (int(mask.shape[0]), int(mask.shape[1]))
    if (
        point_count == int(np.count_nonzero(valid_flat))
        and rgb_width > 0
        and rgb_height > 0
        and (rgb_height, rgb_width) != depth_shape
    ):
        projected, debug = _project_valid_depth_indices_to_mask(
            mask=mask,
            depth=depth,
            capture_meta=capture_meta,
            point_count=point_count,
        )
        if projected is not None:
            return projected, debug

    resized = _resize_mask(mask, (int(depth.shape[1]), int(depth.shape[0])))
    flat_mask = resized.reshape(-1)
    debug = {
        "method": "mask_resized_to_depth_grid",
        "raw_mask_shape": [int(mask_shape[0]), int(mask_shape[1])],
        "depth_shape": [int(depth_shape[0]), int(depth_shape[1])],
        "resized_mask_shape": [int(resized.shape[0]), int(resized.shape[1])],
    }
    if point_count == flat_mask.size:
        selected = np.where(flat_mask)[0].astype(int).tolist()
        debug["selected_point_count"] = len(selected)
        return selected, debug

    valid_mask_values = flat_mask[valid_flat]
    if point_count == valid_mask_values.size:
        selected = np.where(valid_mask_values)[0].astype(int).tolist()
        debug["selected_point_count"] = len(selected)
        debug["valid_depth_count"] = int(np.count_nonzero(valid_flat))
        return selected, debug
    debug["error"] = "mask_depth_point_count_unmapped"
    debug["point_count"] = int(point_count)
    debug["flat_mask_size"] = int(flat_mask.size)
    debug["valid_depth_count"] = int(np.count_nonzero(valid_flat))
    return None, debug


def _write_mask(path: Path, mask: np.ndarray, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_mask = _resize_mask(mask, size)
    Image.fromarray(np.asarray(output_mask, dtype=np.uint8) * 255, mode="L").save(path, format="PNG")


def _read_mask_png_as_bool(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def _write_probability_mask(path: Path, mask: np.ndarray | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mask is None:
        Image.fromarray(np.zeros((1, 1), dtype=np.uint8), mode="L").save(path, format="PNG")
        return
    mask_array = np.asarray(mask, dtype=float)
    if mask_array.size == 0:
        image_array = np.zeros((1, 1), dtype=np.uint8)
    else:
        clipped = np.clip(mask_array, 0.0, 1.0)
        image_array = np.asarray(clipped * 255, dtype=np.uint8)
    Image.fromarray(image_array, mode="L").save(path, format="PNG")


def _write_binary_mask(path: Path, mask: np.ndarray | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mask is None:
        Image.fromarray(np.zeros((1, 1), dtype=np.uint8), mode="L").save(path, format="PNG")
        return
    Image.fromarray(np.asarray(mask, dtype=bool).astype(np.uint8) * 255, mode="L").save(path, format="PNG")


def _raw_detection_records(result: Any) -> list[dict[str, Any]]:
    names = _result_names(result)
    cls_values, conf_values = _boxes(result)
    boxes = _box_coordinates(result)
    records = []
    for index, class_id in enumerate(cls_values):
        raw_mask = _mask_array_from_result(result, index)
        binary_mask = None if raw_mask is None else np.asarray(raw_mask) > 0.5
        label = names.get(class_id, str(class_id))
        records.append(
            {
                "index": index,
                "class_id": class_id,
                "label": label,
                "segment_candidate": _segment_for_label(label),
                "confidence": conf_values[index] if index < len(conf_values) else 0.0,
                "box_xyxy": boxes[index] if index < len(boxes) else None,
                "raw_mask_shape": [int(raw_mask.shape[0]), int(raw_mask.shape[1])] if raw_mask is not None and raw_mask.ndim == 2 else None,
                "raw_mask_min": float(np.nanmin(raw_mask)) if raw_mask is not None and raw_mask.size else None,
                "raw_mask_max": float(np.nanmax(raw_mask)) if raw_mask is not None and raw_mask.size else None,
                "threshold": 0.5,
                "binary_mask_metrics": _mask_metrics(binary_mask),
            }
        )
    return records


def _write_ai1_mask_debug_artifacts(
    *,
    cfg: DimScanConfig,
    view_dir: Path,
    rgb_path: Path,
    result: Any,
    prediction_debug: dict[str, Any],
    model_masks: dict[str, np.ndarray],
    restored_masks: dict[str, np.ndarray],
    mask_restore_debug: dict[str, Any],
    final_object_mask: np.ndarray | None,
) -> dict[str, Any]:
    debug_dir = view_dir / "debug" / "ai1"
    debug_dir.mkdir(parents=True, exist_ok=True)
    rgb_image = Image.open(rgb_path).convert("RGB")
    rgb_input_path = debug_dir / "rgb_input.png"
    rgb_image.save(rgb_input_path, format="PNG")

    detection_records = _raw_detection_records(result)
    raw_mask_paths: dict[str, str] = {}
    for record in detection_records:
        index = int(record["index"])
        raw_mask = _mask_array_from_result(result, index)
        raw_path = debug_dir / f"raw_mask_{index:02d}_model_resolution.png"
        _write_probability_mask(raw_path, raw_mask)
        raw_mask_paths[str(index)] = str(raw_path)

    model_object_mask = model_masks.get("object")
    raw_union_path = debug_dir / "raw_mask_union_model_resolution.png"
    _write_binary_mask(raw_union_path, model_object_mask)

    restored_mask = restored_masks.get("object")
    restored_before_threshold_path = debug_dir / "restored_mask_before_threshold.png"
    restored_binary_path = debug_dir / "restored_binary_mask.png"
    mask_before_morphology_path = debug_dir / "mask_before_morphology.png"
    mask_after_morphology_path = debug_dir / "mask_after_morphology.png"
    final_mask_path = debug_dir / "final_object_mask.png"
    overlay_path = debug_dir / "final_object_mask_overlay_rgb.png"
    _write_binary_mask(restored_before_threshold_path, restored_mask)
    _write_binary_mask(restored_binary_path, restored_mask)
    _write_binary_mask(mask_before_morphology_path, restored_mask)
    _write_binary_mask(mask_after_morphology_path, final_object_mask)
    _write_binary_mask(final_mask_path, final_object_mask)
    if final_object_mask is not None:
        _overlay_mask_on_image(rgb_image, final_object_mask, (255, 64, 64)).save(overlay_path, format="PNG")
    else:
        rgb_image.save(overlay_path, format="PNG")

    raw_summary_path = debug_dir / "raw_detection_summary.json"
    ai1_debug_path = debug_dir / "ai1_mask_debug.json"
    raw_summary = {
        "model_name": configured_model_source(cfg),
        "prompts": list(cfg.yoloe_prompts),
        "confidence_threshold": float(cfg.yoloe_confidence_threshold),
        "original_rgb_size": [int(rgb_image.size[0]), int(rgb_image.size[1])],
        "result_orig_shape": list(getattr(result, "orig_shape", []) or []),
        "detection_count": len(detection_records),
        "detections": detection_records,
        "raw_mask_artifacts": raw_mask_paths,
    }
    write_json_atomic(raw_summary_path, raw_summary)

    stage_metrics = {
        "raw_mask_union_model_resolution": _mask_metrics(model_object_mask),
        "restored_mask_before_threshold": _mask_metrics(restored_mask),
        "restored_binary_mask": _mask_metrics(restored_mask),
        "mask_before_morphology": _mask_metrics(restored_mask),
        "mask_after_morphology": _mask_metrics(final_object_mask),
        "final_object_mask": _mask_metrics(final_object_mask),
    }
    ai1_payload = {
        **raw_summary,
        "model_mask_shape": stage_metrics["raw_mask_union_model_resolution"]["shape"],
        "restored_mask_shape": stage_metrics["restored_binary_mask"]["shape"],
        "selected_indices": prediction_debug.get("assignment_decisions", {}),
        "masks_unioned": prediction_debug.get("object_union", {}).get("masks_unioned", False),
        "object_union": prediction_debug.get("object_union", {}),
        "thresholds": {
            "mask_threshold": 0.5,
            "min_mask_pixel_count": MIN_MASK_PIXEL_COUNT,
            "object_union_min_pixel_count": AI1_OBJECT_UNION_MIN_PIXEL_COUNT,
            "object_union_min_mask_overlap": AI1_OBJECT_UNION_MIN_MASK_OVERLAP,
            "object_union_min_bbox_overlap": AI1_OBJECT_UNION_MIN_BBOX_OVERLAP,
            "object_union_max_bbox_gap_ratio": AI1_OBJECT_UNION_MAX_BBOX_GAP_RATIO,
            "object_union_max_center_distance_ratio": AI1_OBJECT_UNION_MAX_CENTER_DISTANCE_RATIO,
        },
        "morphology_operations": [],
        "stage_metrics": stage_metrics,
        "mask_restoration": mask_restore_debug.get("object", {}),
        "pixel_loss_stage": (
            "none_after_selection_no_morphology"
            if stage_metrics["final_object_mask"]["foreground_pixel_count"]
            == stage_metrics["restored_binary_mask"]["foreground_pixel_count"]
            else "post_selection_restoration_or_morphology"
        ),
        "known_previous_loss_stage": "single_best_detection_selection_before_object_union",
        "artifacts": {
            "rgb_input": str(rgb_input_path),
            "raw_detection_summary": str(raw_summary_path),
            "raw_mask_union_model_resolution": str(raw_union_path),
            "restored_mask_before_threshold": str(restored_before_threshold_path),
            "restored_binary_mask": str(restored_binary_path),
            "mask_before_morphology": str(mask_before_morphology_path),
            "mask_after_morphology": str(mask_after_morphology_path),
            "final_object_mask": str(final_mask_path),
            "final_object_mask_overlay_rgb": str(overlay_path),
            "ai1_mask_debug": str(ai1_debug_path),
            **{f"raw_mask_{int(index):02d}_model_resolution": path for index, path in raw_mask_paths.items()},
        },
    }
    write_json_atomic(ai1_debug_path, ai1_payload)
    return ai1_payload


def _depth_preview_array(depth: np.ndarray) -> np.ndarray:
    preview = np.zeros(depth.shape, dtype=np.uint8)
    valid = _valid_depth_mask(depth)
    if valid.any():
        values = depth[valid]
        low = float(np.percentile(values, 2))
        high = float(np.percentile(values, 98))
        if high <= low:
            low = float(values.min())
            high = float(values.max())
        if high > low:
            normalized = (np.clip(depth, low, high) - low) / (high - low)
            preview[valid] = np.asarray(normalized[valid] * 255, dtype=np.uint8)
    return preview


def _depth_scale_to_meters(capture_meta: dict[str, Any]) -> tuple[float, str]:
    saved_units = str(capture_meta.get("saved_depth_units") or "meters").lower()
    scale_applied = capture_meta.get("depth_scale_applied_to_saved_depth")
    object_scale = capture_meta.get("sdk_depth_scale_m_per_unit")
    if object_scale is None:
        object_scale = capture_meta.get("object_cloud_depth_scale", 1.0)
    try:
        scale = float(object_scale)
    except (TypeError, ValueError) as exc:
        raise ValueError("aligned_depth_object_scale_invalid") from exc
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("aligned_depth_object_scale_invalid")
    if saved_units == "meters":
        if scale_applied is not True:
            raise ValueError("aligned_depth_scale_state_invalid: meters must already have SDK scale applied")
        return scale, saved_units
    if saved_units in {"millimeter", "millimeters", "mm"}:
        if scale_applied is not False:
            raise ValueError("aligned_depth_scale_state_invalid: millimeters must be converted once during cloud generation")
        return scale * 0.001, saved_units
    if saved_units in {"raw_sdk_unit", "raw_sdk_units"}:
        if scale_applied is not False:
            raise ValueError("aligned_depth_scale_state_invalid: raw SDK units must be converted once during cloud generation")
        if scale >= 0.1:
            raise ValueError("aligned_depth_scale_state_invalid: raw SDK scale must be meters_per_unit, not millimeters_per_unit")
        return scale, saved_units
    raise ValueError(f"aligned_depth_units_unsupported:{saved_units}")


def build_object_cloud_from_aligned_depth(
    *,
    object_mask: np.ndarray,
    depth_aligned_to_rgb: np.ndarray,
    rgb_intrinsics: dict[str, Any],
    depth_scale_to_meters: float = 1.0,
    saved_depth_units: str = "meters",
    rgb_image: np.ndarray | None = None,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]] | None, dict[str, Any]]:
    """Build the final object cloud from an RGB-grid mask and SDK D2C aligned depth."""
    mask = np.asarray(object_mask, dtype=bool)
    depth = np.asarray(depth_aligned_to_rgb, dtype=float)
    if mask.ndim != 2:
        raise ValueError(f"object_mask_must_be_2d:{mask.shape}")
    if depth.ndim != 2:
        raise ValueError(f"aligned_depth_must_be_2d:{depth.shape}")
    if mask.shape != depth.shape:
        raise ValueError(f"object_mask_aligned_depth_shape_mismatch:mask={mask.shape}:depth={depth.shape}")

    fx = _intrinsic_value(rgb_intrinsics, ("fx", "focal_length_x"))
    fy = _intrinsic_value(rgb_intrinsics, ("fy", "focal_length_y"))
    cx = _intrinsic_value(rgb_intrinsics, ("cx", "principal_point_x", "ppx"))
    cy = _intrinsic_value(rgb_intrinsics, ("cy", "principal_point_y", "ppy"))
    if fx is None or fy is None or fx == 0 or fy == 0 or cx is None or cy is None:
        raise ValueError("rgb_intrinsics_invalid_for_aligned_depth_object_cloud")
    if depth_scale_to_meters <= 0 or not np.isfinite(depth_scale_to_meters):
        raise ValueError("aligned_depth_scale_to_meters_invalid")

    masked_pixels = int(np.count_nonzero(mask))
    valid = mask & np.isfinite(depth) & (depth > 0)
    rows, cols = np.nonzero(valid)
    z = depth[rows, cols].astype(float) * float(depth_scale_to_meters)
    x = (cols.astype(float) - float(cx)) * z / float(fx)
    y = (rows.astype(float) - float(cy)) * z / float(fy)
    points_array = np.column_stack((x, y, z)) if len(z) else np.empty((0, 3), dtype=float)
    points = [tuple(float(value) for value in point) for point in points_array]

    colors: list[tuple[int, int, int]] | None = None
    if rgb_image is not None:
        rgb = np.asarray(rgb_image, dtype=np.uint8)
        if rgb.shape[:2] != mask.shape:
            raise ValueError(f"rgb_image_aligned_depth_shape_mismatch:rgb={rgb.shape[:2]}:depth={depth.shape}")
        colors = [(int(r), int(g), int(b)) for r, g, b in rgb[rows, cols, :3]]

    xyz_min = points_array.min(axis=0).tolist() if len(points_array) else None
    xyz_max = points_array.max(axis=0).tolist() if len(points_array) else None
    xyz_spans = (points_array.max(axis=0) - points_array.min(axis=0)).tolist() if len(points_array) else None
    selected_depth_raw = depth[rows, cols].astype(float)
    if z.size:
        median_z = float(np.median(z))
        max_z = float(z.max())
        if not (0.1 <= median_z <= 10.0):
            raise ValueError(f"converted_depth_median_implausible:{median_z}")
        if max_z > 20.0:
            raise ValueError(f"converted_depth_max_implausible:{max_z}")
        if saved_depth_units in {"raw_sdk_unit", "raw_sdk_units"} and float(depth_scale_to_meters) >= 0.1:
            raise ValueError("raw_sdk_depth_scale_would_create_meter_scale_from_millimeter_values")
    debug = {
        "generation_method": "aligned_depth_rgb_mask",
        "mask_shape": [int(mask.shape[0]), int(mask.shape[1])],
        "aligned_depth_shape": [int(depth.shape[0]), int(depth.shape[1])],
        "selected_pixel_count": masked_pixels,
        "valid_selected_depth_count": int(len(points)),
        "invalid_masked_depth_count": int(masked_pixels - len(points)),
        "rgb_intrinsics_used": dict(rgb_intrinsics),
        "saved_depth_units": saved_depth_units,
        "sdk_reported_depth_scale": None,
        "sdk_reported_depth_scale_units": None,
        "sdk_depth_scale_m_per_unit": float(depth_scale_to_meters),
        "scale_applied_during_cloud_generation": float(depth_scale_to_meters),
        "valid_depth_min_before_conversion": float(selected_depth_raw.min()) if selected_depth_raw.size else None,
        "valid_depth_median_before_conversion": float(np.median(selected_depth_raw)) if selected_depth_raw.size else None,
        "valid_depth_max_before_conversion": float(selected_depth_raw.max()) if selected_depth_raw.size else None,
        "valid_z_min_m": float(z.min()) if z.size else None,
        "valid_z_median_m": float(np.median(z)) if z.size else None,
        "valid_z_max_m": float(z.max()) if z.size else None,
        "xyz_min": [float(value) for value in xyz_min] if xyz_min is not None else None,
        "xyz_max": [float(value) for value in xyz_max] if xyz_max is not None else None,
        "xyz_spans": [float(value) for value in xyz_spans] if xyz_spans is not None else None,
        "final_point_count": int(len(points)),
        "old_fallback_used": False,
        "no_mask_resize": True,
    }
    return points, colors, debug


def _overlay_mask_on_image(base: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    base_rgb = np.asarray(base.convert("RGB"), dtype=np.uint8).copy()
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape != base_rgb.shape[:2]:
        mask_bool = _resize_mask(mask_bool, base.size)
    overlay = np.asarray(color, dtype=np.uint8)
    base_rgb[mask_bool] = np.asarray(base_rgb[mask_bool] * 0.45 + overlay * 0.55, dtype=np.uint8)
    return Image.fromarray(base_rgb, mode="RGB")


def _point_span(points: list[tuple[float, float, float]]) -> dict[str, Any]:
    if not points:
        return {"point_count": 0, "span": None}
    array = np.asarray(points, dtype=float)
    return {
        "point_count": int(len(points)),
        "span": {
            "x": float(array[:, 0].max() - array[:, 0].min()),
            "y": float(array[:, 1].max() - array[:, 1].min()),
            "z": float(array[:, 2].max() - array[:, 2].min()),
        },
    }


def _xyz_stats(points: np.ndarray) -> dict[str, Any]:
    if points.size == 0:
        return {
            "xyz_min": None,
            "xyz_max": None,
            "xyz_spans": None,
        }
    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    return {
        "xyz_min": [float(value) for value in xyz_min],
        "xyz_max": [float(value) for value in xyz_max],
        "xyz_spans": [float(value) for value in (xyz_max - xyz_min)],
    }


def _object_roi_bounds(cfg: DimScanConfig) -> dict[str, Any]:
    margin = float(cfg.roi_margin_m or 0.0)
    return {
        "min_x_m": -float(cfg.roi_half_width_m) - margin,
        "max_x_m": float(cfg.roi_half_width_m) + margin,
        "min_y_m": None,
        "max_y_m": None,
        "min_z_m": float(cfg.roi_min_depth_m) - margin,
        "max_z_m": float(cfg.roi_max_depth_m) + margin,
        "margin_m": cfg.roi_margin_m,
    }


def _apply_object_cloud_roi(
    *,
    points: list[tuple[float, float, float]],
    colors: list[tuple[int, int, int]] | None,
    cfg: DimScanConfig,
) -> tuple[
    list[tuple[float, float, float]],
    list[tuple[int, int, int]] | None,
    list[tuple[float, float, float]],
    list[tuple[int, int, int]] | None,
    dict[str, Any],
]:
    roi_frame = str(getattr(cfg, "roi_coordinate_frame", "") or "")
    roi_units = str(getattr(cfg, "roi_units", "") or "")
    if not bool(getattr(cfg, "roi_enabled", False)):
        raise ValueError("object_cloud_roi_disabled")
    if roi_frame != "rgb_camera":
        raise ValueError(f"object_cloud_roi_coordinate_frame_mismatch:{roi_frame or 'missing'}")
    if roi_units != "meters":
        raise ValueError(f"object_cloud_roi_units_mismatch:{roi_units or 'missing'}")

    point_array = np.asarray(points, dtype=float).reshape((-1, 3))
    bounds = _object_roi_bounds(cfg)
    pre_stats = _xyz_stats(point_array)
    if point_array.size == 0:
        raise ValueError("object_cloud_roi_empty_input")

    inside = (
        (point_array[:, 0] >= float(bounds["min_x_m"]))
        & (point_array[:, 0] <= float(bounds["max_x_m"]))
        & (point_array[:, 2] >= float(bounds["min_z_m"]))
        & (point_array[:, 2] <= float(bounds["max_z_m"]))
    )
    kept_array = point_array[inside]
    rejected_array = point_array[~inside]
    if len(kept_array) < OBJECT_CLOUD_ROI_MIN_POINTS:
        raise ValueError(f"object_cloud_roi_too_few_points:{len(kept_array)}")

    color_array = np.asarray(colors, dtype=np.uint8).reshape((-1, 3)) if colors is not None else None
    kept_colors = None
    rejected_colors = None
    if color_array is not None:
        if len(color_array) != len(point_array):
            raise ValueError("object_cloud_roi_color_count_mismatch")
        kept_colors = [(int(r), int(g), int(b)) for r, g, b in color_array[inside]]
        rejected_colors = [(int(r), int(g), int(b)) for r, g, b in color_array[~inside]]

    kept_points = [tuple(float(value) for value in point) for point in kept_array]
    rejected_points = [tuple(float(value) for value in point) for point in rejected_array]
    post_stats = _xyz_stats(kept_array)
    kept_percentage = float(len(kept_points) / len(point_array) * 100.0)
    debug = {
        "object_cloud_coordinate_frame": "rgb_camera",
        "roi_coordinate_frame": roi_frame,
        "roi_units": roi_units,
        "roi_bounds": bounds,
        "unfiltered_masked_point_count": int(len(point_array)),
        "points_kept_by_roi": int(len(kept_points)),
        "points_rejected_by_roi": int(len(rejected_points)),
        "kept_percentage": kept_percentage,
        "pre_roi_xyz_min": pre_stats["xyz_min"],
        "pre_roi_xyz_max": pre_stats["xyz_max"],
        "pre_roi_xyz_spans": pre_stats["xyz_spans"],
        "post_roi_xyz_min": post_stats["xyz_min"],
        "post_roi_xyz_max": post_stats["xyz_max"],
        "post_roi_xyz_spans": post_stats["xyz_spans"],
        "geometry_uses_roi_filtered_object_cloud": True,
    }
    return kept_points, kept_colors, rejected_points, rejected_colors, debug


def _write_object_cloud_debug_artifacts(
    *,
    view_dir: Path,
    raw_mask: np.ndarray,
    restored_mask: np.ndarray,
    selected_depth_mask: np.ndarray,
    projected_rgb_mask: np.ndarray,
    depth: np.ndarray,
    rgb_size: tuple[int, int],
    final_points: list[tuple[float, float, float]],
    pixel_map_points: list[tuple[float, float, float]],
    direct_points: list[tuple[float, float, float]],
    old_points: list[tuple[float, float, float]] | None,
    pixel_map_debug: dict[str, Any],
    direct_debug: dict[str, Any],
    old_projection_debug: dict[str, Any] | None,
    capture_meta: dict[str, Any],
    final_cloud_path: Path,
    pixel_map_cloud_path: Path,
    direct_cloud_path: Path,
    old_cloud_path: Path | None,
    pixel_map_path: Path,
    fallback_used: bool,
) -> dict[str, Any]:
    debug_dir = view_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = view_dir / "rgb.png"
    rgb_image = Image.open(rgb_path).convert("RGB")

    raw_path = debug_dir / "object_mask_raw.png"
    restored_path = debug_dir / "object_mask_restored.png"
    overlay_rgb_path = debug_dir / "object_mask_overlay_rgb.png"
    overlay_depth_path = debug_dir / "object_mask_overlay_depth.png"
    projected_rgb_path = debug_dir / "projected_depth_pixels_on_rgb.png"
    selected_depth_path = debug_dir / "object_selected_depth_pixels.png"
    debug_json_path = debug_dir / "object_cloud_debug.json"

    Image.fromarray(np.asarray(raw_mask, dtype=np.uint8) * 255, mode="L").save(raw_path, format="PNG")
    Image.fromarray(np.asarray(restored_mask, dtype=np.uint8) * 255, mode="L").save(restored_path, format="PNG")
    _overlay_mask_on_image(rgb_image, restored_mask, (255, 64, 64)).save(overlay_rgb_path, format="PNG")
    _overlay_mask_on_image(rgb_image, projected_rgb_mask, (64, 144, 255)).save(projected_rgb_path, format="PNG")
    depth_preview = Image.fromarray(_depth_preview_array(depth), mode="L").convert("RGB")
    _overlay_mask_on_image(depth_preview, selected_depth_mask, (255, 64, 64)).save(overlay_depth_path, format="PNG")
    _overlay_mask_on_image(depth_preview, selected_depth_mask, (255, 64, 64)).save(selected_depth_path, format="PNG")

    valid_depth_count = int(np.count_nonzero(_valid_depth_mask(depth)))
    old_cloud_point_count = len(old_points) if old_points is not None else None
    depth_aligned_to_rgb = bool(
        pixel_map_debug.get("depth_aligned_to_rgb")
        or capture_meta.get("depth_aligned_to_rgb")
        or capture_meta.get("aligned_depth_to_rgb")
    )
    debug_payload = {
        "rgb_shape": [int(rgb_size[1]), int(rgb_size[0])],
        "depth_shape": [int(depth.shape[0]), int(depth.shape[1])],
        "rgb_frame_shape": [int(rgb_size[1]), int(rgb_size[0])],
        "depth_frame_shape": [int(depth.shape[0]), int(depth.shape[1])],
        "model_input_shape": None,
        "raw_mask_shape": [int(raw_mask.shape[0]), int(raw_mask.shape[1])],
        "restored_mask_shape": [int(restored_mask.shape[0]), int(restored_mask.shape[1])],
        "depth_scale": capture_meta.get("depth_scale"),
        "depth_intrinsics": capture_meta.get("depth_intrinsics"),
        "rgb_intrinsics": capture_meta.get("rgb_intrinsics"),
        "extrinsics_available": bool(pixel_map_debug.get("extrinsics_available")),
        "extrinsics_source": pixel_map_debug.get("extrinsics_source"),
        "depth_aligned_to_rgb": depth_aligned_to_rgb,
        "valid_depth_pixel_count": int(pixel_map_debug.get("valid_pixel_count") or valid_depth_count),
        "raw_cloud_point_count": int(pixel_map_debug.get("raw_cloud_point_count") or 0),
        "pixel_map_count": int(pixel_map_debug.get("pixel_map_count") or 0),
        "point_count_matches": bool(pixel_map_debug.get("point_count_matches")),
        "valid_pixel_count": int(pixel_map_debug.get("valid_pixel_count") or 0),
        "duplicate_pixel_count": int(pixel_map_debug.get("duplicate_pixel_count") or 0),
        "missing_or_invalid_depth_count": int(pixel_map_debug.get("missing_or_invalid_depth_count") or 0),
        "projected_in_bounds_count": int(pixel_map_debug.get("projected_in_bounds_count") or 0),
        "selected_object_point_count": int(pixel_map_debug.get("selected_object_point_count") or len(pixel_map_points)),
        "selected_object_pixel_count": int(pixel_map_debug.get("selected_object_point_count") or len(pixel_map_points)),
        "final_object_point_count": int(len(final_points)),
        "primary_generation_method": "raw_cloud_with_explicit_pixel_map",
        "fallback_used": bool(fallback_used),
        "old_cloud_point_count": old_cloud_point_count,
        "pixel_map_path": str(pixel_map_path),
        "final_object_cloud_path": str(final_cloud_path),
        "aligned_depth_shape": [int(depth.shape[0]), int(depth.shape[1])],
        "object_mask_pixel_count": int(np.count_nonzero(restored_mask)),
        "valid_masked_depth_pixel_count": int(pixel_map_debug.get("selected_object_point_count") or len(final_points)),
        "invalid_or_zero_depth_count": int(depth.size - valid_depth_count),
        "point_count_after_deprojection": int(len(pixel_map_points)),
        "point_count_after_roi_or_crop": None,
        "point_count_after_table_removal": None,
        "point_count_after_any_filter": int(len(final_points)),
        "final_object_cloud_point_count": int(len(final_points)),
        "fallback_logic_used": bool(fallback_used),
        "clustering_used": False,
        "source_path_or_branch": "yoloe_object_mask_raw_cloud_with_explicit_pixel_map",
        "projection": pixel_map_debug,
        "pixel_map_projection": pixel_map_debug,
        "direct_depth_projection": direct_debug,
        "old_index_masking_projection": old_projection_debug,
        "capture_alignment": {
            "rgb_width": capture_meta.get("rgb_width"),
            "rgb_height": capture_meta.get("rgb_height"),
            "depth_width": capture_meta.get("depth_width"),
            "depth_height": capture_meta.get("depth_height"),
            "rgb_intrinsics": capture_meta.get("rgb_intrinsics"),
            "depth_intrinsics": capture_meta.get("depth_intrinsics"),
        },
        "cloud_spans": {
            "from_pixel_map": _point_span(pixel_map_points),
            "direct_from_depth": _point_span(direct_points),
            "old_index_masking": _point_span(old_points or []),
            "final": _point_span(final_points),
        },
        "artifacts": {
            "object_mask_raw": str(raw_path),
            "object_mask_restored": str(restored_path),
            "object_mask_overlay_rgb": str(overlay_rgb_path),
            "projected_depth_pixels_on_rgb": str(projected_rgb_path),
            "object_mask_overlay_depth": str(overlay_depth_path),
            "object_selected_depth_pixels": str(selected_depth_path),
            "object_cloud_from_pixel_map": str(pixel_map_cloud_path),
            "object_cloud_direct_from_depth": str(direct_cloud_path),
            "object_cloud_old_index_masking": str(old_cloud_path) if old_cloud_path is not None else None,
            "final_object_cloud": str(final_cloud_path),
        },
    }
    write_json_atomic(debug_json_path, debug_payload)
    return debug_payload


def _write_aligned_object_cloud_debug_artifacts(
    *,
    view_dir: Path,
    object_mask: np.ndarray,
    aligned_depth: np.ndarray,
    rgb_image: Image.Image,
    cloud_debug: dict[str, Any],
    capture_meta: dict[str, Any],
    final_cloud_path: Path,
) -> dict[str, Any]:
    debug_dir = view_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    depth_preview = Image.fromarray(_depth_preview_array(aligned_depth), mode="L").convert("RGB")
    valid_depth_mask = _valid_depth_mask(aligned_depth)
    masked_valid_depth = np.asarray(object_mask, dtype=bool) & valid_depth_mask

    aligned_preview_path = debug_dir / "depth_aligned_visualization.png"
    valid_overlay_path = debug_dir / "valid_aligned_depth_overlay_rgb.png"
    object_overlay_path = debug_dir / "object_mask_overlay_rgb.png"
    masked_overlay_path = debug_dir / "object_masked_valid_aligned_depth.png"
    debug_json_path = debug_dir / "object_cloud_debug.json"

    depth_preview.save(aligned_preview_path, format="PNG")
    _overlay_mask_on_image(rgb_image, valid_depth_mask, (64, 200, 80)).save(valid_overlay_path, format="PNG")
    _overlay_mask_on_image(rgb_image, object_mask, (255, 64, 64)).save(object_overlay_path, format="PNG")
    _overlay_mask_on_image(depth_preview, masked_valid_depth, (255, 64, 64)).save(masked_overlay_path, format="PNG")

    debug_payload = {
        **cloud_debug,
        "rgb_shape": [int(rgb_image.size[1]), int(rgb_image.size[0])],
        "d2c_mode": capture_meta.get("d2c_mode") or capture_meta.get("alignment_method"),
        "selected_sdk_alignment_api": capture_meta.get("selected_sdk_alignment_api"),
        "hardware_alignment_used": capture_meta.get("hardware_alignment_used"),
        "final_object_cloud_path": str(final_cloud_path),
        "explicit_confirmation_no_old_fallback_used": True,
        "old_paths_disabled_for_final_output": [
            "select_cloud_points_from_pixel_map",
            "cloud_pixel_indices",
            "raw_cloud_row_selection",
            "build_segment_cloud_from_depth_raw_depth",
            "_indices_for_mask",
            "mask_resize_onto_raw_depth",
            "object_cloud_old_index_masking",
            "intrinsics_only_depth_to_rgb_projection",
        ],
        "artifacts": {
            "aligned_depth_visualization": str(aligned_preview_path),
            "valid_aligned_depth_overlay_rgb": str(valid_overlay_path),
            "object_mask_overlay_rgb": str(object_overlay_path),
            "object_masked_valid_aligned_depth": str(masked_overlay_path),
            "object_cloud_masked_before_roi": str(debug_dir / "object_cloud_masked_before_roi.ply"),
            "object_cloud_rejected_by_roi": str(debug_dir / "object_cloud_rejected_by_roi.ply"),
            "final_object_cloud": str(final_cloud_path),
        },
    }
    write_json_atomic(debug_json_path, debug_payload)
    return debug_payload


def _write_object_cloud_failure_debug(
    *,
    view_dir: Path,
    reason: str,
    mask: np.ndarray | None,
    depth_aligned_to_rgb: np.ndarray | None,
    capture_meta: dict[str, Any],
    debug_mode: bool = True,
) -> None:
    if not debug_mode:
        return
    debug_dir = view_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generation_method": "aligned_depth_rgb_mask",
        "status": "failed",
        "error": reason,
        "mask_shape": [int(mask.shape[0]), int(mask.shape[1])] if mask is not None and mask.ndim == 2 else None,
        "aligned_depth_shape": (
            [int(depth_aligned_to_rgb.shape[0]), int(depth_aligned_to_rgb.shape[1])]
            if depth_aligned_to_rgb is not None and depth_aligned_to_rgb.ndim == 2
            else None
        ),
        "saved_depth_units": capture_meta.get("saved_depth_units"),
        "sdk_reported_depth_scale": capture_meta.get("aligned_sdk_reported_depth_scale", capture_meta.get("sdk_reported_depth_scale")),
        "sdk_reported_depth_scale_units": capture_meta.get(
            "aligned_sdk_reported_depth_scale_units",
            capture_meta.get("sdk_reported_depth_scale_units"),
        ),
        "sdk_depth_scale_m_per_unit": capture_meta.get("sdk_depth_scale_m_per_unit"),
        "sdk_depth_scale": capture_meta.get("sdk_depth_scale"),
        "object_cloud_depth_scale": capture_meta.get("object_cloud_depth_scale"),
        "d2c_mode": capture_meta.get("d2c_mode") or capture_meta.get("alignment_method"),
        "selected_sdk_alignment_api": capture_meta.get("selected_sdk_alignment_api"),
        "explicit_confirmation_no_old_fallback_used": True,
        "old_fallback_used": False,
    }
    write_json_atomic(debug_dir / "object_cloud_debug.json", payload)


def _write_segment_cloud(
    *,
    view_dir: Path,
    segment_name: str,
    mask: np.ndarray,
    depth: np.ndarray,
    cfg: DimScanConfig,
    depth_aligned_to_rgb: np.ndarray | None = None,
    warnings: list[str],
    capture_meta: dict[str, Any] | None = None,
    rgb_size: tuple[int, int] | None = None,
    debug: dict[str, Any] | None = None,
    debug_mode: bool = True,
) -> str | None:
    cloud_path = view_dir / "cloud.ply"
    if not cloud_path.is_file() and segment_name != "object":
        warnings.append(f"{segment_name}_cloud_projection_missing_cloud")
        return None

    if segment_name == "object" and rgb_size is not None:
        capture_meta = capture_meta if isinstance(capture_meta, dict) else {}
        rgb_intrinsics = capture_meta.get("rgb_intrinsics")
        if not isinstance(rgb_intrinsics, dict):
            warnings.append("object_cloud_aligned_depth_missing_rgb_intrinsics")
            _write_object_cloud_failure_debug(
                view_dir=view_dir,
                reason="object_cloud_aligned_depth_missing_rgb_intrinsics",
                mask=np.asarray(mask),
                depth_aligned_to_rgb=None,
                capture_meta=capture_meta,
                debug_mode=debug_mode,
            )
            return None
        if depth_aligned_to_rgb is None:
            warnings.append("object_cloud_aligned_depth_missing")
            _write_object_cloud_failure_debug(
                view_dir=view_dir,
                reason="object_cloud_aligned_depth_missing",
                mask=np.asarray(mask),
                depth_aligned_to_rgb=None,
                capture_meta=capture_meta,
                debug_mode=debug_mode,
            )
            return None

        object_mask = np.asarray(mask, dtype=bool)
        aligned_depth = np.asarray(depth_aligned_to_rgb, dtype=float)
        if object_mask.ndim != 2 or aligned_depth.ndim != 2 or object_mask.shape != aligned_depth.shape:
            reason = (
                "object_cloud_aligned_depth_shape_mismatch:"
                f"mask={tuple(object_mask.shape)}:aligned_depth={tuple(aligned_depth.shape)}"
            )
            warnings.append(reason)
            _write_object_cloud_failure_debug(
                view_dir=view_dir,
                reason=reason,
                mask=object_mask,
                depth_aligned_to_rgb=aligned_depth,
                capture_meta=capture_meta,
                debug_mode=debug_mode,
            )
            return None
        output_path = view_dir / "object_cloud.ply"
        debug_dir = view_dir / "debug"
        pre_roi_cloud_path = debug_dir / "object_cloud_masked_before_roi.ply"
        rejected_roi_cloud_path = debug_dir / "object_cloud_rejected_by_roi.ply"
        try:
            depth_scale_to_meters, saved_units = _depth_scale_to_meters(capture_meta)
            rgb_image = Image.open(view_dir / "rgb.png").convert("RGB")
            rgb_array = np.asarray(rgb_image, dtype=np.uint8)
            masked_points, masked_colors, cloud_debug = build_object_cloud_from_aligned_depth(
                object_mask=object_mask,
                depth_aligned_to_rgb=aligned_depth,
                rgb_intrinsics=rgb_intrinsics,
                depth_scale_to_meters=depth_scale_to_meters,
                saved_depth_units=saved_units,
                rgb_image=rgb_array,
            )
            if debug_mode:
                if masked_colors:
                    write_ascii_ply_with_colors(pre_roi_cloud_path, masked_points, masked_colors)
                else:
                    write_ascii_ply(pre_roi_cloud_path, masked_points)
            final_points, final_colors, rejected_points, rejected_colors, roi_debug = _apply_object_cloud_roi(
                points=masked_points,
                colors=masked_colors,
                cfg=cfg,
            )
            if debug_mode and rejected_points:
                if rejected_colors:
                    write_ascii_ply_with_colors(rejected_roi_cloud_path, rejected_points, rejected_colors)
                else:
                    write_ascii_ply(rejected_roi_cloud_path, rejected_points)
            cloud_debug["sdk_reported_depth_scale"] = capture_meta.get(
                "aligned_sdk_reported_depth_scale",
                capture_meta.get("sdk_reported_depth_scale"),
            )
            cloud_debug["sdk_reported_depth_scale_units"] = capture_meta.get(
                "aligned_sdk_reported_depth_scale_units",
                capture_meta.get("sdk_reported_depth_scale_units"),
            )
            cloud_debug["sdk_depth_scale_m_per_unit"] = depth_scale_to_meters
            cloud_debug.update(roi_debug)
            cloud_debug["generation_method"] = "aligned_depth_rgb_mask_roi_filtered"
            cloud_debug["masked_before_roi_point_count"] = len(masked_points)
            cloud_debug["final_point_count"] = len(final_points)
            cloud_debug["xyz_min"] = roi_debug["post_roi_xyz_min"]
            cloud_debug["xyz_max"] = roi_debug["post_roi_xyz_max"]
            cloud_debug["xyz_spans"] = roi_debug["post_roi_xyz_spans"]
            cloud_debug["final_cloud_is_roi_filtered"] = True
            cloud_debug["object_mask_sha256_consumed_by_cloud"] = _mask_sha256(object_mask)
        except Exception as exc:
            reason = f"object_cloud_aligned_depth_failed:{exc}"
            warnings.append(reason)
            _write_object_cloud_failure_debug(
                view_dir=view_dir,
                reason=reason,
                mask=object_mask,
                depth_aligned_to_rgb=aligned_depth,
                capture_meta=capture_meta,
                debug_mode=debug_mode,
            )
            return None

        if not final_points:
            warnings.append("object_cloud_aligned_depth_empty")
            _write_object_cloud_failure_debug(
                view_dir=view_dir,
                reason="object_cloud_aligned_depth_empty",
                mask=object_mask,
                depth_aligned_to_rgb=aligned_depth,
                capture_meta=capture_meta,
                debug_mode=debug_mode,
            )
            return None

        if final_colors:
            write_ascii_ply_with_colors(output_path, final_points, final_colors)
        else:
            write_ascii_ply(output_path, final_points)
        if debug_mode:
            object_debug = _write_aligned_object_cloud_debug_artifacts(
                view_dir=view_dir,
                object_mask=object_mask,
                aligned_depth=aligned_depth,
                rgb_image=rgb_image,
                cloud_debug=cloud_debug,
                capture_meta=capture_meta,
                final_cloud_path=output_path,
            )
            if debug is not None:
                debug["object_cloud_debug"] = object_debug
        elif debug is not None:
            debug["object_cloud_debug"] = {
                **cloud_debug,
                "final_object_cloud_path": str(output_path),
                "object_cloud_coordinate_frame": "rgb_camera",
                "point_cloud_frame": "rgb_camera",
                "point_cloud_units": "meters",
            }
        spans = cloud_debug.get("xyz_spans")
        logger.info(
            "Object cloud generated method=aligned_depth_rgb_mask_roi_filtered points=%s xyz_spans=%s",
            len(final_points),
            spans,
        )
        return str(output_path)

    if not cloud_path.is_file():
        warnings.append(f"{segment_name}_cloud_projection_missing_cloud")
        return None

    try:
        points, colors = _read_ascii_ply(cloud_path)
        indices, projection_debug = _indices_for_mask(mask, depth, len(points), capture_meta=capture_meta)
    except Exception as exc:
        warnings.append(f"{segment_name}_cloud_projection_failed:{exc}")
        return None

    if indices is None:
        warnings.append(f"{segment_name}_cloud_projection_unmapped")
        return None
    if not indices:
        warnings.append(f"{segment_name}_cloud_projection_empty")
        return None

    selected_points = [points[index] for index in indices]
    output_path = view_dir / f"{segment_name}_cloud.ply"
    if colors:
        selected_colors = [colors[index] for index in indices]
        write_ascii_ply_with_colors(output_path, selected_points, selected_colors)
    else:
        write_ascii_ply(output_path, selected_points)
    return str(output_path)


def _cloud_metrics_for_mask(
    *,
    view_dir: Path,
    mask: np.ndarray,
    depth: np.ndarray,
    output_path: Path | None = None,
    capture_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "point_count_in_pot_cloud": 0,
        "depth_bbox_width_in": None,
        "depth_bbox_height_in": None,
    }
    cloud_path = view_dir / "cloud.ply"
    if not cloud_path.is_file():
        return metrics
    try:
        points, colors = _read_ascii_ply(cloud_path)
        indices, projection_debug = _indices_for_mask(mask, depth, len(points), capture_meta=capture_meta)
        metrics["projection"] = projection_debug
    except Exception as exc:
        metrics["projection_error"] = str(exc)
        return metrics
    if not indices:
        return metrics

    selected_points = [points[index] for index in indices]
    metrics["point_count_in_pot_cloud"] = len(selected_points)
    point_array = np.asarray(selected_points, dtype=float)
    if point_array.size:
        extents_raw = point_array.max(axis=0) - point_array.min(axis=0)
        if float(np.max(extents_raw)) > 20.0:
            point_array = point_array / 1000.0
        extents = point_array.max(axis=0) - point_array.min(axis=0)
        xy = sorted([float(extents[0]), float(extents[1])], reverse=True)
        metrics["depth_bbox_width_in"] = xy[0] * 39.37007874015748
        metrics["depth_bbox_height_in"] = float(extents[2]) * 39.37007874015748

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if colors:
            selected_colors = [colors[index] for index in indices]
            write_ascii_ply_with_colors(output_path, selected_points, selected_colors)
        else:
            write_ascii_ply(output_path, selected_points)
    return metrics


def _pot_quality_record(
    *,
    cfg: DimScanConfig,
    pot_mask: np.ndarray | None,
    object_mask: np.ndarray | None,
    pot_confidence: float | None,
    depth_metrics: dict[str, Any] | None = None,
    is_fallback: bool = False,
    image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    depth_metrics = depth_metrics or {}
    metrics: dict[str, Any] = {
        "pot_confidence": pot_confidence,
        "pot_mask_pixel_count": int(np.count_nonzero(pot_mask)) if pot_mask is not None else 0,
        "object_mask_pixel_count": int(np.count_nonzero(object_mask)) if object_mask is not None else 0,
        "pot_to_object_area_ratio": 0.0,
        "pot_object_overlap_ratio": 0.0,
        "pot_bbox_xyxy": None,
        "object_bbox_xyxy": None,
        "pot_bbox_center_x": None,
        "object_bbox_center_x": None,
        "center_alignment_score": 0.0,
        "lower_object_position_score": 0.0,
        "mask_fill_ratio": 0.0,
        "visible_pot_width_px": None,
        "visible_pot_height_px": None,
        "point_count_in_pot_cloud": depth_metrics.get("point_count_in_pot_cloud", 0),
        "depth_bbox_width_in": depth_metrics.get("depth_bbox_width_in"),
        "depth_bbox_height_in": depth_metrics.get("depth_bbox_height_in"),
    }
    if depth_metrics.get("projection_error"):
        metrics["projection_error"] = depth_metrics["projection_error"]

    if pot_mask is None:
        return {
            "status": "missing",
            "usable_for_geometry": False,
            "usable_for_model": False,
            "trust_score": 0.0,
            "reason": "pot_mask_missing",
            "metrics": metrics,
        }

    pot_box = _mask_bbox(pot_mask)
    object_box = _mask_bbox(object_mask)
    metrics["pot_bbox_xyxy"] = pot_box
    metrics["object_bbox_xyxy"] = object_box
    if pot_box:
        metrics["visible_pot_width_px"] = pot_box[2] - pot_box[0]
        metrics["visible_pot_height_px"] = pot_box[3] - pot_box[1]
        pot_center = _box_center(pot_box)
        if pot_center:
            metrics["pot_bbox_center_x"] = pot_center[0]
    if object_box:
        object_center = _box_center(object_box)
        if object_center:
            metrics["object_bbox_center_x"] = object_center[0]

    object_pixels = metrics["object_mask_pixel_count"]
    pot_pixels = metrics["pot_mask_pixel_count"]
    if object_pixels > 0:
        metrics["pot_to_object_area_ratio"] = float(pot_pixels / object_pixels)
    metrics["pot_object_overlap_ratio"] = _mask_overlap_ratio(pot_mask, object_mask)

    pot_area = _box_area(pot_box)
    if pot_area > 0:
        metrics["mask_fill_ratio"] = float(pot_pixels / pot_area)

    if pot_box and object_box:
        object_width = max(1.0, object_box[2] - object_box[0])
        object_height = max(1.0, object_box[3] - object_box[1])
        pot_center = _box_center(pot_box)
        object_center = _box_center(object_box)
        if pot_center and object_center:
            x_delta = abs(pot_center[0] - object_center[0])
            metrics["center_alignment_score"] = max(0.0, 1.0 - (x_delta / (object_width * 0.5)))
            metrics["lower_object_position_score"] = max(0.0, min(1.0, (pot_center[1] - object_box[1]) / object_height))

    thresholds = {
        "pot_min_confidence": float(cfg.pot_min_confidence),
        "pot_min_object_overlap_ratio": float(cfg.pot_min_object_overlap_ratio),
        "pot_min_area_ratio_to_object": float(cfg.pot_min_area_ratio_to_object),
        "pot_max_area_ratio_to_object": float(cfg.pot_max_area_ratio_to_object),
        "pot_min_lower_position_score": float(cfg.pot_min_lower_position_score),
        "pot_min_center_alignment_score": float(cfg.pot_min_center_alignment_score),
        "pot_min_mask_fill_ratio": float(cfg.pot_min_mask_fill_ratio),
        "pot_min_point_count": int(cfg.pot_min_point_count),
        "pot_min_depth_width_in": float(cfg.pot_min_depth_width_in),
        "pot_max_depth_width_in": float(cfg.pot_max_depth_width_in),
        "pot_min_depth_height_in": float(cfg.pot_min_depth_height_in),
        "pot_max_depth_height_in": float(cfg.pot_max_depth_height_in),
    }
    metrics["thresholds"] = thresholds

    rejection_reasons: list[str] = []
    if is_fallback:
        rejection_reasons.append("fallback_pot_mask_debug_only")
    if pot_confidence is not None and pot_confidence < cfg.pot_min_confidence:
        rejection_reasons.append("pot_confidence_below_threshold")
    if metrics["pot_to_object_area_ratio"] < cfg.pot_min_area_ratio_to_object:
        rejection_reasons.append("pot_area_too_small_relative_to_object")
    if metrics["pot_to_object_area_ratio"] > cfg.pot_max_area_ratio_to_object:
        rejection_reasons.append("pot_area_too_large_relative_to_object")
    if object_mask is not None and metrics["pot_object_overlap_ratio"] < cfg.pot_min_object_overlap_ratio:
        rejection_reasons.append("pot_object_overlap_below_threshold")
    if metrics["lower_object_position_score"] < cfg.pot_min_lower_position_score:
        rejection_reasons.append("pot_not_low_enough_in_object")
    if metrics["center_alignment_score"] < cfg.pot_min_center_alignment_score:
        rejection_reasons.append("pot_badly_off_center")
    if metrics["mask_fill_ratio"] < cfg.pot_min_mask_fill_ratio:
        rejection_reasons.append("pot_mask_fill_ratio_too_low")
    if metrics["point_count_in_pot_cloud"] < cfg.pot_min_point_count:
        rejection_reasons.append("pot_depth_point_count_too_low")

    width_in = metrics.get("depth_bbox_width_in")
    height_in = metrics.get("depth_bbox_height_in")
    if width_in is not None and not (cfg.pot_min_depth_width_in <= width_in <= cfg.pot_max_depth_width_in):
        rejection_reasons.append("pot_depth_width_physically_implausible")
    if height_in is not None and not (cfg.pot_min_depth_height_in <= height_in <= cfg.pot_max_depth_height_in):
        rejection_reasons.append("pot_depth_height_physically_implausible")

    if image_size and pot_box and object_box:
        image_width, _ = image_size
        touches_edge = pot_box[0] <= image_width * EDGE_MARGIN_RATIO or pot_box[2] >= image_width * (1 - EDGE_MARGIN_RATIO)
        metrics["touches_image_edge"] = touches_edge
        if touches_edge and metrics["pot_object_overlap_ratio"] < cfg.pot_min_object_overlap_ratio:
            rejection_reasons.append("pot_at_image_edge_without_strong_object_overlap")

    confidence_score = min(1.0, (pot_confidence or 0.0) / max(0.01, cfg.pot_min_confidence))
    trust_inputs = [
        confidence_score,
        min(1.0, metrics["pot_object_overlap_ratio"] / max(0.01, cfg.pot_min_object_overlap_ratio)),
        min(1.0, metrics["center_alignment_score"] / max(0.01, cfg.pot_min_center_alignment_score)),
        min(1.0, metrics["lower_object_position_score"] / max(0.01, cfg.pot_min_lower_position_score)),
        min(1.0, metrics["mask_fill_ratio"] / max(0.01, cfg.pot_min_mask_fill_ratio)),
    ]
    trust_score = float(max(0.0, min(1.0, sum(trust_inputs) / len(trust_inputs))))

    if not rejection_reasons:
        return {
            "status": "trusted",
            "usable_for_geometry": True,
            "usable_for_model": True,
            "trust_score": trust_score,
            "reason": "pot_mask_passed_quality_checks",
            "metrics": metrics,
        }

    if is_fallback:
        status = "fallback"
    elif "pot_confidence_below_threshold" in rejection_reasons:
        status = "rejected_low_confidence"
    elif "pot_object_overlap_below_threshold" in rejection_reasons:
        status = "rejected_low_overlap"
    elif "pot_area_too_small_relative_to_object" in rejection_reasons:
        status = "rejected_too_small"
    elif "pot_area_too_large_relative_to_object" in rejection_reasons:
        status = "rejected_too_large"
    elif any(reason.startswith("pot_depth") for reason in rejection_reasons):
        status = "rejected_partial"
    else:
        status = "rejected_bad_location"
    return {
        "status": status,
        "usable_for_geometry": False,
        "usable_for_model": False,
        "trust_score": trust_score,
        "reason": ";".join(rejection_reasons),
        "metrics": metrics,
    }


def _selected_box(prediction_debug: dict[str, Any], segment_name: str) -> list[float] | None:
    index = prediction_debug.get("assignment_decisions", {}).get(segment_name)
    if index is None:
        return None
    detections = prediction_debug.get("detections", [])
    for detection in detections:
        if int(detection.get("index", -1)) == int(index):
            box = detection.get("box_xyxy")
            return [float(value) for value in box] if box else None
    return None


def _crop_box_around(box: list[float], image_size: tuple[int, int], margin_ratio: float = 0.14) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = box
    margin = max(x2 - x1, y2 - y1) * margin_ratio
    left = max(0, int(np.floor(x1 - margin)))
    top = max(0, int(np.floor(y1 - margin)))
    right = min(width, int(np.ceil(x2 + margin)))
    bottom = min(height, int(np.ceil(y2 + margin)))
    if right <= left or bottom <= top:
        return 0, 0, width, height
    return left, top, right, bottom


def _mask_to_full_image(mask: np.ndarray, crop_box: tuple[int, int, int, int], full_size: tuple[int, int]) -> np.ndarray:
    full_width, full_height = full_size
    left, top, right, bottom = crop_box
    crop_width = max(1, right - left)
    crop_height = max(1, bottom - top)
    resized = _resize_mask(mask, (crop_width, crop_height))
    full_mask = np.zeros((full_height, full_width), dtype=bool)
    full_mask[top:bottom, left:right] = resized[: bottom - top, : right - left]
    return full_mask


def _box_to_full_image(box: list[float] | None, crop_box: tuple[int, int, int, int]) -> list[float] | None:
    if not box:
        return None
    left, top, _, _ = crop_box
    return [float(box[0] + left), float(box[1] + top), float(box[2] + left), float(box[3] + top)]


def _crop_prediction_debug_to_full(
    prediction_debug: dict[str, Any],
    crop_box: tuple[int, int, int, int],
    object_mask: np.ndarray,
) -> dict[str, Any]:
    detections = []
    for detection in prediction_debug.get("detections", []):
        updated = dict(detection)
        updated["box_xyxy_crop"] = detection.get("box_xyxy")
        updated["box_xyxy"] = _box_to_full_image(detection.get("box_xyxy"), crop_box)
        detections.append(updated)
    updated_debug = dict(prediction_debug)
    updated_debug["boxes_crop"] = prediction_debug.get("boxes", [])
    updated_debug["boxes"] = [_box_to_full_image(box, crop_box) for box in prediction_debug.get("boxes", [])]
    updated_debug["detections"] = detections
    updated_debug["object_overlap_scores"] = {}
    for detection in detections:
        segment_name = detection.get("segment_candidate")
        if segment_name not in {"pot", "leaf"}:
            continue
        index = detection.get("index")
        updated_debug["object_overlap_scores"][str(index)] = detection.get("spatial_scores", {}).get(
            "mask_overlap_ratio_with_object"
        )
    updated_debug["object_mask_pixels"] = int(np.count_nonzero(object_mask))
    return updated_debug


def _upper_object_fallback_mask(object_mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(object_mask)
    if len(xs) < MIN_MASK_PIXEL_COUNT:
        return None
    y_min = int(ys.min())
    y_max = int(ys.max())
    cutoff = y_min + int((y_max - y_min + 1) * 0.62)
    fallback = object_mask.copy()
    fallback[cutoff + 1 :, :] = False
    return fallback if int(np.count_nonzero(fallback)) >= MIN_MASK_PIXEL_COUNT else None


def _lower_central_object_fallback_mask(object_mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(object_mask)
    if len(xs) < MIN_MASK_PIXEL_COUNT:
        return None
    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    width = max(1, x_max - x_min + 1)
    center = (x_min + x_max) / 2.0
    left = int(max(0, center - width * 0.34))
    right = int(min(object_mask.shape[1], center + width * 0.34))
    top = y_min + int((y_max - y_min + 1) * 0.48)
    fallback = np.zeros_like(object_mask, dtype=bool)
    fallback[top : y_max + 1, left:right] = object_mask[top : y_max + 1, left:right]
    return fallback if int(np.count_nonzero(fallback)) >= max(200, MIN_MASK_PIXEL_COUNT // 3) else None


def _object_union_candidate_decision(
    detection: dict[str, Any],
    *,
    primary: dict[str, Any],
    union_mask: np.ndarray,
    image_diag: float,
) -> tuple[bool, dict[str, Any]]:
    mask = detection.get("mask")
    segment_name = detection.get("segment_candidate")
    if segment_name not in AI1_OBJECT_UNION_SEGMENTS:
        return False, {"accepted": False, "reason": "segment_not_part_of_full_object_union"}
    if mask is None or not detection.get("has_mask"):
        return False, {"accepted": False, "reason": "missing_mask"}
    mask_pixel_count = int(detection.get("mask_pixel_count") or 0)
    if mask_pixel_count < AI1_OBJECT_UNION_MIN_PIXEL_COUNT:
        return False, {
            "accepted": False,
            "reason": "mask_pixel_count_below_object_union_minimum",
            "mask_pixel_count": mask_pixel_count,
        }
    rejection_reasons = set(detection.get("rejection_reasons") or [])
    blocking_reasons = rejection_reasons - {"mask_pixel_count_below_minimum"}
    if blocking_reasons and segment_name == "object":
        return False, {"accepted": False, "reason": "candidate_has_blocking_rejection", "rejection_reasons": sorted(blocking_reasons)}

    mask_overlap_ratio = _mask_overlap_ratio(mask, union_mask)
    bbox_overlap_ratio = _box_intersection_area(detection.get("box_xyxy"), primary.get("box_xyxy")) / max(
        1.0,
        _box_area(detection.get("box_xyxy")),
    )
    gap_px = _bbox_gap(detection.get("box_xyxy"), _mask_bbox(union_mask))
    gap_ratio = None if gap_px is None or image_diag <= 0 else float(gap_px / image_diag)
    center_distance_ratio = _center_distance_ratio(detection.get("box_xyxy"), primary.get("box_xyxy"), image_diag)
    spatially_consistent = (
        mask_overlap_ratio >= AI1_OBJECT_UNION_MIN_MASK_OVERLAP
        or bbox_overlap_ratio >= AI1_OBJECT_UNION_MIN_BBOX_OVERLAP
        or (gap_ratio is not None and gap_ratio <= AI1_OBJECT_UNION_MAX_BBOX_GAP_RATIO)
    )
    decision = {
        "accepted": bool(spatially_consistent),
        "segment_candidate": segment_name,
        "mask_pixel_count": mask_pixel_count,
        "mask_overlap_ratio_with_object_union": float(mask_overlap_ratio),
        "bbox_overlap_ratio_with_primary_object": float(bbox_overlap_ratio),
        "bbox_gap_px": gap_px,
        "bbox_gap_ratio": gap_ratio,
        "center_distance_ratio_with_primary_object": center_distance_ratio,
        "thresholds": {
            "min_pixel_count": AI1_OBJECT_UNION_MIN_PIXEL_COUNT,
            "min_mask_overlap": AI1_OBJECT_UNION_MIN_MASK_OVERLAP,
            "min_bbox_overlap": AI1_OBJECT_UNION_MIN_BBOX_OVERLAP,
            "max_bbox_gap_ratio": AI1_OBJECT_UNION_MAX_BBOX_GAP_RATIO,
            "max_center_distance_ratio": AI1_OBJECT_UNION_MAX_CENTER_DISTANCE_RATIO,
            "allowed_segments": sorted(AI1_OBJECT_UNION_SEGMENTS),
        },
    }
    if not spatially_consistent:
        decision["reason"] = "not_spatially_consistent_with_primary_object"
    return bool(spatially_consistent), decision


def _union_spatially_consistent_object_parts(
    *,
    detections: list[dict[str, Any]],
    primary_object: dict[str, Any],
    image_diag: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    union_mask = np.asarray(primary_object["mask"], dtype=bool).copy()
    unioned_indices = [int(primary_object["index"])]
    candidate_decisions: dict[str, Any] = {}
    for detection in detections:
        index = int(detection.get("index", -1))
        if index == int(primary_object["index"]):
            candidate_decisions[str(index)] = {"accepted": True, "reason": "primary_object_anchor"}
            continue
        accepted, decision = _object_union_candidate_decision(
            detection,
            primary=primary_object,
            union_mask=union_mask,
            image_diag=image_diag,
        )
        candidate_decisions[str(index)] = decision
        detection["object_union_decision"] = decision
        if accepted:
            union_mask |= np.asarray(detection["mask"], dtype=bool)
            unioned_indices.append(index)
            detection["included_in_object_union"] = True
    debug = {
        "primary_index": int(primary_object["index"]),
        "unioned_indices": unioned_indices,
        "masks_unioned": len(unioned_indices) > 1,
        "candidate_decisions": candidate_decisions,
        "primary_metrics": _mask_metrics(primary_object["mask"]),
        "union_metrics": _mask_metrics(union_mask),
        "pixels_added_by_union": int(np.count_nonzero(union_mask) - np.count_nonzero(primary_object["mask"])),
    }
    return union_mask, debug


def _select_masks(result: Any) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
    names = _result_names(result)
    cls_values, conf_values = _boxes(result)
    boxes = _box_coordinates(result)
    detections: list[dict[str, Any]] = []
    for index, class_id in enumerate(cls_values):
        label = names.get(class_id, str(class_id))
        segment_name = _segment_for_label(label)
        confidence = conf_values[index] if index < len(conf_values) else 0.0
        mask = _mask_from_result(result, index)
        has_mask = mask is not None
        mask_pixel_count = int(np.count_nonzero(mask)) if has_mask else 0
        box = boxes[index] if index < len(boxes) else None
        if segment_name is None:
            detections.append(
                {
                    "index": index,
                    "class_id": class_id,
                    "label": label,
                    "confidence": confidence,
                    "box_xyxy": box,
                    "has_mask": has_mask,
                    "mask_pixel_count": mask_pixel_count,
                    "mask": mask,
                    "segment_candidate": None,
                    "score": 0.0,
                    "spatial_scores": {},
                    "assignment_decision": None,
                    "rejection_reasons": ["label_not_mapped_to_dimscan_segment"],
                }
            )
            continue
        rejection_reasons: list[str] = []
        if mask is None:
            rejection_reasons.append("missing_mask")
        if mask_pixel_count < MIN_MASK_PIXEL_COUNT:
            rejection_reasons.append("mask_pixel_count_below_minimum")
        detections.append(
            {
                "index": index,
                "class_id": class_id,
                "label": label,
                "confidence": confidence,
                "box_xyxy": box,
                "has_mask": has_mask,
                "mask_pixel_count": mask_pixel_count,
                "mask": mask,
                "segment_candidate": segment_name,
                "score": 0.0,
                "spatial_scores": {},
                "assignment_decision": None,
                "rejection_reasons": rejection_reasons,
            }
        )

    image_height, image_width = _image_shape_from_result(result, [detection.get("mask") for detection in detections])
    image_area = max(1.0, float(image_height * image_width))
    image_diag = max(1.0, float((image_height**2 + image_width**2) ** 0.5))

    selected: dict[str, tuple[float, np.ndarray]] = {}
    selected_indices: dict[str, int] = {}
    object_candidates = [
        detection
        for detection in detections
        if detection["segment_candidate"] == "object" and detection["has_mask"] and not detection["rejection_reasons"]
    ]
    for detection in object_candidates:
        box_area_ratio = _box_area(detection["box_xyxy"]) / image_area
        detection["spatial_scores"]["box_area_ratio"] = box_area_ratio
        detection["score"] = float(detection["confidence"] + min(box_area_ratio, 0.35))
    if object_candidates:
        selected_object = max(object_candidates, key=lambda item: item["score"])
        object_union_mask, object_union_debug = _union_spatially_consistent_object_parts(
            detections=detections,
            primary_object=selected_object,
            image_diag=image_diag,
        )
        selected["object"] = (float(selected_object["confidence"]), object_union_mask)
        selected_indices["object"] = int(selected_object["index"])
    else:
        object_union_debug = {
            "primary_index": None,
            "unioned_indices": [],
            "masks_unioned": False,
            "candidate_decisions": {},
        }

    object_mask = selected.get("object", (None, None))[1]
    object_box = None
    if "object" in selected_indices:
        object_box = detections[selected_indices["object"]]["box_xyxy"]

    for detection in detections:
        segment_name = detection["segment_candidate"]
        if not segment_name or detection["rejection_reasons"]:
            continue
        box = detection["box_xyxy"]
        box_area_ratio = _box_area(box) / image_area
        bbox_overlap_ratio = 0.0
        if object_box is not None:
            bbox_overlap_ratio = _box_intersection_area(box, object_box) / max(1.0, _box_area(box))
        mask_overlap_ratio = _mask_overlap_ratio(detection["mask"], object_mask)
        distance_ratio = _center_distance_ratio(box, object_box, image_diag)
        center = _box_center(box)
        center_y_ratio = center[1] / image_height if center and image_height > 0 else None
        touches_edge = bool(
            box
            and (
                box[0] <= image_width * EDGE_MARGIN_RATIO
                or box[2] >= image_width * (1.0 - EDGE_MARGIN_RATIO)
            )
        )
        detection["spatial_scores"].update(
            {
                "box_area_ratio": box_area_ratio,
                "bbox_overlap_ratio_with_object": bbox_overlap_ratio,
                "mask_overlap_ratio_with_object": mask_overlap_ratio,
                "center_distance_ratio_with_object": distance_ratio,
                "center_y_ratio": center_y_ratio,
                "touches_image_edge": touches_edge,
                "thresholds": {
                    "min_mask_pixel_count": MIN_MASK_PIXEL_COUNT,
                    "pot_min_confidence": POT_MIN_CONFIDENCE,
                    "pot_min_object_mask_overlap": POT_MIN_OBJECT_MASK_OVERLAP,
                    "pot_min_object_bbox_overlap": POT_MIN_OBJECT_BBOX_OVERLAP,
                    "pot_max_center_distance_ratio": POT_MAX_CENTER_DISTANCE_RATIO,
                    "tiny_edge_area_ratio": TINY_EDGE_AREA_RATIO,
                    "table_min_area_ratio": TABLE_MIN_AREA_RATIO,
                    "table_min_center_y_ratio": TABLE_MIN_CENTER_Y_RATIO,
                },
            }
        )

        if segment_name == "object":
            if selected_indices.get("object") != detection["index"]:
                if detection.get("included_in_object_union"):
                    detection["assignment_decision"] = "unioned_into_object"
                else:
                    detection["rejection_reasons"].append("lower_quality_than_selected_object")
            continue

        if segment_name == "pot":
            if detection["confidence"] < POT_MIN_CONFIDENCE:
                detection["rejection_reasons"].append("pot_confidence_below_minimum")
            close_enough = distance_ratio is not None and distance_ratio <= POT_MAX_CENTER_DISTANCE_RATIO
            overlaps_object = (
                mask_overlap_ratio >= POT_MIN_OBJECT_MASK_OVERLAP
                or bbox_overlap_ratio >= POT_MIN_OBJECT_BBOX_OVERLAP
            )
            if object_mask is not None and not overlaps_object and not close_enough:
                detection["rejection_reasons"].append("pot_not_spatially_consistent_with_object")
            if touches_edge and box_area_ratio < TINY_EDGE_AREA_RATIO and not overlaps_object:
                detection["rejection_reasons"].append("tiny_edge_pot_without_strong_object_overlap")
            if detection["rejection_reasons"]:
                continue
            detection["score"] = float(
                detection["confidence"]
                + mask_overlap_ratio * 2.0
                + bbox_overlap_ratio
                - (distance_ratio or 0.0)
                - (0.25 if touches_edge and box_area_ratio < TINY_EDGE_AREA_RATIO else 0.0)
            )
        elif segment_name == "table":
            if box_area_ratio < TABLE_MIN_AREA_RATIO:
                detection["rejection_reasons"].append("table_area_below_minimum")
            if center_y_ratio is not None and center_y_ratio < TABLE_MIN_CENTER_Y_RATIO:
                detection["rejection_reasons"].append("table_not_low_enough_in_image")
            if detection["rejection_reasons"]:
                continue
            detection["score"] = float(detection["confidence"] + box_area_ratio + (center_y_ratio or 0.0))
        elif segment_name == "leaf":
            if object_mask is not None and mask_overlap_ratio < POT_MIN_OBJECT_MASK_OVERLAP:
                detection["rejection_reasons"].append("leaf_not_overlapping_selected_object")
            if detection["rejection_reasons"]:
                continue
            detection["score"] = float(detection["confidence"] + mask_overlap_ratio)

        previous_index = selected_indices.get(segment_name)
        if previous_index is None or detection["score"] > detections[previous_index]["score"]:
            selected[segment_name] = (float(detection["confidence"]), detection["mask"])
            selected_indices[segment_name] = int(detection["index"])

    for detection in detections:
        segment_name = detection["segment_candidate"]
        if not segment_name:
            continue
        selected_index = selected_indices.get(segment_name)
        if segment_name == "object" and detection.get("included_in_object_union"):
            detection["assignment_decision"] = "unioned_into_object"
        elif selected_index == detection["index"]:
            detection["assignment_decision"] = f"selected_{segment_name}"
        elif detection["rejection_reasons"]:
            detection["assignment_decision"] = f"rejected_{segment_name}"
        else:
            detection["assignment_decision"] = f"rejected_lower_quality_for_{segment_name}"
            detection["rejection_reasons"].append(f"lower_quality_than_selected_{segment_name}")

    debug_detections = []
    for detection in detections:
        clean_detection = {
            key: value for key, value in detection.items() if key != "mask"
        }
        debug_detections.append(clean_detection)

    debug = {
        "raw_detection_count": len(cls_values),
        "labels": [detection["label"] for detection in debug_detections],
        "classes": [detection["class_id"] for detection in debug_detections],
        "confidences": [detection["confidence"] for detection in debug_detections],
        "boxes": [detection["box_xyxy"] for detection in debug_detections],
        "has_mask": [detection["has_mask"] for detection in debug_detections],
        "mask_pixel_counts": [detection["mask_pixel_count"] for detection in debug_detections],
        "detections": debug_detections,
        "assignment_decisions": {
            segment_name: selected_indices.get(segment_name) for segment_name in selected
        },
        "object_union": object_union_debug,
        "rejection_reasons": {
            str(detection["index"]): detection["rejection_reasons"]
            for detection in debug_detections
            if detection["rejection_reasons"]
        },
    }
    return (
        {segment_name: item[1] for segment_name, item in selected.items()},
        {segment_name: item[0] for segment_name, item in selected.items()},
        debug,
    )


def _configure_yoloe_text_prompts(model: Any, prompts: list[str]) -> str | None:
    if not hasattr(model, "set_classes"):
        return "yoloe_set_classes_api_unavailable"
    prompt_tuple = tuple(str(prompt) for prompt in prompts)
    with timed_stage("yoloe_prompt_cache_lookup_s"):
        prompt_cache_hit = getattr(model, "_dimscan_active_prompt_tuple", None) == prompt_tuple
        resolved_source = str(
            getattr(model, "ckpt_path", None)
            or getattr(model, "model_name", None)
            or getattr(model, "_dimscan_resolved_model_source", "")
        )
        if resolved_source and _looks_prompt_free_model_source(resolved_source):
            return f"prompt_free_yoloe_checkpoint_rejected:{resolved_source}"
    if prompt_cache_hit:
        return None
    try:
        with timed_stage("yoloe_prompt_cache_build_s"):
            import torch

            with torch.inference_mode(False):
                with torch.no_grad():
                    model.set_classes(list(prompt_tuple))
                    setattr(model, "_dimscan_active_prompt_tuple", prompt_tuple)
                    configured = getattr(model, "_dimscan_configured_prompt_tuples", set())
                    if not isinstance(configured, set):
                        configured = set(configured)
                    configured.add(prompt_tuple)
                    setattr(model, "_dimscan_configured_prompt_tuples", configured)
    except ModuleNotFoundError as exc:
        if exc.name == "clip":
            return "clip_dependency_missing"
        return f"yoloe_prompt_dependency_missing:{exc.name}"
    except AssertionError as exc:
        text = str(exc)
        if "prompt-free" in text.lower() or "prompt free" in text.lower():
            return f"prompt_free_yoloe_checkpoint_rejected:{resolved_source or _model_type_name(model)}"
        return f"yoloe_set_classes_failed:{text}"
    except Exception as exc:
        text = str(exc)
        if "prompt-free" in text.lower() or "prompt free" in text.lower():
            return f"prompt_free_yoloe_checkpoint_rejected:{resolved_source or _model_type_name(model)}"
        return f"yoloe_set_classes_failed:{text}"
    return None


def _detection_summary(result: Any) -> dict[str, Any]:
    names = _result_names(result)
    cls_values, conf_values = _boxes(result)
    labels = [names.get(class_id, str(class_id)) for class_id in cls_values]
    return {
        "detection_count": len(cls_values),
        "labels": labels[:20],
        "confidences": conf_values[:20],
    }


def _run_focused_crop_pass(
    *,
    cfg: DimScanConfig,
    model_source: str,
    rgb_path: Path,
    view_path: Path,
    object_mask: np.ndarray,
    object_box: list[float],
    debug_mode: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any], str | None]:
    with timed_stage("yoloe_crop_prepare_s"):
        rgb_image = Image.open(rgb_path).convert("RGB")
        full_size = rgb_image.size
        crop_box = _crop_box_around(object_box, full_size)
        crop_image = rgb_image.crop(crop_box)
        temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if debug_mode:
            debug_dir = view_path / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            crop_path = debug_dir / "yoloe_object_crop.png"
        else:
            temp_dir = tempfile.TemporaryDirectory(prefix="dimscan_yoloe_crop_")
            crop_path = Path(temp_dir.name) / "yoloe_object_crop.png"
        crop_image.save(crop_path, format="PNG")

    try:
        with timed_stage("yoloe_crop_model_cache_getter_s"):
            crop_model = _cached_yoloe_model(model_source, list(FOCUSED_CROP_PROMPTS))
        with timed_stage("yoloe_prompt_setup_s"):
            prompt_error = _configure_yoloe_text_prompts(crop_model, FOCUSED_CROP_PROMPTS)
        if prompt_error is not None:
            return {}, {}, {"crop_bbox": list(crop_box), "error": prompt_error}, prompt_error

        try:
            with timed_stage("yoloe_inference_s"):
                with timed_stage("yoloe_predict_call_s"):
                    crop_results = crop_model.predict(
                        source=str(crop_path),
                        conf=float(cfg.yoloe_confidence_threshold),
                        verbose=False,
                    )
        except Exception as exc:
            return {}, {}, {"crop_bbox": list(crop_box), "error": f"crop_prediction_failed:{exc}"}, str(exc)

        if not crop_results:
            return {}, {}, {"crop_bbox": list(crop_box), "error": "crop_yoloe_returned_no_results"}, None

        with timed_stage("yoloe_crop_result_conversion_s"):
            crop_masks, crop_confidences, crop_debug = _select_masks(crop_results[0])
    finally:
        if temp_dir is not None:
            with timed_stage("yoloe_crop_temp_cleanup_s"):
                temp_dir.cleanup()
    with timed_stage("yoloe_crop_mask_mapping_s"):
        mapped_masks: dict[str, np.ndarray] = {}
        mapped_confidences: dict[str, float] = {}
        decisions: dict[str, Any] = {}
        for segment_name in ("pot", "leaf"):
            mask = crop_masks.get(segment_name)
            if mask is None:
                decisions[segment_name] = {"decision": "missing_from_crop"}
                continue
            full_mask = _mask_to_full_image(mask, crop_box, full_size)
            overlap = _mask_overlap_ratio(full_mask, object_mask)
            if overlap < POT_MIN_OBJECT_MASK_OVERLAP:
                decisions[segment_name] = {
                    "decision": "rejected_crop_segment",
                    "reason": "crop_segment_not_overlapping_object",
                    "object_overlap": overlap,
                    "minimum_overlap": POT_MIN_OBJECT_MASK_OVERLAP,
                }
                continue
            mapped_masks[segment_name] = full_mask
            mapped_confidences[segment_name] = crop_confidences.get(segment_name, 0.0)
            decisions[segment_name] = {
                "decision": "selected_crop_segment",
                "object_overlap": overlap,
                "confidence": mapped_confidences[segment_name],
            }

        mapped_debug = _crop_prediction_debug_to_full(crop_debug, crop_box, object_mask)
        mapped_debug.update(
            {
                "crop_bbox": list(crop_box),
                "crop_image": str(crop_path),
                "prompts": list(FOCUSED_CROP_PROMPTS),
                "assignment_decisions": decisions,
            }
        )
    return mapped_masks, mapped_confidences, mapped_debug, None


def run_yoloe_segmentation(
    cfg: DimScanConfig,
    view_dir: str | Path,
    *,
    force: bool = False,
    debug_mode: bool = True,
    object_extraction_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run optional YOLOE-style segmentation for one captured view."""
    view_path = Path(view_dir)
    with timed_stage("ai1_existing_segmentation_read_s"):
        existing = read_json_if_exists(view_path / "segmentation.json")
    if existing is not None and not force:
        add_note("ai1_existing_segmentation_reused_without_inference")
        return existing
    with timed_stage("ai1_stale_output_cleanup_s"):
        _clear_stale_outputs(view_path, preserve_object_cloud=object_extraction_result is not None)
    if cfg.segmentation_backend != "yoloe":
        add_note(f"ai1_skipped_backend:{cfg.segmentation_backend}")
        with timed_stage("ai1_skipped_record_write_s"):
            return _skipped_record(
                cfg,
                view_path,
                reason=f"unsupported_segmentation_backend:{cfg.segmentation_backend}",
            )

    with timed_stage("ai1_path_model_setup_s"):
        rgb_path = view_path / cfg.rgb_filename
        depth_path = view_path / cfg.depth_raw_filename
        depth_aligned_to_rgb_path = view_path / getattr(cfg, "depth_aligned_to_rgb_filename", "depth_aligned_to_rgb.npy")
        configured_source = configured_model_source(cfg)
        model_source = resolve_yoloe_model_source(configured_source) if configured_source is not None else None
    debug: dict[str, Any] = {
        "model_name": model_source,
        "configured_model_source": configured_source,
        "prompts": list(cfg.yoloe_prompts),
        "stage_reached": "dependency_check",
        "error": None,
        "raw_detection_count": None,
        "labels": [],
        "classes": [],
        "confidences": [],
        "boxes": [],
        "has_mask": [],
        "mask_pixel_counts": [],
        "assignment_decisions": {},
        "rejection_reasons": {},
    }
    with timed_stage("ai1_required_artifact_check_s"):
        missing_rgb = not rgb_path.is_file()
        missing_depth = not depth_path.is_file()
    if missing_rgb:
        add_note("ai1_skipped_missing_rgb")
        with timed_stage("ai1_skipped_record_write_s"):
            return _skipped_record(cfg, view_path, reason=f"missing_rgb:{rgb_path}")
    if missing_depth:
        add_note("ai1_skipped_missing_depth_raw")
        with timed_stage("ai1_skipped_record_write_s"):
            return _skipped_record(cfg, view_path, reason=f"missing_depth_raw:{depth_path}")

    object_extraction: dict[str, Any] | None = None
    object_extraction_warnings: list[str] = []
    if object_extraction_result is not None:
        with timed_stage("ai1_object_extraction_runtime_reuse_s"):
            object_extraction = object_extraction_result
            extraction_error = object_extraction.get("error") if isinstance(object_extraction, dict) else None
            if extraction_error:
                object_extraction_warnings.append(f"geometry_primary_object_extraction_failed:{extraction_error}")
    else:
        object_extraction_debug_path = view_path / "debug" / "object_extraction_debug.json"
        with timed_stage("ai1_object_extraction_debug_read_s"):
            existing_object_extraction = read_json_if_exists(object_extraction_debug_path)
        if (view_path / "object_cloud.ply").is_file() and isinstance(existing_object_extraction, dict):
            with timed_stage("ai1_object_extraction_reuse_s"):
                object_extraction = existing_object_extraction
        else:
            try:
                object_extraction = extract_geometry_primary_object_cloud(cfg, view_path, debug_mode=debug_mode)
            except Exception as exc:
                with timed_stage("ai1_object_extraction_failure_record_s"):
                    object_extraction_warnings.append(f"geometry_primary_object_extraction_failed:{exc}")
                    if debug_mode:
                        object_extraction_debug_path.parent.mkdir(parents=True, exist_ok=True)
                        write_json_atomic(
                            object_extraction_debug_path,
                            {
                                "extraction_method": "roi_table_cluster",
                                "authoritative_object_source": "geometry_cluster",
                                "status": "failed",
                                "error": str(exc),
                                "ai1_role": "validation_only",
                                "ai1_used_for_object_extraction": False,
                            },
                        )

    if not cfg.segmentation_enabled or not bool(getattr(cfg, "enable_ai1_validation", True)):
        add_note("ai1_validation_disabled_no_yoloe_inference")
        with timed_stage("ai1_geometry_primary_record_write_s"):
            debug["stage_reached"] = "ai1_validation_skipped"
            debug["ai1_role"] = "validation_only"
            debug["ai1_used_for_object_extraction"] = False
            debug["object_extraction"] = _without_runtime_keys(object_extraction)
            _write_yoloe_debug(view_path, debug, debug_mode=debug_mode)
            return _geometry_primary_record(
                cfg,
                view_path,
                model_source=model_source,
                object_extraction=object_extraction,
                ai1_reason="ai1_validation_disabled",
                warnings=object_extraction_warnings,
                debug_mode=debug_mode,
            )

    if model_source is None:
        add_note("ai1_model_not_configured_no_yoloe_inference")
        with timed_stage("ai1_geometry_primary_record_write_s"):
            return _geometry_primary_record(
                cfg,
                view_path,
                model_source=model_source,
                object_extraction=object_extraction,
                ai1_reason="ai1_validation_model_not_configured",
                warnings=[
                    *object_extraction_warnings,
                    "Set DIMSCAN_YOLOE_MODEL_PATH or cfg.yoloe_model_path to a local YOLOE segmentation model.",
                ],
                debug_mode=debug_mode,
            )
    if _looks_prompt_free_model_source(model_source):
        add_note(f"prompt_free_yoloe_checkpoint_rejected:{model_source}")
        with timed_stage("ai1_geometry_primary_record_write_s"):
            return _geometry_primary_record(
                cfg,
                view_path,
                model_source=model_source,
                object_extraction=object_extraction,
                ai1_reason=f"prompt_free_yoloe_checkpoint_rejected:{model_source}",
                warnings=object_extraction_warnings,
                debug_mode=debug_mode,
            )
    if (os.sep in model_source or model_source.endswith(".pt")) and not Path(model_source).is_file():
        add_note(f"ai1_model_not_found_no_yoloe_inference:{model_source}")
        with timed_stage("ai1_geometry_primary_record_write_s"):
            return _geometry_primary_record(
                cfg,
                view_path,
                model_source=model_source,
                object_extraction=object_extraction,
                ai1_reason=f"ai1_validation_model_not_found:{model_source}",
                warnings=object_extraction_warnings,
                debug_mode=debug_mode,
            )

    model = None
    try:
        debug["stage_reached"] = "model_load"
        with timed_stage("yoloe_cache_getter_s"):
            model = _cached_yoloe_model(model_source, list(cfg.yoloe_prompts))
        debug["model_trace"] = _model_trace(model, configured_source or model_source)
        debug["stage_reached"] = "prompt_setup"
        with timed_stage("yoloe_prompt_setup_s"):
            prompt_error = _configure_yoloe_text_prompts(model, list(cfg.yoloe_prompts))
        if prompt_error is not None:
            debug["error"] = prompt_error
            add_note(f"ai1_prompt_setup_failed_no_yoloe_inference:{prompt_error}")
            with timed_stage("ai1_geometry_primary_record_write_s"):
                _write_yoloe_debug(view_path, debug, debug_mode=debug_mode)
                return _geometry_primary_record(
                    cfg,
                    view_path,
                    model_source=model_source,
                    object_extraction=object_extraction,
                    ai1_reason=f"ai1_validation_{prompt_error}",
                    warnings=[
                        *object_extraction_warnings,
                        "pip install git+https://github.com/ultralytics/CLIP.git",
                    ],
                    debug_mode=debug_mode,
                )
        debug["stage_reached"] = "prediction"
        with timed_stage("yoloe_inference_s"):
            with timed_stage("yoloe_predict_call_s"):
                results = model.predict(
                    source=str(rgb_path),
                    conf=float(cfg.yoloe_confidence_threshold),
                    verbose=False,
                )
    except ImportError:
        debug["error"] = "ultralytics_not_installed"
        add_note("ai1_ultralytics_not_installed_no_yoloe_inference")
        with timed_stage("ai1_geometry_primary_record_write_s"):
            _write_yoloe_debug(view_path, debug, debug_mode=debug_mode)
            return _geometry_primary_record(
                cfg,
                view_path,
                model_source=model_source,
                object_extraction=object_extraction,
                ai1_reason="ai1_validation_ultralytics_not_installed",
                warnings=[
                    *object_extraction_warnings,
                    "Install ultralytics in the project venv and configure a local YOLOE segmentation model.",
                ],
                debug_mode=debug_mode,
            )
    except Exception as exc:
        debug["error"] = str(exc)
        add_note(f"ai1_inference_failed:{exc}")
        with timed_stage("ai1_geometry_primary_record_write_s"):
            _write_yoloe_debug(view_path, debug, debug_mode=debug_mode)
            return _geometry_primary_record(
                cfg,
                view_path,
                model_source=model_source,
                object_extraction=object_extraction,
                ai1_reason=f"ai1_validation_inference_failed:{exc}",
                warnings=object_extraction_warnings,
                debug_mode=debug_mode,
            )

    if not results:
        debug["error"] = "yoloe_returned_no_results"
        add_note("ai1_yoloe_returned_no_results")
        with timed_stage("ai1_geometry_primary_record_write_s"):
            _write_yoloe_debug(view_path, debug, debug_mode=debug_mode)
            return _geometry_primary_record(
                cfg,
                view_path,
                model_source=model_source,
                object_extraction=object_extraction,
                ai1_reason="ai1_validation_returned_no_results",
                warnings=object_extraction_warnings,
                debug_mode=debug_mode,
            )

    with timed_stage("ai1_rgb_size_decode_s"):
        rgb_size = Image.open(rgb_path).size
    with timed_stage("ai1_mask_select_restore_s"):
        model_masks, confidences, prediction_debug = _select_masks(results[0])
        masks, mask_restore_debug = _restore_masks_to_rgb_shape(model_masks, rgb_size=rgb_size)
    with timed_stage("ai1_mask_debug_write_s"):
        if debug_mode:
            ai1_mask_debug = _write_ai1_mask_debug_artifacts(
                cfg=cfg,
                view_dir=view_path,
                rgb_path=rgb_path,
                result=results[0],
                prediction_debug=prediction_debug,
                model_masks=model_masks,
                restored_masks=masks,
                mask_restore_debug=mask_restore_debug,
                final_object_mask=masks.get("object"),
            )
        else:
            ai1_mask_debug = None
    debug.update(prediction_debug)
    debug["full_image"] = prediction_debug
    debug["mask_restoration"] = mask_restore_debug
    debug["ai1_mask_debug"] = ai1_mask_debug
    debug["crop"] = {}
    debug["fallback_decisions"] = {}
    object_box = _selected_box(prediction_debug, "object")
    if "object" in masks and object_box is not None:
        crop_masks, crop_confidences, crop_debug, crop_error = _run_focused_crop_pass(
            cfg=cfg,
            model_source=model_source,
            rgb_path=rgb_path,
            view_path=view_path,
            object_mask=masks["object"],
            object_box=object_box,
            debug_mode=debug_mode,
        )
        debug["crop"] = crop_debug
        if crop_error is not None:
            debug["crop"]["error"] = crop_error
        for segment_name in ("pot", "leaf"):
            if segment_name in crop_masks and segment_name not in masks:
                masks[segment_name] = crop_masks[segment_name]
                confidences[segment_name] = crop_confidences.get(segment_name, 0.0)
                debug["assignment_decisions"][segment_name] = f"selected_{segment_name}_from_crop"
            elif segment_name in crop_masks:
                existing_confidence = confidences.get(segment_name, 0.0)
                crop_confidence = crop_confidences.get(segment_name, 0.0)
                if crop_confidence > existing_confidence:
                    masks[segment_name] = crop_masks[segment_name]
                    confidences[segment_name] = crop_confidence
                    debug["assignment_decisions"][segment_name] = f"selected_{segment_name}_from_crop"
    elif "object" in masks:
        debug["crop"] = {"error": "object_bbox_missing"}

    fallback_segments: set[str] = set()
    with timed_stage("ai1_fallback_mask_handling_s"):
        if "object" in masks:
            if "leaf" not in masks:
                leaf_fallback = _upper_object_fallback_mask(masks["object"])
                if leaf_fallback is not None:
                    masks["leaf"] = leaf_fallback
                    confidences["leaf"] = 0.0
                    fallback_segments.add("leaf")
                    debug["fallback_decisions"]["leaf"] = {
                        "decision": "created_fallback_from_upper_object_mask",
                        "mask_pixel_count": int(np.count_nonzero(leaf_fallback)),
                    }
                else:
                    debug["fallback_decisions"]["leaf"] = {"decision": "fallback_impossible"}
            if "pot" not in masks:
                pot_fallback = _lower_central_object_fallback_mask(masks["object"])
                if pot_fallback is not None:
                    masks["pot"] = pot_fallback
                    confidences["pot"] = 0.0
                    fallback_segments.add("pot")
                    debug["fallback_decisions"]["pot"] = {
                        "decision": "created_fallback_from_lower_central_object_mask",
                        "mask_pixel_count": int(np.count_nonzero(pot_fallback)),
                    }
                else:
                    debug["fallback_decisions"]["pot"] = {"decision": "fallback_impossible"}

    debug["final_mask_shapes"] = {
        segment_name: {
            "shape": [int(mask.shape[0]), int(mask.shape[1])] if np.asarray(mask).ndim == 2 else list(np.asarray(mask).shape),
            "foreground_pixel_count": int(np.count_nonzero(mask)),
            "matches_rgb_shape": tuple(np.asarray(mask).shape[:2]) == (int(rgb_size[1]), int(rgb_size[0])),
        }
        for segment_name, mask in masks.items()
    }
    with timed_stage("ai1_detection_summary_s"):
        _write_yoloe_debug(view_path, debug, debug_mode=debug_mode)
        detection_summary = _detection_summary(results[0])
        warnings: list[str] = list(object_extraction_warnings)
        if "leaf" in fallback_segments:
            warnings.append("leaf_mask_fallback_from_object")
        if "pot" in fallback_segments:
            warnings.append("pot_mask_fallback_from_object")
        segment_statuses = {
            name: ("fallback" if name in fallback_segments else ("ok" if name in masks else "missing"))
            for name in SEGMENT_CLASS_HINTS
        }
        for segment_name in ("pot", "leaf", "table"):
            if segment_statuses.get(segment_name) == "missing":
                warnings.append(f"{segment_name}_segment_missing")
        useful_segments = [name for name in ("object", "pot", "leaf", "table") if name in masks]
    if not useful_segments:
        return _failed_record(
            cfg,
            view_path,
            reason="no_useful_segments_found",
            warnings=[
                "No object, pot, leaf, or table masks matched configured class hints.",
                f"raw_detection_labels:{detection_summary['labels']}",
            ],
            confidence_summary=detection_summary,
        )
    with timed_stage("ai1_depth_meta_load_s"):
        depth = np.load(depth_path)
        depth_aligned_to_rgb = np.load(depth_aligned_to_rgb_path) if depth_aligned_to_rgb_path.is_file() else None
        capture_meta = read_json_if_exists(view_path / "capture_meta.json", default={})
        if not isinstance(capture_meta, dict):
            capture_meta = {}
        masks_dir = view_path / "masks"
        artifacts: dict[str, str] = {}
        if object_extraction is not None and object_extraction.get("final_point_count", 0) > 0:
            object_cloud_path = view_path / "object_cloud.ply"
            if object_cloud_path.is_file():
                artifacts["object_cloud"] = str(object_cloud_path)
            if debug_mode:
                artifacts["object_extraction_debug"] = str(view_path / "debug" / "object_extraction_debug.json")
    pot_depth_metrics = (
            _cloud_metrics_for_mask(
                view_dir=view_path,
                mask=masks["pot"],
                depth=depth,
                capture_meta=capture_meta,
            )
        if "pot" in masks
        else {}
    )
    with timed_stage("ai1_pot_quality_s"):
        pot_quality = _pot_quality_record(
            cfg=cfg,
            pot_mask=masks.get("pot"),
            object_mask=masks.get("object"),
            pot_confidence=confidences.get("pot"),
            depth_metrics=pot_depth_metrics,
            is_fallback="pot" in fallback_segments,
            image_size=rgb_size,
        )
        debug["pot_quality"] = pot_quality
        _write_yoloe_debug(view_path, debug, debug_mode=debug_mode)

    if pot_quality["status"] == "trusted":
        segment_statuses["pot"] = "ok"
    elif pot_quality["status"] == "missing":
        segment_statuses["pot"] = "missing"
        warnings.append("pot_segment_missing")
    else:
        segment_statuses["pot"] = "fallback" if pot_quality["status"] == "fallback" else "rejected"
        warnings.append("pot_mask_rejected_not_used_for_geometry")

    with timed_stage("ai1_mask_artifact_handling_s"):
        for segment_name, mask in masks.items():
            mask_path = masks_dir / f"{segment_name}_mask.png"
            _write_mask(mask_path, mask, rgb_size)
            artifacts[f"{segment_name}_mask"] = str(mask_path)
            if segment_name == "object":
                explicit_object_mask_path = view_path / "object_mask.png"
                _write_mask(explicit_object_mask_path, mask, rgb_size)
                saved_object_mask = _read_mask_png_as_bool(explicit_object_mask_path)
                mask_hash = _mask_sha256(mask)
                saved_mask_hash = _mask_sha256(saved_object_mask)
                if mask_hash != saved_mask_hash:
                    raise ValueError("saved_object_mask_checksum_mismatch")
                artifacts["object_mask"] = str(explicit_object_mask_path)
                debug["object_mask_artifact"] = str(explicit_object_mask_path)
                debug["object_mask_sha256_saved"] = saved_mask_hash
                debug["ai1_role"] = "validation_only"
                debug["ai1_used_for_object_extraction"] = False
                debug["object_mask_sha256_validation_only"] = mask_hash
                continue
            if segment_name == "pot" and not pot_quality.get("usable_for_geometry"):
                rejected_path = view_path / "debug" / "pot_cloud_rejected.ply"
                debug_metrics = _cloud_metrics_for_mask(
                    view_dir=view_path,
                    mask=mask,
                    depth=depth,
                    output_path=rejected_path if debug_mode else None,
                    capture_meta=capture_meta,
                )
                pot_quality["metrics"].update(debug_metrics)
                if debug_mode and rejected_path.is_file():
                    artifacts["pot_cloud_rejected"] = str(rejected_path)
                continue
            cloud_output = _write_segment_cloud(
                view_dir=view_path,
                segment_name=segment_name,
                mask=mask,
                depth=depth,
                cfg=cfg,
                depth_aligned_to_rgb=depth_aligned_to_rgb if segment_name == "object" else None,
                warnings=warnings,
                capture_meta=capture_meta,
                rgb_size=rgb_size,
                debug=debug,
                debug_mode=debug_mode,
            )
            if segment_name == "object" and debug.get("object_cloud_debug"):
                cloud_hash = debug["object_cloud_debug"].get("object_mask_sha256_consumed_by_cloud")
                if cloud_hash != debug.get("object_mask_sha256_consumed_by_cloud"):
                    raise ValueError("object_mask_consumed_by_cloud_checksum_mismatch")
            if cloud_output:
                artifacts[f"{segment_name}_cloud"] = cloud_output

    if "object" in masks and not (
        object_extraction is not None and object_extraction.get("final_point_count", 0) > 0
    ):
        segment_statuses["object"] = "failed"
        warnings.append("object_cloud_unavailable_geometry_blocked")

    with timed_stage("ai1_record_assembly_write_s"):
        _write_yoloe_debug(view_path, debug, debug_mode=debug_mode)

        status = "ok" if all(segment_statuses.get(name) == "ok" for name in ("object", "pot", "leaf", "table")) else "partial"
        record = make_segmentation_record(
            status=status,
            model_backend=cfg.segmentation_backend,
            model_name=model_source,
            model_version=getattr(model, "version", None),
            prompts=list(cfg.yoloe_prompts),
            segment_statuses=segment_statuses,
            confidence_summary=confidences,
            pot_quality=pot_quality,
            warnings=warnings,
            artifacts=artifacts,
        )
        if object_extraction is not None and object_extraction.get("final_point_count", 0) > 0:
            runtime_points = object_extraction.get("_runtime_final_object_points")
            if runtime_points is not None:
                record["_runtime_final_object_points"] = runtime_points
        return _write_record(view_path, record)
