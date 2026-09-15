"""RGB-D artifact writers for DimScan capture flows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pipeline.profiling import timed_stage
from utils.io import utc_now_iso


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"


def _image_from_encoded_bytes(data: bytes) -> Image.Image | None:
    if not data.startswith(PNG_SIGNATURE) and not data.startswith(JPEG_SIGNATURE):
        return None
    try:
        from io import BytesIO

        return Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        return None


def _raw_rgb_image(data: bytes, metadata: dict[str, Any]) -> Image.Image:
    width = int(metadata.get("rgb_width") or metadata.get("color_width") or metadata.get("width") or 0)
    height = int(metadata.get("rgb_height") or metadata.get("color_height") or metadata.get("height") or 0)
    rgb_format = str(metadata.get("rgb_format") or metadata.get("color_format") or "RGB").upper()

    if width <= 0 or height <= 0:
        pixel_count = max(1, len(data) // 3)
        width = pixel_count
        height = 1

    expected_rgb = width * height * 3
    expected_rgba = width * height * 4
    if len(data) >= expected_rgba and rgb_format in {"RGBA", "BGRA"}:
        array = np.frombuffer(data[:expected_rgba], dtype=np.uint8).reshape((height, width, 4))
        if rgb_format == "BGRA":
            array = array[:, :, [2, 1, 0, 3]]
        return Image.fromarray(array, mode="RGBA").convert("RGB")
    if len(data) >= expected_rgb:
        array = np.frombuffer(data[:expected_rgb], dtype=np.uint8).reshape((height, width, 3))
        if rgb_format == "BGR":
            array = array[:, :, ::-1]
        return Image.fromarray(array, mode="RGB")

    fallback = np.zeros((height, width, 3), dtype=np.uint8)
    usable = min(fallback.size, len(data))
    if usable:
        fallback.reshape(-1)[:usable] = np.frombuffer(data[:usable], dtype=np.uint8)
    return Image.fromarray(fallback, mode="RGB")


def write_rgb_png(
    path: str | Path,
    rgb_bytes: bytes,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write RGB bytes as a real PNG image."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = _image_from_encoded_bytes(rgb_bytes) or _raw_rgb_image(rgb_bytes, metadata or {})
    image.save(output_path, format="PNG", compress_level=1)
    return output_path


def write_rgb_placeholder(path: str | Path, rgb_bytes: bytes) -> Path:
    """Backward-compatible wrapper that writes a valid PNG image."""
    return write_rgb_png(path, rgb_bytes)


def _depth_array(depth_values: list[list[float]]) -> np.ndarray:
    return np.asarray(depth_values, dtype=np.float32)


def write_depth_raw_npy(path: str | Path, depth_values: list[list[float]]) -> Path:
    """Write raw numeric depth values as a NumPy array."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, _depth_array(depth_values))
    return output_path


def write_depth_aligned_to_rgb_npy(path: str | Path, depth_values: list[list[float]]) -> Path:
    """Write RGB-grid aligned depth values as a NumPy array."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, _depth_array(depth_values))
    return output_path


def write_depth_preview_png(path: str | Path, depth_values: list[list[float]]) -> Path:
    """Write a normalized viewable depth preview PNG."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with timed_stage("depth_preview_prepare_s"):
        depth = _depth_array(depth_values)
        preview = np.zeros(depth.shape, dtype=np.uint8)
        valid = np.isfinite(depth) & (depth > 0)

        if valid.any():
            valid_values = depth[valid]
            low = float(np.percentile(valid_values, 2))
            high = float(np.percentile(valid_values, 98))
            if high <= low:
                high = float(valid_values.max())
                low = float(valid_values.min())
            if high > low:
                normalized = (np.clip(depth, low, high) - low) / (high - low)
                preview[valid] = np.asarray(normalized[valid] * 255, dtype=np.uint8)

    with timed_stage("depth_preview_png_write_s"):
        Image.fromarray(preview, mode="L").save(output_path, format="PNG")
    return output_path


def write_depth_json(
    path: str | Path,
    depth_values: list[list[float]],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write placeholder depth values as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "depth_values": depth_values,
        "metadata": metadata or {},
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=False)
        file.write("\n")

    return output_path


def depth_stats(depth_values: list[list[float]]) -> dict[str, Any]:
    """Return basic depth array stats for capture metadata."""
    depth = _depth_array(depth_values)
    if depth.size == 0:
        return {
            "depth_min": None,
            "depth_max": None,
            "depth_nonzero_ratio": 0.0,
        }

    finite = depth[np.isfinite(depth)]
    valid = finite[finite > 0]
    return {
        "depth_min": float(valid.min()) if valid.size else 0.0,
        "depth_median": float(np.median(valid)) if valid.size else 0.0,
        "depth_max": float(valid.max()) if valid.size else 0.0,
        "depth_nonzero_ratio": float(np.count_nonzero(depth > 0) / depth.size),
    }


def _array_depth_stats(depth: np.ndarray | None, *, prefix: str) -> dict[str, Any]:
    if depth is None or depth.size == 0:
        return {
            f"{prefix}_min": None,
            f"{prefix}_median": None,
            f"{prefix}_max": None,
            f"{prefix}_valid_pixel_count": None,
            f"{prefix}_valid_percentage": None,
        }
    valid = depth[np.isfinite(depth) & (depth > 0)]
    return {
        f"{prefix}_min": float(valid.min()) if valid.size else 0.0,
        f"{prefix}_median": float(np.median(valid)) if valid.size else 0.0,
        f"{prefix}_max": float(valid.max()) if valid.size else 0.0,
        f"{prefix}_valid_pixel_count": int(valid.size),
        f"{prefix}_valid_percentage": float(valid.size / depth.size * 100.0),
    }


def depth_to_points(
    depth_values: list[list[float]],
    *,
    scale: float = 1.0,
) -> list[tuple[float, float, float]]:
    """Convert depth cells to fake XYZ points without camera projection."""
    points: list[tuple[float, float, float]] = []
    for row_index, row in enumerate(depth_values):
        for col_index, depth_value in enumerate(row):
            points.append((col_index * scale, row_index * scale, float(depth_value)))
    return points


def write_ascii_ply(path: str | Path, points: list[tuple[float, float, float]]) -> Path:
    """Write a minimal ASCII PLY point cloud file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("end_header\n")
        for x, y, z in points:
            file.write(f"{x} {y} {z}\n")

    return output_path


def write_ascii_ply_with_colors(
    path: str | Path,
    points: list[tuple[float, float, float]],
    colors: list[tuple[int, int, int]] | None = None,
) -> Path:
    """Write an ASCII PLY point cloud with optional RGB colors."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    has_colors = colors is not None and len(colors) == len(points)

    with output_path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        if has_colors:
            file.write("property uchar red\n")
            file.write("property uchar green\n")
            file.write("property uchar blue\n")
        file.write("end_header\n")
        for index, (x, y, z) in enumerate(points):
            if has_colors:
                red, green, blue = colors[index]
                file.write(f"{x} {y} {z} {red} {green} {blue}\n")
            else:
                file.write(f"{x} {y} {z}\n")

    return output_path


def _intrinsic_value(intrinsics: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = intrinsics.get(name)
        if value is not None:
            return float(value)
    return None


def metric_points_from_depth(
    depth_values: list[list[float]],
    intrinsics: dict[str, Any],
) -> list[tuple[float, float, float]]:
    """Project depth pixels into XYZ using pinhole intrinsics."""
    fx = _intrinsic_value(intrinsics, ("fx", "focal_length_x"))
    fy = _intrinsic_value(intrinsics, ("fy", "focal_length_y"))
    cx = _intrinsic_value(intrinsics, ("cx", "principal_point_x", "ppx"))
    cy = _intrinsic_value(intrinsics, ("cy", "principal_point_y", "ppy"))
    if not fx or not fy or cx is None or cy is None:
        return []

    depth = _depth_array(depth_values)
    points: list[tuple[float, float, float]] = []
    for row_index in range(depth.shape[0]):
        for col_index in range(depth.shape[1]):
            z = float(depth[row_index, col_index])
            if not np.isfinite(z) or z <= 0:
                continue
            x = (col_index - cx) * z / fx
            y = (row_index - cy) * z / fy
            points.append((x, y, z))
    return points


def valid_depth_pixel_indices_from_depth(depth_values: list[list[float]]) -> np.ndarray:
    """Return [row, col] depth pixels in the same order as metric_points_from_depth."""
    depth = _depth_array(depth_values)
    indices: list[tuple[int, int]] = []
    for row_index in range(depth.shape[0]):
        for col_index in range(depth.shape[1]):
            z = float(depth[row_index, col_index])
            if not np.isfinite(z) or z <= 0:
                continue
            indices.append((row_index, col_index))
    return np.asarray(indices, dtype=np.int32).reshape((-1, 2))


def write_cloud_pixel_indices_npy(path: str | Path, pixel_indices: np.ndarray) -> Path:
    """Write the point-row to depth-pixel map for a compact raw cloud."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.asarray(pixel_indices, dtype=np.int32))
    return output_path


def colors_from_rgb_bytes(
    rgb_bytes: bytes,
    metadata: dict[str, Any],
    *,
    target_width: int,
    target_height: int,
) -> list[tuple[int, int, int]] | None:
    """Return per-pixel RGB colors when RGB dimensions match target dimensions."""
    rgb_width = int(metadata.get("rgb_width") or metadata.get("width") or 0)
    rgb_height = int(metadata.get("rgb_height") or metadata.get("height") or 0)
    if rgb_width != target_width or rgb_height != target_height:
        return None

    image = _image_from_encoded_bytes(rgb_bytes) or _raw_rgb_image(rgb_bytes, metadata)
    if image.size != (target_width, target_height):
        return None

    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).reshape(-1, 3)
    return [(int(red), int(green), int(blue)) for red, green, blue in pixels]


def write_capture_meta(
    path: str | Path,
    metadata: dict[str, Any],
    depth_values: list[list[float]],
    *,
    cloud_type: str,
    cloud_pixel_indices_path: str | Path | None = None,
    cloud_pixel_indices_count: int | None = None,
    depth_aligned_to_rgb_values: list[list[float]] | None = None,
    depth_aligned_to_rgb_path: str | Path | None = None,
) -> Path:
    """Write view-level capture metadata."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    depth = _depth_array(depth_values)
    aligned_depth = _depth_array(depth_aligned_to_rgb_values) if depth_aligned_to_rgb_values is not None else None
    stats = depth_stats(depth_values)
    aligned_valid = (
        np.isfinite(aligned_depth) & (aligned_depth > 0)
        if aligned_depth is not None and aligned_depth.ndim == 2
        else None
    )
    record = dict(metadata)
    record.update(
        {
            "timestamp": utc_now_iso(),
            "rgb_width": metadata.get("rgb_width") or metadata.get("width"),
            "rgb_height": metadata.get("rgb_height") or metadata.get("height"),
            "depth_width": int(depth.shape[1]) if depth.ndim == 2 else 0,
            "depth_height": int(depth.shape[0]) if depth.ndim == 2 else 0,
            "depth_dtype": str(depth.dtype),
            "raw_depth_shape": [int(depth.shape[0]), int(depth.shape[1])] if depth.ndim == 2 else None,
            "aligned_depth_shape": (
                [int(aligned_depth.shape[0]), int(aligned_depth.shape[1])]
                if aligned_depth is not None and aligned_depth.ndim == 2
                else None
            ),
            "depth_aligned_to_rgb": aligned_depth is not None,
            "depth_aligned_to_rgb_path": str(depth_aligned_to_rgb_path) if depth_aligned_to_rgb_path is not None else None,
            "aligned_depth_valid_pixel_count": int(np.count_nonzero(aligned_valid)) if aligned_valid is not None else None,
            "aligned_depth_valid_percentage": (
                float(np.count_nonzero(aligned_valid) / aligned_depth.size * 100.0)
                if aligned_valid is not None and aligned_depth.size
                else None
            ),
            "cloud_type": cloud_type,
            "cloud_pixel_indices": str(cloud_pixel_indices_path) if cloud_pixel_indices_path is not None else None,
            "cloud_pixel_indices_count": cloud_pixel_indices_count,
            "cloud_pixel_indices_debug_only": cloud_pixel_indices_path is not None,
            **_array_depth_stats(depth, prefix="raw_depth"),
            **_array_depth_stats(aligned_depth, prefix="aligned_depth"),
            **stats,
        }
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=2, sort_keys=False)
        file.write("\n")
    return output_path


def write_pointcloud_from_depth(
    path: str | Path,
    depth_values: list[list[float]],
    *,
    scale: float = 1.0,
) -> Path:
    """Write a fake point cloud generated from placeholder depth values."""
    points = depth_to_points(depth_values, scale=scale)
    return write_ascii_ply(path, points)
