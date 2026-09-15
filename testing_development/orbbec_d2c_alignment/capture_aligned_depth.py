"""Capture Orbbec SDK depth-to-color aligned depth for hardware validation.

This experiment is intentionally isolated from DimScan production capture,
segmentation, geometry, and feature code. It uses Orbbec's official hardware
D2C mode when available, otherwise Orbbec's AlignFilter software D2C path.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
COLOR_SIZE = (1280, 720)
DEPTH_SIZE = (1280, 800)
PREFERRED_FPS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orbbec official D2C alignment capture experiment")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mask", type=Path, default=None, help="Optional RGB-grid object mask PNG")
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--timeout-ms", type=int, default=3000)
    parser.add_argument("--force-sw", action="store_true", help="Skip HW D2C and use AlignFilter D2C")
    parser.add_argument("--prefer-hw", action="store_true", default=True, help="Prefer hardware D2C when available")
    parser.add_argument("--list-profiles", action="store_true", help="Print profiles and exit without capture")
    return parser.parse_args()


def sdk_import() -> tuple[Any, str, str]:
    module = importlib.import_module("pyorbbecsdk")
    try:
        version = importlib.metadata.version("pyorbbecsdk2")
        package = "pyorbbecsdk2"
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
        package = "pyorbbecsdk"
    return module, package, version


def enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if callable(name):
        name = name()
    if isinstance(name, str):
        return name
    return str(value).rsplit(".", 1)[-1]


def call_optional(obj: Any, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        value = getattr(obj, name, None)
        if callable(value):
            try:
                return value()
            except Exception:
                continue
        if value is not None:
            return value
    return default


def stream_profiles(pipeline: Any, sensor_type: Any) -> list[Any]:
    profile_list = pipeline.get_stream_profile_list(sensor_type)
    count = int(profile_list.get_count())
    profiles: list[Any] = []
    for index in range(count):
        profile = profile_list.get_stream_profile_by_index(index)
        if hasattr(profile, "as_video_stream_profile"):
            profile = profile.as_video_stream_profile()
        if profile_int(profile, "get_width") > 0 and profile_int(profile, "get_height") > 0:
            profiles.append(profile)
    return profiles


def profile_int(profile: Any, method_name: str) -> int:
    method = getattr(profile, method_name, None)
    if callable(method):
        return int(method())
    return 0


def profile_format(profile: Any) -> str:
    method = getattr(profile, "get_format", None)
    return enum_name(method()) if callable(method) else "UNKNOWN"


def profile_summary(profile: Any) -> dict[str, Any]:
    return {
        "width": profile_int(profile, "get_width"),
        "height": profile_int(profile, "get_height"),
        "fps": profile_int(profile, "get_fps"),
        "format": profile_format(profile),
        "description": (
            f"{profile_int(profile, 'get_width')}x{profile_int(profile, 'get_height')} "
            f"{profile_format(profile)} {profile_int(profile, 'get_fps')}fps"
        ),
    }


def print_profiles(pipeline: Any, ob: Any) -> None:
    for label, sensor in (("color", ob.OBSensorType.COLOR_SENSOR), ("depth", ob.OBSensorType.DEPTH_SENSOR)):
        print(f"{label}_profiles:")
        for profile in stream_profiles(pipeline, sensor):
            print(f"  - {profile_summary(profile)['description']}")


def profile_score(profile: Any, *, target_size: tuple[int, int], preferred_formats: tuple[str, ...]) -> tuple[int, int, int, int, int]:
    size = (profile_int(profile, "get_width"), profile_int(profile, "get_height"))
    fmt = profile_format(profile)
    resolution_score = 0 if size == target_size else 10
    format_score = preferred_formats.index(fmt) if fmt in preferred_formats else 10
    fps = profile_int(profile, "get_fps")
    fps_score = 0 if fps == PREFERRED_FPS else abs(fps - PREFERRED_FPS) + 1 if fps > 0 else 100
    return resolution_score, format_score, fps_score, -size[0], -size[1]


def select_color_profile(profiles: list[Any]) -> Any:
    return sorted(profiles, key=lambda profile: profile_score(profile, target_size=COLOR_SIZE, preferred_formats=("RGB", "BGR", "MJPG", "YUYV")))[0]


def select_depth_profile(profiles: list[Any]) -> Any:
    return sorted(profiles, key=lambda profile: profile_score(profile, target_size=DEPTH_SIZE, preferred_formats=("Y16", "Z16")))[0]


def enable_stream(config: Any, profile: Any) -> None:
    config.enable_stream(profile)


def config_hardware_d2c(pipeline: Any, ob: Any) -> tuple[Any | None, Any | None, Any | None, list[str]]:
    warnings: list[str] = []
    color_profiles = sorted(
        stream_profiles(pipeline, ob.OBSensorType.COLOR_SENSOR),
        key=lambda profile: profile_score(profile, target_size=COLOR_SIZE, preferred_formats=("RGB",)),
    )
    for color_profile in color_profiles:
        if profile_format(color_profile) != "RGB":
            continue
        try:
            depth_list = pipeline.get_d2c_depth_profile_list(color_profile, ob.OBAlignMode.HW_MODE)
        except Exception as exc:
            warnings.append(f"hardware_d2c_profile_query_failed:{exc}")
            continue
        depth_profiles = []
        for index in range(int(depth_list.get_count())):
            profile = depth_list.get_stream_profile_by_index(index)
            if hasattr(profile, "as_video_stream_profile"):
                profile = profile.as_video_stream_profile()
            depth_profiles.append(profile)
        if not depth_profiles:
            continue
        depth_profile = sorted(depth_profiles, key=lambda profile: profile_score(profile, target_size=COLOR_SIZE, preferred_formats=("Y16", "Z16")))[0]
        config = ob.Config()
        enable_stream(config, depth_profile)
        enable_stream(config, color_profile)
        config.set_align_mode(ob.OBAlignMode.HW_MODE)
        set_full_frame_mode(config, ob)
        return config, color_profile, depth_profile, warnings
    warnings.append("hardware_d2c_not_supported_for_available_rgb_profiles")
    return None, None, None, warnings


def config_software_d2c(pipeline: Any, ob: Any) -> tuple[Any, Any, Any]:
    config = ob.Config()
    color_profile = select_color_profile(stream_profiles(pipeline, ob.OBSensorType.COLOR_SENSOR))
    depth_profile = select_depth_profile(stream_profiles(pipeline, ob.OBSensorType.DEPTH_SENSOR))
    enable_stream(config, color_profile)
    enable_stream(config, depth_profile)
    set_full_frame_mode(config, ob)
    return config, color_profile, depth_profile


def set_full_frame_mode(config: Any, ob: Any) -> None:
    mode = getattr(getattr(ob, "OBFrameAggregateOutputMode", None), "FULL_FRAME_REQUIRE", None)
    method = getattr(config, "set_frame_aggregate_output_mode", None)
    if callable(method) and mode is not None:
        method(mode)


def get_frame(frames: Any, name: str) -> Any | None:
    method = getattr(frames, name, None)
    return method() if callable(method) else None


def frame_bytes(frame: Any) -> bytes:
    data = frame.get_data()
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    return memoryview(data).tobytes()


def frame_to_rgb(frame: Any, ob: Any) -> np.ndarray:
    width = int(frame.get_width())
    height = int(frame.get_height())
    fmt = enum_name(frame.get_format())
    data = np.frombuffer(frame_bytes(frame), dtype=np.uint8)
    if fmt == "RGB":
        return data.reshape((height, width, 3)).copy()
    if fmt == "BGR":
        return data.reshape((height, width, 3))[:, :, ::-1].copy()
    if fmt == "MJPG":
        import cv2

        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("Could not decode MJPG color frame")
        return bgr[:, :, ::-1].copy()
    convert_cls = getattr(ob, "FormatConvertFilter", None)
    convert_format = getattr(getattr(ob, "OBConvertFormat", None), f"{fmt}_TO_RGB888", None)
    if convert_cls is not None and convert_format is not None:
        converter = convert_cls()
        converter.set_format_convert_format(convert_format)
        rgb_frame = converter.process(frame)
        return frame_to_rgb(rgb_frame, ob)
    raise RuntimeError(f"Unsupported color frame format for experiment: {fmt}")


def depth_frame_to_raw(frame: Any) -> np.ndarray:
    width = int(frame.get_width())
    height = int(frame.get_height())
    data = np.frombuffer(frame_bytes(frame), dtype=np.uint16)
    expected = width * height
    if data.size < expected:
        raise RuntimeError(f"Depth frame has {data.size} uint16 values, expected {expected}")
    return data[:expected].reshape((height, width)).copy()


def depth_scale(frame: Any) -> float:
    for name in ("get_depth_scale", "get_value_scale", "depth_scale"):
        value = getattr(frame, name, None)
        if callable(value):
            return float(value())
        if value is not None:
            return float(value)
    return 0.001


def frame_timestamp(frame: Any) -> float | None:
    value = call_optional(frame, ("get_timestamp", "get_system_timestamp", "timestamp"), None)
    return float(value) if value is not None else None


def intrinsic_to_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    result: dict[str, Any] = {}
    for name in ("width", "height", "fx", "fy", "cx", "cy"):
        attr = getattr(value, name, None)
        if attr is not None:
            result[name] = float(attr)
    return result or None


def distortion_to_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    result: dict[str, Any] = {}
    for name in ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2"):
        attr = getattr(value, name, None)
        if attr is not None:
            result[name] = float(attr)
    return result or None


def extrinsic_to_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    rot = getattr(value, "rot", None)
    transform = getattr(value, "transform", None)
    result: dict[str, Any] = {}
    if rot is not None:
        result["rotation_row_major_3x3"] = np.asarray(rot, dtype=float).reshape((3, 3)).tolist()
    if transform is not None:
        result["translation_mm"] = np.asarray(transform, dtype=float).reshape((3,)).tolist()
    return result or None


def device_summary(pipeline: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    try:
        device = pipeline.get_device()
        info = device.get_device_info()
    except Exception:
        return values
    for method_name in ("get_name", "get_serial_number", "get_firmware_version", "get_connection_type", "get_pid", "get_vid"):
        method = getattr(info, method_name, None)
        if callable(method):
            try:
                values[method_name.removeprefix("get_")] = method()
            except Exception:
                pass
    return values


def depth_visualization(depth: np.ndarray) -> np.ndarray:
    preview = np.zeros(depth.shape, dtype=np.uint8)
    valid = depth > 0
    if np.any(valid):
        values = depth[valid].astype(np.float32)
        lo = float(np.percentile(values, 2))
        hi = float(np.percentile(values, 98))
        if hi <= lo:
            lo = float(values.min())
            hi = float(values.max())
        if hi > lo:
            preview[valid] = np.asarray((np.clip(depth.astype(np.float32), lo, hi)[valid] - lo) / (hi - lo) * 255, dtype=np.uint8)
    return preview


def overlay_valid_depth(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    valid = depth > 0
    green = np.array([0, 255, 80], dtype=np.uint8)
    out[valid] = np.asarray(out[valid] * 0.55 + green * 0.45, dtype=np.uint8)
    return out


def overlay_depth_on_rgb(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    preview = depth_visualization(depth)
    color = np.zeros((*preview.shape, 3), dtype=np.uint8)
    color[:, :, 0] = preview
    color[:, :, 1] = np.asarray(preview * 0.35, dtype=np.uint8)
    edges = np.zeros_like(preview, dtype=bool)
    edges[:, 1:] |= np.abs(preview[:, 1:].astype(int) - preview[:, :-1].astype(int)) > 12
    edges[1:, :] |= np.abs(preview[1:, :].astype(int) - preview[:-1, :].astype(int)) > 12
    out = np.asarray(rgb * 0.70 + color * 0.30, dtype=np.uint8)
    out[edges & (depth > 0)] = np.array([255, 240, 40], dtype=np.uint8)
    return out


def save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="PNG")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_ascii_ply(path: Path, points: list[tuple[float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("end_header\n")
        for x, y, z in points:
            file.write(f"{x} {y} {z}\n")


def object_cloud_from_aligned_depth(mask_path: Path, aligned_depth: np.ndarray, color_intrinsic: dict[str, Any], depth_scale_value: float, output_path: Path) -> int:
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    if mask.shape != aligned_depth.shape:
        raise RuntimeError(f"Mask shape {mask.shape} must equal aligned depth shape {aligned_depth.shape}; no resize is allowed")
    fx = float(color_intrinsic["fx"])
    fy = float(color_intrinsic["fy"])
    cx = float(color_intrinsic["cx"])
    cy = float(color_intrinsic["cy"])
    rows, cols = np.where(mask & (aligned_depth > 0))
    z = aligned_depth[rows, cols].astype(float) * depth_scale_value
    x = (cols.astype(float) - cx) * z / fx
    y = (rows.astype(float) - cy) * z / fy
    points = [(float(px), float(py), float(pz)) for px, py, pz in zip(x, y, z, strict=True)]
    write_ascii_ply(output_path, points)
    return len(points)


def capture(args: argparse.Namespace) -> int:
    ob, sdk_package, sdk_version = sdk_import()
    pipeline = ob.Pipeline()
    warnings: list[str] = []

    if args.list_profiles:
        print(f"sdk_package: {sdk_package}")
        print(f"sdk_version: {sdk_version}")
        print_profiles(pipeline, ob)
        return 0

    selected_api = "hardware_d2c"
    if args.force_sw:
        config, color_profile, depth_profile = config_software_d2c(pipeline, ob)
        selected_api = "software_d2c_align_filter"
        alignment_method = "software_d2c"
    else:
        config, color_profile, depth_profile, hw_warnings = config_hardware_d2c(pipeline, ob)
        warnings.extend(hw_warnings)
        if config is None or color_profile is None or depth_profile is None:
            config, color_profile, depth_profile = config_software_d2c(pipeline, ob)
            selected_api = "software_d2c_align_filter"
            alignment_method = "software_d2c"
            warnings.append("hardware_d2c_unavailable_used_official_software_align_filter")
        else:
            alignment_method = "hardware_d2c"

    align_filter = None
    if alignment_method == "software_d2c":
        align_filter = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)

    try:
        try:
            pipeline.enable_frame_sync()
        except Exception as exc:
            warnings.append(f"frame_sync_enable_warning:{exc}")
        pipeline.start(config)

        frames = None
        raw_color_frame = None
        raw_depth_frame = None
        aligned_depth_frame = None
        for index in range(max(1, args.warmup_frames) + 1):
            frames = pipeline.wait_for_frames(int(args.timeout_ms))
            if frames is None:
                continue
            raw_color_frame = get_frame(frames, "get_color_frame")
            raw_depth_frame = get_frame(frames, "get_depth_frame")
            if raw_color_frame is None or raw_depth_frame is None:
                continue
            if alignment_method == "software_d2c":
                aligned_frames = align_filter.process(frames)
                if aligned_frames is None:
                    continue
                aligned_depth_frame = get_frame(aligned_frames, "get_depth_frame")
            else:
                aligned_depth_frame = raw_depth_frame
            if index >= args.warmup_frames and aligned_depth_frame is not None:
                break

        if raw_color_frame is None or raw_depth_frame is None or aligned_depth_frame is None:
            raise RuntimeError("Timed out waiting for synchronized color/depth/aligned-depth frames")

        rgb = frame_to_rgb(raw_color_frame, ob)
        raw_depth = depth_frame_to_raw(raw_depth_frame)
        aligned_depth = depth_frame_to_raw(aligned_depth_frame)
        scale = depth_scale(aligned_depth_frame)

        if rgb.shape != (720, 1280, 3):
            raise RuntimeError(f"RGB shape assertion failed: expected (720, 1280, 3), got {rgb.shape}")
        if aligned_depth.shape != rgb.shape[:2]:
            raise RuntimeError(f"Aligned depth shape assertion failed: expected {rgb.shape[:2]}, got {aligned_depth.shape}")

        cam_param = pipeline.get_camera_param()
        color_intrinsic_obj = getattr(cam_param, "rgb_intrinsic", None)
        depth_intrinsic_obj = getattr(cam_param, "depth_intrinsic", None)
        color_intrinsic = intrinsic_to_dict(color_intrinsic_obj)
        depth_intrinsic = intrinsic_to_dict(depth_intrinsic_obj)
        if color_intrinsic is None:
            color_intrinsic = intrinsic_to_dict(color_profile.get_intrinsic())
        if depth_intrinsic is None:
            depth_intrinsic = intrinsic_to_dict(depth_profile.get_intrinsic())

        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "depth_raw.npy", raw_depth)
        np.save(output_dir / "depth_aligned_to_rgb.npy", aligned_depth)
        save_png(output_dir / "rgb.png", rgb)
        save_png(output_dir / "depth_raw_visualization.png", depth_visualization(raw_depth))
        save_png(output_dir / "depth_aligned_visualization.png", depth_visualization(aligned_depth))
        save_png(output_dir / "aligned_depth_overlay_rgb.png", overlay_depth_on_rgb(rgb, aligned_depth))
        save_png(output_dir / "valid_aligned_depth_overlay_rgb.png", overlay_valid_depth(rgb, aligned_depth))

        raw_valid = int(np.count_nonzero(raw_depth > 0))
        aligned_valid = int(np.count_nonzero(aligned_depth > 0))
        rgb_ts = frame_timestamp(raw_color_frame)
        raw_depth_ts = frame_timestamp(raw_depth_frame)
        timestamp_diff = abs(rgb_ts - raw_depth_ts) if rgb_ts is not None and raw_depth_ts is not None else None

        object_cloud_count = None
        if args.mask is not None:
            if color_intrinsic is None:
                raise RuntimeError("Cannot build object cloud because color intrinsics are unavailable")
            object_cloud_count = object_cloud_from_aligned_depth(
                args.mask,
                aligned_depth,
                color_intrinsic,
                scale,
                output_dir / "object_cloud_from_aligned_depth.ply",
            )

        calibration = {
            "device": device_summary(pipeline),
            "sdk_package_name": sdk_package,
            "sdk_version": sdk_version,
            "color_stream": profile_summary(color_profile),
            "raw_depth_stream": profile_summary(depth_profile),
            "depth_scale": scale,
            "depth_scale_units": "meters_per_unit according to pyorbbecsdk get_depth_scale()",
            "rgb_intrinsics": color_intrinsic,
            "raw_depth_intrinsics": depth_intrinsic,
            "depth_to_color_extrinsics": extrinsic_to_dict(getattr(cam_param, "transform", None)),
            "rgb_distortion": distortion_to_dict(getattr(cam_param, "rgb_distortion", None)),
            "raw_depth_distortion": distortion_to_dict(getattr(cam_param, "depth_distortion", None)),
            "alignment_method": alignment_method,
            "selected_sdk_alignment_api": selected_api,
            "aligned_depth_width": int(aligned_depth.shape[1]),
            "aligned_depth_height": int(aligned_depth.shape[0]),
            "coordinate_frame_used_by_aligned_depth": "color/RGB image plane; metric z values from Orbbec D2C aligned depth",
            "timestamps": {
                "rgb": rgb_ts,
                "raw_depth": raw_depth_ts,
                "aligned_depth": frame_timestamp(aligned_depth_frame),
            },
        }
        debug_payload = {
            "rgb_shape": list(rgb.shape),
            "raw_depth_shape": list(raw_depth.shape),
            "aligned_depth_shape": list(aligned_depth.shape),
            "rgb_timestamp": rgb_ts,
            "depth_timestamp": raw_depth_ts,
            "timestamp_difference": timestamp_diff,
            "raw_valid_depth_count": raw_valid,
            "aligned_valid_depth_count": aligned_valid,
            "invalid_aligned_depth_count": int(aligned_depth.size - aligned_valid),
            "aligned_valid_depth_percentage": float(aligned_valid / aligned_depth.size * 100.0),
            "minimum_valid_aligned_depth": int(aligned_depth[aligned_depth > 0].min()) if aligned_valid else None,
            "maximum_valid_aligned_depth": int(aligned_depth[aligned_depth > 0].max()) if aligned_valid else None,
            "selected_sdk_alignment_api": selected_api,
            "alignment_method": alignment_method,
            "hardware_alignment_used": alignment_method == "hardware_d2c",
            "sdk_warnings_or_fallback_decisions": warnings,
            "object_cloud_from_aligned_depth_point_count": object_cloud_count,
        }
        write_json(output_dir / "calibration.json", calibration)
        write_json(output_dir / "capture_debug.json", debug_payload)
        print(json.dumps(debug_payload, indent=2))
        return 0
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass


def main() -> int:
    try:
        return capture(parse_args())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
