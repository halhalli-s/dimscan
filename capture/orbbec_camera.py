"""Real Orbbec camera adapter for DimScan data collection."""

from __future__ import annotations

import logging
from typing import Any, Callable

from capture.camera import CameraInterface, CaptureFrame
from pipeline.profiling import timed_stage


logger = logging.getLogger(__name__)
SDK_ERROR_MESSAGE = (
    "pyorbbecsdk is not installed or not available. Install/configure Orbbec SDK "
    "before using OrbbecCamera."
)
COLOR_FORMATS = ("RGB", "BGR", "MJPG")
COLOR_RESOLUTIONS = ((1280, 720), (640, 480))
DEPTH_RESOLUTIONS = ((1280, 800), (640, 480), (1280, 720))
PREFERRED_FPS = 30


class OrbbecCamera(CameraInterface):
    """Orbbec camera adapter with lazy pyorbbecsdk import."""

    def __init__(self, **kwargs: Any) -> None:
        try:
            import pyorbbecsdk as ob
        except ImportError as exc:
            raise RuntimeError(SDK_ERROR_MESSAGE) from exc

        self._ob = ob
        self.options = kwargs
        self._pipeline = None
        self._config = None
        self._align_filter = None
        self._alignment_method: str | None = None
        self._selected_sdk_alignment_api: str | None = None
        self._alignment_warnings: list[str] = []
        self._profile_summary: dict[str, str] = {}

    def available_profile_summary(self) -> dict[str, list[str]]:
        """Return readable color/depth profile descriptions without starting streams."""
        ob = self._ob
        if not hasattr(ob, "Pipeline"):
            raise RuntimeError("pyorbbecsdk does not expose Pipeline.")

        pipeline = ob.Pipeline()
        return {
            "color": [
                self._describe_profile(profile)
                for profile in self._video_profiles_for_sensor(
                    pipeline,
                    self._enum_value(ob.OBSensorType, "COLOR_SENSOR"),
                    "color",
                )
            ],
            "depth": [
                self._describe_profile(profile)
                for profile in self._video_profiles_for_sensor(
                    pipeline,
                    self._enum_value(ob.OBSensorType, "DEPTH_SENSOR"),
                    "depth",
                )
            ],
        }

    def capture_frame(self) -> CaptureFrame:
        """Capture one RGB-D frame from an Orbbec camera.

        The exact SDK object names can vary by installed pyorbbecsdk version. This method
        keeps the integration isolated and raises clear runtime errors if the local SDK
        API differs from the expected pipeline/profile/frame shape.
        """
        ob = self._ob

        try:
            pipeline = self._ensure_started(ob)
            frames = self._wait_for_rgbd_frames(pipeline)
            color_frame = self._get_frame(frames, ("get_color_frame", "color_frame"))
            depth_frame = self._get_frame(frames, ("get_depth_frame", "depth_frame"))
            with timed_stage("d2c_alignment_s"):
                aligned_depth_frame = self._aligned_depth_frame(frames, depth_frame)

            rgb_bytes = self._frame_to_bytes(color_frame)
            rgb_width = self._frame_int(color_frame, ("get_width", "width"))
            rgb_height = self._frame_int(color_frame, ("get_height", "height"))
            rgb_format = self._frame_format_name(color_frame)
            depth_scale = self._depth_scale(depth_frame)
            aligned_depth_scale = self._depth_scale(aligned_depth_frame)
            depth_scale_m_per_unit, depth_scale_units = self._depth_scale_m_per_unit(depth_scale)
            aligned_depth_scale_m_per_unit, aligned_depth_scale_units = self._depth_scale_m_per_unit(aligned_depth_scale)
            depth_values = self._depth_frame_to_raw_values(depth_frame)
            depth_aligned_to_rgb_values = self._depth_frame_to_raw_values(aligned_depth_frame)
            raw_height = len(depth_values)
            raw_width = len(depth_values[0]) if depth_values else 0
            aligned_height = len(depth_aligned_to_rgb_values)
            aligned_width = len(depth_aligned_to_rgb_values[0]) if depth_aligned_to_rgb_values else 0
            if aligned_height != rgb_height or aligned_width != rgb_width:
                raise RuntimeError(
                    "Aligned depth frame must match RGB grid; "
                    f"rgb={rgb_height}x{rgb_width}, aligned_depth={aligned_height}x{aligned_width}."
                )
            if raw_height == aligned_height and raw_width == aligned_width:
                raise RuntimeError(
                    "Raw depth frame has the same shape as aligned depth; refusing to label an aligned frame as raw. "
                    f"raw_depth={raw_height}x{raw_width}, aligned_depth={aligned_height}x{aligned_width}."
                )
            if not self._has_valid_depth(depth_aligned_to_rgb_values):
                raise RuntimeError("Aligned depth frame contains no valid depth pixels.")
            point_cloud_points = self._try_point_cloud_points(ob, depth_frame)
            rgb_ts = self._frame_timestamp(color_frame)
            raw_depth_ts = self._frame_timestamp(depth_frame)
            aligned_depth_ts = self._frame_timestamp(aligned_depth_frame)
            timestamp_diff = (
                abs(rgb_ts - aligned_depth_ts)
                if rgb_ts is not None and aligned_depth_ts is not None
                else None
            )
            metadata = {
                "camera_type": "orbbec",
                "sdk_module": "pyorbbecsdk",
                "sdk_device": self._device_summary(pipeline),
                "rgb_width": rgb_width,
                "rgb_height": rgb_height,
                "rgb_format": rgb_format,
                "depth_height": raw_height,
                "depth_width": raw_width,
                "aligned_depth_height": aligned_height,
                "aligned_depth_width": aligned_width,
                "sdk_reported_depth_scale": depth_scale,
                "sdk_reported_depth_scale_units": depth_scale_units,
                "sdk_depth_scale_m_per_unit": depth_scale_m_per_unit,
                "aligned_sdk_reported_depth_scale": aligned_depth_scale,
                "aligned_sdk_reported_depth_scale_units": aligned_depth_scale_units,
                "aligned_sdk_depth_scale_m_per_unit": aligned_depth_scale_m_per_unit,
                "sdk_depth_scale": depth_scale_m_per_unit,
                "aligned_sdk_depth_scale": aligned_depth_scale_m_per_unit,
                "depth_scale": depth_scale_m_per_unit,
                "saved_depth_units": "raw_sdk_units",
                "depth_scale_applied_to_saved_depth": False,
                "object_cloud_depth_scale": aligned_depth_scale_m_per_unit,
                "scale_applied_during_cloud_generation": aligned_depth_scale_m_per_unit,
                "object_cloud_depth_units": "meters",
                "depth_intrinsics": self._frame_intrinsics(depth_frame),
                "rgb_intrinsics": self._frame_intrinsics(color_frame),
                "raw_depth_intrinsics": self._frame_intrinsics(depth_frame),
                "aligned_depth_intrinsics": self._frame_intrinsics(aligned_depth_frame),
                "point_count": len(point_cloud_points),
                "color_profile": self._profile_summary.get("color"),
                "depth_profile": self._profile_summary.get("depth"),
                "d2c_mode": self._alignment_method,
                "alignment_method": self._alignment_method,
                "selected_sdk_alignment_api": self._selected_sdk_alignment_api,
                "hardware_alignment_used": False,
                "sdk_warnings_or_fallback_decisions": list(self._alignment_warnings),
                "timestamps": {
                    "rgb": rgb_ts,
                    "raw_depth": raw_depth_ts,
                    "aligned_depth": aligned_depth_ts,
                },
                "rgb_aligned_depth_timestamp_difference": timestamp_diff,
                "notes": "Raw depth is preserved from the unaligned frameset; aligned depth is produced by Orbbec AlignFilter. Saved depth values are raw SDK units.",
            }
            logger.info(
                "Orbbec capture D2C mode=%s api=%s rgb=%sx%s raw_depth=%sx%s aligned_depth=%sx%s "
                "saved_depth_units=%s sdk_depth_scale=%s",
                self._alignment_method,
                self._selected_sdk_alignment_api,
                rgb_height,
                rgb_width,
                raw_height,
                raw_width,
                aligned_height,
                aligned_width,
                metadata["saved_depth_units"],
                depth_scale_m_per_unit,
            )

            return CaptureFrame(
                rgb_bytes=rgb_bytes,
                depth_values=depth_values,
                metadata=metadata,
                point_cloud_points=point_cloud_points,
                depth_aligned_to_rgb_values=depth_aligned_to_rgb_values,
            )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Failed to capture Orbbec frame: {exc}") from exc

    def _ensure_started(self, ob: Any) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        return self._create_pipeline(ob)

    def _create_pipeline(self, ob: Any) -> Any:
        if not hasattr(ob, "Pipeline"):
            raise RuntimeError("pyorbbecsdk does not expose Pipeline.")
        if not hasattr(ob, "Config"):
            raise RuntimeError("pyorbbecsdk does not expose Config.")

        pipeline = ob.Pipeline()
        config = ob.Config()

        self._alignment_warnings = []
        config, color_profile, depth_profile = self._software_d2c_config(pipeline, ob)
        self._alignment_method = "software_d2c"
        self._selected_sdk_alignment_api = "ob.AlignFilter.process"
        self._alignment_warnings.append("hardware_d2c_disabled_to_preserve_true_raw_depth")
        self._align_filter = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)

        self._set_full_frame_aggregation(ob, config)
        pipeline.start(config)
        self._enable_frame_sync(pipeline)

        self._pipeline = pipeline
        self._config = config
        self._profile_summary = {
            "color": self._describe_profile(color_profile),
            "depth": self._describe_profile(depth_profile),
        }
        return pipeline

    def _hardware_d2c_config(self, pipeline: Any, ob: Any) -> tuple[Any | None, Any | None, Any | None, list[str]]:
        warnings: list[str] = []
        color_profiles = sorted(
            self._video_profiles_for_sensor(
                pipeline,
                self._enum_value(ob.OBSensorType, "COLOR_SENSOR"),
                "color",
            ),
            key=self._color_profile_score,
        )
        for color_profile in color_profiles:
            if self._profile_format_name(color_profile) != "RGB":
                continue
            try:
                depth_list = pipeline.get_d2c_depth_profile_list(color_profile, ob.OBAlignMode.HW_MODE)
            except Exception as exc:
                warnings.append(f"hardware_d2c_profile_query_failed:{exc}")
                continue
            depth_profiles = self._profiles_from_list(depth_list)
            if not depth_profiles:
                continue
            depth_profile = self._best_profile(
                depth_profiles,
                lambda profile: self._d2c_depth_profile_score(profile, color_profile=color_profile),
                label="hardware_d2c_depth",
                all_profiles=depth_profiles,
            )
            config = ob.Config()
            self._enable_stream(config, depth_profile, "depth")
            self._enable_stream(config, color_profile, "color")
            config.set_align_mode(ob.OBAlignMode.HW_MODE)
            return config, color_profile, depth_profile, warnings
        warnings.append("hardware_d2c_not_supported_for_available_rgb_profiles")
        return None, None, None, warnings

    def _software_d2c_config(self, pipeline: Any, ob: Any) -> tuple[Any, Any, Any]:
        config = ob.Config()
        color_profiles = self._video_profiles_for_sensor(
            pipeline,
            self._enum_value(ob.OBSensorType, "COLOR_SENSOR"),
            "color",
        )
        depth_profiles = self._video_profiles_for_sensor(
            pipeline,
            self._enum_value(ob.OBSensorType, "DEPTH_SENSOR"),
            "depth",
        )
        color_profile = self._select_color_profile(color_profiles)
        depth_profile = self._select_depth_profile(depth_profiles, color_profile=color_profile)
        self._enable_stream(config, color_profile, "color")
        self._enable_stream(config, depth_profile, "depth")
        return config, color_profile, depth_profile

    def _set_full_frame_aggregation(self, ob: Any, config: Any) -> None:
        method = getattr(config, "set_frame_aggregate_output_mode", None)
        mode = getattr(getattr(ob, "OBFrameAggregateOutputMode", None), "FULL_FRAME_REQUIRE", None)
        if callable(method) and mode is not None:
            method(mode)

    def _enable_frame_sync(self, pipeline: Any) -> None:
        method = getattr(pipeline, "enable_frame_sync", None)
        if callable(method):
            try:
                method()
            except Exception:
                pass

    def _video_profiles_for_sensor(self, pipeline: Any, sensor_type: Any, label: str) -> list[Any]:
        if not hasattr(pipeline, "get_stream_profile_list"):
            raise RuntimeError("Orbbec pipeline does not expose stream profile lists.")

        try:
            profile_list = pipeline.get_stream_profile_list(sensor_type)
        except Exception as exc:
            raise RuntimeError(f"Could not query available {label} stream profiles: {exc}") from exc

        profiles = self._profiles_from_list(profile_list)
        if not profiles:
            raise RuntimeError(f"No available {label} stream profiles reported by Orbbec SDK.")
        return profiles

    def _profiles_from_list(self, profile_list: Any) -> list[Any]:
        profiles: list[Any] = []
        count = self._profile_list_count(profile_list)
        for index in range(count):
            profile = profile_list.get_stream_profile_by_index(index)
            if hasattr(profile, "as_video_stream_profile"):
                profile = profile.as_video_stream_profile()
            if self._profile_int(profile, "get_width") > 0 and self._profile_int(profile, "get_height") > 0:
                profiles.append(profile)
        return profiles

    def _profile_list_count(self, profile_list: Any) -> int:
        count = getattr(profile_list, "get_count", None)
        if callable(count):
            return int(count())
        raise RuntimeError("Orbbec stream profile list does not expose get_count().")

    def _select_color_profile(self, profiles: list[Any]) -> Any:
        supported = [
            profile for profile in profiles
            if self._profile_format_name(profile) in COLOR_FORMATS
        ]
        candidates = supported or profiles
        return self._best_profile(
            candidates,
            self._color_profile_score,
            label="color",
            all_profiles=profiles,
        )

    def _select_depth_profile(self, profiles: list[Any], *, color_profile: Any) -> Any:
        def score(profile: Any) -> tuple[int, int, int, int]:
            size = (
                self._profile_int(profile, "get_width"),
                self._profile_int(profile, "get_height"),
            )
            resolution_score = DEPTH_RESOLUTIONS.index(size) if size in DEPTH_RESOLUTIONS else 10
            return (
                resolution_score,
                self._fps_score(profile),
                -self._profile_int(profile, "get_width"),
                -self._profile_int(profile, "get_height"),
            )

        return self._best_profile(profiles, score, label="depth", all_profiles=profiles)

    def _d2c_depth_profile_score(self, profile: Any, *, color_profile: Any) -> tuple[int, int, int, int, int]:
        color_size = (
            self._profile_int(color_profile, "get_width"),
            self._profile_int(color_profile, "get_height"),
        )
        size = (
            self._profile_int(profile, "get_width"),
            self._profile_int(profile, "get_height"),
        )
        format_name = self._profile_format_name(profile)
        format_score = 0 if format_name in {"Y16", "Z16"} else 10
        return (
            0 if size == color_size else 10,
            format_score,
            self._fps_score(profile),
            -self._profile_int(profile, "get_width"),
            -self._profile_int(profile, "get_height"),
        )

    def _best_profile(
        self,
        profiles: list[Any],
        score: Callable[[Any], tuple[int, ...]],
        *,
        label: str,
        all_profiles: list[Any],
    ) -> Any:
        if not profiles:
            raise RuntimeError(
                f"Could not select {label} stream profile. Available {label} profiles: "
                f"{self._format_profile_list(all_profiles)}"
            )
        return sorted(profiles, key=score)[0]

    def _color_profile_score(self, profile: Any) -> tuple[int, int, int, int, int]:
        format_name = self._profile_format_name(profile)
        size = (
            self._profile_int(profile, "get_width"),
            self._profile_int(profile, "get_height"),
        )
        format_score = COLOR_FORMATS.index(format_name) if format_name in COLOR_FORMATS else 10
        resolution_score = COLOR_RESOLUTIONS.index(size) if size in COLOR_RESOLUTIONS else 10
        return (
            resolution_score,
            format_score,
            self._fps_score(profile),
            -self._profile_int(profile, "get_width"),
            -self._profile_int(profile, "get_height"),
        )

    def _fps_score(self, profile: Any) -> int:
        fps = self._profile_int(profile, "get_fps")
        if fps == PREFERRED_FPS:
            return 0
        if fps > 0:
            return abs(fps - PREFERRED_FPS) + 1
        return 100

    def _enable_stream(self, config: Any, profile: Any, label: str) -> None:
        for method_name in ("enable_stream", "enable_video_stream"):
            method = getattr(config, method_name, None)
            if callable(method):
                try:
                    method(profile)
                    return
                except TypeError:
                    continue
                except Exception as exc:
                    raise RuntimeError(
                        f"Could not enable {label} stream profile {self._describe_profile(profile)}: {exc}"
                    ) from exc
        raise RuntimeError("Orbbec Config does not support enabling video stream profiles.")

    def _enum_value(self, enum: Any, name: str) -> Any:
        value = getattr(enum, name, None)
        if value is None:
            raise RuntimeError(f"pyorbbecsdk enum is missing {name}.")
        return value

    def _profile_int(self, profile: Any, method_name: str) -> int:
        method = getattr(profile, method_name, None)
        if callable(method):
            return int(method())
        return 0

    def _profile_format_name(self, profile: Any) -> str:
        method = getattr(profile, "get_format", None)
        if not callable(method):
            return "UNKNOWN"
        return self._enum_name(method())

    def _frame_format_name(self, frame: Any) -> str:
        method = getattr(frame, "get_format", None)
        if not callable(method):
            return "UNKNOWN"
        return self._enum_name(method())

    def _enum_name(self, value: Any) -> str:
        name = getattr(value, "name", None)
        if callable(name):
            name = name()
        if isinstance(name, str):
            return name
        text = str(value)
        return text.rsplit(".", 1)[-1]

    def _describe_profile(self, profile: Any) -> str:
        return (
            f"{self._profile_int(profile, 'get_width')}x"
            f"{self._profile_int(profile, 'get_height')} "
            f"{self._profile_format_name(profile)} "
            f"{self._profile_int(profile, 'get_fps')}fps"
        )

    def _format_profile_list(self, profiles: list[Any]) -> str:
        return ", ".join(self._describe_profile(profile) for profile in profiles) or "none"

    def _wait_for_frames(self, pipeline: Any) -> Any:
        timeout_ms = int(self.options.get("timeout_ms", 5000))
        if hasattr(pipeline, "wait_for_frames"):
            return pipeline.wait_for_frames(timeout_ms)
        if hasattr(pipeline, "wait_for_frame"):
            return pipeline.wait_for_frame(timeout_ms)
        raise RuntimeError("Orbbec pipeline does not support frame waiting.")

    def _wait_for_rgbd_frames(self, pipeline: Any) -> Any:
        attempts = int(self.options.get("frame_attempts", 10))
        last_error = "no frames returned"

        for _ in range(max(1, attempts)):
            frames = self._wait_for_frames(pipeline)
            if frames is None:
                continue

            color_frame = self._maybe_get_frame(frames, ("get_color_frame", "color_frame"))
            depth_frame = self._maybe_get_frame(frames, ("get_depth_frame", "depth_frame"))
            if color_frame is not None and depth_frame is not None:
                return frames

            missing: list[str] = []
            if color_frame is None:
                missing.append("color")
            if depth_frame is None:
                missing.append("depth")
            last_error = f"missing {' and '.join(missing)} frame"

        raise RuntimeError(
            f"Timed out waiting for synchronized color/depth frames ({last_error}). "
            f"Selected color profile: {self._profile_summary.get('color', 'unknown')}; "
            f"selected depth profile: {self._profile_summary.get('depth', 'unknown')}."
        )

    def _get_frame(self, frames: Any, method_names: tuple[str, ...]) -> Any:
        frame = self._maybe_get_frame(frames, method_names)
        if frame is not None:
            return frame
        raise RuntimeError(f"Captured frame set is missing {method_names[0]}.")

    def _aligned_depth_frame(self, frames: Any, raw_depth_frame: Any) -> Any:
        if self._alignment_method == "software_d2c":
            if self._align_filter is None:
                raise RuntimeError("Software D2C selected but Orbbec AlignFilter is not initialized.")
            aligned_frames = self._align_filter.process(frames)
            if aligned_frames is None:
                raise RuntimeError("Orbbec AlignFilter returned no aligned frames.")
            return self._get_frame(aligned_frames, ("get_depth_frame", "depth_frame"))
        if self._alignment_method == "hardware_d2c":
            return raw_depth_frame
        raise RuntimeError("Orbbec D2C alignment method was not selected.")

    def _maybe_get_frame(self, frames: Any, method_names: tuple[str, ...]) -> Any:
        for method_name in method_names:
            method = getattr(frames, method_name, None)
            if callable(method):
                frame = method()
                if frame is not None:
                    return frame
            elif method is not None:
                return method
        return None

    def _frame_to_bytes(self, frame: Any) -> bytes:
        data = self._frame_data(frame)
        if isinstance(data, bytes):
            return data
        if isinstance(data, bytearray):
            return bytes(data)
        try:
            return memoryview(data).tobytes()
        except TypeError as exc:
            raise RuntimeError("Color frame data could not be converted to bytes.") from exc

    def _depth_frame_to_values(self, frame: Any) -> list[list[float]]:
        width = self._frame_int(frame, ("get_width", "width"))
        height = self._frame_int(frame, ("get_height", "height"))
        scale = self._depth_scale(frame)
        data = self._frame_data(frame)

        values = self._matrix_from_nested_data(data, scale=scale)
        if values:
            return values

        raw_bytes = self._bytes_from_data(data)
        bytes_per_value = 2
        expected_values = width * height
        if width <= 0 or height <= 0 or len(raw_bytes) < expected_values * bytes_per_value:
            raise RuntimeError("Depth frame dimensions/data are not usable.")

        rows: list[list[float]] = []
        for row_index in range(height):
            row: list[float] = []
            for col_index in range(width):
                offset = (row_index * width + col_index) * bytes_per_value
                raw_value = int.from_bytes(raw_bytes[offset : offset + bytes_per_value], "little")
                row.append(raw_value * scale)
            rows.append(row)
        return rows

    def _depth_frame_to_raw_values(self, frame: Any) -> list[list[float]]:
        width = self._frame_int(frame, ("get_width", "width"))
        height = self._frame_int(frame, ("get_height", "height"))
        data = self._frame_data(frame)

        values = self._matrix_from_nested_data(data, scale=1.0)
        if values:
            return values

        raw_bytes = self._bytes_from_data(data)
        bytes_per_value = 2
        expected_values = width * height
        if width <= 0 or height <= 0 or len(raw_bytes) < expected_values * bytes_per_value:
            raise RuntimeError("Depth frame dimensions/data are not usable.")

        rows: list[list[float]] = []
        for row_index in range(height):
            row: list[float] = []
            for col_index in range(width):
                offset = (row_index * width + col_index) * bytes_per_value
                raw_value = int.from_bytes(raw_bytes[offset : offset + bytes_per_value], "little")
                row.append(float(raw_value))
            rows.append(row)
        return rows

    def _has_valid_depth(self, depth_values: list[list[float]]) -> bool:
        for row in depth_values:
            for value in row:
                if value and value > 0:
                    return True
        return False

    def _try_point_cloud_points(self, ob: Any, depth_frame: Any) -> list[tuple[float, float, float]]:
        # TODO: Replace this best-effort path with calibrated point cloud generation.
        point_cloud_cls = getattr(ob, "PointCloudFilter", None) or getattr(ob, "PointCloud", None)
        if point_cloud_cls is None:
            return []

        try:
            point_cloud = point_cloud_cls()
            if hasattr(point_cloud, "process"):
                result = point_cloud.process(depth_frame)
            elif hasattr(point_cloud, "calculate"):
                result = point_cloud.calculate(depth_frame)
            else:
                return []
            return self._points_from_data(result)
        except Exception:
            return []

    def _points_from_data(self, data: Any) -> list[tuple[float, float, float]]:
        if hasattr(data, "get_data"):
            data = data.get_data()

        points: list[tuple[float, float, float]] = []
        try:
            for point in data:
                if len(point) >= 3:
                    points.append((float(point[0]), float(point[1]), float(point[2])))
        except TypeError:
            return []
        return points

    def _matrix_from_nested_data(self, data: Any, *, scale: float) -> list[list[float]]:
        if not hasattr(data, "__iter__") or isinstance(data, bytes | bytearray):
            return []

        rows: list[list[float]] = []
        try:
            for row in data:
                if isinstance(row, int | float):
                    return []
                rows.append([float(value) * scale for value in row])
        except TypeError:
            return []
        return rows

    def _frame_data(self, frame: Any) -> Any:
        for method_name in ("get_data", "data"):
            method = getattr(frame, method_name, None)
            if callable(method):
                return method()
            if method is not None:
                return method
        raise RuntimeError("Frame object does not expose data.")

    def _bytes_from_data(self, data: Any) -> bytes:
        if isinstance(data, bytes):
            return data
        if isinstance(data, bytearray):
            return bytes(data)
        try:
            return memoryview(data).tobytes()
        except TypeError as exc:
            raise RuntimeError("Frame data could not be converted to bytes.") from exc

    def _frame_int(self, frame: Any, names: tuple[str, ...]) -> int:
        for name in names:
            value = getattr(frame, name, None)
            if callable(value):
                return int(value())
            if value is not None:
                return int(value)
        return 0

    def _depth_scale(self, frame: Any) -> float:
        for name in ("get_depth_scale", "get_value_scale", "depth_scale"):
            value = getattr(frame, name, None)
            if callable(value):
                return float(value())
            if value is not None:
                return float(value)
        return 0.001

    def _depth_scale_m_per_unit(self, reported_scale: float) -> tuple[float, str]:
        if reported_scale <= 0 or not reported_scale < float("inf"):
            raise RuntimeError(f"Invalid SDK depth scale: {reported_scale!r}")
        if reported_scale >= 0.1:
            return reported_scale * 0.001, "millimeters_per_unit"
        return reported_scale, "meters_per_unit"

    def _frame_timestamp(self, frame: Any) -> float | None:
        for name in ("get_timestamp", "get_system_timestamp", "timestamp"):
            value = getattr(frame, name, None)
            if callable(value):
                try:
                    return float(value())
                except Exception:
                    continue
            if value is not None:
                return float(value)
        return None

    def _frame_intrinsics(self, frame: Any) -> dict[str, Any] | None:
        profile = None
        get_profile = getattr(frame, "get_stream_profile", None)
        if callable(get_profile):
            profile = get_profile()
        if profile is not None and hasattr(profile, "as_video_stream_profile"):
            profile = profile.as_video_stream_profile()
        get_intrinsic = getattr(profile, "get_intrinsic", None)
        if not callable(get_intrinsic):
            return None

        try:
            intrinsic = get_intrinsic()
        except Exception:
            return None

        values: dict[str, Any] = {}
        for name in ("fx", "fy", "cx", "cy", "width", "height"):
            value = getattr(intrinsic, name, None)
            if callable(value):
                value = value()
            if value is not None:
                values[name] = float(value)
        return values or None

    def _device_summary(self, pipeline: Any) -> dict[str, Any] | None:
        get_device = getattr(pipeline, "get_device", None)
        if not callable(get_device):
            return None
        try:
            device = get_device()
        except Exception:
            return None

        values: dict[str, Any] = {}
        info = getattr(device, "get_device_info", None)
        if callable(info):
            try:
                device = info()
            except Exception:
                pass

        for name in (
            "get_name",
            "get_pid",
            "get_vid",
            "get_serial_number",
            "get_firmware_version",
            "get_connection_type",
        ):
            method = getattr(device, name, None)
            if callable(method):
                try:
                    values[name.removeprefix("get_")] = method()
                except Exception:
                    pass
        return values or None

    def _stop_pipeline(self, pipeline: Any) -> None:
        if pipeline is not None and hasattr(pipeline, "stop"):
            pipeline.stop()

    def stop(self) -> None:
        """Stop the Orbbec pipeline if it has been started."""
        self._stop_pipeline(self._pipeline)
        self._pipeline = None
        self._config = None
        self._align_filter = None
        self._alignment_method = None
        self._selected_sdk_alignment_api = None
        self._alignment_warnings = []
        self._profile_summary = {}

    def close(self) -> None:
        """Close the Orbbec camera adapter."""
        self.stop()


def create_orbbec_camera(**kwargs: Any) -> OrbbecCamera:
    """Create an Orbbec camera adapter."""
    return OrbbecCamera(**kwargs)
