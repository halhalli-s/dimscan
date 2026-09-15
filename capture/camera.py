"""Camera interfaces and fake capture implementation for DimScan development."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CaptureFrame:
    """Simple RGB-D capture result container."""

    rgb_bytes: bytes
    depth_values: list[list[float]]
    metadata: dict[str, Any]
    point_cloud_points: list[tuple[float, float, float]] | None = None
    depth_aligned_to_rgb_values: list[list[float]] | None = None


class CameraInterface:
    """Minimal camera interface for pipeline callers."""

    def capture_frame(self) -> CaptureFrame:
        """Capture a frame from a camera implementation."""
        raise NotImplementedError


class FakeCamera(CameraInterface):
    """Fake camera that returns deterministic placeholder RGB-D data."""

    def __init__(
        self,
        *,
        width: int = 4,
        height: int = 3,
        depth_value: float = 1.0,
    ) -> None:
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ValueError("width must be a positive integer.")
        if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
            raise ValueError("height must be a positive integer.")

        depth = float(depth_value)
        if depth <= 0:
            raise ValueError("depth_value must be positive.")

        self.width = width
        self.height = height
        self.depth_value = depth

    def capture_frame(self) -> CaptureFrame:
        """Return a deterministic fake capture frame."""
        return CaptureFrame(
            rgb_bytes=b"DIMSCAN_FAKE_RGB",
            depth_values=[
                [self.depth_value for _ in range(self.width)]
                for _ in range(self.height)
            ],
            metadata={
                "camera_type": "fake",
                "width": self.width,
                "height": self.height,
                "depth_units": "meters",
            },
            point_cloud_points=None,
        )


def create_camera(camera_type: str = "fake", **kwargs: Any) -> CameraInterface:
    """Create a camera implementation by type."""
    if camera_type == "fake":
        return FakeCamera(**kwargs)
    raise ValueError(f"Unsupported camera type: {camera_type!r}")
