"""Import-safe camera preview helpers for the DimScan web console."""

from __future__ import annotations

import io
import time
from typing import Any


JPEG_MIME = "image/jpeg"
PNG_MIME = "image/png"


def _image_mime_from_bytes(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return JPEG_MIME
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return PNG_MIME
    return None


def _rgb_dimensions(frame: Any) -> tuple[int, int]:
    metadata = getattr(frame, "metadata", {}) or {}
    width = metadata.get("rgb_width") or metadata.get("color_width") or metadata.get("width")
    height = metadata.get("rgb_height") or metadata.get("color_height") or metadata.get("height")
    return int(width or 0), int(height or 0)


def _raw_rgb_to_jpeg(data: bytes, width: int, height: int) -> bytes:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("PIL is required to convert raw RGB frames to JPEG.") from exc

    if width <= 0 or height <= 0:
        raise RuntimeError("Raw RGB frame conversion requires width and height metadata.")

    expected_rgb = width * height * 3
    expected_rgba = width * height * 4
    if len(data) == expected_rgba:
        mode = "RGBA"
        usable = data[:expected_rgba]
    elif len(data) >= expected_rgb:
        mode = "RGB"
        usable = data[:expected_rgb]
    else:
        raise RuntimeError("Raw RGB frame data is smaller than its reported dimensions.")

    image = Image.frombytes(mode, (width, height), usable)
    if image.mode != "RGB":
        image = image.convert("RGB")

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=85)
    return output.getvalue()


def encode_rgb_frame_to_jpeg_bytes(frame: Any) -> bytes:
    """Return JPEG bytes for a captured RGB frame."""
    data = getattr(frame, "rgb_bytes", None)
    if isinstance(data, bytearray):
        data = bytes(data)

    if isinstance(data, bytes):
        mime = _image_mime_from_bytes(data)
        if mime == JPEG_MIME:
            return data
        if mime == PNG_MIME:
            try:
                from PIL import Image
            except ImportError as exc:
                raise RuntimeError("PIL is required to convert PNG frames to MJPEG.") from exc

            image = Image.open(io.BytesIO(data)).convert("RGB")
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=85)
            return output.getvalue()
        width, height = _rgb_dimensions(frame)
        return _raw_rgb_to_jpeg(data, width, height)

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("PIL is required to convert RGB arrays to JPEG.") from exc

    try:
        image = Image.fromarray(data)
    except Exception as exc:
        raise RuntimeError("RGB frame could not be converted to an image.") from exc

    output = io.BytesIO()
    image.convert("RGB").save(output, format="JPEG", quality=85)
    return output.getvalue()


def encode_rgb_frame_to_response_bytes(frame: Any) -> tuple[bytes, str]:
    """Return displayable image bytes and MIME type for a captured RGB frame."""
    data = getattr(frame, "rgb_bytes", None)
    if isinstance(data, bytearray):
        data = bytes(data)
    if isinstance(data, bytes):
        mime = _image_mime_from_bytes(data)
        if mime is not None:
            return data, mime
    return encode_rgb_frame_to_jpeg_bytes(frame), JPEG_MIME


def capture_snapshot_bytes() -> tuple[bytes, str]:
    """Return the latest shared camera frame as displayable image bytes."""
    from app.camera_manager import get_camera_manager

    frame = get_camera_manager().capture_frame()
    return encode_rgb_frame_to_response_bytes(frame)


def _error_frame_bytes(message: str) -> bytes | None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    image = Image.new("RGB", (960, 540), color=(250, 252, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 960, 540), outline=(185, 196, 210), width=4)
    draw.text((32, 32), "Camera preview unavailable", fill=(31, 41, 55))
    draw.text((32, 68), message[:120], fill=(127, 29, 29))

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=85)
    return output.getvalue()


def _mjpeg_part(image_bytes: bytes) -> bytes:
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n"
        + image_bytes
        + b"\r\n"
    )


def generate_mjpeg_frames(delay_seconds: float = 0.1) -> Any:
    """Yield MJPEG stream parts from the shared camera frame cache."""
    from app.camera_manager import get_camera_manager

    manager = get_camera_manager()
    manager.start()
    while True:
        try:
            frame = manager.latest_frame()
            yield _mjpeg_part(encode_rgb_frame_to_jpeg_bytes(frame))
        except Exception as exc:
            error_frame = _error_frame_bytes(str(exc))
            if error_frame is not None:
                yield _mjpeg_part(error_frame)
        time.sleep(delay_seconds)
