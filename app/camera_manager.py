"""Shared camera manager for live preview and capture snapshots."""

from __future__ import annotations

import threading
import time
from typing import Any

from capture.camera import CameraInterface, CaptureFrame
from capture.orbbec_camera import create_orbbec_camera


class CameraManager(CameraInterface):
    """Keep one camera instance alive and cache the latest RGB-D frame."""

    def __init__(self, *, interval_seconds: float = 0.08, stale_seconds: float = 2.0) -> None:
        self.interval_seconds = interval_seconds
        self.stale_seconds = stale_seconds
        self._camera: Any = None
        self._latest_frame: CaptureFrame | None = None
        self._latest_at = 0.0
        self._last_error: str | None = None
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background capture loop if it is not already running."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._capture_loop, name="dimscan-camera", daemon=True)
            self._thread.start()

    def _ensure_camera(self) -> Any:
        if self._camera is None:
            self._camera = create_orbbec_camera()
        return self._camera

    def _capture_once_locked(self) -> CaptureFrame:
        frame = self._ensure_camera().capture_frame()
        self._latest_frame = frame
        self._latest_at = time.monotonic()
        self._last_error = None
        return frame

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    self._capture_once_locked()
            except Exception as exc:
                self._last_error = str(exc)
                self._close_camera_locked()
                time.sleep(max(0.5, self.interval_seconds))
            else:
                time.sleep(self.interval_seconds)

    def _close_camera_locked(self) -> None:
        camera = self._camera
        self._camera = None
        if camera is None:
            return
        for method_name in ("close", "stop"):
            method = getattr(camera, method_name, None)
            if callable(method):
                try:
                    method()
                finally:
                    return

    def latest_frame(self) -> CaptureFrame:
        """Return the latest cached frame, waiting briefly for startup."""
        self.start()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest_frame is not None:
                    return self._latest_frame
                last_error = self._last_error
            if last_error:
                raise RuntimeError(last_error)
            time.sleep(0.05)
        with self._lock:
            if self._last_error:
                raise RuntimeError(self._last_error)
        raise RuntimeError("Camera frame cache is not ready yet.")

    def capture_frame(self) -> CaptureFrame:
        """Return a fresh-enough cached RGB-D frame for the capture pipeline."""
        self.start()
        with self._lock:
            if self._latest_frame is not None and time.monotonic() - self._latest_at <= self.stale_seconds:
                return self._latest_frame
            return self._capture_once_locked()

    def stop(self) -> None:
        """Stop the background loop and close the shared camera."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        with self._lock:
            self._close_camera_locked()
            self._thread = None


_manager: CameraManager | None = None
_manager_lock = threading.Lock()


def get_camera_manager() -> CameraManager:
    """Return the process-wide shared camera manager."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = CameraManager()
        return _manager
