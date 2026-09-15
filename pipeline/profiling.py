"""Temporary request-scoped live profiling helpers for capture latency."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator


_ACTIVE_PROFILE: ContextVar[dict[str, Any] | None] = ContextVar("dimscan_live_profile", default=None)
_STAGE_SUM_EXCLUDED_STAGES = {
    "raw_depth_array_prepare_s",
    "aligned_depth_array_prepare_s",
    "rgb_png_write_s",
    "aligned_depth_npy_write_s",
    "raw_depth_npy_write_s",
    "depth_preview_prepare_s",
    "depth_preview_png_write_s",
    "table_plane_cache_lookup_s",
    "table_plane_cache_validation_s",
    "table_plane_cached_apply_s",
    "table_plane_ransac_fallback_s",
    "table_plane_cache_hit",
    "table_plane_cache_replaced",
    "real_capture_process_job_call_s",
    "yoloe_prompt_cache_lookup_s",
    "yoloe_prompt_cache_build_s",
    "yoloe_predict_call_s",
}


def begin_profile(*, endpoint: str, job_id: str | None, debug_mode: bool) -> Any:
    profile: dict[str, Any] = {
        "endpoint": endpoint,
        "job_id": job_id,
        "debug_mode": bool(debug_mode),
        "started": time.perf_counter(),
        "stages": {},
        "notes": [],
    }
    return _ACTIVE_PROFILE.set(profile)


def end_profile(token: Any, *, ok: bool, status_code: int | None = None, error: str | None = None) -> None:
    profile = _ACTIVE_PROFILE.get()
    try:
        if isinstance(profile, dict):
            total_s = time.perf_counter() - float(profile["started"])
            stages = profile.get("stages") if isinstance(profile.get("stages"), dict) else {}
            stage_sum_s = float(
                sum(
                    float(value)
                    for key, value in stages.items()
                    if str(key) not in _STAGE_SUM_EXCLUDED_STAGES
                )
            )
            payload = {
                "event": "dimscan_collect_profile",
                "endpoint": profile.get("endpoint"),
                "job_id": profile.get("job_id"),
                "debug_mode": profile.get("debug_mode"),
                "ok": bool(ok),
                "status_code": status_code,
                "total_request_s": total_s,
                "stage_sum_s": stage_sum_s,
                "unattributed_s": total_s - stage_sum_s,
                "stages": {key: float(value) for key, value in sorted(stages.items())},
                "notes": profile.get("notes", []),
            }
            if error:
                payload["error"] = error
            print("[DIMSCAN_PROFILE] " + json.dumps(payload, sort_keys=True), flush=True)
    finally:
        _ACTIVE_PROFILE.reset(token)


def set_profile_value(key: str, value: Any) -> None:
    profile = _ACTIVE_PROFILE.get()
    if isinstance(profile, dict):
        profile[key] = value


def add_note(note: str) -> None:
    profile = _ACTIVE_PROFILE.get()
    if isinstance(profile, dict):
        notes = profile.setdefault("notes", [])
        if isinstance(notes, list):
            notes.append(note)


def add_timing(stage: str, elapsed_s: float) -> None:
    profile = _ACTIVE_PROFILE.get()
    if not isinstance(profile, dict):
        return
    stages = profile.setdefault("stages", {})
    if isinstance(stages, dict):
        stages[stage] = float(stages.get(stage, 0.0)) + float(elapsed_s)


@contextmanager
def timed_stage(stage: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        add_timing(stage, time.perf_counter() - started)
