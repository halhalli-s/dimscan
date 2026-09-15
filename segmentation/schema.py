"""Schema helpers for optional AI1 segmentation outputs."""

from __future__ import annotations

from typing import Any

from utils.io import utc_now_iso


SEGMENT_NAMES = ("object", "pot", "leaf", "table")
VALID_STATUSES = {"ok", "partial", "failed", "skipped"}
VALID_SEGMENT_STATUSES = {"ok", "missing", "failed", "fallback", "rejected"}


def empty_segment_statuses(status: str = "missing") -> dict[str, str]:
    """Return default per-segment statuses."""
    if status not in VALID_SEGMENT_STATUSES:
        raise ValueError(f"Invalid segment status: {status!r}")
    return {name: status for name in SEGMENT_NAMES}


def make_segmentation_record(
    *,
    status: str,
    model_backend: str,
    model_name: str | None = None,
    model_version: str | None = None,
    prompts: list[str] | None = None,
    segment_statuses: dict[str, str] | None = None,
    confidence_summary: dict[str, Any] | None = None,
    pot_quality: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    reason: str | None = None,
    artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Create a standard segmentation.json payload."""
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid segmentation status: {status!r}")

    statuses = empty_segment_statuses()
    for name, segment_status in (segment_statuses or {}).items():
        if name not in SEGMENT_NAMES:
            continue
        if segment_status not in VALID_SEGMENT_STATUSES:
            raise ValueError(f"Invalid {name} segment status: {segment_status!r}")
        statuses[name] = segment_status

    return {
        "status": status,
        "reason": reason,
        "model_name": model_name,
        "model_backend": model_backend,
        "model_version": model_version,
        "prompts": prompts or [],
        "segments": statuses,
        "confidence_summary": confidence_summary or {},
        "pot_quality": pot_quality or {},
        "warnings": sorted(set(warnings or [])),
        "artifacts": artifacts or {},
        "created_at": utc_now_iso(),
    }
