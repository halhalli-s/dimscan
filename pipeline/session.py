"""Session helpers for DimScan job status, completed steps, and errors."""

from __future__ import annotations

from typing import Any

from app.config import DimScanConfig
from utils.io import read_json_if_exists, utc_now_iso, write_json_atomic
from utils.paths import get_job_file_path


STATUS_CREATED = "created"
STATUS_ERROR = "error"


def create_session(
    cfg: DimScanConfig,
    job_type: str,
    job_id: str,
    *,
    initial_status: str = STATUS_CREATED,
) -> dict[str, Any]:
    """Create a new session file for a job."""
    now = utc_now_iso()
    session = {
        "job_id": job_id,
        "job_type": job_type,
        "status": initial_status,
        "created_at": now,
        "updated_at": now,
        "captured_views": [],
        "steps": {},
        "errors": [],
    }

    path = get_job_file_path(cfg, job_type, job_id, cfg.session_filename)
    write_json_atomic(path, session)
    return session


def load_session(cfg: DimScanConfig, job_type: str, job_id: str) -> dict[str, Any]:
    """Load a job session, creating one when it does not exist."""
    path = get_job_file_path(cfg, job_type, job_id, cfg.session_filename)
    session = read_json_if_exists(path)
    if session is None:
        return create_session(cfg, job_type, job_id)
    return session


def save_session(
    cfg: DimScanConfig,
    job_type: str,
    job_id: str,
    session: dict[str, Any],
) -> dict[str, Any]:
    """Update and persist a job session."""
    session["updated_at"] = utc_now_iso()
    path = get_job_file_path(cfg, job_type, job_id, cfg.session_filename)
    write_json_atomic(path, session)
    return session


def set_status(cfg: DimScanConfig, job_type: str, job_id: str, status: str) -> dict[str, Any]:
    """Set the status for a job session."""
    session = load_session(cfg, job_type, job_id)
    session["status"] = status
    return save_session(cfg, job_type, job_id, session)


def mark_step(
    cfg: DimScanConfig,
    job_type: str,
    job_id: str,
    step_name: str,
    completed: bool = True,
) -> dict[str, Any]:
    """Mark a named session step as complete or incomplete."""
    session = load_session(cfg, job_type, job_id)
    steps = session.setdefault("steps", {})
    steps[step_name] = completed
    return save_session(cfg, job_type, job_id, session)


def record_captured_view(
    cfg: DimScanConfig,
    job_type: str,
    job_id: str,
    view_name: str,
) -> dict[str, Any]:
    """Record a captured view name in job session order."""
    session = load_session(cfg, job_type, job_id)
    captured_views = session.setdefault("captured_views", [])
    if view_name not in captured_views:
        captured_views.append(view_name)
    return save_session(cfg, job_type, job_id, session)


def add_error(
    cfg: DimScanConfig,
    job_type: str,
    job_id: str,
    message: str,
    *,
    step_name: str | None = None,
) -> dict[str, Any]:
    """Append an error to a session and set its status to error."""
    session = load_session(cfg, job_type, job_id)
    errors = session.setdefault("errors", [])
    errors.append(
        {
            "timestamp": utc_now_iso(),
            "step": step_name,
            "message": message,
        }
    )
    session["status"] = STATUS_ERROR
    return save_session(cfg, job_type, job_id, session)
