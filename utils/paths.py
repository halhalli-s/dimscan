"""Centralized path construction for DimScan jobs, views, segmented clouds, and exports."""

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Literal
import uuid

from app.config import DimScanConfig


JobType = Literal["single", "group"]
SAFE_JOB_ID_RE = re.compile(r"[^a-z0-9_-]+")
JOB_ID_SPACE_RE = re.compile(r"\s+")
SAFE_COLLECTION_SKU_RE = re.compile(r"[^A-Z0-9_-]+")
COLLECTION_JOB_ID_RE = re.compile(r"^(?:[A-Z0-9_-]+|SINGLE|GROUP)_\d{8}_\d{6}$")


def sanitize_job_id(job_id: str) -> str:
    """Return a filesystem-safe operator-provided job ID."""
    normalized = JOB_ID_SPACE_RE.sub("_", job_id.strip().lower())
    return SAFE_JOB_ID_RE.sub("", normalized)


def generate_job_id(job_type: str) -> str:
    """Generate a filesystem-safe technical job ID."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:4]
    return f"{job_type}_{timestamp}_{short_id}"


def sanitize_collection_sku(value: object) -> str:
    """Return an uppercase SKU safe for a collection job folder."""
    return SAFE_COLLECTION_SKU_RE.sub("", str(value or "").strip().upper())


def generate_collection_job_id(
    job_type: str,
    items: list[dict[str, object]] | None = None,
    *,
    now: datetime | None = None,
) -> str:
    """Generate the deterministic local-time ID for a new dataset collection job."""
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    if job_type == "group":
        return f"GROUP_{timestamp}"
    if job_type != "single":
        raise ValueError(f"Unsupported job type: {job_type!r}")

    first = items[0] if isinstance(items, list) and items else {}
    metadata = first.get("metadata") if isinstance(first, dict) else None
    resolved = isinstance(metadata, dict) and metadata.get("known") is True
    sku = sanitize_collection_sku(first.get("sku")) if isinstance(first, dict) and resolved else ""
    return f"{sku or 'SINGLE'}_{timestamp}"


def resolve_collection_job_id(job_type: str, job_id: str | None) -> str:
    """Preserve generated collection IDs while retaining manual-ID sanitization."""
    if isinstance(job_id, str) and COLLECTION_JOB_ID_RE.fullmatch(job_id):
        return job_id
    return resolve_job_id(job_type, job_id)


def resolve_job_id(job_type: str, job_id: str | None) -> str:
    """Sanitize a provided job ID or generate one when blank/unsafe."""
    if job_id is not None:
        sanitized = sanitize_job_id(str(job_id))
        if sanitized:
            return sanitized
    return generate_job_id(job_type)


def get_jobs_root(cfg: DimScanConfig, job_type: str) -> Path:
    """Return the jobs root directory for a supported job type."""
    if job_type == cfg.job_type_single:
        return cfg.single_jobs_dir
    if job_type == cfg.job_type_group:
        return cfg.group_jobs_dir
    raise ValueError(f"Unsupported job type: {job_type!r}")


def get_exports_root(cfg: DimScanConfig, job_type: str) -> Path:
    """Return the exports root directory for a supported job type."""
    if job_type == cfg.job_type_single:
        return cfg.single_exports_dir
    if job_type == cfg.job_type_group:
        return cfg.group_exports_dir
    raise ValueError(f"Unsupported job type: {job_type!r}")


def get_job_dir(cfg: DimScanConfig, job_type: str, job_id: str) -> Path:
    """Return the directory for a job."""
    return get_jobs_root(cfg, job_type) / job_id


def get_view_dir(
    cfg: DimScanConfig,
    job_type: str,
    job_id: str,
    view_name: str,
) -> Path:
    """Return the directory for a view within a job."""
    return get_job_dir(cfg, job_type, job_id) / view_name


def get_segmented_dir(
    cfg: DimScanConfig,
    job_type: str,
    job_id: str,
    view_name: str,
) -> Path:
    """Return the segmented cloud directory for a view."""
    return get_view_dir(cfg, job_type, job_id, view_name) / cfg.segmented_dirname


def get_job_file_path(
    cfg: DimScanConfig,
    job_type: str,
    job_id: str,
    filename: str,
) -> Path:
    """Return a file path inside a job directory."""
    return get_job_dir(cfg, job_type, job_id) / filename


def get_view_file_path(
    cfg: DimScanConfig,
    job_type: str,
    job_id: str,
    view_name: str,
    filename: str,
) -> Path:
    """Return a file path inside a view directory."""
    return get_view_dir(cfg, job_type, job_id, view_name) / filename


def get_segmented_file_path(
    cfg: DimScanConfig,
    job_type: str,
    job_id: str,
    view_name: str,
    filename: str,
) -> Path:
    """Return a file path inside a view's segmented cloud directory."""
    return get_segmented_dir(cfg, job_type, job_id, view_name) / filename


def create_job_folder(
    cfg: DimScanConfig,
    job_type: str,
    job_id: str,
    view_names: list[str] | None = None,
) -> Path:
    """Create a job folder and optional view folders with segmented directories."""
    job_dir = get_job_dir(cfg, job_type, job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    if view_names is not None:
        for view_name in view_names:
            segmented_dir = get_segmented_dir(cfg, job_type, job_id, view_name)
            segmented_dir.mkdir(parents=True, exist_ok=True)

    return job_dir


def list_job_dirs(cfg: DimScanConfig, job_type: str) -> list[Path]:
    """Return sorted job directories for a supported job type."""
    jobs_root = get_jobs_root(cfg, job_type)
    if not jobs_root.exists():
        return []
    return sorted(path for path in jobs_root.iterdir() if path.is_dir())
