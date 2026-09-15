"""Inspect DimScan job folders and report expected artifact presence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import DimScanConfig
from utils.io import read_json_if_exists
from utils.paths import get_job_dir


def expected_job_files(cfg: DimScanConfig) -> list[str]:
    """Return expected job-level filenames."""
    return [
        getattr(cfg, "job_metadata_filename", "job_metadata.json"),
        getattr(cfg, "item_list_filename", "item_list.json"),
        getattr(cfg, "arrangement_filename", "arrangement.json"),
        getattr(cfg, "combined_features_filename", "combined_features.json"),
        getattr(cfg, "ground_truth_filename", "ground_truth.json"),
        getattr(cfg, "session_filename", "session.json"),
        getattr(cfg, "box_suggestion_filename", "box_suggestion.json"),
    ]


def expected_view_files(cfg: DimScanConfig) -> list[str]:
    """Return expected view-level filenames."""
    return [
        getattr(cfg, "rgb_filename", "rgb.png"),
        getattr(cfg, "depth_filename", "depth.json"),
        getattr(cfg, "depth_raw_filename", "depth_raw.npy"),
        getattr(cfg, "capture_meta_filename", "capture_meta.json"),
        getattr(cfg, "cloud_filename", "cloud.ply"),
        getattr(cfg, "geometry_filename", "geometry.json"),
        getattr(cfg, "features_filename", "features.json"),
    ]


def _readable_image(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def inspect_job(
    cfg: DimScanConfig,
    job_type: str,
    job_id: str,
) -> dict[str, Any]:
    """Inspect expected artifacts for a DimScan job."""
    job_dir = get_job_dir(cfg, job_type, job_id)
    exists = job_dir.exists() and job_dir.is_dir()
    job_files = {
        filename: (job_dir / filename).is_file()
        for filename in expected_job_files(cfg)
    }
    session_filename = getattr(cfg, "session_filename", "session.json")
    session = read_json_if_exists(job_dir / session_filename)

    views: dict[str, dict[str, bool]] = {}
    if exists:
        for path in sorted(job_dir.iterdir()):
            if path.is_dir() and path.name.startswith("view_"):
                files = {
                    filename: (path / filename).is_file()
                    for filename in expected_view_files(cfg)
                }
                capture_meta = read_json_if_exists(
                    path / getattr(cfg, "capture_meta_filename", "capture_meta.json"),
                    default={},
                )
                views[path.name] = {
                    **files,
                    "rgb_readable": _readable_image(path / getattr(cfg, "rgb_filename", "rgb.png")),
                    "depth_readable": _readable_image(path / getattr(cfg, "depth_filename", "depth.png")),
                    "cloud_type": capture_meta.get("cloud_type") if isinstance(capture_meta, dict) else None,
                }

    return {
        "job_id": job_id,
        "job_type": job_type,
        "job_dir": str(job_dir),
        "exists": exists,
        "job_files": job_files,
        "views": views,
        "session": session,
    }


def print_job_report(report: dict[str, Any]) -> None:
    """Print a readable job inspection report."""
    print(f"Job: {report['job_id']} ({report['job_type']})")
    print(f"Directory: {report['job_dir']}")
    print(f"Exists: {report['exists']}")

    print("Job files:")
    for filename, exists in report["job_files"].items():
        print(f"  [{'x' if exists else ' '}] {filename}")

    print("Views:")
    views = report["views"]
    if not views:
        print("  none")
    for view_name, files in views.items():
        print(f"  {view_name}:")
        for filename, exists in files.items():
            print(f"    [{'x' if exists else ' '}] {filename}")

    session = report.get("session")
    if isinstance(session, dict):
        print(f"Session status: {session.get('status')}")
    else:
        print("Session status: unavailable")
