"""Public segmentation runner for DimScan view folders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import DimScanConfig
from segmentation.yoloe_segmenter import run_yoloe_segmentation


def segment_view(
    cfg: DimScanConfig,
    view_dir: str | Path,
    *,
    force: bool = False,
    debug_mode: bool = True,
    object_extraction_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the configured segmentation backend for one view."""
    if cfg.segmentation_backend == "yoloe":
        return run_yoloe_segmentation(
            cfg,
            view_dir,
            force=force,
            debug_mode=debug_mode,
            object_extraction_result=object_extraction_result,
        )
    return run_yoloe_segmentation(
        cfg,
        view_dir,
        force=force,
        debug_mode=debug_mode,
        object_extraction_result=object_extraction_result,
    )
