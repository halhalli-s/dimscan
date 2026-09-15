"""Isolated tests for the object axis/span validation tool."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from app.config import DimScanConfig
from segmentation.yoloe_segmenter import _apply_object_cloud_roi, build_object_cloud_from_aligned_depth
from testing_development.object_axis_span_validation.capture_object_spans import M_TO_IN, span_stats


def test_span_calculation_and_meter_conversion() -> None:
    points = [(-0.10, -0.20, 0.70), (0.15, 0.30, 0.80), (0.05, -0.10, 0.75)]
    spans = span_stats(points)
    assert abs(spans["x_span_m"] - 0.25) < 1e-12
    assert abs(spans["y_span_m"] - 0.50) < 1e-12
    assert abs(spans["z_span_m"] - 0.10) < 1e-12
    assert abs(spans["x_span_in"] - 9.842525) < 1e-9
    assert abs(spans["y_span_in"] - 19.68505) < 1e-9
    assert abs(spans["z_span_in"] - 3.93701) < 1e-9


def test_negative_coordinates_are_not_sorted_or_relabelled() -> None:
    points = [(5.0, -10.0, 2.0), (8.0, -2.0, 7.0)]
    spans = span_stats(points)
    assert spans["x_span_m"] == 3.0
    assert spans["y_span_m"] == 8.0
    assert spans["z_span_m"] == 5.0
    assert set(spans) >= {"x_span_m", "y_span_m", "z_span_m"}
    assert "length_in" not in spans
    assert "width_in" not in spans
    assert "height_in" not in spans


def test_roi_filtering_and_color_alignment() -> None:
    cfg = DimScanConfig()
    inside_points = [(0.0, 0.0, 0.8) for _ in range(30)]
    outside_points = [(0.0, 0.0, 2.0), (1.0, 0.0, 0.8)]
    points = inside_points + outside_points
    colors = [(10, 20, 30) for _ in inside_points] + [(200, 0, 0), (0, 200, 0)]
    kept, kept_colors, rejected, rejected_colors, debug = _apply_object_cloud_roi(
        points=points,
        colors=colors,
        cfg=cfg,
    )
    assert len(kept) == 30
    assert len(rejected) == 2
    assert kept_colors == [(10, 20, 30) for _ in inside_points]
    assert rejected_colors == [(200, 0, 0), (0, 200, 0)]
    assert debug["roi_coordinate_frame"] == "rgb_camera"
    assert debug["roi_units"] == "meters"


def test_mask_depth_shape_mismatch_fails() -> None:
    try:
        build_object_cloud_from_aligned_depth(
            object_mask=np.ones((2, 2), dtype=bool),
            depth_aligned_to_rgb=np.ones((3, 2), dtype=np.float32),
            rgb_intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
            depth_scale_to_meters=0.001,
            saved_depth_units="raw_sdk_units",
        )
    except ValueError as exc:
        assert "shape_mismatch" in str(exc)
    else:
        raise AssertionError("mask/depth mismatch must fail")


def test_roi_unit_mismatch_fails() -> None:
    cfg = DimScanConfig(roi_units="millimeters")
    try:
        _apply_object_cloud_roi(
            points=[(0.0, 0.0, 0.8) for _ in range(30)],
            colors=None,
            cfg=cfg,
        )
    except ValueError as exc:
        assert "object_cloud_roi_units_mismatch" in str(exc)
    else:
        raise AssertionError("ROI unit mismatch must fail")


def run_tests() -> None:
    test_span_calculation_and_meter_conversion()
    test_negative_coordinates_are_not_sorted_or_relabelled()
    test_roi_filtering_and_color_alignment()
    test_mask_depth_shape_mismatch_fails()
    test_roi_unit_mismatch_fails()


if __name__ == "__main__":
    run_tests()
    print("test_axis_span_validation.py passed")
