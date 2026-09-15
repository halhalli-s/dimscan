"""Plain-Python foundation tests for DimScan."""

from __future__ import annotations

import os
import sys
import tempfile
import types
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from app.config import DimScanConfig
from app.routes import _items_from_payload
from capture.camera import CaptureFrame, FakeCamera
from capture.sku import normalize_sku, require_valid_sku
from capture.pointcloud import metric_points_from_depth, valid_depth_pixel_indices_from_depth, write_ascii_ply
from geometry.measure import _cloud_quality_record, _geometry_status_from_warnings, _robust_extents, measure_view_geometry
from metadata.parser import (
    arrangement_cell_count,
    normalize_arrangement_type,
    parse_arrangement_dims,
)
from metadata.sku_catalogue import parse_catalogue_spec_pot_prior
from packing.rules import suggest_box_from_candidates
from pipeline.quality import _cloud_quality_status
from pipeline.arrangement import default_view_names, infer_view_mode
from pipeline.object_extraction import apply_calibrated_roi
from pipeline.object_extraction import extract_geometry_primary_object_cloud
from pipeline.object_extraction import remove_table_plane
from pipeline.object_extraction import remove_table_plane_with_cache
from pipeline.object_extraction import reset_table_plane_cache
from pipeline.object_extraction import select_and_merge_clusters
from pipeline.object_extraction import voxel_cluster_points
from pipeline import object_extraction as object_extraction_module
from pipeline.scan_writer import create_item_list, initialize_job, write_capture_artifacts
from segmentation.yoloe_segmenter import build_object_cloud_from_aligned_depth
from segmentation.yoloe_segmenter import _depth_scale_to_meters
from segmentation.yoloe_segmenter import FOCUSED_CROP_PROMPTS
from segmentation.yoloe_segmenter import _cached_yoloe_model
from segmentation.yoloe_segmenter import _configure_yoloe_text_prompts
from segmentation.yoloe_segmenter import _mask_sha256
from segmentation.yoloe_segmenter import _run_focused_crop_pass
from segmentation.yoloe_segmenter import _select_masks
from segmentation.yoloe_segmenter import _write_segment_cloud
from segmentation.yoloe_segmenter import _write_ai1_mask_debug_artifacts
from segmentation.yoloe_segmenter import _restore_masks_to_rgb_shape
from segmentation.yoloe_segmenter import run_yoloe_segmentation
from segmentation.yoloe_segmenter import reset_yoloe_model_cache
from utils.io import read_json, write_json_atomic
from utils.paths import generate_collection_job_id, generate_job_id, resolve_job_id, sanitize_collection_sku, sanitize_job_id


def _temp_cfg(tmp_path: Path) -> DimScanConfig:
    datasets = tmp_path / "datasets"
    return DimScanConfig(
        dataset_root=datasets,
        single_jobs_dir=datasets / "single" / "jobs",
        single_exports_dir=datasets / "single" / "exports",
        group_jobs_dir=datasets / "group" / "jobs",
        group_exports_dir=datasets / "group" / "exports",
    )


def test_collection_job_id_uses_resolved_single_sku_and_local_timestamp() -> None:
    now = datetime(2026, 7, 14, 12, 56, 23)
    job_id = generate_collection_job_id(
        "single",
        [{"sku": "PECC0010", "quantity": 1, "metadata": {"known": True}}],
        now=now,
    )
    assert job_id == "PECC0010_20260714_125623"


def test_collection_job_id_uses_single_fallback_for_unresolved_sku() -> None:
    now = datetime(2026, 7, 14, 1, 2, 3)
    unresolved = [{"sku": "UNKNOWN", "quantity": 1, "metadata": {"known": False}}]
    assert generate_collection_job_id("single", unresolved, now=now) == "SINGLE_20260714_010203"
    assert generate_collection_job_id("single", [], now=now) == "SINGLE_20260714_010203"


def test_collection_job_id_uses_group_prefix_without_sku() -> None:
    now = datetime(2026, 7, 14, 23, 59, 58)
    items = [{"sku": "PECC0010", "quantity": 6, "metadata": {"known": True}}]
    assert generate_collection_job_id("group", items, now=now) == "GROUP_20260714_235958"


def test_collection_sku_sanitization_and_timestamp_format() -> None:
    now = datetime(2026, 11, 9, 4, 5, 6)
    assert sanitize_collection_sku(" pe.cc/00 10!?-_") == "PECC0010-_"
    job_id = generate_collection_job_id(
        "single",
        [{"sku": " pe.cc/00 10!?-_", "quantity": 1, "metadata": {"known": True}}],
        now=now,
    )
    assert job_id == "PECC0010-__20261109_040506"
    assert generate_job_id("single").startswith("single_")


def test_collection_folder_and_job_id_match_and_manual_id_is_preserved() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        items = [{"sku": "PECC0010", "quantity": 1, "metadata": {"known": True}}]
        with patch("pipeline.scan_writer.generate_collection_job_id", return_value="PECC0010_20260714_125623"):
            generated = initialize_job(
                cfg,
                job_id=None,
                mode=cfg.mode_data_collection,
                job_type=cfg.job_type_single,
                arrangement_type="1x1",
                view_mode=cfg.view_mode_single,
                items=items,
            )
        assert generated["job_id"] == "PECC0010_20260714_125623"
        assert generated["job_dir"].name == generated["job_id"]
        assert generated["job_metadata"]["job_id"] == generated["job_id"]

        manual = initialize_job(
            cfg,
            job_id="My Manual Job",
            mode=cfg.mode_data_collection,
            job_type=cfg.job_type_single,
            arrangement_type="1x1",
            view_mode=cfg.view_mode_single,
            items=items,
        )
        assert manual["job_id"] == "my_manual_job"


def test_robust_extents_trim_xz_but_keep_full_y_height() -> None:
    points = np.asarray(
        [
            [float(index), float(index * 2), float(index * -3)]
            for index in range(101)
        ],
        dtype=float,
    )

    expected = np.percentile(points, 99.0, axis=0) - np.percentile(points, 1.0, axis=0)
    expected[1] = points[:, 1].max() - points[:, 1].min()
    actual = _robust_extents(points)

    assert np.allclose(actual, expected)


def test_non_blocking_segmentation_warnings_do_not_degrade_geometry_status() -> None:
    warnings = [
        "leaf_from_fallback_mask",
        "leaf_geometry_from_fallback_mask",
        "leaf_features_suppressed_due_to_fallback",
        "pot_mask_rejected_not_used_for_geometry",
        "table_segment_missing_using_ransac",
    ]

    assert _geometry_status_from_warnings(None, warnings) == "ok"


def test_measure_view_geometry_reuses_in_memory_geometry_primary_object_points() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        view_dir = Path(tmp) / "view_01"
        debug_dir = view_dir / "debug"
        debug_dir.mkdir(parents=True)
        points = np.asarray(
            [
                [x * 0.01, y * 0.01, z * 0.01]
                for x in range(6)
                for y in range(6)
                for z in range(3)
            ],
            dtype=float,
        )
        point_tuples = [tuple(float(value) for value in row) for row in points]
        write_ascii_ply(view_dir / "cloud.ply", point_tuples)
        write_json_atomic(view_dir / "capture_meta.json", {"cloud_type": "metric_xyz"})
        write_json_atomic(view_dir / "segmentation.json", {"status": "partial", "segments": {"object": "ok"}})
        write_json_atomic(
            debug_dir / "object_cloud_debug.json",
            {
                "object_cloud_coordinate_frame": "rgb_camera",
                "point_cloud_frame": "rgb_camera",
                "point_cloud_units": "meters",
                "roi_units": "meters",
            },
        )

        from geometry import measure as measure_module

        original_load_cloud = measure_module._load_cloud
        fast_loads: list[str] = []

        def count_fast_load(o3d, path, warnings, *, unit_contract=None):
            fast_loads.append(Path(path).name)
            return original_load_cloud(o3d, path, warnings, unit_contract=unit_contract)

        with patch("geometry.measure._load_cloud", side_effect=count_fast_load):
            fast_geometry = measure_view_geometry(
                view_dir,
                cfg=_temp_cfg(Path(tmp)),
                debug_mode=False,
                geometry_primary_object_points=points,
            )

        write_ascii_ply(view_dir / "object_cloud.ply", point_tuples)
        fallback_loads: list[str] = []

        def count_fallback_load(o3d, path, warnings, *, unit_contract=None):
            fallback_loads.append(Path(path).name)
            return original_load_cloud(o3d, path, warnings, unit_contract=unit_contract)

        with patch("geometry.measure._load_cloud", side_effect=count_fallback_load):
            fallback_geometry = measure_view_geometry(
                view_dir,
                cfg=_temp_cfg(Path(tmp)),
                debug_mode=False,
            )

        assert "object_cloud.ply" not in fast_loads
        assert "cloud.ply" not in fast_loads
        assert "object_cloud.ply" in fallback_loads
        assert "cloud.ply" in fallback_loads
        assert fast_geometry["object_dimensions_in"] == fallback_geometry["object_dimensions_in"]
        assert fast_geometry["source_cloud"] == fallback_geometry["source_cloud"] == "object_cloud"
        assert fast_geometry["metadata"]["clouds"]["object_cloud"]["in_memory"] is True


def test_save_object_cloud_toggle_preserves_in_memory_points_across_debug_modes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        height, width = 80, 100
        fx = fy = 100.0
        cx, cy = 50.0, 40.0
        aligned = np.zeros((height, width), dtype=np.float32)
        for col in range(12, 88):
            aligned[40, col] = 700.0 + (col - 12) * 3.0
        for row in range(16, 34):
            for col in range(43, 58):
                aligned[row, col] = 790.0 + (row % 3) * 5.0 + (col % 2) * 3.0

        def write_fixture(view_name: str) -> Path:
            fixture_dir = Path(tmp) / view_name
            fixture_dir.mkdir()
            Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8), mode="RGB").save(
                fixture_dir / "rgb.png",
                format="PNG",
            )
            np.save(fixture_dir / "depth_aligned_to_rgb.npy", aligned)
            write_json_atomic(
                fixture_dir / "capture_meta.json",
                {
                    "rgb_width": width,
                    "rgb_height": height,
                    "rgb_intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
                    "saved_depth_units": "raw_sdk_units",
                    "depth_scale_applied_to_saved_depth": False,
                    "sdk_depth_scale_m_per_unit": 0.001,
                    "object_cloud_depth_scale": 0.001,
                },
            )
            return fixture_dir

        baseline_points = None
        baseline_object_cloud_text = None
        for debug_mode, save_object_cloud in ((False, False), (False, True), (True, False), (True, True)):
            view_dir = write_fixture(f"debug_{int(debug_mode)}_save_{int(save_object_cloud)}")
            result = extract_geometry_primary_object_cloud(
                cfg,
                view_dir,
                debug_mode=debug_mode,
                save_object_cloud=save_object_cloud,
            )
            points = result["_runtime_final_object_points"]
            if baseline_points is None:
                baseline_points = points
            assert np.array_equal(points, baseline_points)
            assert result["final_point_count"] == int(len(baseline_points))
            assert result["artifacts"]["object_cloud_status"] == ("ready" if save_object_cloud else "not_saved")
            assert ("object_cloud" in result["artifacts"]) is save_object_cloud
            assert (view_dir / "object_cloud.ply").is_file() is save_object_cloud
            assert result["timing"]["object_cloud_ply_write_s"] >= 0.0
            assert result["timing"]["object_cloud_width_line_debug_write_s"] >= 0.0
            width_line_path = view_dir / "debug" / "object_cloud_width_line.ply"
            if not save_object_cloud:
                assert result["timing"]["object_cloud_ply_write_s"] == 0.0
                assert result["timing"]["object_cloud_width_line_debug_write_s"] == 0.0
            if save_object_cloud:
                object_cloud_text = (view_dir / "object_cloud.ply").read_text(encoding="utf-8")
                if baseline_object_cloud_text is None:
                    baseline_object_cloud_text = object_cloud_text
                assert object_cloud_text == baseline_object_cloud_text
            assert width_line_path.is_file() is (debug_mode and save_object_cloud)
            if debug_mode:
                debug = read_json(view_dir / "debug" / "object_cloud_debug.json")
                assert debug["object_cloud_saved"] is save_object_cloud
                assert ("final_object_cloud_path" in debug) is True
                assert bool(debug["final_object_cloud_path"]) is save_object_cloud
                if save_object_cloud:
                    expected_left, expected_right = np.percentile(points[:, 0], [1.0, 99.0])
                    width_debug = debug["width_line_debug"]
                    assert width_debug["purpose"] == "debug_visualization_only_not_authoritative_geometry"
                    assert width_debug["color_rgb"] == [255, 0, 255]
                    assert width_debug["line_point_count"] == 512
                    assert np.isclose(width_debug["left_endpoint_m"][0], expected_left)
                    assert np.isclose(width_debug["right_endpoint_m"][0], expected_right)
                    assert np.isclose(width_debug["represented_width_m"], expected_right - expected_left)
                    assert np.isclose(width_debug["represented_width_in"], (expected_right - expected_left) * 39.3701)
                    width_line_text = width_line_path.read_text(encoding="utf-8")
                    assert f"element vertex {len(points) + 512}" in width_line_text
                    assert "property uchar red" in width_line_text
                    assert "255 0 255" in width_line_text
                else:
                    assert "width_line_debug" not in debug
            else:
                assert not (view_dir / "debug").exists()


def test_object_geometry_warnings_still_degrade_geometry_status() -> None:
    assert _geometry_status_from_warnings(None, ["object_cloud_too_sparse"]) == "degraded"
    assert _geometry_status_from_warnings("No usable object cluster found.", []) == "degraded"


def test_raw_bbox_mismatch_warning_does_not_degrade_geometry_or_cloud_quality() -> None:
    dimensions = {
        "point_count": 1000,
        "length_in": 10.0,
        "width_in": 10.0,
        "height_in": 20.0,
        "raw_x_span_in": 10.0,
        "raw_y_span_in": 20.0,
        "raw_z_span_in": 10.0,
        "robust_x_span_in": 1.0,
        "robust_y_span_in": 20.0,
        "robust_z_span_in": 1.0,
        "length_robust_in": 1.0,
        "width_robust_in": 1.0,
        "height_robust_in": 20.0,
        "max_raw_to_robust_span_ratio": 10.0,
    }
    record = _cloud_quality_record(
        label="object",
        dimensions=dimensions,
        metadata={"largest_cluster_ratio": 1.0, "cluster_count": 1},
        source="object_cloud",
        fallback_mask_used=False,
    )

    assert "raw_bbox_much_larger_than_robust_bbox" in record["warnings"]
    assert record["quality"] == "ok"
    assert _cloud_quality_status(record) == "pass"
    assert _geometry_status_from_warnings(
        None,
        [
            "leaf_cloud_missing",
            "object_raw_bbox_much_larger_than_robust_bbox",
            "segmentation_failed_manifest_but_object_cloud_available",
            "table_segment_missing_using_ransac",
        ],
    ) == "ok"


class _FakeMasks:
    def __init__(self, data: list[np.ndarray]) -> None:
        self.data = data


class _FakeBoxes:
    def __init__(self, cls: list[int], conf: list[float], xyxy: list[list[float]]) -> None:
        self.cls = np.asarray(cls, dtype=float)
        self.conf = np.asarray(conf, dtype=float)
        self.xyxy = np.asarray(xyxy, dtype=float)


class _FakeYoloResult:
    def __init__(
        self,
        *,
        names: dict[int, str],
        cls: list[int],
        conf: list[float],
        boxes: list[list[float]],
        masks: list[np.ndarray],
        orig_shape: tuple[int, int],
    ) -> None:
        self.names = names
        self.boxes = _FakeBoxes(cls, conf, boxes)
        self.masks = _FakeMasks(masks)
        self.orig_shape = orig_shape


def test_focused_crop_uses_separate_prompt_cached_yoloe_model(tmp_path: Path) -> None:
    reset_yoloe_model_cache()
    constructor_calls = []
    set_class_calls = []

    class FakeYOLOE:
        def __init__(self, model_source: str, *, verbose: bool = False) -> None:
            self.instance_index = len(constructor_calls)
            constructor_calls.append((model_source, verbose))

        def set_classes(self, prompts) -> None:
            self.prompts = list(prompts)
            set_class_calls.append((self.instance_index, tuple(self.prompts)))

        def predict(self, *, source: str, conf: float, verbose: bool):
            crop_mask = np.zeros((24, 21), dtype=float)
            crop_mask[3:21, 3:18] = 1.0
            return [
                _FakeYoloResult(
                    names={0: "plant pot"},
                    cls=[0],
                    conf=[0.80],
                    boxes=[[1, 1, 20, 23]],
                    masks=[crop_mask],
                    orig_shape=(24, 21),
                )
            ]

    cfg = _temp_cfg(tmp_path)
    cfg.yoloe_model_name = "fake-yoloe"
    view_dir = tmp_path / "view_01"
    view_dir.mkdir()
    rgb_path = view_dir / "rgb.png"
    Image.fromarray(np.zeros((80, 100, 3), dtype=np.uint8), mode="RGB").save(rgb_path, format="PNG")
    object_mask = np.zeros((80, 100), dtype=bool)
    object_mask[16:34, 43:58] = True

    class _NoopTorchContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_torch = types.SimpleNamespace(
        inference_mode=lambda enabled=False: _NoopTorchContext(),
        no_grad=lambda: _NoopTorchContext(),
    )
    fake_ultralytics = types.SimpleNamespace(YOLOE=FakeYOLOE)
    with patch.dict(sys.modules, {"ultralytics": fake_ultralytics, "torch": fake_torch}):
        full_model = _cached_yoloe_model("fake-yoloe", list(cfg.yoloe_prompts))
        assert _configure_yoloe_text_prompts(full_model, list(cfg.yoloe_prompts)) is None
        crop_masks, _, crop_debug, crop_error = _run_focused_crop_pass(
            cfg=cfg,
            model_source="fake-yoloe",
            rgb_path=rgb_path,
            view_path=view_dir,
            object_mask=object_mask,
            object_box=[43, 16, 58, 34],
            debug_mode=False,
        )
        assert crop_error is None
        assert isinstance(crop_masks, dict)
        assert crop_debug["prompts"] == list(FOCUSED_CROP_PROMPTS)
        assert _configure_yoloe_text_prompts(full_model, list(cfg.yoloe_prompts)) is None

    assert constructor_calls == [("fake-yoloe", False), ("fake-yoloe", False)]
    assert set_class_calls == [
        (0, tuple(cfg.yoloe_prompts)),
        (1, tuple(FOCUSED_CROP_PROMPTS)),
    ]
    reset_yoloe_model_cache()


def test_sku_payload_allows_multiple_distinct_skus() -> None:
    items = _items_from_payload(
        [
            {"sku": "fern_6in", "quantity": 2},
            {"sku": "pothos_6in", "quantity": 3},
        ],
        allow_unresolved=True,
    )

    assert [item["sku"] for item in items] == ["FERN_6IN", "POTHOS_6IN"]
    assert [item["quantity"] for item in items] == [2, 3]


def test_sku_payload_merges_duplicate_quantities() -> None:
    items = _items_from_payload(
        [
            {"sku": "fern_6in", "quantity": 2},
            {"sku": "fern_6in", "quantity": 3},
        ],
        allow_unresolved=True,
    )

    assert len(items) == 1
    assert items[0]["sku"] == "FERN_6IN"
    assert items[0]["quantity"] == 5


def test_sku_payload_merges_whitespace_and_case_duplicates() -> None:
    items = _items_from_payload(
        [
            {"sku": " fern_6in ", "quantity": 2},
            {"sku": "FERN_6IN", "quantity": 4},
        ],
        allow_unresolved=True,
    )

    assert len(items) == 1
    assert items[0]["sku"] == "FERN_6IN"
    assert items[0]["quantity"] == 6


def test_sku_payload_rejects_non_positive_quantities() -> None:
    for quantity in (0, -1, True):
        try:
            _items_from_payload([{"sku": "fern_6in", "quantity": quantity}])
        except ValueError:
            pass
        else:
            raise AssertionError(f"quantity should be rejected: {quantity!r}")


def test_sku_item_list_preserves_other_skus_after_one_removed() -> None:
    items = _items_from_payload(
        [
            {"sku": "fern_6in", "quantity": 2},
            {"sku": "pothos_6in", "quantity": 3},
        ],
        allow_unresolved=True,
    )

    remaining_items = [item for item in items if item["sku"] != "FERN_6IN"]
    item_list = create_item_list(remaining_items)

    assert item_list["total_quantity"] == 3
    assert item_list["unique_sku_count"] == 1
    assert item_list["items"][0]["sku"] == "POTHOS_6IN"


def test_sku_payload_preserves_existing_single_sku_behavior() -> None:
    items = _items_from_payload([{"sku": "fern_8in", "quantity": 1}], allow_unresolved=True)

    assert items == [
        {
            "sku": "FERN_8IN",
            "quantity": 1,
            "metadata": items[0]["metadata"],
            "allow_unresolved": True,
        }
    ]
    assert items[0]["metadata"]["sku"] == "FERN_8IN"


def _cache_test_points(y_offset: float = 0.0) -> np.ndarray:
    xs = np.linspace(-0.35, 0.35, 42)
    zs = np.linspace(0.58, 0.98, 42)
    table_points = np.asarray([[x, y_offset, z] for x in xs for z in zs], dtype=float)
    object_points = np.asarray(
        [
            [x, y + y_offset, z]
            for x in np.linspace(-0.08, 0.08, 12)
            for y in np.linspace(0.08, 0.24, 10)
            for z in np.linspace(0.68, 0.84, 8)
        ],
        dtype=float,
    )
    return np.vstack([table_points, object_points])


def _cache_test_meta(profile: str = "1280x720 RGB 30fps") -> dict:
    return {
        "camera_type": "orbbec",
        "sdk_device": {"serial_number": "CACHE-TEST"},
        "color_profile": profile,
        "depth_profile": "1280x800 Y16 30fps",
        "rgb_width": 1280,
        "rgb_height": 720,
        "depth_width": 1280,
        "depth_height": 800,
        "aligned_depth_width": 1280,
        "aligned_depth_height": 720,
        "rgb_intrinsics": {"fx": 900.0, "fy": 900.0, "cx": 640.0, "cy": 360.0},
    }


def test_table_plane_cache_empty_then_hit_and_parity() -> None:
    reset_table_plane_cache()
    cfg = DimScanConfig()
    points = _cache_test_points()
    meta = _cache_test_meta()

    with patch.object(object_extraction_module, "_fit_table_plane", wraps=object_extraction_module._fit_table_plane) as fit_mock:
        first_points, _, first_table, _, first_debug = remove_table_plane_with_cache(points, None, cfg, meta)
    assert fit_mock.call_count >= 1
    assert first_debug["cached_plane_used"] is False
    assert first_debug["cache_replaced"] is True

    with patch.object(
        object_extraction_module,
        "_fit_table_plane",
        side_effect=AssertionError("cache hit should not run RANSAC"),
    ):
        second_points, _, second_table, _, second_debug = remove_table_plane_with_cache(points, None, cfg, meta)
    assert second_debug["cached_plane_used"] is True
    assert np.array_equal(first_points, second_points)
    assert np.array_equal(first_table, second_table)


def test_table_plane_cache_stale_plane_falls_back_to_ransac() -> None:
    reset_table_plane_cache()
    cfg = DimScanConfig()
    meta = _cache_test_meta()
    remove_table_plane_with_cache(_cache_test_points(), None, cfg, meta)

    shifted_points = _cache_test_points(y_offset=0.35)
    with patch.object(object_extraction_module, "_fit_table_plane", wraps=object_extraction_module._fit_table_plane) as fit_mock:
        _, _, _, _, debug = remove_table_plane_with_cache(shifted_points, None, cfg, meta)
    assert fit_mock.call_count >= 1
    assert debug["cached_plane_used"] is False
    assert debug["cache_validation"]["valid"] is False


def test_table_plane_cache_invalid_ransac_result_does_not_replace_cache() -> None:
    reset_table_plane_cache()
    cfg = DimScanConfig()
    points = _cache_test_points()
    meta = _cache_test_meta()

    def invalid_fit(fit_points, *, distance_threshold_m, iterations, seed):
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float), np.ones(len(fit_points), dtype=bool)

    with patch.object(object_extraction_module, "_fit_table_plane", side_effect=invalid_fit):
        _, _, _, _, debug = remove_table_plane_with_cache(points, None, cfg, meta)
    assert debug["cache_replaced"] is False
    assert debug["cache_replacement_validation"]["valid"] is False

    with patch.object(object_extraction_module, "_fit_table_plane", wraps=object_extraction_module._fit_table_plane) as fit_mock:
        remove_table_plane_with_cache(points, None, cfg, meta)
    assert fit_mock.call_count >= 1


def test_table_plane_cache_key_mismatch_and_reset_force_fresh_ransac() -> None:
    reset_table_plane_cache()
    cfg = DimScanConfig()
    points = _cache_test_points()
    remove_table_plane_with_cache(points, None, cfg, _cache_test_meta("1280x720 RGB 30fps"))

    camera_meta = _cache_test_meta("1280x720 RGB 30fps")
    camera_meta["sdk_device"] = {"serial_number": "OTHER-CAMERA"}
    with patch.object(object_extraction_module, "_fit_table_plane", wraps=object_extraction_module._fit_table_plane) as fit_mock:
        camera_debug = remove_table_plane_with_cache(points, None, cfg, camera_meta)[4]
    assert fit_mock.call_count >= 1
    assert camera_debug["cached_plane_used"] is False
    assert camera_debug["cache_validation"]["reason"] == "cache_miss"

    with patch.object(object_extraction_module, "_fit_table_plane", wraps=object_extraction_module._fit_table_plane) as fit_mock:
        mismatch_debug = remove_table_plane_with_cache(points, None, cfg, _cache_test_meta("640x480 RGB 30fps"))[4]
    assert fit_mock.call_count >= 1
    assert mismatch_debug["cached_plane_used"] is False
    assert mismatch_debug["cache_validation"]["reason"] == "cache_miss"

    roi_cfg = DimScanConfig()
    roi_cfg.roi_half_width_m = cfg.roi_half_width_m + 0.1
    with patch.object(object_extraction_module, "_fit_table_plane", wraps=object_extraction_module._fit_table_plane) as fit_mock:
        roi_debug = remove_table_plane_with_cache(points, None, roi_cfg, _cache_test_meta("1280x720 RGB 30fps"))[4]
    assert fit_mock.call_count >= 1
    assert roi_debug["cached_plane_used"] is False
    assert roi_debug["cache_validation"]["reason"] == "cache_miss"

    reset_table_plane_cache()
    with patch.object(object_extraction_module, "_fit_table_plane", wraps=object_extraction_module._fit_table_plane) as fit_mock:
        reset_debug = remove_table_plane_with_cache(points, None, cfg, _cache_test_meta("1280x720 RGB 30fps"))[4]
    assert fit_mock.call_count >= 1
    assert reset_debug["cached_plane_used"] is False


def run_tests() -> None:
    """Run foundation assertions."""
    test_robust_extents_trim_xz_but_keep_full_y_height()
    test_non_blocking_segmentation_warnings_do_not_degrade_geometry_status()
    test_measure_view_geometry_reuses_in_memory_geometry_primary_object_points()
    test_object_geometry_warnings_still_degrade_geometry_status()
    test_raw_bbox_mismatch_warning_does_not_degrade_geometry_or_cloud_quality()
    test_table_plane_cache_empty_then_hit_and_parity()
    test_table_plane_cache_stale_plane_falls_back_to_ransac()
    test_table_plane_cache_invalid_ransac_result_does_not_replace_cache()
    test_table_plane_cache_key_mismatch_and_reset_force_fresh_ransac()

    assert normalize_sku(" fern 8in\n") == "FERN-8IN"
    assert require_valid_sku("fern_8in") == "FERN_8IN"
    assert normalize_arrangement_type(" 2 X 5 ") == "2x5"
    assert parse_arrangement_dims("2x5") == (2, 5)
    assert arrangement_cell_count("2x5") == 10
    assert infer_view_mode("1x1") == "single_view"
    assert infer_view_mode("3x3") == "single_view"
    assert infer_view_mode("2x5") == "two_view_rectangle"
    assert default_view_names("2x5") == ["view_01", "view_02"]
    assert sanitize_job_id("  My Job/ID 01  ") == "my_jobid_01"
    assert resolve_job_id("single", "  My Job/ID 01  ") == "my_jobid_01"

    generated_job_id = resolve_job_id("group", "   ")
    assert generated_job_id.startswith("group_")
    assert len(generated_job_id.rsplit("_", 1)[1]) == 4

    unsafe_generated_job_id = resolve_job_id("single", "///")
    assert unsafe_generated_job_id.startswith("single_")

    round_prior = parse_catalogue_spec_pot_prior('4.5" Pot')
    assert round_prior["available"] is True
    assert round_prior["shape"] == "round"
    assert round_prior["diameter_in"] == 4.5
    assert round_prior["width_in"] == 4.5
    assert round_prior["depth_in"] == 4.5

    rectangular_prior = parse_catalogue_spec_pot_prior("10 x 20 Flat")
    assert rectangular_prior["available"] is True
    assert rectangular_prior["shape"] == "rectangular"
    assert rectangular_prior["width_in"] == 10.0
    assert rectangular_prior["depth_in"] == 20.0

    gallon_prior = parse_catalogue_spec_pot_prior("1 Gal")
    assert gallon_prior["available"] is False
    assert "gallon_spec_not_mapped" in gallon_prior["notes"]

    box = suggest_box_from_candidates(
        length_candidate_in=17.2,
        width_candidate_in=12.4,
        height_candidate_in=21.1,
        padding_in=1,
        round_increment_in=1,
    )
    assert box["length_in"] == 19
    assert box["width_in"] == 14
    assert box["height_in"] == 23

    depth = [
        [0.0, 1.0, 2.0],
        [float("nan"), 0.0, 3.0],
        [4.0, -1.0, 5.0],
    ]
    intrinsics = {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}
    points = metric_points_from_depth(depth, intrinsics)
    pixel_indices = valid_depth_pixel_indices_from_depth(depth)

    assert pixel_indices.dtype == np.int32
    assert pixel_indices.tolist() == [[0, 1], [0, 2], [1, 2], [2, 0], [2, 2]]
    assert len(points) == len(pixel_indices)
    for point, (row, col) in zip(points, pixel_indices.tolist()):
        z = depth[row][col]
        assert point == (col * z, row * z, z)

    fake_frame = FakeCamera().capture_frame()
    assert fake_frame.depth_aligned_to_rgb_values is None

    mask = np.asarray([[True, False], [True, True]])
    aligned_depth = np.asarray([[2000.0, 0.0], [float("nan"), 4000.0]])
    rgb = np.asarray(
        [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ],
        dtype=np.uint8,
    )
    object_points, object_colors, object_debug = build_object_cloud_from_aligned_depth(
        object_mask=mask,
        depth_aligned_to_rgb=aligned_depth,
        rgb_intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        depth_scale_to_meters=0.001,
        saved_depth_units="raw_sdk_units",
        rgb_image=rgb,
    )
    assert object_points == [(0.0, 0.0, 2.0), (4.0, 4.0, 4.0)]
    assert object_colors == [(10, 20, 30), (100, 110, 120)]
    assert object_debug["generation_method"] == "aligned_depth_rgb_mask"
    assert object_debug["invalid_masked_depth_count"] == 1
    assert object_debug["valid_depth_median_before_conversion"] == 3000.0
    assert object_debug["valid_z_median_m"] == 3.0
    assert _depth_scale_to_meters(
        {
            "saved_depth_units": "raw_sdk_units",
            "depth_scale_applied_to_saved_depth": False,
            "sdk_depth_scale_m_per_unit": 0.001,
            "object_cloud_depth_scale": 1.0,
        }
    ) == (0.001, "raw_sdk_units")
    try:
        _depth_scale_to_meters(
            {
                "saved_depth_units": "raw_sdk_units",
                "depth_scale_applied_to_saved_depth": False,
                "object_cloud_depth_scale": 1.0,
            }
        )
    except ValueError as exc:
        assert "meters_per_unit" in str(exc)
    else:
        raise AssertionError("raw SDK scale 1.0 must not be accepted as meters per unit")
    try:
        build_object_cloud_from_aligned_depth(
            object_mask=np.asarray([[True]]),
            depth_aligned_to_rgb=np.asarray([[1000.0]]),
            rgb_intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
            depth_scale_to_meters=1.0,
            saved_depth_units="raw_sdk_units",
        )
    except ValueError as exc:
        assert "implausible" in str(exc) or "meter_scale" in str(exc)
    else:
        raise AssertionError("kilometer-scale raw SDK depth must fail")
    try:
        build_object_cloud_from_aligned_depth(
            object_mask=np.asarray([[True]]),
            depth_aligned_to_rgb=np.asarray([[1000.0]]),
            rgb_intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
            depth_scale_to_meters=1.0,
            saved_depth_units="meters",
        )
    except ValueError as exc:
        assert "implausible" in str(exc)
    else:
        raise AssertionError("metadata claiming meters for ~1000 depth values must fail")
    try:
        build_object_cloud_from_aligned_depth(
            object_mask=np.ones((1, 2), dtype=bool),
            depth_aligned_to_rgb=np.ones((2, 2), dtype=float),
            rgb_intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        )
    except ValueError as exc:
        assert "shape_mismatch" in str(exc)
    else:
        raise AssertionError("shape mismatch must fail")

    model_mask = np.zeros((384, 640), dtype=bool)
    model_mask[100:110, 200:210] = True
    restored_masks, restore_debug = _restore_masks_to_rgb_shape(
        {"object": model_mask},
        rgb_size=(1280, 720),
    )
    restored_mask = restored_masks["object"]
    assert restored_mask.shape == (720, 1280)
    assert restored_mask.dtype == bool
    assert set(np.unique(restored_mask).tolist()) == {False, True}
    assert restore_debug["object"]["model_mask_shape"] == [384, 640]
    assert restore_debug["object"]["original_rgb_shape"] == [720, 1280]
    assert restore_debug["object"]["final_mask_shape"] == [720, 1280]
    assert restore_debug["object"]["resize_method"] == "nearest"
    assert restore_debug["object"]["restoration_required"] is True
    assert bool(restored_mask[190, 400]) is True

    cfg_for_extraction = DimScanConfig()
    xs = np.linspace(-0.35, 0.35, 18)
    zs = np.linspace(0.65, 0.95, 14)
    table_points = np.asarray([[x, 0.0, z] for x in xs for z in zs], dtype=float)
    central_cluster = np.asarray(
        [[x, y, z] for x in np.linspace(-0.06, 0.06, 8) for y in np.linspace(-0.30, -0.08, 8) for z in np.linspace(0.72, 0.86, 5)],
        dtype=float,
    )
    nearby_leaf_fragment = np.asarray(
        [[x, y, z] for x in np.linspace(0.13, 0.16, 4) for y in np.linspace(-0.36, -0.32, 4) for z in np.linspace(0.78, 0.84, 3)],
        dtype=float,
    )
    distant_large_object = np.asarray(
        [[x, y, z] for x in np.linspace(0.39, 0.45, 7) for y in np.linspace(-0.26, -0.08, 7) for z in np.linspace(0.74, 0.88, 4)],
        dtype=float,
    )
    full_points = np.vstack([table_points, central_cluster, nearby_leaf_fragment, distant_large_object])
    roi_points, _, rejected_roi, _, roi_debug = apply_calibrated_roi(full_points, None, cfg_for_extraction)
    assert len(roi_points) == len(full_points)
    assert len(rejected_roi) == 0
    assert roi_debug["roi_frame"] == "rgb_camera"
    non_table_points, _, removed_table_points, _, table_debug = remove_table_plane(roi_points, None)
    assert table_debug["ransac_succeeded"] is True
    assert len(removed_table_points) >= len(table_points) * 0.8
    assert len(non_table_points) < len(roi_points)
    labels, clusters, clustering_debug = voxel_cluster_points(non_table_points, voxel_size_m=0.035, min_points=10)
    assert clustering_debug["cluster_count"] >= 3
    selected_mask, selection_debug = select_and_merge_clusters(non_table_points, labels, clusters, cfg_for_extraction)
    selected_points = non_table_points[selected_mask]
    assert len(selected_points) > len(central_cluster)
    assert selection_debug["merged_fragment_indices"]
    rejected_reasons = {item["reason"] for item in selection_debug["rejected_clusters"]}
    assert "centroid_outside_expected_object_zone" in rejected_reasons
    assert abs(float(selected_points[:, 0].mean())) < 0.08

    object_anchor = np.zeros((220, 220), dtype=float)
    object_anchor[80:150, 80:140] = 1.0
    top_leaf = np.zeros((220, 220), dtype=float)
    top_leaf[60:82, 92:115] = 1.0
    side_leaf = np.zeros((220, 220), dtype=float)
    side_leaf[105:130, 140:160] = 1.0
    pot_part = np.zeros((220, 220), dtype=float)
    pot_part[150:178, 92:128] = 1.0
    distant_object = np.zeros((220, 220), dtype=float)
    distant_object[0:40, 0:40] = 1.0
    fake_result = _FakeYoloResult(
        names={0: "potted plant", 1: "leaf", 2: "plant pot", 3: "object"},
        cls=[0, 1, 1, 2, 3],
        conf=[0.95, 0.74, 0.72, 0.80, 0.70],
        boxes=[
            [80, 80, 140, 150],
            [92, 60, 115, 82],
            [140, 105, 160, 130],
            [92, 150, 128, 178],
            [0, 0, 40, 40],
        ],
        masks=[object_anchor, top_leaf, side_leaf, pot_part, distant_object],
        orig_shape=(220, 220),
    )
    selected_masks, selected_confidences, selection_debug = _select_masks(fake_result)
    selected_object_mask = selected_masks["object"]
    assert selected_confidences["object"] == 0.95
    assert bool(selected_object_mask[70, 100]) is True
    assert bool(selected_object_mask[120, 150]) is True
    assert bool(selected_object_mask[160, 110]) is True
    assert bool(selected_object_mask[10, 10]) is False
    assert selection_debug["object_union"]["masks_unioned"] is True
    assert set(selection_debug["object_union"]["unioned_indices"]) == {0, 1, 2, 3}
    assert selection_debug["object_union"]["candidate_decisions"]["4"]["accepted"] is False
    assert "not_spatially_consistent" in selection_debug["object_union"]["candidate_decisions"]["4"]["reason"]
    assert selection_debug["detections"][1]["included_in_object_union"] is True

    with tempfile.TemporaryDirectory() as tmp:
        debug_view = Path(tmp) / "view_01"
        debug_view.mkdir()
        rgb_path = debug_view / "rgb.png"
        Image.fromarray(np.zeros((220, 220, 3), dtype=np.uint8), mode="RGB").save(rgb_path, format="PNG")
        ai1_debug = _write_ai1_mask_debug_artifacts(
            cfg=_temp_cfg(Path(tmp)),
            view_dir=debug_view,
            rgb_path=rgb_path,
            result=fake_result,
            prediction_debug=selection_debug,
            model_masks=selected_masks,
            restored_masks=selected_masks,
            mask_restore_debug={"object": {"final_mask_shape": [220, 220]}},
            final_object_mask=selected_object_mask,
        )
        assert (debug_view / "debug" / "ai1" / "rgb_input.png").is_file()
        assert (debug_view / "debug" / "ai1" / "raw_detection_summary.json").is_file()
        assert (debug_view / "debug" / "ai1" / "raw_mask_00_model_resolution.png").is_file()
        assert (debug_view / "debug" / "ai1" / "raw_mask_union_model_resolution.png").is_file()
        assert (debug_view / "debug" / "ai1" / "restored_mask_before_threshold.png").is_file()
        assert (debug_view / "debug" / "ai1" / "restored_binary_mask.png").is_file()
        assert (debug_view / "debug" / "ai1" / "mask_before_morphology.png").is_file()
        assert (debug_view / "debug" / "ai1" / "mask_after_morphology.png").is_file()
        assert (debug_view / "debug" / "ai1" / "final_object_mask.png").is_file()
        assert (debug_view / "debug" / "ai1" / "final_object_mask_overlay_rgb.png").is_file()
        assert (debug_view / "debug" / "ai1" / "ai1_mask_debug.json").is_file()
        assert ai1_debug["morphology_operations"] == []
        assert ai1_debug["stage_metrics"]["final_object_mask"]["sha256"] == _mask_sha256(selected_object_mask)
        assert ai1_debug["masks_unioned"] is True

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        cfg.enable_ai1_validation = False
        height, width = 80, 100
        fx = fy = 100.0
        cx, cy = 50.0, 40.0
        aligned = np.zeros((height, width), dtype=np.float32)
        for col in range(12, 88):
            aligned[40, col] = 700.0 + (col - 12) * 3.0
        for row in range(16, 34):
            for col in range(43, 58):
                aligned[row, col] = 790.0 + (row % 3) * 5.0 + (col % 2) * 3.0

        def write_geometry_primary_fixture(view_name: str) -> Path:
            fixture_dir = Path(tmp) / view_name
            fixture_dir.mkdir()
            Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8), mode="RGB").save(
                fixture_dir / "rgb.png",
                format="PNG",
            )
            np.save(fixture_dir / "depth_raw.npy", np.ones((10, 10), dtype=np.float32))
            np.save(fixture_dir / "depth_aligned_to_rgb.npy", aligned)
            write_json_atomic(
                fixture_dir / "capture_meta.json",
                {
                    "cloud_type": "metric_xyz",
                    "rgb_width": width,
                    "rgb_height": height,
                    "rgb_intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
                    "saved_depth_units": "raw_sdk_units",
                    "depth_scale_applied_to_saved_depth": False,
                    "sdk_depth_scale_m_per_unit": 0.001,
                    "object_cloud_depth_scale": 0.001,
                },
            )
            return fixture_dir

        view_dir = write_geometry_primary_fixture("view_01")
        record = run_yoloe_segmentation(cfg, view_dir, force=True, debug_mode=True)
        extraction_debug = read_json(view_dir / "debug" / "object_extraction_debug.json")
        assert record["segments"]["object"] == "ok"
        assert record["status"] == "partial"
        assert (view_dir / "object_cloud.ply").is_file()
        assert (view_dir / "debug" / "object_cloud_roi_all_points.ply").is_file()
        assert (view_dir / "debug" / "object_cloud_after_table_removal.ply").is_file()
        assert (view_dir / "debug" / "table_plane.ply").is_file()
        assert (view_dir / "debug" / "object_cloud_rejected_points.ply").is_file()
        assert (view_dir / "debug" / "rejected_clusters").is_dir()
        assert extraction_debug["extraction_method"] == "roi_table_cluster"
        assert extraction_debug["authoritative_object_source"] == "geometry_cluster"
        assert extraction_debug["artifacts"]["debug_ply_artifacts_enabled"] is True
        assert extraction_debug["ai1_used_for_object_extraction"] is False
        assert "ai1_validation_disabled" in record["warnings"]

        off_view_dir = write_geometry_primary_fixture("view_02")
        off_record = run_yoloe_segmentation(cfg, off_view_dir, force=True, debug_mode=False)
        assert off_record["segments"]["object"] == "ok"
        assert off_record["status"] == "partial"
        assert (off_view_dir / "object_cloud.ply").is_file()
        assert not (off_view_dir / "debug").exists()
        assert "object_extraction_debug" not in off_record["artifacts"]
        assert (off_view_dir / "object_cloud.ply").read_text(encoding="utf-8") == (
            view_dir / "object_cloud.ply"
        ).read_text(encoding="utf-8")

        reset_yoloe_model_cache()
        constructor_calls = []
        set_class_calls = []

        class FakeYOLOE:
            def __init__(self, model_source: str, *, verbose: bool = False) -> None:
                self.model_source = model_source
                self.instance_index = len(constructor_calls)
                constructor_calls.append((model_source, verbose))

            def set_classes(self, prompts) -> None:
                self.prompts = list(prompts)
                set_class_calls.append((self.instance_index, tuple(self.prompts)))

            def predict(self, *, source: str, conf: float, verbose: bool):
                if "yoloe_object_crop.png" in str(source):
                    crop_mask = np.ones((24, 21), dtype=float)
                    return [
                        _FakeYoloResult(
                            names={0: "plant pot"},
                            cls=[0],
                            conf=[0.80],
                            boxes=[[1, 1, 20, 23]],
                            masks=[crop_mask],
                            orig_shape=(24, 21),
                        )
                    ]
                object_mask = np.zeros((height, width), dtype=float)
                object_mask[16:34, 43:58] = 1.0
                return [
                    _FakeYoloResult(
                        names={0: "potted plant"},
                        cls=[0],
                        conf=[0.95],
                        boxes=[[43, 16, 58, 34]],
                        masks=[object_mask],
                        orig_shape=(height, width),
                    )
                ]

        cfg.enable_ai1_validation = True
        cfg.yoloe_model_name = "fake-yoloe"
        fake_ultralytics = types.SimpleNamespace(YOLOE=FakeYOLOE)
        cached_view_a = write_geometry_primary_fixture("view_03")
        cached_view_b = write_geometry_primary_fixture("view_04")
        with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            first_cached = run_yoloe_segmentation(cfg, cached_view_a, force=True, debug_mode=False)
            second_cached = run_yoloe_segmentation(cfg, cached_view_b, force=True, debug_mode=False)
        assert len(constructor_calls) == 2
        assert constructor_calls == [("fake-yoloe", False), ("fake-yoloe", False)]
        assert len(set_class_calls) == 2
        assert set_class_calls[0] == (0, tuple(cfg.yoloe_prompts))
        assert set_class_calls[1] != (0, tuple(cfg.yoloe_prompts))
        assert first_cached["segments"]["object"] == "ok"
        assert second_cached["segments"]["object"] == "ok"
        reset_yoloe_model_cache()

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        frame = CaptureFrame(
            rgb_bytes=bytes(
                [
                    10, 20, 30,
                    40, 50, 60,
                    70, 80, 90,
                    100, 110, 120,
                ]
            ),
            depth_values=[[1000.0, 2000.0, 3000.0], [4000.0, 5000.0, 6000.0]],
            depth_aligned_to_rgb_values=[[1500.0, 0.0], [2500.0, 3500.0]],
            metadata={
                "camera_type": "unit_test",
                "rgb_width": 2,
                "rgb_height": 2,
                "rgb_format": "RGB",
                "depth_intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
                "rgb_intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
                "saved_depth_units": "raw_sdk_units",
                "sdk_reported_depth_scale": 1.0,
                "sdk_reported_depth_scale_units": "millimeters_per_unit",
                "sdk_depth_scale_m_per_unit": 0.001,
                "sdk_depth_scale": 0.001,
                "depth_scale_applied_to_saved_depth": False,
                "object_cloud_depth_scale": 0.001,
                "d2c_mode": "software_d2c",
                "selected_sdk_alignment_api": "ob.AlignFilter.process",
            },
        )
        artifacts = write_capture_artifacts(
            cfg,
            cfg.job_type_single,
            "job_aligned_writer",
            view_name="view_01",
            frame=frame,
        )
        assert Path(artifacts["depth_raw"]).is_file()
        assert Path(artifacts["depth_aligned_to_rgb"]).is_file()
        assert Path(artifacts["cloud"]).is_file()
        meta = read_json(Path(artifacts["capture_meta"]))
        assert meta["raw_depth_shape"] == [2, 3]
        assert meta["aligned_depth_shape"] == [2, 2]
        assert meta["depth_aligned_to_rgb"] is True
        assert meta["saved_depth_units"] == "raw_sdk_units"
        assert meta["sdk_reported_depth_scale"] == 1.0
        assert meta["sdk_reported_depth_scale_units"] == "millimeters_per_unit"
        assert meta["sdk_depth_scale_m_per_unit"] == 0.001
        assert meta["depth_scale_applied_to_saved_depth"] is False
        assert meta["object_cloud_depth_scale"] == 0.001
        assert meta["raw_depth_median"] == 3500.0
        assert meta["aligned_depth_median"] == 2500.0
        assert meta["cloud_pixel_indices_debug_only"] is True

        view_dir = Path(artifacts["capture_meta"]).parent
        rgb_roi = np.zeros((6, 6, 3), dtype=np.uint8)
        for row_index in range(6):
            for col_index in range(6):
                rgb_roi[row_index, col_index] = [row_index * 20, col_index * 20, 120]
        Image.fromarray(rgb_roi, mode="RGB").save(view_dir / "rgb.png", format="PNG")
        (view_dir / "cloud_pixel_indices.npy").write_bytes(b"not used by final object cloud")
        roi_meta = dict(meta)
        roi_meta["rgb_intrinsics"] = {"fx": 100.0, "fy": 100.0, "cx": 0.0, "cy": 0.0}
        aligned_roi_depth = np.full((6, 6), 800.0, dtype=float)
        aligned_roi_depth[0, 0] = 5000.0
        roi_mask = np.ones((6, 6), dtype=bool)
        warnings: list[str] = []
        object_cloud = _write_segment_cloud(
            view_dir=view_dir,
            segment_name="object",
            mask=roi_mask,
            depth=np.asarray([[1000.0, 2000.0, 3000.0], [4000.0, 5000.0, 6000.0]]),
            cfg=cfg,
            depth_aligned_to_rgb=aligned_roi_depth,
            warnings=warnings,
            capture_meta=roi_meta,
            rgb_size=(6, 6),
            debug={},
        )
        assert warnings == []
        assert object_cloud is not None
        assert Path(object_cloud).is_file()
        object_debug = read_json(view_dir / "debug" / "object_cloud_debug.json")
        assert object_debug["generation_method"] == "aligned_depth_rgb_mask_roi_filtered"
        assert object_debug["explicit_confirmation_no_old_fallback_used"] is True
        assert object_debug["scale_applied_during_cloud_generation"] == 0.001
        assert object_debug["sdk_depth_scale_m_per_unit"] == 0.001
        assert object_debug["valid_z_max_m"] == 5.0
        assert object_debug["mask_shape"] == [6, 6]
        assert object_debug["unfiltered_masked_point_count"] == 36
        assert object_debug["points_kept_by_roi"] == 35
        assert object_debug["points_rejected_by_roi"] == 1
        assert object_debug["final_point_count"] == 35
        assert object_debug["object_mask_sha256_consumed_by_cloud"] == _mask_sha256(roi_mask)
        assert object_debug["object_cloud_coordinate_frame"] == "rgb_camera"
        assert object_debug["roi_coordinate_frame"] == "rgb_camera"
        assert object_debug["roi_units"] == "meters"
        assert object_debug["geometry_uses_roi_filtered_object_cloud"] is True
        assert (view_dir / "debug" / "object_cloud_masked_before_roi.ply").is_file()
        assert (view_dir / "debug" / "object_cloud_rejected_by_roi.ply").is_file()
        final_lines = Path(object_cloud).read_text(encoding="utf-8").splitlines()
        assert "element vertex 35" in final_lines

        warnings = []
        mismatched_cloud = _write_segment_cloud(
            view_dir=view_dir,
            segment_name="object",
            mask=np.ones((1, 2), dtype=bool),
            depth=np.asarray([[1000.0, 2000.0, 3000.0], [4000.0, 5000.0, 6000.0]]),
            cfg=cfg,
            depth_aligned_to_rgb=np.asarray([[1500.0, 0.0], [2500.0, 3500.0]]),
            warnings=warnings,
            capture_meta=meta,
            rgb_size=(2, 2),
            debug={},
        )
        assert mismatched_cloud is None
        assert any("object_cloud_aligned_depth_shape_mismatch" in warning for warning in warnings)
        mismatch_debug = read_json(view_dir / "debug" / "object_cloud_debug.json")
        assert mismatch_debug["status"] == "failed"
        assert mismatch_debug["mask_shape"] == [1, 2]
        assert mismatch_debug["aligned_depth_shape"] == [2, 2]

        warnings = []
        missing_aligned = _write_segment_cloud(
            view_dir=view_dir,
            segment_name="object",
            mask=np.asarray([[True, False], [False, True]]),
            depth=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            cfg=cfg,
            depth_aligned_to_rgb=None,
            warnings=warnings,
            capture_meta=meta,
            rgb_size=(2, 2),
            debug={},
        )
        assert missing_aligned is None
        assert "object_cloud_aligned_depth_missing" in warnings
        failure_debug = read_json(view_dir / "debug" / "object_cloud_debug.json")
        assert failure_debug["status"] == "failed"
        assert failure_debug["error"] == "object_cloud_aligned_depth_missing"

        duplicate_frame = CaptureFrame(
            rgb_bytes=frame.rgb_bytes,
            depth_values=[[1.0, 2.0], [3.0, 4.0]],
            metadata=frame.metadata,
            depth_aligned_to_rgb_values=[[1.0, 2.0], [3.0, 4.0]],
        )
        try:
            write_capture_artifacts(
                cfg,
                cfg.job_type_single,
                "job_duplicate_depth",
                view_name="view_01",
                frame=duplicate_frame,
            )
        except ValueError as exc:
            assert "identical" in str(exc)
        else:
            raise AssertionError("writer must reject identical raw/aligned depth arrays")


if __name__ == "__main__":
    run_tests()
    print("test_foundation.py passed")
