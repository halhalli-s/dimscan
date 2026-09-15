"""Plain-Python fake pipeline tests for DimScan."""

from __future__ import annotations

import os
import sys
import tempfile
import json
from unittest.mock import patch
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from app.config import DimScanConfig
from ai2.dataset import build_training_table
from capture.camera import FakeCamera
from features.extractor import LEAF_AI2_SUPPRESSED_FIELDS
from features.extractor import combine_object_dimensions_for_arrangement
from features.extractor import write_combined_features, write_view_features
from metadata.sku_lookup import make_item
from pipeline.build_dataset import build_rows
from pipeline.ground_truth import record_ground_truth
from pipeline.inspect_job import inspect_job
from pipeline.process_job import process_job
from pipeline.quality import build_quality_summary
from pipeline.run_scan import run_fake_data_collection_scan
from pipeline.real_capture import run_real_data_collection_scan
from pipeline.scan_writer import initialize_arranged_job
from utils.io import read_json, write_json_atomic
from utils.paths import get_job_dir


M_TO_IN = 39.3701


def _temp_cfg(tmp_path: Path) -> DimScanConfig:
    datasets = tmp_path / "datasets"
    return DimScanConfig(
        dataset_root=datasets,
        single_jobs_dir=datasets / "single" / "jobs",
        single_exports_dir=datasets / "single" / "exports",
        group_jobs_dir=datasets / "group" / "jobs",
        group_exports_dir=datasets / "group" / "exports",
    )


def _commit_test_collection_job(
    cfg: DimScanConfig,
    *,
    job_id: str,
    job_type: str,
    arrangement_type: str,
    items: list[dict[str, object]],
) -> None:
    initialize_arranged_job(
        cfg,
        job_id=job_id,
        mode=cfg.mode_data_collection,
        job_type=job_type,
        arrangement_type=arrangement_type,
        items=items,
        strict_quantity=False,
    )


def _base_geometry(
    *,
    pot_quality: dict,
    pot_dimensions: dict | None,
    pot_segment_status: str = "rejected",
    leaf_dimensions: dict | None = None,
    object_profile: dict | None = None,
    leaf_profile: dict | None = None,
    leaf_segment_status: str = "missing",
    extra_warnings: list[str] | None = None,
    leaf_fallback_quality: bool = False,
) -> dict:
    warnings = ["pot_mask_rejected_not_used_for_geometry"]
    if extra_warnings:
        warnings.extend(extra_warnings)
    geometry = {
        "status": "degraded",
        "reason": None,
        "source_cloud": "object_cloud",
        "cloud_type": "metric_xyz",
        "segmentation_used": True,
        "segmentation_status": "partial",
        "object_dimensions_in": {
            "length_in": 12.0,
            "width_in": 8.0,
            "height_in": 16.0,
            "point_count": 1000,
        },
        "pot_dimensions_in": pot_dimensions,
        "leaf_canopy_dimensions_in": leaf_dimensions,
        "table_plane": None,
        "confidence": {},
        "dimensions_in": {
            "length_in": 12.0,
            "width_in": 8.0,
            "height_in": 16.0,
            "point_count": 1000,
        },
        "warnings": warnings,
        "pot_quality": pot_quality,
        "metadata": {
            "segmentation_manifest": {
                "segments": {
                    "object": "ok",
                    "pot": pot_segment_status,
                    "leaf": leaf_segment_status,
                    "table": "missing",
                },
                "pot_quality": pot_quality,
            },
            "pot_segment": {
                "raw_width_in": 99.0,
                "raw_height_in": 88.0,
            },
        },
        "profiles": {},
    }
    if object_profile is not None:
        geometry["profiles"]["object_cloud"] = object_profile
    if leaf_profile is not None:
        geometry["profiles"]["leaf_cloud"] = leaf_profile
    if leaf_fallback_quality:
        geometry["cloud_quality"] = {
            "leaf": {
                "source": "fallback_mask",
                "fallback_mask_used": True,
            }
        }
    return geometry


def _rgb_camera_geometry(*, horizontal_m: float, vertical_m: float, depth_m: float) -> dict:
    dimensions = {
        "point_cloud_frame": "rgb_camera",
        "point_cloud_units": "meters",
        "x_min_m": 0.0,
        "y_min_m": -vertical_m,
        "z_min_m": 0.70,
        "x_max_m": horizontal_m,
        "y_max_m": 0.0,
        "z_max_m": 0.70 + depth_m,
        "x_span_m": horizontal_m,
        "y_span_m": vertical_m,
        "z_span_m": depth_m,
        "x_span_in": horizontal_m * M_TO_IN,
        "y_span_in": vertical_m * M_TO_IN,
        "z_span_in": depth_m * M_TO_IN,
        "horizontal_span_m": horizontal_m,
        "vertical_span_m": vertical_m,
        "depth_thickness_m": depth_m,
        "horizontal_span_in": horizontal_m * M_TO_IN,
        "vertical_span_in": vertical_m * M_TO_IN,
        "depth_thickness_in": depth_m * M_TO_IN,
        "robust_horizontal_span_m": horizontal_m,
        "robust_vertical_span_m": vertical_m,
        "robust_depth_thickness_m": depth_m,
        "robust_horizontal_span_in": horizontal_m * M_TO_IN,
        "robust_vertical_span_in": vertical_m * M_TO_IN,
        "robust_depth_thickness_in": depth_m * M_TO_IN,
        "length_in": horizontal_m * M_TO_IN,
        "width_in": depth_m * M_TO_IN,
        "height_in": vertical_m * M_TO_IN,
        "length_robust_in": horizontal_m * M_TO_IN,
        "width_robust_in": depth_m * M_TO_IN,
        "height_robust_in": vertical_m * M_TO_IN,
        "point_count": 1000,
    }
    geometry = _base_geometry(
        pot_quality={
            "status": "missing",
            "usable_for_geometry": False,
            "usable_for_model": False,
            "trust_score": 0.0,
            "reason": "missing",
            "metrics": {},
        },
        pot_dimensions=None,
    )
    geometry["object_dimensions_in"] = dimensions
    geometry["dimensions_in"] = dimensions
    geometry["scene"] = {
        "length_in": dimensions["length_in"],
        "width_in": dimensions["width_in"],
        "height_in": dimensions["height_in"],
        "point_count": dimensions["point_count"],
    }
    geometry["metadata"]["clouds"] = {
        "object_cloud": {
            "path": "view_01/object_cloud.ply",
            "point_cloud_frame": "rgb_camera",
            "point_cloud_units": "meters",
            "unit_mode": "meters",
        }
    }
    return geometry


def _write_arranged_geometry_job(
    cfg: DimScanConfig,
    job_type: str,
    job_id: str,
    *,
    arrangement_type: str,
    view_mode: str,
    view_geometries: dict[str, dict],
) -> Path:
    job_dir = get_job_dir(cfg, job_type, job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        job_dir / cfg.job_metadata_filename,
        {
            "job_id": job_id,
            "job_type": job_type,
            "arrangement_type": arrangement_type,
            "view_mode": view_mode,
        },
    )
    write_json_atomic(
        job_dir / cfg.item_list_filename,
        {
            "total_quantity": 1,
            "unique_sku_count": 1,
            "items": [{"sku": "TEST", "quantity": 1, "metadata": {}}],
        },
    )
    for view_name, geometry in view_geometries.items():
        view_dir = job_dir / view_name
        view_dir.mkdir(parents=True, exist_ok=True)
        geometry["metadata"]["clouds"]["object_cloud"]["path"] = f"{view_name}/object_cloud.ply"
        write_json_atomic(view_dir / cfg.geometry_filename, geometry)
    return job_dir


def _write_minimal_job(
    cfg: DimScanConfig,
    job_id: str,
    geometry: dict,
    *,
    item_metadata: dict | None = None,
) -> Path:
    job_dir = get_job_dir(cfg, cfg.job_type_single, job_id)
    view_dir = job_dir / "view_01"
    view_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        job_dir / cfg.job_metadata_filename,
        {
            "job_id": job_id,
            "job_type": cfg.job_type_single,
            "arrangement_type": "1x1",
            "view_mode": "single_view",
        },
    )
    write_json_atomic(
        job_dir / cfg.item_list_filename,
        {
            "total_quantity": 1,
            "unique_sku_count": 1,
            "items": [{"sku": "TEST", "quantity": 1, "metadata": item_metadata or {}}],
        },
    )
    write_json_atomic(view_dir / cfg.geometry_filename, geometry)
    return job_dir


def _leaf_dimensions() -> dict:
    return {
        "length_in": 14.0,
        "width_in": 9.0,
        "height_in": 7.0,
        "point_count": 700,
        "length_robust_in": 13.0,
        "width_robust_in": 8.0,
        "height_robust_in": 6.0,
    }


def _leaf_profile() -> dict:
    return {
        "available": True,
        "point_count": 700,
        "bbox_length_in": 14.0,
        "bbox_width_in": 9.0,
        "bbox_height_in": 7.0,
        "vertical_density_bins": [0.25, 0.75],
        "occupied_bin_count": 2,
        "max_bin_density": 0.75,
        "density_center_of_mass_z": 4.0,
        "compactness_point_count_per_bbox_volume": 0.8,
        "height_percentiles": {
            "z_p10": 1.0,
            "z_p25": 2.0,
            "z_p50": 3.0,
            "z_p75": 4.0,
            "z_p90": 5.0,
            "z_p95": 6.0,
            "z_p99": 7.0,
        },
        "thirds": {
            "lower_third": {"width_in": 8.0, "depth_in": 6.0},
            "middle_third": {"width_in": 10.0, "depth_in": 7.0},
            "upper_third": {"width_in": 9.0, "depth_in": 5.0},
        },
        "canopy_width_in": 14.0,
        "canopy_depth_in": 9.0,
        "canopy_height_in": 7.0,
        "canopy_center_z_in": 4.0,
        "leaf_to_object_point_ratio": 0.7,
        "leaf_to_object_volume_ratio": 1.2,
    }


def _object_profile() -> dict:
    return {
        "available": True,
        "point_count": 1000,
        "bbox_length_in": 12.0,
        "bbox_width_in": 8.0,
        "bbox_height_in": 16.0,
        "vertical_density_bins": [0.4, 0.6],
        "occupied_bin_count": 2,
        "max_bin_density": 0.6,
        "density_center_of_mass_z": 8.0,
        "compactness_point_count_per_bbox_volume": 0.65,
        "height_percentiles": {
            "z_p10": 2.0,
            "z_p25": 4.0,
            "z_p50": 8.0,
            "z_p75": 12.0,
            "z_p90": 14.0,
            "z_p95": 15.0,
            "z_p99": 15.8,
        },
        "thirds": {
            "lower_third": {"width_in": 7.0, "depth_in": 5.0},
            "middle_third": {"width_in": 8.0, "depth_in": 6.0},
            "upper_third": {"width_in": 6.0, "depth_in": 4.0},
        },
    }


def _object_cloud_quality() -> dict:
    return {
        "available": True,
        "source": "object_cloud",
        "fallback_mask_used": False,
        "point_count": 1000,
        "raw_bbox_in": {"length_in": 12.0, "width_in": 8.0, "height_in": 16.0},
        "robust_bbox_in": {"length_in": 12.0, "width_in": 8.0, "height_in": 16.0},
        "largest_cluster_ratio": 1.0,
        "cluster_count": 1,
        "outlier_ratio": 1.0,
        "outlier_sensitive": False,
        "quality": "ok",
        "warnings": [],
    }


def test_rejected_pot_debug_stays_out_of_model_features() -> None:
    """Rejected pot measurements are debug-only."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        job_dir = _write_minimal_job(
            cfg,
            "job_rejected_pot",
            _base_geometry(
                pot_quality={
                    "status": "rejected_partial",
                    "usable_for_geometry": False,
                    "usable_for_model": False,
                    "trust_score": 0.1,
                    "reason": "partial_pot_width",
                    "metrics": {"depth_bbox_width_in": 99.0},
                },
                pot_dimensions={"length_in": 99.0, "width_in": 77.0, "height_in": 88.0},
            ),
        )

        write_view_features(
            cfg,
            cfg.job_type_single,
            "job_rejected_pot",
            view_name="view_01",
            view_role="single_or_length",
        )
        features = read_json(job_dir / "view_01" / cfg.features_filename)
        pot_debug = read_json(job_dir / "view_01" / "debug" / "pot_debug.json")

        assert sorted(features) == ["ai2_features", "feature_schema_version", "quality_flags"]
        assert features["ai2_features"]["pot_diameter_in"] is None
        assert features["ai2_features"]["pot_height_in"] is None
        assert features["quality_flags"]["pot_source"] == "missing_or_rejected"
        assert "rejected_pot_dimensions_in" not in features
        assert "raw_pot_candidate_measurements" not in features
        assert pot_debug["rejected_pot_dimensions_in"]["width_in"] == 77.0
        assert pot_debug["raw_pot_candidate_measurements"]["raw_width_in"] == 99.0


def test_unusable_pot_sources_are_missing_or_rejected_without_prior() -> None:
    """Fallback, rejected, and missing pots do not masquerade as model-facing sources."""
    cases = (
        (
            "fallback",
            {"length_in": 99.0, "width_in": 77.0, "height_in": 88.0},
            "fallback",
        ),
        (
            "rejected_low_confidence",
            {"length_in": 99.0, "width_in": 77.0, "height_in": 88.0},
            "rejected",
        ),
        (
            "missing",
            None,
            "missing",
        ),
    )
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        for status, pot_dimensions, segment_status in cases:
            job_id = f"job_{status}"
            job_dir = _write_minimal_job(
                cfg,
                job_id,
                _base_geometry(
                    pot_quality={
                        "status": status,
                        "usable_for_geometry": False,
                        "usable_for_model": False,
                        "trust_score": 0.0,
                        "reason": status,
                        "metrics": {},
                    },
                    pot_dimensions=pot_dimensions,
                    pot_segment_status=segment_status,
                ),
            )

            write_view_features(
                cfg,
                cfg.job_type_single,
                job_id,
                view_name="view_01",
                view_role="single_or_length",
            )
            features = read_json(job_dir / "view_01" / cfg.features_filename)

            assert features["quality_flags"]["pot_quality_status"] == status
            assert features["quality_flags"]["pot_usable_for_model"] is False
            assert features["quality_flags"]["pot_source"] == "missing_or_rejected"
            assert features["ai2_features"]["pot_diameter_in"] is None
            assert features["ai2_features"]["pot_height_in"] is None


def test_combined_features_stay_clean_with_rejected_pot() -> None:
    """Combined features keep rejected pot debug in a separate job debug file."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        job_dir = _write_minimal_job(
            cfg,
            "job_combined_rejected_pot",
            _base_geometry(
                pot_quality={
                    "status": "rejected_partial",
                    "usable_for_geometry": False,
                    "usable_for_model": False,
                    "trust_score": 0.1,
                    "reason": "partial_pot_width",
                    "metrics": {"depth_bbox_width_in": 99.0},
                },
                pot_dimensions={"length_in": 99.0, "width_in": 77.0, "height_in": 88.0},
            ),
        )

        combined = write_combined_features(
            cfg,
            cfg.job_type_single,
            "job_combined_rejected_pot",
            view_names=["view_01"],
        )
        combined_on_disk = read_json(job_dir / cfg.combined_features_filename)
        combined_debug = read_json(job_dir / "debug" / "combined_features_debug.json")

        assert sorted(combined_on_disk) == ["ai2_features", "feature_schema_version", "quality_flags"]
        assert combined == combined_on_disk
        assert combined_on_disk["ai2_features"]["pot_diameter_in"] is None
        assert combined_on_disk["quality_flags"]["pot_source"] == "missing_or_rejected"
        assert "rejected_pot_dimensions_in" not in combined_on_disk
        assert combined_debug["pot_debug"]["rejected_pot_dimensions_in"]["width_in"] == 77.0


def test_combined_features_use_sku_prior_source_when_prior_supplies_pot() -> None:
    """SKU prior is the source only when it fills model-facing pot fields."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        job_dir = _write_minimal_job(
            cfg,
            "job_sku_prior_pot",
            _base_geometry(
                pot_quality={
                    "status": "fallback",
                    "usable_for_geometry": False,
                    "usable_for_model": False,
                    "trust_score": 0.2,
                    "reason": "fallback_pot_mask_debug_only",
                    "metrics": {},
                },
                pot_dimensions=None,
                pot_segment_status="fallback",
            ),
            item_metadata={
                "pot_prior": {
                    "pot_prior_available": True,
                    "source": "sku_exact",
                    "diameter_in": 4.5,
                    "height_in": 4.0,
                    "volume_qt": None,
                    "confidence": "high",
                }
            },
        )

        combined = write_combined_features(
            cfg,
            cfg.job_type_single,
            "job_sku_prior_pot",
            view_names=["view_01"],
        )
        combined_on_disk = read_json(job_dir / cfg.combined_features_filename)

        assert combined == combined_on_disk
        assert combined_on_disk["ai2_features"]["pot_diameter_in"] == 4.5
        assert combined_on_disk["ai2_features"]["pot_height_in"] == 4.0
        assert combined_on_disk["quality_flags"]["pot_source"] == "sku_prior"
        assert combined_on_disk["quality_flags"]["pot_quality_status"] == "fallback"
        assert combined_on_disk["quality_flags"]["pot_usable_for_model"] is False
        assert "rejected_pot_dimensions_in" not in combined_on_disk


def test_combined_features_use_catalogue_style_pot_prior() -> None:
    """Catalogue-style priors can fill model-facing pot fields."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        job_dir = _write_minimal_job(
            cfg,
            "job_catalogue_prior_pot",
            _base_geometry(
                pot_quality={
                    "status": "fallback",
                    "usable_for_geometry": False,
                    "usable_for_model": False,
                    "trust_score": 0.2,
                    "reason": "fallback_pot_mask_debug_only",
                    "metrics": {},
                },
                pot_dimensions=None,
                pot_segment_status="fallback",
            ),
            item_metadata={
                "pot_prior": {
                    "available": True,
                    "shape": "round",
                    "diameter_in": 4.5,
                    "width_in": 4.5,
                    "depth_in": 4.5,
                    "height_in": None,
                    "source": "sku_catalogue",
                    "confidence": "catalog",
                }
            },
        )

        combined = write_combined_features(
            cfg,
            cfg.job_type_single,
            "job_catalogue_prior_pot",
            view_names=["view_01"],
        )
        combined_on_disk = read_json(job_dir / cfg.combined_features_filename)

        assert combined == combined_on_disk
        assert combined_on_disk["ai2_features"]["pot_diameter_in"] == 4.5
        assert combined_on_disk["ai2_features"]["pot_width_in"] == 4.5
        assert combined_on_disk["ai2_features"]["pot_depth_in"] == 4.5
        assert combined_on_disk["ai2_features"]["pot_height_in"] is None
        assert combined_on_disk["ai2_features"]["pot_prior_shape"] == "round"
        assert combined_on_disk["quality_flags"]["pot_source"] == "sku_prior"
        assert combined_on_disk["quality_flags"]["pot_prior_source"] == "sku_catalogue"
        assert combined_on_disk["quality_flags"]["pot_prior_confidence"] == "high"


def test_combined_features_include_sku_context_only_from_item_metadata() -> None:
    """AI2 SKU context is copied from item metadata without exposing raw catalog fields."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        job_dir = _write_minimal_job(
            cfg,
            "job_sku_context",
            _base_geometry(
                pot_quality={
                    "status": "rejected_partial",
                    "usable_for_geometry": False,
                    "usable_for_model": False,
                    "trust_score": 1.0,
                    "reason": "pot_mask_rejected_not_used_for_geometry",
                    "metrics": {},
                },
                pot_dimensions=None,
            ),
            item_metadata={
                "category": " Annuals ",
                "common_name": " FanciFillers Sea Salt' Wormwood ",
                "spec": " 4.5\" Pot ",
                "unit_price": 36.0,
                "source_file": "catalog.xlsx",
                "uom": "PK 10",
            },
        )
        before_geometry = read_json(job_dir / "view_01" / cfg.geometry_filename)

        combined = write_combined_features(
            cfg,
            cfg.job_type_single,
            "job_sku_context",
            view_names=["view_01"],
        )
        combined_on_disk = read_json(job_dir / cfg.combined_features_filename)
        ai2 = combined_on_disk["ai2_features"]

        assert combined == combined_on_disk
        assert ai2["sku_category"] == "Annuals"
        assert ai2["sku_common_name"] == "FanciFillers Sea Salt' Wormwood"
        assert ai2["sku_spec"] == "4.5\" Pot"
        assert "unit_price" not in ai2
        assert "source_file" not in ai2
        assert "uom" not in ai2
        assert "sku" not in ai2
        assert "raw_sku" not in ai2
        assert read_json(job_dir / "view_01" / cfg.geometry_filename) == before_geometry
        assert combined_on_disk["quality_flags"]["pot_source"] == "missing_or_rejected"
        assert combined_on_disk["ai2_features"]["authoritative_object_source"] == "geometry_primary"


def test_combined_features_sku_context_is_nullable_when_missing() -> None:
    """Missing SKU context stays explicit null in AI2 features."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        job_dir = _write_minimal_job(
            cfg,
            "job_sku_context_missing",
            _base_geometry(
                pot_quality={
                    "status": "rejected_partial",
                    "usable_for_geometry": False,
                    "usable_for_model": False,
                    "trust_score": 1.0,
                    "reason": "pot_mask_rejected_not_used_for_geometry",
                    "metrics": {},
                },
                pot_dimensions=None,
            ),
        )

        combined = write_combined_features(
            cfg,
            cfg.job_type_single,
            "job_sku_context_missing",
            view_names=["view_01"],
        )
        combined_on_disk = read_json(job_dir / cfg.combined_features_filename)
        ai2 = combined_on_disk["ai2_features"]

        assert combined == combined_on_disk
        assert ai2["sku_category"] is None
        assert ai2["sku_common_name"] is None
        assert ai2["sku_spec"] is None


def test_trusted_pot_dimensions_enter_ai2_features() -> None:
    """Trusted pot geometry may enter model-facing fields."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        job_dir = _write_minimal_job(
            cfg,
            "job_trusted_pot",
            _base_geometry(
                pot_quality={
                    "status": "trusted",
                    "usable_for_geometry": True,
                    "usable_for_model": True,
                    "trust_score": 0.95,
                    "reason": None,
                    "metrics": {},
                },
                pot_dimensions={"length_in": 6.0, "width_in": 5.5, "height_in": 5.0},
            ),
        )

        write_view_features(
            cfg,
            cfg.job_type_single,
            "job_trusted_pot",
            view_name="view_01",
            view_role="single_or_length",
        )
        features = read_json(job_dir / "view_01" / cfg.features_filename)

        assert features["ai2_features"]["pot_diameter_in"] == 6.0
        assert features["ai2_features"]["pot_height_in"] == 5.0
        assert features["quality_flags"]["pot_source"] == "segmentation_trusted"
        assert features["quality_flags"]["pot_usable_for_model"] is True


def test_fallback_leaf_features_are_suppressed_from_ai2() -> None:
    """Fallback leaf geometry remains debug/provenance only."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        job_dir = _write_minimal_job(
            cfg,
            "job_fallback_leaf",
            _base_geometry(
                pot_quality={
                    "status": "missing",
                    "usable_for_geometry": False,
                    "usable_for_model": False,
                    "trust_score": 0.0,
                    "reason": "missing",
                    "metrics": {},
                },
                pot_dimensions=None,
                leaf_dimensions=_leaf_dimensions(),
                leaf_profile=_leaf_profile(),
                leaf_segment_status="fallback",
                extra_warnings=["leaf_from_fallback_mask", "leaf_geometry_from_fallback_mask"],
                leaf_fallback_quality=True,
            ),
        )

        write_view_features(
            cfg,
            cfg.job_type_single,
            "job_fallback_leaf",
            view_name="view_01",
            view_role="single_or_length",
        )
        features = read_json(job_dir / "view_01" / cfg.features_filename)
        ai2 = features["ai2_features"]
        flags = features["quality_flags"]

        assert ai2["object_length_in"] == 12.0
        assert ai2["object_width_in"] == 8.0
        assert ai2["object_height_in"] == 16.0
        assert ai2["leaf_available"] is False
        assert ai2["leaf_profile_available"] is False
        for key in LEAF_AI2_SUPPRESSED_FIELDS:
            assert ai2[key] is None
        assert flags["leaf_available"] is False
        assert flags["segments"]["leaf"] == "fallback"
        assert "leaf_from_fallback_mask" in flags["warnings"]
        assert "leaf_geometry_from_fallback_mask" in flags["warnings"]
        assert "leaf_features_suppressed_due_to_fallback" in flags["warnings"]


def test_true_leaf_segmentation_can_populate_ai2_features() -> None:
    """Accepted leaf segmentation remains available to model-facing features."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        job_dir = _write_minimal_job(
            cfg,
            "job_true_leaf",
            _base_geometry(
                pot_quality={
                    "status": "missing",
                    "usable_for_geometry": False,
                    "usable_for_model": False,
                    "trust_score": 0.0,
                    "reason": "missing",
                    "metrics": {},
                },
                pot_dimensions=None,
                leaf_dimensions=_leaf_dimensions(),
                object_profile=_object_profile(),
                leaf_profile=_leaf_profile(),
                leaf_segment_status="ok",
            ),
        )

        write_view_features(
            cfg,
            cfg.job_type_single,
            "job_true_leaf",
            view_name="view_01",
            view_role="single_or_length",
        )
        features = read_json(job_dir / "view_01" / cfg.features_filename)
        ai2 = features["ai2_features"]

        assert ai2["object_length_in"] == 12.0
        assert ai2["object_density_center_of_mass_y"] == 8.0
        assert ai2["object_y_p50"] == 8.0
        assert "object_density_center_of_mass_z" not in ai2
        assert "object_z_p50" not in ai2
        assert "object_bbox_length_in" not in ai2
        assert ai2["leaf_available"] is True
        assert ai2["leaf_canopy_length_in"] == 14.0
        assert ai2["leaf_canopy_point_count"] == 700
        assert ai2["leaf_profile_available"] is True
        assert ai2["leaf_vertical_density_bins"] == [0.25, 0.75]
        assert ai2["leaf_density_center_of_mass_y"] == 4.0
        assert ai2["leaf_y_p50"] == 3.0
        assert ai2["leaf_canopy_center_y_in"] == 4.0
        assert "leaf_density_center_of_mass_z" not in ai2
        assert "leaf_z_p50" not in ai2
        assert "leaf_canopy_center_z_in" not in ai2
        assert "leaf_features_suppressed_due_to_fallback" not in features["quality_flags"]["warnings"]


def test_missing_leaf_segmentation_keeps_ai2_leaf_fields_null() -> None:
    """Missing leaf segmentation stays nullable and does not affect object fields."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        job_dir = _write_minimal_job(
            cfg,
            "job_missing_leaf",
            _base_geometry(
                pot_quality={
                    "status": "missing",
                    "usable_for_geometry": False,
                    "usable_for_model": False,
                    "trust_score": 0.0,
                    "reason": "missing",
                    "metrics": {},
                },
                pot_dimensions=None,
                leaf_dimensions=None,
                leaf_segment_status="missing",
            ),
        )

        write_view_features(
            cfg,
            cfg.job_type_single,
            "job_missing_leaf",
            view_name="view_01",
            view_role="single_or_length",
        )
        features = read_json(job_dir / "view_01" / cfg.features_filename)
        ai2 = features["ai2_features"]

        assert ai2["object_length_in"] == 12.0
        assert ai2["leaf_available"] is False
        assert ai2["leaf_canopy_length_in"] is None
        assert ai2["leaf_canopy_width_in"] is None
        assert ai2["leaf_canopy_height_in"] is None
        assert ai2["leaf_canopy_point_count"] is None
        assert "leaf_profile_available" not in ai2


def test_sku_pot_prior_still_applies_when_leaf_is_fallback() -> None:
    """Leaf fallback suppression does not disturb SKU pot-prior behavior."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        job_dir = _write_minimal_job(
            cfg,
            "job_fallback_leaf_sku_prior",
            _base_geometry(
                pot_quality={
                    "status": "fallback",
                    "usable_for_geometry": False,
                    "usable_for_model": False,
                    "trust_score": 0.2,
                    "reason": "fallback_pot_mask_debug_only",
                    "metrics": {},
                },
                pot_dimensions=None,
                pot_segment_status="fallback",
                leaf_dimensions=_leaf_dimensions(),
                leaf_profile=_leaf_profile(),
                leaf_segment_status="fallback",
                extra_warnings=["leaf_geometry_from_fallback_mask"],
            ),
            item_metadata={
                "pot_prior": {
                    "available": True,
                    "shape": "round",
                    "diameter_in": 4.5,
                    "width_in": 4.5,
                    "depth_in": 4.5,
                    "height_in": None,
                    "source": "sku_catalogue",
                    "confidence": "catalog",
                }
            },
        )

        combined = write_combined_features(
            cfg,
            cfg.job_type_single,
            "job_fallback_leaf_sku_prior",
            view_names=["view_01"],
        )
        combined_on_disk = read_json(job_dir / cfg.combined_features_filename)

        assert combined == combined_on_disk
        assert combined_on_disk["ai2_features"]["leaf_available"] is False
        assert combined_on_disk["ai2_features"]["leaf_canopy_length_in"] is None
        assert combined_on_disk["ai2_features"]["pot_diameter_in"] == 4.5
        assert combined_on_disk["ai2_features"]["pot_width_in"] == 4.5
        assert combined_on_disk["quality_flags"]["pot_source"] == "sku_prior"


def test_square_1x1_object_semantics_use_horizontal_for_length_and_width() -> None:
    """Square single-view semantics use X for both package length and width."""
    features = [
        {
            "view_name": "view_01",
            "view_role": "single_or_length",
            "features": {
                "point_cloud_frame": "rgb_camera",
                "point_cloud_units": "meters",
                "object_horizontal_span_m": 0.25,
                "object_vertical_span_m": 0.50,
                "object_depth_thickness_m": 0.10,
                "object_robust_horizontal_span_m": 0.22,
                "object_robust_vertical_span_m": 0.48,
                "object_robust_depth_thickness_m": 0.08,
                "object_point_count": 100,
            },
        }
    ]
    semantic = combine_object_dimensions_for_arrangement(
        job_metadata={"arrangement_type": "1x1", "view_mode": "single_view"},
        view_features=features,
    )
    dims = semantic["semantic_dimensions_in"]
    assert abs(dims["length_in"] - 8.661422) < 1e-9
    assert abs(dims["width_in"] - 8.661422) < 1e-9
    assert abs(dims["height_in"] - 19.68505) < 1e-9
    assert abs(semantic["per_view"][0]["depth_thickness_in"] - 3.93701) < 1e-9
    assert semantic["semantic_mapping"]["length_source"] == "view_01.horizontal_span_robust_1_99"
    assert semantic["semantic_mapping"]["height_source"] == "view_01.vertical_span"
    assert semantic["square_width_equals_length"] is True
    assert semantic["z_used_as_physical_width"] is False


def test_square_2x2_object_semantics_keep_width_equal_to_length() -> None:
    """Square arrangements larger than 1x1 are still single-view squares."""
    semantic = combine_object_dimensions_for_arrangement(
        job_metadata={"arrangement_type": "2x2", "view_mode": "single_view"},
        view_features=[
            {
                "view_name": "view_01",
                "view_role": "single_or_length",
                "features": {
                    "point_cloud_frame": "rgb_camera",
                    "point_cloud_units": "meters",
                    "object_horizontal_span_m": 0.25,
                    "object_vertical_span_m": 0.50,
                    "object_depth_thickness_m": 0.10,
                },
            }
        ],
    )
    dims = semantic["semantic_dimensions_in"]
    assert abs(dims["length_in"] - dims["width_in"]) < 1e-9
    assert abs(dims["width_in"] - 9.842525) < 1e-9


def test_rectangular_two_view_object_semantics_use_view_horizontal_spans() -> None:
    """Rectangles use view_01 X for length, view_02 X for width, and max Y for height."""
    semantic = combine_object_dimensions_for_arrangement(
        job_metadata={"arrangement_type": "1x1", "view_mode": "two_view_rectangle"},
        view_features=[
            {
                "view_name": "view_01",
                "view_role": "length_view",
                "features": {
                    "point_cloud_frame": "rgb_camera",
                    "point_cloud_units": "meters",
                    "object_horizontal_span_m": 0.40,
                    "object_vertical_span_m": 0.60,
                    "object_depth_thickness_m": 0.10,
                    "object_robust_horizontal_span_m": 0.38,
                },
            },
            {
                "view_name": "view_02",
                "view_role": "width_view",
                "features": {
                    "point_cloud_frame": "rgb_camera",
                    "point_cloud_units": "meters",
                    "object_horizontal_span_m": 0.25,
                    "object_vertical_span_m": 0.58,
                    "object_depth_thickness_m": 0.12,
                    "object_robust_horizontal_span_m": 0.23,
                },
            },
        ],
    )
    dims = semantic["semantic_dimensions_in"]
    assert abs(dims["length_in"] - 14.960638) < 1e-9
    assert abs(dims["width_in"] - 9.055123) < 1e-9
    assert abs(dims["height_in"] - 23.62206) < 1e-9
    assert semantic["semantic_mapping"]["length_source"] == "view_01.horizontal_span_robust_1_99"
    assert semantic["semantic_mapping"]["width_source"] == "view_02.horizontal_span_robust_1_99"


def test_vertical_extent_is_positive_for_inverted_y_coordinates() -> None:
    """Negative or inverted Y coordinates collapse to a positive vertical extent before semantics."""
    semantic = combine_object_dimensions_for_arrangement(
        job_metadata={"arrangement_type": "1x1", "view_mode": "single_view"},
        view_features=[
            {
                "view_name": "view_01",
                "view_role": "single_or_length",
                "features": {
                    "point_cloud_frame": "rgb_camera",
                    "point_cloud_units": "meters",
                    "object_horizontal_span_m": 0.25,
                    "object_vertical_span_m": abs(-0.20 - 0.30),
                    "object_depth_thickness_m": 0.10,
                },
            }
        ],
    )
    assert abs(semantic["semantic_dimensions_in"]["height_in"] - 19.68505) < 1e-9


def test_meters_to_inches_conversion_occurs_once() -> None:
    """Raw meter spans are multiplied by the conversion factor exactly once."""
    semantic = combine_object_dimensions_for_arrangement(
        job_metadata={"arrangement_type": "1x1", "view_mode": "single_view"},
        view_features=[
            {
                "view_name": "view_01",
                "view_role": "single_or_length",
                "features": {
                    "point_cloud_frame": "rgb_camera",
                    "point_cloud_units": "meters",
                    "object_horizontal_span_m": 1.0,
                    "object_vertical_span_m": 1.0,
                    "object_depth_thickness_m": 1.0,
                },
            }
        ],
    )
    assert semantic["semantic_dimensions_in"]["length_in"] == M_TO_IN
    assert semantic["units_conversion_factor"] == M_TO_IN


def test_missing_units_or_unknown_frame_fail_loudly_for_rgb_spans() -> None:
    """Explicit RGB-camera spans require explicit meter units and frame metadata."""
    try:
        combine_object_dimensions_for_arrangement(
            job_metadata={"arrangement_type": "1x1", "view_mode": "single_view"},
            view_features=[
                {
                    "view_name": "view_01",
                    "view_role": "single_or_length",
                    "features": {
                        "object_horizontal_span_m": 0.25,
                        "object_vertical_span_m": 0.50,
                        "object_depth_thickness_m": 0.10,
                    },
                }
            ],
        )
    except ValueError as exc:
        assert "object_dimension_unit_contract_invalid" in str(exc)
    else:
        raise AssertionError("missing point-cloud units/frame must fail")


def test_features_and_combined_features_obey_square_arrangement_policy() -> None:
    """features.json and combined_features.json use square semantics from explicit RGB spans."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        job_dir = _write_arranged_geometry_job(
            cfg,
            cfg.job_type_single,
            "job_square_semantics",
            arrangement_type="1x1",
            view_mode="single_view",
            view_geometries={
                "view_01": _rgb_camera_geometry(
                    horizontal_m=0.25,
                    vertical_m=0.50,
                    depth_m=0.10,
                )
            },
        )
        geometry_path = job_dir / "view_01" / cfg.geometry_filename
        geometry = read_json(geometry_path)
        geometry["status"] = "ok"
        geometry["warnings"].append("metric_unit_assumed_millimeters")
        write_json_atomic(geometry_path, geometry)

        combined = write_combined_features(
            cfg,
            cfg.job_type_single,
            "job_square_semantics",
            view_names=["view_01"],
        )
        features = read_json(job_dir / "view_01" / cfg.features_filename)
        combined_on_disk = read_json(job_dir / cfg.combined_features_filename)
        combined_debug = read_json(job_dir / "debug" / "combined_features_debug.json")
        semantic_debug = read_json(job_dir / "debug" / "geometry_semantic_debug.json")

        assert features["feature_schema_version"] == "v1.1"
        assert features["ai2_features"]["object_length_in"] == 9.842525
        assert features["ai2_features"]["object_width_in"] == 9.842525
        assert features["ai2_features"]["object_height_in"] == 19.68505
        for removed_key in (
            "object_length_robust_in",
            "object_width_robust_in",
            "object_height_robust_in",
            "combined_length_candidate_in",
            "combined_width_candidate_in",
            "combined_height_candidate_in",
        ):
            assert removed_key not in features["ai2_features"]
        assert "metric_unit_assumed_millimeters" not in features["quality_flags"]["warnings"]
        assert combined == combined_on_disk
        assert combined_on_disk["feature_schema_version"] == "v1.1"
        assert combined_on_disk["ai2_features"]["object_length_in"] == 9.842525
        assert combined_on_disk["ai2_features"]["object_width_in"] == 9.842525
        assert combined_on_disk["ai2_features"]["object_height_in"] == 19.68505
        assert combined_on_disk["ai2_features"]["arrangement"] == "1x1"
        assert combined_on_disk["ai2_features"]["geometry_trusted"] is True
        assert combined_on_disk["ai2_features"]["authoritative_object_source"] == "geometry_primary"
        assert "object_density_center_of_mass_z" not in combined_on_disk["ai2_features"]
        assert "object_z_p50" not in combined_on_disk["ai2_features"]
        for removed_key in (
            "object_length_robust_in",
            "object_width_robust_in",
            "object_height_robust_in",
            "combined_length_candidate_in",
            "combined_width_candidate_in",
            "combined_height_candidate_in",
        ):
            assert removed_key not in combined_on_disk["ai2_features"]
        assert combined_debug["combined_candidates"]["combined_length_candidate_in"] == 9.842525
        assert combined_debug["combined_candidates"]["combined_width_candidate_in"] == 9.842525
        assert combined_debug["combined_candidates"]["combined_height_candidate_in"] == 19.68505
        assert "metric_unit_assumed_millimeters" not in combined_on_disk["quality_flags"]["warnings"]
        assert semantic_debug["cloud_path"] == "view_01/object_cloud.ply"
        assert semantic_debug["raw_axis_extents"][0]["x_span_m"] == 0.25
        assert semantic_debug["raw_axis_extents"][0]["y_span_m"] == 0.50
        assert semantic_debug["raw_axis_extents"][0]["z_span_m"] == 0.10
        assert semantic_debug["semantic_source_used_for_length"] == "view_01.horizontal_span_robust_1_99"
        assert semantic_debug["semantic_source_used_for_width"] == "view_01.horizontal_span_robust_1_99"
        assert semantic_debug["semantic_source_used_for_height"] == "view_01.vertical_span"
        assert semantic_debug["final_ai2_dimension_writers"]["object_length_in"] == "features.extractor._combined_ai2_contract"
        assert semantic_debug["square_width_equals_length"] is True
        assert semantic_debug["confirmation_z_not_used_as_physical_width"] is True


def test_quality_summary_uses_geometry_primary_for_object_pass() -> None:
    """Missing AI1 object segmentation is non-blocking when geometry-primary is trusted."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        job_dir = _write_arranged_geometry_job(
            cfg,
            cfg.job_type_single,
            "job_geometry_primary_object_status",
            arrangement_type="1x1",
            view_mode="single_view",
            view_geometries={
                "view_01": _rgb_camera_geometry(
                    horizontal_m=0.25,
                    vertical_m=0.50,
                    depth_m=0.10,
                )
            },
        )
        geometry_path = job_dir / "view_01" / cfg.geometry_filename
        geometry = read_json(geometry_path)
        geometry["status"] = "ok"
        geometry["cloud_quality"] = {"object": _object_cloud_quality()}
        write_json_atomic(geometry_path, geometry)
        write_json_atomic(
            job_dir / "view_01" / "segmentation.json",
            {
                "status": "partial",
                "segments": {
                    "object": "missing",
                    "pot": "missing",
                    "leaf": "missing",
                    "table": "missing",
                },
                "pot_quality": {
                    "status": "missing",
                    "usable_for_model": False,
                    "usable_for_geometry": False,
                },
            },
        )

        combined = write_combined_features(
            cfg,
            cfg.job_type_single,
            "job_geometry_primary_object_status",
            view_names=["view_01"],
        )
        summary = build_quality_summary(
            cfg,
            job_type=cfg.job_type_single,
            job_id="job_geometry_primary_object_status",
            write=False,
        )

        assert combined["ai2_features"]["geometry_trusted"] is True
        assert summary["checks"]["object"]["status"] == "pass"
        assert summary["checks"]["object_cloud"]["status"] == "pass"
        assert summary["decision"] != "recapture"


def test_combined_features_obey_rectangular_arrangement_policy() -> None:
    """combined_features.json maps two rectangular views by horizontal spans."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        job_dir = _write_arranged_geometry_job(
            cfg,
            cfg.job_type_group,
            "job_rect_semantics",
            arrangement_type="2x5",
            view_mode="two_view_rectangle",
            view_geometries={
                "view_01": _rgb_camera_geometry(horizontal_m=0.40, vertical_m=0.60, depth_m=0.10),
                "view_02": _rgb_camera_geometry(horizontal_m=0.25, vertical_m=0.58, depth_m=0.12),
            },
        )

        combined = write_combined_features(
            cfg,
            cfg.job_type_group,
            "job_rect_semantics",
            view_names=["view_01", "view_02"],
        )
        combined_on_disk = read_json(job_dir / cfg.combined_features_filename)
        semantic_debug = read_json(job_dir / "debug" / "geometry_semantic_debug.json")

        assert combined == combined_on_disk
        assert abs(combined_on_disk["ai2_features"]["object_length_in"] - 15.74804) < 1e-9
        assert abs(combined_on_disk["ai2_features"]["object_width_in"] - 9.842525) < 1e-9
        assert abs(combined_on_disk["ai2_features"]["object_height_in"] - 23.62206) < 1e-9
        assert "combined_length_candidate_in" not in combined_on_disk["ai2_features"]
        assert "object_length_robust_in" not in combined_on_disk["ai2_features"]
        assert semantic_debug["final_semantic_mapping"]["length_source"] == "view_01.horizontal_span_robust_1_99"
        assert semantic_debug["final_semantic_mapping"]["width_source"] == "view_02.horizontal_span_robust_1_99"


def test_fake_single_pipeline() -> None:
    """Test the fake single-view scan pipeline."""
    cfg = DimScanConfig()
    job_id = "job_test_fake_pipeline_single_000001"
    result = run_fake_data_collection_scan(
        cfg,
        job_id=job_id,
        job_type=cfg.job_type_single,
        arrangement_type="1x1",
        items=[make_item("fern_8in", quantity=1)],
        operator_id="test_operator",
    )

    assert result["view_names"] == ["view_01"]
    assert "combined_features" in result
    assert "box_suggestion" in result

    report = inspect_job(cfg, cfg.job_type_single, job_id)
    job_dir = get_job_dir(cfg, cfg.job_type_single, job_id)
    assert report["exists"]
    assert report["job_files"][cfg.session_filename]
    assert report["job_files"][cfg.combined_features_filename]
    assert (job_dir / "item_metadata.json").exists()
    assert "view_01" in report["views"]
    assert report["views"]["view_01"][cfg.cloud_filename]
    assert report["views"]["view_01"][cfg.geometry_filename]
    assert report["views"]["view_01"][cfg.features_filename]


def test_fake_group_pipeline() -> None:
    """Test the fake two-view group scan pipeline."""
    cfg = DimScanConfig()
    job_id = "job_test_fake_pipeline_group_000001"
    result = run_fake_data_collection_scan(
        cfg,
        job_id=job_id,
        job_type=cfg.job_type_group,
        arrangement_type="2x5",
        items=[
            make_item("fern_6in", quantity=6),
            make_item("pothos_6in", quantity=4),
        ],
        operator_id="test_operator",
    )

    assert result["view_names"] == ["view_01", "view_02"]
    assert "box_suggestion" in result

    report = inspect_job(cfg, cfg.job_type_group, job_id)
    for view_name in ["view_01", "view_02"]:
        assert view_name in report["views"]
        assert report["views"][view_name][cfg.cloud_filename]
        assert report["views"][view_name][cfg.geometry_filename]
        assert report["views"][view_name][cfg.features_filename]


def test_committed_collection_capture_preserves_exact_case_and_ai2_identity() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        job_id = "PE1G1587_20260714_133747"
        items = [make_item("pe1g1587", quantity=1)]
        _commit_test_collection_job(
            cfg,
            job_id=job_id,
            job_type=cfg.job_type_single,
            arrangement_type="1x1",
            items=items,
        )
        job_dir = get_job_dir(cfg, cfg.job_type_single, job_id)

        def fake_process_job(cfg_arg, *, job_type, job_id, **kwargs):
            combined = {
                "feature_schema_version": "v1.1",
                "ai2_features": {
                    "object_available": True,
                    "geometry_trusted": True,
                    "object_length_in": 10.0,
                    "object_width_in": 8.0,
                    "object_height_in": 20.0,
                },
                "quality_flags": {},
            }
            quality = {"quality_summary": {"decision": "proceed"}}
            write_json_atomic(job_dir / cfg_arg.combined_features_filename, combined)
            write_json_atomic(job_dir / cfg_arg.quality_summary_filename, quality)
            return {
                "view_geometry": {"view_01": {"status": "ok"}},
                "view_segmentation": {"view_01": {"status": "partial"}},
                "combined_features": combined,
                "quality_summary": quality["quality_summary"],
            }

        with patch("pipeline.real_capture.process_job", side_effect=fake_process_job):
            result = run_real_data_collection_scan(
                cfg,
                job_id=job_id,
                job_type=cfg.job_type_single,
                arrangement_type="1x1",
                items=items,
                camera=FakeCamera(),
                prompt_for_views=False,
            )

        record_ground_truth(
            cfg,
            cfg.job_type_single,
            result["job_id"],
            length_in=10,
            width_in=8,
            height_in=20,
        )
        assert result["job_id"] == job_id
        assert sorted(path.name for path in cfg.single_jobs_dir.iterdir()) == [job_id]
        assert not (cfg.single_jobs_dir / job_id.lower()).exists()
        assert read_json(job_dir / cfg.job_metadata_filename)["job_id"] == job_id
        assert read_json(job_dir / cfg.session_filename)["job_id"] == job_id
        for relative_path in (
            "view_01/rgb.png",
            "view_01/depth_raw.npy",
            "view_01/capture_meta.json",
            cfg.combined_features_filename,
            cfg.quality_summary_filename,
            cfg.ground_truth_filename,
        ):
            assert (job_dir / relative_path).is_file()
        rows, report = build_training_table(cfg.single_jobs_dir, job_type="single")
        assert report["included"] == 1
        assert [row["job_id"] for row in rows] == [job_id]


def test_explicit_collection_capture_rejects_missing_or_mismatched_commit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        missing_id = "PE1G1587_20260714_140000"
        items = [make_item("pe1g1587", quantity=1)]
        try:
            run_real_data_collection_scan(
                cfg,
                job_id=missing_id,
                job_type=cfg.job_type_single,
                arrangement_type="1x1",
                items=items,
                camera=FakeCamera(),
                prompt_for_views=False,
            )
        except FileNotFoundError as exc:
            assert "Committed collection job directory does not exist" in str(exc)
        else:
            raise AssertionError("missing committed job was accepted")
        assert not cfg.single_jobs_dir.exists()

        job_id = "PE1G1587_20260714_140001"
        _commit_test_collection_job(
            cfg,
            job_id=job_id,
            job_type=cfg.job_type_single,
            arrangement_type="1x1",
            items=items,
        )
        job_dir = get_job_dir(cfg, cfg.job_type_single, job_id)
        metadata = read_json(job_dir / cfg.job_metadata_filename)
        metadata["job_id"] = "DIFFERENT_20260714_140001"
        write_json_atomic(job_dir / cfg.job_metadata_filename, metadata)
        before = sorted(path.name for path in cfg.single_jobs_dir.iterdir())
        try:
            run_real_data_collection_scan(
                cfg,
                job_id=job_id,
                job_type=cfg.job_type_single,
                arrangement_type="1x1",
                items=items,
                camera=FakeCamera(),
                prompt_for_views=False,
            )
        except ValueError as exc:
            assert "metadata job_id mismatch" in str(exc)
        else:
            raise AssertionError("mismatched committed job metadata was accepted")
        assert sorted(path.name for path in cfg.single_jobs_dir.iterdir()) == before


def test_single_and_group_generated_collection_ids_remain_reusable_for_capture() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        cases = (
            (cfg.job_type_single, "SINGLE_20260714_140002", "1x1"),
            (cfg.job_type_group, "GROUP_20260714_140003", "1x2"),
        )
        for job_type, job_id, arrangement_type in cases:
            items = [make_item("unknown_sku", quantity=1 if job_type == "single" else 2)]
            _commit_test_collection_job(
                cfg,
                job_id=job_id,
                job_type=job_type,
                arrangement_type=arrangement_type,
                items=items,
            )
            result = run_real_data_collection_scan(
                cfg,
                job_id=job_id,
                job_type=job_type,
                arrangement_type=arrangement_type,
                items=items,
                camera=FakeCamera(),
                prompt_for_views=False,
                write_features=False,
            )
            assert result["job_id"] == job_id
            assert get_job_dir(cfg, job_type, job_id).is_dir()


def test_blank_capture_job_id_keeps_intentional_collection_creation_fallback() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        items = [make_item("unknown_sku", quantity=1)]
        result = run_real_data_collection_scan(
            cfg,
            job_id=None,
            job_type=cfg.job_type_single,
            arrangement_type="1x1",
            items=items,
            camera=FakeCamera(),
            prompt_for_views=False,
            write_features=False,
        )
        assert result["job_id"].startswith("SINGLE_")
        assert get_job_dir(cfg, cfg.job_type_single, result["job_id"]).is_dir()


def test_same_job_recapture_replaces_only_selected_view_and_preserves_job_state() -> None:
    """Recapture cleans one selected view while preserving job-level state and other views."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        job_id = "job_same_job_recapture"
        items = [
            make_item("fern_6in", quantity=6),
            make_item("pothos_6in", quantity=4),
        ]
        camera = FakeCamera()
        _commit_test_collection_job(
            cfg,
            job_id=job_id,
            job_type=cfg.job_type_group,
            arrangement_type="2x5",
            items=items,
        )

        run_real_data_collection_scan(
            cfg,
            job_id=job_id,
            job_type=cfg.job_type_group,
            arrangement_type="2x5",
            items=items,
            camera=camera,
            prompt_for_views=False,
            view_id="view_01",
            write_features=False,
        )
        run_real_data_collection_scan(
            cfg,
            job_id=job_id,
            job_type=cfg.job_type_group,
            arrangement_type="2x5",
            items=items,
            camera=camera,
            prompt_for_views=False,
            view_id="view_02",
            write_features=False,
        )
        record_ground_truth(
            cfg,
            cfg.job_type_group,
            job_id,
            length_in=18,
            width_in=12,
            height_in=20,
            source="manual_test",
        )
        job_dir = get_job_dir(cfg, cfg.job_type_group, job_id)
        stale_file = job_dir / "view_01" / "debug" / "stale_from_old_capture.txt"
        stale_file.parent.mkdir(parents=True, exist_ok=True)
        stale_file.write_text("old", encoding="utf-8")
        preserved_file = job_dir / "view_02" / "debug" / "preserve.txt"
        preserved_file.parent.mkdir(parents=True, exist_ok=True)
        preserved_file.write_text("keep", encoding="utf-8")
        original_items = read_json(job_dir / cfg.item_list_filename)
        original_gt = read_json(job_dir / cfg.ground_truth_filename)
        process_calls: list[dict[str, object]] = []

        def fake_process_job(
            cfg_arg,
            *,
            job_type,
            job_id,
            view_names=None,
            process_view_names=None,
            force_segmentation=False,
            debug_mode=True,
            skip_valid_views=False,
            write_combined=True,
            quality_view_name=None,
            geometry_primary_fast=False,
        ):
            process_calls.append(
                {
                    "view_names": list(view_names or []),
                    "process_view_names": list(process_view_names or []),
                    "skip_valid_views": skip_valid_views,
                    "write_combined": write_combined,
                    "quality_view_name": quality_view_name,
                    "geometry_primary_fast": geometry_primary_fast,
                    "force_segmentation": force_segmentation,
                }
            )
            payload = {"ai2_features": {"geometry_trusted": True}, "quality_flags": {}}
            write_json_atomic(get_job_dir(cfg_arg, job_type, job_id) / cfg_arg.combined_features_filename, payload)
            return {
                "view_geometry": {"view_01": {"status": "ok"}},
                "view_segmentation": {"view_01": {"status": "partial"}},
                "combined_features": payload,
                "quality_summary": {"decision": "proceed", "checks": {"object": {"status": "pass"}}},
            }

        with patch("pipeline.real_capture.process_job", side_effect=fake_process_job):
            result = run_real_data_collection_scan(
                cfg,
                job_id=job_id,
                job_type=cfg.job_type_group,
                arrangement_type="2x5",
                items=items,
                camera=camera,
                prompt_for_views=False,
                view_id="view_01",
                overwrite=True,
            )

        assert result["job_id"] == job_id
        assert result["current_view_id"] == "view_01"
        assert result["captured_views"] == ["view_01", "view_02"]
        assert process_calls == [
            {
                "view_names": ["view_01", "view_02"],
                "process_view_names": ["view_01"],
                "skip_valid_views": True,
                "write_combined": True,
                "quality_view_name": "view_01",
                "geometry_primary_fast": False,
                "force_segmentation": True,
            }
        ]
        assert not stale_file.exists()
        assert (job_dir / "view_01" / cfg.cloud_filename).is_file()
        assert preserved_file.read_text(encoding="utf-8") == "keep"
        assert read_json(job_dir / cfg.item_list_filename) == original_items
        assert read_json(job_dir / cfg.ground_truth_filename) == original_gt


def test_rectangular_real_capture_waits_for_second_view_before_finalizing() -> None:
    """Rectangular capture records View 01 progress and finalizes only after View 02."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        job_id = "job_rectangular_capture_sequence"
        items = [make_item("fern_6in", quantity=2)]
        camera = FakeCamera()
        process_calls: list[dict[str, object]] = []
        _commit_test_collection_job(
            cfg,
            job_id=job_id,
            job_type=cfg.job_type_group,
            arrangement_type="1x2",
            items=items,
        )

        def fake_first_process_job(
            cfg_arg,
            *,
            job_type,
            job_id,
            view_names=None,
            process_view_names=None,
            force_segmentation=False,
            debug_mode=True,
            skip_valid_views=False,
            write_combined=True,
            quality_view_name=None,
            geometry_primary_fast=False,
        ):
            process_calls.append(
                {
                    "job_id": job_id,
                    "view_names": list(view_names or []),
                    "process_view_names": list(process_view_names or []),
                    "skip_valid_views": skip_valid_views,
                    "write_combined": write_combined,
                    "quality_view_name": quality_view_name,
                    "geometry_primary_fast": geometry_primary_fast,
                    "force_segmentation": force_segmentation,
                }
            )
            return {
                "view_geometry": {"view_01": {"status": "ok"}},
                "view_segmentation": {"view_01": {"status": "partial"}},
                "combined_features": None,
                "quality_summary": None,
            }

        with patch("pipeline.real_capture.process_job", side_effect=fake_first_process_job):
            first = run_real_data_collection_scan(
                cfg,
                job_id=job_id,
                job_type=cfg.job_type_group,
                arrangement_type="1x2",
                items=items,
                camera=camera,
                prompt_for_views=False,
            )

        job_dir = get_job_dir(cfg, cfg.job_type_group, job_id)
        view_01_cloud = job_dir / "view_01" / cfg.cloud_filename
        assert first["current_view_id"] == "view_01"
        assert first["captured_views"] == ["view_01"]
        assert first["remaining_views"] == ["view_02"]
        assert "Width Side" in first["next_action"]
        assert "combined_features" not in first
        assert not (job_dir / cfg.combined_features_filename).exists()
        assert process_calls == [
            {
                "job_id": job_id,
                "view_names": ["view_01"],
                "process_view_names": ["view_01"],
                "skip_valid_views": False,
                "write_combined": False,
                "quality_view_name": None,
                "geometry_primary_fast": False,
                "force_segmentation": True,
            }
        ]
        assert view_01_cloud.is_file()
        process_calls.clear()

        def fake_process_job(
            cfg_arg,
            *,
            job_type,
            job_id,
            view_names=None,
            process_view_names=None,
            force_segmentation=False,
            debug_mode=True,
            skip_valid_views=False,
            write_combined=True,
            quality_view_name=None,
            geometry_primary_fast=False,
        ):
            process_calls.append(
                {
                    "job_id": job_id,
                    "view_names": list(view_names or []),
                    "process_view_names": list(process_view_names or []),
                    "skip_valid_views": skip_valid_views,
                    "write_combined": write_combined,
                    "quality_view_name": quality_view_name,
                    "geometry_primary_fast": geometry_primary_fast,
                    "force_segmentation": force_segmentation,
                }
            )
            assert debug_mode is False
            return {
                "view_geometry": {"view_01": {"status": "ok"}, "view_02": {"status": "ok"}},
                "view_segmentation": {"view_01": {"status": "partial"}, "view_02": {"status": "partial"}},
                "combined_features": {"ai2_features": {"object_length_in": 10.0}},
                "quality_summary": {"decision": "proceed"},
            }

        with patch("pipeline.real_capture.process_job", side_effect=fake_process_job):
            second = run_real_data_collection_scan(
                cfg,
                job_id=job_id,
                job_type=cfg.job_type_group,
                arrangement_type="1x2",
                items=items,
                camera=camera,
                prompt_for_views=False,
                debug_mode=False,
            )

        assert second["current_view_id"] == "view_02"
        assert second["captured_views"] == ["view_01", "view_02"]
        assert second["remaining_views"] == []
        assert second["combined_features"]["ai2_features"]["object_length_in"] == 10.0
        assert process_calls == [
            {
                "job_id": job_id,
                "view_names": ["view_01", "view_02"],
                "process_view_names": ["view_02"],
                "skip_valid_views": True,
                "write_combined": True,
                "quality_view_name": None,
                "geometry_primary_fast": False,
                "force_segmentation": True,
            }
        ]
        assert view_01_cloud.is_file()
        assert second["capture_artifacts_by_view"]["view_02"]["cloud"] == ""
        assert not (job_dir / "view_02" / cfg.cloud_filename).exists()
        assert (job_dir / "view_02" / cfg.rgb_filename).is_file()
        assert (job_dir / "view_02" / cfg.depth_raw_filename).is_file()
        assert (job_dir / "view_02" / cfg.capture_meta_filename).is_file()


def test_debug_off_dense_cloud_construction_is_skipped_when_yoloe_model_configured() -> None:
    """Debug Mode off skips dense cloud construction even when YOLOE is configured."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        cfg.yoloe_model_name = "configured-yoloe"
        job_id = "job_debug_off_yoloe_cloud"
        items = [make_item("fern_6in", quantity=1)]
        _commit_test_collection_job(
            cfg,
            job_id=job_id,
            job_type=cfg.job_type_single,
            arrangement_type="1x1",
            items=items,
        )

        def fake_process_job(
            cfg_arg,
            *,
            job_type,
            job_id,
            view_names=None,
            process_view_names=None,
            force_segmentation=False,
            debug_mode=True,
            skip_valid_views=False,
            write_combined=True,
            quality_view_name=None,
            geometry_primary_fast=False,
        ):
            return {
                "view_geometry": {"view_01": {"status": "ok"}},
                "view_segmentation": {"view_01": {"status": "partial"}},
                "combined_features": {"ai2_features": {"object_length_in": 10.0}},
                "quality_summary": {"decision": "proceed"},
            }

        with (
            patch("pipeline.real_capture.process_job", side_effect=fake_process_job),
            patch("pipeline.scan_writer.metric_points_from_depth", side_effect=AssertionError("dense metric cloud built")),
            patch("pipeline.scan_writer.valid_depth_pixel_indices_from_depth", side_effect=AssertionError("dense pixel index map built")),
            patch("pipeline.scan_writer.colors_from_rgb_bytes", side_effect=AssertionError("dense color map built")),
            patch("pipeline.scan_writer.write_ascii_ply_with_colors", side_effect=AssertionError("dense colored PLY written")),
            patch("pipeline.scan_writer.write_ascii_ply", side_effect=AssertionError("dense PLY written")),
            patch("pipeline.scan_writer.write_pointcloud_from_depth", side_effect=AssertionError("fallback dense PLY written")),
        ):
            result = run_real_data_collection_scan(
                cfg,
                job_id=job_id,
                job_type=cfg.job_type_single,
                arrangement_type="1x1",
                items=items,
                camera=FakeCamera(),
                prompt_for_views=False,
                debug_mode=False,
            )

        job_dir = get_job_dir(cfg, cfg.job_type_single, job_id)
        view_dir = job_dir / "view_01"
        assert result["capture_artifacts_by_view"]["view_01"]["cloud"] == ""
        assert result["capture_artifacts_by_view"]["view_01"]["cloud_pixel_indices"] == ""
        assert not (view_dir / cfg.cloud_filename).exists()
        assert not (view_dir / "cloud_pixel_indices.npy").exists()
        capture_meta = read_json(view_dir / cfg.capture_meta_filename)
        assert capture_meta["cloud_pixel_indices"] is None
        assert capture_meta["cloud_pixel_indices_count"] is None
        assert capture_meta["cloud_pixel_indices_debug_only"] is False


def test_debug_off_skips_view_debug_files_without_changing_authoritative_outputs() -> None:
    """Debug Mode off skips view diagnostics while production outputs still match Debug Mode on."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        geometry = _base_geometry(
            pot_quality={
                "status": "fallback",
                "usable_for_geometry": False,
                "usable_for_model": False,
                "trust_score": 0.0,
                "reason": "fallback_pot_mask_debug_only",
                "metrics": {},
            },
            pot_dimensions=None,
            pot_segment_status="fallback",
            leaf_dimensions=None,
            leaf_segment_status="fallback",
            leaf_fallback_quality=True,
        )
        job_ids = {True: "job_debug_on_outputs", False: "job_debug_off_outputs"}

        def fake_process_view(
            cfg_arg,
            job_type_arg,
            job_id_arg,
            *,
            view_name,
            force_segmentation=False,
            debug_mode=True,
            geometry_primary_fast=False,
        ):
            job_dir = get_job_dir(cfg, job_type_arg, job_id_arg)
            return {
                "segmentation": {
                    "status": "partial",
                    "segments": {"object": "ok", "pot": "fallback", "leaf": "fallback", "table": "missing"},
                    "pot_quality": geometry["pot_quality"],
                    "warnings": ["leaf_mask_fallback_from_object", "pot_mask_fallback_from_object"],
                },
                "geometry": read_json(job_dir / view_name / cfg.geometry_filename),
            }

        outputs = {}
        for debug_mode in (True, False):
            job_id = job_ids[debug_mode]
            job_dir = _write_minimal_job(cfg, job_id, json.loads(json.dumps(geometry)))
            with patch("pipeline.process_job.process_view", side_effect=fake_process_view):
                result = process_job(
                    cfg,
                    job_type=cfg.job_type_single,
                    job_id=job_id,
                    view_names=["view_01"],
                    process_view_names=["view_01"],
                    debug_mode=debug_mode,
                )
            view_dir = job_dir / "view_01"
            outputs[debug_mode] = {
                "features": read_json(view_dir / cfg.features_filename),
                "combined": read_json(job_dir / cfg.combined_features_filename),
                "quality": result["quality_summary"],
            }
            assert (view_dir / cfg.features_filename).is_file()
            assert (job_dir / cfg.combined_features_filename).is_file()
            assert (view_dir / cfg.quality_summary_filename).is_file()
            if debug_mode:
                assert (view_dir / "debug" / "pot_debug.json").is_file()
                assert (job_dir / "debug" / "combined_features_debug.json").is_file()
            else:
                assert not (view_dir / "debug").exists()

        assert outputs[False]["features"] == outputs[True]["features"]
        assert outputs[False]["combined"] == outputs[True]["combined"]
        assert outputs[False]["quality"]["decision"] == outputs[True]["quality"]["decision"]
        assert outputs[False]["quality"]["status_label"] == outputs[True]["quality"]["status_label"]
        assert outputs[False]["quality"]["checks"]["gt"] == outputs[True]["quality"]["checks"]["gt"]
        for key in LEAF_AI2_SUPPRESSED_FIELDS:
            assert outputs[False]["features"]["ai2_features"][key] is None


def test_real_capture_result_strips_runtime_arrays_before_json_response() -> None:
    """In-memory object points remain internal and never leak into capture JSON."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        job_id = "job_runtime_array_response"
        runtime_points = np.asarray([[0.0, 0.0, 1.0], [0.01, 0.02, 1.03]], dtype=float)
        items = [make_item("fern_6in", quantity=1)]
        _commit_test_collection_job(
            cfg,
            job_id=job_id,
            job_type=cfg.job_type_single,
            arrangement_type="1x1",
            items=items,
        )

        def fake_process_job(
            cfg_arg,
            *,
            job_type,
            job_id,
            view_names=None,
            process_view_names=None,
            force_segmentation=False,
            debug_mode=True,
            skip_valid_views=False,
            write_combined=True,
            quality_view_name=None,
            geometry_primary_fast=False,
        ):
            return {
                "view_geometry": {"view_01": {"status": "ok", "point_count": np.int64(2)}},
                "view_segmentation": {
                    "view_01": {
                        "status": "partial",
                        "_runtime_final_object_points": runtime_points,
                        "warnings": ["object_segment_missing"],
                    }
                },
                "combined_features": {
                    "ai2_features": {
                        "object_available": True,
                        "object_point_count": np.int64(2),
                    },
                    "quality_flags": {},
                },
                "quality_summary": {
                    "decision": "proceed",
                    "checks": {"object": {"status": "pass", "point_count": np.int64(2)}},
                },
            }

        with patch("pipeline.real_capture.process_job", side_effect=fake_process_job):
            result = run_real_data_collection_scan(
                cfg,
                job_id=job_id,
                job_type=cfg.job_type_single,
                arrangement_type="1x1",
                items=items,
                camera=FakeCamera(),
                prompt_for_views=False,
            )

        json.dumps(result)
        assert "_runtime_final_object_points" not in result["segmentation_by_view"]["view_01"]
        assert isinstance(result["geometry_by_view"]["view_01"]["point_count"], int)
        assert isinstance(result["combined_features"]["ai2_features"]["object_point_count"], int)


def test_process_job_single_sequential_pass_reuses_object_extraction() -> None:
    """One processed view extracts once, segments once, measures once, and writes once."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        job_id = "job_single_pass_counts"
        job_dir = get_job_dir(cfg, cfg.job_type_single, job_id)
        view_dir = job_dir / "view_01"
        view_dir.mkdir(parents=True)
        write_json_atomic(
            job_dir / cfg.job_metadata_filename,
            {"job_id": job_id, "job_type": cfg.job_type_single, "view_mode": "single_view"},
        )
        runtime_points = np.asarray([[0.0, 0.0, 1.0], [0.01, 0.02, 1.03]], dtype=float)
        extraction_result = {
            "status": "ok",
            "final_point_count": 2,
            "artifacts": {"object_cloud_status": "ready", "object_cloud": str(view_dir / "object_cloud.ply")},
            "_runtime_final_object_points": runtime_points,
        }
        geometry_writes: list[Path] = []

        def fake_segment_view(
            cfg_arg,
            view_dir_arg,
            *,
            force=False,
            debug_mode=True,
            object_extraction_result=None,
        ):
            assert force is True
            assert debug_mode is False
            assert Path(view_dir_arg) == view_dir
            assert object_extraction_result is extraction_result
            return {
                "status": "partial",
                "segments": {"object": "ok", "pot": "missing", "leaf": "missing", "table": "missing"},
                "_runtime_final_object_points": runtime_points,
            }

        def fake_measure_view_geometry(view_dir_arg, *, cfg, debug_mode=True, geometry_primary_object_points=None):
            assert Path(view_dir_arg) == view_dir
            assert debug_mode is False
            assert geometry_primary_object_points is runtime_points
            return _rgb_camera_geometry(horizontal_m=0.25, vertical_m=0.4, depth_m=0.1)

        def fake_write_json_atomic(path, payload):
            geometry_writes.append(Path(path))

        with (
            patch("pipeline.process_job.extract_geometry_primary_object_cloud", return_value=extraction_result) as extract_mock,
            patch("pipeline.process_job.segment_view", side_effect=fake_segment_view) as segment_mock,
            patch("pipeline.process_job.measure_view_geometry", side_effect=fake_measure_view_geometry) as measure_mock,
            patch("pipeline.process_job.write_json_atomic", side_effect=fake_write_json_atomic),
            patch("pipeline.process_job.write_view_features") as features_mock,
        ):
            result = process_job(
                cfg,
                job_type=cfg.job_type_single,
                job_id=job_id,
                view_names=["view_01"],
                process_view_names=["view_01"],
                force_segmentation=True,
                debug_mode=False,
                write_combined=False,
            )

        assert extract_mock.call_count == 1
        assert segment_mock.call_count == 1
        assert measure_mock.call_count == 1
        assert features_mock.call_count == 1
        assert geometry_writes == [view_dir / cfg.geometry_filename]
        assert result["processed_view_names"] == ["view_01"]


def test_yoloe_debug_off_uses_runtime_object_extraction_without_reextracting() -> None:
    """Debug Mode off does not make YOLOE use optional debug JSON as extraction proof."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        cfg.segmentation_backend = "yoloe"
        view_dir = Path(tmp) / "view_01"
        view_dir.mkdir()
        (view_dir / cfg.rgb_filename).write_bytes(b"rgb")
        (view_dir / cfg.depth_raw_filename).write_bytes(b"depth")
        (view_dir / "object_cloud.ply").write_text("ply\n", encoding="utf-8")
        runtime_points = np.asarray([[0.0, 0.0, 1.0], [0.01, 0.02, 1.03]], dtype=float)
        extraction_result = {
            "status": "ok",
            "final_point_count": 2,
            "artifacts": {"object_cloud_status": "ready", "object_cloud": str(view_dir / "object_cloud.ply")},
            "_runtime_final_object_points": runtime_points,
        }

        from segmentation.yoloe_segmenter import run_yoloe_segmentation

        with patch(
            "segmentation.yoloe_segmenter.extract_geometry_primary_object_cloud",
            side_effect=AssertionError("object extraction reran"),
        ) as extract_mock:
            record = run_yoloe_segmentation(
                cfg,
                view_dir,
                force=True,
                debug_mode=False,
                object_extraction_result=extraction_result,
            )

        assert extract_mock.call_count == 0
        assert record["segments"]["object"] == "ok"
        assert record["artifacts"]["object_cloud"] == str(view_dir / "object_cloud.ply")
        assert not (view_dir / "debug").exists()


def test_process_job_reuses_valid_view_outputs_for_incremental_finalize() -> None:
    """Final View 02 processing reuses valid View 01 artifacts."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        job_id = "job_incremental_reuse"
        job_dir = get_job_dir(cfg, cfg.job_type_group, job_id)
        job_dir.mkdir(parents=True)
        write_json_atomic(
            job_dir / cfg.job_metadata_filename,
            {
                "job_id": job_id,
                "mode": "data_collection",
                "job_type": cfg.job_type_group,
                "arrangement_type": "1x2",
                "view_mode": cfg.view_mode_two_view_rectangle,
            },
        )
        write_json_atomic(
            job_dir / cfg.item_list_filename,
            {"items": [make_item("fern_6in", quantity=1)], "total_quantity": 1, "unique_sku_count": 1},
        )

        def write_valid_view(view_name: str, *, width_m: float) -> None:
            view_dir = job_dir / view_name
            view_dir.mkdir()
            (view_dir / cfg.rgb_filename).write_bytes(b"rgb")
            (view_dir / cfg.depth_raw_filename).write_bytes(b"depth")
            write_json_atomic(view_dir / cfg.capture_meta_filename, {})
            (view_dir / cfg.cloud_filename).write_text("ply\n", encoding="utf-8")
            (view_dir / "object_cloud.ply").write_text("ply\n", encoding="utf-8")
            write_json_atomic(view_dir / "segmentation.json", {"status": "partial", "segments": {"object": "missing"}})
            geometry = _rgb_camera_geometry(horizontal_m=width_m, vertical_m=0.4, depth_m=0.1)
            geometry["source_cloud"] = "object_cloud"
            write_json_atomic(view_dir / cfg.geometry_filename, geometry)
            write_view_features(
                cfg,
                cfg.job_type_group,
                job_id,
                view_name=view_name,
                view_role="length_view" if view_name == "view_01" else "width_view",
            )

        write_valid_view("view_01", width_m=0.20)
        write_valid_view("view_02", width_m=0.30)
        view_01_features_before = (job_dir / "view_01" / cfg.features_filename).read_text(encoding="utf-8")
        processed: list[str] = []

        def fake_process_view(
            cfg_arg,
            job_type_arg,
            job_id_arg,
            *,
            view_name,
            force_segmentation=False,
            debug_mode=True,
            geometry_primary_fast=False,
        ):
            processed.append(view_name)
            return {
                "segmentation": read_json(job_dir / view_name / "segmentation.json"),
                "geometry": read_json(job_dir / view_name / cfg.geometry_filename),
            }

        with patch("pipeline.process_job.process_view", side_effect=fake_process_view):
            result = process_job(
                cfg,
                job_type=cfg.job_type_group,
                job_id=job_id,
                view_names=["view_01", "view_02"],
                process_view_names=["view_02"],
                skip_valid_views=True,
            )

        assert processed == ["view_02"]
        assert result["processed_view_names"] == ["view_02"]
        assert (job_dir / "view_01" / cfg.features_filename).read_text(encoding="utf-8") == view_01_features_before
        combined = read_json(job_dir / cfg.combined_features_filename)
        assert combined["ai2_features"]["object_length_in"] == 0.20 * M_TO_IN
        assert combined["ai2_features"]["object_width_in"] == 0.30 * M_TO_IN


def test_process_job_reprocesses_invalid_prior_view_during_incremental_finalize() -> None:
    """Invalid View 01 outputs are rebuilt before final two-view combination."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        job_id = "job_incremental_reprocess_invalid"
        job_dir = get_job_dir(cfg, cfg.job_type_group, job_id)
        job_dir.mkdir(parents=True)
        write_json_atomic(
            job_dir / cfg.job_metadata_filename,
            {
                "job_id": job_id,
                "mode": "data_collection",
                "job_type": cfg.job_type_group,
                "arrangement_type": "1x2",
                "view_mode": cfg.view_mode_two_view_rectangle,
            },
        )
        write_json_atomic(
            job_dir / cfg.item_list_filename,
            {"items": [make_item("fern_6in", quantity=1)], "total_quantity": 1, "unique_sku_count": 1},
        )
        for view_name in ("view_01", "view_02"):
            view_dir = job_dir / view_name
            view_dir.mkdir()
            (view_dir / cfg.rgb_filename).write_bytes(b"rgb")
            (view_dir / cfg.depth_raw_filename).write_bytes(b"depth")
            write_json_atomic(view_dir / cfg.capture_meta_filename, {})
            (view_dir / cfg.cloud_filename).write_text("ply\n", encoding="utf-8")
            (view_dir / "object_cloud.ply").write_text("ply\n", encoding="utf-8")
            write_json_atomic(view_dir / "segmentation.json", {"status": "partial", "segments": {"object": "missing"}})
            geometry = _rgb_camera_geometry(horizontal_m=0.25 if view_name == "view_01" else 0.35, vertical_m=0.4, depth_m=0.1)
            geometry["source_cloud"] = "object_cloud"
            write_json_atomic(view_dir / cfg.geometry_filename, geometry)
        write_view_features(cfg, cfg.job_type_group, job_id, view_name="view_02", view_role="width_view")

        processed: list[str] = []

        def fake_process_view(
            cfg_arg,
            job_type_arg,
            job_id_arg,
            *,
            view_name,
            force_segmentation=False,
            debug_mode=True,
            geometry_primary_fast=False,
        ):
            processed.append(view_name)
            return {
                "segmentation": read_json(job_dir / view_name / "segmentation.json"),
                "geometry": read_json(job_dir / view_name / cfg.geometry_filename),
            }

        with patch("pipeline.process_job.process_view", side_effect=fake_process_view):
            result = process_job(
                cfg,
                job_type=cfg.job_type_group,
                job_id=job_id,
                view_names=["view_01", "view_02"],
                process_view_names=["view_02"],
                skip_valid_views=True,
            )

        assert processed == ["view_01", "view_02"]
        assert result["processed_view_names"] == ["view_01", "view_02"]
        assert (job_dir / "view_01" / cfg.features_filename).is_file()


def test_dataset_row_after_ground_truth() -> None:
    """Test that a fake scan with ground truth becomes a dataset row."""
    cfg = DimScanConfig()
    job_id = "job_test_fake_pipeline_dataset_000001"
    run_fake_data_collection_scan(
        cfg,
        job_id=job_id,
        job_type=cfg.job_type_single,
        arrangement_type="1x1",
        items=[make_item("fern_8in", quantity=1)],
        operator_id="test_operator",
    )
    record_ground_truth(
        cfg,
        cfg.job_type_single,
        job_id,
        length_in=18,
        width_in=18,
        height_in=24,
        source="manual_test",
        fit="good",
        damage=False,
    )

    rows = build_rows(cfg, cfg.job_type_single)
    row = next(row for row in rows if row.get("job_id") == job_id)
    assert row["features_ai2_features_object_length_in"] is not None
    assert "features_ai2_features_object_z_p50" not in row
    assert "features_ai2_features_object_density_center_of_mass_z" not in row
    for removed_key in (
        "features_ai2_features_object_length_robust_in",
        "features_ai2_features_object_width_robust_in",
        "features_ai2_features_object_height_robust_in",
        "features_ai2_features_object_bbox_length_in",
        "features_ai2_features_object_bbox_width_in",
        "features_ai2_features_object_bbox_height_in",
        "features_ai2_features_combined_length_candidate_in",
        "features_ai2_features_combined_width_candidate_in",
        "features_ai2_features_combined_height_candidate_in",
    ):
        assert removed_key not in row


def run_tests() -> None:
    """Run fake pipeline assertions."""
    test_rejected_pot_debug_stays_out_of_model_features()
    test_unusable_pot_sources_are_missing_or_rejected_without_prior()
    test_combined_features_stay_clean_with_rejected_pot()
    test_combined_features_use_sku_prior_source_when_prior_supplies_pot()
    test_combined_features_use_catalogue_style_pot_prior()
    test_trusted_pot_dimensions_enter_ai2_features()
    test_fallback_leaf_features_are_suppressed_from_ai2()
    test_true_leaf_segmentation_can_populate_ai2_features()
    test_missing_leaf_segmentation_keeps_ai2_leaf_fields_null()
    test_sku_pot_prior_still_applies_when_leaf_is_fallback()
    test_square_1x1_object_semantics_use_horizontal_for_length_and_width()
    test_square_2x2_object_semantics_keep_width_equal_to_length()
    test_rectangular_two_view_object_semantics_use_view_horizontal_spans()
    test_vertical_extent_is_positive_for_inverted_y_coordinates()
    test_meters_to_inches_conversion_occurs_once()
    test_missing_units_or_unknown_frame_fail_loudly_for_rgb_spans()
    test_features_and_combined_features_obey_square_arrangement_policy()
    test_quality_summary_uses_geometry_primary_for_object_pass()
    test_combined_features_obey_rectangular_arrangement_policy()
    test_fake_single_pipeline()
    test_fake_group_pipeline()
    test_same_job_recapture_replaces_only_selected_view_and_preserves_job_state()
    test_real_capture_result_strips_runtime_arrays_before_json_response()
    test_debug_off_dense_cloud_is_kept_when_yoloe_model_configured()
    test_rectangular_real_capture_waits_for_second_view_before_finalizing()
    test_dataset_row_after_ground_truth()


if __name__ == "__main__":
    run_tests()
    print("test_fake_pipeline.py passed")
