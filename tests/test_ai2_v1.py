"""Focused tests for the isolated AI2 v1 baseline."""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from ai2.dataset import build_training_table, matrix_from_rows, model_features_from_ai2
from ai2.predict import default_model_dir, predict_ai2_for_job
from ai2.train import train_ai2_v1
from app.config import DimScanConfig
from app.server import create_app
from utils.io import read_json, write_json_atomic
from features.extractor import group_composition_features
from scripts import train_ai2_v1 as train_cli


def _temp_cfg(tmp_path: Path) -> DimScanConfig:
    datasets = tmp_path / "datasets"
    models = tmp_path / "models"
    return DimScanConfig(
        dataset_root=datasets,
        single_jobs_dir=datasets / "single" / "jobs",
        single_exports_dir=datasets / "single" / "exports",
        group_jobs_dir=datasets / "group" / "jobs",
        group_exports_dir=datasets / "group" / "exports",
        models_root=models,
        single_packing_models_dir=models / "packing" / "single",
        group_packing_models_dir=models / "packing" / "group",
    )


def _write_job(
    jobs_root: Path,
    job_id: str,
    *,
    completed: bool = True,
    object_available: bool = True,
    geometry_trusted: bool = True,
    label: tuple[float, float, float] = (10.0, 8.0, 20.0),
    sku_context: bool = False,
    write_combined: bool = True,
    write_gt: bool = True,
) -> Path:
    job_dir = jobs_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    if write_combined:
        ai2_features = {
            "object_available": object_available,
            "geometry_trusted": geometry_trusted,
            "object_length_in": 9.5,
            "object_width_in": 7.5,
            "object_height_in": 18.5,
            "object_point_count": 1200,
            "leaf_canopy_length_in": None,
            "source_file": "do-not-use.xlsx",
            "unit_price": 99.0,
            "uom": "EA",
            "raw_sku": "SECRET",
            "recorded_at": "2026-07-08T00:00:00Z",
        }
        if sku_context:
            ai2_features.update(
                {
                    "sku_category": "Annuals",
                    "sku_common_name": "Test Plant",
                    "sku_spec": "4.5 in Pot",
                }
            )
        write_json_atomic(
            job_dir / "combined_features.json",
            {
                "feature_schema_version": "v1.1",
                "ai2_features": ai2_features,
                "quality_flags": {},
            },
        )
    if write_gt:
        write_json_atomic(
            job_dir / "ground_truth.json",
            {
                "completed": completed,
                "source": "manual",
                "actual_box": {
                    "length_in": label[0],
                    "width_in": label[1],
                    "height_in": label[2],
                },
                "fit": "good",
                "damage": False,
                "notes": None,
                "recorded_at": "2026-07-08T00:00:00Z",
            },
        )
    return job_dir


def test_dataset_includes_valid_job_and_skips_bad_jobs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "jobs"
        _write_job(root, "valid")
        _write_job(root, "no_gt", write_gt=False)
        _write_job(root, "no_features", write_combined=False)
        _write_job(root, "incomplete", completed=False)
        _write_job(root, "no_object", object_available=False)
        _write_job(root, "untrusted", geometry_trusted=False)

        rows, report = build_training_table(root)

        assert [row["job_id"] for row in rows] == ["valid"]
        assert report["included"] == 1
        assert report["skipped"] == 5
        assert report["skip_reasons"] == {
            "ground_truth_incomplete": 1,
            "missing_combined_features": 1,
            "missing_ground_truth": 1,
            "object_unavailable": 1,
            "geometry_untrusted": 1,
        }


def test_dataset_excludes_gt_and_spreadsheet_noise_but_keeps_safe_sku_fields() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "jobs"
        _write_job(root, "valid", sku_context=True)

        rows, report = build_training_table(root)
        feature_names = set(report["feature_names"])

        assert rows[0]["label"] == {"length_in": 10.0, "width_in": 8.0, "height_in": 20.0}
        assert {"sku_category", "sku_common_name", "sku_spec"} <= feature_names
        assert "object_length_in" in feature_names
        assert "leaf_canopy_length_in" in feature_names
        assert "source_file" not in feature_names
        assert "unit_price" not in feature_names
        assert "uom" not in feature_names
        assert "raw_sku" not in feature_names
        assert "recorded_at" not in feature_names
        assert not any(name.startswith("actual_box") for name in feature_names)
        assert "job_id" not in feature_names


def test_model_feature_vector_flattens_density_arrays_and_approved_booleans() -> None:
    ai2_features = {
        "object_available": True,
        "geometry_trusted": True,
        "object_vertical_density_bins": [index / 10 for index in range(10)],
        "leaf_vertical_density_bins": [1.0 - index / 10 for index in range(10)],
        "leaf_available": True,
        "table_available": False,
        "object_profile_available": True,
        "leaf_profile_available": False,
        "arrangement": "2x3",
        "arrangement_rows": 2,
        "arrangement_columns": 3,
        "object_length_in": 20.0,
        "sku_category": "Perennials",
    }

    features = model_features_from_ai2(ai2_features)

    assert features["object_vertical_density_bin_00"] == 0.0
    assert features["object_vertical_density_bin_09"] == 0.9
    assert features["leaf_vertical_density_bin_00"] == 1.0
    assert abs(features["leaf_vertical_density_bin_09"] - 0.1) < 1e-9
    assert features["leaf_available"] == 1.0
    assert features["table_available"] == 0.0
    assert features["object_profile_available"] == 1.0
    assert features["leaf_profile_available"] == 0.0
    assert features["arrangement_rows"] == 2.0
    assert features["arrangement_columns"] == 3.0
    assert features["sku_category"] == "Perennials"
    assert "object_vertical_density_bins" not in features
    assert "leaf_vertical_density_bins" not in features
    assert "object_available" not in features
    assert "geometry_trusted" not in features
    assert "arrangement" not in features
    assert list(features) == sorted(features)


def test_model_feature_vector_has_stable_columns_for_null_and_missing_values() -> None:
    populated = model_features_from_ai2(
        {
            "object_vertical_density_bins": [0.1] * 10,
            "leaf_vertical_density_bins": [0.2] * 10,
            "leaf_available": True,
            "table_available": False,
            "object_profile_available": True,
            "leaf_profile_available": False,
        }
    )
    nulls = model_features_from_ai2(
        {
            "object_vertical_density_bins": None,
            "leaf_vertical_density_bins": None,
            "leaf_available": None,
            "table_available": None,
        }
    )
    missing = model_features_from_ai2({})

    assert list(populated) == list(nulls) == list(missing)
    assert len([name for name in missing if name.startswith("object_vertical_density_bin_")]) == 10
    assert len([name for name in missing if name.startswith("leaf_vertical_density_bin_")]) == 10
    assert all(value is None for value in nulls.values())
    assert all(value is None for value in missing.values())


def test_training_feature_list_contains_stable_flattened_columns() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "jobs"
        first = _write_job(root, "first")
        second = _write_job(root, "second")
        first_payload = read_json(first / "combined_features.json")
        first_payload["ai2_features"].update(
            {
                "object_vertical_density_bins": [0.1] * 10,
                "leaf_vertical_density_bins": None,
                "leaf_available": False,
                "table_available": True,
                "object_profile_available": True,
                "leaf_profile_available": False,
            }
        )
        write_json_atomic(first / "combined_features.json", first_payload)

        rows, report = build_training_table(root)
        feature_names = report["feature_names"]
        x_rows, _, matrix_names = matrix_from_rows(rows, feature_names)

        assert feature_names == sorted(feature_names)
        assert matrix_names == feature_names
        for prefix in ("object_vertical_density_bin_", "leaf_vertical_density_bin_"):
            expected = [f"{prefix}{index:02d}" for index in range(10)]
            assert all(name in feature_names for name in expected)
        assert "object_vertical_density_bins" not in feature_names
        assert "leaf_vertical_density_bins" not in feature_names
        assert x_rows[0]["object_vertical_density_bin_00"] == 0.1
        assert x_rows[1]["object_vertical_density_bin_00"] is None
        assert x_rows[0]["table_available"] == 1.0
        assert x_rows[1]["table_available"] is None

        output_dir = Path(tmp) / "model"
        training_report = train_ai2_v1(root, output_dir)
        written_feature_names = read_json(output_dir / "feature_list.json")
        assert training_report["status"] == "trained"
        assert written_feature_names == feature_names


def test_missing_model_returns_ok_without_prediction() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        _write_job(cfg.single_jobs_dir, "valid")

        result = predict_ai2_for_job(cfg, cfg.job_type_single, "valid")

        assert result["ok"] is True
        assert result["model_available"] is False
        assert result["prediction"] is None
        assert "ai2_model_missing" in result["warnings"][0]


def test_prediction_does_not_overwrite_ground_truth() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        _write_job(cfg.single_jobs_dir, "train_a", label=(10.0, 8.0, 20.0), sku_context=True)
        _write_job(cfg.single_jobs_dir, "train_b", label=(12.0, 9.0, 22.0), sku_context=True)
        report = train_ai2_v1(cfg.single_jobs_dir, cfg.single_packing_models_dir / "ai2_v1")
        assert report["status"] == "trained"
        gt_path = cfg.single_jobs_dir / "train_a" / "ground_truth.json"
        before = gt_path.read_bytes()

        result = predict_ai2_for_job(cfg, cfg.job_type_single, "train_a")

        assert result["model_available"] is True
        assert result["prediction"]["predicted_box"]
        assert result["ground_truth"]["present"] is True
        assert result["comparison"] is not None
        assert gt_path.read_bytes() == before


def test_flask_ai2_predict_route_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        _write_job(cfg.single_jobs_dir, "train_a", label=(10.0, 8.0, 20.0), sku_context=True)
        _write_job(cfg.single_jobs_dir, "train_b", label=(12.0, 9.0, 22.0), sku_context=True)
        train_ai2_v1(cfg.single_jobs_dir, cfg.single_packing_models_dir / "ai2_v1")
        app = create_app(cfg)

        client = app.test_client()
        response = client.post(
            "/api/ai2/predict",
            json={"job_type": "single", "job_id": "train_a"},
        )
        data = response.get_json()

        assert response.status_code == 200
        assert data["ok"] is True
        assert data["model_available"] is True
        assert data["prediction"]["method"] == "ai2_v1_sklearn_baseline"
        assert data["ground_truth"]["present"] is True


def test_training_scan_ignores_prediction_data_when_pointed_at_datasets() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        datasets_jobs = root / "datasets" / "single" / "jobs"
        prediction_jobs = root / "prediction_data" / "single" / "jobs"
        _write_job(datasets_jobs, "dataset_job")
        _write_job(prediction_jobs, "prediction_job")

        rows, report = build_training_table(datasets_jobs)

        assert [row["job_id"] for row in rows] == ["dataset_job"]
        assert report["included"] == 1


def test_capture_predict_route_writes_prediction_data_without_ground_truth() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        app = create_app(cfg)

        def fake_capture(active_cfg, **kwargs):
            job_id = "single_prediction_test"
            job_dir = active_cfg.single_jobs_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(
                job_dir / "combined_features.json",
                {
                    "feature_schema_version": "v1.1",
                    "ai2_features": {
                        "object_available": True,
                        "geometry_trusted": True,
                        "object_length_in": 10.0,
                        "object_width_in": 8.0,
                        "object_height_in": 20.0,
                    },
                    "quality_flags": {},
                },
            )
            return {
                "job_id": job_id,
                "current_view_id": "view_01",
                "captured_views": ["view_01"],
                "required_views": ["view_01"],
                "quality_summary": {"decision": "proceed"},
            }

        def fake_predict(active_cfg, job_type, job_id):
            assert "prediction_data" in str(active_cfg.single_jobs_dir)
            assert job_id == "single_prediction_test"
            assert not (active_cfg.single_jobs_dir / job_id / "ground_truth.json").exists()
            return {
                "ok": True,
                "model_available": False,
                "prediction": None,
                "ground_truth": {"present": False},
                "comparison": None,
                "warnings": ["ai2_model_missing"],
                "model": {"path": "models/packing/single/ai2_v1/model.joblib"},
            }

        with (
            patch("app.routes.run_real_data_collection_scan", side_effect=fake_capture),
            patch("app.routes.predict_ai2_for_job", side_effect=fake_predict),
            patch("app.camera_manager.get_camera_manager", return_value=object()),
        ):
            response = app.test_client().post(
                "/api/ai2/capture-predict",
                json={"items": [{"sku": "fern_8in", "quantity": 1}]},
            )

        data = response.get_json()
        prediction_job_dir = Path(data["prediction_job_dir"])
        assert response.status_code == 200
        assert data["ok"] is True
        assert data["prediction_job_id"] == "single_prediction_test"
        assert "prediction_data/single/jobs/single_prediction_test" in str(prediction_job_dir)
        assert prediction_job_dir.is_dir()
        assert (prediction_job_dir / "combined_features.json").is_file()
        assert not (prediction_job_dir / "ground_truth.json").exists()


def test_capture_predict_allows_blank_sku_with_context_missing_warning() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        app = create_app(cfg)

        def fake_capture(active_cfg, **kwargs):
            items = kwargs["items"]
            assert items == [
                {
                    "sku": None,
                    "quantity": 1,
                    "metadata": {
                        "known": False,
                        "lookup_status": "not_provided",
                        "category": None,
                        "common_name": None,
                        "spec": None,
                        "pot_prior": None,
                    },
                }
            ]
            job_id = "single_prediction_blank_sku"
            job_dir = active_cfg.single_jobs_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(
                job_dir / "combined_features.json",
                {
                    "feature_schema_version": "v1.1",
                    "ai2_features": {
                        "object_available": True,
                        "geometry_trusted": True,
                        "object_length_in": 10.0,
                        "object_width_in": 8.0,
                        "object_height_in": 20.0,
                        "sku_category": None,
                        "sku_common_name": None,
                        "sku_spec": None,
                    },
                    "quality_flags": {},
                },
            )
            return {
                "job_id": job_id,
                "current_view_id": "view_01",
                "captured_views": ["view_01"],
                "required_views": ["view_01"],
                "quality_summary": {"decision": "proceed"},
            }

        def fake_predict(active_cfg, job_type, job_id):
            return {
                "ok": True,
                "model_available": False,
                "prediction": None,
                "ground_truth": {"present": False},
                "comparison": None,
                "warnings": ["ai2_model_missing"],
                "model": {"path": "models/packing/single/ai2_v1/model.joblib"},
            }

        with (
            patch("app.routes.run_real_data_collection_scan", side_effect=fake_capture),
            patch("app.routes.predict_ai2_for_job", side_effect=fake_predict),
            patch("app.camera_manager.get_camera_manager", return_value=object()),
        ):
            response = app.test_client().post(
                "/api/ai2/capture-predict",
                json={"items": [{"sku": "", "quantity": 1}]},
            )

        data = response.get_json()
        prediction_job_dir = Path(data["prediction_job_dir"])
        assert response.status_code == 200
        assert data["ok"] is True
        assert data["prediction_job_id"] == "single_prediction_blank_sku"
        assert "SKU not provided; prediction used geometry-only/context-missing mode." in data["warnings"]
        assert not (prediction_job_dir / "ground_truth.json").exists()


def _group_item(sku: str, quantity: int, category: str, spec: str, diameter: float | None) -> dict:
    return {
        "sku": sku,
        "quantity": quantity,
        "metadata": {
            "category": category,
            "common_name": f"Name {sku}",
            "spec": spec,
            "pot_prior": {
                "available": diameter is not None,
                "confidence": "high" if diameter is not None else "unknown",
                "diameter_in": diameter,
                "width_in": diameter,
                "depth_in": diameter,
                "height_in": diameter,
            },
        },
    }


def test_group_composition_is_complete_order_independent_and_quantity_weighted() -> None:
    a = _group_item("SKU_A", 2, "Perennials", "1 Gallon", 6.0)
    b = _group_item("SKU_B", 4, "Shrubs", "2 Gallon", 10.0)
    first = group_composition_features({"items": [a, b]})
    reordered = group_composition_features({"items": [b, a]})

    assert first == reordered
    assert first["homogeneous_group"] == 0.0
    assert first["group_category_qty__perennials"] == 2.0
    assert first["group_category_qty__shrubs"] == 4.0
    assert first["group_spec_qty__1_gallon"] == 2.0
    assert first["group_spec_qty__2_gallon"] == 4.0
    assert first["group_pot_prior_diameter_in_weighted_mean"] == (2 * 6.0 + 4 * 10.0) / 6
    assert "SKU_A" not in first["group_composition_summary"]
    assert "Name" not in first["group_composition_summary"]


def test_homogeneous_group_composition_uses_all_quantity() -> None:
    features = group_composition_features(
        {"items": [_group_item("SKU_A", 6, "Perennials", "1 Gallon", 6.0)]}
    )
    assert features["homogeneous_group"] == 1.0
    assert features["group_largest_type_quantity"] == 6.0
    assert features["group_largest_type_ratio"] == 1.0
    assert features["group_pot_prior_quantity"] == 6.0

    mixed_same_metadata = group_composition_features(
        {
            "items": [
                _group_item("SKU_A", 3, "Perennials", "1 Gallon", 6.0),
                _group_item("SKU_B", 3, "Perennials", "1 Gallon", 6.0),
            ]
        }
    )
    assert mixed_same_metadata["homogeneous_group"] == 0.0


def test_single_and_group_default_model_paths_are_separate() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        single = default_model_dir(cfg, "single")
        group = default_model_dir(cfg, "group")
        assert single == cfg.single_packing_models_dir / "ai2_v1"
        assert group == cfg.group_packing_models_dir / "ai2_v1"
        assert single != group


def test_prediction_data_root_is_explicitly_rejected_for_training() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "prediction_data" / "group" / "jobs"
        _write_job(root, "prediction_job")
        rows, report = build_training_table(root, job_type="group")
        assert rows == []
        assert report["skip_reasons"] == {"prediction_data_not_trainable": 1}


def test_group_training_cli_uses_group_only_defaults() -> None:
    report = {
        "status": "trained",
        "dataset": {"included": 2, "skipped": 0, "skip_reasons": {}},
        "label_summary": {},
        "model_path": "models/packing/group/ai2_v1/model.joblib",
    }
    with (
        patch.object(sys, "argv", ["train_ai2_v1.py", "--job-type", "group"]),
        patch("scripts.train_ai2_v1.train_ai2_v1", return_value=report) as trainer,
    ):
        assert train_cli.main() == 0
    trainer.assert_called_once_with(
        "datasets/group/jobs",
        "models/packing/group/ai2_v1",
        job_type="group",
    )


def test_single_rectangular_capture_waits_for_view_02_then_predicts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        app = create_app(cfg)
        calls = []

        def fake_capture(active_cfg, **kwargs):
            calls.append(kwargs["view_id"])
            job_id = kwargs["job_id"] or "single_rect_prediction"
            captured = ["view_01"] if kwargs["view_id"] == "view_01" else ["view_01", "view_02"]
            remaining = ["view_02"] if kwargs["view_id"] == "view_01" else []
            return {
                "job_id": job_id,
                "current_view_id": kwargs["view_id"],
                "captured_views": captured,
                "required_views": ["view_01", "view_02"],
                "remaining_views": remaining,
                "next_action": "Rotate approximately 90 degrees." if remaining else "Complete.",
                "quality_summary": {"decision": "proceed"},
            }

        def fake_predict(active_cfg, job_type, job_id):
            assert job_type == "single"
            assert job_id == "single_rect_prediction"
            return {"ok": True, "model_available": True, "prediction": {"predicted_box": {}}, "warnings": []}

        with (
            patch("app.routes.run_real_data_collection_scan", side_effect=fake_capture),
            patch("app.routes.predict_ai2_for_job", side_effect=fake_predict) as predictor,
            patch("app.camera_manager.get_camera_manager", return_value=object()),
        ):
            first = app.test_client().post(
                "/api/ai2/capture-predict",
                json={"job_type": "single", "shape_mode": "rectangular", "view_id": "view_01", "items": [{"quantity": 1}]},
            ).get_json()
            assert first["ready_for_prediction"] is False
            assert first["remaining_views"] == ["view_02"]
            predictor.assert_not_called()
            second = app.test_client().post(
                "/api/ai2/capture-predict",
                json={"job_type": "single", "shape_mode": "rectangular", "view_id": "view_02", "prediction_job_id": first["prediction_job_id"], "items": [{"quantity": 1}]},
            ).get_json()

        assert calls == ["view_01", "view_02"]
        assert second["ready_for_prediction"] is True
        predictor.assert_called_once()


def test_group_final_view_selects_group_model_and_missing_model_is_clean() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        app = create_app(cfg)

        def fake_capture(active_cfg, **kwargs):
            job_id = kwargs["job_id"] or "group_prediction"
            final = kwargs["view_id"] == "view_02"
            return {
                "job_id": job_id,
                "current_view_id": kwargs["view_id"],
                "captured_views": ["view_01", "view_02"] if final else ["view_01"],
                "required_views": ["view_01", "view_02"],
                "remaining_views": [] if final else ["view_02"],
                "next_action": "Complete." if final else "Rotate approximately 90 degrees.",
                "quality_summary": {"decision": "proceed"},
            }

        def missing_group_model(active_cfg, job_type, job_id):
            assert job_type == "group"
            assert default_model_dir(active_cfg, job_type) == active_cfg.group_packing_models_dir / "ai2_v1"
            return {"ok": True, "model_available": False, "prediction": None, "warnings": ["ai2_model_missing"]}

        payload = {
            "job_type": "group",
            "arrangement_type": "1x2",
            "items": [{"sku": "FERN_8IN", "quantity": 2}],
        }
        with (
            patch("app.routes.run_real_data_collection_scan", side_effect=fake_capture),
            patch("app.routes.predict_ai2_for_job", side_effect=missing_group_model) as predictor,
            patch("app.camera_manager.get_camera_manager", return_value=object()),
        ):
            first = app.test_client().post("/api/ai2/capture-predict", json={**payload, "view_id": "view_01"}).get_json()
            predictor.assert_not_called()
            second = app.test_client().post(
                "/api/ai2/capture-predict",
                json={**payload, "view_id": "view_02", "prediction_job_id": first["prediction_job_id"]},
            ).get_json()

        assert second["ready_for_prediction"] is True
        assert second["status"] == "model_unavailable"
        predictor.assert_called_once()


def _javascript_function_body(source: str, function_name: str) -> str:
    marker = f"function {function_name}("
    start = source.index(marker)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[brace + 1:index]
    raise AssertionError(f"Unclosed JavaScript function: {function_name}")


def test_prediction_page_exposes_new_prediction_and_two_view_actions() -> None:
    html = (PROJECT_ROOT / "app/templates/predict.html").read_text(encoding="utf-8")
    assert 'id="capture-view-01"' in html
    assert "Capture View 02 &amp; Predict" in html
    assert 'id="start-new-prediction" hidden' in html
    assert "Start New Prediction" in html


def test_start_new_prediction_reset_is_local_and_restores_view_01() -> None:
    source = (PROJECT_ROOT / "app/static/predict.js").read_text(encoding="utf-8")
    body = _javascript_function_body(source, "resetPredictionJob")

    assert "predictionJobId = null" in body
    assert "predictionCapturedViews = []" in body
    assert "predictionReady = false" in body
    assert "predictionCompleted = false" in body
    assert 'field("capture-view-01").hidden = false' in body
    assert 'field("capture-view-02").hidden = true' in body
    assert 'field("start-new-prediction").hidden = true' in body
    assert 'output.textContent = "{}"' in body
    assert "postJson" not in body
    assert "fetch(" not in body
    assert "delete" not in body.lower()


def test_completed_one_or_two_view_prediction_shows_reset_without_reusing_job() -> None:
    source = (PROJECT_ROOT / "app/static/predict.js").read_text(encoding="utf-8")
    capture_body = source[source.index("async function captureAndPredict"):source.index("async function predict()")]

    assert "if (requestInProgress || predictionCompleted)" in capture_body
    assert "activeButton.disabled = true" in capture_body
    assert "predictionCompleted = predictionReady" in capture_body
    assert 'field("capture-view-02").hidden = !(response.remaining_views || []).includes("view_02")' in capture_body
    assert 'field("start-new-prediction").hidden = false' in capture_body
    assert 'field("start-new-prediction").addEventListener("click", resetPredictionJob)' in source
    assert 'field("job-type").addEventListener("change", renderMode)' in source
    assert 'field("shape-mode").addEventListener("change", resetPredictionJob)' in source


def test_collection_job_setup_names_resolved_unresolved_group_and_manual_ids() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(_temp_cfg(Path(tmp)))
        client = app.test_client()

        resolved = client.post(
            "/api/job/setup",
            json={"job_type": "single", "arrangement_type": "1x1", "items": [{"sku": "PECC0010", "quantity": 1}]},
        ).get_json()
        unresolved = client.post(
            "/api/job/setup",
            json={"job_type": "single", "arrangement_type": "1x1", "items": [{"sku": "NOT_A_REAL_SKU", "quantity": 1}]},
        ).get_json()
        group = client.post(
            "/api/job/setup",
            json={"job_type": "group", "arrangement_type": "1x1", "items": [{"sku": "PECC0010", "quantity": 1}]},
        ).get_json()
        manual = client.post(
            "/api/job/setup",
            json={"job_type": "single", "job_id": "My Manual Job", "arrangement_type": "1x1"},
        ).get_json()

        assert re.fullmatch(r"PECC0010_\d{8}_\d{6}", resolved["job_id"])
        assert re.fullmatch(r"SINGLE_\d{8}_\d{6}", unresolved["job_id"])
        assert re.fullmatch(r"GROUP_\d{8}_\d{6}", group["job_id"])
        assert resolved["auto_generated_job_id"] is True
        assert unresolved["auto_generated_job_id"] is True
        assert group["auto_generated_job_id"] is True
        assert manual["job_id"] == "my_manual_job"
        assert manual["auto_generated_job_id"] is False


def test_collection_job_id_ui_preserves_manual_override_and_refreshes_on_start() -> None:
    source = (PROJECT_ROOT / "app/static/app.js").read_text(encoding="utf-8")
    refresh_body = source[source.index("function refreshAutoJobId"):source.index("function jobSetupItems")]
    prepare_body = source[source.index("async function prepareJob"):source.index("async function lookupSku")]
    lookup_body = source[source.index("async function lookupSku"):source.index("function addItem")]

    assert 'return "GROUP"' in source
    assert 'return "SINGLE"' in source
    assert 'replace(/[^A-Z0-9_-]+/g, "")' in source
    assert "jobIdField.value.trim() && !jobIdIsAutoGenerated" in refresh_body
    assert "refreshAutoJobId({force: true})" in prepare_body
    assert "job_id: jobIdIsAutoGenerated ? null" in prepare_body
    assert "items: jobSetupItems()" in prepare_body
    assert "refreshAutoJobId()" in lookup_body
    assert 'field("job-id").addEventListener("input"' in source
    assert "jobIdIsAutoGenerated = false" in source
    assert 'job_id: activeJob.job_id' in source


def test_collection_setup_is_folder_free_and_first_item_creates_final_job() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        client = create_app(cfg).test_client()

        setup = client.post(
            "/api/job/setup",
            json={"job_type": "single", "arrangement_type": "1x1"},
        ).get_json()
        assert not cfg.single_jobs_dir.exists() or list(cfg.single_jobs_dir.iterdir()) == []

        created = client.post(
            "/api/job/commit-items",
            json={
                "job_type": "single",
                "job_id": None,
                "arrangement_type": "1x1",
                "operator_id": "operator-a",
                "items": [{"sku": "pe1g1587", "quantity": 1}],
            },
        ).get_json()
        job_id = created["job_id"]
        job_dir = cfg.single_jobs_dir / job_id
        assert setup["auto_generated_job_id"] is True
        assert created["created"] is True
        assert re.fullmatch(r"PE1G1587_\d{8}_\d{6}", job_id)
        assert job_id.startswith("PE1G1587_")
        assert job_dir.name == job_id
        assert read_json(job_dir / "job_metadata.json")["operator_id"] == "operator-a"
        assert read_json(job_dir / "item_list.json")["total_quantity"] == 1

        updated = client.post(
            "/api/job/commit-items",
            json={
                "job_type": "single",
                "job_id": job_id,
                "arrangement_type": "1x1",
                "items": [{"sku": "pe1g1587", "quantity": 2}],
            },
        ).get_json()
        assert updated["created"] is False
        assert updated["job_id"] == job_id
        assert [path.name for path in cfg.single_jobs_dir.iterdir()] == [job_id]
        assert read_json(job_dir / "item_list.json")["total_quantity"] == 2


def test_invalid_sku_requires_explicit_unresolved_commit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        client = create_app(cfg).test_client()

        rejected = client.post(
            "/api/job/commit-items",
            json={
                "job_type": "single",
                "arrangement_type": "1x1",
                "items": [{"sku": "NOT_A_REAL_SKU", "quantity": 1}],
            },
        )
        assert rejected.status_code == 400
        assert "SKU not found" in rejected.get_json()["error"]
        assert not cfg.single_jobs_dir.exists()

        intentional = client.post(
            "/api/job/commit-items",
            json={
                "job_type": "single",
                "arrangement_type": "1x1",
                "items": [{"sku": "NOT_A_REAL_SKU", "quantity": 1, "allow_unresolved": True}],
            },
        )
        payload = intentional.get_json()
        assert intentional.status_code == 200
        assert re.fullmatch(r"SINGLE_\d{8}_\d{6}", payload["job_id"])
        job_dir = cfg.single_jobs_dir / payload["job_id"]
        assert [path.name for path in cfg.single_jobs_dir.iterdir()] == [payload["job_id"]]
        assert not any(path.is_dir() for path in job_dir.iterdir())


def test_final_setup_item_removal_deletes_exact_job_and_next_valid_sku_is_fresh() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        client = create_app(cfg).test_client()
        created = client.post(
            "/api/job/commit-items",
            json={
                "job_type": "single",
                "arrangement_type": "1x1",
                "items": [{"sku": "NOT_A_REAL_SKU", "quantity": 1, "allow_unresolved": True}],
            },
        ).get_json()
        old_job_id = created["job_id"]
        old_job_dir = cfg.single_jobs_dir / old_job_id

        removed = client.post(
            "/api/job/remove-item",
            json={"job_type": "single", "job_id": old_job_id, "items": []},
        )
        assert removed.status_code == 200
        assert removed.get_json()["deleted"] is True
        assert not old_job_dir.exists()

        fresh = client.post(
            "/api/job/commit-items",
            json={
                "job_type": "single",
                "arrangement_type": "1x1",
                "items": [{"sku": "pe1g1587", "quantity": 1}],
            },
        ).get_json()
        assert fresh["job_id"].startswith("PE1G1587_")
        assert (cfg.single_jobs_dir / fresh["job_id"]).is_dir()
        assert [path.name for path in cfg.single_jobs_dir.iterdir()] == [fresh["job_id"]]


def test_final_item_removal_cannot_delete_captured_job() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        client = create_app(cfg).test_client()
        created = client.post(
            "/api/job/commit-items",
            json={
                "job_type": "single",
                "arrangement_type": "1x1",
                "items": [{"sku": "pe1g1587", "quantity": 1}],
            },
        ).get_json()
        job_dir = cfg.single_jobs_dir / created["job_id"]
        capture_path = job_dir / "view_01" / "capture_meta.json"
        write_json_atomic(capture_path, {"captured": True})

        blocked = client.post(
            "/api/job/remove-item",
            json={"job_type": "single", "job_id": created["job_id"], "items": []},
        )
        assert blocked.status_code == 400
        assert "capture or processing data" in blocked.get_json()["error"]
        assert job_dir.is_dir()
        assert capture_path.is_file()


def test_collection_ui_removal_resets_committed_identity_to_draft() -> None:
    source = (PROJECT_ROOT / "app/static/app.js").read_text(encoding="utf-8")
    remove_body = source[source.index("async function removeItem"):source.index("function renderItems")]
    add_body = source[source.index("async function addItem"):source.index("async function captureCollect")]
    assert 'postJson("/api/job/remove-item"' in remove_body
    assert 'activeJob.job_id = ""' in remove_body
    assert 'field("job-id").value = ""' in remove_body
    assert "backendJobCreated = false" in remove_body
    assert "items = nextItems" in remove_body
    assert 'throw {error: "SKU not found"}' in add_body
    assert 'field("allow-unresolved-sku").checked' in add_body
    assert "activeJob.job_id = response.job_id" in add_body
    assert 'field("job-id").value = response.job_id' in add_body


def test_first_unresolved_group_and_manual_item_commits_use_final_names() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        client = create_app(cfg).test_client()

        unresolved = client.post(
            "/api/job/commit-items",
            json={
                "job_type": "single",
                "arrangement_type": "1x1",
                    "items": [{"sku": "NOT_A_REAL_SKU", "quantity": 1, "allow_unresolved": True}],
            },
        ).get_json()
        group = client.post(
            "/api/job/commit-items",
            json={
                "job_type": "group",
                "arrangement_type": "1x2",
                "items": [{"sku": "PECC0010", "quantity": 1}],
            },
        ).get_json()
        manual = client.post(
            "/api/job/commit-items",
            json={
                "job_type": "single",
                "job_id": "my_manual_job",
                "arrangement_type": "1x1",
                "items": [{"sku": "PECC0010", "quantity": 1}],
            },
        ).get_json()

        assert re.fullmatch(r"SINGLE_\d{8}_\d{6}", unresolved["job_id"])
        assert re.fullmatch(r"GROUP_\d{8}_\d{6}", group["job_id"])
        assert manual["job_id"] == "my_manual_job"
        assert (cfg.single_jobs_dir / unresolved["job_id"]).is_dir()
        assert (cfg.group_jobs_dir / group["job_id"]).is_dir()
        assert (cfg.single_jobs_dir / "my_manual_job").is_dir()


def test_final_collection_job_id_is_used_by_capture_and_ground_truth() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _temp_cfg(Path(tmp))
        app = create_app(cfg)
        client = app.test_client()
        created = client.post(
            "/api/job/commit-items",
            json={
                "job_type": "single",
                "arrangement_type": "1x1",
                "items": [{"sku": "PECC0010", "quantity": 1}],
            },
        ).get_json()
        job_id = created["job_id"]

        def fake_capture(active_cfg, **kwargs):
            assert kwargs["job_id"] == job_id
            return {
                "job_id": job_id,
                "current_view_id": "view_01",
                "captured_views": ["view_01"],
                "required_views": ["view_01"],
                "required_view_count": 1,
                "remaining_views": [],
                "next_action": "All captures completed.",
                "quality_summary": {"decision": "proceed"},
                "segmentation_by_view": {"view_01": {"status": "partial", "warnings": []}},
            }

        with (
            patch("app.routes.run_real_data_collection_scan", side_effect=fake_capture),
            patch("app.camera_manager.get_camera_manager", return_value=object()),
        ):
            capture = client.post(
                "/api/collect",
                json={
                    "mode": "data_collection",
                    "job_type": "single",
                    "job_id": job_id,
                    "arrangement_type": "1x1",
                    "items": [{"sku": "PECC0010", "quantity": 1}],
                },
            )
        assert capture.status_code == 200
        gt = client.post(
            "/api/ground-truth",
            json={
                "job_type": "single",
                "job_id": job_id,
                "length_in": 10,
                "width_in": 8,
                "height_in": 20,
                "fit": "good",
            },
        )
        assert gt.status_code == 200
        assert (cfg.single_jobs_dir / job_id / "ground_truth.json").is_file()


def test_collection_ui_commits_items_once_before_capture() -> None:
    source = (PROJECT_ROOT / "app/static/app.js").read_text(encoding="utf-8")
    add_body = source[source.index("async function addItem"):source.index("async function captureCollect")]
    capture_body = source[source.index("async function captureCollect"):source.index("async function recaptureSelectedView")]
    assert "if (itemCommitInProgress)" in add_body
    assert 'postJson("/api/job/commit-items"' in add_body
    assert "backendJobCreated || !activeJob.auto_generated_job_id" in add_body
    assert "backendJobCreated = true" in add_body
    assert "if (!backendJobCreated)" in capture_body
