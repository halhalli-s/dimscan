"""Production-facing route registration for the DimScan web flow."""

from __future__ import annotations

from dataclasses import replace
import traceback
from typing import Any

from app.config import DimScanConfig
from pipeline.job_setup import prepare_job_setup
from metadata.sku_lookup import lookup_sku, make_item_from_sku
from metadata.sku_catalogue import lookup_catalogue_sku
from ai2.predict import predict_ai2_for_job, write_ai2_prediction
from pipeline.ground_truth import record_ground_truth
from pipeline.inspect_job import inspect_job
from pipeline.prediction import predict_box_for_job
from pipeline.profiling import begin_profile, end_profile, timed_stage
from pipeline.real_capture import run_real_data_collection_scan
from pipeline.run_scan import run_fake_data_collection_scan
from pipeline.scan_writer import delete_setup_only_collection_job, initialize_arranged_job, update_collection_job_items
from pipeline.workflow import MODE_DATA_COLLECTION, validate_job_type, validate_mode
from utils.paths import get_job_dir


def _quantity(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("quantity must be a positive integer")
    quantity = int(value)
    if quantity <= 0:
        raise ValueError("quantity must be a positive integer")
    return quantity


def _items_from_payload(raw_items: Any, *, allow_unresolved: bool = False) -> list[dict[str, Any]]:
    """Create deduplicated job items from a client payload."""
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("items are required; add at least one SKU before capture")

    items: list[dict[str, Any]] = []
    item_by_sku: dict[str, dict[str, Any]] = {}
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise ValueError("items must be objects")
        item = make_item_from_sku(
            raw_item["sku"],
            quantity=_quantity(raw_item.get("quantity", 1)),
        )
        unresolved_approved = allow_unresolved or raw_item.get("allow_unresolved") is True
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        if metadata.get("known") is not True and not unresolved_approved:
            raise ValueError(f"SKU not found: {item['sku']}")
        if metadata.get("known") is not True and unresolved_approved:
            item["allow_unresolved"] = True
        sku_key = item["sku"].strip().upper()
        existing = item_by_sku.get(sku_key)
        if existing is None:
            items.append(item)
            item_by_sku[sku_key] = item
        else:
            existing["quantity"] += item["quantity"]
    return items


def _prediction_items_from_payload(raw_items: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Create prediction items, allowing SKU context to be absent."""
    warnings: list[str] = []
    if not isinstance(raw_items, list):
        raw_items = []

    sku_items = [
        item for item in raw_items
        if isinstance(item, dict) and str(item.get("sku", "")).strip()
    ]
    if sku_items:
        return _items_from_payload(sku_items, allow_unresolved=True), warnings

    quantity = 1
    for raw_item in raw_items:
        if isinstance(raw_item, dict) and raw_item.get("quantity") not in (None, ""):
            quantity = _quantity(raw_item.get("quantity", 1))
            break
    warnings.append("SKU not provided; prediction used geometry-only/context-missing mode.")
    return [
        {
            "sku": None,
            "quantity": quantity,
            "metadata": {
                "known": False,
                "lookup_status": "not_provided",
                "category": None,
                "common_name": None,
                "spec": None,
                "pot_prior": None,
            },
        }
    ], warnings


def _prediction_cfg(cfg: DimScanConfig) -> DimScanConfig:
    prediction_root = cfg.dataset_root.parent / "prediction_data"
    return replace(
        cfg,
        dataset_root=prediction_root,
        single_jobs_dir=prediction_root / "single/jobs",
        single_exports_dir=prediction_root / "single/exports",
        group_jobs_dir=prediction_root / "group/jobs",
        group_exports_dir=prediction_root / "group/exports",
        mode_data_collection="prediction",
    )


def register_routes(app: Any, cfg: DimScanConfig) -> Any:
    """Register minimal Flask routes when a Flask-like app is provided."""
    if not hasattr(app, "route"):
        return app

    try:
        from flask import Response, jsonify, render_template, request
    except ImportError:
        return app

    def json_error(
        message: str,
        status_code: int = 400,
        *,
        stage: str = "unknown",
        details: Any = None,
    ) -> Any:
        payload: dict[str, Any] = {"ok": False, "error": message, "stage": stage}
        if details is not None:
            payload["details"] = details
        return jsonify(payload), status_code

    def error_stage(exc: BaseException) -> str:
        text = f"{type(exc).__name__}: {exc}".lower()
        if any(token in text for token in ("camera", "orbbec", "capture", "artifact", "view folder")):
            return "capture"
        if any(token in text for token in ("segment", "yolo", "clip", "mask")):
            return "segmentation"
        if any(token in text for token in ("geometry", "measure", "cloud", "open3d", "plane")):
            return "geometry"
        if any(token in text for token in ("feature", "combined_features")):
            return "features"
        return "unknown"

    def required_text(payload: dict[str, Any], key: str, label: str, missing: list[str]) -> str:
        value = str(payload.get(key, "")).strip()
        if not value:
            missing.append(label)
        return value

    def positive_float(payload: dict[str, Any], key: str, label: str, missing: list[str], invalid: list[str]) -> float:
        raw_value = payload.get(key)
        if raw_value in (None, ""):
            missing.append(label)
            return 0.0
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            invalid.append(label)
            return 0.0
        if value <= 0:
            invalid.append(label)
        return value

    def segmentation_message(segmentation: dict[str, Any]) -> str | None:
        status = str(segmentation.get("status") or "").lower()
        segments = segmentation.get("segments")
        if not isinstance(segments, dict):
            return None
        object_found = segments.get("object") in {"ok", "fallback"}
        missing = [name for name in ("pot", "leaf", "table") if segments.get(name) == "missing"]
        if status == "partial" and object_found and missing:
            return f"Segmentation partial: object found; {'/'.join(missing)} missing"
        if status == "partial":
            return "Segmentation partial"
        if status == "ok":
            return "Segmentation ok"
        return None

    @app.route("/", methods=["GET"])
    @app.route("/collect", methods=["GET"])
    def index() -> Any:
        return render_template("index.html")

    @app.route("/predict", methods=["GET"])
    def predict_page() -> Any:
        return render_template("predict.html")

    @app.route("/health", methods=["GET"])
    def health() -> Any:
        return jsonify({"status": "ok", "service": "dimscan"})

    @app.route("/api/camera/snapshot", methods=["GET"])
    def camera_snapshot() -> Any:
        try:
            from app.camera_stream import capture_snapshot_bytes

            image_bytes, mime_type = capture_snapshot_bytes()
            return Response(image_bytes, mimetype=mime_type)
        except Exception as exc:
            return json_error(f"Camera snapshot unavailable: {exc}", status_code=500)

    @app.route("/video_feed", methods=["GET"])
    def video_feed() -> Any:
        from app.camera_stream import generate_mjpeg_frames

        return Response(
            generate_mjpeg_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/sku/lookup", methods=["POST"])
    def sku_lookup() -> Any:
        payload = request.get_json(silent=True) or {}
        try:
            catalogue_result = lookup_catalogue_sku(payload["sku"])
            metadata = lookup_sku(payload["sku"])
            found = catalogue_result["status"] == "found"
            message = "SKU found" if found else "SKU not found"
            return jsonify(
                {
                    "ok": True,
                    "status": catalogue_result["status"],
                    "sku": metadata["sku"],
                    "known": found,
                    "item": catalogue_result.get("item"),
                    "pot_prior": metadata.get("pot_prior") or catalogue_result.get("pot_prior"),
                    "metadata": metadata,
                    "warning": catalogue_result.get("warning"),
                    "message": message,
                }
            )
        except KeyError:
            return json_error("sku is required")
        except ValueError as exc:
            return json_error(str(exc))

    @app.route("/api/job/setup", methods=["POST"])
    def job_setup() -> Any:
        payload = request.get_json(silent=True) or {}
        try:
            job_type = validate_job_type(payload.get("job_type", cfg.job_type_single))
            raw_items = payload.get("items")
            setup_items = (
                _items_from_payload(raw_items, allow_unresolved=True)
                if isinstance(raw_items, list) and raw_items
                else None
            )
            setup = prepare_job_setup(
                cfg,
                job_type=job_type,
                job_id=payload.get("job_id"),
                arrangement_type=str(payload.get("arrangement_type", "1x1")).strip() or "1x1",
                shape_mode=payload.get("shape_mode"),
                items=setup_items,
            )
            return jsonify({"ok": True, **setup})
        except (KeyError, TypeError, ValueError) as exc:
            return json_error(str(exc), stage="setup")

    @app.route("/api/job/commit-items", methods=["POST"])
    def commit_job_items() -> Any:
        payload = request.get_json(silent=True) or {}
        try:
            job_type = validate_job_type(payload.get("job_type", cfg.job_type_single))
            arrangement_type = str(payload.get("arrangement_type", "1x1")).strip() or "1x1"
            items = _items_from_payload(payload.get("items"))
            requested_job_id = str(payload.get("job_id") or "").strip() or None
            existing_dir = get_job_dir(cfg, job_type, requested_job_id) if requested_job_id else None
            if existing_dir is not None and existing_dir.is_dir():
                result = update_collection_job_items(
                    cfg,
                    job_type=job_type,
                    job_id=requested_job_id,
                    items=items,
                )
                created = False
            else:
                result = initialize_arranged_job(
                    cfg,
                    job_id=requested_job_id,
                    mode=MODE_DATA_COLLECTION,
                    job_type=job_type,
                    arrangement_type=arrangement_type,
                    items=items,
                    operator_id=payload.get("operator_id") or None,
                    strict_quantity=False,
                    shape_mode=payload.get("shape_mode"),
                )
                created = True
            return jsonify(
                {
                    "ok": True,
                    "created": created,
                    "job_id": result["job_id"],
                    "job_dir": str(result["job_dir"]),
                    "job_type": job_type,
                    "arrangement_type": result["job_metadata"].get("arrangement_type"),
                    "view_mode": result.get("view_mode"),
                    "required_views": result.get("view_names") or ["view_01"],
                    "items": result["item_list"],
                }
            )
        except FileExistsError as exc:
            return json_error(str(exc), status_code=409, stage="setup")
        except (KeyError, TypeError, ValueError) as exc:
            return json_error(str(exc), stage="setup")

    @app.route("/api/job/remove-item", methods=["POST"])
    def remove_job_item() -> Any:
        payload = request.get_json(silent=True) or {}
        try:
            job_type = validate_job_type(payload.get("job_type", cfg.job_type_single))
            job_id = str(payload.get("job_id") or "").strip()
            if not job_id:
                raise ValueError("job_id is required")
            raw_items = payload.get("items")
            if not isinstance(raw_items, list):
                raise ValueError("items must be a list")
            if raw_items:
                items = _items_from_payload(raw_items)
                result = update_collection_job_items(
                    cfg,
                    job_type=job_type,
                    job_id=job_id,
                    items=items,
                )
                return jsonify({"ok": True, "deleted": False, "job_id": job_id, "items": result["item_list"]})

            delete_setup_only_collection_job(cfg, job_type=job_type, job_id=job_id)
            return jsonify({"ok": True, "deleted": True, "job_id": job_id})
        except FileNotFoundError as exc:
            return json_error(str(exc), status_code=404, stage="setup")
        except (KeyError, TypeError, ValueError) as exc:
            return json_error(str(exc), stage="setup")

    @app.route("/api/collect", methods=["POST"])
    def collect() -> Any:
        payload = request.get_json(silent=True) or {}
        profile_token = begin_profile(
            endpoint="/api/collect",
            job_id=str(payload.get("job_id") or "") or None,
            debug_mode=bool(payload.get("debug_mode", False)),
        )
        try:
            with timed_stage("collect_request_validation_s"):
                mode = validate_mode(payload.get("mode", MODE_DATA_COLLECTION))
            if mode != MODE_DATA_COLLECTION:
                response = json_error("collection endpoint only supports data_collection mode")
                end_profile(profile_token, ok=False, status_code=400, error="collection endpoint only supports data_collection mode")
                return response

            with timed_stage("collect_job_payload_validation_s"):
                job_type = validate_job_type(payload.get("job_type", cfg.job_type_single))
                if not str(payload.get("arrangement_type", "")).strip():
                    response = json_error("arrangement_type is required", stage="capture")
                    end_profile(profile_token, ok=False, status_code=400, error="arrangement_type is required")
                    return response
                raw_items = payload.get("items")
                if not isinstance(raw_items, list) or not raw_items:
                    response = json_error("items are required; add at least one SKU before capture", stage="capture")
                    end_profile(profile_token, ok=False, status_code=400, error="items are required")
                    return response
            with timed_stage("collect_item_lookup_s"):
                items = _items_from_payload(raw_items)
            with timed_stage("collect_camera_manager_get_s"):
                from app.camera_manager import get_camera_manager

                camera = get_camera_manager()

            result = run_real_data_collection_scan(
                cfg,
                job_id=payload.get("job_id"),
                job_type=job_type,
                arrangement_type=payload["arrangement_type"],
                items=items,
                operator_id=payload.get("operator_id") or None,
                prompt_for_views=False,
                view_id=payload.get("view_id") or None,
                view_index=payload.get("view_index"),
                overwrite=bool(payload.get("overwrite", False)),
                debug_mode=bool(payload.get("debug_mode", False)),
                save_object_cloud=bool(payload.get("save_object_cloud", False)),
                camera=camera,
            )
            with timed_stage("response_assembly_serialization_s"):
                segmentation_by_view = result.get("segmentation_by_view", {})
                current_segmentation = {}
                if isinstance(segmentation_by_view, dict):
                    current_segmentation = segmentation_by_view.get(result["current_view_id"], {}) or {}
                current_segmentation_message = segmentation_message(current_segmentation)
                response_payload = {
                    "ok": True,
                    "job_id": result["job_id"],
                    "current_view_id": result["current_view_id"],
                    "quality_summary": result.get("quality_summary"),
                    "segmentation_status": current_segmentation.get("status"),
                    "segmentation_message": current_segmentation_message,
                    "segmentation_warnings": current_segmentation.get("warnings", []),
                    "captured_views": result["captured_views"],
                    "required_views": result["required_views"],
                    "required_view_count": result["required_view_count"],
                    "remaining_views": result["remaining_views"],
                    "next_action": result["next_action"],
                    "result": result,
                }
                response = jsonify(response_payload)
            end_profile(profile_token, ok=True, status_code=getattr(response, "status_code", None))
            return response
        except FileNotFoundError as exc:
            response = json_error(str(exc), status_code=404, stage=error_stage(exc))
            end_profile(profile_token, ok=False, status_code=404, error=str(exc))
            return response
        except RuntimeError as exc:
            response = json_error(
                str(exc),
                status_code=500,
                stage=error_stage(exc),
                details=traceback.format_exc(),
            )
            end_profile(profile_token, ok=False, status_code=500, error=str(exc))
            return response
        except (KeyError, TypeError, ValueError) as exc:
            response = json_error(str(exc), stage=error_stage(exc))
            end_profile(profile_token, ok=False, status_code=400, error=str(exc))
            return response
        except Exception as exc:
            response = json_error(
                str(exc),
                status_code=500,
                stage=error_stage(exc),
                details=traceback.format_exc(),
            )
            end_profile(profile_token, ok=False, status_code=500, error=str(exc))
            return response

    @app.route("/api/predict", methods=["POST"])
    def predict() -> Any:
        payload = request.get_json(silent=True) or {}
        try:
            prediction = predict_box_for_job(
                cfg,
                validate_job_type(payload["job_type"]),
                payload["job_id"],
            )
            return jsonify({"ok": True, "prediction": prediction})
        except FileNotFoundError as exc:
            return json_error(str(exc), status_code=404)
        except (KeyError, ValueError) as exc:
            return json_error(str(exc))

    @app.route("/api/ai2/predict", methods=["POST"])
    def ai2_predict() -> Any:
        payload = request.get_json(silent=True) or {}
        try:
            result = predict_ai2_for_job(
                cfg,
                validate_job_type(payload["job_type"]),
                payload["job_id"],
            )
            write_ai2_prediction(cfg, validate_job_type(payload["job_type"]), payload["job_id"], result)
            return jsonify(result)
        except FileNotFoundError as exc:
            return json_error(str(exc), status_code=404)
        except (KeyError, ValueError) as exc:
            return json_error(str(exc))

    @app.route("/api/ai2/capture-predict", methods=["POST"])
    def ai2_capture_predict() -> Any:
        payload = request.get_json(silent=True) or {}
        prediction_cfg = _prediction_cfg(cfg)
        try:
            job_type = validate_job_type(payload.get("job_type", cfg.job_type_single))
            shape_mode = str(payload.get("shape_mode") or "square_cylindrical").strip().lower()
            if job_type == cfg.job_type_single and shape_mode not in {"square_cylindrical", "rectangular"}:
                return json_error("shape_mode must be square_cylindrical or rectangular", stage="setup")
            arrangement_type = str(payload.get("arrangement_type", "1x1")).strip() or "1x1"
            if job_type == cfg.job_type_single:
                arrangement_type = "1x1"
            if job_type == cfg.job_type_group:
                items = _items_from_payload(payload.get("items"), allow_unresolved=True)
                context_warnings = []
            else:
                items, context_warnings = _prediction_items_from_payload(payload.get("items"))
            if job_type == cfg.job_type_single:
                if len(items) != 1 or items[0].get("quantity") != 1:
                    return json_error("single prediction requires exactly one item with quantity 1", stage="setup")
            with timed_stage("ai2_capture_predict_camera_manager_get_s"):
                from app.camera_manager import get_camera_manager

                camera = get_camera_manager()
            capture_result = run_real_data_collection_scan(
                prediction_cfg,
                job_id=payload.get("prediction_job_id") or None,
                job_type=job_type,
                arrangement_type=arrangement_type,
                items=items,
                operator_id=payload.get("operator_id") or None,
                prompt_for_views=False,
                view_id=payload.get("view_id") or None,
                overwrite=False,
                debug_mode=bool(payload.get("debug_mode", False)),
                save_object_cloud=bool(payload.get("save_object_cloud", False)),
                camera=camera,
                strict_quantity=True,
                shape_mode=shape_mode if job_type == cfg.job_type_single else None,
            )
            prediction_job_id = capture_result["job_id"]
            prediction_job_dir = get_job_dir(prediction_cfg, job_type, prediction_job_id)
            remaining_views = capture_result.get("remaining_views") or []
            if remaining_views:
                return jsonify(
                    {
                        "ok": True,
                        "status": "capture_in_progress",
                        "model_available": None,
                        "prediction": None,
                        "prediction_job_id": prediction_job_id,
                        "prediction_job_type": job_type,
                        "prediction_job_dir": str(prediction_job_dir),
                        "captured_views": capture_result.get("captured_views") or [],
                        "required_views": capture_result.get("required_views") or [],
                        "remaining_views": remaining_views,
                        "ready_for_prediction": False,
                        "operator_instruction": capture_result.get("next_action"),
                        "quality_summary": capture_result.get("quality_summary"),
                        "warnings": context_warnings,
                    }
                )
            prediction = predict_ai2_for_job(
                prediction_cfg,
                job_type,
                prediction_job_id,
            )
            write_ai2_prediction(prediction_cfg, job_type, prediction_job_id, prediction)
            warnings = [*context_warnings, *(prediction.get("warnings") or [])]
            if prediction.get("prediction") and isinstance(prediction["prediction"], dict):
                prediction["prediction"]["warnings"] = [
                    *context_warnings,
                    *(prediction["prediction"].get("warnings") or []),
                ]
            return jsonify(
                {
                    **prediction,
                    "warnings": warnings,
                    "prediction_job_id": prediction_job_id,
                    "prediction_job_type": job_type,
                    "prediction_job_dir": str(prediction_job_dir),
                    "captured_views": capture_result.get("captured_views") or [],
                    "required_views": capture_result.get("required_views") or [],
                    "remaining_views": [],
                    "ready_for_prediction": True,
                    "status": "predicted" if prediction.get("model_available") else "model_unavailable",
                    "operator_instruction": capture_result.get("next_action"),
                    "capture": {
                        "current_view_id": capture_result.get("current_view_id"),
                        "captured_views": capture_result.get("captured_views"),
                        "required_views": capture_result.get("required_views"),
                        "quality_summary": capture_result.get("quality_summary"),
                    },
                }
            )
        except RuntimeError as exc:
            return json_error(
                str(exc),
                status_code=500,
                stage=error_stage(exc),
                details=traceback.format_exc(),
            )
        except FileNotFoundError as exc:
            return json_error(str(exc), status_code=404)
        except (KeyError, TypeError, ValueError) as exc:
            return json_error(str(exc), stage=error_stage(exc))

    @app.route("/api/ground-truth", methods=["POST"])
    def ground_truth() -> Any:
        payload = request.get_json(silent=True) or {}
        try:
            missing: list[str] = []
            invalid: list[str] = []
            job_type = required_text(payload, "job_type", "job_type", missing)
            job_id = required_text(payload, "job_id", "job_id", missing)
            fit = required_text(payload, "fit", "fit", missing)
            length_in = positive_float(payload, "length_in", "length", missing, invalid)
            width_in = positive_float(payload, "width_in", "width", missing, invalid)
            height_in = positive_float(payload, "height_in", "height", missing, invalid)
            if missing:
                return json_error(
                    f"Ground truth missing: {', '.join(missing)}",
                    details={"missing": missing},
                )
            if invalid:
                return json_error(
                    f"Ground truth invalid: {', '.join(invalid)} must be positive numbers",
                    details={"invalid": invalid},
                )
            active_job_type = validate_job_type(job_type)
            notes = str(payload.get("notes", "")).strip() or None
            ground_truth_record = record_ground_truth(
                cfg,
                active_job_type,
                job_id,
                length_in=length_in,
                width_in=width_in,
                height_in=height_in,
                source=payload.get("source", "manual"),
                fit=fit,
                damage=bool(payload.get("damage", False)),
                notes=notes,
            )
            return jsonify({"ok": True, "ground_truth": ground_truth_record})
        except FileNotFoundError as exc:
            return json_error(str(exc), status_code=404)
        except (KeyError, TypeError, ValueError) as exc:
            return json_error(str(exc))

    @app.route("/api/job/<job_type>/<job_id>", methods=["GET"])
    def job_report(job_type: str, job_id: str) -> Any:
        try:
            return jsonify(inspect_job(cfg, validate_job_type(job_type), job_id))
        except ValueError as exc:
            return json_error(str(exc))

    @app.route("/api/fake-scan", methods=["POST"])
    def fake_scan() -> Any:
        payload = request.get_json(silent=True) or {}
        try:
            result = run_fake_data_collection_scan(
                cfg,
                job_id=payload.get("job_id"),
                job_type=validate_job_type(payload.get("job_type", cfg.job_type_single)),
                arrangement_type=payload["arrangement_type"],
                items=_items_from_payload(payload["items"]),
                operator_id=payload.get("operator_id"),
            )
            return jsonify({"ok": True, "job_id": result["job_id"], "result": result})
        except (KeyError, TypeError, ValueError) as exc:
            return json_error(str(exc))

    return app
