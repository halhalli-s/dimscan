"""Train the minimal AI2 v1 sklearn baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ai2.dataset import LABEL_FIELDS, SAFE_SKU_FIELDS, build_training_table, matrix_from_rows
from utils.io import write_json_atomic


MODEL_VERSION = "ai2_v1_sklearn_baseline"
MIN_TRAINING_SAMPLES = 2


def _label_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for field in LABEL_FIELDS:
        values = [float(row["label"][field]) for row in rows]
        if values:
            summary[field] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
            }
    return summary


def train_ai2_v1(
    jobs_root: str | Path,
    out_dir: str | Path,
    *,
    job_type: str | None = None,
) -> dict[str, Any]:
    """Train and save the AI2 v1 baseline model artifacts."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, dataset_report = build_training_table(jobs_root, job_type=job_type)
    feature_names = list(dataset_report["feature_names"])
    report: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "status": "not_trained",
        "dataset": dataset_report,
        "label_fields": list(LABEL_FIELDS),
        "label_summary": _label_summary(rows),
        "warnings": [],
        "job_type": job_type,
    }

    if len(rows) < MIN_TRAINING_SAMPLES:
        report["warnings"].append(
            f"too_few_samples: need at least {MIN_TRAINING_SAMPLES}, found {len(rows)}"
        )
        write_json_atomic(output_dir / "report.json", report)
        write_json_atomic(output_dir / "feature_list.json", feature_names)
        write_json_atomic(output_dir / "labels.json", list(LABEL_FIELDS))
        return report

    try:
        import joblib
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
    except ImportError as exc:
        report["warnings"].append(f"sklearn_stack_unavailable: {exc}")
        write_json_atomic(output_dir / "report.json", report)
        write_json_atomic(output_dir / "feature_list.json", feature_names)
        write_json_atomic(output_dir / "labels.json", list(LABEL_FIELDS))
        return report

    x_rows, y_rows, feature_names = matrix_from_rows(rows, feature_names)
    categorical_features = [name for name in SAFE_SKU_FIELDS if name in feature_names]
    numeric_features = [name for name in feature_names if name not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_features),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=64,
                    random_state=42,
                    min_samples_leaf=1,
                ),
            ),
        ]
    )
    model.fit(pd.DataFrame(x_rows, columns=feature_names), y_rows)

    joblib.dump(
        {
            "model_version": MODEL_VERSION,
            "model": model,
            "feature_names": feature_names,
            "label_fields": list(LABEL_FIELDS),
        },
        output_dir / "model.joblib",
    )
    report.update(
        {
            "status": "trained",
            "sample_count": len(rows),
            "feature_count": len(feature_names),
            "model_path": str(output_dir / "model.joblib"),
            "quality": "unvalidated_small_data_baseline",
        }
    )
    write_json_atomic(output_dir / "report.json", report)
    write_json_atomic(output_dir / "feature_list.json", feature_names)
    write_json_atomic(output_dir / "labels.json", list(LABEL_FIELDS))
    return report
