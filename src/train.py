"""Train and persist HealthRisk-AI machine-learning models."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    DEFAULT_N_ROWS,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    build_preprocessor,
    default_data_path,
    ensure_dataset,
    split_features_targets,
)


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
CLASSIFIER_FILENAME = "patient_risk_classifier.joblib"
REGRESSOR_FILENAME = "insurance_cost_regressor.joblib"
METRICS_FILENAME = "training_metrics.json"


def classifier_path(model_dir: str | Path = MODEL_DIR) -> Path:
    return Path(model_dir) / CLASSIFIER_FILENAME


def regressor_path(model_dir: str | Path = MODEL_DIR) -> Path:
    return Path(model_dir) / REGRESSOR_FILENAME


def build_classifier_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """Create the patient risk classification pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=240,
                    max_depth=14,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_regressor_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """Create the insurance cost regression pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=260,
                    max_depth=16,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_models(
    data_path: str | Path | None = None,
    model_dir: str | Path = MODEL_DIR,
    reports_dir: str | Path = REPORTS_DIR,
    n_rows: int = DEFAULT_N_ROWS,
    random_state: int = RANDOM_STATE,
    force_data: bool = False,
) -> dict[str, Any]:
    """Generate data if needed, train both models, and save artifacts."""
    resolved_data_path = Path(data_path) if data_path is not None else default_data_path()
    resolved_model_dir = Path(model_dir)
    resolved_reports_dir = Path(reports_dir)
    resolved_model_dir.mkdir(parents=True, exist_ok=True)
    resolved_reports_dir.mkdir(parents=True, exist_ok=True)

    df = ensure_dataset(
        path=resolved_data_path,
        n_rows=n_rows,
        random_state=random_state,
        force=force_data,
    )
    x, y_risk, y_cost = split_features_targets(df)

    x_train, x_test, y_risk_train, y_risk_test, y_cost_train, y_cost_test = train_test_split(
        x,
        y_risk,
        y_cost,
        test_size=0.2,
        random_state=random_state,
        stratify=y_risk,
    )

    classifier = build_classifier_pipeline(random_state=random_state)
    regressor = build_regressor_pipeline(random_state=random_state)

    classifier.fit(x_train, y_risk_train)
    regressor.fit(x_train, y_cost_train)

    risk_predictions = classifier.predict(x_test)
    cost_predictions = regressor.predict(x_test)

    metrics = {
        "rows": int(len(df)),
        "features": FEATURE_COLUMNS,
        "data_path": str(resolved_data_path),
        "classifier_path": str(classifier_path(resolved_model_dir)),
        "regressor_path": str(regressor_path(resolved_model_dir)),
        "classification": {
            "accuracy": round(float(accuracy_score(y_risk_test, risk_predictions)), 4),
            "report": classification_report(
                y_risk_test,
                risk_predictions,
                output_dict=True,
                zero_division=0,
            ),
        },
        "regression": {
            "mae": round(float(mean_absolute_error(y_cost_test, cost_predictions)), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(y_cost_test, cost_predictions))), 2),
            "r2": round(float(r2_score(y_cost_test, cost_predictions)), 4),
        },
    }

    joblib.dump(classifier, classifier_path(resolved_model_dir))
    joblib.dump(regressor, regressor_path(resolved_model_dir))

    metrics_path = resolved_reports_dir / METRICS_FILENAME
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metrics["metrics_path"] = str(metrics_path)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HealthRisk-AI models.")
    parser.add_argument("--rows", type=int, default=DEFAULT_N_ROWS, help="Number of rows to generate.")
    parser.add_argument(
        "--force-data",
        action="store_true",
        help="Regenerate the synthetic dataset before training.",
    )
    args = parser.parse_args()

    metrics = train_models(n_rows=args.rows, force_data=args.force_data)

    print("HealthRisk-AI training complete")
    print(f"Dataset: {metrics['data_path']}")
    print(f"Risk classifier: {metrics['classifier_path']}")
    print(f"Cost regressor: {metrics['regressor_path']}")
    print(f"Metrics: {metrics['metrics_path']}")
    print(f"Classification accuracy: {metrics['classification']['accuracy']}")
    print(f"Cost MAE: {metrics['regression']['mae']}")


if __name__ == "__main__":
    main()
