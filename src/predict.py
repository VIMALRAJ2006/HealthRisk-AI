"""Prediction utilities for HealthRisk-AI."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.preprocessing import FEATURE_COLUMNS
from src.risk_engine import score_patient
from src.train import MODEL_DIR, classifier_path, regressor_path, train_models


RISK_PROBABILITY_ORDER = ["Low", "Medium", "High"]


def _as_dataframe(patient_data: Mapping[str, Any] | pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(patient_data, pd.DataFrame):
        df = patient_data.copy()
    elif isinstance(patient_data, pd.Series):
        df = patient_data.to_frame().T
    elif isinstance(patient_data, Mapping):
        df = pd.DataFrame([patient_data])
    else:
        raise TypeError("patient_data must be a mapping, pandas Series, or pandas DataFrame")

    missing_columns = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required feature columns: {missing_columns}")

    return df[FEATURE_COLUMNS].copy()


def ensure_models_exist(model_dir: str | Path = MODEL_DIR) -> None:
    """Train models when saved artifacts are not available."""
    if not classifier_path(model_dir).exists() or not regressor_path(model_dir).exists():
        train_models(model_dir=model_dir)


def load_models(model_dir: str | Path = MODEL_DIR, auto_train: bool = True) -> tuple[Any, Any]:
    """Load saved classifier and regressor pipelines."""
    if auto_train:
        ensure_models_exist(model_dir)

    classifier = joblib.load(classifier_path(model_dir))
    regressor = joblib.load(regressor_path(model_dir))
    return classifier, regressor


def predict_patient(
    patient_data: Mapping[str, Any] | pd.Series,
    model_dir: str | Path = MODEL_DIR,
    auto_train: bool = True,
) -> dict[str, Any]:
    """Predict risk class, risk score, and insurance cost for one patient."""
    classifier, regressor = load_models(model_dir=model_dir, auto_train=auto_train)
    patient_df = _as_dataframe(patient_data)

    predicted_risk = str(classifier.predict(patient_df)[0])
    predicted_cost = float(regressor.predict(patient_df)[0])
    raw_probabilities = {
        str(label): float(probability)
        for label, probability in zip(classifier.classes_, classifier.predict_proba(patient_df)[0])
    }
    risk_probabilities = {
        label: round(raw_probabilities.get(label, 0.0), 4)
        for label in RISK_PROBABILITY_ORDER
    }
    risk_score = score_patient(patient_df.iloc[0])

    return {
        "patient_risk": predicted_risk,
        "risk_score": risk_score["risk_score"],
        "risk_category": risk_score["risk_category"],
        "risk_probabilities": risk_probabilities,
        "insurance_cost": round(predicted_cost, 2),
    }


def predict_batch(
    patient_data: pd.DataFrame,
    model_dir: str | Path = MODEL_DIR,
    auto_train: bool = True,
) -> pd.DataFrame:
    """Predict risk and cost for a batch of patients."""
    classifier, regressor = load_models(model_dir=model_dir, auto_train=auto_train)
    patient_df = _as_dataframe(patient_data)

    predictions = patient_df.copy()
    predictions["predicted_patient_risk"] = classifier.predict(patient_df)
    predictions["predicted_insurance_cost"] = regressor.predict(patient_df).round(2)

    score_df = score_patient(patient_df)
    predictions["risk_score"] = score_df["risk_score"].values
    predictions["risk_category"] = score_df["risk_category"].values

    return predictions
