"""Rule-based clinical risk scoring utilities.

The machine-learning classifier predicts the learned risk label, while this
module provides a transparent score from 0 to 100 that can be shown alongside
the model output.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


LOW_RISK_THRESHOLD = 35.0
HIGH_RISK_THRESHOLD = 60.0


def _as_dataframe(patient_data: Mapping[str, Any] | pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Convert a single patient or batch of patients into a DataFrame."""
    if isinstance(patient_data, pd.DataFrame):
        return patient_data.copy()
    if isinstance(patient_data, pd.Series):
        return patient_data.to_frame().T
    if isinstance(patient_data, Mapping):
        return pd.DataFrame([patient_data])
    raise TypeError("patient_data must be a mapping, pandas Series, or pandas DataFrame")


def _numeric_column(df: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column not in df:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype("float64")


def _smoker_flag(df: pd.DataFrame) -> pd.Series:
    if "smoker" not in df:
        return pd.Series(False, index=df.index)
    return df["smoker"].astype(str).str.strip().str.lower().isin(
        {"yes", "y", "true", "1", "smoker"}
    )


def calculate_risk_scores(
    patient_data: Mapping[str, Any] | pd.Series | pd.DataFrame,
) -> pd.Series:
    """Calculate transparent health risk scores on a 0 to 100 scale."""
    df = _as_dataframe(patient_data)

    age = _numeric_column(df, "age", 45.0)
    bmi = _numeric_column(df, "bmi", 25.0)
    blood_pressure = _numeric_column(df, "blood_pressure", 120.0)
    glucose = _numeric_column(df, "glucose", 95.0)
    cholesterol = _numeric_column(df, "cholesterol", 180.0)
    smoker = _smoker_flag(df)

    age_component = np.clip((age - 22.0) / 55.0, 0.0, 1.0) * 22.0
    bmi_component = np.clip((bmi - 22.0) / 15.0, 0.0, 1.0) * 16.0
    bp_component = np.clip((blood_pressure - 112.0) / 60.0, 0.0, 1.0) * 18.0
    glucose_component = np.clip((glucose - 90.0) / 110.0, 0.0, 1.0) * 18.0
    cholesterol_component = np.clip((cholesterol - 170.0) / 120.0, 0.0, 1.0) * 14.0
    smoker_component = smoker.astype(float) * 12.0

    score = (
        age_component
        + bmi_component
        + bp_component
        + glucose_component
        + cholesterol_component
        + smoker_component
    )

    return pd.Series(np.round(np.clip(score, 0.0, 100.0), 2), index=df.index, name="risk_score")


def categorize_risk_score(score: float) -> str:
    """Map a numeric risk score to Low, Medium, or High."""
    if score < LOW_RISK_THRESHOLD:
        return "Low"
    if score < HIGH_RISK_THRESHOLD:
        return "Medium"
    return "High"


def assign_risk_categories(scores: pd.Series | np.ndarray | list[float]) -> pd.Series:
    """Assign risk category labels for a sequence of risk scores."""
    score_series = pd.Series(scores)
    return score_series.apply(categorize_risk_score).rename("risk_category")


def score_patient(patient_data: Mapping[str, Any] | pd.Series | pd.DataFrame) -> dict[str, Any] | pd.DataFrame:
    """Return risk score and category for a single patient or a patient batch."""
    df = _as_dataframe(patient_data)
    scores = calculate_risk_scores(df)
    categories = assign_risk_categories(scores)

    result = pd.DataFrame(
        {
            "risk_score": scores.values,
            "risk_category": categories.values,
        },
        index=df.index,
    )

    if isinstance(patient_data, (Mapping, pd.Series)):
        first = result.iloc[0]
        return {
            "risk_score": float(first["risk_score"]),
            "risk_category": str(first["risk_category"]),
        }

    return result
