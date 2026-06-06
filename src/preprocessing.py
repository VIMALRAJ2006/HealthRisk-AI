"""Data generation and preprocessing helpers for HealthRisk-AI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.risk_engine import assign_risk_categories, calculate_risk_scores


RANDOM_STATE = 42
DEFAULT_N_ROWS = 5000

FEATURE_COLUMNS = [
    "age",
    "gender",
    "bmi",
    "blood_pressure",
    "glucose",
    "cholesterol",
    "smoker",
]
NUMERICAL_FEATURES = ["age", "bmi", "blood_pressure", "glucose", "cholesterol"]
CATEGORICAL_FEATURES = ["gender", "smoker"]
CLASSIFICATION_TARGET = "patient_risk"
REGRESSION_TARGET = "insurance_cost"
DATASET_FILENAME = "synthetic_healthcare_data.csv"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_path() -> Path:
    return project_root() / "data" / DATASET_FILENAME


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    """Create the preprocessing pipeline used by both models."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERICAL_FEATURES),
            ("categorical", _make_one_hot_encoder(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def generate_synthetic_healthcare_data(
    n_rows: int = DEFAULT_N_ROWS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generate a realistic synthetic healthcare dataset."""
    rng = np.random.default_rng(random_state)

    age = rng.integers(18, 91, size=n_rows)
    gender = rng.choice(["Female", "Male"], size=n_rows, p=[0.52, 0.48])
    smoker = rng.choice(["No", "Yes"], size=n_rows, p=[0.76, 0.24])
    smoker_flag = smoker == "Yes"

    bmi = rng.normal(26.5 + (age - 45) * 0.035 + smoker_flag * 1.1, 4.8, size=n_rows)
    bmi = np.clip(bmi, 16.0, 45.0)

    blood_pressure = rng.normal(
        108.0 + age * 0.42 + (bmi - 25.0) * 0.75 + smoker_flag * 6.0,
        11.5,
        size=n_rows,
    )
    blood_pressure = np.clip(blood_pressure, 90.0, 205.0)

    glucose = rng.normal(
        78.0 + age * 0.36 + (bmi - 25.0) * 1.10 + smoker_flag * 8.0,
        17.0,
        size=n_rows,
    )
    glucose = np.clip(glucose, 65.0, 260.0)

    cholesterol = rng.normal(
        150.0 + age * 0.72 + (bmi - 25.0) * 1.60 + smoker_flag * 10.0,
        24.0,
        size=n_rows,
    )
    cholesterol = np.clip(cholesterol, 120.0, 340.0)

    df = pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "bmi": np.round(bmi, 1),
            "blood_pressure": np.round(blood_pressure, 0).astype(int),
            "glucose": np.round(glucose, 0).astype(int),
            "cholesterol": np.round(cholesterol, 0).astype(int),
            "smoker": smoker,
        }
    )

    noisy_scores = calculate_risk_scores(df) + rng.normal(0.0, 6.0, size=n_rows)
    noisy_scores = np.clip(noisy_scores, 0.0, 100.0)
    df[CLASSIFICATION_TARGET] = assign_risk_categories(noisy_scores).values

    risk_multiplier = noisy_scores / 100.0
    insurance_cost = (
        1800.0
        + age * 68.0
        + df["bmi"] * 115.0
        + df["blood_pressure"] * 10.0
        + df["glucose"] * 12.0
        + df["cholesterol"] * 6.0
        + smoker_flag.astype(float) * 2900.0
        + risk_multiplier * 8500.0
        + rng.normal(0.0, 1350.0, size=n_rows)
    )
    df[REGRESSION_TARGET] = np.round(np.clip(insurance_cost, 1200.0, None), 2)

    return df[
        FEATURE_COLUMNS
        + [
            CLASSIFICATION_TARGET,
            REGRESSION_TARGET,
        ]
    ]


def save_dataset(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    """Save the generated dataset as CSV."""
    data_path = Path(path) if path is not None else default_data_path()
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)
    return data_path


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the healthcare dataset."""
    data_path = Path(path) if path is not None else default_data_path()
    return pd.read_csv(data_path)


def ensure_dataset(
    path: str | Path | None = None,
    n_rows: int = DEFAULT_N_ROWS,
    random_state: int = RANDOM_STATE,
    force: bool = False,
) -> pd.DataFrame:
    """Load the dataset if it exists, otherwise generate and persist it."""
    data_path = Path(path) if path is not None else default_data_path()
    if force or not data_path.exists():
        df = generate_synthetic_healthcare_data(n_rows=n_rows, random_state=random_state)
        save_dataset(df, data_path)
        return df
    return load_dataset(data_path)


def split_features_targets(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split a full dataset into model inputs and targets."""
    return (
        df[FEATURE_COLUMNS].copy(),
        df[CLASSIFICATION_TARGET].copy(),
        df[REGRESSION_TARGET].copy(),
    )
