# HealthRisk-AI

HealthRisk-AI is a complete Python project for synthetic healthcare risk
analysis. It generates patient data, trains two Random Forest models, saves the
trained pipelines with Joblib, and serves predictions through a Streamlit
dashboard.

## Project Structure

```text
HealthRisk-AI/
├── data/
├── models/
├── notebooks/
├── reports/
├── app/
│   └── streamlit_app.py
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── risk_engine.py
├── requirements.txt
├── README.md
└── main.py
```

## Features

- Generates a 5000-row synthetic healthcare dataset.
- Includes age, gender, BMI, blood pressure, glucose, cholesterol, smoker,
  patient risk, and insurance cost fields.
- Trains a `RandomForestClassifier` for patient risk prediction.
- Trains a `RandomForestRegressor` for insurance cost prediction.
- Uses a shared preprocessing pipeline with scaling and one-hot encoding.
- Saves trained model pipelines to the `models/` folder with Joblib.
- Provides a transparent 0-100 risk scoring engine.
- Includes a Streamlit dashboard with patient inputs, predictions, and Plotly
  visualizations.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Generate Data and Train Models

```powershell
python main.py
```

This creates:

- `data/synthetic_healthcare_data.csv`
- `models/patient_risk_classifier.joblib`
- `models/insurance_cost_regressor.joblib`
- `reports/training_metrics.json`

You can also train directly:

```powershell
python src/train.py --force-data
```

## Run the Dashboard

```powershell
streamlit run app/streamlit_app.py
```

The dashboard automatically generates data and trains models if artifacts are
missing.

## Programmatic Prediction

```python
from src.predict import predict_patient

patient = {
    "age": 52,
    "gender": "Female",
    "bmi": 29.4,
    "blood_pressure": 138,
    "glucose": 118,
    "cholesterol": 214,
    "smoker": "No",
}

prediction = predict_patient(patient)
print(prediction)
```

Example output:

```python
{
    "patient_risk": "Medium",
    "risk_score": 37.41,
    "risk_category": "Medium",
    "risk_probabilities": {"Low": 0.4563, "Medium": 0.5437, "High": 0.0},
    "insurance_cost": 16014.85,
}
```

## Notes

This project uses synthetic data for development and demonstration. It is not a
clinical decision system and should not be used for medical diagnosis,
treatment, underwriting, or real patient triage.
