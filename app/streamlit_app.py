"""Streamlit dashboard for HealthRisk-AI."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.predict import load_models, predict_patient
from src.preprocessing import ensure_dataset


st.set_page_config(
    page_title="HealthRisk-AI",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_health_data() -> pd.DataFrame:
    return ensure_dataset()


@st.cache_resource(show_spinner=False)
def load_health_models():
    return load_models(auto_train=True)


def build_patient_payload() -> dict[str, object]:
    with st.sidebar:
        st.header("Patient Profile")
        with st.form("patient_form"):
            age = st.slider("Age", min_value=18, max_value=90, value=46)
            gender = st.selectbox("Gender", options=["Female", "Male"], index=0)
            bmi = st.slider("BMI", min_value=16.0, max_value=45.0, value=27.5, step=0.1)
            blood_pressure = st.slider("Blood Pressure", min_value=90, max_value=205, value=128)
            glucose = st.slider("Glucose", min_value=65, max_value=260, value=102)
            cholesterol = st.slider("Cholesterol", min_value=120, max_value=340, value=190)
            smoker = st.selectbox("Smoker", options=["No", "Yes"], index=0)
            st.form_submit_button("Run Prediction", use_container_width=True)

    return {
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "blood_pressure": blood_pressure,
        "glucose": glucose,
        "cholesterol": cholesterol,
        "smoker": smoker,
    }


def risk_gauge(score: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#0f766e"},
                "steps": [
                    {"range": [0, 35], "color": "#dcfce7"},
                    {"range": [35, 60], "color": "#fef3c7"},
                    {"range": [60, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#111827", "width": 3},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=10))
    return fig


def probability_chart(probabilities: dict[str, float]) -> go.Figure:
    probability_df = pd.DataFrame(
        {
            "Risk": list(probabilities.keys()),
            "Probability": [value * 100 for value in probabilities.values()],
        }
    )
    fig = px.bar(
        probability_df,
        x="Risk",
        y="Probability",
        color="Risk",
        color_discrete_map={"Low": "#16a34a", "Medium": "#ca8a04", "High": "#dc2626"},
        range_y=[0, 100],
    )
    fig.update_layout(showlegend=False, height=260, margin=dict(l=20, r=20, t=30, b=20))
    fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
    return fig


def render_prediction(patient: dict[str, object]) -> None:
    prediction = predict_patient(patient)

    metric_cols = st.columns(3)
    metric_cols[0].metric("Model Risk", prediction["patient_risk"])
    metric_cols[1].metric("Risk Category", prediction["risk_category"])
    metric_cols[2].metric("Insurance Cost", f"${prediction['insurance_cost']:,.2f}")

    chart_cols = st.columns([1, 1])
    with chart_cols[0]:
        st.subheader("Risk Score")
        st.plotly_chart(risk_gauge(prediction["risk_score"]), use_container_width=True)
    with chart_cols[1]:
        st.subheader("Risk Probabilities")
        st.plotly_chart(
            probability_chart(prediction["risk_probabilities"]),
            use_container_width=True,
        )


def render_dataset_charts(df: pd.DataFrame) -> None:
    st.subheader("Population Overview")

    chart_cols = st.columns(2)
    risk_counts = df["patient_risk"].value_counts().reset_index()
    risk_counts.columns = ["patient_risk", "count"]

    with chart_cols[0]:
        fig = px.bar(
            risk_counts,
            x="patient_risk",
            y="count",
            color="patient_risk",
            color_discrete_map={"Low": "#16a34a", "Medium": "#ca8a04", "High": "#dc2626"},
            labels={"patient_risk": "Risk", "count": "Patients"},
        )
        fig.update_layout(showlegend=False, height=330, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with chart_cols[1]:
        fig = px.box(
            df,
            x="patient_risk",
            y="insurance_cost",
            color="patient_risk",
            color_discrete_map={"Low": "#16a34a", "Medium": "#ca8a04", "High": "#dc2626"},
            labels={"patient_risk": "Risk", "insurance_cost": "Insurance Cost"},
        )
        fig.update_layout(showlegend=False, height=330, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    scatter = px.scatter(
        df.sample(min(len(df), 1200), random_state=7),
        x="age",
        y="insurance_cost",
        color="patient_risk",
        size="bmi",
        hover_data=["gender", "blood_pressure", "glucose", "cholesterol", "smoker"],
        color_discrete_map={"Low": "#16a34a", "Medium": "#ca8a04", "High": "#dc2626"},
        labels={"age": "Age", "insurance_cost": "Insurance Cost", "patient_risk": "Risk"},
    )
    scatter.update_layout(height=430, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(scatter, use_container_width=True)


def main() -> None:
    df = load_health_data()
    load_health_models()

    st.title("HealthRisk-AI")
    patient = build_patient_payload()

    render_prediction(patient)
    st.divider()

    summary_cols = st.columns(4)
    summary_cols[0].metric("Patients", f"{len(df):,}")
    summary_cols[1].metric("Average Age", f"{df['age'].mean():.1f}")
    summary_cols[2].metric("Average BMI", f"{df['bmi'].mean():.1f}")
    summary_cols[3].metric("Average Cost", f"${df['insurance_cost'].mean():,.0f}")

    render_dataset_charts(df)


if __name__ == "__main__":
    main()
