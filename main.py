"""Entry point for generating data and training HealthRisk-AI models."""

from __future__ import annotations

from src.train import train_models


def main() -> None:
    metrics = train_models(force_data=True)

    print("HealthRisk-AI setup complete")
    print(f"Rows generated: {metrics['rows']}")
    print(f"Dataset: {metrics['data_path']}")
    print(f"Risk classifier: {metrics['classifier_path']}")
    print(f"Cost regressor: {metrics['regressor_path']}")
    print(f"Metrics: {metrics['metrics_path']}")
    print(f"Classification accuracy: {metrics['classification']['accuracy']}")
    print(f"Insurance cost MAE: {metrics['regression']['mae']}")
    print("Launch dashboard with: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
