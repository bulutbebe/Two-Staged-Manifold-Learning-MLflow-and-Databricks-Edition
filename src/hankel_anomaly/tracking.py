"""Small helpers for configuring MLflow experiment tracking."""

import os

import mlflow

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_EXPERIMENT_NAME = "hankel-anomaly-detection"


def configure_mlflow(experiment_name: str = DEFAULT_EXPERIMENT_NAME) -> str:
    """Configure MLflow to use the local server, unless overridden by environment."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return tracking_uri
