"""MLflow setup: local SQLite backend + local artifact folder (no server needed)."""

from __future__ import annotations

import os

import mlflow

from aml.config import PROJECT_ROOT, load_train_config

os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")


def tracking_uri() -> str:
    return "sqlite:///" + (PROJECT_ROOT / "mlflow.db").as_posix()


def setup_mlflow() -> str:
    """Point MLflow at the project's database and create the experiment if needed.
    Returns the experiment id."""
    cfg = load_train_config()["mlflow"]
    mlflow.set_tracking_uri(tracking_uri())
    name = cfg["experiment_name"]
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        art = (PROJECT_ROOT / "mlartifacts").as_uri()
        exp_id = mlflow.create_experiment(name, artifact_location=art)
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment_id=exp_id)
    return exp_id
