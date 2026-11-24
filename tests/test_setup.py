"""Step 1 smoke tests: environment and configs are in place."""

import importlib

import pytest

from aml.config import load_schema, load_train_config

REQUIRED = ["pandas", "polars", "pyarrow", "sklearn", "statsmodels", "lightgbm",
            "xgboost", "optuna", "mlflow", "shap", "fastapi", "streamlit"]


@pytest.mark.parametrize("module", REQUIRED)
def test_library_imports(module):
    importlib.import_module(module)


def test_schema_has_required_columns():
    cols = load_schema()["columns"]
    for key in ["timestamp", "from_account", "to_account", "amount_paid", "label"]:
        assert key in cols


def test_split_fractions_sum_to_one():
    split = load_train_config()["split"]
    assert abs(sum(split.values()) - 1.0) < 1e-9


def test_laptop_safety_limits():
    cfg = load_train_config()
    assert cfg["compute"]["n_jobs"] <= 2
    assert cfg["optuna"]["timeout_seconds"] <= 300
