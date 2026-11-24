"""Checks on the saved Step 6 models (skipped if models/ is absent)."""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from aml.config import PROJECT_ROOT
from aml.data import PROCESSED, load_split

MODELS = PROJECT_ROOT / "models"
pytestmark = pytest.mark.skipif(not (MODELS / "model_meta.json").exists() or
                                not (PROCESSED / "val_sample.parquet").exists(),
                                reason="trained models / data not present")


@pytest.fixture(scope="module")
def setup():
    meta = json.loads((MODELS / "model_meta.json").read_text())
    booster = lgb.Booster(model_file=str(MODELS / "lgbm_monotone.txt"))
    X = load_split("val_sample")[meta["features"]].sample(300, random_state=0).reset_index(drop=True)
    for c, cats in meta["categories"].items():
        X[c] = pd.Categorical(X[c].astype(str), categories=cats)
    return meta, booster, X


def test_monotonic_constraints_hold(setup):
    """Raising a +1 feature never lowers the score; raising a -1 feature never raises it."""
    meta, booster, X = setup
    for feat, sign in meta["monotone_constraints"].items():
        lo = booster.predict(X)
        X_hi = X.copy(); X_hi[feat] = X_hi[feat] * 2 + 5
        hi = booster.predict(X_hi)
        assert (sign * (hi - lo) >= -1e-12).all(), f"constraint violated for {feat}"


def test_model_uses_expected_features(setup):
    meta, booster, _ = setup
    assert booster.feature_name() == meta["features"]
    assert "weekday" not in meta["features"]


def test_champion_registered_in_mlflow():
    import mlflow
    from aml.tracking import setup_mlflow
    setup_mlflow()
    mv = mlflow.MlflowClient().get_model_version_by_alias("aml-lgbm", "champion")
    assert int(mv.version) >= 1
