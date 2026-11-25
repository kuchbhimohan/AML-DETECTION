"""Tests for SHAP explanations / reason codes (skipped if model or data absent)."""

import numpy as np
import pytest

from aml.config import PROJECT_ROOT
from aml.data import PROCESSED, load_split

pytestmark = pytest.mark.skipif(not (PROJECT_ROOT / "models" / "model_meta.json").exists() or
                                not (PROCESSED / "val_sample.parquet").exists(), reason="model/data not present")


@pytest.fixture(scope="module")
def X():
    from aml.explain import prepare
    df = load_split("val_sample")
    pos = df[df.label == 1].head(60); neg = df[df.label == 0].sample(140, random_state=0)
    import pandas as pd
    return prepare(pd.concat([pos, neg]))


def test_every_feature_has_a_reason_template():
    from aml.explain import REASONS, load_model
    _, meta = load_model()
    assert set(meta["features"]) <= set(REASONS)


def test_shap_is_additive(X):
    from aml.explain import load_model, shap_values
    booster, _ = load_model()
    contrib, base = shap_values(X)
    raw = booster.predict(X, raw_score=True)
    assert np.allclose(base + contrib.sum(axis=1).to_numpy(), raw, atol=1e-8)


def test_reason_codes_are_distinct_positive_and_readable(X):
    from aml.explain import GROUPS, reason_codes
    for e in reason_codes(X):
        rs = e["reasons"]
        assert len(rs) <= 3 and 0 <= e["probability"] <= 1
        contribs = [r["contribution"] for r in rs]
        assert all(c > 0 for c in contribs) and contribs == sorted(contribs, reverse=True)
        feats = [r["feature"] for r in rs]
        assert len(set(feats)) == len(feats)
        for members in GROUPS.values():                 # merged features never appear separately
            assert not any(m in feats for m in members)
        assert all(isinstance(r["reason"], str) and len(r["reason"]) > 5 and "{" not in r["reason"] for r in rs)


def test_laundering_examples_get_reasons(X):
    from aml.explain import reason_codes
    top = reason_codes(X.head(60))                       # first 60 rows are laundering
    assert sum(len(e["reasons"]) == 3 for e in top) >= 50
