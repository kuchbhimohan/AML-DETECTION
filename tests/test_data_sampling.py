"""Tests for data loading/validation and case-control sampling."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from aml.data import (PROCESSED, DataValidationError, align_categories, load_raw_transactions,
                      load_split, validate, xyw)
from aml.sampling import case_control_sample, correct_probability, weighted_positive_rate

FEATS = ["f1", "payment_format"]


def _frame(n=20):
    return pd.DataFrame({"row_id": np.arange(n), "label": [1] + [0] * (n - 1),
                         "timestamp": pd.date_range("2022-09-01", periods=n, freq="min"),
                         "f1": np.arange(n, dtype=float), "payment_format": ["ACH", "Wire"] * (n // 2)})


# ---------------- validation ----------------
def test_validate_accepts_good_frame():
    validate(_frame(), FEATS)


@pytest.mark.parametrize("breaker, msg", [
    (lambda d: d.drop(columns="f1"), "missing"),
    (lambda d: d.assign(label=2), "label"),
    (lambda d: d.assign(f1=np.nan), "nulls"),
    (lambda d: d.assign(row_id=0), "duplicate"),
    (lambda d: d.iloc[::-1], "sorted"),
])
def test_validate_rejects_bad_frames(breaker, msg):
    with pytest.raises(DataValidationError, match=msg):
        validate(breaker(_frame()), FEATS)


def test_align_categories_uses_training_categories():
    train = _frame(); other = _frame().assign(payment_format=["Bitcoin"] * 20)
    (aligned,) = align_categories([other], reference=train)
    assert list(aligned["payment_format"].cat.categories) == ["ACH", "Wire"]
    assert aligned["payment_format"].isna().all()   # unseen category -> missing


def test_xyw_defaults_weight_to_one():
    X, y, w = xyw(_frame(), FEATS)
    assert X.shape == (20, 2) and y.sum() == 1 and (w == 1).all()


def test_load_raw_transactions_with_custom_schema(tmp_path):
    p = tmp_path / "t.csv"
    pd.DataFrame({"when": ["2022/09/02 10:00", "2022/09/01 09:00"], "src": ["a", "b"], "dst": ["b", "a"],
                  "amt": [5.0, 7.0]}).to_csv(p, index=False)
    schema = {"columns": {"timestamp": "when", "from_account": "src", "to_account": "dst",
                          "amount_paid": "amt", "label": None}, "timestamp_format": "%Y/%m/%d %H:%M"}
    df = load_raw_transactions(p, schema)
    assert list(df["from_account"]) == ["b", "a"]          # sorted by time
    assert (df["label"] == 0).all()                         # unlabelled data allowed


# ---------------- sampling ----------------
def test_case_control_keeps_all_positives_and_weights():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"label": (rng.random(100_000) < 0.002).astype(int)})
    s = case_control_sample(df, 0.1, seed=1)
    assert s["label"].sum() == df["label"].sum()
    assert len(s[s.label == 0]) == pytest.approx(0.1 * (df.label == 0).sum(), rel=0.05)
    # weighted rate in the sample estimates the true rate
    assert weighted_positive_rate(s["label"], s["weight"]) == pytest.approx(df["label"].mean(), rel=0.1)


def test_prior_correction_recovers_true_scale():
    """Unweighted model on a 5% negative sample over-predicts ~20x; correction fixes it."""
    rng = np.random.default_rng(0)
    n = 400_000
    x = rng.normal(size=(n, 1))
    p_true = 1 / (1 + np.exp(-(-7 + 1.5 * x[:, 0])))
    y = (rng.random(n) < p_true).astype(int)
    df = pd.DataFrame({"x": x[:, 0], "label": y})
    s = case_control_sample(df, 0.05, seed=2)
    model = LogisticRegression().fit(s[["x"]], s["label"])
    p_raw = model.predict_proba(df[["x"]])[:, 1]
    p_fix = correct_probability(p_raw, 0.05)
    assert p_raw.mean() > 5 * y.mean()                       # badly mis-calibrated
    assert p_fix.mean() == pytest.approx(y.mean(), rel=0.15)  # fixed
    assert np.array_equal(np.argsort(p_raw), np.argsort(p_fix))  # ranking unchanged


# ---------------- real files (skipped if Kaggle outputs are absent) ----------------
real = pytest.mark.skipif(not (PROCESSED / "feature_list.json").exists(), reason="Kaggle outputs not present")


@real
def test_real_split_files_are_valid():
    info = json.loads((PROCESSED / "split_info.json").read_text())
    for name in ["train_sample", "val_sample", "tail"]:
        df = load_split(name)                                  # validates
        assert int(df["label"].sum()) == info["files"][f"{name}.parquet"]["laundering"]


@real
def test_real_splits_do_not_overlap_in_time():
    ends = {s: load_split(s, columns=["timestamp"])["timestamp"] for s in ["train_sample", "val_full", "test_full"]}
    assert ends["train_sample"].max() < ends["val_full"].min()
    assert ends["val_full"].max() < ends["test_full"].min()
