"""Correctness + leakage tests for src/aml/features.py.

1. Brute force: recompute features the slow, obvious way (filter rows strictly
   before t) and compare with the fast sliding-window implementation.
2. No-future test: deleting all future transactions must not change any feature
   of the remaining (earlier) transactions.
"""

import numpy as np
import pandas as pd
import pytest

from aml.features import WINDOWS, build_features, estimate_usd_rates

RATES = {"US Dollar": 1.0, "Euro": 1.1, "Yuan": 0.14}


def make_transactions(n=1500, n_accounts=40, seed=0):
    rng = np.random.default_rng(seed)
    minutes = np.sort(rng.integers(0, 9 * 24 * 60, n))
    minutes[10:20] = minutes[10]  # force same-minute ties
    ts = pd.Timestamp("2022-09-01") + pd.to_timedelta(minutes, unit="m")
    banks = rng.integers(1, 4, (n, 2))
    accs = rng.integers(0, n_accounts, (n, 2))
    cur = rng.choice(list(RATES), n)
    amt = np.round(rng.lognormal(6, 1.5, n), 2)
    return pd.DataFrame({
        "timestamp": ts, "from_bank": banks[:, 0], "from_account": [f"A{a}" for a in accs[:, 0]],
        "to_bank": banks[:, 1], "to_account": [f"A{a}" for a in accs[:, 1]],
        "amount_received": amt, "receiving_currency": cur, "amount_paid": amt,
        "payment_currency": cur, "payment_format": rng.choice(["ACH", "Wire", "Cash"], n),
        "label": (rng.random(n) < 0.05).astype(int),
    })


@pytest.fixture(scope="module")
def data():
    df = make_transactions()
    return df, build_features(df, RATES)


def _brute(df, i):
    """Slow reference implementation for row i."""
    row = df.iloc[i]
    s = (row.from_bank, row.from_account); r = (row.to_bank, row.to_account)
    past = df[df["timestamp"] < row.timestamp]
    src = list(zip(past.from_bank, past.from_account)); dst = list(zip(past.to_bank, past.to_account))
    past = past.assign(src=src, dst=dst)
    out = {}
    for w, ns in WINDOWS.items():
        win = past[past["timestamp"] >= row.timestamp - pd.Timedelta(ns, unit="ns")]
        so = win[win["src"] == s]; ri = win[win["dst"] == r]
        out[f"s_out_cnt_{w}"] = len(so); out[f"s_out_uniq_{w}"] = so["dst"].nunique()
        out[f"r_in_cnt_{w}"] = len(ri); out[f"r_in_uniq_{w}"] = ri["src"].nunique()
        if w == "24h":
            si = win[win["dst"] == s]; ro = win[win["src"] == r]
            out["s_in_cnt_24h"] = len(si); out["r_out_cnt_24h"] = len(ro)
            usd = so["amount_paid"] * so["payment_currency"].astype(object).map(RATES).astype(float)
            out["s_out_usd_24h"] = usd.sum()
        if w == "7d":
            out["reverse_flow_7d"] = int(((win["src"] == r) & (win["dst"] == s)).any())
    out["pair_prev_cnt"] = int(((past["src"] == s) & (past["dst"] == r)).sum())
    return out


def test_matches_brute_force(data):
    df, f = data
    rng = np.random.default_rng(1)
    for i in rng.choice(len(df), 120, replace=False):
        expected = _brute(df, i)
        for col, val in expected.items():
            assert f.loc[i, col] == pytest.approx(val, rel=1e-4, abs=1e-2), (i, col)


def test_same_minute_transactions_do_not_see_each_other(data):
    df, f = data
    tied = df.index[df["timestamp"] == df.loc[10, "timestamp"]]
    assert len(tied) >= 10
    # all tied rows have the same timestamp, so none can be in another's "past"
    for i in tied:
        assert f.loc[i, "pair_prev_cnt"] == _brute(df, i)["pair_prev_cnt"]


def test_deleting_the_future_changes_nothing(data):
    df, f_full = data
    cutoff = df["timestamp"].quantile(0.6)
    early = df[df["timestamp"] < cutoff]
    f_early = build_features(early, RATES)
    pd.testing.assert_frame_equal(f_full.loc[early.index].reset_index(drop=True),
                                  f_early.reset_index(drop=True), check_dtype=False,
                                  check_categorical=False, rtol=1e-4)


def test_usd_rate_estimation():
    df = pd.DataFrame({"amount_paid": [100.0, 200.0, 50.0], "amount_received": [110.0, 220.0, 350.0],
                       "payment_currency": ["Euro", "Euro", "US Dollar"],
                       "receiving_currency": ["US Dollar", "US Dollar", "Yuan"]})
    rates = estimate_usd_rates(df)
    assert rates["Euro"] == pytest.approx(1.1)
    assert rates["Yuan"] == pytest.approx(1 / 7.0)


def test_usd_rate_chains_through_other_currencies():
    # Yen is only ever exchanged with Euro; Euro with USD.
    df = pd.DataFrame({"amount_paid": [100.0, 1000.0], "amount_received": [110.0, 6.0],
                       "payment_currency": ["Euro", "Yen"], "receiving_currency": ["US Dollar", "Euro"]})
    rates = estimate_usd_rates(df)
    assert rates["Yen"] == pytest.approx(0.006 * 1.1)
