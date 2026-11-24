"""Leak-free transaction features.

Every feature for a transaction at time t uses ONLY transactions with timestamp < t
(strictly earlier). Transactions in the same minute are treated as simultaneous and
never see each other. This is what makes the time-based evaluation honest.

Uses only numpy + pandas, so the same file runs on Kaggle and on the laptop.

Input: a DataFrame with the standard columns from configs/schema.yaml
(timestamp, from_bank, from_account, to_bank, to_account, amount_received,
receiving_currency, amount_paid, payment_currency, payment_format, [label]).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

HOUR_NS = 3_600 * 10**9
WINDOWS = {"1h": 1 * HOUR_NS, "24h": 24 * HOUR_NS, "7d": 7 * 24 * HOUR_NS}
ALL_HISTORY = 10**18  # effectively infinite window (≈31 years in ns)

CATEGORICAL = ["payment_format", "payment_currency"]


# --------------------------------------------------------------------------- #
# Core algorithm: sliding-window statistics with two pointers + a hash map
# --------------------------------------------------------------------------- #
def window_stats(q_key, q_time, e_key, e_time, e_cpty, e_amount, windows: dict[str, int]):
    """For each query (key k, time t) and each window W, aggregate the events with
    the same key and event time in [t - W, t):
        cnt, distinct counterparties, sum and sum of squares of amount.
    Also returns `last_t`: time of the latest event with time < t (or -1).

    One pass over sorted queries/events, O(n) amortised. Results are returned
    in the ORIGINAL query order.
    """
    q_key = np.asarray(q_key, dtype=np.int64); q_time = np.asarray(q_time, dtype=np.int64)
    e_key = np.asarray(e_key, dtype=np.int64); e_time = np.asarray(e_time, dtype=np.int64)
    e_cpty = np.asarray(e_cpty, dtype=np.int64); e_amount = np.asarray(e_amount, dtype=np.float64)

    qo = np.lexsort((q_time, q_key)); eo = np.lexsort((e_time, e_key))
    QK, QT = q_key[qo].tolist(), q_time[qo].tolist()
    EK, ET = e_key[eo].tolist(), e_time[eo].tolist()
    EC, EA = e_cpty[eo].tolist(), e_amount[eo].tolist()
    nq, ne = len(QK), len(EK)
    names = list(windows); widths = [windows[w] for w in names]; nw = len(names)

    out_cnt = [[0] * nq for _ in range(nw)]; out_dst = [[0] * nq for _ in range(nw)]
    out_sum = [[0.0] * nq for _ in range(nw)]; out_ss = [[0.0] * nq for _ in range(nw)]
    out_last = [-1] * nq

    add = 0
    rem = [0] * nw
    cnt_map = [dict() for _ in range(nw)]
    n_in = [0] * nw; s = [0.0] * nw; ss = [0.0] * nw
    cur_key = None; last_t = -1

    for qi in range(nq):
        k = QK[qi]; t = QT[qi]
        if k != cur_key:  # new account: reset all state
            cur_key = k; last_t = -1
            while add < ne and EK[add] < k:
                add += 1
            for w in range(nw):
                rem[w] = add; cnt_map[w].clear(); n_in[w] = 0; s[w] = 0.0; ss[w] = 0.0
        # add events of this key that happened strictly before t
        while add < ne and EK[add] == k and ET[add] < t:
            c = EC[add]; a = EA[add]; last_t = ET[add]
            for w in range(nw):
                m = cnt_map[w]; m[c] = m.get(c, 0) + 1
                n_in[w] += 1; s[w] += a; ss[w] += a * a
            add += 1
        # drop events older than t - W (per window)
        for w in range(nw):
            lo = t - widths[w]; r = rem[w]; m = cnt_map[w]
            while r < add and ET[r] < lo:
                c = EC[r]; a = EA[r]; v = m[c] - 1
                if v: m[c] = v
                else: del m[c]
                n_in[w] -= 1; s[w] -= a; ss[w] -= a * a
                r += 1
            rem[w] = r
            out_cnt[w][qi] = n_in[w]; out_dst[w][qi] = len(m)
            out_sum[w][qi] = s[w]; out_ss[w][qi] = ss[w]
        out_last[qi] = last_t

    inv = np.empty(nq, dtype=np.int64); inv[qo] = np.arange(nq)
    res = {}
    for w, name in enumerate(names):
        res[f"cnt_{name}"] = np.asarray(out_cnt[w])[inv]
        res[f"distinct_{name}"] = np.asarray(out_dst[w])[inv]
        res[f"sum_{name}"] = np.asarray(out_sum[w])[inv]
        res[f"sumsq_{name}"] = np.asarray(out_ss[w])[inv]
    res["last_t"] = np.asarray(out_last)[inv]
    return res


# --------------------------------------------------------------------------- #
# Exchange rates (estimated from the data itself)
# --------------------------------------------------------------------------- #
def estimate_usd_rates(df: pd.DataFrame) -> dict[str, float]:
    """USD per unit of each currency, estimated from cross-currency transactions
    (median of amount_received / amount_paid). Pass TRAIN-period rows only."""
    rates = {"US Dollar": 1.0}
    pc = df["payment_currency"].astype(str); rc = df["receiving_currency"].astype(str)
    ok = (df["amount_paid"] > 0) & (df["amount_received"] > 0)
    to_usd = df[ok & (rc == "US Dollar") & (pc != "US Dollar")]
    for cur, g in to_usd.groupby(pc[to_usd.index]):
        rates[cur] = float((g["amount_received"] / g["amount_paid"]).median())
    from_usd = df[ok & (pc == "US Dollar") & (rc != "US Dollar")]
    for cur, g in from_usd.groupby(rc[from_usd.index]):
        if cur not in rates:
            rates[cur] = float(1.0 / (g["amount_received"] / g["amount_paid"]).median())
    # Chain through other currencies (e.g. Yen -> Euro -> USD) for any still missing.
    cross = df[ok & (pc != rc)]
    pair_ratio = (cross["amount_received"] / cross["amount_paid"]).groupby(
        [pc[cross.index], rc[cross.index]]).median()            # units of rc per unit of pc
    changed = True
    while changed:
        changed = False
        for (p, r), ratio in pair_ratio.items():
            if p not in rates and r in rates:      # 1 p = ratio r  -> usd(p) = ratio * usd(r)
                rates[p] = float(ratio * rates[r]); changed = True
            elif r not in rates and p in rates:    # 1 r = p / ratio
                rates[r] = float(rates[p] / ratio); changed = True
    return rates


# --------------------------------------------------------------------------- #
# Feature builder
# --------------------------------------------------------------------------- #
def _std(cnt, s, ss):
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
        var = np.where(cnt > 1, ss / np.maximum(cnt, 1) - mean**2, np.nan)
    return mean, np.sqrt(np.clip(var, 0, None))


def build_features(df: pd.DataFrame, usd_rates: dict[str, float]) -> pd.DataFrame:
    """Return a DataFrame of features, aligned row-by-row with `df`."""
    df = df.reset_index(drop=True)
    # Force nanosecond unit: pandas 3 may store datetimes in s/us, which would
    # silently make every window 1000x+ too wide.
    t = df["timestamp"].astype("datetime64[ns]").astype("int64").to_numpy()

    # Accounts are identified by (bank, account) to avoid ID collisions across banks.
    src = df["from_bank"].astype(str) + "_" + df["from_account"].astype(str)
    dst = df["to_bank"].astype(str) + "_" + df["to_account"].astype(str)
    codes, uniques = pd.factorize(pd.concat([src, dst], ignore_index=True))
    n = len(df); n_acc = len(uniques)
    s_code = codes[:n].astype(np.int64); r_code = codes[n:].astype(np.int64)

    rate = df["payment_currency"].astype(str).map(usd_rates).astype(float).fillna(1.0).to_numpy()
    amt = df["amount_paid"].to_numpy(dtype=np.float64) * rate

    f = pd.DataFrame(index=df.index)

    # 1) Sender's outgoing history (velocity, fan-out, amount behaviour)
    so = window_stats(s_code, t, s_code, t, r_code, amt, WINDOWS)
    for w in WINDOWS:
        f[f"s_out_cnt_{w}"] = so[f"cnt_{w}"]
        f[f"s_out_uniq_{w}"] = so[f"distinct_{w}"]
    f["s_out_usd_24h"] = so["sum_24h"]; f["s_out_usd_7d"] = so["sum_7d"]
    mean7, std7 = _std(so["cnt_7d"], so["sum_7d"], so["sumsq_7d"])
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (amt - mean7) / std7
    f["s_amt_z_7d"] = np.where(np.isfinite(z), z, 0.0)
    f["s_amt_ratio_7d"] = np.where(so["cnt_7d"] > 0, amt / np.maximum(mean7, 1e-9), 1.0)
    f["s_hours_since_prev_out"] = np.where(so["last_t"] >= 0, (t - so["last_t"]) / HOUR_NS, -1.0)

    # 2) Receiver's incoming history (fan-in)
    ri = window_stats(r_code, t, r_code, t, s_code, amt, WINDOWS)
    for w in WINDOWS:
        f[f"r_in_cnt_{w}"] = ri[f"cnt_{w}"]
        f[f"r_in_uniq_{w}"] = ri[f"distinct_{w}"]
    f["r_in_usd_24h"] = ri["sum_24h"]

    # 3) Cross-role: did the sender just RECEIVE money? does the receiver SEND onward?
    #    (pass-through / scatter-gather behaviour)
    w24 = {"24h": WINDOWS["24h"]}
    si = window_stats(s_code, t, r_code, t, s_code, amt, w24)
    f["s_in_cnt_24h"] = si["cnt_24h"]; f["s_in_uniq_24h"] = si["distinct_24h"]
    f["s_in_usd_24h"] = si["sum_24h"]
    ro = window_stats(r_code, t, s_code, t, r_code, amt, w24)
    f["r_out_cnt_24h"] = ro["cnt_24h"]; f["r_out_uniq_24h"] = ro["distinct_24h"]
    f["s_passthrough_ratio_24h"] = np.where(si["sum_24h"] > 0, so["sum_24h"] / np.maximum(si["sum_24h"], 1e-9), 0.0)

    # 4) Pair history: first time paying this counterparty? did money flow back before?
    pair = s_code * n_acc + r_code
    rev = r_code * n_acc + s_code
    zeros = np.zeros(n, dtype=np.int64); zf = np.zeros(n)
    ph = window_stats(pair, t, pair, t, zeros, zf, {"all": ALL_HISTORY})
    f["pair_prev_cnt"] = ph["cnt_all"]
    f["pair_first_time"] = (ph["cnt_all"] == 0).astype(np.int8)
    rv = window_stats(rev, t, pair, t, zeros, zf, {"7d": WINDOWS["7d"]})
    f["reverse_flow_7d"] = (rv["cnt_7d"] > 0).astype(np.int8)  # 2-hop cycle: r -> s seen before

    # 5) Account age (hours since first appearance in any role; first appearance <= t)
    first = pd.Series(np.concatenate([t, t])).groupby(codes).min().to_numpy()
    f["s_age_hours"] = (t - first[s_code]) / HOUR_NS
    f["r_age_hours"] = (t - first[r_code]) / HOUR_NS

    # 6) Transaction attributes
    f["log_amount_usd"] = np.log1p(np.clip(amt, 0, None))
    f["round_amount"] = ((df["amount_paid"] % 100 == 0) & (df["amount_paid"] > 0)).astype(np.int8).to_numpy()
    f["is_ach"] = (df["payment_format"].astype(str) == "ACH").astype(np.int8).to_numpy()
    f["cross_bank"] = (df["from_bank"] != df["to_bank"]).astype(np.int8).to_numpy()
    f["self_transfer"] = (s_code == r_code).astype(np.int8)
    f["cross_currency"] = (df["payment_currency"].astype(str) != df["receiving_currency"].astype(str)).astype(np.int8).to_numpy()
    f["hour"] = df["timestamp"].dt.hour.to_numpy(np.int8)
    f["weekday"] = df["timestamp"].dt.day_of_week.to_numpy(np.int8)
    for c in CATEGORICAL:
        f[c] = df[c].astype(str).astype("category")

    # compact dtypes
    for c in f.columns:
        if f[c].dtype == np.float64:
            f[c] = f[c].astype(np.float32)
        elif f[c].dtype == np.int64:
            f[c] = f[c].astype(np.int32)
    return f


def feature_names(f: pd.DataFrame) -> list[str]:
    return list(f.columns)
