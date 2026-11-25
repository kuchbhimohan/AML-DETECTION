"""Build serving + dashboard artefacts (run once after Step 7).

Outputs
  models/serving.json     isotonic calibrator + risk-band thresholds (chosen on VALIDATION)
  app/data/alerts.parquet top test-period alerts with probabilities + SHAP reason codes
  app/data/curves.json    recall/precision vs alert budget (champion, logistic, rules) on the full test period
  app/data/profile.json   percentiles of key features among NORMAL traffic (for "how unusual" bars)
  app/data/*.json         copies of the reports the dashboard shows (self-contained for deployment)

Run:  uv run python -m aml.demo_data
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from aml.config import PROJECT_ROOT, REPORTS_DIR, load_train_config
from aml.data import load_split
from aml.explain import prepare, reason_codes

CFG = load_train_config()
MODELS = PROJECT_ROOT / "models"
APP_DATA = PROJECT_ROOT / "app" / "data"
CHAMP = "LightGBM monotone (champion)"
N_ALERTS = 2500
PROFILE_FEATURES = ["s_out_uniq_24h", "s_out_cnt_1h", "s_out_cnt_24h", "r_in_uniq_24h", "r_in_uniq_7d",
                    "s_in_cnt_24h", "log_amount_usd", "s_hours_since_prev_out", "pair_prev_cnt", "s_amt_ratio_7d"]


def _cached_scores():
    z = np.load(PROJECT_ROOT / "data" / "processed" / "scores_cache.npz", allow_pickle=True)
    return z["scores"].item()


def build_serving(val_scores: np.ndarray, y_val: np.ndarray) -> dict:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(val_scores, y_val)
    cal_val = iso.predict(val_scores)
    results = json.loads((REPORTS_DIR / "results.json").read_text())
    high = results["f1_operating_point"]["threshold_from_val"]          # best-F1 threshold (calibrated)
    medium = float(np.quantile(cal_val, 0.99))                         # top-1% of validation traffic
    # Cap: isotonic can output exactly 0/1 at the extremes; no model should claim certainty.
    serving = {"calibrator": {"type": "isotonic", "x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist(),
                              "max_probability": 0.99},
               "bands": {"high": float(high), "medium": float(min(medium, high))},
               "band_meaning": {"high": "best-F1 threshold chosen on validation (~76% precision on test)",
                                "medium": "inside the top-1% alert budget on validation",
                                "low": "below the alert budget"}}
    (MODELS / "serving.json").write_text(json.dumps(serving))
    return serving


def calibrate(p: np.ndarray, serving: dict) -> np.ndarray:
    c = serving["calibrator"]
    return np.minimum(np.interp(p, c["x"], c["y"]), c.get("max_probability", 1.0))


def curves(y, scores: dict, n=160) -> dict:
    grid = np.unique(np.round(np.geomspace(1e-4, 0.1, n) * len(y)).astype(int))
    out = {"budget": (grid / len(y)).tolist(), "n_transactions": int(len(y)), "n_laundering": int(y.sum()),
           "days": 2}
    for name, s in scores.items():
        caught = np.cumsum(y[np.argsort(-s, kind="stable")])
        out[name] = {"recall": (caught[grid - 1] / y.sum()).tolist(), "precision": (caught[grid - 1] / grid).tolist()}
    return out


def main():
    APP_DATA.mkdir(parents=True, exist_ok=True)
    cache = _cached_scores()
    val = load_split("val_full", columns=["label"]); test = load_split("test_full")
    yv, yt = val["label"].to_numpy(), test["label"].to_numpy()
    serving = build_serving(cache["val"][CHAMP], yv)
    print("serving bands:", serving["bands"])

    p_test = cache["test"][CHAMP]; cal = calibrate(p_test, serving)
    # curves: champion + logistic + rules (logistic refit is fast and deterministic)
    from aml.evaluate import _add_cheap_scores
    train = load_split("train_sample")
    tmp = _add_cheap_scores({"test": {}}, train, test)
    cur = curves(yt, {"LightGBM (champion)": p_test, "Logistic regression": tmp["test"]["Logistic regression"],
                      "Rules (ACH, amount)": tmp["test"]["Rules (ACH, amount)"]})
    (APP_DATA / "curves.json").write_text(json.dumps(cur))

    # alert queue: top-N riskiest test transactions, with reason codes
    top = np.argsort(-p_test, kind="stable")[:N_ALERTS]
    alerts = test.iloc[top].copy()
    alerts["p_raw"] = p_test[top]; alerts["p_cal"] = cal[top]
    alerts["rank"] = np.arange(1, len(top) + 1)
    alerts["band"] = np.where(alerts["p_cal"] >= serving["bands"]["high"], "High",
                              np.where(alerts["p_cal"] >= serving["bands"]["medium"], "Medium", "Low"))
    rc = reason_codes(prepare(alerts))
    for k in range(3):
        alerts[f"reason_{k + 1}"] = [e["reasons"][k]["reason"] if len(e["reasons"]) > k else "" for e in rc]
        alerts[f"shap_{k + 1}"] = [e["reasons"][k]["contribution"] if len(e["reasons"]) > k else 0.0 for e in rc]
    alerts["reasons_json"] = [json.dumps(e["reasons"]) for e in rc]
    alerts["amount_usd"] = np.expm1(alerts["log_amount_usd"]).round(2)
    for c in ("payment_format", "payment_currency", "split"):
        alerts[c] = alerts[c].astype(str)
    alerts.to_parquet(APP_DATA / "alerts.parquet", index=False)

    # Bin-level calibration check on the alert queue (test period): is "70%" really ~70%?
    p_uncapped = np.interp(p_test[top], serving["calibrator"]["x"], serving["calibrator"]["y"])
    bins = [(0.999, 1.01, "100% (capped to 99%)"), (0.9, 0.999, "90–99.9%"), (0.5, 0.9, "50–90%"),
            (0.3, 0.5, "30–50%"), (0.05, 0.3, "5–30%"), (0.0, 0.05, "<5%")]
    lab = alerts["label"].to_numpy()
    cal_bins = []
    for lo, hi, name in bins:
        m = (p_uncapped >= lo) & (p_uncapped < hi)
        if m.sum():
            cal_bins.append({"bin": name, "n": int(m.sum()), "mean_predicted": float(p_uncapped[m].mean()),
                             "actual": float(lab[m].mean())})
    (APP_DATA / "calibration_bins.json").write_text(json.dumps(cal_bins, indent=1))
    print(f"alerts: {len(alerts):,} rows, {int(alerts['label'].sum()):,} truly laundering, "
          f"bands {alerts['band'].value_counts().to_dict()}")

    # feature percentiles among normal traffic
    normal = test.loc[yt == 0, PROFILE_FEATURES]
    prof = {f: np.quantile(normal[f].astype(float), np.linspace(0, 1, 101)).tolist() for f in PROFILE_FEATURES}
    (APP_DATA / "profile.json").write_text(json.dumps(prof))

    for f in ["results.json", "extra_models.json", "explainability.json", "split_info.json", "ablation_age.json"]:
        shutil.copy(REPORTS_DIR / f, APP_DATA / f)
    shutil.copy(MODELS / "model_meta.json", APP_DATA / "model_meta.json")
    print("wrote app/data/ and models/serving.json")


if __name__ == "__main__":
    main()
