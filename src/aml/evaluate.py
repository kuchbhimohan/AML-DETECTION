"""Step 7 — final evaluation on the UNTOUCHED test period (Sep 9–10).

Produces reports/results.md + reports/results.json. These are the ONLY source of
numbers quoted on the resume / README. Deterministic: same inputs -> same numbers.

Run:  uv run python -m aml.evaluate
"""

from __future__ import annotations

import json
import time

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, precision_recall_curve

from aml.baselines import LinearPrep
from aml.config import PROJECT_ROOT, REPORTS_DIR, load_train_config
from aml.data import CATEGORICAL, align_categories, load_split
from aml.metrics import _top_k_mask, gains_curve, ranking_metrics
from aml.sampling import correct_probability
from aml.stats import delong_test, mcnemar_test, poisson_bootstrap, psi, wilson_interval
from aml.tracking import setup_mlflow

CFG = load_train_config()
MODELS = PROJECT_ROOT / "models"
FIG = REPORTS_DIR / "figures"
R = CFG["sampling"]["negative_rate"]; BUDGETS = CFG["alert_budgets"]
CHAMP = "LightGBM monotone (champion)"


def _prep(frames, meta):
    out = []
    for f in frames:
        f = f.copy()
        for c in CATEGORICAL:
            v = f[c].astype(str)
            f[c] = pd.Categorical(v.where(v.isin(meta["categories"][c])), categories=meta["categories"][c])
        out.append(f)
    return out


CACHE = PROJECT_ROOT / "data" / "processed" / "scores_cache.npz"   # git-ignored


def _model_fingerprint() -> str:
    """Changes whenever any model file changes -> cache is invalidated automatically."""
    import hashlib
    h = hashlib.md5()
    for f in sorted(MODELS.glob("*")):
        h.update(f.name.encode()); h.update(f.read_bytes())
    return h.hexdigest()


def score_all(train, val, test, tail, meta):
    """Scores needed for the report. LightGBM scoring of ~1M rows takes minutes on
    2 cores, so boosted-model scores are cached on disk (keyed by model files)."""
    fp = _model_fingerprint()
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        if str(z["fingerprint"]) == fp:
            print("using cached model scores")
            cached = z["scores"].item()
            return _add_cheap_scores(cached, train, test)
    feats = meta["features"]
    lg_m = lgb.Booster(model_file=str(MODELS / "lgbm_monotone.txt"))
    lg_u = lgb.Booster(model_file=str(MODELS / "lgbm_unconstrained.txt"))
    xg = xgb.Booster(); xg.load_model(str(MODELS / "xgb_monotone.json")); xg.set_param({"nthread": CFG["compute"]["n_jobs"]})
    it = meta["xgb_best_iteration"]; nt = CFG["compute"]["n_jobs"]

    def champ(df):
        return correct_probability(lg_m.predict(df[feats], num_threads=nt), R)

    t0 = time.time()
    out = {"val": {CHAMP: champ(val)}, "train": {CHAMP: champ(train)}, "tail": {CHAMP: champ(tail)},
           "test": {CHAMP: champ(test),
                    "LightGBM unconstrained": correct_probability(lg_u.predict(test[feats], num_threads=nt), R),
                    "XGBoost monotone": correct_probability(
                        xg.predict(xgb.DMatrix(test[feats], enable_categorical=True), iteration_range=(0, it + 1)), R)}}
    print(f"scored boosted models in {time.time() - t0:.0f}s (cached for next time)")
    np.savez_compressed(CACHE, fingerprint=fp, scores=np.array(out, dtype=object))
    return _add_cheap_scores(out, train, test)


def _add_cheap_scores(out, train, test):
    """Logistic regression (refit exactly as in Step 5, deterministic) and rules: fast."""
    prep = LinearPrep(list(json.loads((MODELS / "model_meta.json").read_text())["features"])).fit(
        train, train["label"].to_numpy())
    logit = sm.Logit(train["label"].to_numpy(), sm.add_constant(prep.transform(train))).fit(disp=0, maxiter=300)
    out["test"]["Logistic regression"] = correct_probability(
        logit.predict(sm.add_constant(prep.transform(test), has_constant="add")), R)
    out["test"]["Rules (ACH, amount)"] = test["is_ach"].to_numpy() * 100 + test["log_amount_usd"].to_numpy()
    return out


_T = [time.time()]


def _tick(label):
    now = time.time(); print(f"  [{now - _T[0]:6.1f}s] {label}"); _T[0] = now


def main():
    t_start = time.time()
    meta = json.loads((MODELS / "model_meta.json").read_text())
    train = load_split("train_sample"); val = load_split("val_full"); test = load_split("test_full"); tail = load_split("tail")
    train, val, test, tail = _prep([train, val, test, tail], meta)
    yv, yt = val["label"].to_numpy(), test["label"].to_numpy()
    print(f"test: {len(test):,} txns, {yt.sum():,} laundering | scoring models...")
    S = score_all(train, val, test, tail, meta)
    _tick("scores ready")
    st, sv = S["test"], S["val"]
    R_ = {"meta": {"test_rows": int(len(test)), "test_laundering": int(yt.sum()),
                   "test_period": f"{test['timestamp'].min()} → {test['timestamp'].max()}",
                   "bootstrap_resamples": CFG["bootstrap"]["n_resamples"]}}

    # 1) point metrics, all models
    R_["test_metrics"] = {m: ranking_metrics(yt, s, BUDGETS) for m, s in st.items()}

    _tick("point metrics")
    # 2) bootstrap CIs + paired differences
    print("bootstrap...")
    boot_models = [CHAMP, "LightGBM unconstrained", "Logistic regression"]
    ci, diffs = poisson_bootstrap(yt, {m: st[m] for m in boot_models}, BUDGETS,
                                  n_resamples=CFG["bootstrap"]["n_resamples"], ci=CFG["bootstrap"]["ci"],
                                  seed=CFG["seed"], diff_pairs=[(CHAMP, "Logistic regression"),
                                                                (CHAMP, "LightGBM unconstrained")])
    R_["bootstrap_ci"] = ci; R_["bootstrap_diff"] = diffs

    _tick("bootstrap")
    # 3) significance tests
    R_["delong"] = {"champion_vs_logreg": delong_test(yt, st[CHAMP], st["Logistic regression"]),
                    "champion_vs_unconstrained": delong_test(yt, st[CHAMP], st["LightGBM unconstrained"])}
    caught_c = _top_k_mask(st[CHAMP], 0.01)[yt == 1]
    caught_l = _top_k_mask(st["Logistic regression"], 0.01)[yt == 1]
    R_["mcnemar_1pct_on_laundering"] = mcnemar_test(caught_c, caught_l)

    _tick("DeLong + McNemar")
    # 4) calibration: isotonic fitted on VALIDATION, applied to test
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(sv[CHAMP], yv)
    p_raw = st[CHAMP]; p_cal = iso.predict(p_raw)
    R_["calibration"] = {"brier_raw": float(brier_score_loss(yt, p_raw)), "brier_isotonic": float(brier_score_loss(yt, p_cal)),
                         "brier_constant_baseline": float(brier_score_loss(yt, np.full(len(yt), yv.mean()))),
                         "mean_pred_raw": float(p_raw.mean()), "mean_pred_isotonic": float(p_cal.mean()),
                         "actual_rate": float(yt.mean())}

    # 5) operating points
    #   a) best-F1 threshold chosen on validation (calibrated) -> applied to test
    pv = iso.predict(sv[CHAMP]); prec, rec, thr = precision_recall_curve(yv, pv)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12); best = int(np.nanargmax(f1[:-1]))
    th = float(thr[best]); alert = p_cal >= th
    tp = int((alert & (yt == 1)).sum()); fp = int((alert & (yt == 0)).sum()); fn = int((~alert & (yt == 1)).sum())
    R_["f1_operating_point"] = {"threshold_from_val": th, "val_f1": float(f1[best]),
                                "test_precision": tp / max(tp + fp, 1), "test_recall": tp / max(tp + fn, 1),
                                "test_f1": 2 * tp / max(2 * tp + fp + fn, 1), "alerts": int(alert.sum()),
                                "alert_rate": float(alert.mean())}
    #   b) cost-based: alert if P(laundering) * cost_missed > cost_review
    R_["cost_operating_points"] = {}
    for ratio in (50, 100, 300, 1000):
        a = p_cal > 1.0 / ratio
        tp = int((a & (yt == 1)).sum())
        R_["cost_operating_points"][f"missed_case_costs_{ratio}x_a_review"] = {
            "threshold": 1.0 / ratio, "alert_rate": float(a.mean()), "alerts": int(a.sum()),
            "recall": tp / yt.sum(), "precision": tp / max(a.sum(), 1)}

    _tick("calibration + operating points")
    # 6) drift: PSI train(weighted sample) -> test, per feature and for the score
    w_tr = train["weight"].to_numpy()
    num_feats = [f for f in meta["features"] if f not in CATEGORICAL]
    psi_feat = {f: psi(train[f].to_numpy(), test[f].to_numpy(), w_expected=w_tr) for f in num_feats}
    R_["drift"] = {"score_psi_train_to_test": psi(S["train"][CHAMP], p_raw, w_expected=w_tr),
                   "score_psi_val_to_test": psi(sv[CHAMP], p_raw),
                   "feature_psi_top10": dict(sorted(psi_feat.items(), key=lambda kv: -kv[1])[:10]),
                   "n_features_psi_gt_0_25": int(sum(v > 0.25 for v in psi_feat.values())),
                   "n_features_psi_0_1_to_0_25": int(sum(0.1 < v <= 0.25 for v in psi_feat.values())),
                   "base_rate_train": float((train["label"] * w_tr).sum() / w_tr.sum()),
                   "base_rate_val": float(yv.mean()), "base_rate_test": float(yt.mean())}

    _tick("drift")
    # 7) per-pattern recall at 1% budget (Wilson CIs)
    pat = test["pattern"].to_numpy()[yt == 1]
    R_["pattern_recall_1pct"] = {}
    for p in sorted(pd.unique(pat), key=lambda x: -int((pat == x).sum())):
        idx = pat == p; n = int(idx.sum())
        kc = int(caught_c[idx].sum()); kl = int(caught_l[idx].sum())
        R_["pattern_recall_1pct"][p] = {"n": n, "champion": kc / n, "champion_ci": wilson_interval(kc, n),
                                        "logreg": kl / n}

    # 8) tail days (out-of-distribution sanity check)
    ytail = tail["label"].to_numpy()
    R_["tail"] = {"rows": int(len(tail)), "laundering": int(ytail.sum()),
                  **{k: v for k, v in ranking_metrics(ytail, S["tail"][CHAMP], ()).items() if k in ("roc_auc", "pr_auc", "base_rate")}}
    R_["meta"]["runtime_seconds"] = round(time.time() - t_start, 1)

    _tick("patterns + tail")
    _figures(yt, st, p_raw, p_cal, R_)
    _write(R_)
    setup_mlflow()
    with mlflow.start_run(run_name="final_evaluation_test"):
        mlflow.set_tags({"stage": "test_evaluation"})
        mlflow.log_metrics({f"test_{k}": v for k, v in R_["test_metrics"][CHAMP].items()})
        mlflow.log_artifact(str(REPORTS_DIR / "results.md")); mlflow.log_artifact(str(REPORTS_DIR / "results.json"))
        for f in FIG.glob("step7_*.png"):
            mlflow.log_artifact(str(f))
    print(f"done in {time.time() - t_start:.0f}s -> reports/results.md")


# --------------------------------------------------------------------------- #
def _figures(yt, st, p_raw, p_cal, R_):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for m in [CHAMP, "Logistic regression", "Rules (ACH, amount)"]:
        x, g = gains_curve(yt, st[m], 500); ax.plot(x, g, label=m)
    ax.plot([0, 1], [0, 1], "--", c="grey", label="random")
    for b in BUDGETS:
        ax.axvline(b, c="k", lw=0.5, ls=":")
    ax.set(xscale="log", xlim=(1e-4, 1), xlabel="fraction of transactions reviewed (log)",
           ylabel="fraction of laundering caught", title="Cumulative gains — TEST period")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "step7_gains_test.png", dpi=120); plt.close(fig)

    edges = np.array([0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0001])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for p, lab in [(p_raw, "raw (prior-corrected)"), (p_cal, "isotonic (fit on val)")]:
        b = np.digitize(p, edges) - 1; xs, ys = [], []
        for i in np.unique(b):
            m = b == i
            if m.sum() >= 30:
                xs.append(p[m].mean()); ys.append(max(yt[m].mean(), 1e-5))
        ax.plot(xs, ys, "o-", label=lab)
    ax.plot([1e-5, 1], [1e-5, 1], "--", c="grey", label="perfect")
    ax.set(xscale="log", yscale="log", xlabel="predicted probability", ylabel="observed laundering rate",
           title="Reliability — TEST period"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "step7_reliability.png", dpi=120); plt.close(fig)

    pr = R_["pattern_recall_1pct"]; names = list(pr)
    fig, ax = plt.subplots(figsize=(7, 4))
    y0 = np.arange(len(names)); c = [pr[n]["champion"] for n in names]
    err = np.array([[pr[n]["champion"] - pr[n]["champion_ci"][0], pr[n]["champion_ci"][1] - pr[n]["champion"]] for n in names]).T
    ax.barh(y0 + 0.2, c, 0.4, xerr=err, capsize=2, label="LightGBM (champion)")
    ax.barh(y0 - 0.2, [pr[n]["logreg"] for n in names], 0.4, label="Logistic regression")
    ax.set_yticks(y0); ax.set_yticklabels([f"{n} (n={pr[n]['n']})" for n in names], fontsize=8)
    ax.set(xlabel="recall at 1% alert budget", xlim=(0, 1), title="Which laundering patterns are caught? — TEST")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "step7_pattern_recall.png", dpi=120); plt.close(fig)

    d = R_["drift"]["feature_psi_top10"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(list(d)[::-1], list(d.values())[::-1])
    for x, c in [(0.1, "orange"), (0.25, "red")]:
        ax.axvline(x, c=c, ls="--", lw=1)
    ax.set(xlabel="PSI (train → test)", title="Top-10 feature drift"); ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout(); fig.savefig(FIG / "step7_psi.png", dpi=120); plt.close(fig)


def _pct(x):
    return f"{100 * x:.1f}%"


def _rate(x):
    return f"{100 * x:.3f}%"


def _write(R_):
    (REPORTS_DIR / "results.json").write_text(json.dumps(R_, indent=1, default=float), encoding="utf-8")
    m = R_["meta"]; ci = R_["bootstrap_ci"]; tm = R_["test_metrics"]
    L = ["# Final results — TEST period (untouched until now)", "",
         f"- Test period: {m['test_period']} · {m['test_rows']:,} transactions · {m['test_laundering']:,} laundering",
         f"- Models trained on Sep 1–6, tuned on Sep 7–8. 95% CIs from a paired Poisson bootstrap ({m['bootstrap_resamples']} resamples).",
         "- Regenerate with `uv run python -m aml.evaluate` (deterministic).", "",
         "## 1. Headline (alert budget = share of transactions investigators review)", "",
         "| Model | PR-AUC | ROC-AUC | Recall @0.5% | Recall @1% | Recall @2% | Precision @1% |",
         "|---|---|---|---|---|---|---|"]
    for name, x in tm.items():
        def c(k):
            if name in ci:
                p, lo, hi = ci[name][k]; return f"{p:.3f} [{lo:.3f}–{hi:.3f}]"
            return f"{x[k]:.3f}"
        L.append(f"| {name} | {c('pr_auc')} | {c('roc_auc')} | {c('recall_at_0_5pct')} | {c('recall_at_1pct')} | "
                 f"{c('recall_at_2pct')} | {x['precision_at_1pct']:.3f} |")
    L += ["", "## 2. Is the champion significantly better?", ""]
    for pair, dd in R_["bootstrap_diff"].items():
        d, lo, hi = dd["pr_auc"]; d2, lo2, hi2 = dd["recall_at_1pct"]
        L.append(f"- **{pair}**: ΔPR-AUC = {d:+.3f} [{lo:+.3f}, {hi:+.3f}], Δrecall@1% = {d2:+.3f} [{lo2:+.3f}, {hi2:+.3f}]")
    dl = R_["delong"]
    L += [f"- DeLong (ROC-AUC) champion vs logistic: Δ = {dl['champion_vs_logreg']['diff']:+.4f}, z = {dl['champion_vs_logreg']['z']:.2f}, "
          f"p = {dl['champion_vs_logreg']['p_value']:.2e}",
          f"- DeLong champion vs unconstrained: Δ = {dl['champion_vs_unconstrained']['diff']:+.4f}, p = {dl['champion_vs_unconstrained']['p_value']:.2e}"]
    mc = R_["mcnemar_1pct_on_laundering"]
    L += [f"- McNemar at 1% budget (laundering cases): caught only by champion = {mc['a_only']}, only by logistic = {mc['b_only']}, "
          f"both = {mc['both']}, neither = {mc['neither']}; exact p = {mc['p_value']:.2e}", ""]
    cal = R_["calibration"]
    L += ["## 3. Calibration (champion)", "",
          f"- Brier: raw {cal['brier_raw']:.6f} · isotonic (fit on validation) {cal['brier_isotonic']:.6f} · "
          f"constant-rate baseline {cal['brier_constant_baseline']:.6f}",
          f"- Mean predicted: raw {_rate(cal['mean_pred_raw'])}, isotonic {_rate(cal['mean_pred_isotonic'])}; actual {_rate(cal['actual_rate'])}",
          "- Figure: `figures/step7_reliability.png`", ""]
    f = R_["f1_operating_point"]
    L += ["## 4. Operating points", "",
          f"- **Best-F1 threshold (chosen on validation)** → test: precision {_pct(f['test_precision'])}, recall {_pct(f['test_recall'])}, "
          f"F1 {f['test_f1']:.3f}, {f['alerts']:,} alerts ({_pct(f['alert_rate'])} of transactions)", "",
          "",
          "Cost-based rule (calibrated probabilities): alert when P(laundering) × cost of a missed case > cost of a review.",
          "Isotonic calibration is a step function, so nearby thresholds can give identical alert sets.", "",
          "| Cost of a missed case vs one review | Threshold | Alert rate | Recall | Precision |", "|---|---|---|---|---|"]
    for k, v in R_["cost_operating_points"].items():
        ratio = k.split("_")[3]
        L.append(f"| {ratio} | {v['threshold']:.4f} | {_pct(v['alert_rate'])} | {_pct(v['recall'])} | {_pct(v['precision'])} |")
    dr = R_["drift"]
    L += ["", "## 5. Drift (PSI; <0.1 stable, 0.1–0.25 moderate, >0.25 major)", "",
          f"- Score PSI train→test {dr['score_psi_train_to_test']:.3f}; val→test {dr['score_psi_val_to_test']:.3f}",
          f"- Features with PSI > 0.25: {dr['n_features_psi_gt_0_25']}; 0.1–0.25: {dr['n_features_psi_0_1_to_0_25']}",
          f"- Base rate: train {_rate(dr['base_rate_train'])} → val {_rate(dr['base_rate_val'])} → test {_rate(dr['base_rate_test'])}",
          "- Why so much feature drift: the data starts on Sep 1, so account age and long-window/all-history features "
          "(`*_age_hours`, `pair_prev_cnt`, `*_7d`) keep growing during the first week ('history warm-up'). "
          "The model still generalises (see §1), and §8 shows what happens without the age features.", "",
          "| Feature | PSI |", "|---|---|", *[f"| `{k}` | {v:.3f} |" for k, v in dr["feature_psi_top10"].items()], "",
          "## 6. Recall by laundering pattern (1% budget, Wilson 95% CI)", "",
          "| Pattern | n | Champion | 95% CI | Logistic |", "|---|---|---|---|---|"]
    for p, v in R_["pattern_recall_1pct"].items():
        L.append(f"| {p} | {v['n']} | {_pct(v['champion'])} | {_pct(v['champion_ci'][0])}–{_pct(v['champion_ci'][1])} | {_pct(v['logreg'])} |")
    t = R_["tail"]
    L += ["", "## 7. Sparse tail days (Sep 11–18, excluded from the main test)", "",
          f"- {t['rows']:,} transactions, {t['laundering']:,} laundering ({_pct(t['base_rate'])}); champion ROC-AUC {t['roc_auc']:.3f}, PR-AUC {t['pr_auc']:.3f}",
          "- Simulator artefact (laundering chains finishing after normal traffic stops); reported only as a sanity check.", "",
          ]
    ab_path = REPORTS_DIR / "ablation_age.json"
    if ab_path.exists():
        ab = json.loads(ab_path.read_text())
        L += ["## 8. Ablation: are the account-age features doing the work? (validation)", "",
              "| | PR-AUC | Recall @1% |", "|---|---|---|",
              f"| with age features (champion) | {ab['with']['pr_auc']:.3f} | {_pct(ab['with']['recall_at_1pct'])} |",
              f"| without `s_age_hours`, `r_age_hours` | {ab['without']['pr_auc']:.3f} | {_pct(ab['without']['recall_at_1pct'])} |", "",
              "Age partly reflects 'time since the data starts' (only 10 days observed), so it is a known limitation. "
              "Removing it lowers PR-AUC but leaves recall at the 1% alert budget almost unchanged, "
              "so the headline business result does not rely on it.", ""]
    L += [f"_Runtime: {m['runtime_seconds']}s._", ""]
    (REPORTS_DIR / "results.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
