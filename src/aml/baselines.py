"""Step 5 — baseline models, tracked in MLflow.

1. Rules baseline       : what a simple rules engine would do (ACH first, then by amount).
2. Logistic regression  : statsmodels, interpretable, odds ratios with 95% CIs.
3. Isolation Forest     : unsupervised ("what if we had no labels?").

All are trained on train_sample and evaluated on the FULL validation period
(the test period stays untouched until Step 7).

Run:  uv run python -m aml.baselines
"""

from __future__ import annotations

import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor

from aml.config import REPORTS_DIR, load_train_config
from aml.data import align_categories, feature_list, load_split
from aml.metrics import gains_curve, ranking_metrics
from aml.sampling import correct_probability
from aml.tracking import setup_mlflow

FIG = REPORTS_DIR / "figures"
BINARY = ["pair_first_time", "reverse_flow_7d", "round_amount", "is_ach", "cross_bank",
          "self_transfer", "cross_currency"]


# --------------------------------------------------------------------------- #
# Feature preparation for the linear / tree-free models
# --------------------------------------------------------------------------- #
def vif_fast(Z: pd.DataFrame) -> pd.Series:
    """Variance inflation factors, VIF_i = 1 / (1 - R²_i), where R²_i comes from
    regressing column i on all the others. Computed from the Gram matrix with one
    small least-squares solve per column (p×p), instead of p regressions on n rows.
    Equivalent to statsmodels' variance_inflation_factor; perfectly collinear
    columns get VIF = inf."""
    X = Z.to_numpy(dtype=np.float64)
    G = X.T @ X
    out = np.empty(X.shape[1])
    for i in range(X.shape[1]):
        rest = np.delete(np.arange(X.shape[1]), i)
        beta, *_ = np.linalg.lstsq(G[np.ix_(rest, rest)], G[rest, i], rcond=None)
        ss_res = G[i, i] - G[i, rest] @ beta          # residual sum of squares
        r2 = 1 - ss_res / G[i, i]
        out[i] = np.inf if r2 >= 1 - 1e-12 else 1 / (1 - r2)
    return pd.Series(out, index=Z.columns)


class LinearPrep:
    """log1p for heavy-tailed counts/amounts, cyclic hour, one-hot payment format,
    standardisation (fitted on train only)."""

    def __init__(self, features: list[str]):
        self.features = features

    def _raw(self, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        for c in self.features:
            if c in BINARY:
                X[c] = df[c].astype(float)
            elif c == "s_hours_since_prev_out":
                X["s_no_prev_out"] = (df[c] < 0).astype(float)
                X[c] = np.log1p(df[c].clip(lower=0))
            elif c == "s_amt_z_7d":
                X[c] = df[c].clip(-10, 10)
            elif c == "hour":
                X["hour_sin"] = np.sin(2 * np.pi * df[c] / 24); X["hour_cos"] = np.cos(2 * np.pi * df[c] / 24)
            elif c == "payment_format":
                d = pd.get_dummies(df[c].astype(str), prefix="fmt", dtype=float)
                X = pd.concat([X, d], axis=1)
            elif c in ("payment_currency", "weekday"):
                continue  # 15-level currency left to the tree models; weekday excluded
            elif c == "log_amount_usd":
                X[c] = df[c]
            else:
                X[c] = np.log1p(df[c].clip(lower=0))
        return X.astype(float)

    def fit(self, df: pd.DataFrame, y: np.ndarray):
        X = self._raw(df)
        self.dropped = {}
        # 1) separation: binary/dummy columns whose "1" level never/always co-occurs with laundering
        for c in X.columns:
            vals = np.unique(X[c])
            if len(vals) <= 2:
                pos_in = y[X[c].to_numpy() == 1].sum(); n_in = (X[c] == 1).sum()
                if n_in == 0 or pos_in == 0 or pos_in == n_in:
                    self.dropped[c] = "separation/constant (no laundering in level 1)"
        if "fmt_Cheque" in X.columns and "fmt_Cheque" not in self.dropped:
            self.dropped["fmt_Cheque"] = "reference category"
        X = X.drop(columns=list(self.dropped))
        self.mean, self.std = X.mean(), X.std().replace(0, 1)
        Z = (X - self.mean) / self.std
        # 2) multicollinearity: iteratively drop the highest-VIF column until all VIF <= 10
        sub = Z.sample(min(60_000, len(Z)), random_state=0)
        while True:
            vif = vif_fast(sub)
            if vif.max() <= 10:
                break
            worst = vif.idxmax(); self.dropped[worst] = f"VIF {vif.max():.1f} > 10"
            sub = sub.drop(columns=worst)
        self.columns = list(sub.columns); self.vif = vif
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._raw(df).reindex(columns=list(self.mean.index), fill_value=0.0)
        return ((X - self.mean) / self.std)[self.columns]


# --------------------------------------------------------------------------- #
def _plot_curves(results: dict, y_val: np.ndarray, path_pr, path_gain):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, s in results.items():
        p, r, _ = precision_recall_curve(y_val, s)
        ax.plot(r, p, label=name)
    ax.axhline(y_val.mean(), ls="--", c="grey", label="random")
    ax.set(xlabel="recall", ylabel="precision", title="Precision–recall (validation)", yscale="log")
    ax.legend(); fig.tight_layout(); fig.savefig(path_pr, dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, s in results.items():
        x, g = gains_curve(y_val, s, 400); ax.plot(x, g, label=name)
    ax.plot([0, 1], [0, 1], ls="--", c="grey", label="random")
    for b in (0.01, 0.02):
        ax.axvline(b, c="k", lw=0.5, ls=":")
    ax.set(xscale="log", xlim=(1e-4, 1), xlabel="fraction of transactions reviewed (log)",
           ylabel="fraction of laundering caught", title="Cumulative gains (validation)")
    ax.legend(); fig.tight_layout(); fig.savefig(path_gain, dpi=120); plt.close(fig)


def _plot_odds(coefs: pd.DataFrame, path):
    c = coefs.drop(index="const").sort_values("odds_ratio")
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(c) + 1))
    ax.errorbar(c["odds_ratio"], range(len(c)), xerr=[c["odds_ratio"] - c["or_ci_low"], c["or_ci_high"] - c["odds_ratio"]],
                fmt="o", ms=4, capsize=2)
    ax.axvline(1, c="grey", ls="--"); ax.set_xscale("log")
    ax.set_yticks(range(len(c))); ax.set_yticklabels(c.index, fontsize=8)
    ax.set(xlabel="odds ratio per 1 SD (log scale, 95% CI)", title="Logistic regression: what raises laundering odds?")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    cfg = load_train_config(); r = cfg["sampling"]["negative_rate"]; budgets = cfg["alert_budgets"]
    feats = [f for f in feature_list() if f not in cfg["excluded_features"]]
    FIG.mkdir(parents=True, exist_ok=True)
    setup_mlflow()

    train = load_split("train_sample"); val = load_split("val_full")
    val, = align_categories([val], reference=train)
    y_tr = train["label"].to_numpy(); w_tr = train["weight"].to_numpy(); y_val = val["label"].to_numpy()
    print(f"train_sample {len(train):,} rows ({y_tr.sum():,} laundering) | val_full {len(val):,} rows ({y_val.sum():,})")

    scores, all_metrics = {}, {}

    # ---- 1. rules baseline -------------------------------------------------
    with mlflow.start_run(run_name="baseline_rules"):
        mlflow.set_tags({"stage": "baseline", "model_family": "rules"})
        s = val["is_ach"].to_numpy() * 100 + val["log_amount_usd"].to_numpy()
        m = ranking_metrics(y_val, s, budgets); mlflow.log_metrics(m)
        mlflow.log_param("rule", "ACH first, then larger USD amount")
        scores["rules (ACH, amount)"] = s; all_metrics["rules (ACH, amount)"] = m

    # ---- 2. logistic regression -------------------------------------------
    with mlflow.start_run(run_name="baseline_logreg"):
        mlflow.set_tags({"stage": "baseline", "model_family": "logistic_regression"})
        t0 = time.time()
        prep = LinearPrep(feats).fit(train, y_tr)
        Xtr = sm.add_constant(prep.transform(train))
        # Case-control design: fit UNWEIGHTED. Slopes (odds ratios) are unbiased under
        # outcome-based sampling; only the intercept shifts by -log(r) and is corrected.
        res = sm.Logit(y_tr, Xtr).fit(disp=0, maxiter=300)
        ci = res.conf_int()
        coefs = pd.DataFrame({"coef": res.params, "p_value": res.pvalues,
                              "odds_ratio": np.exp(res.params), "or_ci_low": np.exp(ci[0]), "or_ci_high": np.exp(ci[1])})
        Xval = sm.add_constant(prep.transform(val), has_constant="add")
        p_sample = res.predict(Xval)
        p_val = correct_probability(p_sample, r)
        m = ranking_metrics(y_val, p_val, budgets)
        m["mean_pred_prob"] = float(p_val.mean()); m["fit_seconds"] = time.time() - t0
        m["pseudo_r2"] = float(res.prsquared); m["n_features_used"] = len(prep.columns)
        mlflow.log_metrics(m)
        mlflow.log_params({"negative_rate": r, "vif_threshold": 10, "n_dropped": len(prep.dropped)})
        coefs.to_csv(REPORTS_DIR / "logreg_coefficients.csv")
        _plot_odds(coefs, FIG / "step5_logreg_odds_ratios.png")
        mlflow.log_artifact(str(REPORTS_DIR / "logreg_coefficients.csv"))
        mlflow.log_artifact(str(FIG / "step5_logreg_odds_ratios.png"))
        mlflow.log_dict(prep.dropped, "dropped_features.json")
        scores["logistic regression"] = p_val; all_metrics["logistic regression"] = m

    # ---- 3. isolation forest ----------------------------------------------
    with mlflow.start_run(run_name="baseline_isolation_forest"):
        mlflow.set_tags({"stage": "baseline", "model_family": "isolation_forest"})
        params = {"n_estimators": 100, "max_samples": 512, "random_state": cfg["seed"], "n_jobs": cfg["compute"]["n_jobs"]}
        t0 = time.time()
        Xtr_if = prep.transform(train); Xval_if = prep.transform(val)
        iso = IsolationForest(**params).fit(Xtr_if, sample_weight=w_tr)   # labels NOT used
        s = -iso.score_samples(Xval_if)
        m = ranking_metrics(y_val, s, budgets); m["fit_seconds"] = time.time() - t0
        mlflow.log_params(params); mlflow.log_metrics(m)
        scores["isolation forest (no labels)"] = s; all_metrics["isolation forest (no labels)"] = m

    # ---- comparison plots + report ----------------------------------------
    _plot_curves(scores, y_val, FIG / "step5_pr_curves.png", FIG / "step5_gains.png")
    with mlflow.start_run(run_name="baseline_comparison"):
        mlflow.set_tags({"stage": "baseline", "model_family": "comparison"})
        mlflow.log_artifact(str(FIG / "step5_pr_curves.png")); mlflow.log_artifact(str(FIG / "step5_gains.png"))

    tbl = pd.DataFrame(all_metrics).T
    cols = ["pr_auc", "roc_auc", "gini", "recall_at_0_5pct", "recall_at_1pct", "recall_at_2pct", "precision_at_1pct"]
    lines = ["# Step 5 — Baselines (validation period, full data)", "",
             f"Validation: {len(val):,} transactions, {int(y_val.sum()):,} laundering (base rate {y_val.mean():.4%}).",
             "Trained on train_sample (all laundering + 8% of normal). Test period untouched.", "",
             tbl[cols].to_markdown(floatfmt=".4f"), "",
             "## Logistic regression", "",
             f"- Features used after separation + VIF pruning: {len(prep.columns)}; pseudo-R² = {res.prsquared:.3f}",
             f"- Mean corrected predicted probability on validation: {all_metrics['logistic regression']['mean_pred_prob']:.4%} "
             f"vs actual {y_val.mean():.4%}. The train period's rate is lower "
             f"({train['label'].mul(train['weight']).sum() / train['weight'].sum():.4%}), so the gap is a base-rate "
             "shift between periods, not a modelling error (examined again in Step 7: drift + calibration).", "",
             "### Dropped features", "", *[f"- `{k}`: {v}" for k, v in prep.dropped.items()], "",
             "### Odds ratios per 1 SD (sorted by |coef|)", "",
             coefs.reindex(coefs["coef"].abs().sort_values(ascending=False).index)
                  .round(4).to_markdown(), ""]
    (REPORTS_DIR / "baselines.md").write_text("\n".join(lines), encoding="utf-8")
    print(tbl[cols].round(4).to_string())
    print("\nwrote reports/baselines.md")


if __name__ == "__main__":
    main()
