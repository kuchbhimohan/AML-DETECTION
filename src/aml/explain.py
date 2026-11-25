"""Step 8 — explanations: why was this transaction flagged?

Uses TreeSHAP via LightGBM's built-in `pred_contrib` (exact Shapley values for
tree models, no extra approximation). For each transaction:

    model log-odds = base value + Σ feature contributions        (additivity)

Contributions are in log-odds of the model trained on the case-control sample.
The prior correction only shifts the BASE value by log(r); feature contributions
are unchanged, so explanations are valid on the real-world scale too.

Run:  uv run python -m aml.explain      -> reports/explainability.md + figures
"""

from __future__ import annotations

import json
from functools import lru_cache

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aml.config import PROJECT_ROOT, REPORTS_DIR, load_train_config
from aml.sampling import correct_probability

MODELS = PROJECT_ROOT / "models"
FIG = REPORTS_DIR / "figures"
CFG = load_train_config()

# Plain-English templates. {v} = the transaction's value for that feature.
REASONS = {
    "s_out_cnt_1h": "Sender made {v:.0f} payments in the previous hour",
    "s_out_cnt_24h": "Sender made {v:.0f} payments in the previous 24 hours",
    "s_out_cnt_7d": "Sender made {v:.0f} payments in the previous 7 days",
    "s_out_uniq_1h": "Sender paid {v:.0f} different accounts in the previous hour",
    "s_out_uniq_24h": "Sender paid {v:.0f} different accounts in the previous 24 hours (fan-out)",
    "s_out_uniq_7d": "Sender paid {v:.0f} different accounts in the previous 7 days (fan-out)",
    "s_out_usd_24h": "Sender sent ${v:,.0f} in the previous 24 hours",
    "s_out_usd_7d": "Sender sent ${v:,.0f} in the previous 7 days",
    "s_amt_z_7d": "Amount is {v:+.1f} standard deviations from the sender's usual amounts",
    "s_amt_ratio_7d": "Amount is {v:.1f}x the sender's average over the previous 7 days",
    "s_hours_since_prev_out": "Sender's previous payment was {v:.1f} hours earlier",
    "r_in_cnt_1h": "Receiver got {v:.0f} payments in the previous hour",
    "r_in_cnt_24h": "Receiver got {v:.0f} payments in the previous 24 hours",
    "r_in_cnt_7d": "Receiver got {v:.0f} payments in the previous 7 days",
    "r_in_uniq_1h": "Receiver was paid by {v:.0f} different accounts in the previous hour",
    "r_in_uniq_24h": "Receiver was paid by {v:.0f} different accounts in the previous 24 hours (fan-in)",
    "r_in_uniq_7d": "Receiver was paid by {v:.0f} different accounts in the previous 7 days (fan-in)",
    "r_in_usd_24h": "Receiver got ${v:,.0f} in the previous 24 hours",
    "s_in_cnt_24h": "Sender itself received {v:.0f} payments in the previous 24 hours (pass-through)",
    "s_in_uniq_24h": "Sender received money from {v:.0f} different accounts in the previous 24 hours",
    "s_in_usd_24h": "Sender received ${v:,.0f} in the previous 24 hours",
    "r_out_cnt_24h": "Receiver made {v:.0f} onward payments in the previous 24 hours",
    "r_out_uniq_24h": "Receiver paid {v:.0f} different accounts in the previous 24 hours",
    "s_passthrough_ratio_24h": "Sender is passing on {v:.1f}x what it received in the previous 24 hours",
    "pair_prev_cnt": "These two accounts had {v:.0f} earlier payments between them",
    "pair_first_time": "First-ever payment from this sender to this receiver",
    "reverse_flow_7d": "Receiver sent money back to this sender in the previous 7 days (circular flow)",
    "s_age_hours": "Sender first seen {v:.0f} hours ago",
    "r_age_hours": "Receiver first seen {v:.0f} hours ago",
    "log_amount_usd": "Amount ≈ ${amt:,.0f}",
    "round_amount": "Round amount (multiple of 100)",
    "is_ach": "Paid by ACH",
    "cross_bank": "Sender and receiver are at different banks",
    "self_transfer": "Transfer to the same account",
    "cross_currency": "Different payment and receiving currency",
    "hour": "Made at {v:.0f}:00",
    "payment_format": "Payment format: {s}",
    "payment_currency": "Currency: {s}",
}
BINARY_TRUE_ONLY = {"pair_first_time", "reverse_flow_7d", "round_amount", "is_ach", "cross_bank",
                    "self_transfer", "cross_currency"}


@lru_cache(maxsize=1)
def load_model():
    meta = json.loads((MODELS / "model_meta.json").read_text())
    booster = lgb.Booster(model_file=str(MODELS / f"{meta['champion']}.txt"))
    return booster, meta


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Select model features and apply the training category lists."""
    _, meta = load_model()
    X = df[meta["features"]].copy()
    for c, cats in meta["categories"].items():
        v = X[c].astype(str)
        X[c] = pd.Categorical(v.where(v.isin(cats)), categories=cats)
    return X


def shap_values(X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Returns (contributions per feature [log-odds], base values)."""
    booster, meta = load_model()
    contrib = booster.predict(X, pred_contrib=True, num_threads=CFG["compute"]["n_jobs"])
    return pd.DataFrame(contrib[:, :-1], columns=meta["features"], index=X.index), contrib[:, -1]


def _phrase(feat: str, value) -> str:
    t = REASONS.get(feat, feat + " = {v}")
    if feat == "log_amount_usd":
        return t.format(amt=float(np.expm1(value)))
    if feat in ("payment_format", "payment_currency"):
        return t.format(s=str(value))
    try:
        return t.format(v=float(value))
    except (ValueError, TypeError):
        return f"{feat} = {value}"


# Features that describe the SAME fact are explained together. Shapley values are
# additive, so a group's contribution is simply the sum of its members' SHAP values.
GROUPS = {"payment_method": ["payment_format", "is_ach"],
          "pair_history": ["pair_prev_cnt", "pair_first_time"]}
_MEMBER_TO_GROUP = {m: g for g, ms in GROUPS.items() for m in ms}


def _group_phrase(group: str, row: pd.Series) -> str:
    if group == "payment_method":
        return _phrase("payment_format", row["payment_format"])
    if group == "pair_history":
        n = float(row["pair_prev_cnt"])
        return ("First-ever payment from this sender to this receiver" if n == 0
                else f"These two accounts had {n:.0f} earlier payments between them")
    raise KeyError(group)


def _value(v):
    if pd.isna(v):
        return None
    return v if isinstance(v, str) else float(v)


def reason_codes(X: pd.DataFrame, top_k: int = 3) -> list[dict]:
    """Top-k DISTINCT reasons that pushed each transaction's risk UP, in plain English."""
    contrib, base = shap_values(X)
    _, meta = load_model()
    r = meta["negative_rate"]
    out = []
    for i, idx in enumerate(X.index):
        c = contrib.loc[idx]; row = X.loc[idx]
        logit = base[i] + c.sum()
        p = float(correct_probability(1 / (1 + np.exp(-logit)), r))
        # collapse grouped features into one contribution each
        items = {}
        for feat, v in c.items():
            key = _MEMBER_TO_GROUP.get(feat, feat)
            items[key] = items.get(key, 0.0) + float(v)
        ups = []
        for key, v in sorted(items.items(), key=lambda kv: -kv[1]):
            if v <= 0 or len(ups) == top_k:
                break
            if key in GROUPS:
                ups.append({"feature": key, "value": {m: _value(row[m]) for m in GROUPS[key]},
                            "contribution": v, "reason": _group_phrase(key, row)})
                continue
            if key in BINARY_TRUE_ONLY and not row[key]:
                continue  # e.g. "not cross-bank" raising risk is not a useful reason for an analyst
            ups.append({"feature": key, "value": _value(row[key]), "contribution": v, "reason": _phrase(key, row[key])})
        out.append({"probability": p, "log_odds": float(logit), "reasons": ups})
    return out


# --------------------------------------------------------------------------- #
# Global analysis + report
# --------------------------------------------------------------------------- #
def main(n_negatives: int = 15_000):
    from aml.data import load_split
    test = load_split("test_full")
    rng = np.random.default_rng(CFG["seed"])
    pos = np.flatnonzero(test["label"].to_numpy() == 1)
    neg = rng.choice(np.flatnonzero(test["label"].to_numpy() == 0), n_negatives, replace=False)
    sample = test.iloc[np.sort(np.concatenate([pos, neg]))]
    X = prepare(sample)
    contrib, base = shap_values(X)
    booster, meta = load_model()

    # additivity check: base + Σ contributions == raw model log-odds
    raw = booster.predict(X, raw_score=True, num_threads=CFG["compute"]["n_jobs"])
    add_err = float(np.max(np.abs(base + contrib.sum(axis=1).to_numpy() - raw)))

    # global importance (weight negatives back to the real ratio so it reflects all traffic)
    w = np.where(sample["label"].to_numpy() == 1, 1.0, (len(test) - len(pos)) / n_negatives)
    imp = (contrib.abs().mul(w, axis=0).sum() / w.sum()).sort_values(ascending=False)
    imp_pos = contrib[sample["label"].to_numpy() == 1].abs().mean().sort_values(ascending=False)

    # direction: does a higher feature value push risk up? (Spearman rho of value vs SHAP)
    direction = {}
    for f in meta["features"]:
        if f in ("payment_format", "payment_currency"):
            continue
        v = pd.Series(X[f].astype(float).to_numpy()); s = pd.Series(contrib[f].to_numpy())
        direction[f] = float(v.rank().corr(s.rank())) if v.nunique() > 1 else 0.0

    # agreement with logistic regression odds ratios (Step 5)
    coefs = pd.read_csv(REPORTS_DIR / "logreg_coefficients.csv", index_col=0)
    agree_rows = []
    for f, row in coefs.drop(index="const").iterrows():
        if f in direction and row["p_value"] < 0.05 and abs(direction[f]) > 0.1:
            agree_rows.append((f, np.sign(row["coef"]), np.sign(direction[f])))
    n_agree = sum(a == b for _, a, b in agree_rows)

    # monotone constraints must show up as non-negative/non-positive direction
    mono_ok = {f: (np.sign(direction[f]) in (0, s)) for f, s in meta["monotone_constraints"].items()}

    # example alerts: top-3 riskiest laundering transactions + top-ranked false alarm
    p = correct_probability(1 / (1 + np.exp(-raw)), meta["negative_rate"])
    lab = sample["label"].to_numpy()
    ex_idx = list(X.index[np.argsort(-p * (lab == 1))[:3]]) + list(X.index[np.argsort(-p * (lab == 0))[:1]])
    examples = reason_codes(X.loc[ex_idx])

    # figures
    FIG.mkdir(parents=True, exist_ok=True)
    top = imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5)); ax.barh(top.index, top.values)
    ax.set(xlabel="mean |SHAP| (log-odds), all test traffic", title="What drives the risk score?")
    ax.tick_params(axis="y", labelsize=8); fig.tight_layout(); fig.savefig(FIG / "step8_shap_importance.png", dpi=120); plt.close(fig)
    import shap
    feats15 = list(imp.head(15).index)
    Xnum = X[feats15].copy()
    for c in Xnum.columns:
        if isinstance(Xnum[c].dtype, pd.CategoricalDtype):
            Xnum[c] = Xnum[c].cat.codes
    shap.summary_plot(contrib[feats15].to_numpy(), Xnum.astype(float), show=False, max_display=15)
    plt.title("SHAP beeswarm (test sample)"); plt.tight_layout(); plt.savefig(FIG / "step8_shap_beeswarm.png", dpi=120); plt.close("all")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, f in zip(axes, [f for f in imp.index if f not in ("payment_format", "payment_currency")][:3]):
        ax.scatter(X[f].astype(float), contrib[f], s=3, alpha=0.3, c=lab, cmap="coolwarm")
        ax.set(xlabel=f, ylabel="SHAP (log-odds)", xscale="symlog")
    fig.suptitle("Dependence plots (red = laundering)"); fig.tight_layout(); fig.savefig(FIG / "step8_dependence.png", dpi=120); plt.close(fig)

    # report
    L = ["# Step 8 — Explainability (SHAP)", "",
         f"- Exact TreeSHAP (LightGBM `pred_contrib`) on a test sample: all {len(pos):,} laundering + {n_negatives:,} random normal transactions;",
         "  importance re-weighted to the real normal:laundering ratio.",
         f"- **Additivity check:** max |base + Σ SHAP − model log-odds| = {add_err:.2e} (exact).", "",
         "## Global importance (mean |SHAP|, log-odds)", "", "| Feature | All traffic | Laundering only | Direction (ρ value vs SHAP) |",
         "|---|---|---|---|",
         *[f"| `{f}` | {imp[f]:.3f} | {imp_pos[f]:.3f} | "
           f"{(format(direction[f], '+.2f') if f in direction else '— (categorical)')} |" for f in imp.head(15).index], "",
         "Note: `is_ach` and `payment_format` carry the same ACH signal, so the model splits credit between them; "
         "alert reasons merge them (and `pair_prev_cnt` + `pair_first_time`) into one reason each. "
         "The age features have near-zero direction: their effect is non-monotone (see Step 7 ablation).", "",
         "## Sanity checks", "",
         f"- **Agreement with logistic regression:** of {len(agree_rows)} features that are significant in the logistic model "
         f"(p<0.05) and have a clear SHAP direction, **{n_agree} agree in sign**." +
         ("" if n_agree == len(agree_rows) else " Disagreements: " + ", ".join(f"`{f}`" for f, a, b in agree_rows if a != b)
          + " (logistic coefficients are *conditional* on correlated features; SHAP directions are marginal, so some flips are expected)."),
         f"- **Monotone constraints visible in SHAP:** " + ", ".join(f"`{f}` {'✅' if ok else '❌'}" for f, ok in mono_ok.items()), "",
         "## Example alerts (top-3 reasons that pushed risk up)", ""]
    for k, (idx, e) in enumerate(zip(ex_idx, examples)):
        kind = "laundering" if lab[X.index.get_loc(idx)] == 1 else "**false alarm** (normal transaction)"
        L += [f"**Example {k + 1}** — {kind}, P(laundering) = {e['probability']:.1%}", ""]
        L += [f"{j + 1}. {r['reason']} (SHAP {r['contribution']:+.2f})" for j, r in enumerate(e["reasons"])] + [""]
    L += ["Figures: `figures/step8_shap_importance.png`, `figures/step8_shap_beeswarm.png`, `figures/step8_dependence.png`", ""]
    (REPORTS_DIR / "explainability.md").write_text("\n".join(L), encoding="utf-8")
    (REPORTS_DIR / "explainability.json").write_text(json.dumps(
        {"additivity_max_error": add_err, "importance_all": imp.to_dict(), "importance_laundering": imp_pos.to_dict(),
         "direction": direction, "logreg_agreement": {"n": len(agree_rows), "agree": n_agree},
         "monotone_ok": mono_ok, "examples": examples}, indent=1, default=str))
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
