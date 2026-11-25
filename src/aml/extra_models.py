"""Step 7b — additional algorithms, compared against the (fixed) champion.

Honesty rules:
- The champion was chosen on VALIDATION in Step 6 and does not change here.
- These models use fixed, sensible settings: no tuning on the test period.
- Each command is short (laptop-safe); test-period scores are cached on disk.

Commands:
  uv run python -m aml.extra_models fit --model rf      # Random Forest (bagging)
  uv run python -m aml.extra_models fit --model et      # Extra Trees
  uv run python -m aml.extra_models fit --model mlp     # neural network
  uv run python -m aml.extra_models imbalance           # none vs SMOTE vs class weights (LightGBM)
  uv run python -m aml.extra_models report              # + rank-average ensemble -> reports/extra_models.md
"""

from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from aml.baselines import LinearPrep
from aml.config import PROJECT_ROOT, REPORTS_DIR, load_train_config
from aml.data import load_split
from aml.metrics import ranking_metrics
from aml.sampling import correct_probability
from aml.stats import poisson_bootstrap

CFG = load_train_config()
SEED = CFG["seed"]; NJ = CFG["compute"]["n_jobs"]; R = CFG["sampling"]["negative_rate"]
MODELS = PROJECT_ROOT / "models"
SCORES = PROJECT_ROOT / "data" / "processed" / "extra_scores"     # git-ignored
CHAMP = "LightGBM monotone (champion)"


def _matrices(standardise: bool = False):
    """Numeric design matrix for models without native categorical support:
    log-transformed counts/amounts + one-hot payment format (no VIF pruning)."""
    feats = json.loads((MODELS / "model_meta.json").read_text())["features"]
    train = load_split("train_sample"); vs = load_split("val_sample"); test = load_split("test_full")
    prep = LinearPrep(feats)
    Xtr = prep._raw(train); cols = list(Xtr.columns)
    Xvs = prep._raw(vs).reindex(columns=cols, fill_value=0.0)
    Xte = prep._raw(test).reindex(columns=cols, fill_value=0.0)
    if standardise:
        mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
        Xtr, Xvs, Xte = [(X - mu) / sd for X in (Xtr, Xvs, Xte)]
    return (Xtr.to_numpy(np.float32), train["label"].to_numpy(), Xvs.to_numpy(np.float32), vs,
            Xte.to_numpy(np.float32), test["label"].to_numpy(), cols)


def _slug(name: str) -> str:
    """Safe file name (Windows forbids ':' — it silently creates a hidden data stream)."""
    import re
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def _save(name: str, p: np.ndarray, seconds: float, extra: dict | None = None):
    SCORES.mkdir(parents=True, exist_ok=True)
    np.save(SCORES / f"{_slug(name)}.npy", p)
    info = {"name": name, "fit_and_score_seconds": round(seconds, 1), **(extra or {})}
    (SCORES / f"{_slug(name)}.json").write_text(json.dumps(info, indent=1))
    print(f"{name}: saved test scores ({seconds:.0f}s)")


def fit(model: str):
    t0 = time.time()
    Xtr, ytr, _, _, Xte, yte, cols = _matrices(standardise=(model == "mlp"))
    if model == "rf":
        clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_features="sqrt",
                                     n_jobs=NJ, random_state=SEED)
        name, params = "Random Forest", clf.get_params()
    elif model == "et":
        clf = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=5, max_features="sqrt",
                                   n_jobs=NJ, random_state=SEED)
        name, params = "Extra Trees", clf.get_params()
    elif model == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512, learning_rate_init=1e-3,
                            max_iter=40, early_stopping=True, n_iter_no_change=5, random_state=SEED)
        name, params = "Neural network (MLP 64-32)", clf.get_params()
    else:
        raise ValueError(model)
    with threadpool_limits(NJ):
        clf.fit(Xtr, ytr)                                   # unweighted on case-control sample
        p = correct_probability(clf.predict_proba(Xte)[:, 1], R)
    keep = {k: v for k, v in params.items() if isinstance(v, (int, float, str, tuple)) and k not in ("verbose",)}
    _save(name, p, time.time() - t0, {"params": {k: str(v) for k, v in keep.items()}, "n_features": len(cols)})


def imbalance():
    """Same LightGBM settings, three ways of handling the 1:100 imbalance in the sample:
    (a) nothing, (b) SMOTE synthetic positives up to 10%, (c) class weights."""
    Xtr, ytr, Xvs, vs, Xte, yte, cols = _matrices()
    # Identical model + identical fixed budget for all variants (no early stopping: it stopped one
    # variant after 3 trees purely from metric noise, which would make the comparison unfair).
    base = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 200,
            "lambda_l2": 5.0, "feature_fraction": 0.7, "bagging_fraction": 0.9, "bagging_freq": 1,
            "seed": SEED, "num_threads": NJ, "verbosity": -1}
    N_ROUNDS = 300
    n_pos, n_neg = int(ytr.sum()), int((ytr == 0).sum())
    variants = {}
    # (a) no resampling
    variants["Imbalance: none (case-control sample)"] = (Xtr, ytr, base, lambda p: correct_probability(p, R))
    # (b) SMOTE: synthetic positives until positives = 10% of negatives
    t = time.time()
    Xs, ys = SMOTE(sampling_strategy=0.10, random_state=SEED, k_neighbors=5).fit_resample(Xtr, ytr)
    smote_secs = time.time() - t
    r_smote = R * n_pos / ys.sum()   # positives inflated too: combined prior correction
    variants["Imbalance: SMOTE to 10%"] = (Xs, ys, base, lambda p, r=r_smote: correct_probability(p, r))
    # (c) class weights: up-weight positives to balance classes
    variants["Imbalance: class weights (balanced)"] = (Xtr, ytr, {**base, "scale_pos_weight": n_neg / n_pos},
                                                      lambda p: correct_probability(p, R * n_pos / n_neg))
    for name, (X, y, params, fix) in variants.items():
        t0 = time.time()
        bst = lgb.train(params, lgb.Dataset(X, y), num_boost_round=N_ROUNDS)
        p = fix(bst.predict(Xte, num_threads=NJ))
        _save(name, p, time.time() - t0, {"train_rows": int(len(y)), "train_positives": int(y.sum()),
                                          "n_trees": N_ROUNDS,
                                          **({"smote_seconds": round(smote_secs, 1)} if "SMOTE" in name else {})})


def report():
    test = load_split("test_full", columns=["label"]); y = test["label"].to_numpy()
    cache = np.load(PROJECT_ROOT / "data" / "processed" / "scores_cache.npz", allow_pickle=True)["scores"].item()["test"]
    scores = {CHAMP: cache[CHAMP], "XGBoost monotone": cache["XGBoost monotone"]}
    infos = {}
    for f in sorted(SCORES.glob("*.npy")):
        meta_f = f.with_suffix(".json")
        info = json.loads(meta_f.read_text()) if meta_f.exists() else {}
        if "name" not in info or f.stem != _slug(info["name"]):
            continue                                  # skip outputs from older versions of this script
        scores[info["name"]] = np.load(f); infos[info["name"]] = info
    # Rank-average ensemble (no fitted weights -> nothing tuned on test)
    members = [CHAMP, "XGBoost monotone", "Random Forest"]
    if all(m in scores for m in members):
        ranks = [pd.Series(scores[m]).rank(pct=True).to_numpy() for m in members]
        scores["Ensemble: rank-average (LGBM + XGB + RF)"] = np.mean(ranks, axis=0)
    budgets = CFG["alert_budgets"]
    pm = {m: ranking_metrics(y, s, budgets) for m, s in scores.items()}
    others = [m for m in scores if m != CHAMP]
    print("bootstrap vs champion...")
    _, diffs = poisson_bootstrap(y, scores, budgets, n_resamples=500, seed=SEED,
                                 diff_pairs=[(m, CHAMP) for m in others])
    order = sorted(scores, key=lambda m: -pm[m]["recall_at_1pct"])
    L = ["# Step 7b — Other algorithms vs the champion (TEST period)", "",
         "The champion (monotone LightGBM) was fixed on validation in Step 6. The models below use fixed, standard",
         "settings (no tuning on test) and are reported for comparison only. Δ = model − champion, 95% CI from a",
         "paired Poisson bootstrap (500 resamples); a CI that includes 0 means no significant difference.", "",
         "| Model | PR-AUC | Recall @0.5% | Recall @1% | Recall @2% | Δ recall @1% vs champion [95% CI] |",
         "|---|---|---|---|---|---|"]
    for m in order:
        x = pm[m]
        if m == CHAMP:
            d = "—"
        else:
            dd, lo, hi = diffs[f"{m} - {CHAMP}"]["recall_at_1pct"]; d = f"{dd:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        L.append(f"| {m} | {x['pr_auc']:.3f} | {x['recall_at_0_5pct']:.3f} | {x['recall_at_1pct']:.3f} | "
                 f"{x['recall_at_2pct']:.3f} | {d} |")
    L += ["", "## Settings and cost", ""]
    for m, info in infos.items():
        extra = {k: v for k, v in info.items() if k not in ("params", "name")}
        L.append(f"- **{m}**: {json.dumps(extra)}")
    L += ["", "Regenerate: run the `fit`/`imbalance` commands, then `uv run python -m aml.extra_models report`.", ""]
    (REPORTS_DIR / "extra_models.md").write_text("\n".join(L), encoding="utf-8")
    (REPORTS_DIR / "extra_models.json").write_text(json.dumps(
        {"metrics": pm, "diff_vs_champion": diffs, "settings": infos}, indent=1, default=float))
    start = L.index("|---|---|---|---|---|---|") - 1
    print("\n".join(L[start:start + 2 + len(order)]))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit"); f.add_argument("--model", choices=["rf", "et", "mlp"], required=True)
    sub.add_parser("imbalance"); sub.add_parser("report")
    a = ap.parse_args()
    {"fit": lambda: fit(a.model), "imbalance": imbalance, "report": report}[a.cmd]()


if __name__ == "__main__":
    main()
