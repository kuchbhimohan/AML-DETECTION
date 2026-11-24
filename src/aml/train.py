"""Step 6 — gradient boosting: LightGBM (with monotonic constraints) + XGBoost,
tuned with Optuna, every trial tracked in MLflow, best model registered.

Laptop-safe: 2 cores, each tuning batch stops after `timeout_seconds`.
Studies are stored in optuna.db, so several short batches add up.

Usage:
  uv run python -m aml.train tune --model lgbm --trials 10
  uv run python -m aml.train tune --model xgb  --trials 10
  uv run python -m aml.train final
"""

from __future__ import annotations

import argparse
import json
import time
from functools import lru_cache

import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

from aml.config import PROJECT_ROOT, REPORTS_DIR, load_train_config
from aml.data import CATEGORICAL, align_categories, feature_list, load_split
from aml.metrics import ranking_metrics
from aml.sampling import correct_probability
from aml.tracking import setup_mlflow

MODELS_DIR = PROJECT_ROOT / "models"
CFG = load_train_config()
SEED = CFG["seed"]; N_JOBS = CFG["compute"]["n_jobs"]; R = CFG["sampling"]["negative_rate"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_data(limit: int | None = None):
    feats = [f for f in feature_list() if f not in CFG["excluded_features"]]
    train = load_split("train_sample"); vs = load_split("val_sample"); vf = load_split("val_full")
    if limit:  # smoke tests only
        train = pd.concat([train[train.label == 1].head(limit // 10), train[train.label == 0].head(limit)])
        vs = vs.head(limit); vf = vf.head(limit * 3)
    vs, vf = align_categories([vs, vf], reference=train)
    for c in CATEGORICAL:
        train[c] = train[c].astype(str).astype(pd.CategoricalDtype(vs[c].cat.categories))
    return feats, train, vs, vf


def monotone_vector(feats: list[str], use: bool) -> list[int]:
    m = CFG["monotone_constraints"]
    return [int(m.get(f, 0)) if use else 0 for f in feats]


# --------------------------------------------------------------------------- #
# Model fitting
# --------------------------------------------------------------------------- #
def fit_lgbm(params: dict, feats, train, vs, monotone: bool):
    p = {"objective": "binary", "metric": "average_precision", "verbosity": -1, "seed": SEED,
         "num_threads": N_JOBS, "bagging_freq": 1, **params,
         "monotone_constraints": monotone_vector(feats, monotone),
         "monotone_constraints_method": "advanced"}
    dtr = lgb.Dataset(train[feats], train["label"])               # unweighted (case-control)
    dva = lgb.Dataset(vs[feats], vs["label"], weight=vs["weight"], reference=dtr)
    booster = lgb.train(p, dtr, num_boost_round=3000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    return booster


def fit_xgb(params: dict, feats, train, vs, monotone: bool):
    p = {"objective": "binary:logistic", "eval_metric": "aucpr", "tree_method": "hist",
         "seed": SEED, "nthread": N_JOBS, **params}
    if monotone:
        p["monotone_constraints"] = "(" + ",".join(map(str, monotone_vector(feats, True))) + ")"
    dtr = xgb.DMatrix(train[feats], train["label"], enable_categorical=True)
    dva = xgb.DMatrix(vs[feats], vs["label"], weight=vs["weight"], enable_categorical=True)
    return xgb.train(p, dtr, num_boost_round=3000, evals=[(dva, "val")],
                     early_stopping_rounds=100, verbose_eval=False)


def predict(model, X: pd.DataFrame) -> np.ndarray:
    """Real-world-scale probability (prior-corrected for the 8% negative sampling)."""
    if isinstance(model, lgb.Booster):
        p = model.predict(X, num_iteration=model.best_iteration)
    else:
        p = model.predict(xgb.DMatrix(X, enable_categorical=True),
                          iteration_range=(0, model.best_iteration + 1))
    return correct_probability(p, R)


# --------------------------------------------------------------------------- #
# Optuna search spaces
# --------------------------------------------------------------------------- #
def space_lgbm(t: optuna.Trial) -> dict:
    return {"learning_rate": t.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": t.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": t.suggest_int("min_child_samples", 20, 500, log=True),
            "feature_fraction": t.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": t.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l2": t.suggest_float("lambda_l2", 1e-3, 10, log=True)}


def space_xgb(t: optuna.Trial) -> dict:
    return {"eta": t.suggest_float("eta", 0.02, 0.2, log=True),
            "max_depth": t.suggest_int("max_depth", 3, 10),
            "min_child_weight": t.suggest_float("min_child_weight", 1, 100, log=True),
            "subsample": t.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda": t.suggest_float("lambda", 1e-3, 10, log=True)}


SPACES = {"lgbm": (space_lgbm, fit_lgbm), "xgb": (space_xgb, fit_xgb)}


def tune(model_name: str, n_trials: int, limit: int | None = None):
    feats, train, vs, vf = load_data(limit)
    space, fit = SPACES[model_name]
    setup_mlflow()
    storage = CFG["optuna"]["storage"].replace("sqlite:///", "sqlite:///" + PROJECT_ROOT.as_posix() + "/")
    study = optuna.create_study(study_name=f"{model_name}_constrained" + ("_smoke" if limit else ""),
                                direction="maximize", storage=storage, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=SEED + len(feats)))

    def objective(trial):
        params = space(trial)
        t0 = time.time()
        model = fit(params, feats, train, vs, monotone=True)
        score = float(average_precision_score(vf["label"], predict(model, vf[feats])))  # full validation
        if limit:  # smoke test: don't pollute MLflow
            return score
        with mlflow.start_run(run_name=f"tune_{model_name}_t{trial.number}"):
            mlflow.set_tags({"stage": "tuning", "model_family": model_name, "monotone": "yes"})
            mlflow.log_params(params)
            mlflow.log_metrics({"val_pr_auc": score, "best_iteration": model.best_iteration,
                                "fit_seconds": time.time() - t0})
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, timeout=CFG["optuna"]["timeout_seconds"])
    done = [t for t in study.trials if t.value is not None]
    print(f"{model_name}: {len(done)} trials total | best val PR-AUC {study.best_value:.4f} | {study.best_params}")
    return study


# --------------------------------------------------------------------------- #
# Final models
# --------------------------------------------------------------------------- #
def final():
    feats, train, vs, vf = load_data()
    setup_mlflow()
    storage = CFG["optuna"]["storage"].replace("sqlite:///", "sqlite:///" + PROJECT_ROOT.as_posix() + "/")
    best = {m: optuna.load_study(study_name=f"{m}_constrained", storage=storage) for m in ("lgbm", "xgb")}
    MODELS_DIR.mkdir(exist_ok=True)
    y = vf["label"].to_numpy(); budgets = CFG["alert_budgets"]
    rows, models = {}, {}

    candidates = [("lgbm_monotone", "lgbm", True), ("lgbm_unconstrained", "lgbm", False),
                  ("xgb_monotone", "xgb", True)]
    for name, fam, mono in candidates:
        params = best[fam].best_params
        t0 = time.time()
        model = SPACES[fam][1](params, feats, train, vs, monotone=mono)
        p = predict(model, vf[feats])
        m = ranking_metrics(y, p, budgets)
        m.update(best_iteration=int(model.best_iteration), fit_seconds=time.time() - t0,
                 mean_pred_prob=float(p.mean()))
        with mlflow.start_run(run_name=f"final_{name}"):
            mlflow.set_tags({"stage": "final_candidate", "model_family": fam, "monotone": str(mono)})
            mlflow.log_params({**params, "monotone": mono, "n_features": len(feats)})
            mlflow.log_metrics(m)
        rows[name] = m; models[name] = model
        print(f"{name:20s} PR-AUC {m['pr_auc']:.4f} | recall@1% {m['recall_at_1pct']:.4f} | iters {m['best_iteration']}")

    # Decision rule: keep the monotone LightGBM unless constraints cost > max_constraint_cost.
    cost = 1 - rows["lgbm_monotone"]["pr_auc"] / rows["lgbm_unconstrained"]["pr_auc"]
    champion = "lgbm_monotone" if cost <= CFG["max_constraint_cost"] else "lgbm_unconstrained"
    print(f"constraint cost: {cost:+.2%} relative PR-AUC -> champion = {champion}")

    # Save models locally (used by evaluation, API and dashboard)
    models["lgbm_monotone"].save_model(str(MODELS_DIR / "lgbm_monotone.txt"), num_iteration=models["lgbm_monotone"].best_iteration)
    models["lgbm_unconstrained"].save_model(str(MODELS_DIR / "lgbm_unconstrained.txt"), num_iteration=models["lgbm_unconstrained"].best_iteration)
    models["xgb_monotone"].save_model(str(MODELS_DIR / "xgb_monotone.json"))
    meta = {"features": feats, "categories": {c: list(train[c].cat.categories) for c in CATEGORICAL},
            "negative_rate": R, "champion": champion, "constraint_cost_rel_pr_auc": cost,
            "monotone_constraints": CFG["monotone_constraints"],
            "xgb_best_iteration": int(models["xgb_monotone"].best_iteration),
            "best_params": {m: s.best_params for m, s in best.items()}}
    (MODELS_DIR / "model_meta.json").write_text(json.dumps(meta, indent=1))

    # Register champion in the MLflow Model Registry with alias "champion"
    name_reg = CFG["mlflow"]["registered_model_name"]
    with mlflow.start_run(run_name=f"register_{champion}"):
        mlflow.set_tags({"stage": "registered", "champion": champion})
        mlflow.log_metrics({f"val_{k}": v for k, v in rows[champion].items() if isinstance(v, float)})
        mlflow.log_dict(meta, "model_meta.json")
        info = mlflow.lightgbm.log_model(models[champion], name="model", registered_model_name=name_reg)
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(name_reg, "champion", info.registered_model_version)

    # Report
    n_trials = {m: len([t for t in s.trials if t.value is not None]) for m, s in best.items()}
    cols = ["pr_auc", "roc_auc", "recall_at_0_5pct", "recall_at_1pct", "recall_at_2pct", "precision_at_1pct", "best_iteration"]
    tbl = pd.DataFrame(rows).T[cols]
    lines = ["# Step 6 — Gradient boosting (validation period, full data)", "",
             f"Optuna trials: LightGBM {n_trials['lgbm']}, XGBoost {n_trials['xgb']} (objective: PR-AUC on full validation).",
             "Trained unweighted on train_sample; probabilities prior-corrected for 8% negative sampling.", "",
             tbl.to_markdown(floatfmt=".4f"), "",
             f"**Monotonic constraints cost {cost:+.2%} relative PR-AUC** "
             f"(rule: keep constraints if cost ≤ {CFG['max_constraint_cost']:.0%}) → champion: `{champion}`, "
             f"registered as `{name_reg}` version {info.registered_model_version} (alias `champion`).", "",
             "## Constraints", "", *[f"- `{k}`: {'+' if v > 0 else '−'} (risk must {'not decrease' if v > 0 else 'not increase'})"
                                    for k, v in CFG["monotone_constraints"].items()], "",
             "## Best hyperparameters", "", "```json", json.dumps(meta["best_params"], indent=1), "```", ""]
    (REPORTS_DIR / "boosting.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote reports/boosting.md and models/")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune"); t.add_argument("--model", choices=list(SPACES), required=True)
    t.add_argument("--trials", type=int, default=CFG["optuna"]["n_trials_per_batch"])
    t.add_argument("--limit", type=int, default=None, help="tiny subset for smoke tests")
    sub.add_parser("final")
    a = ap.parse_args()
    if a.cmd == "tune":
        tune(a.model, a.trials, a.limit)
    else:
        final()


if __name__ == "__main__":
    main()
