"""Business-oriented ranking metrics for alerting.

The investigation team can review only a fixed fraction of transactions
("alert budget"). So the key question is: if we alert on the top-k% riskiest
transactions, what share of all laundering do we catch?
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _top_k_mask(score, budget: float):
    score = np.asarray(score, dtype=np.float64)
    k = max(1, int(round(budget * len(score))))
    # stable, deterministic tie-breaking: higher score first, then lower index
    order = np.lexsort((np.arange(len(score)), -score))
    mask = np.zeros(len(score), dtype=bool); mask[order[:k]] = True
    return mask


def recall_at_budget(y, score, budget: float) -> float:
    """Share of all positives that fall in the top `budget` fraction by score."""
    y = np.asarray(y)
    return float(y[_top_k_mask(score, budget)].sum() / max(y.sum(), 1))


def precision_at_budget(y, score, budget: float) -> float:
    """Share of alerts (top `budget` fraction) that are true positives."""
    y = np.asarray(y); m = _top_k_mask(score, budget)
    return float(y[m].mean())


def gains_curve(y, score, n_points: int = 200):
    """Cumulative gains: x = fraction of transactions reviewed, y = fraction of
    laundering caught. Returns (x, y) arrays starting at (0, 0)."""
    y = np.asarray(y); order = np.argsort(-np.asarray(score), kind="stable")
    caught = np.cumsum(y[order]) / max(y.sum(), 1)
    idx = np.unique(np.linspace(0, len(y) - 1, n_points).astype(int))
    return np.r_[0, (idx + 1) / len(y)], np.r_[0, caught[idx]]


def gini_from_auc(auc: float) -> float:
    """Gini (as used in credit/insurance scoring) = 2*AUC - 1."""
    return 2 * auc - 1


def ranking_metrics(y, score, budgets=(0.005, 0.01, 0.02)) -> dict[str, float]:
    auc = roc_auc_score(y, score)
    out = {"pr_auc": float(average_precision_score(y, score)), "roc_auc": float(auc),
           "gini": float(gini_from_auc(auc)), "base_rate": float(np.mean(y))}
    for b in budgets:
        tag = f"{b * 100:g}pct".replace(".", "_")
        out[f"recall_at_{tag}"] = recall_at_budget(y, score, b)
        out[f"precision_at_{tag}"] = precision_at_budget(y, score, b)
    return out
