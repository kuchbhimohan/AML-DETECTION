"""Statistical tools for honest model evaluation.

- Poisson bootstrap        : confidence intervals for any ranking metric (fast, O(n) per resample)
- DeLong test              : are two correlated ROC-AUCs significantly different?
- McNemar test             : do two models catch different laundering cases at a fixed alert budget?
- PSI                      : population stability index (feature / score drift between periods)
- Wilson interval          : CI for a proportion with small counts (per-pattern recall)
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Weighted metrics on a fixed sort order (used by the bootstrap)
# --------------------------------------------------------------------------- #
class SortedScores:
    """Pre-sorts one model's scores once so each bootstrap resample is cheap.

    Positives are rare (~1k of ~1M), so only ONE full-length cumulative sum is
    needed (total weight above each position); everything else is computed at the
    positive positions only. Exact, not an approximation.
    """

    def __init__(self, y, score):
        self.order = np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")
        self.pos = np.flatnonzero(np.asarray(y)[self.order] == 1)   # positions of positives (sorted order)

    def metrics(self, w_sorted: np.ndarray, budgets) -> dict[str, float]:
        """w_sorted: per-row bootstrap weights, already in this model's sort order."""
        cw = np.cumsum(w_sorted, dtype=np.int64)          # total weight at or above each rank
        wp = w_sorted[self.pos].astype(np.float64)         # weights of positives
        cp = np.cumsum(wp)                                 # positive weight at or above each positive
        W = float(cw[-1]); P = cp[-1]; N = W - P
        cw_pos = cw[self.pos].astype(np.float64)
        out = {"pr_auc": float((wp * cp / np.maximum(cw_pos, 1e-12)).sum() / P)}
        neg_above_incl = cw_pos - cp                       # negatives ranked at/above each positive
        out["roc_auc"] = float((wp * (N - neg_above_incl)).sum() / (P * N))
        for b in budgets:
            k = np.searchsorted(cw, b * W, side="left")    # last rank inside the alert budget
            j = np.searchsorted(self.pos, k, side="right")  # positives at ranks <= k
            out[f"recall_at_{b * 100:g}pct".replace(".", "_")] = float(cp[j - 1] / P) if j else 0.0
        return out


def poisson_bootstrap(y, scores: dict[str, np.ndarray], budgets, n_resamples=1000, ci=0.95, seed=42,
                      diff_pairs: list[tuple[str, str]] = ()):
    """Paired Poisson bootstrap: every resample gives each transaction a Poisson(1)
    weight, the SAME weights for every model (so model differences are paired).

    Returns {model: {metric: (point, low, high)}} and {"A - B": {metric: (diff, low, high)}}.
    """
    y = np.asarray(y)
    sorted_ = {m: SortedScores(y, s) for m, s in scores.items()}
    ones = {m: ss.metrics(np.ones(len(y), dtype=np.uint8), budgets) for m, ss in sorted_.items()}
    rng = np.random.default_rng(seed)
    draws = {m: [] for m in scores}
    for _ in range(n_resamples):
        w = rng.poisson(1.0, len(y)).astype(np.uint8)     # Poisson(1) never exceeds 255 in practice
        for m, ss in sorted_.items():
            draws[m].append(ss.metrics(w[ss.order], budgets))
    a = (1 - ci) / 2
    res = {}
    for m in scores:
        keys = draws[m][0].keys()
        res[m] = {k: (ones[m][k], *np.quantile([d[k] for d in draws[m]], [a, 1 - a])) for k in keys}
    diffs = {}
    for A, B in diff_pairs:
        keys = draws[A][0].keys()
        diffs[f"{A} - {B}"] = {k: (ones[A][k] - ones[B][k],
                                   *np.quantile([da[k] - db[k] for da, db in zip(draws[A], draws[B])], [a, 1 - a]))
                               for k in keys}
    return res, diffs


# --------------------------------------------------------------------------- #
# DeLong test for two correlated ROC AUCs (Sun & Xu 2014 fast algorithm)
# --------------------------------------------------------------------------- #
def _structural_components(pos_scores, neg_scores):
    m, n = len(pos_scores), len(neg_scores)
    all_ranks = stats.rankdata(np.concatenate([pos_scores, neg_scores]))
    pos_ranks = stats.rankdata(pos_scores); neg_ranks = stats.rankdata(neg_scores)
    v10 = (all_ranks[:m] - pos_ranks) / n           # per positive: share of negatives below
    v01 = 1 - (all_ranks[m:] - neg_ranks) / m       # per negative: share of positives above
    return v10.mean(), v10, v01


def delong_test(y, score_a, score_b) -> dict[str, float]:
    y = np.asarray(y).astype(bool)
    auc_a, v10a, v01a = _structural_components(np.asarray(score_a)[y], np.asarray(score_a)[~y])
    auc_b, v10b, v01b = _structural_components(np.asarray(score_b)[y], np.asarray(score_b)[~y])
    s10 = np.cov(np.vstack([v10a, v10b])); s01 = np.cov(np.vstack([v01a, v01b]))
    cov = s10 / y.sum() + s01 / (~y).sum()
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if not var > 1e-18:  # (near-)identical rankings: no evidence of a difference
        return {"auc_a": float(auc_a), "auc_b": float(auc_b), "diff": float(auc_a - auc_b),
                "se": 0.0, "z": 0.0, "p_value": 1.0}
    z = (auc_a - auc_b) / np.sqrt(var)
    return {"auc_a": float(auc_a), "auc_b": float(auc_b), "diff": float(auc_a - auc_b),
            "se": float(np.sqrt(var)), "z": float(z), "p_value": float(2 * stats.norm.sf(abs(z)))}


# --------------------------------------------------------------------------- #
# McNemar test on paired binary outcomes
# --------------------------------------------------------------------------- #
def mcnemar_test(correct_a, correct_b) -> dict[str, float]:
    """b = A right & B wrong, c = A wrong & B right. Exact binomial test on discordant pairs."""
    a = np.asarray(correct_a, bool); bb = np.asarray(correct_b, bool)
    b = int((a & ~bb).sum()); c = int((~a & bb).sum())
    p = float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    return {"a_only": b, "b_only": c, "both": int((a & bb).sum()), "neither": int((~a & ~bb).sum()), "p_value": p}


# --------------------------------------------------------------------------- #
# Population stability index
# --------------------------------------------------------------------------- #
def psi(expected, actual, bins: int = 10, w_expected=None) -> float:
    """PSI = Σ (a - e) · ln(a / e) over quantile bins of the EXPECTED (reference) data.
    < 0.1 stable · 0.1–0.25 moderate shift · > 0.25 major shift."""
    e = np.asarray(expected, dtype=np.float64); a_ = np.asarray(actual, dtype=np.float64)
    we = np.ones(len(e)) if w_expected is None else np.asarray(w_expected, dtype=np.float64)
    order = np.argsort(e); cw = np.cumsum(we[order]) / we.sum()
    edges = np.unique(e[order][np.searchsorted(cw, np.linspace(0, 1, bins + 1)[1:-1])])
    eb = np.bincount(np.searchsorted(edges, e, side="right"), weights=we, minlength=len(edges) + 1) / we.sum()
    ab = np.bincount(np.searchsorted(edges, a_, side="right"), minlength=len(edges) + 1) / len(a_)
    eb = np.clip(eb, 1e-6, None); ab = np.clip(ab, 1e-6, None)
    return float(((ab - eb) * np.log(ab / eb)).sum())


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
CAL_EDGES = np.array([0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0000001])


def expected_calibration_error(y, p, edges=CAL_EDGES) -> float:
    """ECE = Σ_bins (share of transactions in bin) · |mean predicted − observed rate|.
    Log-spaced bins, because almost all probabilities here are tiny (base rate ~0.1%)."""
    y = np.asarray(y, dtype=np.float64); p = np.asarray(p, dtype=np.float64)
    b = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
    n = np.bincount(b, minlength=len(edges) - 1)
    sp = np.bincount(b, weights=p, minlength=len(edges) - 1)
    sy = np.bincount(b, weights=y, minlength=len(edges) - 1)
    m = n > 0
    return float((np.abs(sp[m] - sy[m])).sum() / len(y))


def platt_fit(p_val, y_val):
    """Platt scaling on the log-odds of the model's probability: returns a function p -> calibrated p."""
    from sklearn.linear_model import LogisticRegression
    def logit(p):
        p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
        return np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p_val), y_val)   # ~unpenalised
    return lambda p: lr.predict_proba(logit(p))[:, 1], (float(lr.coef_[0, 0]), float(lr.intercept_[0]))


# --------------------------------------------------------------------------- #
def wilson_interval(k: int, n: int, ci: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - (1 - ci) / 2); p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (float(centre - half), float(centre + half))
