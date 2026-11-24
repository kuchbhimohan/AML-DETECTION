import numpy as np
import pytest

from aml.metrics import gains_curve, gini_from_auc, precision_at_budget, ranking_metrics, recall_at_budget


def test_recall_and_precision_at_budget_hand_example():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    s = np.array([.9, .8, .7, .1, .1, .1, .1, .1, .1, .05])
    # top 20% = 2 transactions: scores .9 (pos) and .8 (neg)
    assert recall_at_budget(y, s, 0.2) == pytest.approx(1 / 3)
    assert precision_at_budget(y, s, 0.2) == pytest.approx(0.5)
    assert recall_at_budget(y, s, 1.0) == 1.0


def test_perfect_and_random_scores():
    rng = np.random.default_rng(0)
    y = (rng.random(50_000) < 0.01).astype(int)
    perfect = ranking_metrics(y, y + rng.random(len(y)) * 0.1)
    assert perfect["pr_auc"] == pytest.approx(1.0) and perfect["recall_at_1pct"] > 0.9
    rand = ranking_metrics(y, rng.random(len(y)))
    assert rand["roc_auc"] == pytest.approx(0.5, abs=0.03)
    assert rand["recall_at_1pct"] == pytest.approx(0.01, abs=0.01)


def test_gains_curve_endpoints_and_gini():
    y = np.array([0, 1, 0, 1]); s = np.array([.1, .9, .2, .8])
    x, g = gains_curve(y, s)
    assert (x[0], g[0]) == (0, 0) and g[-1] == 1.0
    assert gini_from_auc(0.5) == 0 and gini_from_auc(1.0) == 1
