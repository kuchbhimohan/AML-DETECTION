"""Check each statistical tool against a reference implementation or known behaviour."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score
from statsmodels.stats.proportion import proportion_confint

from aml.metrics import recall_at_budget
from aml.stats import SortedScores, delong_test, mcnemar_test, poisson_bootstrap, psi, wilson_interval


@pytest.fixture(scope="module")
def sim():
    rng = np.random.default_rng(0)
    n = 60_000
    y = (rng.random(n) < 0.01).astype(int)
    good = y * 1.5 + rng.normal(size=n)          # strong model
    weak = y * 0.7 + rng.normal(size=n)          # weaker model
    return y, good, weak


def test_sorted_scores_match_sklearn_with_unit_weights(sim):
    y, good, _ = sim
    m = SortedScores(y, good).metrics(np.ones(len(y), dtype=np.uint8), [0.01])
    assert m["pr_auc"] == pytest.approx(average_precision_score(y, good), rel=1e-9)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, good), rel=1e-9)
    assert m["recall_at_1pct"] == pytest.approx(recall_at_budget(y, good, 0.01), abs=2 / y.sum())


def test_fast_weighted_metrics_match_sklearn_with_poisson_weights(sim):
    y, good, _ = sim
    w = np.random.default_rng(5).poisson(1.0, len(y)).astype(np.uint8)
    ss = SortedScores(y, good)
    m = ss.metrics(w[ss.order], [0.01])
    assert m["pr_auc"] == pytest.approx(average_precision_score(y, good, sample_weight=w), rel=1e-9)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, good, sample_weight=w), rel=1e-9)


def test_bootstrap_ci_contains_point_and_detects_difference(sim):
    y, good, weak = sim
    res, diffs = poisson_bootstrap(y, {"good": good, "weak": weak}, [0.01], n_resamples=200,
                                   diff_pairs=[("good", "weak")])
    p, lo, hi = res["good"]["pr_auc"]
    assert lo < p < hi
    d, dlo, dhi = diffs["good - weak"]["roc_auc"]
    assert dlo > 0                                 # better model is significantly better


def test_delong_auc_values_and_significance(sim):
    y, good, weak = sim
    r = delong_test(y, good, weak)
    assert r["auc_a"] == pytest.approx(roc_auc_score(y, good), rel=1e-9)
    assert r["auc_b"] == pytest.approx(roc_auc_score(y, weak), rel=1e-9)
    assert r["p_value"] < 1e-6
    same = delong_test(y, good, good + 1e-9 * np.random.default_rng(1).normal(size=len(y)))
    assert same["p_value"] > 0.05                  # identical models: no difference


def test_mcnemar_counts_and_p():
    a = np.array([1] * 30 + [0] * 5 + [1] * 50, bool)
    b = np.array([0] * 30 + [1] * 5 + [1] * 50, bool)
    r = mcnemar_test(a, b)
    assert (r["a_only"], r["b_only"], r["both"]) == (30, 5, 50)
    assert r["p_value"] < 0.001


def test_psi_stable_vs_shifted():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=50_000)
    assert psi(ref, rng.normal(size=50_000)) < 0.01
    assert psi(ref, rng.normal(1.0, 1, size=50_000)) > 0.25


def test_wilson_matches_statsmodels():
    for k, n in [(0, 10), (7, 20), (55, 57)]:
        assert wilson_interval(k, n) == pytest.approx(proportion_confint(k, n, method="wilson"), rel=1e-9)


def test_fast_vif_matches_statsmodels():
    import pandas as pd
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from aml.baselines import vif_fast
    rng = np.random.default_rng(0)
    Z = pd.DataFrame(rng.normal(size=(4000, 5)))
    Z[5] = Z[0] * 0.9 + Z[1] * 0.3 + rng.normal(size=4000) * 0.1     # strongly collinear column
    Z = (Z - Z.mean()) / Z.std()
    ref = np.array([variance_inflation_factor(Z.values, i) for i in range(Z.shape[1])])
    assert np.allclose(vif_fast(Z).values, ref, rtol=1e-3)
    Z[6] = Z[0] + Z[1]                                                # perfectly collinear
    assert np.isinf(vif_fast(Z)[6])
