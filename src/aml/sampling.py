"""Case-control (negative) sampling and probability correction.

Why: 99.9% of transactions are normal. Keeping all laundering rows and a random
fraction r of normal rows shrinks training data ~12x with almost no information loss.

The catch: a model trained on the sample sees laundering ~1/r times more often than
reality, so its raw probabilities are too high. Two equivalent fixes:

1. Train with weights (normal rows weight 1/r) -> probabilities are already on the
   real-world scale.
2. Train unweighted, then correct each probability (prior correction):
       odds_true = r * odds_sample
       p_true    = r * p / (r * p + 1 - p)

Ranking (and therefore PR-AUC / recall@budget) is unchanged by the correction,
because it is a monotonic transform. Calibration (Brier, reliability) is not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def case_control_sample(df: pd.DataFrame, negative_rate: float, seed: int = 42,
                        label_col: str = "label") -> pd.DataFrame:
    """Keep every positive and each negative with probability `negative_rate`.
    Adds a `weight` column (1 for positives, 1/negative_rate for negatives)."""
    if not 0 < negative_rate <= 1:
        raise ValueError("negative_rate must be in (0, 1]")
    rng = np.random.default_rng(seed)
    keep = (df[label_col] == 1).to_numpy() | (rng.random(len(df)) < negative_rate)
    out = df[keep].copy()
    out["weight"] = np.where(out[label_col] == 1, 1.0, 1.0 / negative_rate)
    return out


def correct_probability(p, negative_rate: float):
    """Map probabilities from an UNWEIGHTED model trained on a case-control sample
    back to the real-world scale."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    r = negative_rate
    return r * p / (r * p + 1 - p)


def weighted_positive_rate(y, w) -> float:
    """Estimated real-world positive rate from a weighted sample."""
    y = np.asarray(y); w = np.asarray(w, dtype=np.float64)
    return float((w * y).sum() / w.sum())
