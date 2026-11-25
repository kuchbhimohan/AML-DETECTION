# Step 7b — Other algorithms vs the champion (TEST period)

The champion (monotone LightGBM) was fixed on validation in Step 6. The models below use fixed, standard
settings (no tuning on test) and are reported for comparison only. Δ = model − champion, 95% CI from a
paired Poisson bootstrap (500 resamples); a CI that includes 0 means no significant difference.

| Model | PR-AUC | Recall @0.5% | Recall @1% | Recall @2% | Δ recall @1% vs champion [95% CI] |
|---|---|---|---|---|---|
| Extra Trees | 0.440 | 0.691 | 0.756 | 0.772 | +0.007 [-0.005, +0.021] |
| LightGBM monotone (champion) | 0.534 | 0.667 | 0.749 | 0.778 | — |
| Random Forest | 0.498 | 0.691 | 0.748 | 0.775 | -0.001 [-0.013, +0.013] |
| Ensemble: rank-average (LGBM + XGB + RF) | 0.533 | 0.679 | 0.747 | 0.785 | -0.002 [-0.012, +0.007] |
| XGBoost monotone | 0.519 | 0.668 | 0.736 | 0.769 | -0.013 [-0.022, -0.003] |
| Imbalance: SMOTE to 10% | 0.537 | 0.684 | 0.734 | 0.782 | -0.015 [-0.027, +0.001] |
| Imbalance: none (case-control sample) | 0.521 | 0.664 | 0.728 | 0.765 | -0.021 [-0.031, -0.011] |
| Imbalance: class weights (balanced) | 0.469 | 0.656 | 0.713 | 0.769 | -0.036 [-0.052, -0.021] |
| Neural network (MLP 64-32) | 0.353 | 0.626 | 0.697 | 0.744 | -0.052 [-0.074, -0.031] |

## Settings and cost

- **Extra Trees**: {"fit_and_score_seconds": 84.8, "n_features": 45}
- **Imbalance: class weights (balanced)**: {"fit_and_score_seconds": 29.0, "train_rows": 261880, "train_positives": 2530, "n_trees": 300}
- **Imbalance: none (case-control sample)**: {"fit_and_score_seconds": 25.4, "train_rows": 261880, "train_positives": 2530, "n_trees": 300}
- **Imbalance: SMOTE to 10%**: {"fit_and_score_seconds": 29.4, "train_rows": 285285, "train_positives": 25935, "n_trees": 300, "smote_seconds": 0.2}
- **Neural network (MLP 64-32)**: {"fit_and_score_seconds": 32.7, "n_features": 45}
- **Random Forest**: {"fit_and_score_seconds": 153.8, "n_features": 45}

Regenerate: run the `fit`/`imbalance` commands, then `uv run python -m aml.extra_models report`.
