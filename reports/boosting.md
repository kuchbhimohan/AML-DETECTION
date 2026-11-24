# Step 6 — Gradient boosting (validation period, full data)

Optuna trials: LightGBM 17, XGBoost 6 (objective: PR-AUC on full validation).
Trained unweighted on train_sample; probabilities prior-corrected for 8% negative sampling.

|                    |   pr_auc |   roc_auc |   recall_at_0_5pct |   recall_at_1pct |   recall_at_2pct |   precision_at_1pct |   best_iteration |
|:-------------------|---------:|----------:|-------------------:|-----------------:|-----------------:|--------------------:|-----------------:|
| lgbm_monotone      |   0.5740 |    0.9875 |             0.7346 |           0.8089 |           0.8533 |              0.0868 |         748.0000 |
| lgbm_unconstrained |   0.5795 |    0.9873 |             0.7384 |           0.8098 |           0.8485 |              0.0869 |         618.0000 |
| xgb_monotone       |   0.5633 |    0.9867 |             0.7326 |           0.8012 |           0.8359 |              0.0860 |         784.0000 |

**Monotonic constraints cost +0.95% relative PR-AUC** (rule: keep constraints if cost ≤ 3%) → champion: `lgbm_monotone`, registered as `aml-lgbm` version 1 (alias `champion`).

## Constraints

- `s_out_cnt_1h`: + (risk must not decrease)
- `s_out_cnt_24h`: + (risk must not decrease)
- `s_out_uniq_24h`: + (risk must not decrease)
- `s_out_uniq_7d`: + (risk must not decrease)
- `r_in_uniq_24h`: + (risk must not decrease)
- `r_in_uniq_7d`: + (risk must not decrease)
- `pair_prev_cnt`: − (risk must not increase)

## Best hyperparameters

```json
{
 "lgbm": {
  "learning_rate": 0.022359698854299274,
  "num_leaves": 66,
  "min_child_samples": 219,
  "feature_fraction": 0.6778325619780777,
  "bagging_fraction": 0.9710736814376468,
  "lambda_l2": 5.188869122224808
 },
 "xgb": {
  "eta": 0.05573003550177858,
  "max_depth": 9,
  "min_child_weight": 17.494008773833386,
  "subsample": 0.7646616568086777,
  "colsample_bytree": 0.533258440653874,
  "lambda": 0.0800696472188664
 }
}
```
