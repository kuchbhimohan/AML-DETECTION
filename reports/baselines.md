# Step 5 — Baselines (validation period, full data)

Validation: 965,524 transactions, 1,036 laundering (base rate 0.1073%).
Trained on train_sample (all laundering + 8% of normal). Test period untouched.

|                              |   pr_auc |   roc_auc |   gini |   recall_at_0_5pct |   recall_at_1pct |   recall_at_2pct |   precision_at_1pct |
|:-----------------------------|---------:|----------:|-------:|-------------------:|-----------------:|-----------------:|--------------------:|
| rules (ACH, amount)          |   0.0121 |    0.8962 | 0.7925 |             0.0183 |           0.0222 |           0.1844 |              0.0024 |
| logistic regression          |   0.1321 |    0.9750 | 0.9501 |             0.5888 |           0.7375 |           0.8127 |              0.0791 |
| isolation forest (no labels) |   0.0046 |    0.8803 | 0.7605 |             0.0058 |           0.0135 |           0.0454 |              0.0015 |

## Logistic regression

- Features used after separation + VIF pruning: 28; pseudo-R² = 0.497
- Mean corrected predicted probability on validation: 0.0794% vs actual 0.1073%. The train period's rate is lower (0.0780%), so the gap is a base-rate shift between periods, not a modelling error (examined again in Step 7: drift + calibration).

### Dropped features

- `cross_currency`: separation/constant (no laundering in level 1)
- `fmt_Reinvestment`: separation/constant (no laundering in level 1)
- `fmt_Wire`: separation/constant (no laundering in level 1)
- `fmt_Cheque`: reference category
- `is_ach`: VIF 1000799917193443.5 > 10
- `s_out_uniq_1h`: VIF 247.5 > 10
- `s_out_uniq_24h`: VIF 224.8 > 10
- `s_out_cnt_7d`: VIF 169.4 > 10
- `s_out_uniq_7d`: VIF 55.6 > 10
- `r_in_cnt_7d`: VIF 24.7 > 10
- `s_out_cnt_24h`: VIF 21.8 > 10
- `r_out_uniq_24h`: VIF 16.4 > 10
- `s_in_cnt_24h`: VIF 15.5 > 10
- `r_in_uniq_1h`: VIF 14.7 > 10
- `s_out_usd_7d`: VIF 12.8 > 10
- `s_age_hours`: VIF 12.5 > 10
- `self_transfer`: VIF 11.2 > 10

### Odds ratios per 1 SD (sorted by |coef|)

|                         |    coef |   p_value |   odds_ratio |   or_ci_low |   or_ci_high |
|:------------------------|--------:|----------:|-------------:|------------:|-------------:|
| const                   | -8.2316 |    0      |       0.0003 |      0.0002 |       0.0003 |
| fmt_ACH                 |  1.5524 |    0      |       4.7226 |      4.3617 |       5.1133 |
| pair_prev_cnt           | -1.25   |    0      |       0.2865 |      0.245  |       0.3351 |
| s_out_cnt_1h            |  1.0762 |    0      |       2.9336 |      2.6781 |       3.2133 |
| cross_bank              |  0.7817 |    0      |       2.1851 |      1.8977 |       2.5161 |
| r_age_hours             |  0.7029 |    0      |       2.0196 |      1.8005 |       2.2655 |
| r_in_uniq_7d            |  0.6963 |    0      |       2.0063 |      1.825  |       2.2055 |
| r_in_cnt_1h             | -0.6951 |    0      |       0.499  |      0.469  |       0.5309 |
| log_amount_usd          |  0.4687 |    0      |       1.5979 |      1.5028 |       1.6989 |
| pair_first_time         |  0.4004 |    0      |       1.4924 |      1.3207 |       1.6866 |
| s_hours_since_prev_out  |  0.3307 |    0      |       1.3919 |      1.2825 |       1.5107 |
| fmt_Bitcoin             |  0.3268 |    0      |       1.3865 |      1.2976 |       1.4814 |
| reverse_flow_7d         |  0.3257 |    0      |       1.385  |      1.3012 |       1.4743 |
| s_in_usd_24h            |  0.2663 |    0      |       1.3052 |      1.2108 |       1.4069 |
| s_in_uniq_24h           | -0.221  |    0      |       0.8017 |      0.755  |       0.8514 |
| s_out_usd_24h           |  0.203  |    0.0002 |       1.225  |      1.1016 |       1.3622 |
| s_amt_ratio_7d          | -0.1707 |    0      |       0.843  |      0.8045 |       0.8834 |
| r_in_usd_24h            | -0.1619 |    0.0001 |       0.8505 |      0.7839 |       0.9227 |
| fmt_Credit Card         |  0.1514 |    0.0031 |       1.1634 |      1.0522 |       1.2864 |
| hour_cos                | -0.1457 |    0      |       0.8644 |      0.8237 |       0.9071 |
| r_out_cnt_24h           | -0.1112 |    0.0001 |       0.8948 |      0.8471 |       0.9451 |
| r_in_uniq_24h           | -0.0768 |    0.2986 |       0.926  |      0.8011 |       1.0704 |
| s_passthrough_ratio_24h |  0.0659 |    0.0496 |       1.0682 |      1.0001 |       1.1409 |
| s_amt_z_7d              | -0.0523 |    0.0043 |       0.949  |      0.9156 |       0.9837 |
| fmt_Cash                |  0.0511 |    0.2316 |       1.0525 |      0.9679 |       1.1444 |
| r_in_cnt_24h            |  0.0393 |    0.573  |       1.0401 |      0.9071 |       1.1926 |
| hour_sin                |  0.0373 |    0.0933 |       1.038  |      0.9938 |       1.0843 |
| round_amount            |  0.0111 |    0.4237 |       1.0111 |      0.9841 |       1.039  |
| s_no_prev_out           |  0.0006 |    0.9919 |       1.0006 |      0.8954 |       1.1181 |
