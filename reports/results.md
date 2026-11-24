# Final results — TEST period (untouched until now)

- Test period: 2022-09-09 00:00:00 → 2022-09-10 23:59:00 · 862,792 transactions · 956 laundering
- Models trained on Sep 1–6, tuned on Sep 7–8. 95% CIs from a paired Poisson bootstrap (1000 resamples).
- Regenerate with `uv run python -m aml.evaluate` (deterministic).

## 1. Headline (alert budget = share of transactions investigators review)

| Model | PR-AUC | ROC-AUC | Recall @0.5% | Recall @1% | Recall @2% | Precision @1% |
|---|---|---|---|---|---|---|
| LightGBM monotone (champion) | 0.534 [0.505–0.565] | 0.981 [0.979–0.983] | 0.667 [0.641–0.695] | 0.749 [0.722–0.776] | 0.778 [0.754–0.805] | 0.083 |
| LightGBM unconstrained | 0.538 [0.510–0.568] | 0.979 [0.976–0.982] | 0.667 [0.641–0.695] | 0.730 [0.704–0.759] | 0.771 [0.746–0.798] | 0.081 |
| XGBoost monotone | 0.519 | 0.981 | 0.668 | 0.736 | 0.769 | 0.082 |
| Logistic regression | 0.117 [0.103–0.135] | 0.964 [0.959–0.970] | 0.521 [0.490–0.550] | 0.659 [0.632–0.690] | 0.746 [0.720–0.773] | 0.073 |
| Rules (ACH, amount) | 0.012 | 0.900 | 0.020 | 0.021 | 0.155 | 0.002 |

## 2. Is the champion significantly better?

- **LightGBM monotone (champion) - Logistic regression**: ΔPR-AUC = +0.418 [+0.392, +0.442], Δrecall@1% = +0.090 [+0.065, +0.113]
- **LightGBM monotone (champion) - LightGBM unconstrained**: ΔPR-AUC = -0.004 [-0.007, -0.000], Δrecall@1% = +0.019 [+0.008, +0.029]
- DeLong (ROC-AUC) champion vs logistic: Δ = +0.0168, z = 7.57, p = 3.70e-14
- DeLong champion vs unconstrained: Δ = +0.0020, p = 7.19e-08
- McNemar at 1% budget (laundering cases): caught only by champion = 118, only by logistic = 32, both = 598, neither = 208; exact p = 8.84e-13

## 3. Calibration (champion)

- Brier: raw 0.000700 · isotonic (fit on validation) 0.000683 · constant-rate baseline 0.001107
- Mean predicted: raw 0.099%, isotonic 0.121%; actual 0.111%
- Figure: `figures/step7_reliability.png`

## 4. Operating points

- **Best-F1 threshold (chosen on validation)** → test: precision 75.7%, recall 45.3%, F1 0.567, 572 alerts (0.1% of transactions)


Cost-based rule (calibrated probabilities): alert when P(laundering) × cost of a missed case > cost of a review.
Isotonic calibration is a step function, so nearby thresholds can give identical alert sets.

| Cost of a missed case vs one review | Threshold | Alert rate | Recall | Precision |
|---|---|---|---|---|
| 50x | 0.0200 | 0.6% | 70.2% | 12.1% |
| 100x | 0.0100 | 1.4% | 75.8% | 6.2% |
| 300x | 0.0033 | 3.1% | 80.2% | 2.9% |
| 1000x | 0.0010 | 19.5% | 99.8% | 0.6% |

## 5. Drift (PSI; <0.1 stable, 0.1–0.25 moderate, >0.25 major)

- Score PSI train→test 0.149; val→test 0.060
- Features with PSI > 0.25: 9; 0.1–0.25: 8
- Base rate: train 0.078% → val 0.107% → test 0.111%
- Why so much feature drift: the data starts on Sep 1, so account age and long-window/all-history features (`*_age_hours`, `pair_prev_cnt`, `*_7d`) keep growing during the first week ('history warm-up'). The model still generalises (see §1), and §8 shows what happens without the age features.

| Feature | PSI |
|---|---|
| `r_age_hours` | 8.594 |
| `s_age_hours` | 8.468 |
| `pair_prev_cnt` | 1.706 |
| `r_in_cnt_7d` | 1.334 |
| `s_hours_since_prev_out` | 0.736 |
| `s_out_cnt_7d` | 0.652 |
| `pair_first_time` | 0.558 |
| `r_in_uniq_7d` | 0.505 |
| `self_transfer` | 0.339 |
| `cross_bank` | 0.248 |

## 6. Recall by laundering pattern (1% budget, Wilson 95% CI)

| Pattern | n | Champion | 95% CI | Logistic |
|---|---|---|---|---|
| UNLABELLED | 371 | 37.5% | 32.7%–42.5% | 35.6% |
| GATHER-SCATTER | 127 | 99.2% | 95.7%–99.9% | 90.6% |
| SCATTER-GATHER | 110 | 98.2% | 93.6%–99.5% | 90.0% |
| STACK | 84 | 96.4% | 90.0%–98.8% | 78.6% |
| FAN-OUT | 65 | 100.0% | 94.4%–100.0% | 86.2% |
| FAN-IN | 57 | 96.5% | 88.1%–99.0% | 91.2% |
| CYCLE | 55 | 100.0% | 93.5%–100.0% | 81.8% |
| BIPARTITE | 44 | 100.0% | 92.0%–100.0% | 70.5% |
| RANDOM | 43 | 100.0% | 91.8%–100.0% | 79.1% |

## 7. Sparse tail days (Sep 11–18, excluded from the main test)

- 1,108 transactions, 655 laundering (59.1%); champion ROC-AUC 0.979, PR-AUC 0.976
- Simulator artefact (laundering chains finishing after normal traffic stops); reported only as a sanity check.

## 8. Ablation: are the account-age features doing the work? (validation)

| | PR-AUC | Recall @1% |
|---|---|---|
| with age features (champion) | 0.574 | 80.9% |
| without `s_age_hours`, `r_age_hours` | 0.522 | 80.1% |

Age partly reflects 'time since the data starts' (only 10 days observed), so it is a known limitation. Removing it lowers PR-AUC but leaves recall at the 1% alert budget almost unchanged, so the headline business result does not rely on it.

_Runtime: 202.6s._
