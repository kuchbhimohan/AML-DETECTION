# Step 8 — Explainability (SHAP)

- Exact TreeSHAP (LightGBM `pred_contrib`) on a test sample: all 956 laundering + 15,000 random normal transactions;
  importance re-weighted to the real normal:laundering ratio.
- **Additivity check:** max |base + Σ SHAP − model log-odds| = 5.15e-14 (exact).

## Global importance (mean |SHAP|, log-odds)

| Feature | All traffic | Laundering only | Direction (ρ value vs SHAP) |
|---|---|---|---|
| `pair_prev_cnt` | 0.795 | 1.594 | -0.61 |
| `payment_format` | 0.367 | 1.792 | — (categorical) |
| `is_ach` | 0.310 | 2.072 | +0.67 |
| `s_hours_since_prev_out` | 0.299 | 0.566 | +0.78 |
| `log_amount_usd` | 0.261 | 0.874 | +0.72 |
| `s_age_hours` | 0.253 | 0.518 | +0.01 |
| `s_out_uniq_24h` | 0.243 | 0.217 | +0.35 |
| `s_out_cnt_1h` | 0.190 | 0.169 | +0.11 |
| `self_transfer` | 0.143 | 0.231 | -0.25 |
| `payment_currency` | 0.116 | 0.168 | — (categorical) |
| `r_age_hours` | 0.113 | 0.263 | -0.00 |
| `s_out_uniq_7d` | 0.105 | 0.127 | +0.59 |
| `s_out_usd_7d` | 0.096 | 0.160 | +0.61 |
| `r_in_cnt_7d` | 0.094 | 0.125 | +0.36 |
| `s_out_uniq_1h` | 0.093 | 0.085 | -0.30 |

Note: `is_ach` and `payment_format` carry the same ACH signal, so the model splits credit between them; alert reasons merge them (and `pair_prev_cnt` + `pair_first_time`) into one reason each. The age features have near-zero direction: their effect is non-monotone (see Step 7 ablation).

## Sanity checks

- **Agreement with logistic regression:** of 16 features that are significant in the logistic model (p<0.05) and have a clear SHAP direction, **11 agree in sign**. Disagreements: `s_out_usd_24h`, `s_amt_z_7d`, `s_in_uniq_24h`, `r_out_cnt_24h`, `s_passthrough_ratio_24h` (logistic coefficients are *conditional* on correlated features; SHAP directions are marginal, so some flips are expected).
- **Monotone constraints visible in SHAP:** `s_out_cnt_1h` ✅, `s_out_cnt_24h` ✅, `s_out_uniq_24h` ✅, `s_out_uniq_7d` ✅, `r_in_uniq_24h` ✅, `r_in_uniq_7d` ✅, `pair_prev_cnt` ✅

## Example alerts (top-3 reasons that pushed risk up)

**Example 1** — laundering, P(laundering) = 96.7%

1. Payment format: ACH (SHAP +6.06)
2. First-ever payment from this sender to this receiver (SHAP +2.43)
3. Amount ≈ $11,902 (SHAP +1.80)

**Example 2** — laundering, P(laundering) = 95.8%

1. Payment format: ACH (SHAP +6.03)
2. Amount ≈ $12,099 (SHAP +1.80)
3. First-ever payment from this sender to this receiver (SHAP +1.54)

**Example 3** — laundering, P(laundering) = 95.6%

1. Payment format: ACH (SHAP +5.76)
2. First-ever payment from this sender to this receiver (SHAP +2.61)
3. Amount ≈ $14,437 (SHAP +1.63)

**Example 4** — **false alarm** (normal transaction), P(laundering) = 35.0%

1. Payment format: ACH (SHAP +4.13)
2. First-ever payment from this sender to this receiver (SHAP +2.20)
3. Amount ≈ $13,417 (SHAP +1.60)

Figures: `figures/step8_shap_importance.png`, `figures/step8_shap_beeswarm.png`, `figures/step8_dependence.png`
