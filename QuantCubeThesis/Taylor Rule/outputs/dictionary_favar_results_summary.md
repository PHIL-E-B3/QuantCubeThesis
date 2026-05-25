# Dictionary FAVAR Results Summary
**Document type: FOMC Statements | Date: 2026-05-19**

---

## 1. Setup

- **Target variable**: effective_rate(t+1) — Wu-Xia shadow rate merged with fed funds rate
- **Sample**: N = 726 meeting-date observations (FOMC calendar, all available)
- **Baseline regressors**: 4 PCA factors extracted from the macro panel (vix, gdp, unemployment_gap, inflation_dev_from_target, implied_ffr). Four components explain 91.0% variance.
- **Estimation**: OLS with HC3-robust standard errors
- **OOS**: Expanding-window (init = 60% of sample = 435 obs, 291 OOS predictions)
- **Normalization tested**: wordcount (raw counts ÷ doc length) and zscore (expanding z-score)
- **Extreme-val sweep**: 1.0–3.0 (no effect on dictionary outputs — see note in Section 5)

---

## 2. Baseline FAVAR (No Sentiment)

| Metric | Value |
|--------|-------|
| Adj R² | 0.8716 |
| AIC | 1219.67 |
| OOS RMSE | **0.8889** |
| N | 726 |

---

## 3. Best Model Per Combo — In-Sample vs OOS

All four combos peak in-sample at Spec 4e. The OOS ranking is different.

| Combo | Best IS Spec | IS Adj R² | IS AIC | Best OOS Spec | OOS RMSE |
|-------|-------------|-----------|--------|---------------|----------|
| **Sharpe — wordcount** | 4e | 0.9088 | 976.8 | **4e** | **0.7705** |
| **Sharpe — zscore** | 4e | 0.9069 | 991.7 | **4e** | 0.7954 |
| **Gardner — zscore** | 4e | 0.9089 | 975.9 | **4e** | 0.8125 |
| **Gardner — wordcount** | 4e | 0.8900 | 1112.8 | **4b_consensus** | 0.8483 |
| Baseline (no sentiment) | — | 0.8716 | 1219.7 | — | 0.8889 |

**Key reversal**: Gardner zscore is the best in-sample (adj R² = 0.909), but Sharpe wordcount wins OOS (RMSE = 0.771). The simpler model generalises better.

---

## 4. Full Spec Comparison — Statements (ev=1.0, representative)

Results are identical across extreme_val = 1.0–3.0 for all dictionary approaches.

### 4a. Gardner — Z-score

| Spec | Description | IS Adj R² | IS AIC | OOS RMSE | OOS vs Baseline |
|------|-------------|-----------|--------|----------|-----------------|
| Baseline | — | 0.8716 | 1219.7 | 0.8889 | — |
| 4a | Total sent (additive) | 0.8797 | 1173.5 | 0.8645 | **-2.7%** |
| 4b — sent_sd/var | Std dev / variance | 0.8716 | 1219.7 | 0.8889 | =baseline |
| **4b — consensus** | Per-topic agreement | **0.8959** | 1072.6 | **0.8200** | **-7.7%** |
| 4b — consensus×macro | Consensus × macro | 0.8875 | 1129.2 | 0.8955 | +0.7% |
| 4c | All 5 topic scores | 0.8847 | 1146.7 | 0.8557 | -3.7% |
| 4d | Topic PCA | 0.8849 | 1144.7 | 0.8527 | -4.1% |
| **4e** | **Total × all macro** | **0.9089** | **975.9** | **0.8125** | **-8.6%** |
| 4f | Matched topic × macro | 0.8944 | 1083.0 | 0.8960 | +0.8% |
| 4g | Novelty (first diff) | 0.8724 | 1218.8 | 0.9058 | +1.9% |

### 4b. Gardner — Wordcount

| Spec | IS Adj R² | IS AIC | OOS RMSE | OOS vs Baseline |
|------|-----------|--------|----------|-----------------|
| 4a | 0.8717 | 1220.4 | 0.8927 | +0.4% |
| 4b — sd/var | 0.8716 | 1219.7 | 0.8889 | =baseline |
| **4b — consensus** | **0.8857** | 1140.4 | **0.8483** | **-4.6%** |
| 4b — consensus×macro | 0.8812 | 1168.7 | 0.8872 | -0.2% |
| 4c | 0.8728 | 1217.8 | 0.8995 | +1.2% |
| 4d | 0.8729 | 1216.4 | 0.8964 | +0.8% |
| **4e** | **0.8900** | **1112.8** | **0.8618** | **-3.1%** |
| 4f | 0.8805 | 1172.9 | 0.9017 | +1.4% |
| 4g | 0.8723 | 1218.1 | 0.8968 | +0.9% |

> Note: For Gardner wordcount, the **consensus spec (0.848) beats 4e (0.862) OOS** — the opposite of the in-sample ranking. Consensus appears more robust to overfitting.

### 4c. Sharpe — Wordcount

| Spec | IS Adj R² | IS AIC | OOS RMSE | OOS vs Baseline |
|------|-----------|--------|----------|-----------------|
| 4a | 0.8720 | 1219.0 | 0.8879 | -0.1% |
| 4b — sd/var | 0.8716 | 1219.7 | 0.8889 | =baseline |
| 4b — consensus | 0.8735 | 1210.2 | 0.8823 | -0.7% |
| **4e** | **0.9088** | **976.8** | **0.7705** | **-13.3%** |
| 4f | 0.8716 | 1219.7 | 0.8889 | =baseline |
| 4g | 0.8722 | 1215.1 | 0.8877 | -0.1% |

### 4d. Sharpe — Z-score

| Spec | IS Adj R² | IS AIC | OOS RMSE | OOS vs Baseline |
|------|-----------|--------|----------|-----------------|
| 4a | 0.8805 | 1169.0 | 0.8612 | -3.1% |
| 4b — sd/var | 0.8716 | 1219.7 | 0.8889 | =baseline |
| 4b — consensus | 0.8730 | 1212.9 | 0.8847 | -0.5% |
| **4e** | **0.9069** | **991.7** | **0.7954** | **-10.5%** |
| 4f | 0.8716 | 1219.7 | 0.8889 | =baseline |
| 4g | 0.8721 | 1215.4 | 0.8904 | +0.2% |

---

## 5. Key Findings

### 5.1 Spec 4e (Total × Macro Interactions) Dominates — In-Sample and OOS
Augmenting the FAVAR with `sent_total × {vix, gdp, unemployment_gap, inflation_dev_from_target, implied_ffr}` is the best specification by both metrics for Sharpe (both norms) and Gardner zscore. The OOS gains are large: -8.6% to -13.3% RMSE vs baseline. This confirms the signal is genuine, not an in-sample artefact.

### 5.2 OOS Ranking Reverses the In-Sample Ranking
Gardner zscore wins in-sample (adj R² = 0.909, AIC = 975.9) but Sharpe wordcount wins OOS (RMSE = 0.771). Sharpe's simpler structure — a single net count (positive minus negative words) — appears more robust to overfitting than Gardner's five-topic weighted scoring. This is a material finding for the LLM comparison: in-sample fit is not a reliable guide to which model will generalise.

### 5.3 Consensus Metric is OOS-Robust for Gardner Wordcount
The Gardner wordcount consensus spec (OOS RMSE = 0.848) beats spec 4e (0.862) out-of-sample, despite 4e being the in-sample winner. For Gardner zscore, consensus (0.820) is a close second to 4e (0.813). This suggests that the agreement measure is a lower-variance signal than total-sentiment interactions — useful for the thesis argument that FOMC unanimity matters.

### 5.4 Spec 4f (Matched Topic × Macro) Hurts OOS
Despite meaningful in-sample gains (+1.3–2.3 pp adj R²), 4f is consistently worse than the baseline OOS for all combos (+0.7–1.4% RMSE). The matched interactions overfit: with 5 separate crossed terms, the model can fit in-sample noise that doesn't persist.

### 5.5 Raw Dispersion (SD, Variance) Adds Nothing In- or Out-of-Sample
Specs 4b_sent_sd and 4b_sent_var are identical to the baseline on every metric. Not surprising for dictionary approaches where dispersion is not a meaningful concept (there is no sentence-level distribution — only word counts).

### 5.6 Normalization Effect
Z-score normalization systematically improves both in-sample fit and OOS performance for Gardner. For Sharpe, wordcount is better OOS despite zscore being slightly closer in-sample (0.795 vs 0.770). The expanding z-score strips out long-run tone shifts; wordcount preserves levels. Which is better depends on whether the macro relationship is with the *level* or *deviation* of tone.

### 5.7 Results Invariant to Extreme-Val Parameter
Every metric is identical across extreme_val = 1.0–3.0. Expected: this parameter rescales LLM ±2 labels but has no effect on dictionary word counts.

---

## 6. OOS Bug (Fixed)

The original OOS RMSE values were all NaN due to a one-character bug in `utils.py:oos_rmse()`. The expanding-window loop called `m.predict(X_test)[0]`, which uses pandas label-based indexing. When the DataFrame index is non-zero (after dropna removes some rows), `[0]` raises `KeyError: 0` silently — causing every iteration to fail. Fixed to `.iloc[0]`.

---

## 7. LLM Comparison Benchmarks

| Threshold | Model | IS Adj R² | OOS RMSE |
|-----------|-------|-----------|----------|
| Best IS dictionary | Gardner zscore — 4e | 0.9089 | 0.8125 |
| Best OOS dictionary | Sharpe wordcount — 4e | 0.9088 | **0.7705** |
| Most robust (OOS/IS trade-off) | Gardner zscore — 4e | — | 0.8125 |
| Consensus benchmark | Gardner zscore — 4b_consensus | 0.8959 | 0.8200 |
| Baseline (no sentiment) | FAVAR only | 0.8716 | 0.8889 |

The LLM approach should be evaluated on both metrics. To beat dictionaries on OOS, LLM-based Spec 4e needs RMSE < 0.770.

---

## 8. Change-in-Rate Regressions (delta_rate_next)

Target: **Δeffective_rate = effective_rate(t+1) − effective_rate(t)** — the actual rate move decided at the next meeting.

### 8.1 Baseline Comparison: Level vs Change

| | Level (target_next) | Change (delta_rate_next) |
|--|--|--|
| Baseline Adj R² | 0.872 | **0.107** |
| Baseline OOS RMSE | 0.889 | **0.211** |

The baseline adj R² collapses from 0.872 to 0.107 when switching targets. **~87% of the level regression's explanatory power came from rate persistence** (the current rate predicting the next rate), not from macroeconomic information. The change regression is the harder, more economically meaningful test.

### 8.2 Best Per-Combo Results (delta target, ev=1.0)

| Combo | Best IS Spec | IS Adj R² | Best OOS Spec | OOS RMSE | OOS vs Baseline |
|-------|-------------|-----------|---------------|----------|-----------------|
| Baseline | — | 0.107 | — | 0.211 | — |
| Gardner zscore | 4e | 0.199 | **4b_consensus** | **0.208** | **-1.5%** |
| Sharpe wordcount | 4e | 0.155 | **4e** | **0.209** | **-1.1%** |
| Gardner wordcount | 4c | 0.169 | 4g (novelty) | 0.213 | +0.9% |
| Sharpe zscore | 4e | 0.122 | baseline ~ | 0.211 | ≈0% |

### 8.3 Full Spec OOS Table (delta target, ev=1.0)

Baseline OOS RMSE = **0.211066**. Improvement = lower RMSE.

| Spec | GW OOS | GZ OOS | SW OOS | SZ OOS |
|------|--------|--------|--------|--------|
| Baseline | 0.2111 | 0.2111 | 0.2111 | 0.2111 |
| 4a | 0.2156 | 0.2136 | 0.2117 | 0.2131 |
| 4b — sd/var | 0.2111 | 0.2111 | 0.2111 | 0.2111 |
| **4b — consensus** | 0.2133 | **0.2079** | 0.2116 | 0.2120 |
| 4b — consensus×macro | 0.2259 | 0.2336 | — | — |
| 4c | 0.2181 | 0.2137 | — | — |
| 4d | 0.2187 | 0.2123 | — | — |
| **4e** | 0.2224 | 0.2264 | **0.2086** | 0.2197 |
| 4f | 0.2297 | 0.2291 | 0.2111 | 0.2111 |
| 4g | 0.2128 | 0.2108 | 0.2118 | 0.2124 |

*(GW = Gardner wordcount, GZ = Gardner zscore, SW = Sharpe wordcount, SZ = Sharpe zscore)*

### 8.4 Key Findings — Change Regressions

**Finding 1 — Persistence dominates the level regression.**
The baseline macro FAVAR explains only 10.7% of the variance in rate *changes* vs 87.2% of rate *levels*. The apparent predictive power in the level regressions was mostly autocorrelation.

**Finding 2 — Sentiment adds almost nothing to change prediction.**
Most specs are OOS-*worse* than the macro-only baseline for the change target. The largest OOS improvements are only −1.5% (Gardner zscore consensus) and −1.1% (Sharpe wordcount 4e) — compared to −8.6% and −13.3% for the level target. This is not evidence that sentiment is useful for predicting the direction of the next move.

**Finding 3 — Spec 4e reverses: it helps levels but hurts change OOS.**
Spec 4e (total sentiment × macro) was the dominant in-sample and OOS winner for the level target. For the change target, it is one of the *worst* specs OOS (up to +7.2% worse RMSE for Gardner zscore). The interaction terms capture something about the steady-state *level* of policy, not its movements.

**Finding 4 — Consensus is the most robust signal across both targets.**
Gardner zscore consensus is the only spec that beats the baseline OOS in both the level regression (−7.7%) and the change regression (−1.5%). It appears to capture a genuine signal about policy stance direction that is robust to the target definition.

**Finding 5 — In-sample fit is uninformative for change regressions.**
All specs show positive in-sample adj R² gains over the baseline (e.g. Gardner zscore 4e: 0.107 → 0.199). But most are OOS-worse. The ratio of OOS gain to IS gain is much lower for changes than for levels, indicating more overfitting.

### 8.5 Implication for LLM Comparison

For the change target, the LLM benchmark to beat is:

| Metric | Best dictionary (change) | Baseline (change) |
|--------|--------------------------|-------------------|
| IS Adj R² | 0.199 (Gardner zscore 4e) | 0.107 |
| OOS RMSE | **0.208** (Gardner zscore consensus) | 0.211 |

The LLM approach should be evaluated on both level and change targets. If LLM labels genuinely capture the Fed's *decision* (not just its general tone), we expect meaningfully better OOS RMSE on the change target — which the dictionary approaches largely fail to deliver.

---

## 9. Open Issues

1. **Other document types** (minutes, speeches, press conferences) not yet in this summary. Full results exist in `dict_favar_models.csv`.
2. **CFNAI interaction (sent_macro)**: Applicable to LLM approach only; not relevant for dictionary runs.
3. **Change regression power is low** (baseline adj R² = 0.107). Consider augmenting macro regressors with lagged rate changes or market-implied moves before concluding dictionaries cannot predict changes.
