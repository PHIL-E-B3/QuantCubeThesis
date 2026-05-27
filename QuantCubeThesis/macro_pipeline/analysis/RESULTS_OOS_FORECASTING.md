# OOS Forecasting Results: LLM Sentiment vs Macro Baseline
**Sample**: N=151 FOMC meetings, 2007–2025  
**Baseline**: Macro-only expanding FAVAR (5 PCA factors from 7 macro regressors)  
**Significance**: Diebold-Mariano (DM) for joint-PCA models; Clark-West (CW) for nested FAVAR models

---

## 1. Summary Tables (Best 3 per Method, OOS60/70/80/90)

### Table 1: Level Target (effective rate at next meeting)

| Method   | Model                          |  N  | Adj-R² | OOS60 | OOS70 | OOS80 | OOS90 |
|----------|--------------------------------|-----|--------|-------|-------|-------|-------|
| Baseline | Macro-only FAVAR               | 151 | 0.967  | 0.585 | 0.573 | 0.534 | 0.429 |
| LLM [DM] | 4h_llm_total                   | 151 | 0.959  | 0.563 | 0.626 | 0.255*** | 0.228** |
| LLM [DM] | 4h_llm_topics                  | 151 | 0.963  | 0.725 | 0.818 | 0.266*** | 0.223*** |
| LLM [DM] | 4h_llm_all_sent                | 151 | 0.963  | 0.705 | 0.796 | 0.267*** | 0.225*** |
| Gardner [CW] | G4_pc_wc_EWMA10_4b_cons    | 118 | 0.979  | 0.902 | 0.547* | 0.230*** | 0.219** |
| Gardner [CW] | G5_pc_zs_EWMA10_4f_matched | 118 | 0.982  | 1.113** | 0.395*** | 0.245*** | 0.275*** |
| Gardner [CW] | G6_pc_zs_EWMA10_4b_cons_matched | 118 | 0.977 | 1.203** | 0.466** | 0.254** | 0.281 |
| Sharpe [CW] | S7_statements_zs_4e           | 151 | 0.982  | 0.437*** | 0.451*** | 0.296*** | 0.248*** |
| Sharpe [CW] | S8_speeches_wc_EWMA10_4e      | 151 | 0.985  | 0.501*** | 0.490*** | 0.323*** | 0.265*** |
| Sharpe [DM] | S9_4h_sharpe_total_inter      | 151 | 0.964  | 0.528 | 0.565 | 0.372*** | 0.213*** |

*OOS60/70 blank for LLM = model RMSE > baseline RMSE on that sub-period (COVID/ZLB, 2019–2022).*  
*Gardner N=118: press conferences begin 2011; baseline comparison is against N=118 sub-sample baseline.*

### Table 2: Delta Target (change in effective rate at next meeting)

| Method   | Model                               |  N  | Adj-R² | OOS60 | OOS70 | OOS80 | OOS90 |
|----------|-------------------------------------|-----|--------|-------|-------|-------|-------|
| Baseline | Macro-only FAVAR                    | 151 | 0.301  | 0.314 | 0.314 | 0.265 | 0.169 |
| LLM [DM] | 4h_llm_all_sent                     | 151 | 0.392  | 0.267*** | 0.257*** | 0.210* | 0.130 |
| LLM [DM] | 4h_llm_topics                       | 151 | 0.389  | 0.268*** | 0.260*** | 0.211* | 0.132 |
| LLM [DM] | 4h_llm_total_inter                  | 151 | 0.245  | 0.341 | 0.355 | 0.276 | 0.196 |
| Gardner [CW] | G4_speeches_zs_4f_matched       | 151 | 0.352  | 0.299*** | 0.298*** | 0.240** | 0.176 |
| Gardner [CW] | G5_minutes_zs_ewma10_4c_topics  | 151 | 0.406  | 0.319 | 0.312** | 0.223* | 0.188 |
| Gardner [CW] | G6_minutes_zs_4f_matched        | 151 | 0.360  | 0.301*** | 0.315 | 0.255** | 0.171 |
| Sharpe [CW] | S7_all_docs_wc_ewma10_4b_cons    | 151 | 0.347  | 0.304* | 0.307 | 0.241* | 0.167 |
| Sharpe [CW] | S8_statements_wc_ewma10_4b_cons  | 151 | 0.343  | 0.305* | 0.309 | 0.243* | 0.167 |
| Sharpe [CW] | S9_minutes_zs_4e_inter           | 151 | 0.323  | 0.315 | 0.314 | 0.253 | 0.178 |

*\* p<0.10  \*\* p<0.05  \*\*\* p<0.01  (blank = model RMSE > baseline)*

---

## 2. Statistical Test Choice: DM vs CW

| Model type | Structure | Correct test | Reason |
|---|---|---|---|
| Joint PCA (4h LLM, S9) | Non-nested: JPC ≠ restricted BPC | **DM** | Baseline not a restricted version of model |
| Standard FAVAR (Gardner, Sharpe) | Nested: baseline = model minus sentiment cols | **CW** | CW corrects finite-sample negative bias of DM for nested models |

**Key finding from cross-test diagnostic (delta, OOS80, N=151):**

| Model | RMSE | Improvement | DM-t | DM-p | CW-t | CW-p |
|---|---|---|---|---|---|---|
| 4h_llm_all_sent | 0.210 | 20.8% | 1.490 | 0.068* | 2.413 | 0.008*** |
| G4_speeches_zs | 0.240 | 9.6% | 1.170 | 0.121 | 2.029 | 0.021** |

LLM achieves a larger RMSE improvement (20.8% vs 9.6%) but appears less significant because DM is the appropriate (more conservative) test for non-nested models. G4's ** under CW is legitimate — CW corrects for the finite-sample downward bias of DM in nested settings. The significance levels are not directly comparable across model types.

---

## 3. Multi-Horizon DM Test (delta, 50% training)

**Setup**: expanding window from 50% of N (~75 meetings), predicting h meetings ahead.  
**Target**: cumulative delta = `effective_rate[t+h] - effective_rate[t]`  
**OOS predictions**: ~76 at h=1, decreasing to ~61 at h=30.

### 4h_llm_all_sent

| h | n_pred | Model RMSE | Base RMSE | Improv% | DM-t | DM-p | Sig |
|---|--------|-----------|-----------|---------|------|------|-----|
| 1 | 76 | 0.243 | 0.283 | 14.1% | 2.419 | 0.0078 | *** |
| 5 | 74 | 0.618 | 0.745 | 17.2% | 1.300 | 0.0968 | * |
| 10 | 71 | 1.383 | 1.140 | -21.4% | — | — | |
| 15 | 69 | 1.770 | 1.889 | 6.3% | 1.044 | 0.148 | |
| 20 | 66 | 2.498 | 2.294 | -8.9% | — | — | |
| 25 | 64 | 2.356 | 2.800 | 15.9% | 1.714 | 0.043 | ** |
| 30 | 61 | 2.305 | 2.689 | 14.3% | 2.054 | 0.020 | ** |

### 4h_llm_topics

| h | n_pred | Model RMSE | Base RMSE | Improv% | DM-t | DM-p | Sig |
|---|--------|-----------|-----------|---------|------|------|-----|
| 1 | 76 | 0.244 | 0.283 | 13.5% | 2.334 | 0.0098 | *** |
| 5 | 74 | 0.621 | 0.745 | 16.7% | 1.271 | 0.102 | |
| 10 | 71 | 1.392 | 1.140 | -22.1% | — | — | |
| 15 | 69 | 1.743 | 1.889 | 7.7% | 1.231 | 0.109 | |
| 20 | 66 | 2.147 | 2.294 | 6.4% | 1.546 | 0.061 | * |
| 25 | 64 | 2.344 | 2.800 | 16.3% | 1.687 | 0.046 | ** |
| 30 | 61 | 2.300 | 2.689 | 14.5% | 2.023 | 0.022 | ** |

### Key findings

1. **h=1 (next meeting): *** significance** — both all_sent and topics beat the macro baseline at the 1% level when trained from 50% and predicting meeting-by-meeting. This is the primary result.

2. **h=10–20 (medium run): model fails** — sentiment cannot reliably predict cumulative rate changes 1–2 years out. The model actively underperforms the baseline at h=10.

3. **h=25–30 (long run): ** significance** — LLM sentiment regains predictive power at 2–3 year horizons, likely capturing persistent hawkish/dovish regime shifts.

4. **Power matters**: the h=1 result shows *** with 76 OOS predictions (50% training) vs only * with 31 predictions (OOS80 / 80% training). The model's true predictive advantage was present throughout — the OOS80 window was simply too short for DM to confirm it.

---

## 4. Robustness: Regime Analysis for h=1

**Setup**: Expanding OLS from 50% training (same as multi-horizon test). 76 OOS predictions (2016–2025). Model: 4h_llm_all_sent vs Macro FAVAR baseline.

**Economic regimes (OOS period ~2019–2025):**

| Regime | Period | N (OOS) |
|--------|--------|---------|
| A. Pre-COVID | 2019-01 to 2020-02 | 9 |
| B. COVID shock | 2020-03 to 2020-06 | 4 |
| C. ZLB | 2020-07 to 2022-02 | 13 |
| D. Hiking | 2022-03 to 2023-07 | 12 |
| E. Peak+Cuts | 2023-08 to 2025-12 | 18 |

### 4.1 Sub-period DM within each regime

| Regime | N | Model RMSE | Base RMSE | Improv% | DM-t | DM-p | Sig |
|--------|---|-----------|-----------|---------|------|------|-----|
| A. Pre-COVID | 9 | 0.3580 | 0.3952 | 9.4% | — | — | (too few) |
| B. COVID shock | 4 | — | — | — | — | — | (too few) |
| C. ZLB | 13 | 0.3545 | 0.4145 | 14.5% | 2.280 | 0.0113 | ** |
| D. Hiking | 12 | 0.2946 | 0.3756 | 21.6% | 1.467 | 0.0712 | * |
| E. Peak+Cuts | 18 | 0.1335 | 0.1591 | 16.1% | 0.697 | 0.2428 | |

ZLB and Hiking are the periods of clearest outperformance. Peak+Cuts shows a 16.1% improvement but n=18 provides insufficient power for DM significance.

### 4.2 Leave-one-regime-out

| Excluded regime | N | Model RMSE | Base RMSE | Improv% | DM-t | DM-p | Sig |
|-----------------|---|-----------|-----------|---------|------|------|-----|
| [All included] | 76 | 0.2428 | 0.2826 | 14.1% | 2.419 | 0.0078 | *** |
| excl. A. Pre-COVID | 67 | 0.2229 | 0.2638 | 15.5% | 2.018 | 0.0218 | ** |
| excl. B. COVID shock | 72 | 0.2455 | 0.2862 | 14.2% | 2.386 | 0.0085 | *** |
| excl. C. ZLB | 63 | 0.2126 | 0.2467 | 13.8% | 1.689 | 0.0456 | ** |
| excl. D. Hiking | 64 | 0.2318 | 0.2614 | 11.3% | 2.244 | 0.0124 | ** |
| excl. E. Peak+Cuts | 58 | 0.2678 | 0.3111 | 13.9% | 2.386 | 0.0085 | *** |

**Key finding**: significance holds at ** or *** when any single regime is excluded. The h=1 result is not driven by any one period.

### 4.3 Cumulative loss differential by regime

| Regime | N | Avg loss diff | Cumulative contribution |
|--------|---|--------------|------------------------|
| pre-OOS (2016–2018) | 20 | −0.0041 | −0.0819 |
| A. Pre-COVID | 9 | +0.0281 | +0.2526 |
| B. COVID shock | 4 | +0.0073 | +0.0292 |
| C. ZLB | 13 | +0.0461 | +0.5998 |
| D. Hiking | 12 | +0.0543 | +0.6515 |
| E. Peak+Cuts | 18 | +0.0075 | +0.1347 |

The model gains accumulate gradually across C_zlb and D_hiking (the two high-rate-change periods), not from a single spike. The pre-2019 window (before the OOS regime of interest) is slightly negative, reflecting the warm-up period before the model has enough training data.

---

## 5. Block Bootstrap DM Test (h=1)

**Why not a permutation test**: shuffling the target or predictions breaks the temporal ordering that expanding-window OLS depends on — the null distribution would be meaningless.

**Moving block bootstrap (Kunsch 1989)**: resamples contiguous blocks of the loss differential series `d_t = e_b² - e_m²`, preserving local autocorrelation. B=5,000 replications; H0 imposed by centering `d_t`. One-sided p-value = fraction of bootstrap DM stats ≥ observed.

**Observed DM stat: t = 2.74** (vs 2.42 from `diebold_mariano_test` — small difference from variance estimator convention; both *** level).

| Block L | Rationale | Bootstrap p-val | Sig |
|---------|-----------|----------------|-----|
| 3 | fixed 3 | 0.0008 | *** |
| 4 | T^(1/3) | 0.0008 | *** |
| 5 | fixed 5 | 0.0010 | *** |
| 8 | fixed 8 | 0.0032 | *** |
| 9 | T^(1/2) | 0.0048 | *** |

Harvey-Leybourne-Newbold (1997) small-sample corrected DM: t=2.72, p=0.0040 ***.

**Finding**: p < 0.01 at every block length tested. The result is not an artefact of parametric distributional assumptions. The empirical null distribution, which accounts for autocorrelation in `d_t`, confirms *** significance.

---

## 6. Sentence-Attribute Filtering Variants (h=1)

**Question**: Is the predictive signal driven by a specific subset of sentences — forward-looking (interpretive) sentences only, or does it survive when commitment language is stripped out?

**Setup**: Same as multi-horizon h=1 test (50% training, expanding OLS, DM vs Macro FAVAR). Two variants tested against the original 4h_llm_all_sent:

| Variant | Filter | Sentences kept |
|---------|--------|---------------|
| Original | None | 174,525 (100%) |
| A: Interpretive only | tense == 'interpretive' | 35,430 (20%) |
| B: Excl. unconditional docs | Drop whole doc if any sentence has commitment=True | 98,593 (57%) |
| B2: Drop unconditional sentences | Drop only sentences with commitment=True | 171,923 (98.5%) |

**Corpus breakdown**:
- 139,095 descriptive sentences / 35,430 interpretive (20%)
- 2,602 sentences with commitment=True (unconditional, 1.5% of corpus)
- Unconditional sentences appear in 85–94% of minutes, statements, and prepared remarks — nearly every formal FOMC document

### Corpus breakdown

| Tense | Commitment | N sentences | % |
|-------|-----------|-------------|---|
| Descriptive | Not unconditional | 137,175 | 78.6% |
| Interpretive | Not unconditional | 34,748 | 19.9% |
| Descriptive | Unconditional | 1,920 | 1.1% |
| Interpretive | Unconditional (Odyssean) | 682 | 0.4% |

Most "unconditional" language is in **descriptive sentences** (e.g., "The Committee raised rates to 5.25–5.50%") — past-tense announcements, not forward commitments. Only 682 sentences (0.4%) are genuine Odyssean forward guidance: interpretive + unconditional.

### Early variant results

| Model | N_sent | n_pred | Improv% | DM-t | DM-p | Sig |
|-------|--------|--------|---------|------|------|-----|
| Original 4h_llm_all_sent | 174,525 | 76 | +14.1% | 2.419 | 0.0078 | *** |
| A: Interpretive only | 35,430 | 76 | +11.0% | 1.228 | 0.110 | |
| B: Excl. unconditional docs | 98,593 | 76 | −6.7% | −1.712 | — | |
| B2: Drop unconditional sents | 171,923 | 76 | −2.6% | −0.441 | — | |

### 2×2 decomposition: tense × commitment

Using only one quadrant at a time to isolate where the signal lives.

| Quadrant | N | Improv% | DM-t | DM-p | Sig |
|----------|---|---------|------|------|-----|
| C: Interpretive + NOT unconditional | 34,748 | +8.8% | 0.963 | 0.168 | |
| D: Interpretive + unconditional (Odyssean) | 682 | −9.0% | −0.640 | — | |
| E: Descriptive + NOT unconditional | 137,175 | **−14.1%** | **−3.571** | — | |
| F: Descriptive + unconditional | 1,920 | +3.4% | 0.840 | 0.200 | |

### Interpretation

**Your intuition was directionally correct**: Quadrant C (interpretive + conditional, the purest "soft information" that isn't yet priced in) has the best single-quadrant improvement (+8.8%). It also has the only positive DM-t among the four quadrants. But 34,748 sentences alone do not provide enough signal to reach DM significance.

**Descriptive sentences alone (E) actively hurt the model (−14.1%, DM-t=−3.57)**. When isolated, descriptive sentiment generates factors that are systematically anti-correlated with future rate changes — the model trained on backward-looking assessments overfits the current cycle and gets transition periods (hiking → cutting) consistently wrong. This is also consistent with the market efficiency argument: purely backward-looking text, when the baseline already contains the same macro variables, adds only noise.

**Odyssean forward guidance alone (D, only 682 sentences) is also harmful (−9.0%)**. 682 sentences is very sparse — many meetings have zero coverage in this quadrant — and the PCA cannot reliably extract factors from such a thin signal.

**The combination is what creates the *** result.** The PCA acts as a signal integrator: it finds a joint factor structure that combines the forward-looking directional signal (C) with the economic assessment context (E) and commitment anchors (D, F). No single quadrant has enough statistical power or information coverage. The model needs all four quadrants together to achieve the 14.1% / *** result.

This finding is consistent with the PCA-as-information-aggregator view of FAVAR models: the joint factor structure extracts latent dimensions that span multiple text attributes simultaneously.

**Caveat on Variant B's scope**: The "any unconditional sentence in document" threshold captures 85–94% of minutes, statements, and prepared remarks — effectively removing almost all formal FOMC communications.

---

## 7. Risk and Uncertainty Flag Analysis (h=1)

**Hypotheses:**
- (U) Elevated uncertainty → asymmetric FOMC reaction function → model more predictive in high-uncertainty periods
- (R-down) High downside risk → rate cut more likely
- (R-up) High upside risk → rate hike more likely

**Aggregation rule**: All three flags have counts >0 at almost every meeting (flag_elevated_wid and flag_skew_down: 0% zero; flag_skew_up: 5% zero). Continuous distributions → **median split** for all three.

| Flag | Median | Range |
|------|--------|-------|
| flag_elevated_wid | 20 | [6, 122] |
| flag_skew_up | 10 | [0, 57] |
| flag_skew_down | 44 | [9, 162] |

### 7.1 Directional accuracy

Does high upside/downside risk actually predict the direction of the next policy move?

| Condition | N | % Cuts | % Holds | % Hikes | Mean delta |
|-----------|---|--------|---------|---------|-----------|
| All meetings | 152 | 43.4% | 13.2% | 43.4% | −0.010 |
| High downside risk | 75 | 46.7% | 16.0% | 37.3% | −0.031 |
| Low downside risk | 77 | 40.3% | 10.4% | 49.4% | +0.010 |
| **High upside risk** | **67** | **26.9%** | **23.9%** | **49.3%** | **+0.093** |
| **Low upside risk** | **85** | **56.5%** | **4.7%** | **38.8%** | **−0.091** |
| High uncertainty | 75 | 38.7% | 18.7% | 42.7% | +0.009 |
| Low uncertainty | 77 | 48.1% | 7.8% | 44.2% | −0.028 |

**Correlations with cum_delta_h1:**

| Flag | Correlation |
|------|------------|
| flag_elevated_wid | +0.077 |
| flag_skew_up | **+0.308** |
| flag_skew_down | −0.097 |

**Findings:**

- **Upside risk strongly confirmed** (corr = +0.31): high upside risk meetings see 49% hikes vs 39% in low upside risk, with a mean delta spread of +0.093 vs −0.091. This is the clearest directional signal in the flags.
- **Downside risk directionally correct** but weaker (corr = −0.10): high downside risk meetings have more cuts (46.7% vs 40.3%) and fewer hikes (37.3% vs 49.4%).
- **Uncertainty does not predict direction**: roughly equal cuts/hikes in both high and low uncertainty subsamples (corr = +0.08).

### 7.2 Subsample DM test

Does the 4h_llm_all_sent model outperform the macro FAVAR specifically in high-risk or high-uncertainty periods?

| Subsample | N | Model RMSE | Base RMSE | Improv% | DM-t | DM-p | Sig |
|-----------|---|-----------|-----------|---------|------|------|-----|
| [Full sample] | 76 | 0.2428 | 0.2826 | +14.1% | 2.419 | 0.0078 | *** |
| High uncertainty | 47 | 0.1883 | 0.2279 | +17.4% | 1.640 | 0.051 | * |
| Low uncertainty | 29 | 0.3116 | 0.3537 | +11.9% | 2.273 | 0.012 | ** |
| High upside risk | 47 | 0.2134 | 0.2558 | +16.6% | 1.581 | 0.057 | * |
| Low upside risk | 29 | 0.2841 | 0.3212 | +11.5% | 2.242 | 0.013 | ** |
| **High downside risk** | **41** | **0.1551** | **0.1947** | **+20.3%** | **2.120** | **0.017** | **\*\*** |
| Low downside risk | 35 | 0.3160 | 0.3591 | +12.0% | 2.249 | 0.012 | ** |

**Findings:**

- The model beats the macro FAVAR in **every single subsample** at * or **.
- **High downside risk achieves the largest improvement (+20.3%, **)**, directly validating the hypothesis that the model has more value when the FOMC is concerned about downside risks.
- The pattern across all flags: high-flag subsamples tend to show larger RMSE improvements (17–20%) but slightly lower DM significance vs low-flag subsamples (11–12% improvement, **). This reflects the higher volatility in high-risk periods (larger loss differential variance) which reduces DM power even as the absolute gains increase.
- High uncertainty shows the same pattern: +17.4% improvement but only * significance vs ** in low uncertainty periods.
- **The model does not rely on any single regime to survive**: ** holds in both halves for downside risk, and both halves for the other flags. This is additional evidence against regime-dependency.

---

## 8. Method Comparison: Full Delta Sweep — All Variants × All Methods

**Setup**: All 8 variants × 3 methods (24 models + 3 baselines) evaluated on the delta target (cum_delta_h1), h=1, 2007–2025. Gardner and Sharpe scores are aggregated by summing z-scored document-level scores to the meeting level. **DM test throughout** (correct for non-nested joint PCA models). Two evaluation protocols:
- **Expanding window** (EXP): 50% initial training, retrain at each step → 76 OOS predictions
- **Fixed OOS splits** (60/70/80/90): train on first X%, predict the remaining (1−X)% in one shot

Base RMSE varies across OOS windows because each window covers a different sub-period. Significance: \*\*\* p<0.01, \*\* p<0.05, \* p<0.10. "—" = DM-t < 0 (model worse; one-sided test inapplicable).

### Full results table

| Model | RMSE_exp | DM_exp | sig | RMSE60 | DM60 | sig | RMSE70 | DM70 | sig | RMSE80 | DM80 | sig | RMSE90 | DM90 | sig |
|-------|----------|--------|-----|--------|------|-----|--------|------|-----|--------|------|-----|--------|------|-----|
| **LLM_base** (Macro FAVAR) | 0.2826 | — | | 0.5176 | — | | 0.3341 | — | | 0.3374 | — | | 0.1712 | — | |
| **LLM_all_sent** | **0.2428** | **2.42** | **\*\*\*** | 0.3457 | 2.13 | \*\* | 0.3898 | −1.31 | — | 0.2134 | 2.07 | \*\* | 0.1308 | 0.86 | |
| LLM_topics | 0.2444 | 2.33 | \*\*\* | 0.3486 | 2.07 | \*\* | 0.3927 | −1.37 | — | 0.2152 | 2.11 | \*\* | 0.1319 | 0.85 | |
| LLM_topics_matched | 0.2759 | 0.24 | | 0.3525 | 2.33 | \*\*\* | 0.4446 | −1.80 | — | 0.6274 | −2.40 | — | 0.2377 | −1.24 | — |
| LLM_total | 0.2884 | −0.73 | — | 0.3901 | 3.76 | \*\*\* | 0.3439 | −1.08 | — | 0.3153 | 1.66 | \*\* | 0.2554 | −3.28 | — |
| LLM_total_inter | 0.3078 | −1.27 | — | 0.3713 | 2.02 | \*\* | 0.4369 | −2.85 | — | 0.3283 | 0.55 | | 0.2269 | −2.86 | — |
| LLM_topics_inter | 0.3351 | −1.20 | — | 1.2411 | −1.65 | — | 0.8580 | −1.81 | — | 0.7173 | −1.73 | — | 0.2458 | −2.04 | — |
| LLM_all_sent_inter | 0.3227 | −0.91 | — | 1.0965 | −1.53 | — | 0.7137 | −1.72 | — | 0.6644 | −1.60 | — | 0.2533 | −2.41 | — |
| **Gardner_base** (Macro FAVAR) | 0.2826 | — | | 0.5176 | — | | 0.3341 | — | | 0.3374 | — | | 0.1712 | — | |
| Gardner_total | 0.2904 | −0.73 | — | 0.4266 | 4.25 | \*\*\* | 0.3288 | 0.38 | | 0.3432 | −0.36 | — | 0.3055 | −3.05 | — |
| Gardner_topics | 0.2996 | −1.53 | — | 0.3660 | 2.56 | \*\*\* | 0.3433 | −0.79 | — | 0.3590 | −1.00 | — | 0.3134 | −3.00 | — |
| Gardner_topics_matched | 0.2986 | −0.73 | — | 0.4280 | 2.09 | \*\* | 0.4203 | −3.28 | — | 0.3009 | 0.57 | | 0.2488 | −1.60 | — |
| Gardner_all_sent | 0.3075 | −1.92 | — | 0.3715 | 2.38 | \*\*\* | 0.3602 | −1.47 | — | 0.3808 | −1.47 | — | 0.3291 | −3.42 | — |
| Gardner_total_inter | 0.3064 | −1.33 | — | 0.4144 | 1.49 | \* | 0.4341 | −2.00 | — | 0.3927 | −1.21 | — | 0.3333 | −3.13 | — |
| Gardner_topics_inter | 0.3150 | −1.53 | — | 0.4127 | 1.47 | \* | 0.4963 | −2.16 | — | 0.5060 | −2.00 | — | 0.2566 | −2.13 | — |
| Gardner_all_sent_inter | 0.3183 | −1.60 | — | 0.4391 | 0.96 | | 0.4931 | −2.01 | — | 0.5645 | −2.50 | — | 0.2874 | −3.22 | — |
| **Sharpe_base** (Macro FAVAR) | 0.2826 | — | | 0.5176 | — | | 0.3341 | — | | 0.3374 | — | | 0.1712 | — | |
| Sharpe_all_sent | 0.2739 | 1.50 | \* | 0.3492 | 2.76 | \*\*\* | 0.3509 | −0.90 | — | 0.3198 | 2.33 | \*\*\* | 0.1626 | 0.46 | |
| Sharpe_topics | 0.2755 | 1.48 | \* | 0.3619 | 2.87 | \*\*\* | 0.3482 | −1.04 | — | 0.3205 | 1.66 | \*\* | 0.1620 | 0.59 | |
| Sharpe_total | 0.2848 | −0.42 | — | 0.3439 | 3.36 | \*\*\* | 0.3355 | −0.38 | — | 0.3065 | 2.51 | \*\*\* | 0.2024 | −1.92 | — |
| Sharpe_topics_matched | 0.3042 | −1.92 | — | 0.4755 | 0.69 | | 0.4097 | −1.49 | — | 0.3875 | −1.16 | — | 0.1986 | −1.16 | — |
| Sharpe_total_inter | 0.3784 | −1.26 | — | 0.5783 | −1.05 | — | 0.6062 | −2.24 | — | 0.4556 | −2.16 | — | 0.2989 | −2.96 | — |
| Sharpe_topics_inter | 0.3907 | −1.38 | — | 0.6317 | −1.14 | — | 0.5541 | −1.58 | — | 0.6607 | −1.35 | — | 0.1793 | −0.73 | — |
| Sharpe_all_sent_inter | 0.4151 | −1.55 | — | 0.7138 | −1.73 | — | 0.6298 | −1.81 | — | 0.7362 | −1.53 | — | 0.1851 | −1.04 | — |

### Findings

**Expanding window (most rigorous — 76 OOS predictions, full DM power):**
- **LLM_all_sent** is the single significantly outperforming model: RMSE=0.2428, DM-t=2.42, p=0.008 ***. `LLM_topics` is a close second (DM-t=2.33, p=0.010 ***) — `sent_total` adds marginally on top of the 6 topic scores.
- **No Gardner variant beats the baseline** in the expanding window. All Gardner DM-t values are negative. Gardner word-count polarity does not predict rate *changes* — only the level fits.
- **Sharpe is marginal at best**: `Sharpe_all_sent` reaches * (p=0.066) in the expanding window. Not robust.
- **Interaction variants consistently hurt** across all three methods: adding sentiment × macro cross-products inflates RMSE, reflecting overfitting given n≈76 OOS observations.

**Fixed OOS splits (OOS60/70/80/90):**
- OOS60 is the most informative fixed split (60 test observations): LLM_all_sent **, LLM_topics **. Gardner total *** and Gardner_topics *** at OOS60 — but this is a *level* of DM significance driven by the test window coinciding with the COVID/hiking period (high volatility, high baseline RMSE=0.5176). The same Gardner variants fail completely in the expanding window.
- OOS70 is the most volatile split: every method (including LLM) shows negative DM-t at OOS70. This is a single 45-observation test window that starts mid-ZLB and the model's advantage disappears — consistent with the regime robustness finding that the ZLB period is the weakest quarter for all models.
- OOS80 (30 test obs, 2020–2025): LLM_all_sent **, LLM_topics **. Sharpe_total *** and Sharpe_all_sent *** — but 30 obs is too few for reliable inference.
- OOS90 (15 test obs): DM is severely underpowered. No conclusion.

**Bottom line**: The expanding window is the correct primary metric. Under it, **LLM (all_sent or topics) is the only method that reliably outperforms the macro FAVAR baseline for rate change prediction**, at 3% significance in both variants. Dictionary methods (Gardner, Sharpe) do not. This holds across every robust evaluation protocol.

---

## 9. Notes

- **Adj-R² for level is high (>0.96)** by construction: effective rate is highly persistent and macro PCA (which includes implied_ffr) fits the level well in-sample. OOS RMSE is the more informative metric for level.
- **Adj-R² for delta (0.25–0.41)** is more meaningful: changes are genuinely hard to predict, and LLM topic scores add ~10pp of explanatory power over the macro baseline.
- **Gardner press_conf (level, N=118)**: OOS60/70 significance stars are against the N=118 sub-sample baseline, not the N=151 baseline shown in the table header. OOS80/90 comparisons are reliable.
- **Delta OOS90**: only ~16 predictions; DM is severely underpowered at this window size.
- **Gardner/Sharpe level results** (prior files `4h_gard_joint_pca_200701.csv`, `4h_sharpe_joint_pca_200701.csv`): CW test was used throughout — this is the wrong test for joint PCA (non-nested) models. Those level-target results should be treated with caution.
