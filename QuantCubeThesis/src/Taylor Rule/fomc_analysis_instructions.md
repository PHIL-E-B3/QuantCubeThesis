# FOMC NLP Sentiment Analysis Pipeline — Execution Instructions

## Environment & File Assumptions
- **Sentence-level labels**: output JSON from the LLM labelling pipeline. Each record contains: `id`, `sentence`, `source`, `doc_type`, `date`, `top` (list), `sen` (int in {-2,-1,0,1,2}), `ten`, `hor`, `com`, `con`, `ris`, `wid`. Expected as a single merged JSON file; the pipeline also accepts a directory of per-batch files.
- **Macro data**: `data/macro_inputs/Master_Macro.csv` — stacked FOMC panel (blackout_date, meeting_date, minutes_date rows per meeting) containing macro variables: `inflation_dev_from_target`, `unemployment_gap`, `implied_ffr`, `effective_rate`, `fed_funds_rate`, `gdp`, `vix`.
- **`effective_rate`** is the merged shadow-rate series: Wu-Xia shadow rate when available (pre-2022 / ZLB episodes), otherwise `fed_funds_rate`. Use `effective_rate` as the target variable throughout unless Step 7a specifies otherwise.
- **Dictionary sentiment scores**: re-computed by Step D (see below) from the raw FOMC text in `data/raw/structured_json_*/`. Two normalization strategies are tested: `'wordcount'` (divide counts by document word count) and `'zscore'` (expanding-window z-score applied on top of word-count normalised scores). Outputs land in `Taylor Rule/nlp_output/`.
- **Dependent variable**: `effective_rate` at meeting `t+1` (one-period forward shift).
- **Canonical topic labels**: `monetary_policy`, `inflation`, `unemployment`, `economic_activity`, `financial_conditions`, `macro`, `no_topic`, `boilerplate`. Exclude `no_topic` and `boilerplate` from all sentiment aggregations.
- **Speaker categories** (for speech doc_type): `chair`, `vice_chair`, `others`. Stored as the `speaker_type` metadata field on each sentence record.

All outputs → `Taylor Rule/outputs/`. Figures saved as `.png` at 150 dpi with a consistent colour palette. All intermediate DataFrames → `Taylor Rule/outputs/intermediate/`. Random seed = 42 globally.

---

## Step D — Dictionary Re-computation (prerequisite)

Run once (or re-run whenever the raw text corpus changes) to produce the canonical dictionary scores used in Step 2 and as baselines throughout.

For each combination of:
- **Dictionaries**: Gardner et al. (2022), Sharpe et al. (2022)
- **Document types**: statements, minutes, speeches, press_conferences (prepared + qa)
- **Normalization**: `'wordcount'`, `'zscore'` (expanding z-score on word-count-normalised scores)

produce one CSV file per combination, saved to `Taylor Rule/nlp_output/`, with naming convention:
`{Dictionary}_{doc_types}_{normalization}_nlp.csv`

Each CSV contains: `date`, `meeting_date` (FOMC meeting the document aligns to), and the dictionary-specific score columns.

**Temporal alignment rules** (no look-ahead):
- `statements` → `meeting_date` row (released on decision day)
- `minutes` → `minutes_date` row (~3 weeks after meeting; lag=1)
- `speeches` → aggregate all speeches between two consecutive FOMC meeting dates into one observation per meeting window; aligned to `blackout_date` row. Retain `speaker_type` tag (`chair`, `vice_chair`, `others`) as a metadata column on each source record so speaker-mask analyses (Step 5f) can filter downstream.
- `press_conferences` → `meeting_date` row

Log any speech whose date does not fall cleanly within a meeting window to `Taylor Rule/outputs/date_alignment_warnings.txt`.

---

## Step 0 — Sentence-to-Document Aggregation

Foundation for all subsequent steps. Build `aggregate_to_document(df_sentences, agg_func='sum', rescale_extreme=2.0)` that:

1. Filters out sentences where `top` is `['no_topic']` or `['boilerplate']`.
2. Applies `rescale_sen()` (Step 6a) using `rescale_extreme` before any aggregation.
3. Groups by `(source, date, doc_type)` — the document key.
4. Computes per-document columns (all prefixed `sent_`):

   **Per-topic sentiment** (one per topic in `{monetary_policy, inflation, unemployment, economic_activity, financial_conditions, macro}`):
   - `sent_{topic}`: aggregated `sen` for sentences where `top` contains that topic.

   **Aggregate sentiment**:
   - `sent_total`: aggregated `sen` across all non-excluded sentences.
   - `sent_sd`: standard deviation of `sen` (proxy for internal disagreement).
   - `sent_var`: variance of `sen`.

   **Flag aggregations**:
   - `flag_elevated_wid`: count of sentences where `wid == 'elevated'`.
   - `flag_skew_up`: count where `ris == 'skewed_upside'`.
   - `flag_skew_down`: count where `ris == 'skewed_downside'`.
   - `flag_unconditional_forward`: binary 1 if any sentence has `com == 'unconditional'` AND `ten == 'forward'`.
   - `flag_conditional`: count where `com == 'conditional'`.
   - `speaker_types_present`: set of `speaker_type` values in the document (for speech filtering).

5. `agg_func` supports `'sum'`, `'power'` (sign(x)·|x|^1.1), `'log'` (sign(x)·log(1+|x|)).

6. Merges onto `Master_Macro` by backward as-of: each document is assigned to the nearest FOMC meeting on or after its date. Retain only rows where macro data is available.

7. Speech aggregation: pool all speeches between two consecutive FOMC meeting dates into one document-level observation per meeting. Log alignment warnings (see Step D).

---

## Step 1 — Stationarity Checks

**Input**: document-level DataFrame from Step 0 (`agg_func='sum'`).

For each `sent_` column:
1. ADF test (`statsmodels.tsa.stattools.adfuller`, regression='c', autolag='AIC').
2. KPSS test (`statsmodels.tsa.stattools.kpss`, regression='c', nlags='auto').
3. Summary DataFrame: `series`, `adf_stat`, `adf_pvalue`, `adf_stationary`, `kpss_stat`, `kpss_pvalue`, `kpss_stationary`, `conclusion` (stationary / non-stationary / conflicting).
4. Save: `outputs/stationarity_results.csv`.
5. Plot time series (6 per figure), non-stationary series in red. Save: `outputs/stationarity_plots.png`.

---

## Step 2 — Correlation with Dictionary Approaches

**Input**: `sent_total`, per-topic columns; Gardner and Sharpe scores from Step D (both normalization variants).

1. Pearson + Spearman correlations:
   - `sent_total` vs. Gardner (wordcount) and Gardner (zscore).
   - `sent_total` vs. Sharpe (wordcount) and Sharpe (zscore).
   - Gardner vs. Sharpe (benchmark).
   - Per-topic `sent_{topic}` vs. closest dictionary analogue.
2. Correlation heatmap (`seaborn.heatmap`). Save: `outputs/correlation_heatmap.png`.
3. Scatter plots with regression lines (3-panel). Save: `outputs/correlation_scatter.png`.
4. Save: `outputs/correlation_summary.csv`.

---

## Step 3 — Baseline FAVAR

**Regressors**: `['inflation_dev_from_target', 'unemployment_gap', 'implied_ffr', 'effective_rate', 'gdp', 'vix']`.
**Target**: `effective_rate` at `t+1`.
- Standardise regressors; PCA retaining ≥ 90% cumulative variance. Store factor loadings.
- OLS on PCA factors → `model_baseline`.

**Output metrics** (computed for every model in Steps 3–5):
`R², Adjusted R², AIC, BIC, F-statistic, F-p-value, N obs`
Plus for NLP vs baseline comparisons:
- **LRT** statistic and p-value.
- **OOS RMSE** via expanding-window CV (min 60% initial window, step=1 meeting).

All model results appended to `outputs/model_comparison.csv` with `model_label` column.

---

## Step 4 — NLP Regressor Tests

Add NLP regressors to baseline PCA factors. For each specification, also run all `extreme_val` variants (Step 6a: {1.0, 1.5, 2.0, 2.5, 3.0}) and the three normalization strategies for dictionary-based benchmarks (Step 6c). Report best by Adjusted R².

**4a. Total sentiment only** — add `sent_total`.

**4b. Sentiment dispersion** — add `sent_sd` and `sent_var` (separately then jointly).

**4c. Per-topic sentiment (all topics)** — add all `sent_{topic}` simultaneously. Report VIF; flag VIF > 10.

**4d. PCA on topic sentiment** — PCA on `sent_{topic}` columns (≥ 85% variance). Add factors to baseline.

**4e. Interaction: total sentiment × macro regressors** — `sent_total × vix`, `× gdp`, `× unemployment_gap`, `× inflation_dev_from_target`, `× implied_ffr`. Add jointly.

**4f. Interaction: topic sentiment × matched regressor**:
- `sent_economic_activity × gdp`
- `sent_financial_conditions × vix`
- `sent_unemployment × unemployment_gap`
- `sent_monetary_policy × implied_ffr`
- `sent_inflation × inflation_dev_from_target`
- `sent_macro × gdp` *(closest available aggregate — note this choice)*

**4g. Novelty: sentiment change between consecutive documents**:
- Speeches: `delta_sent_{topic}(t) = sent_{topic}(t) − sent_{topic}(t−1)` per meeting window.
- Statements: `delta_sent_total(t) = sent_total(t) − sent_total(t−1)`.
- Add novelty terms to baseline.

---

## Step 5 — Mask-Based Investigations

Helper: `apply_mask(df, condition, label)` — filters document-level DataFrame, reruns the **best Step 4 model** on the subsample. Report full metric set + distribution of shadow-rate direction (up/flat/down) as bar chart. Skip mask and log warning if subsample < 30 observations.

**5a. Uncertainty mask**:
- Mask A: re-aggregate using only `wid == 'elevated'` sentences (Step 0 restricted).
- Mask B: `flag_elevated_wid > median(flag_elevated_wid)`.
- Compare F-stat vs. full sample. Plot conditional shadow-rate direction.

**5b. Tail risk mask**:
- `|sen| == 2` sentences only.
- Documents where `flag_skew_up > 0`.
- Documents where `flag_skew_down > 0`.

**5c. Tense mask**:
- Mask 1: `ten == 'forward'` only.
- Mask 2: `ten in ['forward', 'present']`.
- Compare predictability between masks and vs. full-sentence baseline.

**5d. Commitment / Odyssean guidance mask**:
- Mask 1: `flag_unconditional_forward == 1` (Odyssean guidance present).
- Mask 2: `flag_unconditional_forward == 0` (purely Delphic).
- Mask 3: `flag_conditional > 0`.
- Repeat with `doc_type` filters: (a) all, (b) minutes only, (c) external communication (statements + press_conference_prepared).

**5e. Shadow rate direction mask** *(look-ahead — flag with `[LOOKAHEAD]` in all output)*:
- Mask Up: `effective_rate(t+1) > effective_rate(t)`.
- Mask Down: `effective_rate(t+1) < effective_rate(t)`.

**5f. Speaker mask** *(speeches only)*:
- Mask Chair-only: aggregate speeches filtering to `speaker_type == 'chair'`.
- Mask Chair + Vice Chair: `speaker_type in ['chair', 'vice_chair']`.
- Mask All speakers (baseline).
- Hypothesis: test whether excluding non-chair speeches loses signal (lower Adj R²) or removes noise (higher Adj R²). Report Adj R², F-stat, and OOS RMSE for each mask. Plot coefficient of `sent_total` across the three masks with 95% CI.

---

## Step 6 — Scaling & Normalisation Exploration

**6a. Extreme sentiment rescaling (`|2|` value)**

`rescale_sen(df_sentences, extreme_val)` replaces `sen == ±2` with `±extreme_val`. Test `extreme_val ∈ {1.0, 1.5, 2.0, 2.5, 3.0}`. For each, re-run Step 0 (sum only) then the two best Step 4 models. Plot Adj R² vs. `extreme_val`. Save: `outputs/scaling_extreme_val.png`.

**6b. Aggregation function**

Three variants: linear (`f(x) = x`), power (`sign(x)·|x|^1.1`), log (`sign(x)·log(1+|x|)`). Re-run Step 0 and two best Step 4 models per variant. Save comparison table: `outputs/aggregation_func_comparison.csv`.

**6c. Normalisation strategy** *(new)*

For dictionary-based scores (Gardner and Sharpe), compare three normalisation strategies:
- `'wordcount'`: divide counts by document word count.
- `'zscore'`: expanding-window z-score on word-count-normalised scores.
- `'none'`: raw counts.

Re-run the two best Step 4 models substituting each normalisation variant of the dictionary scores. Report Adj R², AIC, BIC in a table. Save: `outputs/normalization_comparison.csv`.

---

## Step 7 — Robustness Checks

**7a. Swap DV**: replace `effective_rate(t+1)` with `fed_funds_rate(t+1)`. Re-run model 4f. Compare.

**7b. Rolling window stability**: re-run best Step 4 model with 5-year rolling window. Plot key NLP coefficient(s) + 95% CI over time. Save: `outputs/rolling_coefficients.png`.

**7c. Leave-one-out by doc_type**: exclude one doc_type at a time. Report Adj R² per exclusion.

**7d. Placebo test**: shuffle `sent_total` (seed=42, 500 reps). Record null Adj R² distribution. Empirical p-value. Plot histogram with observed value marked. Save: `outputs/placebo_test.png`.

---

## Output Summary

After all steps, generate `outputs/RESULTS_SUMMARY.md`:
1. Table of all model specifications: R², Adj R², AIC, BIC, F-stat, OOS RMSE.
2. Best-performing model overall.
3. Best-performing mask.
4. Optimal `extreme_val`, aggregation function, and normalisation strategy.
5. Speaker mask finding: signal gain/loss from excluding non-chair speeches.
6. Key findings in 3–5 bullets per step.

---

## Implementation Notes

- Libraries: `pandas`, `numpy`, `statsmodels`, `sklearn`, `scipy`, `seaborn`, `matplotlib`.
- HC3 heteroskedasticity-robust standard errors on all regressions.
- Suppress statsmodels convergence warnings; log to `outputs/warnings.log`.
- Each Step = one function, callable independently with the Step 0 DataFrame as primary input.
- Step 4 must return the best model object (by Adj R²) so Steps 5, 6, 7 can reuse it without re-identifying it.
- If a mask produces < 30 observations, skip and log to `outputs/warnings.log`.
