# Prompt Engineering Log — FOMC Sentence Classification

## Overview

This document records the iterative prompt engineering process for classifying FOMC (Federal Open Market Committee) sentences using `unsloth/Meta-Llama-3.1-8B-Instruct` via vLLM. The goal was to maximise macro-F1 on a 7-field annotation schema applied to 591 held-out sentences from FOMC minutes, press conference transcripts, speeches, and statements.

---

## Task

Each sentence is classified into 7 fields:

| Field | Type | Values |
|-------|------|--------|
| `topic` | multi-label array | inflation, labor_market, economic_activity, macro, financial_conditions, monetary_policy, boilerplate, no_topic |
| `tense` | string | descriptive, interpretive |
| `sentiment` | string | strongly_hawkish, hawkish, neutral, dovish, strongly_dovish, na |
| `horizon` | boolean | true / false |
| `commitment` | string | unconditional, conditional, none |
| `risk` | string | skewed_upside, skewed_downside, symmetric, na |
| `width` | string | elevated, contested, none |

---

## Evaluation Setup

- **Validation set**: `eval_labelled_merged_corrected.json` (591 sentences, human-corrected labels)
- **Training seed**: `all_labelled_sentences.json` (632 sentences) + `final_extreme_seed.json` (128 sentences, oversampled ±2 sentiment and rare risk/width labels) = 760 total
- **Primary metric**: Macro F1 averaged over topic, sentiment, risk, width (equal weight)
- **Secondary metric**: SumF1* = same but sentiment excludes `na` (more informative when boilerplate is absent from prompt schema)
- **Infrastructure**: RunPod GPU (NVIDIA A100-SXM4-80GB / RTX 4090), vLLM 0.17.1, PyTorch 2.10.0, CUDA 12.8

---

## Label Distribution (seed + extreme, N=760)

| Field | Dominant label | Rare labels |
|-------|---------------|-------------|
| sentiment | dovish 28%, neutral 26%, hawkish 21% | strongly_hawkish 8.6%, strongly_dovish 12.1%, na 4.1% |
| risk | na 87.4% | skewed_downside 8.6%, skewed_upside 3.2%, symmetric 0.9% |
| width | none 91.7% | elevated 6.1%, contested 2.2% |
| horizon | False 93% | True 7% |

The scarcity of rare labels (especially `symmetric` risk, `contested` width, `strongly_hawkish`) was a persistent challenge throughout.

---

## Key Insight: Inflated Aggregate Scores

Risk and width macro-F1 scores appeared high (~0.60–0.65) but were inflated by the dominant `na`/`none` defaults. When evaluated on **non-default labels only**:

- Risk drops from ~0.65 → ~0.55
- Width drops from ~0.58 → ~0.42
- Sentiment stays ~0.40 (never had a dominant default to inflate it)

All three fields are genuinely performing at 0.40–0.55 on the hard cases. Sentiment is not uniquely bad — it just lacked the easy-default crutch.

---

## Prompt Iterations

### Early Prompts (P0–P8, lowercase p1–p6)

Baseline prompts with minimal schema, no few-shot examples, or simple schema-only instructions.

- **SumF1**: ~0.15 across all early variants
- These effectively produced random outputs — the model could not follow the schema without worked examples

---

### P2_medium_3shot_final

First generation of "final" prompts. Medium-detail schema with 3 few-shot examples.

- **SumF1**: 0.509
- Notable: hawkish recall very high (0.904) but precision very low (0.325) — model defaulted to hawkish as a safe label
- `contested` width: 0.000 — completely undetected

---

### P3_medium_5shot_final / P3_medium_5shot_final_v3

Expanded to 5 shots. Modest improvement.

- **SumF1**: 0.552 / 0.531
- Best topic F1 across early prompts (0.629 for P3)
- Sentiment and contested still weak

---

### P5_high_3shot_final (baseline champion)

High-detail schema, 3 carefully selected few-shot examples. Became the baseline to beat for all subsequent iterations.

- **SumF1**: 0.567
- Best overall for a long period
- Weaknesses: `strongly_hawkish` (0.190), `strongly_dovish` (0.171), `contested` (0.083)

---

### P5_high_3shot_final_v3

Added error-correction rules targeting known failure modes.

- **SumF1**: 0.563 (marginal drop overall)
- Improved: `strongly_hawkish` (0.261), `contested` (0.304), width (0.627)
- The v3 error-correction rules made the model more conservative

---

### P7_high_5shot_final / P7_high_5shot_final_v3

5-shot variant of P5's high-detail schema.

- **SumF1**: 0.555 / 0.562
- P7 had best `unconditional` commitment F1 (0.697)
- v3 improved risk at cost of sentiment

---

### P8_high_12shot_final_v4

12 few-shot examples — the most shots attempted.

- **SumF1**: 0.521
- Counterintuitively worse than P5 (3-shot)
- More shots hurt: long prompt pushed the model past its effective context, diluting signal
- Lesson: **more shots ≠ better performance for 8B models**

---

### Chain-of-Thought Variants (P5-CoT, P7-CoT, P7v3-CoT, P8-CoT)

Added explicit step-by-step reasoning before the JSON output.

**Key findings:**
- CoT **dramatically improved** `strongly_hawkish` for P7-CoT: 0.105 → **0.462**
- CoT consistently improved `neutral` detection across all variants
- CoT **hurt** `strongly_dovish` (collapsed to 0.000 on some variants)
- CoT helped `horizon True` slightly
- Net effect: P7-CoT (0.566 SumF1) nearly matched P5 but didn't clearly beat it
- CoT is most valuable for intensity detection when combined with the right base examples

---

### FOMC Annotation Scheme (zero-shot)

Tested the full 8000-token annotation scheme document as a zero-shot prompt.

- **SumF1**: 0.153
- Complete failure — the 8B model could not apply the full ruleset without examples
- Lesson: **detailed rules without examples do not work for small models**

---

### P5_high_3shot_v4 / P7_high_5shot_v4

V4 variants collapsed `boilerplate` into `no_topic` and removed `na` from sentiment.

- P5v4 **SumF1**: 0.555 raw, **0.573 SumF1*** (second best when na excluded)
- `contested` improved significantly: 0.083 → 0.321
- `strongly_dovish` improved: 0.171 → 0.111 → 0.233 (mixed)
- Boilerplate/na removal penalised raw score but represented genuine schema simplification

---

### P_final_v5 / P5_final_v6 / P5_v7

Iterative refinements focusing on schema clarity and risk compression.

- P_final_v5: **SumF1**: 0.540 — strong width (0.644) and commitment (0.671)
- P5_v7: compressed `na` risk into `symmetric` — concept valid but sentiment degraded badly (0.302), collapsed risk score
- Lesson: `na` removal in risk hurt more than helped at this model size

---

### P5_FINAL

Added two targeted few-shot examples from the extreme seed:
- **Strongly hawkish**: *"The January employment report came in substantially stronger than most forecasters expected"* — illustrates `substantially stronger than expected` as intensity marker
- **Strongly dovish**: *"The unemployment rate surged in April by more than 10 percentage points to 14.7 percent, an 80-year high"* — illustrates stacked intensity markers + `skewed_downside` interaction

Results:
- **SumF1**: 0.565 (raw), **0.585 SumF1*** (best at the time on hard cases)
- `contested` width: 0.083 → **0.370** (largest single improvement)
- `strongly_hawkish`: 0.190 → 0.308
- `unconditional` commitment: 0.584 → **0.693**
- `economic_activity` topic: 0.563 → 0.661
- Trade-offs: horizon detection degraded; the new examples didn't cover long-term horizon language

---

### P5_v10 — Final Prompt

Further refinement incorporating lessons from P5_FINAL. Addressed precision failures and added more calibrated examples.

- **SumF1**: **0.606** (raw), **0.626 SumF1*** — new best by a significant margin (+0.04 over previous best)

#### Per-label results:

| Field | Label | Prec | Rec | F1 |
|-------|-------|------|-----|----|
| topic | inflation | 0.887 | 0.881 | **0.884** |
| topic | labor_market | 0.808 | 0.860 | **0.833** |
| topic | monetary_policy | 0.740 | 0.816 | **0.776** |
| topic | economic_activity | 0.568 | 0.892 | 0.695 |
| topic | financial_conditions | 0.564 | 0.824 | 0.670 |
| topic | no_topic | 0.355 | 0.717 | 0.475 |
| topic | macro | 0.404 | 0.610 | 0.486 |
| sentiment | neutral | 0.543 | 0.744 | **0.628** |
| sentiment | dovish | 0.621 | 0.594 | **0.607** |
| sentiment | hawkish | 0.531 | 0.675 | **0.595** |
| sentiment | strongly_hawkish | 0.667 | 0.222 | 0.333 |
| sentiment | strongly_dovish | 0.200 | 0.160 | 0.178 |
| risk | na | 0.972 | 0.920 | **0.945** |
| risk | symmetric | 0.700 | 0.778 | **0.737** |
| risk | skewed_upside | 0.714 | 0.714 | **0.714** |
| risk | skewed_downside | 0.544 | 0.782 | 0.642 |
| width | none | 0.950 | 0.971 | **0.960** |
| width | elevated | 0.931 | 0.574 | 0.711 |
| width | contested | 0.308 | 0.400 | 0.348 |
| commitment | none | 0.968 | 0.856 | **0.908** |
| commitment | unconditional | 0.636 | 0.686 | 0.660 |
| commitment | conditional | 0.396 | 0.833 | 0.537 |
| horizon | False | 0.970 | 0.752 | **0.847** |
| horizon | True | 0.314 | 0.831 | 0.456 |

#### Remaining failure modes in P5_v10:

| Pattern | Labels affected | Root cause |
|---------|----------------|-----------|
| **Recall failure** | `strongly_hawkish` (rec=0.222), `elevated` (rec=0.574) | Model too conservative — fires only when very certain |
| **Precision failure** | `economic_activity`, `financial_conditions`, `no_topic` (prec~0.35–0.57), `conditional` (prec=0.396), `horizon True` (prec=0.314) | Model over-fires — too many false positives |
| **Both weak** | `strongly_dovish` (F1=0.178), `contested` (F1=0.348), `macro` (F1=0.486) | Genuinely rare/ambiguous labels — insufficient training signal |
| **Schema gap** | `boilerplate` (F1=0.000), `na` sentiment (F1=0.000) | Intentionally removed from prompt schema |

---

## Summary of Prompt Evolution

| Prompt | SumF1 | SumF1* | Key change |
|--------|-------|--------|------------|
| P0–P8 (early) | ~0.15 | — | Baseline/no examples |
| P2_medium_3shot_final | 0.509 | — | First few-shot |
| P5_high_3shot_final | 0.567 | 0.562 | High-detail schema, 3 shots |
| P7_high_5shot_final_CoT | 0.566 | 0.579 | CoT — best strongly_hawkish |
| P5_high_3shot_v4 | 0.555 | 0.573 | Removed boilerplate/na |
| P5_FINAL | 0.565 | 0.585 | Added ±2 targeted examples |
| **P5_v10** | **0.606** | **0.626** | Final refinement — new best |

---

## Key Lessons

1. **More shots ≠ better**: P8 with 12 shots underperformed P5 with 3. Sweet spot for 8B models appears to be 3–5 well-chosen examples.

2. **Intensity detection requires targeted examples**: The ±2 labels (strongly_hawkish/dovish) only improved meaningfully once explicit intensity-marker examples were added to the prompt. Generic schema rules were insufficient.

3. **CoT helps selectively**: Chain-of-thought reasoning improved intensity detection and neutral classification but hurt strongly_dovish detection. Most valuable for intensity when combined with the right base prompt (P7-CoT).

4. **Aggregate scores are misleading**: The dominant `na`/`none` defaults inflate risk and width macro-F1 by ~0.10–0.15. Evaluating on non-default labels only gives a more honest picture.

5. **Boilerplate/na removal is a net win**: Removing the boilerplate topic and na sentiment from the schema simplifies the task and improves performance on substantive labels — the raw F1 drop is an artefact of evaluation methodology, not real degradation.

6. **Full annotation schemes don't work zero-shot**: The complete 8000-token annotation document scored 0.153 — worse than random on most fields. Small models need examples, not rules.

7. **Precision and recall failures require different fixes**: Over-firing labels (e.g., `horizon True`, `conditional`) need negative examples; under-firing labels (e.g., `strongly_hawkish`, `elevated`) need more positive examples and explicit recall-encouraging instructions.
