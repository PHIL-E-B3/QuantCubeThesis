# Prompt Evaluation — Identified Issues
**Date:** 2026-05-17  
**Scope:** Base model prompt comparison P1, P3, P5, P7 (Llama-3.1-8B-Instruct, no SFT, 587 eval sentences)

---

## Issue 1 — Critical label mismatch: `unemployment` vs `labor_market`

**Severity: Critical**

All four prompts (P1, P3, P5, P7) list `"unemployment"` as a valid topic value in the schema definition and in the few-shot examples. The ground truth and annotation schema use `"labor_market"`. The model is not hallucinating — it is doing exactly what the prompt instructs.

**Result:** `labor_market` F1 = **0.000 across every prompt**. Of 47 ground-truth `labor_market` sentences, ~33–35 are predicted as `unemployment` per prompt — a label that does not exist in the schema.

| Prompt | True=labor_market (n=47) | Predicted `unemployment` |
|--------|--------------------------|--------------------------|
| P1 | 0 correct | 35 |
| P3 | 0 correct | 33 |
| P5 | 0 correct | 33 |
| P7 | 0 correct | 34 |

**Fix:** Replace `"unemployment"` with `"labor_market"` in the schema definition and in all few-shot examples across all prompts. One-line change per prompt file.

---

## Issue 2 — Sentiment polarity inversion

**Severity: High**

The model systematically confuses hawkish and dovish sentiment, especially in P1 and P3. When the true label is `-1` (dovish), P1 predicts `2` (strongly hawkish) in 75 of 163 cases. P3 predicts `2` in 61 of 163 cases. P5 and P7 are better but still flip the sign — predicting `1` (hawkish) for true `-1` (dovish) in ~50–60 cases.

**Result:**

| Prompt | True=-1 (n=163) | Correct | Top wrong prediction |
|--------|-----------------|---------|----------------------|
| P1 | 1 | `2` (hawkish): 75, blank: 61 |
| P3 | 0 | `2` (hawkish): 61, blank: 55 |
| P5 | 32 | `1` (hawkish): 60, blank: 46 |
| P7 | 29 | `1` (hawkish): 53, blank: 51 |

**Root cause — magnitude-without-sign problem (not a scale direction issue):**

The current scale is linguistically correct: bad economic conditions (high unemployment, weak growth) map to negative values (dovish pressure), and hot conditions (high inflation, tight labour market) map to positive values (hawkish pressure). This is intuitive and Llama's pretraining prior should support it.

The `2` prediction is not a sign confusion — it is the model detecting **strong, directional economic language** and outputting `2` as its "high-intensity signal" escape hatch, without correctly resolving the sign. Three prompt bugs prevent correct sign resolution:

1. **No negative few-shot anchors.** P1 and P5 show only `"na"`, `1`, `0` — the value `-1` is never demonstrated in the output. P3 has one `-2` example but no `-1`. The model has never seen how to resolve strong language as negative, so it defaults to the high-magnitude positive value it has been shown.

2. **Asymmetric polarity rules.** The rules list explicit hawkish triggers ("inflation rising → hawkish, unemployment falling → hawkish") but do not state the dovish inverses. The model must infer that unemployment *rising* → dovish, which it fails to do reliably when the only confirmed examples go one direction.

3. **Level-vs-direction instruction error.** P1/P3 say: `"IMPORTANT: when a sentence shows direction-of-change AND level (e.g. 'fell but remained high'), label the direction of change, not the level."` This directly contradicts the annotation instructions, which say to label the **level**, not the direction of change. P5 is internally contradictory — the header says "LEVEL OVERRIDES DIRECTION" but the body text then says "label the direction of change" — so the wrong instruction wins at inference time.

**Fix:**
1. Add at least one `sen = -1` and one `sen = -2` few-shot example to every prompt.
2. Add explicit dovish polarity anchors to the rules (e.g. "unemployment rising = dovish (−), inflation falling = dovish (−)").
3. Correct the level-vs-direction instruction in P1/P3 to say "label the level". Remove the contradictory body text in P5.

---

## Issue 3 — `wid = contested` defaults to `none`

**Severity: High**

When the true label is `wid = contested`, P1 and P3 predict `none` almost universally (14/14 and 15/15 errors respectively go to `none`). P5 and P7 perform substantially better (10/17 and 9/17 correct) but still default to `none` for remaining errors.

| Prompt | True=contested (n=17) | Correct | Predicted `none` |
|--------|-----------------------|---------|------------------|
| P1 | 3 | 14 |
| P3 | 2 | 15 |
| P5 | 10 | 6 |
| P7 | 9 | 7 |

**Likely cause:** `wid = contested` requires recognising two simultaneously operative opposing economic forces — a genuinely complex multi-clause inference. P1/P3 lack sufficient explanation of this distinction. P5/P7 include an explicit definition and recover most cases.

**Note:** This failure persists in SFT — WID F1 remains ~0.36 even after fine-tuning, the weakest field across all adapters. More `contested` training examples are likely needed (currently only 16 in the seed).

---

## Issue 4 — `ris = symmetric` confused with `skewed_downside`

**Severity: Medium**

When the true label is `ris = symmetric`, the model does not default to `na` — it makes a directional guess, predominantly `skewed_downside`.

| Prompt | True=symmetric (n=19) | Correct | Predicted `skewed_downside` |
|--------|-----------------------|---------|-----------------------------|
| P1 | 10 | 4 |
| P3 | 10 | 5 |
| P5 | 10 | 5 |
| P7 | 12 | 3 |

**Likely cause:** The model recognises risk language but defaults to the more common directional class rather than recognising "balanced" framing. `symmetric` is rare in the training data (n=7 in seed) and the prompts do not include a `symmetric` few-shot example.

**Fix:** Add a few-shot example where `ris = symmetric` is the correct label (e.g. a sentence with "risks to both sides" or "roughly balanced").

---

## Issue 5 — `top = no_topic` defaults to `monetary_policy`

**Severity: Medium**

Of 49 ground-truth `no_topic` sentences, the model predicts `no_topic` zero times across all four prompts. The most common wrong prediction is `monetary_policy` (~19–21 per prompt).

| Prompt | True=no_topic (n=49) | Correct | Top wrong prediction |
|--------|----------------------|---------|----------------------|
| P1 | 0 | `monetary_policy`: 19 |
| P3 | 0 | `monetary_policy`: 21 |
| P5 | 0 | `boilerplate`: 14 |
| P7 | 0 | `monetary_policy`: 20 |

**Likely cause:** The model cannot recognise that a generic sentence carries no specific economic content. It defaults to the most topically salient word in the sentence.

**Fix:** Add a `no_topic` few-shot example. The definition in the prompts ("generic statements that convey no specific economic content") may need a negative example showing what is *not* `no_topic`.

---

## Issue 6 — CoT prompts produce high parse failure rates

**Severity: Medium** (affects P2, P4, P6, P8 — not in primary set but relevant for future)

| Prompt | Parse failure rate |
|--------|--------------------|
| P2_medium_3shot_cot | 13.6% |
| P4_medium_5shot_cot | 14.1% |
| P6_high_3shot_cot | 31.2% |
| P8_high_5shot_cot | 33.6% |

The model produces reasoning text that breaks JSON parsing. CoT prompts need a strict `ANSWER: {json}` post-amble with a parser that extracts only the JSON block after the `ANSWER:` tag.

---

## Summary Table

| Issue | Fields affected | Worst prompts | Fix complexity |
|-------|----------------|---------------|----------------|
| `unemployment` label bug | `top` | All | Trivial — rename in prompt |
| Sentiment polarity inversion | `sen` | P1, P3 (worst) | Medium — add negative few-shot anchors + dovish polarity rules + fix level-vs-direction error |
| `contested` defaults to `none` | `wid` | P1, P3 | Medium — add examples + more seed data |
| `symmetric` confused with `skewed_downside` | `ris` | All | Easy — add one few-shot example |
| `no_topic` invisible | `top` | All | Easy — add one few-shot example |
| CoT parse failures | all | P2/P4/P6/P8 | Medium — add structured output post-amble |
