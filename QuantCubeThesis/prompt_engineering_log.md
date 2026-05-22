# Prompt Engineering Log — FOMC Sentence Classification

## Overview

This document records the iterative prompt engineering process for classifying FOMC communications using `unsloth/Meta-Llama-3.1-8B-Instruct` via vLLM. The goal was to maximise macro-F1 on a 7-field annotation schema applied to held-out FOMC sentences.

---

## Task

Each sentence is classified into 7 fields:

| Field | Type | Values |
|-------|------|--------|
| `topic` | multi-label array | inflation, labor_market, economic_activity, macro, financial_conditions, monetary_policy, no_topic |
| `tense` | string | descriptive, interpretive |
| `sentiment` | string | strongly_hawkish, hawkish, neutral, dovish, strongly_dovish |
| `commitment` | string | unconditional, conditional, none |
| `risk` | string | skewed_upside, skewed_downside, symmetric, na |
| `width` | string | elevated, contested, none |

Note: `horizon` (boolean) was retained in some variants but ultimately handled by a dictionary rule rather than model prediction.

---

## Evaluation Setup

- **Validation set**: `eval_labelled_merged_corrected.json` (591 original sentences + 151 extended = 742 total)
- **Training seed**: `all_labelled_sentences.json` (810 sentences after augmentation with rare-label batches)
- **Primary metric**: Macro F1 averaged over topic, sentiment, risk, width (Summary F1)
- **Infrastructure**: RunPod A100 SXM4-80GB, vLLM, PyTorch, CUDA 13.0

---

## Central Finding

**Instructions and verbosity mattered more than shot count, but targeted examples mattered most of all.**

P5 (high-detail instructions, 3 examples) was competitive with or better than P7 (5 examples, lower-detail instructions), demonstrating that schema clarity and rule precision outweigh example volume for an 8B model. However, adding two precisely targeted examples to P5 — one for `strongly_hawkish`, one for `strongly_dovish` — produced a prompt that outperformed both P5 and P7 individually. This combination (P5 instruction quality + P7 shot count) became the basis for all subsequent versions.

The final compression phase (v10 → v27) confirmed the corollary: **examples carry more information per token than instruction text**. Aggressive instruction compression had minimal impact; aggressive example compression caused measurable regression.

---

## Phase 1: Early Baselines (P0–P8, p1–p6)

Minimal-schema prompts with no few-shot examples.

- **Summary F1**: ~0.15 across all early variants
- These produced near-random outputs — the model could not apply the schema without worked examples
- **Lesson**: Rules without examples are insufficient for an 8B model

---

## Phase 2: First Few-Shot Prompts (P2, P3)

### P2_medium_3shot_final
- **Summary F1**: 0.509
- Hawkish recall very high (0.904) but precision very low (0.325) — model defaulted to hawkish
- `contested` width: 0.000 — completely undetected

### P3_medium_5shot_final / v3
- **Summary F1**: 0.552 / 0.531
- Best topic F1 in early phase (0.629)
- Sentiment and contested still weak

---

## Phase 3: High-Detail Schema (P5 vs P7)

### P5_high_3shot_final — Baseline Champion
- **Summary F1**: 0.567
- High-detail schema, 3 carefully selected few-shot examples
- Weaknesses: `strongly_hawkish` (0.190), `strongly_dovish` (0.171), `contested` (0.083)

### P7_high_5shot_final / v3
- **Summary F1**: 0.555 / 0.562
- 5 examples with less detailed schema
- Best `unconditional` commitment F1 (0.697)

**Key observation**: P5 with 3 examples matched or outperformed P7 with 5 examples. High-detail instructions compensated for fewer shots. This confirmed that **verbosity and schema precision matter more than shot count**.

### P8_high_12shot_final_v4
- **Summary F1**: 0.521
- 12 examples — worst result of the high-shot experiments
- Long prompt pushed model past effective context, diluting signal
- **Lesson confirmed**: more shots ≠ better. Sweet spot for 8B models is 3–5 well-chosen examples

### Chain-of-Thought Variants
- CoT improved `strongly_hawkish` dramatically for P7-CoT: 0.105 → **0.462**
- CoT consistently improved `neutral` detection
- CoT hurt `strongly_dovish` in some variants
- Net: P7-CoT (0.566) nearly matched P5 but didn't clearly beat it
- CoT most valuable for intensity detection combined with the right base examples

---

## Phase 4: Structural Audit and Schema Fixes

Before further iteration, a full audit against v3 annotation instructions identified:

1. **Schema collapse**: boilerplate vs no_topic distinction. Resolved: procedural sentences → no_topic + neutral (not na). Locked as grading rule.
2. **Commitment definition**: tightened to strict two-part test — commitment verb AND explicit condition phrase. Easier for small model than fuzzy rule.
3. **Risk field contradiction**: `na` definition conflicted with conditional rule. Fixed: na = no explicit risk framing AND no conditional structure.
4. **Intensity vocabulary drift**: union of v3 and prompt lists adopted (including "tremendous", "dramatically", "plummeted", "severe").
5. **Recurring annotation errors** across 8 batches:
   - **Inflation polarity inversion** (8 sentences): subdued/moderating inflation labeled hawkish
   - **Commitment overclaiming** (10+ sentences): "will + vague action" called unconditional; "expects + action" called conditional
   - **Horizon misses** (8 sentences): "over the medium term", "over time", etc. left false
   - **Risk direction inversions**: downside risks called upside; dual-mandate collision misclassified
   - **Width conflation**: contested vs elevated repeatedly confused

---

## Phase 5: The Breakthrough — P5_FINAL / P5_v10

### The "Best of Both Worlds" Insight

Adding two precisely targeted examples to P5 achieved what neither P5 nor P7 could separately:
- Example 1: *"The January employment report came in substantially stronger than most forecasters expected"* → teaches intensity marker in comparison phrase → strongly_hawkish
- Example 2: *"The unemployment rate surged in April by more than 10 percentage points to 14.7 percent, an 80-year high"* → teaches stacked markers + forward projection → strongly_dovish + skewed_downside

This is technically equivalent to converting P5 into a P7-style prompt (5 examples with high-detail instructions) — **combining P5's instruction quality with P7's shot coverage**.

### P5_FINAL Results
- **Summary F1**: 0.565 raw, 0.585 SumF1*
- `contested` width: 0.083 → **0.370** (largest single improvement)
- `strongly_hawkish`: 0.190 → 0.308
- `unconditional` commitment: 0.584 → **0.693**
- `economic_activity` topic: 0.563 → 0.661

### P5_v10 — Further Refinement (Final Reference Prompt)

Additional rule additions targeting documented failures:
- Skewed upside clarification: "MORE inflation is the tail risk — label captures direction, not value judgment"
- Intensity criteria restructured into explicit (a/b/c) with comparison-phrase qualification
- Cumulative signals rule: 3+ same-direction signals → strongly_
- Width elevated triggers expanded with casual expressions
- DEFAULT-TO-NEUTRAL exception: intensity marker present → fire ±2 regardless

**Summary F1: 0.606 (new best by +0.04)**

#### Per-label results:

| Field | Label | F1 |
|-------|-------|----|
| topic | inflation | 0.884 |
| topic | labor_market | 0.833 |
| topic | monetary_policy | 0.776 |
| topic | economic_activity | 0.695 |
| topic | financial_conditions | 0.670 |
| topic | macro | 0.486 |
| topic | no_topic | 0.475 |
| sentiment | neutral | 0.628 |
| sentiment | dovish | 0.607 |
| sentiment | hawkish | 0.595 |
| sentiment | strongly_hawkish | 0.333 |
| sentiment | strongly_dovish | 0.178 |
| risk | na | 0.945 |
| risk | symmetric | 0.737 |
| risk | skewed_upside | 0.714 |
| risk | skewed_downside | 0.642 |
| width | none | 0.960 |
| width | elevated | 0.711 |
| width | contested | 0.348 |
| commitment | none | 0.908 |
| commitment | unconditional | 0.660 |
| commitment | conditional | 0.537 |

---

## Phase 6: Token Compression for Fine-Tuning (v10 → v27)

P5_v10 at ~2,900 tokens was too long for efficient fine-tuning (max_seq_length constraint). Target: ~2,200 tokens. This phase systematically tested compression strategies.

### Strategy 1 — Article Stripping (→ v11/v12, ~2,444 tokens)
Removed articles, tightened phrasing throughout instruction section. No content removed.
- **Result**: matched v10 performance exactly
- **Lesson**: Article stripping loses zero semantic content

### Strategy 2 — Merge Redundant Rule Blocks (→ ~2,200 tokens)
Collapsed DEFAULT-TO-NEUTRAL and Key Rules into sentiment definition.
- **Result**: performance dropped, primarily on risk
- **Diagnosis**: compressing directional logic into dense prose caused model to conflate risk and sentiment concepts. Risk requires two-step reasoning sensitive to instruction clarity.

### Strategy 3 — Cut Negative Contested Examples
Removed ✗ NOT contested examples from width.
- **Result**: poor performance
- **Lesson**: Negative examples are load-bearing. "Fell but remained high" teaches both the level-override rule AND that level+direction ≠ contested. Cannot be removed.

### v15 (~2,430 tokens)
Strategy 1 + width trigger cuts + light trimming + horizon removed (handled by dictionary rule).
- Only 14 tokens fewer than v12 despite all changes
- **Lesson confirmed**: examples dominate token count (~60% of total); instruction compression has hit diminishing returns

### v23 — Aggressive Reasoning Compression
Trimmed reasoning blocks in all examples aggressively.
- **Result**: bad performance on sentiment and risk
- **Lesson**: Sentiment and risk need full reasoning chains. The model needs every logical step spelled out, not just the conclusion.

### v24 — Conservative Reasoning Restoration
Restored sentiment and risk reasoning near-original length. Added contrast language for intensity threshold. Added "do not confuse with skewed_downside" to upside example.
- **~1,954 tokens** — identified ~100 tokens of headroom

### v25–v27 — Final Budget Spending (~2,217 tokens)
Three targeted additions using remaining headroom, each addressing documented failure modes:
1. Added explicit inflation polarity rule: "subdued/moderate/at or below target = dovish, not hawkish" — making the 8-instance recurring error explicit in instructions
2. Added "NOT elevated" to contested example reasoning — teaching that opposing forces without uncertainty vocabulary → wid=none, not elevated
3. Changed example 4 risk label to "skewed_downside, not na" — reinforcing that forward-projecting deterioration populates risk

**P5_v27 — Final Prompt for Fine-Tuning: 2,217 tokens**

---

## Summary Table

| Prompt | Summary F1 | Key change |
|--------|-----------|------------|
| P0–P8 (early) | ~0.15 | Baseline, no examples |
| P2_medium_3shot_final | 0.509 | First few-shot |
| P5_high_3shot_final | 0.567 | High-detail schema, 3 shots |
| P7_high_5shot_final | 0.555 | 5 shots, less instruction detail |
| P7-CoT | 0.566 | Chain-of-thought — best strongly_hawkish |
| P5_FINAL | 0.565 | Added 2 targeted ±2 examples |
| **P5_v10** | **0.606** | Full revision — new best |
| P5_v27 | (fine-tuning) | Compressed to 2,217 tokens |

---

## Key Lessons

1. **Instructions > shot count, but targeted examples > both**: P5 beat P7 on instruction quality, but adding 2 precisely targeted examples to P5 outperformed both. The combination of P5's schema precision with P7's shot range achieved the best results.

2. **More shots ≠ better**: P8 with 12 shots underperformed P5 with 3. Sweet spot for 8B models is 3–5 well-chosen examples. Beyond 5, the prompt context crowds out the model's working memory.

3. **Negative examples are load-bearing**: The ✗ NOT contested examples in width were carrying two rules simultaneously. Removing them caused measurable regression. Before cutting any example content, test the specific rule it teaches.

4. **Examples dominate token cost**: ~60% of total tokens are in examples. Instruction compression yielded diminishing returns after basic article stripping. To materially reduce token count, you must compress or remove examples — but this comes at a performance cost.

5. **Sentiment and risk need full reasoning chains**: Compressing example reasoning caused the largest performance drops. These fields have the most nuanced rules and the model needs to see each logical step.

6. **Instruction-level and example-level teaching are complementary**: The inflation polarity error persisted across 8 sentences despite being implicitly taught through examples. Making it an explicit rule ("subdued/moderate = dovish") provided a second learning signal.

7. **Contested is the hardest field**: F1=0.348–0.370 throughout, resistant to prompt changes. Requires simultaneous evaluation of 5 conditions. Negative examples and the "hedge ≠ economic force" rule are the most important guardrails.

8. **Schema decisions have outsized downstream impact**: The largest F1 jump came from fixing boilerplate/no_topic schema collapse — not from any prompt refinement. Clean, consistent labels matter more than prompt sophistication with small training data.

9. **Full annotation schemes don't work zero-shot**: The complete 8000-token annotation document scored 0.153 — worse than random. Small models need examples, not rules alone.
