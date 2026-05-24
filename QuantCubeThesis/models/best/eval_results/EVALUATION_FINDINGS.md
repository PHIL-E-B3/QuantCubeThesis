# Evaluation Findings — checkpoint-258 vs New Adapter

**Date:** 2026-05-24  
**Eval set used:** `data/eval_merged_labelled_corrected_3-class_com_con.json` (n=711)

---

## Why Results Varied Across Sessions

1. **Output files not in git** — `Taylor Rule/` is untracked, so CSV results were overwritten silently across runs with no versioning.
2. **Eval set changed three times:**
   - n=742 (`eval_labelled_merged_corrected.json`) — original, 5-class sentiment
   - n=712 — intermediate cleaned version
   - n=711 (`eval_merged_labelled_corrected_3-class_com_con.json`) — current, 3-class sentiment, 2-class commitment, no contested width
3. **Summary F1 definition varied** — primary fields were sometimes 4 (topic/sentiment/risk/width), correctly should be 5 (+ commitment).
4. **Label compression not applied to predictions** — the GT was already relabelled to 3-class but model predictions still used 5-class labels (strongly_hawkish etc.), unfairly penalising the model.

---

## Correct Evaluation Methodology

### Label compression (applied to BOTH predictions and GT)
- `strongly_hawkish` → `hawkish`
- `strongly_dovish` → `dovish`
- `contested` (width) → `none`
- `conditional` (commitment) → `none`

### Forcing rules (applied to predictions only, as post-processing)
- Topic does not include `monetary_policy` → `commitment = none`
- Topic is `no_topic` only → `sentiment = neutral`
- **Finding:** forcing rules have negligible effect (~0.001 F1) — the model already learns these constraints from the prompt.

### Sentiment: exclude `na` from macro F1
- The GT contains 3 sentences labelled `sentiment=na` (out of 711).
- Neither model ever predicts `na` (and the prompt doesn't ask for it).
- Including `na` as a class in macro F1 zeros out that class and artificially depresses sentiment F1 from ~0.77 to ~0.58.
- **Fix:** exclude `na` from sentiment evaluation → true 3-class eval (hawkish / neutral / dovish).

### Topic: use fixed valid label set
- P5_v41b causes the model to hallucinate `fiscal_policy` on 1 sentence.
- Including `fiscal_policy` as a spurious class in the macro F1 denominator inflated the apparent drop in topic F1 (0.731 apparent vs 0.836 true).
- **Fix:** restrict topic evaluation to the 7 valid labels.

---

## Step-by-Step Contribution (checkpoint-258, old prompt)

| Step | Summary F1 | Change |
|---|---|---|
| 1. Raw, 4-field primary | 0.649 | Baseline (what was previously reported) |
| 2. + commitment in primary (5-field) | 0.639 | −0.010 (commitment was bad uncompressed) |
| 3. + compress GT | 0.639 | 0.000 (GT already 3-class) |
| 4. + compress predictions | 0.807 | **+0.168** (biggest factor) |
| 5. + forcing rules | 0.806 | ~0.000 (negligible) |
| 6. + exclude na from sentiment | **0.845** | +0.039 |

---

## Final Results — Same 711 Sentences, Correct Methodology

| Field | ckpt-258 old prompt | ckpt-258 + P5_v41b | New adapter + P5_v41b |
|---|---|---|---|
| topic | 0.842 | 0.836 | 0.825 |
| sentiment | 0.773 | 0.778 | 0.721 |
| commitment | 0.924 | 0.917 | 0.898 |
| risk | 0.810 | 0.801 | 0.690 |
| width | 0.878 | 0.914 | 0.919 |
| **Summary F1** | **0.845** | **0.828** | **0.810** |

### Key takeaways
- **checkpoint-258 is the better adapter** — the new fine-tuning regressed on risk (−0.111) and sentiment (−0.057), despite improving on width (+0.041).
- **P5_v41b already outputs compressed labels** — no post-processing needed when using this prompt. Old prompt required compression as post-processing.
- **Forcing rules are redundant with P5_v41b** — the prompt already enforces them.
- **New adapter improved commitment** (0.924 → 0.898 slight drop) and width, but hurt risk significantly — training data changes likely targeted commitment/conditionality at the expense of risk.

### Decision
Use **checkpoint-258** with **P5_v41b** for full inference. Summary F1 = **0.828** without any post-processing.

---

## Scripts

- `recompute_f1.py` — raw per-field F1 from saved raw output files
- `recompute_compressed_f1.py` — compressed + forced evaluation with na exclusion
