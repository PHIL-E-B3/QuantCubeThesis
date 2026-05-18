# v3 _final Prompt Eval — Brief Report

**Date:** 2026-05-18
**Eval set:** eval_labelled_merged.json (591 sentences, 5 few-shot example IDs excluded)
**Model:** unsloth/Meta-Llama-3.1-8B-Instruct (base, no SFT)
**Inference:** vLLM batched, bf16, max_new_tokens=256

## What changed v1 → v3

Three prompt edits applied to all 4 _final prompts (P2/P3/P5/P7):

1. **DEFAULT-TO-NEUTRAL rules** — explicit guidance to default to neutral on ambiguous sentiment cases. Targeted the hawkish bias in v1 (70% of P5 SEN errors were hawk-biased; `dovish→hawkish` and `neutral→hawkish` were the top two patterns).
2. **Width=`contested` with concrete examples** — added 4 positive and 3 negative examples, plus an explicit test ("imagine each force in isolation, do they push policy in opposite directions?"). Targeted the 82–88% miss rate on `contested`.
3. **Stricter `commitment` rules** — `conditional` now requires BOTH an explicit action verb AND an explicit condition. Default to `none` when uncertain. Targeted the 16–20% over-prediction of `conditional`.

## Headline F1 results

| Prompt | v1 F1 | v3 F1 | Δ |
|--------|------:|------:|--:|
| P2_medium_3shot | 0.4691 | 0.4793 | +0.0102 |
| P3_medium_5shot | 0.5108 | 0.5001 | -0.0106 |
| P5_high_3shot | 0.5185 | 0.5112 | -0.0074 |
| **P7_high_5shot** | 0.5145 | **0.5248** | **+0.0103** |

**New winner: P7_high_5shot_final_v3 (Summary F1 = 0.5248)** — narrowly past the previous best (P5_high_3shot_final at 0.5185).

## Per-field deltas for P5_v3 vs P5_v1

| Field | v1 F1 | v3 F1 | Δ |
|-------|------:|------:|--:|
| topic | 0.575 | 0.554 | -0.021 |
| tense | 0.399 | 0.378 | -0.021 |
| **sentiment** | 0.415 | 0.427 | **+0

cat > models/prompt_eval/SUMMARY_v3_final.md << 'EOF'
# v3 _final Prompt Eval — Brief Report

**Date:** 2026-05-18
**Eval set:** eval_labelled_merged.json (591 sentences, 5 few-shot example IDs excluded)
**Model:** unsloth/Meta-Llama-3.1-8B-Instruct (base, no SFT)
**Inference:** vLLM batched, bf16, max_new_tokens=256

## What changed v1 → v3

Three prompt edits applied to all 4 _final prompts (P2/P3/P5/P7):

1. **DEFAULT-TO-NEUTRAL rules** — explicit guidance to default to neutral on ambiguous sentiment cases. Targeted the hawkish bias in v1 (70% of P5 SEN errors were hawk-biased; `dovish→hawkish` and `neutral→hawkish` were the top two patterns).
2. **Width=`contested` with concrete examples** — added 4 positive and 3 negative examples, plus an explicit test ("imagine each force in isolation, do they push policy in opposite directions?"). Targeted the 82–88% miss rate on `contested`.
3. **Stricter `commitment` rules** — `conditional` now requires BOTH an explicit action verb AND an explicit condition. Default to `none` when uncertain. Targeted the 16–20% over-prediction of `conditional`.

## Headline F1 results

| Prompt | v1 F1 | v3 F1 | Δ |
|--------|------:|------:|--:|
| P2_medium_3shot | 0.4691 | 0.4793 | +0.0102 |
| P3_medium_5shot | 0.5108 | 0.5001 | -0.0106 |
| P5_high_3shot | 0.5185 | 0.5112 | -0.0074 |
| **P7_high_5shot** | 0.5145 | **0.5248** | **+0.0103** |

**New winner: P7_high_5shot_final_v3 (Summary F1 = 0.5248)** — narrowly past the previous best (P5_high_3shot_final at 0.5185).

## Per-field deltas for P5_v3 vs P5_v1

| Field | v1 F1 | v3 F1 | Δ |
|-------|------:|------:|--:|
| topic | 0.575 | 0.554 | -0.021 |
| tense | 0.399 | 0.378 | -0.021 |
| **sentiment** | 0.415 | 0.427 | **+0.012** |
| horizon | 0.477 | 0.429 | -0.048 |
| commitment | 0.478 | 0.472 | -0.006 |
| risk | 0.646 | 0.579 | -0.067 |
| **width** | 0.439 | 0.485 | **+0.046** |

## Interpretation

**The fix worked where targeted:**
- `width` +0.046 — concrete contested examples + the "imagine each force in isolation" test let the model recognize the pattern. Best confirmation that the prompt edit had its intended effect.
- `sentiment` +0.012 — DEFAULT-TO-NEUTRAL helped slightly, though gains are modest. May need stronger framing in a future iteration.

**The fix caused unintended regressions:**
- `risk` -0.067 — the neutrality framing seems to have spilled over into risk attribution: the model is more reluctant to assign `skewed_downside` / `skewed_upside` when ambiguous.
- `horizon` -0.048 — same suspicion: the added neutrality wording may have downweighted forward-looking signals.
- `topic` and `tense` slightly worse (-0.021 each) — likely noise from longer prompt + a 4th in-context example shifting topic predictions.

**Net effect:** the v3 fixes shift errors around rather than eliminate them. P7_v3 wins by +0.010 overall, but P3 and P5 regressed slightly. The pattern suggests **risk is now the highest-leverage target** — it dropped sharply on the prompt change despite being one of the strongest fields in v1.

## Next step

Use `P7_high_5shot_final_v3` as the working prompt for downstream steps (base-model inference on the 120K unlabelled pool, or as the training prompt for the next SFT run).
