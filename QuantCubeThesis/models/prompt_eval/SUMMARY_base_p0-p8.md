# Base Model Prompt Comparison (Llama-3.1-8B-Instruct, no SFT)

**Date:** 2026-05-17
**Eval set:** 587 sentences (eval_labelled_merged.json, 5 few-shot example IDs excluded)
**Inference:** vLLM batched, bf16, max_new_tokens=256 (CoT: 512)

## Ranked by Summary F1 (mean of top, sen, ris, wid macro-F1)

| Prompt                    | Summary F1 | Top   | Sen   | Ris   | Wid   | ParseFail |
|---------------------------|-----------:|------:|------:|------:|------:|----------:|
| **P7_high_5shot**         | **0.4785** | 0.495 | 0.386 | 0.554 | 0.480 |   0.0%   |
| P5_high_3shot             | 0.4577     | 0.473 | 0.399 | 0.498 | 0.460 |   0.0%   |
| P1_medium_3shot           | 0.4489     | 0.508 | 0.254 | 0.568 | 0.466 |   0.0%   |
| P2_medium_3shot_cot       | 0.4228     | 0.408 | 0.283 | 0.594 | 0.406 |  13.6%   |
| P3_medium_5shot           | 0.4194     | 0.497 | 0.259 | 0.524 | 0.397 |   0.0%   |
| P4_medium_5shot_cot       | 0.3865     | 0.408 | 0.257 | 0.468 | 0.413 |  14.1%   |
| P0_minimal                | 0.3529     | 0.388 | 0.213 | 0.558 | 0.252 |   0.5%   |
| P6_high_3shot_cot         | 0.3433     | 0.292 | 0.286 | 0.435 | 0.361 |  31.2%   |
| P8_high_5shot_cot         | 0.3348     | 0.283 | 0.297 | 0.391 | 0.369 |  33.6%   |

**Winner:** `P7_high_5shot` (Summary F1 = 0.4785)

## Observations

- **High-detail prompts (P5, P7) win for the base model.** More detailed system instructions + 5-shot examples gives Llama enough scaffolding to produce reliable JSON.
- **CoT prompts (P2, P4, P6, P8) consistently parse-fail** at 13–33%. Llama produces reasoning text that breaks JSON parsing more often than it helps accuracy. Worth investigating: maybe a stricter "ANSWER: {json}" format in CoT prompts would lower parse failures.
- **5-shot > 3-shot, but not always.** P7 (5-shot) > P5 (3-shot), but P1 (3-shot medium) > P3 (5-shot medium). Suggests detail × examples interacts non-trivially.
- **P0_minimal** is near the bottom as expected: no examples, no schema = no consistency.

## Cross-comparison with SFT adapters

Despite P7 being best for base model, **`sft_p3` (P3-trained adapter) won the SFT eval** at F1=0.4930. Key insight: in-context learning preferences ≠ fine-tuning preferences. P3's medium-detail prompt may be a better "format teacher" during SFT, while P7's verbose instructions help the un-trained base model reason but become redundant once the LoRA learns the schema.

| Comparison           | Best prompt        | Summary F1 |
|----------------------|--------------------|-----------:|
| Base model (587)     | P7_high_5shot      | 0.4785    |
| SFT adapter (114)    | sft_p3 / P3_medium_5shot | 0.4930 |

Note: the comparison uses different eval sets (587 vs 114) so the +0.015 gap is suggestive, not definitive.
