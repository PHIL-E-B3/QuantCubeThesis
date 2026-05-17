# SFT Adapter Comparison: P1 / P3 / P5 / P7

**Date:** 2026-05-17
**Eval set:** 114 held-out test sentences (stratified 80/20 split by `sen`, seed=42)
**Training data:** 455 sentences (includes 72 new rare-class annotations)

## Summary F1 (mean of top, sen, ris, wid macro-F1)

| Adapter   | Summary F1 | top    | ten    | sen    | com    | hor    | ris    | wid    |
|-----------|-----------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| sft_p1    | 0.4887     | 0.458  | 0.408  | 0.435  | 0.504  | 0.541  | 0.697  | 0.364  |
| **sft_p3** | **0.4930** | **0.520** | 0.408 | 0.399 | 0.504 | 0.541 | 0.683 | 0.370 |
| sft_p5    | 0.4644     | 0.500  | 0.408  | 0.380  | 0.504  | 0.541  | 0.620  | 0.357  |
| sft_p7    | 0.4665     | 0.454  | 0.408  | 0.356  | 0.504  | 0.541  | 0.699  | 0.358  |

**Winner:** `sft_p3` (Summary F1 = 0.4930)

## Observations

- **Medium-detail prompts (P1, P3) beat high-detail (P5, P7).** Likely because 455 training examples isn't enough for the model to learn from extra prompt verbosity — simpler instructions generalize better at small scale.
- **RIS F1 jumped to ~0.68–0.70** across all adapters, vs. 0.33 in the original sft_p3 run. The 72 new rare-class annotations (24 downside, 24 upside, 20 elevated, 4 contested) clearly worked for RIS.
- **WID F1 still weak (~0.36).** Even with more elevated/contested examples, the model struggles to identify uncertainty/contestedness. Possible next step: more WID annotations or rule-based post-processing.
- **All adapters within 0.03 of each other on Summary F1** — differences may be within noise on a 114-sentence test set. The winner margin is small.

## Next step
Run `sft_p3` on the 120K unlabelled FOMC sentences using vLLM batched inference.
