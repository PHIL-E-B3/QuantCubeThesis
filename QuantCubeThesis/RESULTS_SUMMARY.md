# Results Summary — May 17, 2026

End-to-end SFT pipeline run on a fresh A100 SXM 80GB. This document summarises everything we trained, evaluated, and learned in one place.

## TL;DR

| Comparison           | Best                    | Summary F1 |
|----------------------|-------------------------|-----------:|
| **SFT adapters**     | `sft_p3` (P3_medium_5shot)  | **0.4930** |
| **Base model prompts** | `P7_high_5shot`         | 0.4785     |

- **4 SFT adapters trained** (P1, P3, P5, P7) on the expanded training set including 72 new rare-class annotations.
- **9 base-model prompts evaluated** (P0–P8) on the full eval set (587 sentences).
- **Key insight**: the prompt-style best for base model (P7) is NOT the prompt-style best for SFT (P3). Detailed prompts help in-context learning but become redundant once a LoRA learns the schema directly.

---

## 1. Data

| File | Records |
|------|---------|
| `data/eval_labelled_merged.json` (after merging 72 new rare-class annotations) | 592 |
| Few-shot example IDs excluded from train/test | 5 |
| Available for stratified split | 587 |
| **Training set (80%)** | 455 |
| **Held-out test set (20%)** | 114 |

**Stratified by `sen`**, seed=42 — same split used across all 4 adapters (so directly comparable).

The 72 new annotations came from `data/QuantCube_Seed_Labelled/seed_*_annotated.json`:

- `seed_ris_skewed_downside_annotated.json`: 24 records
- `seed_ris_skewed_upside_annotated.json`: 24 records
- `seed_wid_elevated_annotated.json`: 20 records
- `seed_wid_contested_annotated.json`: 4 records (after curation)

They were targeted at rare-class examples that the original training was light on. These were merged into `eval_labelled_merged.json` before training.

---

## 2. SFT Adapter Results (114 held-out test sentences)

All adapters use Llama-3.1-8B-Instruct base + LoRA (r=16, α=32, dropout=0.05, target = q/k/v/o/gate/up/down).

Training: 3 epochs, effective batch size 16 (batch=8 × grad_accum=2), lr=2e-4 cosine, max_seq_len=1850.

**Quantization**: bf16 (no 4-bit) due to a CUDA 13 / bitsandbytes runtime conflict on this RunPod template. See `RUNPOD_SETUP_GUIDE.md`. Optimizer is `adamw_torch`.

### Summary F1 (mean of top, sen, ris, wid macro-F1)

| Adapter   | Summary F1 | top    | ten    | sen    | com    | hor    | ris    | wid    | ParseFail |
|-----------|-----------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|----------:|
| sft_p1    | 0.4887     | 0.458  | 0.408  | 0.435  | 0.504  | 0.541  | 0.697  | 0.364  | 0.0% |
| **sft_p3** | **0.4930** | **0.520** | 0.408 | 0.399 | 0.504 | 0.541 | 0.683 | 0.370 | 0.0% |
| sft_p5    | 0.4644     | 0.500  | 0.408  | 0.380  | 0.504  | 0.541  | 0.620  | 0.357  | 0.0% |
| sft_p7    | 0.4665     | 0.454  | 0.408  | 0.356  | 0.504  | 0.541  | 0.699  | 0.358  | 0.0% |

**Winner: `sft_p3` (Summary F1 = 0.4930)**, narrowly edging out sft_p1.

### Observations

- **Medium-detail prompts beat high-detail prompts for SFT.** With only 455 training examples, the model doesn't benefit from verbose prompt scaffolding — simpler instructions generalize better.
- **RIS F1 jumped from ~0.33 (previous sft_p3 run) to ~0.68** across all adapters. The 48 new RIS annotations (24 upside + 24 downside) had a clear effect.
- **WID F1 still weak (~0.36).** Even with new `wid=elevated` (20) and `wid=contested` (4) examples, the model struggles. Possible causes: too few `wid=contested` examples; the multi-clause contrastive structure is genuinely hard.
- **All four adapters within 0.03 of each other.** On a 114-sentence test set with imbalanced classes, the winner margin could be within noise. P3 vs P1 looks essentially tied.
- **0% parse failure across all SFT adapters** — fine-tuning eliminated JSON formatting errors entirely.

---

## 3. Base Model Prompt Comparison (587 sentences)

Llama-3.1-8B-Instruct, no SFT, vLLM batched inference. Each prompt template wraps each sentence as a chat message; we measure F1 on the parsed JSON output.

### Ranked by Summary F1

| Prompt                    | Summary F1 | top   | sen   | ris   | wid   | ParseFail |
|---------------------------|-----------:|------:|------:|------:|------:|----------:|
| **P7_high_5shot**         | **0.4785** | 0.495 | 0.386 | 0.554 | 0.480 |   0.0%    |
| P5_high_3shot             | 0.4577     | 0.473 | 0.399 | 0.498 | 0.460 |   0.0%    |
| P1_medium_3shot           | 0.4489     | 0.508 | 0.254 | 0.568 | 0.466 |   0.0%    |
| P2_medium_3shot_cot       | 0.4228     | 0.408 | 0.283 | 0.594 | 0.406 |  13.6%    |
| P3_medium_5shot           | 0.4194     | 0.497 | 0.259 | 0.524 | 0.397 |   0.0%    |
| P4_medium_5shot_cot       | 0.3865     | 0.408 | 0.257 | 0.468 | 0.413 |  14.1%    |
| P0_minimal                | 0.3529     | 0.388 | 0.213 | 0.558 | 0.252 |   0.5%    |
| P6_high_3shot_cot         | 0.3433     | 0.292 | 0.286 | 0.435 | 0.361 |  31.2%    |
| P8_high_5shot_cot         | 0.3348     | 0.283 | 0.297 | 0.391 | 0.369 |  33.6%    |

**Winner: `P7_high_5shot` (Summary F1 = 0.4785)**.

### Observations

- **High-detail prompts (P5, P7) dominate the top of the table.** Detailed system instructions + 5-shot examples give the un-tuned base model enough scaffolding to produce reliable JSON.
- **All CoT prompts have high parse-failure rates** (13–34%). The model produces reasoning text that breaks JSON parsing more often than CoT helps accuracy. CoT prompts may need a stricter `ANSWER: {json}` post-amble to recover usable output.
- **More few-shot examples helps non-uniformly**: P7 (5-shot) > P5 (3-shot), but P1 (3-shot) > P3 (5-shot). Suggests the *detail level × shot count* interaction matters more than either alone.
- **P0_minimal is near the bottom**, as expected. No examples, no schema = no consistency.

---

## 4. Cross-Comparison: Base vs SFT

| | Best prompt           | Summary F1 |
|---|-----------------------|-----------:|
| **Base model** (587 sentences) | P7_high_5shot         | 0.4785     |
| **SFT adapter** (114 test)     | sft_p3 (P3_medium_5shot) | 0.4930     |

The +0.015 gap is suggestive but on different sample sizes (587 vs 114). For an apples-to-apples comparison, the next step would be evaluating each SFT adapter on the same 587-sentence set the base model was tested on (excluding the 455 training IDs).

### Why does the best base prompt differ from the best SFT prompt?

P7's verbose instructions help base Llama infer the labelling rules at inference time. Once you fine-tune, the LoRA encodes those rules directly in its weights — and verbose prompts add noise rather than information. P3's medium detail seems to be a "format teacher" sweet spot: enough structure for the model to learn the JSON schema, not so much that it overfits to prompt phrasing.

---

## 5. Reproducibility

**Code state**: GitHub commit `2d9e40e` on `main`.

**Key files**:
- `scripts/sft_train.py` — trains a LoRA adapter (bf16, adamw_torch)
- `scripts/sft_eval_vllm.py` — fast vLLM-based eval (use this instead of `sft_eval.py`)
- `scripts/prompt_eval.py` — base-model prompt comparison with vLLM
- `models/sft_p{1,3,5,7}/` — adapter weights (in Dropbox, gitignored), training_config.json, split_info.json, eval_results/ (all in git)
- `models/prompt_eval/` — base model prompt results (all in git)
- `models/eval_summary_p1357.{json,md}` — consolidated SFT comparison
- `models/prompt_eval/SUMMARY_base_p0-p8.md` — consolidated base prompt comparison

**To reproduce sft_p3** (the winning adapter):

```bash
python scripts/sft_train.py \
    --prompt P3_medium_5shot \
    --model unsloth/Meta-Llama-3.1-8B-Instruct \
    --batch-size 8 --grad-accum 2 \
    --max-seq-length 1850 \
    --epochs 3 --lr 2e-4 \
    --lora-r 16 --lora-alpha 32 \
    --seed 42
```

Then evaluate:

```bash
python scripts/sft_eval_vllm.py \
    --adapter models/sft_p3/adapter --compare-base
```

---

## 6. Next Steps

1. **120K-sentence inference** with sft_p3 on `data/all_unlabelled_sentences/master_unlabelled_pool.json` (not yet run, ~1.5–2 hrs on A100).
2. **Apples-to-apples eval**: run sft_p3 against the full 587-sentence eval set, not just 114, for a cleaner base-vs-SFT delta.
3. **WID-specific improvement**: WID F1 is still ~0.36 — the weakest field. Either annotate more `wid=elevated/contested` examples or post-process with rules.
4. **CoT prompts need fixing**: 14–34% parse failures wastes signal. Adding a strict `ANSWER: ` post-amble to CoT templates should recover most of it.
