# QuantCubeThesis — FOMC Sentence Classification

Multi-field classification of Federal Reserve communications (statements, minutes, speeches, press conferences) using LoRA-fine-tuned Llama-3.1-8B. Each sentence is mapped to a 7-field JSON: `top` (topic), `ten` (tense), `sen` (sentiment −2 to 2), `hor` (forward-looking), `com` (commitment), `ris` (risk skew), `wid` (uncertainty).

## Current Status (May 2026)

**Best SFT adapter**: `sft_p3` — Summary F1 = 0.4930 on held-out 114-sentence test set.

**Best base-model prompt**: `P7_high_5shot` — Summary F1 = 0.4785 on full 587-sentence eval.

See [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md) for the full comparison table, per-field metrics, observations, and next steps.

## Project Structure

```
├── prompts/                       # Prompt templates P0–P9 + variants
├── data/
│   ├── eval_labelled_merged.json  # Main eval/training corpus (~592 records)
│   ├── QuantCube_Seed_Labelled/   # Per-batch annotation files
│   └── all_unlabelled_sentences/  # Raw unlabelled pool for inference (~120K)
├── scripts/
│   ├── sft_train.py               # QLoRA fine-tuning (bf16, currently)
│   ├── sft_eval.py                # SFT eval via transformers (legacy, has loading bug)
│   ├── sft_eval_vllm.py           # SFT eval via vLLM (use this)
│   ├── prompt_eval.py             # Base-model prompt comparison (vLLM)
│   └── cloud_setup.sh             # RunPod dep install + training automation
├── models/
│   ├── sft_p{1,3,5,7}/
│   │   ├── adapter/               # LoRA weights (gitignored)
│   │   ├── checkpoints/           # Training snapshots (gitignored)
│   │   ├── split_info.json        # Train/test IDs (in git)
│   │   ├── training_config.json   # Hyperparameters (in git)
│   │   └── eval_results/          # Per-adapter metrics (in git)
│   └── prompt_eval/               # Base-model P0–P8 results
├── RESULTS_SUMMARY.md             # Full comparison of all runs
├── RUNPOD_SETUP_GUIDE.md          # Step-by-step pod setup (read this for cloud)
└── requirements-cloud.txt         # Pinned cloud dependencies
```

## Setup

### Cloud GPU (recommended for training + inference)

See **[`RUNPOD_SETUP_GUIDE.md`](RUNPOD_SETUP_GUIDE.md)** for a step-by-step walkthrough on A100 SXM. Covers:
- Pod configuration (50 GB container disk, 100 GB network volume)
- SSH setup from local PowerShell
- Dependency installation (with gotchas — there are several missing transitive deps not in `requirements-cloud.txt`)
- Training, evaluation, and base-prompt comparison commands
- The bitsandbytes / CUDA-13 / transformers strict-loading workarounds

### Local (data exploration only)

The current Llama-3.1-8B training does not fit comfortably on consumer GPUs in bf16 mode (~16 GB needed, more with LoRA gradients). Quantization (4-bit) makes it work on ≥16 GB, but `bitsandbytes` has CUDA version compatibility issues on RunPod images — see the setup guide. For local development you can still use this repo for data manipulation, annotation, and prompt design.

```bash
git clone https://github.com/PHIL-E-B3/QuantCubeThesis.git
cd QuantCubeThesis
# Windows: setup_env.bat
# Mac/Linux: chmod +x setup_env.sh && ./setup_env.sh
```

## Usage

### Train an SFT adapter

```bash
python scripts/sft_train.py \
    --prompt P3_medium_5shot \
    --model unsloth/Meta-Llama-3.1-8B-Instruct \
    --batch-size 8 --grad-accum 2 \
    --max-seq-length 1850 \
    --epochs 3 --lr 2e-4 \
    --seed 42
```

The script loads `data/eval_labelled_merged.json`, does an 80/20 stratified split by `sen`, trains a LoRA adapter, and saves to `models/sft_<prompt_short>/adapter/` plus `split_info.json` and `training_config.json`.

Available prompts (define a corresponding `models/sft_<short>/` folder in `PROMPT_CONFIGS` to add new ones):
- `P1_medium_3shot` → `sft_p1`
- `P3_medium_5shot` → `sft_p3`
- `P5_high_3shot` → `sft_p5`
- `P7_high_5shot` → `sft_p7`

### Evaluate an SFT adapter

```bash
python scripts/sft_eval_vllm.py \
    --adapter models/sft_p3/adapter \
    --compare-base
```

Evaluates on the 114 held-out sentences from training (test_ids in `split_info.json`). Use `--compare-base` to also evaluate the base model on the same set.

### Compare base-model prompts

```bash
python scripts/prompt_eval.py \
    --val-set data/eval_labelled_merged.json \
    --resume --vllm \
    --model unsloth/Meta-Llama-3.1-8B-Instruct
```

Runs every prompt matching `prompts/P*.txt` on the full eval set, ranks by Summary F1, saves to `models/prompt_eval/`.

## Hardware Requirements

**For training (recommended):**
- **GPU**: A100 SXM 80GB or equivalent (H100, RTX PRO 6000)
- **RAM**: 64 GB+ system RAM (provided by RunPod template)
- **Disk**: 50 GB container disk + 100 GB persistent network volume

**Why not 4-bit on smaller GPUs?** `bitsandbytes` has a CUDA 13 runtime dependency issue on current RunPod images (looks for `libnvJitLink.so.13` which isn't present). Until that's resolved, training runs in bf16, which requires ~25–30 GB VRAM. See [`RUNPOD_SETUP_GUIDE.md` §9](RUNPOD_SETUP_GUIDE.md) for workarounds.

## Label Taxonomy (7 fields)

| Field | Type   | Values |
|-------|--------|--------|
| `top` | multi  | inflation, unemployment, economic_activity, macro, financial_conditions, monetary_policy, boilerplate, no_topic |
| `ten` | single | descriptive, interpretive |
| `sen` | single | -2, -1, 0, 1, 2, na |
| `hor` | single | True, False (forward-looking) |
| `com` | single | unconditional, conditional, none |
| `ris` | single | skewed_downside, skewed_upside, symmetric, na |
| `wid` | single | elevated, contested, none |

Primary fields for headline F1: `top`, `sen`, `ris`, `wid` (the four most informative for downstream macro analysis).

See `data/FOMC_annotation_instructions_v3_corrected.md` for the full annotation rulebook.
