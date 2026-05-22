# Prompt Engineering Setup — RunPod A100

Step-by-step guide for running prompt evaluation on RunPod. Tested May 2026 on A100 SXM4 80GB with CUDA 13.0.

---

## 1. Pod selection

- **GPU**: NVIDIA A100 SXM4 80GB
- **Cloud type**: Secure Cloud (NOT Community Cloud — older drivers)
- **CUDA version**: must show **13.0** in the pod selector before deploying
- **Container disk**: 50 GB (default 20 GB fills up with vLLM/HF caches)
- **Storage**: attach existing Network Volume at `/workspace`

Verify after connecting:
```bash
nvidia-smi
```
Should show `CUDA Version: 13.0` and `Driver Version: 580.x`.

---

## 2. Environment setup

Run these exports at the start of every session (add to `~/.bashrc` to make permanent):

```bash
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export VLLM_CACHE_ROOT=/workspace/.cache/vllm
export FLASHINFER_WORKSPACE_BASE=/workspace/.cache/flashinfer
export XDG_CACHE_HOME=/workspace/.cache
export HF_TOKEN=your_huggingface_token

mkdir -p /workspace/.cache/{huggingface,vllm,flashinfer}
```

---

## 3. Clone the repo

**Important**: clone into the home directory, not `/workspace`. Do NOT `cd /workspace` first.

```bash
git clone https://github.com/PHIL-E-B3/QuantCubeThesis.git
cd QuantCubeThesis/QuantCubeThesis
```

---

## 4. Install dependencies

**Critical rule**: do NOT pin or reinstall `transformers` manually. Let vLLM install the version it needs (>=4.56.0). Overriding this is what breaks everything.

```bash
pip install -r requirements-cloud.txt --no-deps
pip install vllm
pip install scipy joblib threadpoolctl pandas pyarrow dill multiprocess xxhash
```

That's it. Three commands. No transformers pin.

---

## 5. Run prompt evaluation

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 python scripts/prompt_eval.py \
    --val-set data/eval_labelled_merged_corrected.json \
    --vllm \
    --prompt P5_v10.txt \
    --model unsloth/Meta-Llama-3.1-8B-Instruct
```

To evaluate a specific prompt variant:
```bash
VLLM_USE_FLASHINFER_SAMPLER=0 python scripts/prompt_eval.py \
    --val-set data/eval_labelled_merged_corrected.json \
    --vllm \
    --prompt P5_v11.txt \
    --model unsloth/Meta-Llama-3.1-8B-Instruct
```

To resume an interrupted run (skips already-completed sentences):
```bash
VLLM_USE_FLASHINFER_SAMPLER=0 python scripts/prompt_eval.py \
    --val-set data/eval_labelled_merged_corrected.json \
    --vllm \
    --prompt P5_v10.txt \
    --model unsloth/Meta-Llama-3.1-8B-Instruct \
    --resume
```

---

## 6. Key gotchas

### Do not pin transformers
Running `pip install "transformers==4.46.3"` after vLLM breaks everything — vLLM 0.21.0 requires transformers >=4.56.0. Never override this.

### CUDA version matters
- CUDA 13.0 (driver 580+): works natively, no patches needed
- CUDA 12.4 or 12.8: vLLM 0.21.0 compiled for CUDA 13, will fail with binary incompatibility errors
- Always check `nvidia-smi` first and confirm CUDA 13.0 before proceeding

### Model caching
If the model was previously downloaded to `/workspace/.cache/huggingface`, it loads instantly with no HuggingFace API calls. If starting fresh, the 16 GB download happens on first run only.

### VLLM_USE_FLASHINFER_SAMPLER=0
Always prepend this. It disables the flashinfer sampler which can cause issues even on CUDA 13 pods.

### Cloning into home, not workspace
Clone the repo into `/root` (the default home directory), not `/workspace`. The workspace is for cache only. Running from `/root/QuantCube...` is correct.

---

## 7. Results location

Results are saved to `models/prompt_eval/` inside the project directory:
- `{prompt_name}_metrics.json` — F1, accuracy, parse failure rate per field
- `{prompt_name}_raw.json` — raw model responses for debugging
