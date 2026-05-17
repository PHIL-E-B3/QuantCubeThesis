#!/bin/bash
# ============================================================
# Cloud GPU Setup Script (RunPod / Lambda / Vast.ai)
# ============================================================
# Sets up environment + runs training/eval pipeline on a cloud GPU.
# Updated May 2026 after live debugging on A100 SXM (see RUNPOD_SETUP_GUIDE.md).
#
# Usage:
#   bash scripts/cloud_setup.sh install      # dependencies only
#   bash scripts/cloud_setup.sh train-p1     # train P1 adapter only
#   bash scripts/cloud_setup.sh train-p3     # train P3 adapter only
#   bash scripts/cloud_setup.sh train-p5     # train P5 adapter only
#   bash scripts/cloud_setup.sh train-p7     # train P7 adapter only
#   bash scripts/cloud_setup.sh train-all    # train all 4 (P1, P3, P5, P7)
#   bash scripts/cloud_setup.sh eval         # vLLM eval on all 4 adapters
#   bash scripts/cloud_setup.sh prompt-eval  # base model on all P0-P8 prompts
#   bash scripts/cloud_setup.sh all          # install + train all + eval
# ============================================================

set -e
STEP=${1:-all}

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
log()  { echo -e "${GREEN}[cloud]${NC} $1"; }
warn() { echo -e "${YELLOW}[cloud]${NC} $1"; }

# ── 1. INSTALL DEPENDENCIES ──
install_deps() {
    log "Installing Python dependencies..."

    # DO NOT install torch — pre-installed on cloud images.
    # DO NOT use --no-deps — we need transitive deps (scipy, joblib, pandas, etc.)

    pip install --break-system-packages -r requirements-cloud.txt

    # Verify everything imports cleanly
    python -c "
import torch, transformers, accelerate, peft, bitsandbytes, vllm
import scipy, sklearn, joblib, threadpoolctl, pandas, pyarrow
print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('transformers:', transformers.__version__)
print('vllm:', vllm.__version__)
print('all imports clean')
"

    # GPU check
    log "GPU check:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    log "Dependencies installed."
}

# ── 2. CACHE REDIRECTION ──
setup_caches() {
    log "Redirecting caches to /workspace (persistent network volume)..."
    if ! grep -q "HF_HOME" ~/.bashrc 2>/dev/null; then
        cat >> ~/.bashrc << 'EOF'
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export VLLM_CACHE_ROOT=/workspace/.cache/vllm
export FLASHINFER_WORKSPACE_BASE=/workspace/.cache/flashinfer
export XDG_CACHE_HOME=/workspace/.cache
EOF
    fi
    mkdir -p /workspace/.cache/{huggingface,vllm,flashinfer}
    # Apply immediately for this shell
    export HF_HOME=/workspace/.cache/huggingface
    export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
    export VLLM_CACHE_ROOT=/workspace/.cache/vllm
    export FLASHINFER_WORKSPACE_BASE=/workspace/.cache/flashinfer
    export XDG_CACHE_HOME=/workspace/.cache
    log "Caches set."
}

# ── 3. DETECT GPU AND SET OPTIMAL PARAMS ──
get_gpu_params() {
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')

    if [ "$VRAM_MB" -ge 70000 ]; then
        # A100 80GB / H100 — bf16 full precision, large batches
        BATCH_SIZE=8
        GRAD_ACCUM=2
        MAX_SEQ=1850
        log "Detected high-VRAM GPU (${VRAM_MB}MB). Using bf16, batch=$BATCH_SIZE, grad_accum=$GRAD_ACCUM, max_seq=$MAX_SEQ"
    elif [ "$VRAM_MB" -ge 40000 ]; then
        BATCH_SIZE=8
        GRAD_ACCUM=2
        MAX_SEQ=1850
        log "Detected mid-VRAM GPU (${VRAM_MB}MB). bf16, batch=$BATCH_SIZE, grad_accum=$GRAD_ACCUM"
    elif [ "$VRAM_MB" -ge 20000 ]; then
        BATCH_SIZE=4
        GRAD_ACCUM=4
        MAX_SEQ=1850
        warn "Only ${VRAM_MB}MB VRAM. bf16 may be tight; consider QLoRA (4-bit) but watch for bitsandbytes/CUDA13 conflict."
    else
        BATCH_SIZE=4
        GRAD_ACCUM=4
        MAX_SEQ=1280
        warn "Low VRAM (${VRAM_MB}MB). May not fit bf16 8B model."
    fi
}

# ── 4. TRAIN ──
train_prompt() {
    local prompt=$1
    get_gpu_params
    log "Training $prompt adapter..."
    mkdir -p logs
    python scripts/sft_train.py \
        --prompt $prompt \
        --model unsloth/Meta-Llama-3.1-8B-Instruct \
        --batch-size $BATCH_SIZE \
        --grad-accum $GRAD_ACCUM \
        --max-seq-length $MAX_SEQ \
        --epochs 3 \
        2>&1 | tee logs/sft_${prompt}.log
}

train_p1() { train_prompt P1_medium_3shot; }
train_p3() { train_prompt P3_medium_5shot; }
train_p5() { train_prompt P5_high_3shot; }
train_p7() { train_prompt P7_high_5shot; }

train_all() {
    train_p1
    train_p3
    train_p5
    train_p7
    log "All 4 adapters trained (sft_p1, sft_p3, sft_p5, sft_p7)."
}

# ── 5. EVALUATE (vLLM-based, fast) ──
eval_adapters() {
    log "Evaluating all 4 adapters via vLLM..."
    mkdir -p logs
    for i in 1 3 5 7; do
        if [ ! -d "models/sft_p${i}/adapter" ]; then
            warn "models/sft_p${i}/adapter not found — skipping (train first)"
            continue
        fi
        log "Eval sft_p${i}..."
        if [ $i -eq 1 ]; then
            python scripts/sft_eval_vllm.py --adapter models/sft_p${i}/adapter --compare-base \
                2>&1 | tee logs/eval_vllm_sft_p${i}.log
        else
            python scripts/sft_eval_vllm.py --adapter models/sft_p${i}/adapter \
                2>&1 | tee logs/eval_vllm_sft_p${i}.log
        fi
    done

    log "Summary:"
    for i in 1 3 5 7; do
        python -c "
import json, os
p = 'models/sft_p${i}/eval_results/sft_sft_p${i}_metrics.json'
if not os.path.exists(p):
    print(f'sft_p${i}: (no results)')
else:
    m = json.load(open(p))
    print(f'sft_p${i}: F1 = {m[\"summary_f1\"]:.4f}')
" 2>/dev/null
    done
}

# ── 6. BASE-MODEL PROMPT EVAL ──
prompt_eval() {
    log "Running base-model prompt comparison (P0-P8)..."
    mkdir -p logs prompts/extras

    # Move variants out of the way
    for f in P3_medium_5shot_v2 P7_high_5shot_v2 P7v2_high_5shot P7v3_high_5shot P9_high_5shot_cot_tight; do
        [ -f "prompts/${f}.txt" ] && mv "prompts/${f}.txt" prompts/extras/
    done
    log "$(ls prompts/P*.txt | wc -l) prompts will be evaluated"

    python scripts/prompt_eval.py \
        --val-set data/eval_labelled_merged.json \
        --resume --vllm \
        --model unsloth/Meta-Llama-3.1-8B-Instruct \
        2>&1 | tee logs/base_prompt_eval.log

    # Restore variants
    mv prompts/extras/*.txt prompts/ 2>/dev/null
    rmdir prompts/extras 2>/dev/null

    log "Prompt eval complete. Results in models/prompt_eval/"
}

# ── 7. PACKAGE RESULTS ──
package_results() {
    log "Packaging results for download..."
    tar czf cloud_results.tar.gz \
        models/sft_p*/adapter/ \
        models/sft_p*/split_info.json \
        models/sft_p*/training_config.json \
        models/sft_p*/eval_results/ \
        models/prompt_eval/ \
        models/eval_summary_*.{md,json} \
        2>/dev/null || warn "Some result files missing"
    log "Results packaged: cloud_results.tar.gz"
    log "Download with: scp <instance>:~/QuantCubeThesis/cloud_results.tar.gz ."
}

# ── MAIN ──
case $STEP in
    install)     install_deps; setup_caches ;;
    train-p1)    setup_caches; train_p1 ;;
    train-p3)    setup_caches; train_p3 ;;
    train-p5)    setup_caches; train_p5 ;;
    train-p7)    setup_caches; train_p7 ;;
    train-all)   setup_caches; train_all ;;
    eval)        setup_caches; eval_adapters ;;
    prompt-eval) setup_caches; prompt_eval ;;
    package)     package_results ;;
    all)
        install_deps
        setup_caches
        train_all
        eval_adapters
        prompt_eval
        package_results
        log "============================================"
        log "ALL DONE."
        nvidia-smi --query-gpu=name --format=csv,noheader
        log "Download results: scp <instance>:~/QuantCubeThesis/cloud_results.tar.gz ."
        log "============================================"
        ;;
    *)
        echo "Usage: bash scripts/cloud_setup.sh [install|train-p1|train-p3|train-p5|train-p7|train-all|eval|prompt-eval|package|all]"
        exit 1
        ;;
esac
