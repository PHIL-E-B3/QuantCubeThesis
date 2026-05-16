#!/bin/bash
# ============================================================
# Cloud GPU Setup Script (RunPod / Lambda / Vast.ai)
# ============================================================
# This script sets up the environment on a cloud GPU instance,
# runs training and evaluation. Your local code is unchanged.
#
# Usage:
#   1. SSH into your cloud instance
#   2. Clone your repo: git clone <your-repo-url> && cd QuantCubeThesis
#   3. Run: bash scripts/cloud_setup.sh
#
# Or run individual steps:
#   bash scripts/cloud_setup.sh install    # dependencies only
#   bash scripts/cloud_setup.sh train-p3   # train P3 adapter only
#   bash scripts/cloud_setup.sh train-p7   # train P7 adapter only
#   bash scripts/cloud_setup.sh train-all  # train both adapters
#   bash scripts/cloud_setup.sh eval       # evaluate both adapters
#   bash scripts/cloud_setup.sh all        # full pipeline
# ============================================================

set -e  # Exit on error

STEP=${1:-all}

# ── Colors for output ──
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[cloud]${NC} $1"; }
warn() { echo -e "${YELLOW}[cloud]${NC} $1"; }

# ── 1. INSTALL DEPENDENCIES ──
install_deps() {
    log "Installing Python dependencies..."
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || true
    pip install transformers accelerate peft bitsandbytes datasets scikit-learn
    pip install vllm

    # Check GPU
    log "GPU check:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
    log "Dependencies installed."
}

# ── 2. DETECT GPU AND SET OPTIMAL PARAMS ──
get_gpu_params() {
    # Detect VRAM and set batch size accordingly
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')

    if [ "$VRAM_MB" -ge 70000 ]; then
        # A100 80GB / H100 — can use larger batches, no quantization needed
        BATCH_SIZE=8
        GRAD_ACCUM=2
        log "Detected high-VRAM GPU (${VRAM_MB}MB). Using batch_size=$BATCH_SIZE, grad_accum=$GRAD_ACCUM"
    elif [ "$VRAM_MB" -ge 40000 ]; then
        # A100 40GB / A6000
        BATCH_SIZE=8
        GRAD_ACCUM=2
        log "Detected mid-VRAM GPU (${VRAM_MB}MB). Using batch_size=$BATCH_SIZE, grad_accum=$GRAD_ACCUM"
    elif [ "$VRAM_MB" -ge 20000 ]; then
        # RTX 4090 / 3090
        BATCH_SIZE=4
        GRAD_ACCUM=4
        log "Detected 24GB GPU (${VRAM_MB}MB). Using batch_size=$BATCH_SIZE, grad_accum=$GRAD_ACCUM"
    else
        # 16GB or less — same as local
        BATCH_SIZE=4
        GRAD_ACCUM=4
        log "Detected <=16GB GPU (${VRAM_MB}MB). Using local defaults."
    fi
}

# ── 3. TRAIN ──
train_p3() {
    get_gpu_params
    log "Training P3 adapter..."
    python scripts/sft_train.py \
        --prompt P3_medium_5shot \
        --model unsloth/Meta-Llama-3.1-8B-Instruct \
        --batch-size $BATCH_SIZE \
        --grad-accum $GRAD_ACCUM \
        --epochs 3
    log "P3 adapter complete. Saved to models/sft_p3/"
}

train_p7() {
    get_gpu_params
    log "Training P7 adapter..."
    python scripts/sft_train.py \
        --prompt P7_high_5shot \
        --model unsloth/Meta-Llama-3.1-8B-Instruct \
        --batch-size $BATCH_SIZE \
        --grad-accum $GRAD_ACCUM \
        --epochs 3
    log "P7 adapter complete. Saved to models/sft_p7/"
}

train_all() {
    train_p3
    train_p7
    log "Both adapters trained."
}

# ── 4. EVALUATE ──
eval_adapters() {
    log "Evaluating P3 adapter (with base model comparison)..."
    python scripts/sft_eval.py \
        --adapter models/sft_p3/adapter \
        --compare-base

    log "Evaluating P7 adapter..."
    python scripts/sft_eval.py \
        --adapter models/sft_p7/adapter

    log "Evaluation complete. Results in models/sft_p3/eval_results/ and models/sft_p7/eval_results/"
}

# ── 5. DOWNLOAD RESULTS ──
package_results() {
    log "Packaging results for download..."
    tar czf cloud_results.tar.gz \
        models/sft_p3/adapter/ \
        models/sft_p3/split_info.json \
        models/sft_p3/training_config.json \
        models/sft_p3/eval_results/ \
        models/sft_p7/adapter/ \
        models/sft_p7/split_info.json \
        models/sft_p7/training_config.json \
        models/sft_p7/eval_results/ \
        2>/dev/null || warn "Some result files not found (training may not have run yet)"
    log "Results packaged: cloud_results.tar.gz"
    log "Download with: scp <instance>:~/QuantCubeThesis/cloud_results.tar.gz ."
}

# ── MAIN ──
case $STEP in
    install)    install_deps ;;
    train-p3)   train_p3 ;;
    train-p7)   train_p7 ;;
    train-all)  train_all ;;
    eval)       eval_adapters ;;
    package)    package_results ;;
    all)
        install_deps
        train_all
        eval_adapters
        package_results
        log "============================================"
        log "ALL DONE. Total GPU time:"
        nvidia-smi --query-gpu=name --format=csv,noheader
        log "Download results: scp <instance>:~/QuantCubeThesis/cloud_results.tar.gz ."
        log "============================================"
        ;;
    *)
        echo "Usage: bash scripts/cloud_setup.sh [install|train-p3|train-p7|train-all|eval|package|all]"
        exit 1
        ;;
esac
