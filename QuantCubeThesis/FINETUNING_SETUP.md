# Fine-Tuning Setup Guide — LoRA on A100 SXM4 80GB

Step-by-step guide for running the Optuna hyperparameter search + final LoRA fine-tune on RunPod. Tested May 2026 on A100 SXM4 80GB with CUDA 13.0.

---

## 1. Pod Selection

- **GPU**: NVIDIA A100 SXM4 80GB
- **Cloud type**: Secure Cloud (NOT Community Cloud — older drivers)
- **CUDA version**: must show **13.0** in the pod selector before deploying
- **Container disk**: 50 GB minimum
- **Storage**: attach existing Network Volume at `/workspace`

Verify after connecting:
```bash
nvidia-smi
```
Should show `CUDA Version: 13.0` and `Driver Version: 580.x`.

---

## 2. Environment Setup

Run these exports at the start of every session:

```bash
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_TOKEN=your_huggingface_token

mkdir -p /workspace/.cache/huggingface
```

---

## 3. Clone the Repo

Clone into the home directory (NOT `/workspace`):

```bash
git clone https://github.com/PHIL-E-B3/QuantCubeThesis.git
cd QuantCubeThesis/QuantCubeThesis
```

---

## 4. Install Dependencies

**Critical rules:**
- Do NOT install bitsandbytes — CUDA 13 has no compatible binary; training uses bf16 full precision instead
- Do NOT install flash-attn — CUDA toolkit mismatch prevents compilation; training uses sdpa automatically
- Do NOT pin or reinstall transformers after this step

```bash
pip install -r requirements-cloud.txt --no-deps
pip install peft optuna colorlog alembic sqlalchemy
pip install scipy joblib threadpoolctl pandas pyarrow dill multiprocess xxhash
pip install "transformers==4.46.3" --force-reinstall
```

**Why transformers==4.46.3**: newer versions raise a RuntimeError on weight conversion entries when loading models for LoRA fine-tuning on this pod configuration.

---

## 5. Verify Setup

Run the baseline check — loads data, tokenises, prints stats. No GPU used, exits immediately:

```bash
python scripts/train.py --config configs/default.yaml --prompt prompts/P5_v41b.txt --baseline
```

Expected output:
```
Model:      unsloth/Meta-Llama-3.1-8B-Instruct
Prompt:     prompts/P5_v41b.txt
Max length: 2560
Device:     NVIDIA A100-SXM4-80GB
Loaded 906 labelled sentences from 3 file(s)
Generative split — train: 770, val: 136  (eval set held out separately)
Dataset built. Exiting (--baseline mode).
```

If you see 0 skipped examples and the correct sentence counts, proceed to training.

---

## 6. Run Final Retrain (using existing best Optuna params)

Since Optuna has already been run and best hyperparameters are stored in `models/optuna/fomc_qlora.db`, use `--retrain-best` to skip the search and go straight to the 3-epoch retrain with the expanded dataset and best prompt:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train.py --config configs/default.yaml --prompt prompts/P5_v41b.txt --retrain-best
```

### To run a fresh Optuna search instead (optional)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train.py --config configs/default.yaml --prompt prompts/P5_v41b.txt --optuna
```

### What this does

1. **Optuna search** (25 trials, 1 epoch each):
   - Searches over: lora_r ∈ {2, 4, 8}, lora_alpha_multiplier ∈ {1, 2}, lora_dropout ∈ {0.05, 0.1, 0.2}, learning_rate ∈ [0.0001, 0.0005], weight_decay ∈ {0.01, 0.05, 0.1}, batch_size ∈ {8, 16}
   - Each trial trains for 1 epoch and evaluates on the validation set
   - Results saved to `models/optuna/fomc_qlora.db` after each completed trial
   - If a trial OOMs it is pruned and Optuna moves to the next

2. **Automatic final retrain** with best hyperparameters for **3 full epochs**

3. **Saves LoRA adapter** to `models/best/adapter/`

### Expected timing (without flash attention)
- ~10–12 min per Optuna trial
- 25 trials ≈ 4–5 hours
- Final 3-epoch retrain ≈ 35–40 min
- **Total: ~5–6 hours**

### Progress indicators

After each completed trial you will see:
```
────────────────────────────────────────────────────────────
  Trial   5/25  [OK]  loss=0.0548
  Params: {'lora_r': 4, 'lora_alpha_multiplier': 2, ...}
  Best so far: trial #4  loss=0.0554
────────────────────────────────────────────────────────────
```

After all trials, automatic retraining starts:
```
=== Retraining with best parameters ===
```

Final output:
```
Adapter saved to models/best/adapter
```

---

## 7. Resuming an Interrupted Run

The Optuna study is saved in `models/optuna/fomc_qlora.db`. If training is interrupted (SSH drop, timeout, manual stop):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train.py --config configs/default.yaml --optuna
```

It will automatically resume from where it left off. Completed trials are not re-run. Only the current in-progress trial is lost.

---

## 8. Saving Results

Once training completes, push the adapter and results to git:

```bash
git add models/best/
git add models/optuna/
git commit -m "Add fine-tuned LoRA adapter and Optuna results"
git push origin main
```

**Note**: adapter weights are ~177 MB per checkpoint. The `.gitignore` excludes `models/*/checkpoints/` but includes `models/best/adapter/`.

---

## 9. Key Configuration (configs/default.yaml)

| Parameter | Value | Notes |
|---|---|---|
| `model.name` | unsloth/Meta-Llama-3.1-8B-Instruct | Base model |
| `model.max_seq_length` | 2560 | P5_v27 (2217 tokens) + buffer |
| `training.num_epochs` | 3 | For the final retrain after Optuna |
| `training.per_device_train_batch_size` | 16 | A100 80GB handles this |
| `optuna.n_trials` | 25 | Hyperparameter search trials |
| `lora.target_modules` | q/k/v/o/gate/up/down | All 7 linear layers |
| `paths.seed_data_merged` | all_labelled_sentences.json + final_extreme_seed.json + active_learning_all.json | Training data only |
| `paths.eval_data` | eval_merged_labelled_corrected_3-class_com_con.json | True held-out test set (711 records) |

---

## 10. Known Issues and Workarounds

### bitsandbytes error at startup
```
bitsandbytes library load error: Configured CUDA binary not found at .../libbitsandbytes_cuda130.so
```
**This is a harmless warning.** bitsandbytes has no CUDA 13 binary but we are not using 4-bit quantization. Training uses bf16 full precision and proceeds normally. Ignore this message.

### flash_attention_2 unavailable
```
flash_attention_2 unavailable, using sdpa
```
**Expected behaviour.** The CUDA toolkit on this pod is 12.4 while PyTorch was compiled for CUDA 13, preventing flash-attn compilation. The code automatically falls back to sdpa (PyTorch's native efficient attention). Training proceeds normally, ~20% slower than with flash attention.

### Trainer.tokenizer deprecation warning
```
Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
```
**Harmless.** Ignore.

### Trial count exceeds n_trials
If the trial counter shows e.g. 26/25, this is because the Optuna study database already contained failed/pruned trials from previous runs. The extra trial was part of the requested search. Normal behaviour.

### OOM during Optuna trial
If a trial OOMs (especially batch_size=16), it will be automatically caught, cleaned up, and marked as PRUNED. Optuna learns to avoid that configuration. You will see:
```
Trial  X/25  [PRUNED]  loss=n/a
```
Training continues with the next trial.

---

## 11. Architecture Notes

- **No 4-bit quantization**: A100 80GB has 80GB VRAM; Llama 3.1 8B in bf16 uses ~16GB, leaving 64GB free. 4-bit would be needed only on smaller GPUs.
- **LoRA targets all 7 linear layers**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj. This matches the standard QLoRA setup from Dettmers et al.
- **Training prompt**: P5_v27 (2,217 tokens) — the result of systematic compression from P5_v10 while preserving performance. See `prompt_engineering_log.md` for full history.
- **Training data**: ~770 train / ~136 val from 906 unique sentences (741 seed + 165 active learning; extreme seed fully contained in main seed). Eval set (`eval_merged_labelled_corrected_3-class_com_con.json`, 711 sentences) is never touched during training. The 69 records that overlapped between the original seed file and the eval set have been removed from training.
