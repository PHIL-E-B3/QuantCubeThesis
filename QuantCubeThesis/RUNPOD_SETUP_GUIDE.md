# RunPod Setup Guide — A100 SXM 80GB

Step-by-step setup for running QuantCubeThesis SFT training + eval on RunPod. Tested May 2026. Reflects fixes for several version-mismatch and CUDA bugs we hit live; do not skip steps unless you know why.

---

## 1. RunPod console — deploy pod

### GPU choice
Pick **A100 SXM 80GB on Secure Cloud** ($1.49/hr at time of writing). Reasons:
- 80GB VRAM lets us train without 4-bit quantization (avoids `bitsandbytes` + CUDA-13 conflict — see §6).
- Cheaper than H100, more available than B200/H200.
- Same GPU previous training runs used, so configs port directly.

Fallback if unavailable: RTX PRO 6000 ($1.69/hr Community Cloud, 96GB VRAM).

### Storage
- **Container disk: 50 GB.** The 20 GB default fills up with HF caches + vLLM JIT cache and crashes.
- **Network volume: 100 GB**, attached at `/workspace`. Created via Storage → Network Volumes. Pick the region matching your chosen GPU (e.g. EU-RO-1).
  - Network volumes persist across pod terminations and are billed at $0.07/GB/month (~$7/month for 100 GB).
  - Don't use the per-pod "Volume disk" option — it dies with the pod.

### SSH setup
- In RunPod Settings → SSH Public Keys, paste your local `~/.ssh/id_ed25519.pub` (one-time per account, persists across pods).
- After the pod is running, grab the **"SSH over exposed TCP"** line from the Connect tab. Looks like `ssh root@IP -p PORT -i ~/.ssh/id_ed25519`.
- Update your local `~/.ssh/config`:
  ```
  Host runpod
      HostName <IP_from_runpod>
      Port <PORT_from_runpod>
      User root
      IdentityFile C:\Users\<you>\.ssh\id_ed25519
      ServerAliveInterval 60
  ```
  (Each pod restart gives new IP/port — update this file accordingly.)
- Test: `ssh runpod` from local PowerShell. If first time, type `yes` on host fingerprint prompt.

---

## 2. Pod prep — first connection

Once SSH'd into the pod:

```bash
# Redirect every cache to the persistent network volume.
# Stops the container disk filling up and lets you reuse downloads across pods.
cat >> ~/.bashrc << 'EOF'
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export VLLM_CACHE_ROOT=/workspace/.cache/vllm
export FLASHINFER_WORKSPACE_BASE=/workspace/.cache/flashinfer
export XDG_CACHE_HOME=/workspace/.cache
EOF
mkdir -p /workspace/.cache/{huggingface,vllm,flashinfer}
source ~/.bashrc

# Verify GPU + storage
nvidia-smi | head -10
df -h | grep -E "overlay|workspace"   # overlay = container disk (50GB), workspace = network volume (100GB)
```

Expected: A100-SXM4-80GB shown, `overlay 50G`, `/workspace 100G+` (or ~117 GB if EU mfs).

---

## 3. Clone repo + install dependencies

```bash
cd /workspace
git clone https://github.com/PHIL-E-B3/QuantCubeThesis.git
cd QuantCubeThesis/QuantCubeThesis

# Configure git for pushes from pod
git config --global user.email "<your_email>"
git config --global user.name "<your_github_username>"
git config --global credential.helper 'cache --timeout=86400'

# Install Python deps — this is the painful part.
# Don't just run cloud_setup.sh blindly; it auto-launches training before
# verifying deps and we hit a chain of missing transitive packages.

# 1. Base pinned packages (your friend's pinned versions; will be partially overridden below)
pip install --break-system-packages -r requirements-cloud.txt --no-deps

# 2. vLLM (not in requirements.txt; chunky download ~1 GB)
pip install --break-system-packages vllm

# 3. Upgrade transformers + accelerate + peft + bitsandbytes
#    The pinned 4.46.0 conflicts with vLLM's transitive deps.
pip install --break-system-packages --upgrade transformers accelerate peft bitsandbytes

# 4. Missing transitive deps that the above don't auto-install
pip install --break-system-packages scipy joblib threadpoolctl pandas pyarrow dill multiprocess xxhash

# Verify everything imports
python -c "
import torch, transformers, accelerate, peft, bitsandbytes, vllm
import scipy, sklearn, joblib, threadpoolctl, pandas, pyarrow
print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('transformers:', transformers.__version__)
print('vllm:', vllm.__version__)
print('peft:', peft.__version__)
print('all good')
"
```

Should print `all good`. If any import fails, install that one package, retry.

---

## 4. Install tmux (so training survives SSH drops)

```bash
apt-get update -qq && apt-get install -y -qq tmux
```

All long-running jobs go in tmux sessions. Detach with `Ctrl+B then D`, reattach with `tmux attach -t <name>`.

---

## 5. Run training (4 SFT adapters in sequence)

```bash
mkdir -p logs

# Optional: merge new rare-class annotations into eval set before training.
# (Skip if already merged — check len of eval_labelled_merged.json first.)
python << 'EOF'
import json
with open('data/eval_labelled_merged.json') as f:
    eval_data = json.load(f)
print(f'Before: {len(eval_data)} records')
existing_ids = set(r['id'] for r in eval_data)
for nf in [
    'data/QuantCube_Seed_Labelled/seed_ris_skewed_downside_annotated.json',
    'data/QuantCube_Seed_Labelled/seed_ris_skewed_upside_annotated.json',
    'data/QuantCube_Seed_Labelled/seed_wid_elevated_annotated.json',
    'data/QuantCube_Seed_Labelled/seed_wid_contested_annotated.json',
]:
    for r in json.load(open(nf)):
        if r['id'] not in existing_ids:
            eval_data.append(r); existing_ids.add(r['id'])
print(f'After: {len(eval_data)} records')
with open('data/eval_labelled_merged.json', 'w') as f:
    json.dump(eval_data, f, indent=2)
EOF

# Train all 4 adapters in tmux
tmux new -s train
# Inside tmux:
for prompt in P1_medium_3shot P3_medium_5shot P5_high_3shot P7_high_5shot; do
    echo "[$(date +%H:%M:%S)] Training $prompt"
    python scripts/sft_train.py \
        --prompt $prompt \
        --batch-size 8 --grad-accum 2 \
        --max-seq-length 1850 \
        2>&1 | tee logs/sft_${prompt}.log
done
echo "[$(date +%H:%M:%S)] DONE"
# Ctrl+B D to detach
```

Each adapter takes ~8 min on A100 SXM. Total ~35 min for all 4.

**Time-saver**: `prompt_eval.py` and `sft_train.py` both load the base model once per script invocation. If you want max speed, run all 4 in a single Python process — but the loop above is simpler.

---

## 6. Run eval (4 adapters)

Use the vLLM-based eval script — `sft_eval.py` (transformers-based) currently hits a strict-loading bug on the newer transformers version.

```bash
for i in 1 3 5 7; do
    if [ $i -eq 1 ]; then
        python scripts/sft_eval_vllm.py --adapter models/sft_p${i}/adapter --compare-base 2>&1 | tee logs/eval_vllm_sft_p${i}.log
    else
        python scripts/sft_eval_vllm.py --adapter models/sft_p${i}/adapter 2>&1 | tee logs/eval_vllm_sft_p${i}.log
    fi
done

# Summary
for i in 1 3 5 7; do
    python -c "
import json
m = json.load(open('models/sft_p${i}/eval_results/sft_sft_p${i}_metrics.json'))
print(f'sft_p${i}: F1 = {m[\"summary_f1\"]:.4f}')
"
done
```

Each eval ~2 min (vLLM is fast). Results saved to `models/sft_p{N}/eval_results/`.

---

## 7. Run base-model prompt comparison

```bash
# Temporarily move variant prompts so only P0-P8 main versions run
mkdir -p prompts/extras
mv prompts/P3_medium_5shot_v2.txt prompts/P7_high_5shot_v2.txt \
   prompts/P7v2_high_5shot.txt prompts/P7v3_high_5shot.txt \
   prompts/P9_high_5shot_cot_tight.txt prompts/extras/ 2>/dev/null
ls prompts/P*.txt | wc -l   # should be 9

# Run all 9 prompts on the full eval set (~25 min on A100 with vLLM)
python scripts/prompt_eval.py \
    --val-set data/eval_labelled_merged.json \
    --resume --vllm \
    --model unsloth/Meta-Llama-3.1-8B-Instruct \
    2>&1 | tee logs/base_prompt_eval_9prompts.log

# Restore variants
mv prompts/extras/*.txt prompts/
rmdir prompts/extras

# Print ranked summary
python << 'EOF'
import json
from pathlib import Path
rows = []
for f in sorted(Path('models/prompt_eval').glob('P*_metrics.json')):
    if any(v in f.stem for v in ['_v2', 'v2_', 'v3_', 'P9_', 'tight']):
        continue
    m = json.load(open(f))
    rows.append((m['prompt_name'], m['summary_f1'], m['parse_failure_rate']))
rows.sort(key=lambda x: x[1], reverse=True)
for name, f1, pf in rows:
    print(f"{name:<28} F1={f1:.4f}  PF={pf:.1%}")
EOF
```

---

## 8. Backup before terminating pod

Run from your local PowerShell (uses your SSH config `runpod`):

```powershell
cd "<your-local-path>/QuantCubeThesis"

# Pull adapter weights to local Dropbox
New-Item -ItemType Directory -Force -Path models\sft_p1, models\sft_p3, models\sft_p5, models\sft_p7 | Out-Null
foreach ($i in 1, 3, 5, 7) {
    scp -r runpod:/workspace/QuantCubeThesis/QuantCubeThesis/models/sft_p$i/adapter models\sft_p$i\
}
```

Each adapter is ~177 MB. Total ~700 MB.

**Don't `scp` the `checkpoints/` folders unless you need to resume training** — they're ~750 MB each, and training is only 8 min on A100 so re-training is cheaper than storing them.

Then commit small files (eval results, configs, summaries) to git from the pod or local — the `.gitignore` already keeps `models/*/adapter/` and `models/*/checkpoints/` out.

---

## 9. Known gotchas / debug notes

### bitsandbytes + CUDA 13 mismatch
- RunPod's A100 image ships CUDA 13.0 driver but no CUDA 13 runtime libs.
- Latest `bitsandbytes` wheels are compiled against CUDA 13, looking for `libnvJitLink.so.13` which doesn't exist on this container.
- **Workaround**: don't use 4-bit quantization. `sft_train.py` now defaults to bf16 (full-precision weights, ~16 GB VRAM, fits easily on 80 GB A100). The model trains slightly slower but the LoRA outcome is the same (and arguably better — no quantization noise).
- If you really need 4-bit (e.g. on a smaller GPU), you'd need to install the CUDA 13 runtime libs separately or pin `bitsandbytes==0.46.1` (CUDA 12 build).

### transformers strict-loading bug
- Newer transformers (5.x dev) raises `RuntimeError` when its loading report contains CONVERSION entries — even though those are harmless for our use case.
- Triggered when `sft_eval.py` calls `PeftModel.from_pretrained` after loading the base. Same code in `sft_train.py` doesn't trigger it (different call path).
- **Workaround**: use `sft_eval_vllm.py` instead. vLLM has its own loading path that bypasses the issue entirely.

### Disk-full crash on vLLM first run
- vLLM JIT-compiles FlashInfer kernels into `/root/.cache/flashinfer/` on first run.
- With the default 20 GB container disk this fills up fast.
- **Fixes (both)**: bigger container disk (50 GB) + redirect caches to `/workspace` via the env vars in §2. The `enforce_eager=True` flag in our vLLM calls also disables the heaviest JIT pass.

### `python scripts/prompt_eval.py --val-set ...` picks up too many prompts
- The script does `prompts/P*.txt` glob → picks up all 14 P*.txt files including v2/v3 variants and P9.
- **Workaround**: temporarily move variants to `prompts/extras/`, run eval, move back. See §7.

### `git pull` rejected mid-session
- If you merge new annotations into `eval_labelled_merged.json` on the pod and your friend pushed corrections to the same file, pull will refuse.
- **Workaround**: `git checkout -- data/eval_labelled_merged.json` to discard local change, then `git pull`, then re-run the merge script.

### GitHub auth on pod
- HTTPS push asks for username/password but GitHub disabled passwords.
- **Workaround**: generate a Personal Access Token (Settings → Developer settings → Tokens classic, `repo` scope), use it as the password. Cache it: `git config --global credential.helper 'cache --timeout=86400'`.

---

## 10. Pod lifecycle

- **Stop pod** (not terminate) between sessions if you'll be back within a day — keeps the per-pod container disk state, charges $0.006/hr for idle disk.
- **Terminate pod** when done — fully tears down the pod. Your `/workspace` network volume persists separately (~$7/month).
- **Restart**: the IP/port change every time. Update `~/.ssh/config` on local before retrying SSH.
