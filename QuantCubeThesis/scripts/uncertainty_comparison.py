"""
Uncertainty Method Comparison
==============================
Compares three uncertainty signals on the eval set:
  1. Label-token logprob  — -logprob of the specific field label token only
  2. Sampling consistency — entropy over 5 predictions at T=0.5
  3. Checkpoint ensemble  — disagreement across checkpoint-86 / 172 / 258

Baseline (mean neg-logprob over all tokens) included for comparison.

Usage:
    VLLM_USE_FLASHINFER_SAMPLER=0 python scripts/uncertainty_comparison.py \
        --eval-set data/eval_labelled_merged_corrected.json \
        --prompt prompts/P5_v27.txt
"""
import argparse
import json
import re
import numpy as np
from collections import Counter
from math import log
from pathlib import Path
from scipy import stats as scipy_stats
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

PROJECT_ROOT = Path(__file__).parent.parent

CHECKPOINTS = {
    "cp86":  "models/best/checkpoint-86",
    "cp172": "models/best/checkpoint-172",
    "cp258": "models/best/checkpoint-258",
}

SINGLE_FIELDS = {
    "sentiment":  "sen",
    "risk":       "ris",
    "width":      "wid",
    "commitment": "com",
    "tense":      "ten",
}

N_SAMPLES = 5
SAMPLE_TEMP = 0.5


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for m in reversed(list(re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL))):
        try:
            parsed = json.loads(m.group())
            if any(k in parsed for k in ["sentiment", "sen", "topic"]):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def get_label(parsed, field):
    if parsed is None:
        return "na"
    return str(parsed.get(field, "na")).lower().strip()


def mean_neg_lp(output):
    lps_list = output.outputs[0].logprobs
    tids = output.outputs[0].token_ids
    if not lps_list:
        return float("nan")
    lps = [lps_list[i][tid].logprob for i, tid in enumerate(tids)
           if lps_list[i] and tid in lps_list[i]]
    return -np.mean(lps) if lps else float("nan")


def label_token_neg_lp(output, text, field, tokenizer):
    """
    Extract -logprob of the first token of the predicted label value
    for a specific field (e.g. 'hawkish' in 'sentiment': 'hawkish').
    Uses fast-tokenizer offset mapping for robust character alignment.
    """
    lps_list = output.outputs[0].logprobs
    tids = output.outputs[0].token_ids
    if not lps_list or not text:
        return float("nan")

    m = re.search(rf'"{re.escape(field)}"\s*:\s*"', text)
    if not m:
        return float("nan")
    label_char_start = m.end()

    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
    except Exception:
        return float("nan")

    for i, (start, end) in enumerate(offsets):
        if start <= label_char_start < end or start == label_char_start:
            if i < len(lps_list) and lps_list[i] and tids[i] in lps_list[i]:
                return -lps_list[i][tids[i]].logprob
            return float("nan")
    return float("nan")


def label_entropy(labels):
    """Shannon entropy over a list of label strings (e.g. 5 sample predictions)."""
    counts = Counter(labels)
    n = len(labels)
    return -sum((c / n) * log(c / n) for c in counts.values() if c > 0)


def ensemble_disagreement(label_list):
    """
    Proportion of checkpoints that disagree with the majority vote.
    0 = all agree, 1 = maximum disagreement.
    """
    counts = Counter(label_list)
    majority_count = counts.most_common(1)[0][1]
    n = len(label_list)
    return 1.0 - majority_count / n


# ── Inference helpers ─────────────────────────────────────────────────────────

def run_inference(llm, conversations, lora_request, sampling):
    return llm.chat(messages=conversations, sampling_params=sampling,
                    lora_request=lora_request, use_tqdm=True)


def build_conversations(eval_data, prompt_template):
    convs = []
    for s in eval_data:
        p = prompt_template.replace("{sentence}", s.get("sentence") or "")
        p = p.replace("{context_question}", s.get("context_question") or "")
        convs.append([{"role": "user", "content": p}])
    return convs


# ── Analysis ──────────────────────────────────────────────────────────────────

def mwu_p(wrong_u, correct_u):
    if not wrong_u or not correct_u:
        return float("nan")
    _, p = scipy_stats.mannwhitneyu(wrong_u, correct_u, alternative="greater")
    return p


def sig_stars(p):
    if np.isnan(p):  return "n/a"
    if p < 0.001:    return "***"
    if p < 0.01:     return "**"
    if p < 0.05:     return "*"
    return ""


def analyse(records, gt_map, method_keys):
    """
    records: list of dicts with id, gt, and one key per method containing uncertainty score.
    Returns per-field p-values and per-class sentiment breakdown.
    """
    # ── Correct vs Incorrect p-values ─────────────────────────────────────────
    print(f"\n{'='*80}")
    print("CORRECT vs INCORRECT  (Mann-Whitney U, one-sided, higher U = more uncertain)")
    print(f"{'='*80}")
    header = f"{'Field':<14}" + "".join(f"{k:>16}" for k in method_keys)
    print(header)
    print("-" * (14 + 16 * len(method_keys)))

    for field, gt_key in SINGLE_FIELDS.items():
        correct = {k: [] for k in method_keys}
        wrong   = {k: [] for k in method_keys}
        for r in records:
            gt   = str(r["gt"].get(gt_key, "na")).lower().strip()
            pred = str(r["preds"].get("cp258", {}).get(field, "na")).lower().strip()
            for k in method_keys:
                u = r[k].get(field, float("nan"))
                if np.isnan(u):
                    continue
                (correct[k] if gt == pred else wrong[k]).append(u)

        row = f"{field:<14}"
        for k in method_keys:
            p = mwu_p(wrong[k], correct[k])
            row += f"  {p:>7.4f} {sig_stars(p):>4}"
        print(row)

    # ── Sentiment per-class ───────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("SENTIMENT PER-CLASS UNCERTAINTY  (mean U per true label, with accuracy)")
    print(f"{'='*80}")
    header = f"  {'Class':<22}" + "".join(f"{k:>16}" for k in method_keys) + f"  {'Acc':>6}  {'n':>5}"
    print(header)
    print(f"  {'-'*( 22 + 16*len(method_keys) + 14)}")

    class_data = {}
    for r in records:
        gt = str(r["gt"].get("sen", "na")).lower().strip()
        pred = str(r["preds"].get("cp258", {}).get("sentiment", "na")).lower().strip()
        if gt in ("na", ""):
            continue
        if gt not in class_data:
            class_data[gt] = {"correct": 0, "total": 0, **{k: [] for k in method_keys}}
        class_data[gt]["total"] += 1
        if gt == pred:
            class_data[gt]["correct"] += 1
        for k in method_keys:
            u = r[k].get("sentiment", float("nan"))
            if not np.isnan(u):
                class_data[gt][k].append(u)

    for cls in ["neutral", "dovish", "hawkish", "strongly_dovish", "strongly_hawkish"]:
        if cls not in class_data:
            continue
        d = class_data[cls]
        acc = d["correct"] / d["total"] if d["total"] else 0
        row = f"  {cls:<22}"
        for k in method_keys:
            mu = np.mean(d[k]) if d[k] else float("nan")
            row += f"  {mu:>14.4f}"
        row += f"  {acc:>6.2f}  {d['total']:>5}"
        print(row)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--eval-set", default="data/eval_labelled_merged_corrected.json")
    parser.add_argument("--prompt", default="prompts/P5_v27.txt")
    parser.add_argument("--max-model-len", type=int, default=2560)
    parser.add_argument("--output", default="models/best/eval_results/uncertainty_comparison.json")
    args = parser.parse_args()

    prompt_template = Path(args.prompt).read_text(encoding="utf-8").strip()
    with open(args.eval_set, encoding="utf-8") as f:
        eval_data = json.load(f)
    print(f"Loaded {len(eval_data)} sentences")

    conversations = build_conversations(eval_data, prompt_template)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        enforce_eager=True,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=3,
    )
    tokenizer = llm.get_tokenizer()

    lora_requests = {
        name: LoRARequest(name, i + 1, str((PROJECT_ROOT / path).resolve()))
        for i, (name, path) in enumerate(CHECKPOINTS.items())
    }

    # ── Option 1 + Baseline: single pass with logprobs (checkpoint-258) ───────
    print("\n[1/3] Option 1 — label-token logprob (checkpoint-258, logprobs=1)...")
    sp_lp = SamplingParams(temperature=0.01, top_p=0.95, max_tokens=256, logprobs=1)
    out_258 = run_inference(llm, conversations, lora_requests["cp258"], sp_lp)

    # ── Option 2: N samples at T=0.5 (checkpoint-258) ────────────────────────
    print(f"\n[2/3] Option 2 — sampling consistency ({N_SAMPLES} samples, T={SAMPLE_TEMP})...")
    sp_sample = SamplingParams(temperature=SAMPLE_TEMP, top_p=0.95, max_tokens=256)
    sample_outputs = []
    for i in range(N_SAMPLES):
        print(f"  Sample {i+1}/{N_SAMPLES}...")
        sample_outputs.append(run_inference(llm, conversations, lora_requests["cp258"], sp_sample))

    # ── Option 3: checkpoint ensemble ─────────────────────────────────────────
    print("\n[3/3] Option 3 — checkpoint ensemble (86 / 172 / 258)...")
    sp_greedy = SamplingParams(temperature=0.01, top_p=0.95, max_tokens=256)
    ensemble_outputs = {}
    for name, lora_req in lora_requests.items():
        print(f"  Running {name}...")
        ensemble_outputs[name] = run_inference(llm, conversations, lora_req, sp_greedy)

    # ── Build records ─────────────────────────────────────────────────────────
    print("\nComputing uncertainty scores...")
    records = []
    for idx, s in enumerate(eval_data):
        o258 = out_258[idx]
        text258 = o258.outputs[0].text.strip()
        parsed258 = extract_json(text258) or {}

        # Baseline & Option 1 per field
        baseline_u, label_lp_u = {}, {}
        baseline_mean = mean_neg_lp(o258)
        for field in SINGLE_FIELDS:
            baseline_u[field] = baseline_mean
            label_lp_u[field] = label_token_neg_lp(o258, text258, field, tokenizer)

        # Option 2: sampling entropy per field
        sample_preds = []
        for so in sample_outputs:
            p = extract_json(so[idx].outputs[0].text.strip()) or {}
            sample_preds.append(p)
        sampling_u = {}
        for field in SINGLE_FIELDS:
            labels = [get_label(p, field) for p in sample_preds]
            sampling_u[field] = label_entropy(labels)

        # Option 3: ensemble disagreement per field
        ens_preds = {name: extract_json(ensemble_outputs[name][idx].outputs[0].text.strip()) or {}
                     for name in CHECKPOINTS}
        ensemble_u = {}
        for field in SINGLE_FIELDS:
            labels = [get_label(ens_preds[name], field) for name in CHECKPOINTS]
            ensemble_u[field] = ensemble_disagreement(labels)

        records.append({
            "id": s["id"],
            "sentence": (s.get("sentence") or "")[:120],
            "gt": {k: s.get(k) for k in ["top", "ten", "sen", "com", "hor", "ris", "wid"]},
            "preds": {"cp258": parsed258, **{n: ens_preds[n] for n in CHECKPOINTS}},
            "baseline":  baseline_u,
            "label_lp":  label_lp_u,
            "sampling":  sampling_u,
            "ensemble":  ensemble_u,
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Scores saved to {args.output}")

    METHOD_KEYS = ["baseline", "label_lp", "sampling", "ensemble"]
    analyse(records, None, METHOD_KEYS)


if __name__ == "__main__":
    main()
