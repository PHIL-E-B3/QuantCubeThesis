"""
Uncertainty Validation Script
==============================
Validates whether the mean negative log-probability of generated tokens
is a meaningful uncertainty signal: sentences the model gets wrong should
be more uncertain than sentences it gets right.

Usage:
    VLLM_USE_FLASHINFER_SAMPLER=0 python scripts/uncertainty_eval.py \
        --adapter models/best/checkpoint-258 \
        --eval-set data/eval_labelled_merged_corrected.json \
        --prompt prompts/P5_v27.txt
"""
import argparse
import json
import re
import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy import stats as scipy_stats
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

PROJECT_ROOT = Path(__file__).parent.parent

SINGLE_FIELDS = {
    "sentiment":  "sen",
    "tense":      "ten",
    "commitment": "com",
    "risk":       "ris",
    "width":      "wid",
}


def extract_json(response):
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    for match in reversed(list(re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL))):
        try:
            parsed = json.loads(match.group())
            if any(k in parsed for k in ["sentiment", "sen", "topic", "top"]):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def mean_neg_logprob(output):
    """Mean negative log-probability of generated tokens — higher = more uncertain."""
    logprobs_list = output.outputs[0].logprobs
    token_ids = output.outputs[0].token_ids
    if not logprobs_list:
        return float("nan")
    lps = [
        logprobs_list[i][tid].logprob
        for i, tid in enumerate(token_ids)
        if logprobs_list[i] and tid in logprobs_list[i]
    ]
    return -np.mean(lps) if lps else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="models/best/checkpoint-258")
    parser.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--eval-set", default="data/eval_labelled_merged_corrected.json")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-model-len", type=int, default=2560)
    parser.add_argument("--output", default="models/best/eval_results/uncertainty_scores.json")
    args = parser.parse_args()

    prompt_template = Path(args.prompt).read_text(encoding="utf-8").strip()
    with open(args.eval_set, encoding="utf-8") as f:
        eval_data = json.load(f)
    print(f"Loaded {len(eval_data)} sentences")

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        enforce_eager=True,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=1,
    )
    lora_request = LoRARequest("best", 1, str(Path(args.adapter).resolve()))
    sampling = SamplingParams(temperature=0.01, top_p=0.95, max_tokens=256, logprobs=1)

    conversations = []
    for s in eval_data:
        p = prompt_template.replace("{sentence}", s.get("sentence") or "")
        p = p.replace("{context_question}", s.get("context_question") or "")
        conversations.append([{"role": "user", "content": p}])

    print("Running inference with logprobs...")
    outputs = llm.chat(messages=conversations, sampling_params=sampling,
                       lora_request=lora_request, use_tqdm=True)

    results = []
    for s, o in zip(eval_data, outputs):
        parsed = extract_json(o.outputs[0].text.strip()) or {}
        results.append({
            "id": s["id"],
            "sentence": (s.get("sentence") or "")[:150],
            "uncertainty": mean_neg_logprob(o),
            "parsed": parsed,
            "gt": {k: s.get(k) for k in ["top", "ten", "sen", "com", "hor", "ris", "wid"]},
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Scores saved to {args.output}")

    # ── Correct vs Incorrect ──────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("UNCERTAINTY: CORRECT vs INCORRECT  (Mann-Whitney U, one-sided)")
    print(f"{'='*65}")
    print(f"{'Field':<14} {'Correct μ':>10} {'Wrong μ':>10} {'Δ':>8} {'p-value':>10}  sig?")
    print("-" * 65)

    for field, gt_key in SINGLE_FIELDS.items():
        correct_u, wrong_u = [], []
        for r in results:
            gt = str(r["gt"].get(gt_key, "na")).lower().strip()
            pred = str(r["parsed"].get(field, "na")).lower().strip()
            u = r["uncertainty"]
            if np.isnan(u):
                continue
            (correct_u if gt == pred else wrong_u).append(u)

        if correct_u and wrong_u:
            _, p = scipy_stats.mannwhitneyu(wrong_u, correct_u, alternative="greater")
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            print(f"{field:<14} {np.mean(correct_u):>10.4f} {np.mean(wrong_u):>10.4f} "
                  f"{np.mean(wrong_u)-np.mean(correct_u):>+8.4f} {p:>10.4f}  {sig}")

    # ── Per-class uncertainty breakdown ───────────────────────────────────────
    print(f"\n{'='*65}")
    print("UNCERTAINTY BY CLASS  (mean ± std, with per-class accuracy)")
    print(f"{'='*65}")

    for field, gt_key in SINGLE_FIELDS.items():
        class_u = defaultdict(list)
        class_correct = defaultdict(int)
        for r in results:
            gt = str(r["gt"].get(gt_key, "na")).lower().strip()
            pred = str(r["parsed"].get(field, "na")).lower().strip()
            u = r["uncertainty"]
            if np.isnan(u) or gt in ("na", ""):
                continue
            class_u[gt].append(u)
            if gt == pred:
                class_correct[gt] += 1

        print(f"\n  {field}:")
        print(f"  {'Class':<22} {'Mean U':>8} {'Std':>7} {'Acc':>6} {'n':>5}")
        print(f"  {'-'*48}")
        for cls, us in sorted(class_u.items(), key=lambda x: -np.mean(x[1])):
            acc = class_correct[cls] / len(us)
            print(f"  {cls:<22} {np.mean(us):>8.4f} {np.std(us):>7.4f} {acc:>6.2f} {len(us):>5}")

    # ── Most uncertain sentences ───────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("TOP 15 MOST UNCERTAIN SENTENCES")
    print(f"{'='*65}")
    sorted_r = sorted([r for r in results if not np.isnan(r["uncertainty"])],
                      key=lambda x: -x["uncertainty"])
    for r in sorted_r[:15]:
        sen_pred = r["parsed"].get("sentiment", "?")
        sen_gt   = str(r["gt"].get("sen", "?")).lower()
        correct  = "✓" if sen_pred == sen_gt else "✗"
        print(f"  U={r['uncertainty']:.4f} sen:{sen_gt}->{sen_pred}{correct}  "
              f"{r['sentence'][:90]}")


if __name__ == "__main__":
    main()
