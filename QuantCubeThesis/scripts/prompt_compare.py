"""
Prompt Comparison Script
=========================
Runs all specified prompts against both base and SFT model in a single session.
Loads the model once, iterates over prompts.

Usage:
    VLLM_USE_FLASHINFER_SAMPLER=0 python scripts/prompt_compare.py \
        --adapter models/best/checkpoint-258 \
        --eval-set data/eval_labelled_merged_corrected.json \
        --prompts prompts/P5_v27.txt prompts/P5_v28.txt prompts/P5_v26_2.txt
"""
import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

PROJECT_ROOT = Path(__file__).parent.parent

LABEL_SCHEMA = {
    "topic":      {"type": "multi",  "values": []},
    "tense":      {"type": "single", "values": []},
    "sentiment":  {"type": "single", "values": []},
    "commitment": {"type": "single", "values": []},
    "risk":       {"type": "single", "values": []},
    "width":      {"type": "single", "values": []},
}
GT_FIELD_MAP = {
    "top": "topic", "ten": "tense", "sen": "sentiment",
    "com": "commitment", "ris": "risk", "wid": "width",
}
PRIMARY_FIELDS = ["topic", "sentiment", "risk", "width"]


def extract_json(response):
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    for m in reversed(list(re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL))):
        try:
            parsed = json.loads(m.group())
            if any(k in parsed for k in ["sentiment", "sen", "topic", "top"]):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def normalize(rec, is_pred=True):
    if not is_pred:
        rec = {GT_FIELD_MAP.get(k, k): v for k, v in rec.items()}
    out = {}
    for field, schema in LABEL_SCHEMA.items():
        val = rec.get(field)
        if val is None or val == "":
            out[field] = "na" if schema["type"] == "single" else ["na"]
            continue
        if schema["type"] == "multi":
            out[field] = [str(v).lower().strip() for v in val] if isinstance(val, list) else [str(val).lower().strip()]
        else:
            out[field] = str(val).lower().strip()
    return out


def compute_metrics(predictions, ground_truths):
    results = {}
    for field, schema in LABEL_SCHEMA.items():
        if schema["type"] == "single":
            pairs = [(gt.get(field, "na"), p.get(field, "na"))
                     for gt, p in zip(ground_truths, predictions)
                     if gt.get(field) not in ("", None)]
            if not pairs:
                results[field] = {"f1_macro": 0, "accuracy": 0, "n": 0}
                continue
            yt, yp = zip(*pairs)
            results[field] = {
                "f1_macro":  f1_score(yt, yp, average="macro", zero_division=0),
                "accuracy":  accuracy_score(yt, yp),
                "n": len(pairs),
            }
        else:
            yts = [set(gt.get(field, ["na"])) for gt in ground_truths]
            yps = [set(p.get(field,  ["na"])) for p in predictions]
            jaccard = [len(t & p) / len(t | p) if (t | p) else 1.0 for t, p in zip(yts, yps)]
            all_labels = sorted({l for s in yts + yps for l in s})
            yb_t = [[1 if l in s else 0 for l in all_labels] for s in yts]
            yb_p = [[1 if l in s else 0 for l in all_labels] for s in yps]
            results[field] = {
                "f1_macro":     f1_score(yb_t, yb_p, average="macro", zero_division=0),
                "jaccard_mean": sum(jaccard) / len(jaccard),
                "accuracy":     sum(1 for t, p in zip(yts, yps) if t == p) / len(yts),
                "n": len(yts),
            }
    summary = sum(results[f]["f1_macro"] for f in PRIMARY_FIELDS) / len(PRIMARY_FIELDS)
    return results, summary


def run_eval(llm, test_data, prompt_template, lora_request, tag, results_dir):
    conversations = [[{"role": "user", "content":
                        prompt_template.replace("{sentence}", s.get("sentence") or "")
                                       .replace("{context_question}", s.get("context_question") or "")}]
                     for s in test_data]
    sampling = SamplingParams(temperature=0.01, top_p=0.95, max_tokens=256)
    outputs = llm.chat(messages=conversations, sampling_params=sampling,
                       lora_request=lora_request, use_tqdm=True)

    raw, predictions, ground_truths = [], [], []
    parse_fails = 0
    for s, o in zip(test_data, outputs):
        resp = o.outputs[0].text.strip()
        parsed = extract_json(resp) or {}
        if not parsed:
            parse_fails += 1
        raw.append({"id": s["id"], "sentence": (s.get("sentence") or "")[:100],
                    "raw_response": resp[:500], "parsed": parsed})
        predictions.append(normalize(parsed, True))
        ground_truths.append(normalize(s, False))

    metrics, summary = compute_metrics(predictions, ground_truths)
    result = {
        "tag": tag, "n": len(predictions),
        "parse_fail_rate": parse_fails / len(raw) if raw else 0,
        "summary_f1": summary, "field_metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{tag}_metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(results_dir / f"{tag}_raw.json", "w") as f:
        json.dump(raw, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="models/best/checkpoint-258")
    parser.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--eval-set", default="data/eval_labelled_merged_corrected.json")
    parser.add_argument("--max-model-len", type=int, default=2560)
    parser.add_argument("--prompts", nargs="+", required=True)
    args = parser.parse_args()

    eval_data = json.load(open(PROJECT_ROOT / args.eval_set, encoding="utf-8"))
    print(f"Eval set: {len(eval_data)} sentences")

    adapter_path = (PROJECT_ROOT / args.adapter).resolve()
    results_dir = PROJECT_ROOT / "models/best/eval_results/prompt_compare"

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
               enforce_eager=True, enable_lora=True, max_lora_rank=16, max_loras=1)
    lora_request = LoRARequest("best", 1, str(adapter_path))

    all_results = {}
    for prompt_path in args.prompts:
        p = Path(prompt_path)
        prompt_name = p.stem
        template = p.read_text(encoding="utf-8").strip()
        print(f"\n{'='*60}\nPrompt: {prompt_name}\n{'='*60}")

        print("  [SFT]")
        sft = run_eval(llm, eval_data, template, lora_request, f"sft_{prompt_name}", results_dir)
        print(f"  Summary F1: {sft['summary_f1']:.4f}  (parse fails: {sft['parse_fail_rate']:.1%})")

        print("  [Base]")
        base = run_eval(llm, eval_data, template, None, f"base_{prompt_name}", results_dir)
        print(f"  Summary F1: {base['summary_f1']:.4f}  (parse fails: {base['parse_fail_rate']:.1%})")

        all_results[prompt_name] = {"sft": sft, "base": base}

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("SUMMARY TABLE — Summary F1 (topic+sentiment+risk+width macro avg)")
    print(f"{'='*90}")
    print(f"  {'Prompt':<30} {'SFT F1':>9} {'Base F1':>9} {'Delta':>9}  {'SFT Sen F1':>12} {'Base Sen F1':>12}")
    print(f"  {'-'*82}")
    for prompt_name, res in all_results.items():
        s = res["sft"]
        b = res["base"]
        delta = s["summary_f1"] - b["summary_f1"]
        s_sen = s["field_metrics"]["sentiment"]["f1_macro"]
        b_sen = b["field_metrics"]["sentiment"]["f1_macro"]
        print(f"  {prompt_name:<30} {s['summary_f1']:>9.4f} {b['summary_f1']:>9.4f} {delta:>+9.4f}  {s_sen:>12.4f} {b_sen:>12.4f}")

    with open(results_dir / "prompt_comparison_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {results_dir}")


if __name__ == "__main__":
    main()
