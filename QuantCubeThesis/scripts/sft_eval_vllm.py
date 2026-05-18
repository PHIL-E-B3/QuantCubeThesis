"""
SFT Evaluation via vLLM (fast batched inference, supports LoRA adapters).

Drop-in replacement for sft_eval.py — avoids transformers strict-loading issues
and is ~10x faster thanks to vLLM continuous batching.

Usage:
    python scripts/sft_eval_vllm.py --adapter models/sft_p1/adapter --compare-base
    python scripts/sft_eval_vllm.py --adapter models/sft_p3/adapter
"""
import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# ── PATHS ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
EVAL_PATH = PROJECT_ROOT / "data" / "eval_labelled_merged.json"
DEFAULT_ADAPTER = PROJECT_ROOT / "models" / "sft_p3" / "adapter"

PROMPT_MAP = {
    "sft_p2": "P2_medium_3shot_final",
    "sft_p3": "P3_medium_5shot_final",
    "sft_p5": "P5_high_3shot_final",
    "sft_p7": "P7_high_5shot_final",
}

LABEL_SCHEMA = {
    "topic":      {"type": "multi",  "values": ["inflation", "labor_market", "economic_activity",
                   "macro", "financial_conditions", "monetary_policy", "boilerplate", "no_topic"]},
    "tense":      {"type": "single", "values": ["descriptive", "interpretive"]},
    "sentiment":  {"type": "single", "values": ["strongly_hawkish", "hawkish", "neutral",
                   "dovish", "strongly_dovish", "na"]},
    "horizon":    {"type": "single", "values": ["true", "false"]},
    "commitment": {"type": "single", "values": ["unconditional", "conditional", "none"]},
    "risk":       {"type": "single", "values": ["skewed_downside", "skewed_upside", "symmetric", "na"]},
    "width":      {"type": "single", "values": ["elevated", "contested", "none"]},
}

# Ground truth files use old abbreviated field names — map them to the full names
GT_FIELD_MAP = {
    "top": "topic", "ten": "tense", "sen": "sentiment",
    "hor": "horizon", "com": "commitment", "ris": "risk", "wid": "width",
}

PRIMARY_FIELDS = ["topic", "sentiment", "risk", "width"]


# ── PROMPT + PARSING ─────────────────────────────────────────────────────────

def build_prompt(template: str, sentence_data: dict) -> str:
    """Substitute sentence and context into prompt template."""
    sentence = sentence_data.get("sentence") or ""
    context_q = sentence_data.get("context_question") or ""
    prompt = template.replace("{sentence}", sentence)
    prompt = prompt.replace("{context_question}", context_q)
    return prompt


def extract_json(response: str) -> Optional[dict]:
    """Extract JSON object from model response."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    matches = list(re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL))
    for match in reversed(matches):
        try:
            parsed = json.loads(match.group())
            if any(k in parsed for k in ["top", "ten", "sen"]):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def normalize(rec: dict, is_pred: bool = True) -> dict:
    # Ground truth records use abbreviated field names — remap them first
    if not is_pred:
        rec = {GT_FIELD_MAP.get(k, k): v for k, v in rec.items()}
    out = {}
    for field, schema in LABEL_SCHEMA.items():
        val = rec.get(field)
        if val is None or val == "":
            out[field] = "na" if schema["type"] == "single" else ["na"]
            continue
        if schema["type"] == "multi":
            if isinstance(val, str):
                out[field] = [val.lower().strip()]
            elif isinstance(val, list):
                out[field] = [str(v).lower().strip() for v in val]
            else:
                out[field] = [str(val)]
        else:
            out[field] = str(val).lower().strip()
    return out


# ── METRICS ──────────────────────────────────────────────────────────────────

def compute_field_metrics(preds, gts) -> Dict:
    results = {}
    for field, schema in LABEL_SCHEMA.items():
        if schema["type"] == "single":
            y_t = [gt.get(field, "na") for gt in gts]
            y_p = [p.get(field, "na") for p in preds]
            valid = [(t, p) for t, p in zip(y_t, y_p) if t not in ("", None)]
            if not valid:
                results[field] = {"accuracy": 0, "f1_macro": 0, "precision": 0, "recall": 0, "n": 0}
                continue
            yt, yp = zip(*valid)
            results[field] = {
                "accuracy": accuracy_score(yt, yp),
                "f1_macro": f1_score(yt, yp, average="macro", zero_division=0),
                "precision": precision_score(yt, yp, average="macro", zero_division=0),
                "recall": recall_score(yt, yp, average="macro", zero_division=0),
                "n": len(valid),
            }
        else:
            yts = [set(gt.get(field, ["na"])) for gt in gts]
            yps = [set(p.get(field, ["na"])) for p in preds]
            exact = sum(1 for t, p in zip(yts, yps) if t == p)
            jaccard = [len(t & p) / len(t | p) if (t | p) else 1.0 for t, p in zip(yts, yps)]
            all_labels = sorted({l for s in yts + yps for l in s})
            y_t_bin = [[1 if l in s else 0 for l in all_labels] for s in yts]
            y_p_bin = [[1 if l in s else 0 for l in all_labels] for s in yps]
            results[field] = {
                "accuracy": exact / len(yts),
                "f1_macro": f1_score(y_t_bin, y_p_bin, average="macro", zero_division=0),
                "precision": precision_score(y_t_bin, y_p_bin, average="macro", zero_division=0),
                "recall": recall_score(y_t_bin, y_p_bin, average="macro", zero_division=0),
                "jaccard_mean": sum(jaccard) / len(jaccard),
                "n": len(yts),
            }
    return results


def compute_summary(field_metrics):
    return sum(field_metrics[f]["f1_macro"] for f in PRIMARY_FIELDS) / len(PRIMARY_FIELDS)


# ── INFERENCE ────────────────────────────────────────────────────────────────

def run_vllm_eval(llm, test_data, prompt_template, lora_request, name, results_dir):
    print(f"\n{'=' * 60}\nEvaluating: {name}\n{'=' * 60}")
    prompts = [build_prompt(prompt_template, s) for s in test_data]
    conversations = [[{"role": "user", "content": p}] for p in prompts]
    sampling = SamplingParams(temperature=0.01, top_p=0.95, max_tokens=256)

    outputs = llm.chat(
        messages=conversations,
        sampling_params=sampling,
        lora_request=lora_request,
        use_tqdm=True,
    )

    raw_outputs = []
    predictions, ground_truths = [], []
    parse_fails = 0
    for s, o in zip(test_data, outputs):
        resp = o.outputs[0].text.strip()
        parsed = extract_json(resp)
        if parsed is None:
            parse_fails += 1
            parsed = {}
        raw_outputs.append({
            "id": s["id"],
            "sentence": s["sentence"][:100],
            "raw_response": resp[:1000],
            "parsed": parsed,
        })
        predictions.append(normalize(parsed, True))
        ground_truths.append(normalize(s, False))

    field_metrics = compute_field_metrics(predictions, ground_truths)
    summary = compute_summary(field_metrics)

    result = {
        "name": name,
        "n_sentences": len(predictions),
        "parse_failure_rate": parse_fails / len(raw_outputs) if raw_outputs else 0,
        "summary_f1": summary,
        "field_metrics": field_metrics,
        "timestamp": datetime.now().isoformat(),
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{name}_raw.json", "w") as f:
        json.dump(raw_outputs, f, indent=2)
    with open(results_dir / f"{name}_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Summary F1: {summary:.4f}  ParseFail: {result['parse_failure_rate']:.1%}")
    for field, m in field_metrics.items():
        marker = " *" if field in PRIMARY_FIELDS else ""
        print(f"    {field:5s}: acc={m['accuracy']:.3f}  F1={m['f1_macro']:.3f}  (n={m['n']}){marker}")
    return result


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default=str(DEFAULT_ADAPTER))
    parser.add_argument("--model", type=str, default="unsloth/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--compare-base", action="store_true",
                        help="Also evaluate base model (no adapter)")
    parser.add_argument("--max-model-len", type=int, default=2048)
    args = parser.parse_args()

    adapter_path = Path(args.adapter).resolve()
    model_dir = adapter_path.parent
    model_name_short = model_dir.name  # e.g. "sft_p1"

    split_info_path = model_dir / "split_info.json"
    results_dir = model_dir / "eval_results"

    prompt_key = PROMPT_MAP.get(model_name_short, "P3_medium_5shot")
    prompt_path = PROJECT_ROOT / "prompts" / f"{prompt_key}.txt"
    print(f"Adapter:    {adapter_path}")
    print(f"Prompt:     {prompt_key}")

    with open(split_info_path) as f:
        split_info = json.load(f)
    test_ids = set(split_info["test_ids"])

    with open(EVAL_PATH, encoding="utf-8") as f:
        all_data = json.load(f)
    test_data = [s for s in all_data if s["id"] in test_ids]
    print(f"Test set:   {len(test_data)} sentences")
    print(f"SEN dist:   {dict(sorted(Counter(str(s.get('sen')) for s in test_data).items()))}")

    prompt_template = prompt_path.read_text(encoding="utf-8").strip()

    # Load vLLM once with LoRA support
    print(f"\nLoading vLLM (model={args.model})...")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        enforce_eager=True,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=1,
    )

    # SFT eval (with LoRA adapter)
    lora_request = LoRARequest(model_name_short, 1, str(adapter_path))
    sft_result = run_vllm_eval(llm, test_data, prompt_template, lora_request,
                               f"sft_{model_name_short}", results_dir)

    # Optional base eval (no LoRA)
    if args.compare_base:
        base_result = run_vllm_eval(llm, test_data, prompt_template, None,
                                    f"base_{model_name_short}", results_dir)

        print(f"\n{'=' * 60}\nHEAD-TO-HEAD: SFT vs Base\n{'=' * 60}")
        print(f"{'Field':<8} {'Base F1':>10} {'SFT F1':>10} {'Delta':>10}")
        print("-" * 40)
        for field in LABEL_SCHEMA:
            b = base_result["field_metrics"].get(field, {}).get("f1_macro", 0)
            s = sft_result["field_metrics"].get(field, {}).get("f1_macro", 0)
            marker = " *" if field in PRIMARY_FIELDS else ""
            print(f"{field:<8} {b:>10.3f} {s:>10.3f} {s - b:>+10.3f}{marker}")
        print("-" * 40)
        print(f"{'Summary':<8} {base_result['summary_f1']:>10.4f} "
              f"{sft_result['summary_f1']:>10.4f} "
              f"{sft_result['summary_f1'] - base_result['summary_f1']:>+10.4f}")

        comparison = {
            "base": base_result,
            "sft": sft_result,
            "delta_summary_f1": sft_result["summary_f1"] - base_result["summary_f1"],
            "timestamp": datetime.now().isoformat(),
        }
        with open(results_dir / "comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)

    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    main()
