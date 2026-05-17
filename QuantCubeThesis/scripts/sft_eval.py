"""
SFT Evaluation Script
======================
Evaluates the fine-tuned LoRA adapter on the held-out test set,
computes per-field metrics, and compares against the base model.

Usage:
    python scripts/sft_eval.py
    python scripts/sft_eval.py --adapter models/sft_p3/adapter
    python scripts/sft_eval.py --compare-base   # also run base model for head-to-head
"""

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# ── PATHS ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
EVAL_PATH = PROJECT_ROOT / "data" / "eval_labelled_merged.json"
DEFAULT_ADAPTER = PROJECT_ROOT / "models" / "sft_p3" / "adapter"

# Prompt and output dirs are resolved dynamically from the adapter path in main()
PROMPT_MAP = {
    "sft_p1": "P1_medium_3shot",
    "sft_p3": "P3_medium_5shot",
    "sft_p5": "P5_high_3shot",
    "sft_p7": "P7_high_5shot",
}

LABEL_SCHEMA = {
    "top": {"type": "multi", "values": ["inflation", "unemployment", "economic_activity",
            "macro", "financial_conditions", "monetary_policy", "boilerplate", "no_topic"]},
    "ten": {"type": "single", "values": ["descriptive", "interpretive"]},
    "sen": {"type": "single", "values": ["-2", "-1", "0", "1", "2", "na"]},
    "com": {"type": "single", "values": ["unconditional", "conditional", "none"]},
    "hor": {"type": "single", "values": ["True", "False"]},
    "ris": {"type": "single", "values": ["skewed_downside", "skewed_upside", "symmetric", "na"]},
    "wid": {"type": "single", "values": ["elevated", "contested", "none"]},
}

PRIMARY_FIELDS = ["top", "sen", "ris", "wid"]
LABEL_FIELDS = ["top", "ten", "sen", "hor", "com", "ris", "wid"]


# ── MODEL LOADING ────────────────────────────────────────────────────────────

def load_base_model(model_name: str):
    """Load base model in bf16 (no quantization — CUDA 13 lib conflict workaround)."""
    print(f"\nLoading base model: {model_name}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    print(f"Base model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    return model, tokenizer


def load_sft_model(adapter_path: str, model_name: str):
    """Load base model + LoRA adapter."""
    model, tokenizer = load_base_model(model_name)

    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    trainable, total = model.get_nb_trainable_parameters()
    print(f"Adapter loaded. Trainable: {trainable:,} / {total:,}")
    print(f"VRAM after adapter: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    return model, tokenizer


# ── INFERENCE ────────────────────────────────────────────────────────────────

def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate a single response (near-deterministic)."""
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.01,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


# ── JSON PARSING ─────────────────────────────────────────────────────────────

def extract_json(response: str) -> Optional[dict]:
    """Extract JSON from model response."""
    # Try direct parse first (SFT model should output clean JSON)
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try finding JSON object
    json_matches = list(re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL))
    if json_matches:
        for match in reversed(json_matches):
            try:
                parsed = json.loads(match.group())
                if any(k in parsed for k in ["top", "ten", "sen"]):
                    return parsed
            except json.JSONDecodeError:
                continue

    return None


def normalize_prediction(pred: dict) -> dict:
    """Normalize parsed prediction to match expected format."""
    normalized = {}
    for field, schema in LABEL_SCHEMA.items():
        val = pred.get(field)
        if val is None:
            normalized[field] = "na" if schema["type"] == "single" else ["na"]
            continue

        if schema["type"] == "multi":
            if isinstance(val, str):
                normalized[field] = [val.lower().strip()] if val != "na" else ["na"]
            elif isinstance(val, list):
                normalized[field] = [str(v).lower().strip() for v in val]
            else:
                normalized[field] = [str(val)]
        else:
            normalized[field] = str(val).lower().strip()

    return normalized


def normalize_ground_truth(gt: dict) -> dict:
    """Normalize ground truth labels."""
    normalized = {}
    for field, schema in LABEL_SCHEMA.items():
        val = gt.get(field)
        if val is None or val == "":
            normalized[field] = "na" if schema["type"] == "single" else ["na"]
            continue

        if schema["type"] == "multi":
            if isinstance(val, str):
                normalized[field] = [val.lower().strip()]
            elif isinstance(val, list):
                normalized[field] = [str(v).lower().strip() for v in val]
            else:
                normalized[field] = [str(val)]
        else:
            normalized[field] = str(val).lower().strip()

    return normalized


# ── METRICS ──────────────────────────────────────────────────────────────────

def compute_field_metrics(predictions, ground_truths) -> Dict:
    """Compute per-field accuracy, precision, recall, F1."""
    results = {}

    for field, schema in LABEL_SCHEMA.items():
        if schema["type"] == "single":
            y_true = [gt.get(field, "na") for gt in ground_truths]
            y_pred = [pred.get(field, "na") for pred in predictions]

            valid = [(t, p) for t, p in zip(y_true, y_pred) if t not in ("", None)]
            if not valid:
                results[field] = {"accuracy": 0, "f1_macro": 0, "precision": 0, "recall": 0, "n": 0}
                continue

            y_true_v, y_pred_v = zip(*valid)
            results[field] = {
                "accuracy": accuracy_score(y_true_v, y_pred_v),
                "f1_macro": f1_score(y_true_v, y_pred_v, average="macro", zero_division=0),
                "precision": precision_score(y_true_v, y_pred_v, average="macro", zero_division=0),
                "recall": recall_score(y_true_v, y_pred_v, average="macro", zero_division=0),
                "n": len(valid),
            }
        else:
            # Multi-label: set-based exact match and Jaccard
            y_true_sets = [set(gt.get(field, ["na"])) for gt in ground_truths]
            y_pred_sets = [set(pred.get(field, ["na"])) for pred in predictions]

            exact_matches = sum(1 for t, p in zip(y_true_sets, y_pred_sets) if t == p)
            jaccard_scores = [
                len(t & p) / len(t | p) if len(t | p) > 0 else 1.0
                for t, p in zip(y_true_sets, y_pred_sets)
            ]

            n = len(y_true_sets)
            results[field] = {
                "accuracy": exact_matches / n if n > 0 else 0,
                "f1_macro": float(np.mean(jaccard_scores)) if jaccard_scores else 0,
                "precision": float(np.mean(jaccard_scores)) if jaccard_scores else 0,
                "recall": float(np.mean(jaccard_scores)) if jaccard_scores else 0,
                "n": n,
            }

    return results


def compute_summary_f1(field_metrics: Dict) -> float:
    """Summary F1: unweighted mean of F1-macro across primary fields."""
    total_f1 = 0
    count = 0
    for field in PRIMARY_FIELDS:
        if field in field_metrics and field_metrics[field]["n"] > 0:
            total_f1 += field_metrics[field]["f1_macro"]
            count += 1
    return total_f1 / count if count > 0 else 0


# ── EVALUATION ───────────────────────────────────────────────────────────────

def run_eval(model, tokenizer, test_data, prompt_template, label: str, results_dir: Path):
    """Run inference on test set and compute metrics."""
    print(f"\n{'=' * 60}")
    print(f"EVALUATING: {label}")
    print(f"{'=' * 60}")
    print(f"Test sentences: {len(test_data)}")

    raw_outputs = []
    predictions = []
    ground_truths = []
    parse_failures = 0

    # Check for resume
    raw_path = results_dir / f"{label}_raw.json"
    completed_ids = set()
    if raw_path.exists():
        with open(raw_path) as f:
            existing = json.load(f)
        completed_ids = {r["id"] for r in existing}
        raw_outputs = existing
        print(f"  Resuming: {len(completed_ids)} already completed")

    remaining = [s for s in test_data if s["id"] not in completed_ids]

    for i, ex in enumerate(remaining):
        idx = len(completed_ids) + i + 1
        prompt = prompt_template.replace("{sentence}", ex["sentence"])

        t0 = time.time()
        try:
            response = generate_response(model, tokenizer, prompt)
        except Exception as e:
            print(f"  [{idx}/{len(test_data)}] ERROR: {e}")
            response = ""
        elapsed = time.time() - t0

        parsed = extract_json(response)
        if parsed is None:
            parse_failures += 1
            status = "PARSE FAIL"
            parsed = {}
        else:
            status = "OK"

        print(f"  [{idx}/{len(test_data)}] {status} — {elapsed:.1f}s")

        raw_outputs.append({
            "id": ex["id"],
            "sentence": ex["sentence"][:100],
            "raw_response": response[:500],
            "parsed": parsed,
            "elapsed_s": round(elapsed, 1),
        })

        # Save incrementally
        with open(raw_path, "w") as f:
            json.dump(raw_outputs, f, indent=2)

    # Compute metrics over all results
    all_parsed = {r["id"]: r.get("parsed", {}) for r in raw_outputs}

    for ex in test_data:
        if ex["id"] in all_parsed:
            gt = normalize_ground_truth(ex)
            pred = normalize_prediction(all_parsed[ex["id"]])
            ground_truths.append(gt)
            predictions.append(pred)

    field_metrics = compute_field_metrics(predictions, ground_truths)
    summary_f1 = compute_summary_f1(field_metrics)

    total_parse_failures = sum(1 for r in raw_outputs if not r.get("parsed"))

    result = {
        "label": label,
        "n_sentences": len(predictions),
        "parse_failure_rate": total_parse_failures / len(raw_outputs) if raw_outputs else 0,
        "summary_f1": summary_f1,
        "field_metrics": field_metrics,
        "timestamp": datetime.now().isoformat(),
    }

    # Save metrics
    with open(results_dir / f"{label}_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    print(f"\n  Summary F1 (primary fields): {summary_f1:.4f}")
    print(f"  Parse failure rate: {total_parse_failures}/{len(raw_outputs)}")
    print(f"  Per-field results:")
    for field, m in field_metrics.items():
        marker = " *" if field in PRIMARY_FIELDS else ""
        print(f"    {field:5s}: acc={m['accuracy']:.3f}  F1={m['f1_macro']:.3f}  "
              f"P={m['precision']:.3f}  R={m['recall']:.3f}{marker}")

    # SEN confusion matrix
    print(f"\n  SEN confusion matrix (rows=truth, cols=pred):")
    sen_vals = ["-2", "-1", "0", "1", "2", "na"]
    cm = defaultdict(lambda: defaultdict(int))
    for ex in test_data:
        sid = ex["id"]
        if sid not in all_parsed:
            continue
        tv = str(ex.get("sen"))
        pv = str(all_parsed[sid].get("sen", "FAIL"))
        cm[tv][pv] += 1

    header = f"  {'':>6}" + "".join(f"{v:>6}" for v in sen_vals)
    print(header)
    for tv in sen_vals:
        row = f"  {tv:>6}"
        for pv in sen_vals:
            row += f"{cm[tv][pv]:>6}"
        print(row)

    # SEN error direction
    hawk = dove = other = total_err = 0
    for ex in test_data:
        sid = ex["id"]
        if sid not in all_parsed:
            continue
        true_sen = ex.get("sen")
        pred_sen = all_parsed[sid].get("sen")
        try:
            tn = int(true_sen) if str(true_sen) != "na" else None
            pn = int(pred_sen) if str(pred_sen) != "na" else None
        except (ValueError, TypeError):
            continue
        if tn is None and pn is None:
            continue
        if tn is None or pn is None:
            total_err += 1
            other += 1
            continue
        if pn == tn:
            continue
        total_err += 1
        if pn > tn:
            hawk += 1
        else:
            dove += 1

    print(f"\n  SEN error direction: hawk={hawk} dove={dove} other={other} "
          f"total_err={total_err} hawk%={hawk/total_err*100:.0f}%" if total_err else "")

    return result


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SFT Evaluation on Held-Out Test Set")
    parser.add_argument("--adapter", type=str, default=str(DEFAULT_ADAPTER),
                        help="Path to LoRA adapter directory")
    parser.add_argument("--model", type=str,
                        default="unsloth/Meta-Llama-3.1-8B-Instruct",
                        help="Base model name")
    parser.add_argument("--compare-base", action="store_true",
                        help="Also evaluate base model (no adapter) for comparison")
    args = parser.parse_args()

    # Derive paths from adapter location: models/sft_p3/adapter → model_dir = models/sft_p3
    adapter_path = Path(args.adapter)
    model_dir = adapter_path.parent  # e.g. models/sft_p7
    model_name_short = model_dir.name  # e.g. "sft_p7"

    split_info_path = model_dir / "split_info.json"
    results_dir = model_dir / "eval_results"

    # Resolve prompt template
    prompt_key = PROMPT_MAP.get(model_name_short, "P3_medium_5shot")
    prompt_path = PROJECT_ROOT / "prompts" / f"{prompt_key}.txt"
    print(f"Model dir:  {model_dir}")
    print(f"Prompt:     {prompt_key}")

    # Load split info
    if split_info_path.exists():
        with open(split_info_path) as f:
            split_info = json.load(f)
        test_ids = set(split_info["test_ids"])
        print(f"Loaded split info: {split_info['train_size']} train, {split_info['test_size']} test")
    else:
        print(f"ERROR: split_info.json not found at {split_info_path}. Run sft_train.py first.")
        return

    # Load eval data and filter to test set
    with open(EVAL_PATH, encoding="utf-8") as f:
        all_data = json.load(f)

    test_data = [s for s in all_data if s["id"] in test_ids]
    print(f"Test set: {len(test_data)} sentences")

    # Label distribution
    sen_dist = Counter(str(s.get("sen")) for s in test_data)
    print(f"SEN distribution: {dict(sorted(sen_dist.items()))}")

    # Load prompt
    prompt_template = prompt_path.read_text(encoding="utf-8").strip()

    # Setup results dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Evaluate SFT model ──
    sft_model, tokenizer = load_sft_model(args.adapter, args.model)
    sft_result = run_eval(sft_model, tokenizer, test_data, prompt_template,
                          f"sft_{model_name_short}", results_dir)

    # Free VRAM
    del sft_model
    torch.cuda.empty_cache()

    # ── Optionally evaluate base model ──
    if args.compare_base:
        base_model, tokenizer = load_base_model(args.model)
        base_result = run_eval(base_model, tokenizer, test_data, prompt_template,
                               f"base_{model_name_short}", results_dir)

        del base_model
        torch.cuda.empty_cache()

        # ── Head-to-head comparison ──
        print(f"\n{'=' * 60}")
        print("HEAD-TO-HEAD COMPARISON (SFT vs Base)")
        print(f"{'=' * 60}")
        print(f"{'Field':<8} {'Base Acc':>10} {'SFT Acc':>10} {'Base F1':>10} {'SFT F1':>10} {'Delta F1':>10}")
        print("-" * 58)
        for field in LABEL_SCHEMA:
            b = base_result["field_metrics"].get(field, {})
            s = sft_result["field_metrics"].get(field, {})
            delta = s.get("f1_macro", 0) - b.get("f1_macro", 0)
            marker = " *" if field in PRIMARY_FIELDS else ""
            print(f"{field:<8} {b.get('accuracy', 0):>10.3f} {s.get('accuracy', 0):>10.3f} "
                  f"{b.get('f1_macro', 0):>10.3f} {s.get('f1_macro', 0):>10.3f} "
                  f"{delta:>+10.3f}{marker}")

        print("-" * 58)
        print(f"{'Summary':<8} {'':>10} {'':>10} "
              f"{base_result['summary_f1']:>10.4f} {sft_result['summary_f1']:>10.4f} "
              f"{sft_result['summary_f1'] - base_result['summary_f1']:>+10.4f}")

        # Save comparison
        comparison = {
            "base": base_result,
            "sft": sft_result,
            "delta_summary_f1": sft_result["summary_f1"] - base_result["summary_f1"],
            "timestamp": datetime.now().isoformat(),
        }
        with open(results_dir / "comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Results saved to: {results_dir}")
    print(f"  Raw outputs: {results_dir / f'sft_{model_name_short}_raw.json'}")
    print(f"  Metrics:     {results_dir / f'sft_{model_name_short}_metrics.json'}")
    if args.compare_base:
        print(f"  Comparison:  {results_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
