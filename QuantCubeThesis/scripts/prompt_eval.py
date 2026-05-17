"""
Prompt Engineering Evaluation Harness
======================================
Runs multiple prompt variants through non-fine-tuned Llama 3.1 8B
on a labelled validation set, computes per-field metrics.

Usage:
    # Run all 9 prompts on a 60-sentence sample:
    python scripts/prompt_eval.py --val-set data/eval_labelled_merged.json --sample 60

    # Run a single prompt:
    python scripts/prompt_eval.py --val-set data/eval_labelled_merged.json --prompt P1_medium_3shot.txt --sample 60

    # Run all prompts on the full eval set (slow):
    python scripts/prompt_eval.py --val-set data/eval_labelled_merged.json

    # Resume from where you left off (skips already-completed prompt/sentence combos):
    python scripts/prompt_eval.py --val-set data/eval_labelled_merged.json --resume
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

# ── CONFIGURATION ────────────────────────────────────────────────────────────

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
RESULTS_DIR = Path(__file__).parent.parent / "models" / "prompt_eval"

# Fields and their valid values (from annotation schema v2)
LABEL_SCHEMA = {
    "top": {
        "type": "multi",
        "values": ["inflation", "labor_market", "economic_activity", "macro",
                   "financial_conditions", "monetary_policy", "boilerplate", "no_topic"],
    },
    "ten": {
        "type": "single",
        "values": ["descriptive", "interpretive"],
    },
    "sen": {
        "type": "single",
        "values": ["-2", "-1", "0", "1", "2", "na"],
    },
    "com": {
        "type": "single",
        "values": ["unconditional", "conditional", "none"],
    },
    "hor": {
        "type": "single",
        "values": ["True", "False"],
    },
    "ris": {
        "type": "single",
        "values": ["skewed_downside", "skewed_upside", "symmetric", "na"],
    },
    "wid": {
        "type": "single",
        "values": ["elevated", "contested", "none"],
    },
}

# Fields that are most important for the thesis (weighted in summary)
PRIMARY_FIELDS = ["top", "sen", "ris", "wid"]

VLLM_CHUNK_SIZE = 50  # sentences per save checkpoint when using vLLM


# ── MODEL LOADING ────────────────────────────────────────────────────────────

def load_model(model_name: str):
    """Load quantized Llama 3.1 8B for local inference."""
    print(f"\nLoading model: {model_name}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    print(f"Model loaded. Memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB\n")
    return model, tokenizer


# ── vLLM BACKEND ─────────────────────────────────────────────────────────────

def load_model_vllm(model_name: str):
    """Load model with vLLM for fast batched inference (no quantization needed on A100)."""
    from vllm import LLM
    print(f"\nLoading model with vLLM: {model_name}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    llm = LLM(model=model_name, dtype="bfloat16", max_model_len=4096, enforce_eager=True)
    print("vLLM model loaded.\n")
    return llm


def generate_batch_vllm(
    llm,
    prompts: List[str],
    max_new_tokens: int = 256,
    temperature: float = 0.01,
) -> List[str]:
    """Batch inference via vLLM chat API. Returns one decoded string per prompt."""
    from vllm import SamplingParams
    conversations = [[{"role": "user", "content": p}] for p in prompts]
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        top_p=0.95 if temperature > 0 else 1.0,
    )
    outputs = llm.chat(messages=conversations, sampling_params=sampling_params, use_tqdm=False)
    return [o.outputs[0].text.strip() for o in outputs]


# ── INFERENCE (transformers) ──────────────────────────────────────────────────

def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.01,  # Near-deterministic for eval
) -> str:
    """Generate a single response from the model."""
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.95 if temperature > 0 else 1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


# ── JSON PARSING ──────────────────────────────────────────────────────────────

def extract_json(response: str) -> Optional[dict]:
    """
    Extract JSON from model response. Handles:
    - Pure JSON responses
    - JSON embedded in text
    - CoT responses with "ANSWER: {json}" format
    """
    # Try 1: Look for "ANSWER: " prefix (CoT prompts)
    answer_match = re.search(r'ANSWER:\s*(\{.*\})', response, re.DOTALL)
    if answer_match:
        try:
            return json.loads(answer_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try 2: Find the last JSON object in the response
    # (CoT prompts may have JSON in reasoning AND answer)
    json_matches = list(re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL))
    if json_matches:
        # Take the last one (most likely to be the final answer)
        for match in reversed(json_matches):
            try:
                parsed = json.loads(match.group())
                # Validate it has at least some expected fields
                if any(k in parsed for k in ["top", "ten", "sen"]):
                    return parsed
            except json.JSONDecodeError:
                continue

    # Try 3: Whole response is JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    return None


def normalize_prediction(pred: dict) -> dict:
    """Normalize a parsed prediction to match expected format."""
    normalized = {}

    for field, schema in LABEL_SCHEMA.items():
        val = pred.get(field)

        if val is None:
            # Missing field — use default
            normalized[field] = "na" if schema["type"] == "single" else ["na"]
            continue

        if schema["type"] == "multi":
            # Ensure it's a list
            if isinstance(val, str):
                if val == "na":
                    normalized[field] = ["na"]
                else:
                    normalized[field] = [val]
            elif isinstance(val, list):
                normalized[field] = [str(v).lower().strip() for v in val]
            else:
                normalized[field] = [str(val)]
        else:
            # Single value
            normalized[field] = str(val).lower().strip()

    return normalized


def normalize_ground_truth(gt: dict) -> dict:
    """Normalize ground truth labels to match prediction format."""
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


# ── METRICS ───────────────────────────────────────────────────────────────────

def compute_field_metrics(
    predictions: List[dict],
    ground_truths: List[dict],
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-field accuracy, precision, recall, F1.
    Handles both single-value and multi-value fields.
    """
    results = {}

    for field, schema in LABEL_SCHEMA.items():
        if schema["type"] == "single":
            y_true = [gt.get(field, "na") for gt in ground_truths]
            y_pred = [pred.get(field, "na") for pred in predictions]

            # Filter out cases where ground truth is empty/missing
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
            # Multi-label: use set-based exact match and Jaccard
            y_true_sets = [set(gt.get(field, ["na"])) for gt in ground_truths]
            y_pred_sets = [set(pred.get(field, ["na"])) for pred in predictions]

            exact_matches = sum(1 for t, p in zip(y_true_sets, y_pred_sets) if t == p)
            jaccard_scores = []
            for t, p in zip(y_true_sets, y_pred_sets):
                if len(t | p) == 0:
                    jaccard_scores.append(1.0)
                else:
                    jaccard_scores.append(len(t & p) / len(t | p))

            n = len(y_true_sets)
            results[field] = {
                "accuracy": exact_matches / n if n > 0 else 0,
                "f1_macro": np.mean(jaccard_scores) if jaccard_scores else 0,  # Jaccard as proxy
                "precision": np.mean(jaccard_scores) if jaccard_scores else 0,
                "recall": np.mean(jaccard_scores) if jaccard_scores else 0,
                "n": n,
            }

    return results


def compute_summary_score(field_metrics: Dict) -> float:
    """Weighted summary score across primary fields."""
    total_f1 = 0
    count = 0
    for field in PRIMARY_FIELDS:
        if field in field_metrics and field_metrics[field]["n"] > 0:
            total_f1 += field_metrics[field]["f1_macro"]
            count += 1
    return total_f1 / count if count > 0 else 0


# ── MAIN EVALUATION LOOP ─────────────────────────────────────────────────────

def _build_prompt(prompt_template: str, sentence_data: dict) -> str:
    """Apply sentence (and optional context_question) into a prompt template."""
    sentence = sentence_data["sentence"]
    prompt = prompt_template.replace("{sentence}", sentence)
    if sentence_data.get("context_question"):
        context_insert = f'\nContext (question being answered): "{sentence_data["context_question"]}"\n'
        prompt = prompt.replace(
            f'Sentence: "{sentence}"',
            f'{context_insert}Sentence: "{sentence}"',
        )
    return prompt


def evaluate_prompt(
    prompt_path: Path,
    validation_data: List[dict],
    model,
    tokenizer,
    results_dir: Path,
    resume: bool = False,
    max_new_tokens: int = 256,
    vllm_model=None,
) -> Dict:
    """Run a single prompt on all validation sentences and compute metrics.

    Pass vllm_model to use the fast batched vLLM path instead of the default
    per-sentence transformers path.
    """
    prompt_name = prompt_path.stem
    prompt_template = prompt_path.read_text(encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"EVALUATING: {prompt_name}  [{'vLLM' if vllm_model else 'transformers'}]")
    print(f"{'='*60}")

    # Check for existing partial results
    raw_results_path = results_dir / f"{prompt_name}_raw.json"
    existing_results = []
    completed_ids = set()

    if resume and raw_results_path.exists():
        with open(raw_results_path) as f:
            existing_results = json.load(f)
        completed_ids = {r["id"] for r in existing_results}
        print(f"  Resuming: {len(completed_ids)} already completed")

    raw_outputs = list(existing_results)
    predictions: List[dict] = []
    ground_truths: List[dict] = []
    parse_failures = 0
    remaining = [s for s in validation_data if s["id"] not in completed_ids]
    total = len(validation_data)

    # ── vLLM batched path ─────────────────────────────────────────────────────
    if vllm_model is not None:
        n_chunks = (len(remaining) + VLLM_CHUNK_SIZE - 1) // VLLM_CHUNK_SIZE
        for chunk_idx in range(n_chunks):
            chunk = remaining[chunk_idx * VLLM_CHUNK_SIZE:(chunk_idx + 1) * VLLM_CHUNK_SIZE]
            prompts = [_build_prompt(prompt_template, s) for s in chunk]

            t0 = time.time()
            try:
                responses = generate_batch_vllm(vllm_model, prompts, max_new_tokens=max_new_tokens)
            except Exception as e:
                print(f"  ERROR in vLLM batch {chunk_idx + 1}/{n_chunks}: {e}")
                responses = [""] * len(chunk)
            elapsed = time.time() - t0

            for j, (sentence_data, response) in enumerate(zip(chunk, responses)):
                sid = sentence_data["id"]
                global_idx = len(completed_ids) + chunk_idx * VLLM_CHUNK_SIZE + j + 1
                parsed = extract_json(response)
                if parsed is None:
                    parse_failures += 1
                    print(f"  [{global_idx}/{total}] PARSE FAIL ({sid})")
                    parsed = {}
                else:
                    print(f"  [{global_idx}/{total}] OK ({sid})")

                raw_outputs.append({
                    "id": sid,
                    "sentence": sentence_data["sentence"][:100],
                    "raw_response": response[:1000],
                    "parsed": parsed,
                    "elapsed_s": round(elapsed / len(chunk), 2),
                })

            with open(raw_results_path, "w") as f:
                json.dump(raw_outputs, f, indent=2)
            print(f"  Chunk {chunk_idx + 1}/{n_chunks}: {len(chunk)} sentences in {elapsed:.1f}s "
                  f"({elapsed/len(chunk):.2f}s/sent)")

    # ── transformers per-sentence path ────────────────────────────────────────
    else:
        for i, sentence_data in enumerate(remaining):
            idx = len(completed_ids) + i + 1
            sid = sentence_data["id"]
            prompt = _build_prompt(prompt_template, sentence_data)

            t0 = time.time()
            try:
                response = generate_response(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
            except Exception as e:
                print(f"  [{idx}/{total}] ERROR generating for {sid}: {e}")
                response = ""
            elapsed = time.time() - t0

            parsed = extract_json(response)
            if parsed is None:
                parse_failures += 1
                print(f"  [{idx}/{total}] PARSE FAIL ({sid}) — {elapsed:.1f}s")
                parsed = {}
            else:
                print(f"  [{idx}/{total}] OK ({sid}) — {elapsed:.1f}s")

            raw_outputs.append({
                "id": sid,
                "sentence": sentence_data["sentence"][:100],
                "raw_response": response[:1000],
                "parsed": parsed,
                "elapsed_s": round(elapsed, 1),
            })

            with open(raw_results_path, "w") as f:
                json.dump(raw_outputs, f, indent=2)

    # Now compute metrics over ALL results (existing + new)
    all_parsed = {r["id"]: r.get("parsed", {}) for r in raw_outputs}

    for sentence_data in validation_data:
        sid = sentence_data["id"]
        if sid in all_parsed:
            gt = normalize_ground_truth(sentence_data)
            pred = normalize_prediction(all_parsed[sid])
            ground_truths.append(gt)
            predictions.append(pred)

    if not predictions:
        print("  No predictions to evaluate!")
        return {}

    # Compute metrics
    field_metrics = compute_field_metrics(predictions, ground_truths)
    summary_score = compute_summary_score(field_metrics)

    total_parse_failures = sum(1 for r in raw_outputs if not r.get("parsed"))

    result = {
        "prompt_name": prompt_name,
        "n_sentences": len(predictions),
        "parse_failure_rate": total_parse_failures / len(raw_outputs) if raw_outputs else 0,
        "summary_f1": summary_score,
        "field_metrics": field_metrics,
        "timestamp": datetime.now().isoformat(),
    }

    # Save metrics
    metrics_path = results_dir / f"{prompt_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    print(f"\n  Summary F1 (primary fields): {summary_score:.4f}")
    print(f"  Parse failure rate: {result['parse_failure_rate']:.1%}")
    print(f"  Per-field accuracy:")
    for field, metrics in field_metrics.items():
        marker = " *" if field in PRIMARY_FIELDS else ""
        print(f"    {field:5s}: acc={metrics['accuracy']:.3f}  F1={metrics['f1_macro']:.3f}  (n={metrics['n']}){marker}")

    return result


    # IDs of sentences used as few-shot examples in prompts — exclude from evaluation
EXAMPLE_IDS = {
    "e5418507-7ab1-476e-8db9-fc28796c584f",  # boilerplate
    "99122de6-aab0-49fc-9183-d4bd3fc33e27",  # level vs change
    "c7346ec3-ca84-449c-a104-7e0c8d3543ba",  # contested
    "cd96b673-cdc4-4ab3-a428-3880ba0bd1dd",  # strong dovish
    "995fddd0-966b-4ea0-9eaf-879c5f7fbeed",  # risk upside
}


def run_all_prompts(
    val_path: str,
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    prompt_filter: Optional[str] = None,
    resume: bool = False,
    sample_size: Optional[int] = None,
    max_new_tokens: int = 256,
    use_vllm: bool = False,
):
    """Run evaluation across all (or selected) prompts."""
    # Load validation data
    with open(val_path, encoding="utf-8") as f:
        validation_data = json.load(f)
    print(f"Loaded {len(validation_data)} validation sentences from {val_path}")

    # Exclude few-shot example sentences
    validation_data = [s for s in validation_data if s["id"] not in EXAMPLE_IDS]
    print(f"After excluding few-shot examples: {len(validation_data)} sentences")

    # Subsample if requested (stratified by sen to keep label distribution)
    if sample_size and sample_size < len(validation_data):
        np.random.seed(42)
        indices = np.random.choice(len(validation_data), size=sample_size, replace=False)
        validation_data = [validation_data[i] for i in sorted(indices)]
        print(f"Sampled {len(validation_data)} sentences for evaluation")

    # Load model once
    if use_vllm:
        vllm_model = load_model_vllm(model_name)
        model, tokenizer = None, None
    else:
        model, tokenizer = load_model(model_name)
        vllm_model = None

    # Find prompts — new naming convention: P0_minimal.txt, P1_medium_3shot.txt, etc.
    if prompt_filter:
        prompt_files = [PROMPTS_DIR / prompt_filter]
        if not prompt_files[0].exists():
            prompt_files = list(PROMPTS_DIR.glob(f"*{prompt_filter}*"))
    else:
        prompt_files = sorted(PROMPTS_DIR.glob("P*.txt"))

    print(f"Prompts to evaluate: {[p.stem for p in prompt_files]}")

    # Results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run each prompt
    all_results = []
    for prompt_path in prompt_files:
        # CoT prompts need more output tokens for reasoning
        is_cot = "cot" in prompt_path.stem.lower()
        tokens = max_new_tokens if not is_cot else max(max_new_tokens, 512)

        result = evaluate_prompt(
            prompt_path, validation_data, model, tokenizer, RESULTS_DIR, resume,
            max_new_tokens=tokens, vllm_model=vllm_model,
        )
        if result:
            all_results.append(result)

    # ── Final comparison table ────────────────────────────────────────────
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("PROMPT COMPARISON — RANKED BY SUMMARY F1")
        print("=" * 80)

        comparison = []
        for r in sorted(all_results, key=lambda x: x["summary_f1"], reverse=True):
            row = {
                "prompt": r["prompt_name"],
                "summary_f1": f"{r['summary_f1']:.4f}",
                "parse_fail": f"{r['parse_failure_rate']:.1%}",
            }
            for field in PRIMARY_FIELDS:
                fm = r["field_metrics"].get(field, {})
                row[f"{field}_f1"] = f"{fm.get('f1_macro', 0):.3f}"
            comparison.append(row)

        df = pd.DataFrame(comparison)
        print(df.to_string(index=False))

        # Save comparison
        comp_path = RESULTS_DIR / "comparison.json"
        with open(comp_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nFull results saved to {RESULTS_DIR}/")

        # Identify winner
        best = sorted(all_results, key=lambda x: x["summary_f1"], reverse=True)[0]
        print(f"\n>>> BEST PROMPT: {best['prompt_name']} (summary F1 = {best['summary_f1']:.4f})")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FOMC Prompt Engineering Evaluation")
    parser.add_argument("--val-set", type=str, required=True,
                        help="Path to labelled validation JSON")
    parser.add_argument("--model", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Specific prompt file to test (filename or glob)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from partial results")
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of sentences to sample (default: use all)")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Max output tokens (default: 256, CoT prompts auto-use 512)")
    parser.add_argument("--vllm", action="store_true",
                        help="Use vLLM for fast batched inference instead of transformers+BitsAndBytes")
    args = parser.parse_args()

    run_all_prompts(
        val_path=args.val_set,
        model_name=args.model,
        prompt_filter=args.prompt,
        resume=args.resume,
        sample_size=args.sample,
        max_new_tokens=args.max_new_tokens,
        use_vllm=args.vllm,
    )


if __name__ == "__main__":
    main()
