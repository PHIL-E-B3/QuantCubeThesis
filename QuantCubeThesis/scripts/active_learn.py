"""
Active Learning — Generative Model (vLLM + LoRA)
=================================================
Selects the most uncertain unlabelled sentences for human annotation,
using mean negative log-probability of generated tokens as the uncertainty
signal. Higher uncertainty = model is less confident in its output.

Workflow:
    # Step 1: score unlabelled pool and export candidates
    VLLM_USE_FLASHINFER_SAMPLER=0 python scripts/active_learn.py --select \
        --cycle 1 \
        --adapter models/best/checkpoint-258 \
        --prompt prompts/P5_v27.txt

    # Step 2: human annotates data/active_learning/candidates_cycle_01.json

    # Step 3: integrate annotations into training data
    python scripts/active_learn.py --integrate --cycle 1

    # Step 4: retrain (run separately using train.py --retrain-best)
"""
import argparse
import json
import re
import numpy as np
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
UNLABELLED_POOL = PROJECT_ROOT / "data" / "all_unlabelled_sentences" / "master_unlabelled_pool.json"
TRAIN_FILES = [
    PROJECT_ROOT / "data" / "QuantCube_Seed_Labelled" / "all_labelled_sentences.json",
    PROJECT_ROOT / "data" / "QuantCube_Seed_Batches" / "final_extreme_seed.json",
]
EVAL_FILE = PROJECT_ROOT / "data" / "eval_labelled_merged_corrected.json"
AL_DIR = PROJECT_ROOT / "data" / "active_learning"


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


def load_known_ids():
    """IDs already in training or eval — never select these."""
    known = set()
    for f in TRAIN_FILES:
        if f.exists():
            for r in json.load(open(f, encoding="utf-8")):
                known.add(r["id"])
    if EVAL_FILE.exists():
        for r in json.load(open(EVAL_FILE, encoding="utf-8")):
            known.add(r["id"])
    # Also exclude anything already selected in previous AL cycles
    for f in sorted(AL_DIR.glob("candidates_cycle_*.json")):
        for r in json.load(open(f, encoding="utf-8")):
            known.add(r["id"])
    return known


def select(args):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    known_ids = load_known_ids()
    pool = [r for r in json.load(open(UNLABELLED_POOL, encoding="utf-8"))
            if r["id"] not in known_ids]
    print(f"Unlabelled pool (excluding known): {len(pool)} sentences")

    # Optionally subsample to keep runtime manageable
    if args.pool_size and args.pool_size < len(pool):
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(pool), size=args.pool_size, replace=False)
        pool = [pool[i] for i in sorted(idx)]
        print(f"Subsampled to {len(pool)} sentences (--pool-size {args.pool_size})")

    prompt_template = Path(args.prompt).read_text(encoding="utf-8").strip()

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
    for s in pool:
        p = prompt_template.replace("{sentence}", s.get("sentence") or "")
        p = p.replace("{context_question}", s.get("context_question") or "")
        conversations.append([{"role": "user", "content": p}])

    print(f"Scoring {len(pool)} sentences...")
    outputs = llm.chat(messages=conversations, sampling_params=sampling,
                       lora_request=lora_request, use_tqdm=True)

    scored = []
    for s, o in zip(pool, outputs):
        parsed = extract_json(o.outputs[0].text.strip()) or {}
        scored.append({
            **s,
            "uncertainty": mean_neg_logprob(o),
            "model_prediction": parsed,
        })

    # Sort by uncertainty descending and take top-K
    scored = [r for r in scored if not np.isnan(r["uncertainty"])]
    scored.sort(key=lambda x: -x["uncertainty"])

    candidates = scored[:args.query_size]

    # Print summary
    print(f"\nTop {args.query_size} most uncertain sentences selected")
    print(f"  Uncertainty range: {candidates[-1]['uncertainty']:.4f} – {candidates[0]['uncertainty']:.4f}")
    sent_dist = Counter(str(r.get("model_prediction", {}).get("sentiment", "?")) for r in candidates)
    print(f"  Predicted sentiment distribution: {dict(sent_dist)}")

    # Clear model_prediction from export (annotator fills in labels fresh)
    # but keep it as a reference column
    export = []
    for r in candidates:
        export.append({
            "id": r["id"],
            "sentence": r.get("sentence", ""),
            "source": r.get("source", ""),
            "doc_type": r.get("doc_type", ""),
            "date": r.get("date", ""),
            "context_question": r.get("context_question", ""),
            "uncertainty": round(r["uncertainty"], 6),
            "model_prediction": r["model_prediction"],
            # Empty fields for human annotator to fill:
            "top": "",
            "ten": "",
            "sen": "",
            "com": "",
            "ris": "",
            "wid": "",
            "notes": "",
        })

    AL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = AL_DIR / f"candidates_cycle_{args.cycle:02d}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)

    # Log the cycle
    log_path = AL_DIR / "active_learning_log.json"
    log = json.load(open(log_path)) if log_path.exists() else []
    log.append({
        "cycle": args.cycle,
        "timestamp": datetime.now().isoformat(),
        "pool_scored": len(scored),
        "query_size": args.query_size,
        "mean_uncertainty_selected": float(np.mean([r["uncertainty"] for r in candidates])),
        "mean_uncertainty_pool": float(np.mean([r["uncertainty"] for r in scored])),
    })
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nCandidates saved to {out_path}")
    print(f"Next step: annotate {out_path.name}, then run --integrate --cycle {args.cycle}")


def integrate(args):
    candidates_path = AL_DIR / f"candidates_cycle_{args.cycle:02d}.json"
    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates file not found: {candidates_path}")

    candidates = json.load(open(candidates_path, encoding="utf-8"))

    # Filter to only rows that were actually annotated (sen field filled in)
    annotated = [r for r in candidates if r.get("sen") not in ("", None)]
    if not annotated:
        print("No annotated rows found. Fill in the label fields first.")
        return

    # Strip model_prediction and uncertainty from what goes into training
    clean = []
    for r in annotated:
        clean.append({k: r[k] for k in
                      ["id", "sentence", "source", "doc_type", "date",
                       "context_question", "top", "ten", "sen", "com", "ris", "wid"]
                      if k in r})

    # Append to all_labelled_sentences.json (primary training file)
    target = TRAIN_FILES[0]
    existing = json.load(open(target, encoding="utf-8"))
    existing_ids = {r["id"] for r in existing}
    new = [r for r in clean if r["id"] not in existing_ids]

    if not new:
        print("All annotated sentences already in training set.")
        return

    existing.extend(new)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"Integrated {len(new)} new sentences into {target.name}")
    print(f"Training set now: {len(existing)} sentences")
    print(f"\nNext step: retrain with expanded dataset:")
    print(f"  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train.py "
          f"--config configs/default.yaml --retrain-best")


def main():
    parser = argparse.ArgumentParser(description="Active Learning — Generative Model")
    parser.add_argument("--select", action="store_true", help="Score unlabelled pool and export candidates")
    parser.add_argument("--integrate", action="store_true", help="Integrate annotated candidates into training data")
    parser.add_argument("--cycle", type=int, required=True, help="Active learning cycle number")
    # Select options
    parser.add_argument("--adapter", default="models/best/checkpoint-258")
    parser.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--prompt", default="prompts/P5_v27.txt")
    parser.add_argument("--max-model-len", type=int, default=2560)
    parser.add_argument("--query-size", type=int, default=80, help="Sentences to select per cycle")
    parser.add_argument("--pool-size", type=int, default=None,
                        help="Max unlabelled sentences to score (None = all). "
                             "Use e.g. 10000 to limit runtime.")
    args = parser.parse_args()

    if args.select:
        select(args)
    elif args.integrate:
        integrate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
