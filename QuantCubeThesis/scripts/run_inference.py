"""
run_inference.py — Full-corpus LLM inference for FOMC sentence labelling.

Reads  data/full_corpus_ordered.json   (174,525 sentences, stable IDs)
Writes Taylor Rule/nlp_output/nlp_labels.json   (one labelled record per sentence)

Progress is check-pointed after every batch: a JSONL sidecar file lets the job
resume exactly where it left off if the A100 run is interrupted.

Usage (on the A100):
    python scripts/run_inference.py
    python scripts/run_inference.py --batch-size 1024 --adapter models/best/checkpoint-258
    python scripts/run_inference.py --resume          # skip already-done records
    python scripts/run_inference.py --consolidate-only  # merge JSONL → final JSON only

Arguments:
    --corpus        Path to full_corpus_ordered.json (default: data/full_corpus_ordered.json)
    --adapter       LoRA adapter directory            (default: models/best/checkpoint-258)
    --model         Base model HF name or local path  (default: unsloth/Meta-Llama-3.1-8B-Instruct)
    --prompt        Prompt template .txt file         (default: prompts/P5_v41b.txt)
    --output        Final output JSON path            (default: Taylor Rule/nlp_output/nlp_labels.json)
    --checkpoint    JSONL checkpoint file             (default: <output_dir>/nlp_labels_checkpoint.jsonl)
    --batch-size    vLLM batch size                   (default: 512)
    --max-model-len Max token context window          (default: 2560)
    --resume        Skip records already in checkpoint (default: True)
    --no-resume     Force re-inference of all records
    --consolidate-only  Skip inference; just merge checkpoint → final JSON
"""

import argparse
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent

DEFAULT_CORPUS  = ROOT / "data" / "full_corpus_ordered.json"
DEFAULT_ADAPTER = ROOT / "models" / "best" / "checkpoint-258"
DEFAULT_PROMPT  = ROOT / "prompts" / "P5_v41b.txt"
DEFAULT_OUTPUT  = ROOT / "Taylor Rule" / "nlp_output" / "nlp_labels.json"

# Label schema — used for parsing and normalisation
LABEL_SCHEMA = {
    "topic":      {"type": "multi",  "values": ["inflation", "labor_market", "economic_activity",
                   "macro", "financial_conditions", "monetary_policy", "no_topic"]},
    "tense":      {"type": "single", "values": ["descriptive", "interpretive"]},
    "sentiment":  {"type": "single", "values": ["hawkish", "neutral", "dovish"]},
    "commitment": {"type": "bool"},
    "risk":       {"type": "single", "values": ["skewed_downside", "skewed_upside", "symmetric", "na"]},
    "width":      {"type": "single", "values": ["elevated", "none"]},
}


# ── Parsing helpers ────────────────────────────────────────────────────────────

def extract_json(response: str) -> Optional[dict]:
    """Extract the first valid JSON object from a model response string."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    for match in re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL):
        try:
            parsed = json.loads(match.group())
            if any(k in parsed for k in ("topic", "sentiment", "tense")):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def normalise_labels(raw: dict) -> dict:
    """Coerce parsed JSON to clean, schema-conformant label values."""
    out = {}
    for field, schema in LABEL_SCHEMA.items():
        val = raw.get(field)
        if schema["type"] == "multi":
            if val is None:
                out[field] = ["no_topic"]
            elif isinstance(val, list):
                out[field] = [str(v).lower().strip() for v in val] or ["no_topic"]
            else:
                out[field] = [str(val).lower().strip()]
        elif schema["type"] == "bool":
            if isinstance(val, bool):
                out[field] = val
            elif isinstance(val, str):
                out[field] = val.lower() in ("true", "yes", "1")
            else:
                out[field] = False
        else:  # single
            out[field] = str(val).lower().strip() if val is not None else "na"
    return out


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: Path) -> dict:
    """Return {id: labelled_record} for all already-processed sentences."""
    done = {}
    if not ckpt_path.exists():
        return done
    with open(ckpt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    done[rec["id"]] = rec
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def append_checkpoint(ckpt_path: Path, records: list[dict]) -> None:
    """Append a batch of labelled records to the JSONL checkpoint file."""
    with open(ckpt_path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def consolidate(ckpt_path: Path, out_path: Path, corpus: list[dict]) -> None:
    """Merge checkpoint JSONL into final JSON, preserving corpus order."""
    done = load_checkpoint(ckpt_path)
    print(f"Checkpoint: {len(done)} labelled records")

    corpus_ids = [r["id"] for r in corpus]
    missing = [cid for cid in corpus_ids if cid not in done]
    if missing:
        print(f"WARNING: {len(missing)} corpus records not in checkpoint — they will be absent from output")

    ordered = [done[cid] for cid in corpus_ids if cid in done]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=None)
    print(f"Saved {len(ordered)} records to {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


# ── Core inference ─────────────────────────────────────────────────────────────

def run_inference(args) -> None:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    corpus_path  = Path(args.corpus)
    adapter_path = Path(args.adapter).resolve()
    prompt_path  = Path(args.prompt)
    out_path     = Path(args.output)
    ckpt_path    = Path(args.checkpoint) if args.checkpoint else out_path.parent / "nlp_labels_checkpoint.jsonl"

    # Load corpus
    print(f"Loading corpus from {corpus_path} ...")
    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"  {len(corpus):,} sentences total")

    # Load prompt template
    prompt_template = prompt_path.read_text(encoding="utf-8").strip()
    print(f"Prompt: {prompt_path.name}")

    # Load checkpoint — skip already-processed records
    done_ids: set[str] = set()
    if args.resume:
        done = load_checkpoint(ckpt_path)
        done_ids = set(done.keys())
        print(f"Checkpoint: {len(done_ids)} already done, resuming")

    todo = [r for r in corpus if r["id"] not in done_ids]
    print(f"To infer: {len(todo):,} sentences")

    if not todo:
        print("Nothing to do — all records already in checkpoint.")
        consolidate(ckpt_path, out_path, corpus)
        return

    # Load vLLM
    print(f"\nLoading vLLM (model={args.model}) ...")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        enforce_eager=True,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=1,
    )
    lora_request = LoRARequest("best", 1, str(adapter_path))
    sampling = SamplingParams(temperature=0.01, top_p=0.95, max_tokens=256)
    print("vLLM ready.")

    # Batch inference
    batch_size  = args.batch_size
    n_batches   = (len(todo) + batch_size - 1) // batch_size
    total_done  = 0
    parse_fails = 0
    t0          = time.time()

    print(f"\nStarting inference: {len(todo):,} sentences in {n_batches} batches of {batch_size}\n")

    for batch_idx in range(n_batches):
        batch = todo[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        prompts = [prompt_template.replace("{sentence}", r["sentence"]) for r in batch]
        conversations = [[{"role": "user", "content": p}] for p in prompts]

        outputs = llm.chat(
            messages=conversations,
            sampling_params=sampling,
            lora_request=lora_request,
            use_tqdm=False,
        )

        labelled_batch = []
        for rec, out in zip(batch, outputs):
            resp  = out.outputs[0].text.strip()
            raw   = extract_json(resp)
            ok    = raw is not None
            if not ok:
                parse_fails += 1
                raw = {}
            labels = normalise_labels(raw)
            labelled_batch.append({
                "id":           rec["id"],
                "source":       rec["source"],
                "doc_type":     rec["doc_type"],
                "date":         rec["date"],
                "sentence_idx": rec["sentence_idx"],
                "sentence":     rec["sentence"],
                "parse_ok":     ok,
                **labels,
            })

        append_checkpoint(ckpt_path, labelled_batch)
        total_done += len(batch)

        # Progress line with ETA
        elapsed   = time.time() - t0
        rate      = total_done / elapsed if elapsed > 0 else 1
        remaining = len(todo) - total_done
        eta_s     = remaining / rate if rate > 0 else 0
        eta_str   = str(timedelta(seconds=int(eta_s)))
        pct       = 100 * total_done / len(todo)
        print(
            f"  Batch {batch_idx + 1:>4}/{n_batches}  "
            f"{total_done:>7,}/{len(todo):,}  ({pct:5.1f}%)  "
            f"rate={rate:.0f} s/s  ETA={eta_str}  "
            f"parse_fail={parse_fails}"
        )

    elapsed_total = time.time() - t0
    print(f"\nInference complete in {timedelta(seconds=int(elapsed_total))}")
    print(f"Parse failures: {parse_fails}/{total_done} ({100*parse_fails/total_done:.1f}%)")

    consolidate(ckpt_path, out_path, corpus)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full-corpus FOMC inference with vLLM + LoRA")
    parser.add_argument("--corpus",           default=str(DEFAULT_CORPUS))
    parser.add_argument("--adapter",          default=str(DEFAULT_ADAPTER))
    parser.add_argument("--model",            default="unsloth/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--prompt",           default=str(DEFAULT_PROMPT))
    parser.add_argument("--output",           default=str(DEFAULT_OUTPUT))
    parser.add_argument("--checkpoint",       default=None,
                        help="JSONL checkpoint file (default: <output_dir>/nlp_labels_checkpoint.jsonl)")
    parser.add_argument("--batch-size",       type=int, default=512)
    parser.add_argument("--max-model-len",    type=int, default=2560)
    parser.add_argument("--resume",           action="store_true", default=True,
                        help="Skip records already in checkpoint (default: on)")
    parser.add_argument("--no-resume",        dest="resume", action="store_false",
                        help="Re-infer all records, ignoring checkpoint")
    parser.add_argument("--consolidate-only", action="store_true",
                        help="Skip inference; merge checkpoint → final JSON only")
    args = parser.parse_args()

    if args.consolidate_only:
        corpus_path = Path(args.corpus)
        out_path    = Path(args.output)
        ckpt_path   = Path(args.checkpoint) if args.checkpoint else out_path.parent / "nlp_labels_checkpoint.jsonl"
        with open(corpus_path, encoding="utf-8") as f:
            corpus = json.load(f)
        consolidate(ckpt_path, out_path, corpus)
    else:
        run_inference(args)


if __name__ == "__main__":
    main()
