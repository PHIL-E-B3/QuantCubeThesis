"""
count_sentence_tokens.py
-------------------------
Counts token lengths of all sentences (labelled + unlabelled pool)
and reports distribution statistics.

Usage:
    python scripts/count_sentence_tokens.py --token YOUR_HF_TOKEN
"""

import argparse
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

FILES = {
    "seed (all_labelled_sentences)": ROOT / "data/QuantCube_Seed_Labelled/all_labelled_sentences.json",
    "eval (eval_labelled_merged)":    ROOT / "data/eval_labelled_merged.json",
    "unlabelled pool":                ROOT / "data/all_unlabelled_sentences/master_unlabelled_pool.json",
}


def get_sentences(path: Path) -> list:
    with open(path, encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    return [r["sentence"] for r in data if isinstance(r, dict) and r.get("sentence")]


def report(name: str, lengths: list):
    a = np.array(lengths)
    print(f"\n{'='*55}")
    print(f"  {name}  ({len(a):,} sentences)")
    print(f"{'='*55}")
    print(f"  Min:         {a.min():>6}")
    print(f"  25th pct:    {np.percentile(a, 25):>6.0f}")
    print(f"  Median:      {np.percentile(a, 50):>6.0f}")
    print(f"  75th pct:    {np.percentile(a, 75):>6.0f}")
    print(f"  90th pct:    {np.percentile(a, 90):>6.0f}")
    print(f"  95th pct:    {np.percentile(a, 95):>6.0f}")
    print(f"  99th pct:    {np.percentile(a, 99):>6.0f}")
    print(f"  Max:         {a.max():>6}")
    print(f"  Mean:        {a.mean():>6.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=True, help="HuggingFace access token")
    args = parser.parse_args()

    from huggingface_hub import login
    from transformers import AutoTokenizer

    login(token=args.token, add_to_git_credential=False)
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    all_sentences = []

    for name, path in FILES.items():
        if not path.exists():
            print(f"\nSkipping (not found): {path.name}")
            continue
        sentences = get_sentences(path)
        print(f"\nTokenizing {name} ({len(sentences):,} sentences)...")
        lengths = [len(tokenizer(s, add_special_tokens=False)["input_ids"]) for s in sentences]
        report(name, lengths)
        all_sentences.extend(lengths)

    if all_sentences:
        report("ALL COMBINED", all_sentences)


if __name__ == "__main__":
    main()
