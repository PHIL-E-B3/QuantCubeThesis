"""Find and report duplicate sentences in eval."""
import json
from pathlib import Path
import re, string
from collections import defaultdict

BASE = Path(r"C:\Users\Javier\OneDrive - HEC Paris\Documentos\QuantCubeThesis\QuantCubeThesis\data")

def normalize_text(s):
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s

with open(BASE / "eval_labelled_merged.json", encoding="utf-8") as f:
    eval_data = json.load(f)

# Find duplicates
seen = defaultdict(list)
for i, s in enumerate(eval_data):
    norm = normalize_text(s["sentence"])
    seen[norm].append(i)

dups = {k: v for k, v in seen.items() if len(v) > 1}
print(f"Duplicate groups: {len(dups)}")
for norm, indices in dups.items():
    print(f"\nNorm text (first 100): {norm[:100]}")
    for idx in indices:
        s = eval_data[idx]
        print(f"  Index {idx}: id={s.get('id')} | sen={s.get('sen')} | ris={s.get('ris')}")
        print(f"    Sentence: {s['sentence'][:120]}")
