"""Verify no duplicates exist within eval or across eval+seed."""
import json
from pathlib import Path
import re, string
from collections import defaultdict

BASE = Path(r"C:\Users\Javier\OneDrive - HEC Paris\Documentos\QuantCubeThesis\QuantCubeThesis\data")
SEED_DIR = BASE / "QuantCube_Seed_Labelled"

def normalize_text(s):
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s

with open(BASE / "eval_labelled_merged.json", encoding="utf-8") as f:
    eval_data = json.load(f)

eval_norms = [normalize_text(s["sentence"]) for s in eval_data]
eval_norm_set = set(eval_norms)

# Check for duplicates within eval
dup_within_eval = len(eval_norms) - len(eval_norm_set)
print(f"Eval size: {len(eval_data)}")
print(f"Duplicates within eval: {dup_within_eval}")

# Check for cross-contamination with seed
seed_files = sorted(p for p in SEED_DIR.glob("*.json") if not p.name.endswith("_tolabel.json"))
all_seed = []
for sf in seed_files:
    with open(sf, encoding="utf-8") as f:
        data = json.load(f)
    all_seed.extend(data)

overlap = sum(1 for s in all_seed if normalize_text(s["sentence"]) in eval_norm_set)
print(f"Seed sentences also in eval (should be 0 if proper removal): {overlap}")
if overlap > 0:
    print("  OVERLAP DETECTED - investigate!")
    for s in all_seed:
        if normalize_text(s["sentence"]) in eval_norm_set:
            print(f"  OVERLAP: [{s.get('__source_file__','?')}] {s['sentence'][:80]}")

print(f"\nTotal seed sentences remaining: {len(all_seed)}")
print("Verification complete.")
