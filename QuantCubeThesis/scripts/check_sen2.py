"""Check how sen=2 is stored in seed files and what values are actually available."""
import json
from pathlib import Path
from collections import defaultdict
import re, string

BASE = Path(r"C:\Users\Javier\OneDrive - HEC Paris\Documentos\QuantCubeThesis\QuantCubeThesis\data")
SEED_DIR = BASE / "QuantCube_Seed_Labelled"

def normalize_text(s):
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s

# load eval
with open(BASE / "eval_labelled_merged.json", encoding="utf-8") as f:
    eval_data = json.load(f)

eval_norms = {normalize_text(s["sentence"]) for s in eval_data}

# count sen=2 in eval
sen2_eval = [s for s in eval_data if str(s.get("sen")) == "2"]
print(f"sen=2 in eval: {len(sen2_eval)}")

# scan all seed files
seed_files = sorted(p for p in SEED_DIR.glob("*.json") if not p.name.endswith("_tolabel.json"))
sen2_in_seed = []
for sf in seed_files:
    with open(sf, encoding="utf-8") as f:
        data = json.load(f)
    for s in data:
        if str(s.get("sen")) == "2":
            norm = normalize_text(s["sentence"])
            if norm not in eval_norms:
                sen2_in_seed.append((sf.name, s["sentence"][:80]))

print(f"\nsen=2 sentences remaining in seed (not in eval): {len(sen2_in_seed)}")
for fname, snip in sen2_in_seed:
    print(f"  [{fname}] {snip}")

# Check all unique sen values across seed files
print("\nAll unique 'sen' values seen in seed pool (not in eval):")
vals = defaultdict(int)
for sf in seed_files:
    with open(sf, encoding="utf-8") as f:
        data = json.load(f)
    for s in data:
        if normalize_text(s["sentence"]) not in eval_norms:
            vals[repr(s.get("sen"))] += 1
for v, c in sorted(vals.items()):
    print(f"  sen={v}: {c}")
