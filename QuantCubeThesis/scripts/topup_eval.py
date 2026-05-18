"""
Top-up eval batch so every label VALUE has at least 15 examples.
Moves sentences from seed labelled files to eval_labelled_merged.json.
"""

import json
import re
import string
from collections import defaultdict
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(r"C:\Users\Javier\OneDrive - HEC Paris\Documentos\QuantCubeThesis\QuantCubeThesis\data")
EVAL_FILE = BASE / "eval_labelled_merged.json"
SEED_DIR  = BASE / "QuantCube_Seed_Labelled"

# ── Label schema ─────────────────────────────────────────────────────────────
LABEL_FIELDS = {
    "ris": ["skewed_downside", "skewed_upside", "symmetric", "na"],
    "sen": ["-2", "-1", "0", "1", "2", "na"],
    "wid": ["elevated", "contested", "none"],
    "com": ["unconditional", "conditional", "none"],
    "hor": ["True", "False"],
    "ten": ["descriptive", "interpretive"],
}
TOP_VALUES = ["inflation", "labor_market", "economic_activity", "macro",
              "financial_conditions", "monetary_policy", "boilerplate", "no_topic"]

MIN_COUNT = 15


# ── Helpers ──────────────────────────────────────────────────────────────────
def normalize_text(s: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_label(field: str, val) -> str:
    """Return a canonical string for any label value."""
    if field == "hor":
        if isinstance(val, bool):
            return str(val)          # True / False
        if isinstance(val, str):
            return val.capitalize()  # "true" -> "True"
        return str(val)
    if field == "sen":
        # int 0, string "0", etc. → "-2"/"-1"/"0"/"1"/"2"/"na"
        return str(val)
    return str(val)


def get_label_values(sentence: dict, field: str):
    """Return a list of canonical label values for this field on this sentence."""
    if field == "top":
        tops = sentence.get("top", [])
        if isinstance(tops, list):
            return tops
        return [str(tops)]
    val = sentence.get(field)
    if val is None:
        return []
    return [normalize_label(field, val)]


# ── Step 1: Load eval and count distribution ──────────────────────────────────
print("=" * 60)
print("STEP 1 — Loading eval and counting distribution")
print("=" * 60)

with open(EVAL_FILE, encoding="utf-8") as f:
    eval_data = json.load(f)

print(f"  Eval sentences loaded: {len(eval_data)}")

# Count distributions
eval_counts = {}  # field -> value -> count
for field in list(LABEL_FIELDS.keys()) + ["top"]:
    eval_counts[field] = defaultdict(int)

for sent in eval_data:
    for field in LABEL_FIELDS:
        for v in get_label_values(sent, field):
            eval_counts[field][v] += 1
    for v in get_label_values(sent, "top"):
        eval_counts["top"][v] += 1

print("\nFull eval label distribution:")
for field, val_dict in eval_counts.items():
    print(f"\n  {field}:")
    if field == "top":
        expected = TOP_VALUES
    else:
        expected = LABEL_FIELDS[field]
    for v in expected:
        cnt = val_dict.get(v, 0)
        flag = "  <<< BELOW 15" if cnt < MIN_COUNT else ""
        print(f"    {v:30s}: {cnt:4d}{flag}")
    # also show any unexpected values
    for v, cnt in sorted(val_dict.items()):
        if v not in expected:
            print(f"    [unexpected] {v:24s}: {cnt:4d}")


# ── Step 2: Load seed files ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Loading seed files")
print("=" * 60)

seed_files = sorted(
    p for p in SEED_DIR.glob("*.json")
    if not p.name.endswith("_tolabel.json")
)
print(f"  Seed files found (excluding _tolabel): {len(seed_files)}")
for sf in seed_files:
    print(f"    {sf.name}")

# load all seed sentences, track origin file
seed_pool = []   # list of dicts with extra key "__source_file__"
for sf in seed_files:
    with open(sf, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print(f"  WARNING: {sf.name} is not a list, skipping")
        continue
    for sent in data:
        sent["__source_file__"] = sf.name
    seed_pool.extend(data)
    print(f"  {sf.name}: {len(data)} sentences")

print(f"\n  Total seed sentences: {len(seed_pool)}")


# ── Step 3: Remove duplicates already in eval ─────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Deduplicating seed vs eval")
print("=" * 60)

eval_norm_texts = {normalize_text(s["sentence"]) for s in eval_data}

before = len(seed_pool)
seed_pool = [s for s in seed_pool if normalize_text(s["sentence"]) not in eval_norm_texts]
removed = before - len(seed_pool)
print(f"  Removed {removed} seed sentences already present in eval.")
print(f"  Seed pool after dedup: {len(seed_pool)}")


# ── Step 4: Top-up eval ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Topping up eval to 15 per label value")
print("=" * 60)

# We'll process fields in a fixed order; top is treated like the rest
all_fields_ordered = list(LABEL_FIELDS.keys()) + ["top"]

sentences_added_to_eval = []  # will hold the moved sentences
moved_from_file = defaultdict(int)  # filename -> count moved

# Keep a set of "used" norm-texts to prevent double-picks within this run
used_in_this_run = set()

def count_non_na_flags(sent: dict) -> int:
    """
    Heuristic 'purity' score: how many label fields have non-trivial values.
    Lower = more 'pure' (fewer other signals firing at once).
    """
    score = 0
    for fld in LABEL_FIELDS:
        vals = get_label_values(sent, fld)
        for v in vals:
            if v not in ("na", "none", "False", "no_topic", "0"):
                score += 1
    tops = get_label_values(sent, "top")
    score += max(0, len(tops) - 1)  # multi-topic adds noise
    return score


shortfall_summary = {}  # (field, value) -> shortfall

for field in all_fields_ordered:
    if field == "top":
        expected_vals = TOP_VALUES
    else:
        expected_vals = LABEL_FIELDS[field]

    for target_val in expected_vals:
        current_count = eval_counts[field].get(target_val, 0)
        if current_count >= MIN_COUNT:
            continue

        needed = MIN_COUNT - current_count
        shortfall_summary[(field, target_val)] = needed

        # Find candidates in seed pool (not yet used in this run)
        candidates = []
        for sent in seed_pool:
            norm = normalize_text(sent["sentence"])
            if norm in used_in_this_run:
                continue
            vals = get_label_values(sent, field)
            if target_val in vals:
                candidates.append(sent)

        if not candidates:
            print(f"  {field}={target_val}: need {needed}, found 0 candidates in seed. SKIPPING.")
            continue

        # Sort by purity (ascending = prefer pure sentences)
        candidates.sort(key=count_non_na_flags)

        picked = candidates[:needed]
        print(f"  {field}={target_val}: need {needed}, found {len(candidates)} candidates, picking {len(picked)}.")

        for sent in picked:
            src_file = sent["__source_file__"]
            norm = normalize_text(sent["sentence"])
            used_in_this_run.add(norm)
            sentences_added_to_eval.append(sent)
            # Update live count so later iterations see updated numbers
            for fld2 in all_fields_ordered:
                for v2 in get_label_values(sent, fld2):
                    eval_counts[fld2][v2] = eval_counts[fld2].get(v2, 0) + 1
            moved_from_file[src_file] += 1

        # Also remove picked from the seed_pool list
        picked_norms = {normalize_text(s["sentence"]) for s in picked}
        seed_pool = [s for s in seed_pool if normalize_text(s["sentence"]) not in picked_norms]


# ── Step 5: Save updated eval ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Saving updated eval file")
print("=" * 60)

# Strip helper key before saving
for sent in sentences_added_to_eval:
    sent.pop("__source_file__", None)
    sent.pop("context_question", None)  # not in the eval schema; actually keep it if present

# Actually keep all fields; just remove our internal tracking key
# Re-add __source_file__ so we know what to strip vs keep
# Let's reload with tracking key intact for the seed update step
# The sentences_added_to_eval already had __source_file__ popped above — redo cleanly.

# Rebuild: we need to re-examine which sentences we moved. Let's redo with tracking.
# Actually sentences_added_to_eval had __source_file__ popped. That's fine for eval.
# For seed file updating we use moved_from_file counts which are already built.
# But we need to know *which* sentences (by normalized text) came from each file.

# Let's collect the norms that were added
added_norms = used_in_this_run.copy()

# Build updated eval list
eval_data_updated = eval_data + sentences_added_to_eval
print(f"  Original eval size: {len(eval_data)}")
print(f"  Sentences added: {len(sentences_added_to_eval)}")
print(f"  New eval size: {len(eval_data_updated)}")

with open(EVAL_FILE, "w", encoding="utf-8") as f:
    json.dump(eval_data_updated, f, ensure_ascii=False, indent=2)
print(f"  Saved: {EVAL_FILE}")


# ── Step 6: Update seed files ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — Updating seed files (removing moved sentences)")
print("=" * 60)

for sf in seed_files:
    if sf.name not in moved_from_file:
        continue
    with open(sf, encoding="utf-8") as f:
        original = json.load(f)

    kept = [s for s in original if normalize_text(s["sentence"]) not in added_norms]
    n_removed = len(original) - len(kept)
    print(f"  {sf.name}: removed {n_removed} sentences ({len(original)} -> {len(kept)})")

    with open(sf, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)


# ── Step 7: Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 — SUMMARY")
print("=" * 60)

print("\nLabel values that were below 15 and by how much:")
for (field, val), shortfall in sorted(shortfall_summary.items()):
    print(f"  {field}={val}: was short by {shortfall}")

print("\nSentences moved by seed file:")
for fname, cnt in sorted(moved_from_file.items()):
    print(f"  {fname}: {cnt} sentences moved")

print(f"\nTotal sentences moved: {len(sentences_added_to_eval)}")

print("\nFinal eval distribution:")
final_counts = {}
for field in list(LABEL_FIELDS.keys()) + ["top"]:
    final_counts[field] = defaultdict(int)

for sent in eval_data_updated:
    for field in LABEL_FIELDS:
        for v in get_label_values(sent, field):
            final_counts[field][v] += 1
    for v in get_label_values(sent, "top"):
        final_counts["top"][v] += 1

for field, val_dict in final_counts.items():
    print(f"\n  {field}:")
    if field == "top":
        expected = TOP_VALUES
    else:
        expected = LABEL_FIELDS[field]
    for v in expected:
        cnt = val_dict.get(v, 0)
        flag = "  <<< STILL BELOW 15" if cnt < MIN_COUNT else ""
        print(f"    {v:30s}: {cnt:4d}{flag}")
    for v, cnt in sorted(val_dict.items()):
        if v not in expected:
            print(f"    [unexpected] {v:24s}: {cnt:4d}")

print("\nDone.")
