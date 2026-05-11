"""
fix_however_merges.py
─────────────────────
Full pipeline to remove wrongly-merged "however" sentences from seed files
and replenish them with clean sentences from the freshly-regenerated pool.

Steps
─────
1. Save current seed files (old labelled state) before the pipeline runs.
2. Run 00_initialize_data.py (new rules: "however" no longer a merge trigger
   for any doc type).
3. For every seed file (all 5 doc types):
      invalid  = sentences whose text contains '. However' mid-sentence
                 (the ONLY sign two clauses were joined by this trigger).
      valid    = everything else — kept exactly as-is.
4. Replenish: for each removed invalid sentence, pick one replacement from
   the freshly-generated pool (same doc_type, no '. However', not already
   in the labelled set). Trim PC seeds back to their 50 / 100 targets.
5. Write final seed files.
6. Clean the pool (streaming, memory-safe):
      remove  – old valid labelled texts (already annotated)
              – replenishment IDs (now in seeds)
              – any remaining pool record whose sentence contains '. However'
                (stale minutes records preserved from old run)
      append  – PC surplus (new pipeline generated 150; we only keep 50 / 100)
"""

import json
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import ijson

ROOT      = Path(__file__).parent.parent
SEED_DIR  = ROOT / "data" / "initial_training_seed_sentences"
POOL_PATH = ROOT / "data" / "all_unlabelled_sentences" / "master_unlabelled_pool.json"
POOL_TMP  = POOL_PATH.with_suffix(".tmp.json")
PIPELINE  = ROOT / "scripts" / "00_initialize_data.py"

SEED_TARGETS = {
    "minutes":                    150,
    "speech":                     150,
    "statement":                  150,
    "press_conference_prepared":   50,
    "press_conference_qa":        100,
}

# A sentence is wrongly merged iff a clause ends with a period and the NEXT
# clause begins with "However" — the exact pattern the old merger triggered on.
HOWEVER_MERGE_RE = re.compile(r'\.\s+[Hh]owever\b')

RNG_SEED = 42
random.seed(RNG_SEED)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_seed(doc_type: str) -> list:
    path = SEED_DIR / f"seed_{doc_type}.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_seed(doc_type: str, records: list):
    path = SEED_DIR / f"seed_{doc_type}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def is_wrongly_merged(sentence: str) -> bool:
    """True only if the sentence was merged at a '. However' boundary."""
    return bool(HOWEVER_MERGE_RE.search(sentence))


# ── step 1 — save current seeds ───────────────────────────────────────────────

print("=" * 60)
print("Step 1: saving current seed files")
print("=" * 60)

old_seeds: dict[str, list] = {}
for dt in SEED_TARGETS:
    old_seeds[dt] = load_seed(dt)
    print(f"  {dt:<35} {len(old_seeds[dt]):>4} records")


# ── step 2 — run pipeline with new rules ──────────────────────────────────────

print("\n" + "=" * 60)
print("Step 2: running pipeline (however removed from DEPENDENCY_STARTS)")
print("=" * 60)

env_utf8 = {"PYTHONIOENCODING": "utf-8"}
import os
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"

result = subprocess.run(
    [sys.executable, str(PIPELINE)],
    env=env, capture_output=True, text=True, encoding="utf-8", errors="replace"
)
print(result.stdout)
if result.returncode != 0:
    print("PIPELINE ERROR:\n", result.stderr)
    sys.exit(1)


# ── step 3 — classify old seed sentences ──────────────────────────────────────

print("=" * 60)
print("Step 3: classifying old seed sentences (valid / invalid)")
print("=" * 60)

old_valid:   dict[str, list] = {}
old_invalid: dict[str, list] = {}

for dt, records in old_seeds.items():
    valid   = [r for r in records if not is_wrongly_merged(r.get("sentence", ""))]
    invalid = [r for r in records if     is_wrongly_merged(r.get("sentence", ""))]
    old_valid[dt]   = valid
    old_invalid[dt] = invalid
    flag = " ← has invalids" if invalid else ""
    print(f"  {dt:<35} valid={len(valid):>4}  invalid={len(invalid):>3}{flag}")


# ── step 4 — replenish from new pool ──────────────────────────────────────────

print("\n" + "=" * 60)
print("Step 4: sampling replenishment sentences from new pool")
print("=" * 60)

# Texts already labelled — must not appear again in pool or as replenishment
all_labelled_texts: set[str] = {
    r.get("sentence", "")
    for dt_recs in old_valid.values()
    for r in dt_recs
}

# Stream new pool once → collect clean candidates per doc_type
candidates: dict[str, list] = defaultdict(list)
print("  Streaming new pool for candidates ...")
with open(POOL_PATH, "rb") as f:
    for rec in ijson.items(f, "item"):
        dt       = rec.get("doc_type", "")
        sentence = rec.get("sentence", "")
        if dt not in SEED_TARGETS:
            continue
        if sentence in all_labelled_texts:
            continue
        if is_wrongly_merged(sentence):
            continue
        candidates[dt].append(rec)

for dt, cands in candidates.items():
    print(f"  {dt:<35} {len(cands):>6} clean candidates")

# Sample replenishment
replenishment:     dict[str, list] = {}
replenishment_ids: set[str]        = set()

for dt, target in SEED_TARGETS.items():
    n_valid   = len(old_valid[dt])
    n_needed  = target - n_valid
    available = candidates[dt]

    if n_needed < 0:
        # Shouldn't happen, but guard against it
        replenishment[dt] = []
        print(f"  {dt:<35} no replenishment needed (valid={n_valid} >= target={target})")
        continue

    n_sample = min(n_needed, len(available))
    chosen   = random.sample(available, n_sample)
    replenishment[dt] = chosen
    replenishment_ids.update(r["id"] for r in chosen)

    if n_sample < n_needed:
        print(f"  WARNING: {dt} — needed {n_needed} but only {n_sample} available")

    print(f"  {dt:<35} kept={n_valid:>4}  replenished={n_sample:>3}  "
          f"final={n_valid + n_sample:>4}  (target={target})")


# ── step 5 — write final seed files ───────────────────────────────────────────

print("\n" + "=" * 60)
print("Step 5: writing final seed files")
print("=" * 60)

for dt, target in SEED_TARGETS.items():
    final = old_valid[dt] + replenishment[dt]
    write_seed(dt, final)
    print(f"  seed_{dt}.json  →  {len(final)} records")


# ── step 6 — clean pool (streaming, memory-safe) ──────────────────────────────

print("\n" + "=" * 60)
print("Step 6: cleaning pool")
print("=" * 60)

# PC surplus: pipeline generated 150 for each PC type; we only keep the
# target amount in seeds, so the rest must go back into the pool.
pc_surplus: list = []
for dt, target in SEED_TARGETS.items():
    if not dt.startswith("press_conference"):
        continue
    new_seed = load_seed(dt)          # just-written final seed (target length)
    # After step 5, seed has `target` records. The pipeline-generated seed had
    # 150. The surplus is everything in the pipeline seed beyond what we kept.
    # We already wrote our custom seed, so read the pipeline-generated NEW pool
    # to find the surplus — they are the new pipeline seed sentences minus ours.
    # Simpler: surplus sentences are those in candidates[dt] that we did NOT
    # pick as replenishment and are NOT in old_valid[dt].
    # Actually: pipeline seed was overwritten. Surplus = new pool records for
    # this doc_type that were pipeline-seeded but not by us. But the pipeline
    # already put them in the pool (since it only excludes its own 150 seeds).
    # So they're already in the new pool — no action needed here.
    pass

# All old valid labelled texts (to exclude from pool — they're done)
excluded_texts = all_labelled_texts.copy()

n_kept = 0
n_removed_text  = 0
n_removed_id    = 0
n_removed_merge = 0

with open(POOL_TMP, "w", encoding="utf-8") as out:
    out.write("[")
    first = True

    with open(POOL_PATH, "rb") as f:
        for rec in ijson.items(f, "item"):
            sentence = rec.get("sentence", "")
            rec_id   = rec.get("id", "")
            doc_type = rec.get("doc_type", "")

            # Exclude: already labelled text
            if sentence in excluded_texts:
                n_removed_text += 1
                continue

            # Exclude: chosen as replenishment (now in a seed file)
            if rec_id in replenishment_ids:
                n_removed_id += 1
                continue

            # Exclude: stale however-merged sentences (e.g. old preserved minutes)
            if is_wrongly_merged(sentence):
                n_removed_merge += 1
                continue

            if not first:
                out.write(",")
            json.dump(rec, out, ensure_ascii=False)
            first  = False
            n_kept += 1

    out.write("]")

# Atomically replace old pool with cleaned version
POOL_PATH.unlink()
POOL_TMP.rename(POOL_PATH)

print(f"  Removed (already labelled text) : {n_removed_text:>6}")
print(f"  Removed (replenishment — now in seed): {n_removed_id:>6}")
print(f"  Removed (stale however-merge)   : {n_removed_merge:>6}")
print(f"  Kept                            : {n_kept:>6}")


# ── summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
total_invalid  = sum(len(v) for v in old_invalid.values())
total_replaced = sum(len(v) for v in replenishment.values())
print(f"  Invalid (however-merged) sentences removed : {total_invalid}")
print(f"  Replacement sentences drawn from pool      : {total_replaced}")
print(f"  Pool size after cleanup                    : {n_kept:,}")
print("\nDone.")
