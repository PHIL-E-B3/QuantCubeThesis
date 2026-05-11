"""
extract_contrastive_merges.py
──────────────────────────────
Collect every sentence in the pool + seeds where a contrastive/contradictory
connective appears mid-sentence (after a period), indicating two originally
separate sentences were merged by the chunker.

Pattern searched:  [end of clause]  .  [space]  [contrastive word]  ...
e.g. "Inflation remained elevated. However, the labour market showed signs..."

For each match the output records:
  - full merged sentence
  - which trigger word fired
  - the clause BEFORE the trigger  ("Inflation remained elevated.")
  - the clause AFTER  the trigger  ("the labour market showed signs...")
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import ijson

ROOT      = Path(__file__).parent.parent
POOL_PATH = ROOT / "data" / "all_unlabelled_sentences" / "master_unlabelled_pool.json"
SEED_DIR  = ROOT / "data" / "initial_training_seed_sentences"
OUT_PATH  = ROOT / "data" / "contrastive_merges.json"

# ── CONTRASTIVE PATTERNS ──────────────────────────────────────────────────────
# Restricted to the EXACT words in DEPENDENCY_STARTS / PC_DEPENDENCY_STARTS
# that are contradictory in nature, so we only catch sentences that were
# actually merged by the pipeline — not natural multi-clause source sentences.
#
#   All doc types  →  "however"
#   Press conf only →  "but"  (in PC_DEPENDENCY_STARTS)

# Matches any doc type: sentence ends with period, then "However"
_HOWEVER_RE = re.compile(r'\.\s+(however)\b', re.IGNORECASE)

# Matches press conference doc types only: sentence ends with period, then "But"
_BUT_RE = re.compile(r'\.\s+(but)\b', re.IGNORECASE)

PC_DOC_TYPES = {"press_conference_prepared", "press_conference_qa"}


def find_contrastive_splits(sentence: str, doc_type: str) -> list[dict]:
    """
    Return one dict per pipeline-merged contrastive hinge found in the sentence.
    Only checks "however" (all docs) and "but" (press conference only).
    """
    patterns = [_HOWEVER_RE]
    if doc_type in PC_DOC_TYPES:
        patterns.append(_BUT_RE)

    matches = []
    for pattern in patterns:
        for m in pattern.finditer(sentence):
            trigger   = m.group(1).lower()
            split_pos = m.start()
            before    = sentence[:split_pos].strip()
            after     = sentence[m.end():].strip()
            matches.append({
                "trigger":        trigger,
                "before_trigger": before,
                "after_trigger":  after,
            })
    return matches


# ── RECORD ITERATORS ──────────────────────────────────────────────────────────

def iter_pool():
    with open(POOL_PATH, "rb") as f:
        yield from ijson.items(f, "item")


def iter_seeds():
    for seed_file in sorted(SEED_DIR.glob("seed_*.json")):
        with open(seed_file, encoding="utf-8") as f:
            for rec in json.load(f):
                yield rec


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results      = []
    seen_ids     = set()
    total_scanned = 0

    print("Scanning pool + seeds for contrastive merges ...")

    for label, source in [("pool", iter_pool()), ("seeds", iter_seeds())]:
        for rec in source:
            total_scanned += 1
            uid      = rec.get("id", "")
            sentence = rec.get("sentence", "")

            if uid in seen_ids or not sentence:
                continue

            splits = find_contrastive_splits(sentence, rec.get("doc_type", ""))
            if not splits:
                continue

            seen_ids.add(uid)
            base = {
                "id":               uid,
                "sentence":         sentence,
                "source":           rec.get("source", ""),
                "doc_type":         rec.get("doc_type", ""),
                "date":             str(rec.get("date", "")),
                "context_question": rec.get("context_question"),
            }
            # One result entry per contrastive hinge in the sentence
            for split in splits:
                results.append({**base, **split})

        print(f"  [{label}] done")

    # ── Summary ───────────────────────────────────────────────────────────────
    trigger_counts = Counter(r["trigger"]   for r in results)
    doc_counts     = Counter(r["doc_type"]  for r in results)
    # Unique sentence count (a sentence with two hinges appears twice above)
    unique_sentences = len(seen_ids)

    print(f"\n── Results ─────────────────────────────────────────────────────────")
    print(f"  Total sentences scanned         : {total_scanned:,}")
    print(f"  Sentences with contrastive merge: {unique_sentences:,}  "
          f"({100*unique_sentences/total_scanned:.1f}%)")
    print(f"  Total contrastive hinges found  : {len(results):,}")

    print(f"\n  By trigger word:")
    for t, n in trigger_counts.most_common():
        print(f"    {t:<25} {n:>5}")

    print(f"\n  By doc type:")
    for dt, n in sorted(doc_counts.items(), key=lambda x: -x[1]):
        print(f"    {dt:<35} {n:>5}")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "summary": {
            "total_scanned":          total_scanned,
            "unique_sentences":       unique_sentences,
            "match_rate_pct":         round(100 * unique_sentences / total_scanned, 2),
            "total_hinges":           len(results),
            "trigger_counts":         dict(trigger_counts.most_common()),
            "doc_type_counts":        dict(doc_counts),
        },
        "sentences": results,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} entries → {OUT_PATH}")
