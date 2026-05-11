"""
inflation_unemployment_collisions.py
─────────────────────────────────────
Find every sentence in the unlabelled pool (+ seed files) that simultaneously
mentions inflation AND unemployment/employment, then use Gardner sub-scores to
classify the relationship between the two signals.

Signal taxonomy
───────────────
  Gardner inf_score  > 0  →  inflationary pressure   (hawkish for inflation)
  Gardner inf_score  < 0  →  disinflationary pressure (dovish  for inflation)
  Gardner labor_score > 0  →  strong labour market     (hawkish for employment)
  Gardner labor_score < 0  →  weak   labour market     (dovish  for employment)

  COLLISION  (dual-mandate tension):
    "hawk_inf_dove_labor"  inf > 0 AND labor < 0  — rising prices + weak jobs
    "dove_inf_hawk_labor"  inf < 0 AND labor > 0  — low inflation + tight labour

  REINFORCEMENT (both mandates point the same way):
    "both_hawkish"         inf > 0 AND labor > 0
    "both_dovish"          inf < 0 AND labor < 0

  AMBIGUOUS  (one or both scores == 0 after keyword filtering):
    "ambiguous"
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import ijson

ROOT      = Path(__file__).parent.parent
POOL_PATH = ROOT / "data" / "all_unlabelled_sentences" / "master_unlabelled_pool.json"
SEED_DIR  = ROOT / "data" / "initial_training_seed_sentences"
OUT_PATH  = ROOT / "data" / "inflation_unemployment_collisions.json"

sys.path.insert(0, str(ROOT))
from src.training.Gardner import NLP as gardner_nlp

# ── PRE-FILTER PATTERNS ───────────────────────────────────────────────────────
# Broad keyword sets — intentionally inclusive so we catch all candidate sentences.

_INF_KW = re.compile(
    r'\b(inflation|inflationary|disinflation\w*|deflation\w*|'
    r'price (?:level|stability|pressure|target|index)|'
    r'consumer prices?|pce|cpi|core (?:inflation|pce|cpi)|'
    r'price (?:rise|increase|decline|decrease|fall)|'
    r'2 percent (?:goal|target|objective))\b',
    re.I,
)

_UNEMP_KW = re.compile(
    r'\b(unemployment|unemployed|jobless|labor market|labour market|'
    r'employment|payroll|job (?:gain|loss|opening|market)|'
    r'hiring|workers?|workforce|labor force|'
    r'maximum employment|full employment|'
    r'nonfarm payroll|jolts)\b',
    re.I,
)


def mentions_both(sentence: str) -> bool:
    return bool(_INF_KW.search(sentence)) and bool(_UNEMP_KW.search(sentence))


# ── SIGNAL CLASSIFIER ─────────────────────────────────────────────────────────

def classify_signals(sentence: str) -> dict:
    """
    Run Gardner NLP, extract inf and labor sub-scores, and return a dict with
    the raw scores, collision type, and a human-readable description.
    """
    result     = gardner_nlp(sentence)
    inf_score  = result["gardner_inf"]
    lab_score  = result["gardner_labor"]

    if inf_score > 0 and lab_score < 0:
        rel  = "hawk_inf_dove_labor"
        desc = "Inflationary pressure + weak labour (dual-mandate tension)"
    elif inf_score < 0 and lab_score > 0:
        rel  = "dove_inf_hawk_labor"
        desc = "Disinflationary pressure + strong labour (dual-mandate tension)"
    elif inf_score > 0 and lab_score > 0:
        rel  = "both_hawkish"
        desc = "Both inflation and labour signal hawkish conditions"
    elif inf_score < 0 and lab_score < 0:
        rel  = "both_dovish"
        desc = "Both inflation and labour signal dovish conditions"
    else:
        rel  = "ambiguous"
        desc = "One or both scores zero — no clear directional signal"

    return {
        "inf_score":  round(inf_score,  4),
        "labor_score": round(lab_score, 4),
        "relationship": rel,
        "description":  desc,
    }


# ── RECORD SOURCES ────────────────────────────────────────────────────────────

def iter_pool():
    """Stream the unlabelled pool with ijson."""
    with open(POOL_PATH, "rb") as f:
        yield from ijson.items(f, "item")


def iter_seeds():
    """Yield records from all seed JSON files."""
    for seed_file in sorted(SEED_DIR.glob("seed_*.json")):
        with open(seed_file, encoding="utf-8") as f:
            for rec in json.load(f):
                yield rec


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results   = []
    seen_ids  = set()
    total_scanned = 0

    print("Scanning pool and seed files for inflation + unemployment sentences ...")

    for source_label, source_iter in [("pool", iter_pool()), ("seeds", iter_seeds())]:
        for rec in source_iter:
            total_scanned += 1
            uid      = rec.get("id", "")
            sentence = rec.get("sentence", "")

            if uid in seen_ids or not sentence:
                continue
            if not mentions_both(sentence):
                continue

            seen_ids.add(uid)
            signals = classify_signals(sentence)

            results.append({
                "id":               uid,
                "sentence":         sentence,
                "source":           rec.get("source", ""),
                "doc_type":         rec.get("doc_type", ""),
                "date":             rec.get("date",   ""),
                "context_question": rec.get("context_question"),
                "inf_score":        signals["inf_score"],
                "labor_score":      signals["labor_score"],
                "relationship":     signals["relationship"],
                "description":      signals["description"],
            })

        print(f"  [{source_label}] done")

    # ── Summary ───────────────────────────────────────────────────────────────
    rel_counts = Counter(r["relationship"] for r in results)
    doc_counts = Counter(r["doc_type"]     for r in results)

    total_collision = (
        rel_counts.get("hawk_inf_dove_labor", 0) +
        rel_counts.get("dove_inf_hawk_labor", 0)
    )
    collision_rate = total_collision / len(results) * 100 if results else 0

    print(f"\n── Results ─────────────────────────────────────────────────────────")
    print(f"  Total sentences scanned : {total_scanned:,}")
    print(f"  Mention both signals    : {len(results):,}  "
          f"({100*len(results)/total_scanned:.1f}% of pool)")
    print(f"\n  Signal relationship breakdown:")
    for rel, n in sorted(rel_counts.items(), key=lambda x: -x[1]):
        pct = 100 * n / len(results)
        print(f"    {rel:<30} {n:>5}  ({pct:.1f}%)")
    print(f"\n  Collision rate (dual-mandate tension): "
          f"{total_collision}/{len(results)} = {collision_rate:.1f}%")
    print(f"\n  By doc type:")
    for dt, n in sorted(doc_counts.items(), key=lambda x: -x[1]):
        print(f"    {dt:<35} {n:>5}")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "summary": {
            "total_scanned":      total_scanned,
            "total_matched":      len(results),
            "match_rate_pct":     round(100 * len(results) / total_scanned, 2),
            "collision_total":    total_collision,
            "collision_rate_pct": round(collision_rate, 2),
            "relationship_counts": dict(rel_counts),
            "doc_type_counts":     dict(doc_counts),
        },
        "sentences": results,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} sentences → {OUT_PATH}")
