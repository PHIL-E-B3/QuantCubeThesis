"""
build_full_corpus.py — Extract ALL FOMC sentences from raw docs into a complete,
ordered corpus suitable for full-corpus LLM inference and regression analysis.

Unlike master_unlabelled_pool.json (built for active learning), this corpus:
  - Includes EVERY sentence from every document — no de-duplication
  - Identical sentences in different documents get distinct records with distinct IDs
  - Sentences are ordered: by date, then source, then position within document
  - IDs are STABLE: uuid5(NAMESPACE, "{source}::{sentence_idx}") so re-running
    the script always produces the same IDs

Output: data/full_corpus_ordered.json

Usage:
    python scripts/build_full_corpus.py
    python scripts/build_full_corpus.py --output data/full_corpus_ordered.json
    python scripts/build_full_corpus.py --dry-run   # count sentences only, no write
"""
import argparse
import importlib.util
import json
import uuid
from collections import Counter
from pathlib import Path

# ── Load processing functions from 00_initialize_data.py ─────────────────────
# (importlib needed because the filename starts with a digit)
_INIT_PATH = Path(__file__).parent / "00_initialize_data.py"
_spec = importlib.util.spec_from_file_location("initialize_data", _INIT_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

clean_text              = _mod.clean_text
build_annotatable_records = _mod.build_annotatable_records
load_minutes            = _mod.load_minutes
load_statements         = _mod.load_statements
load_press_conferences  = _mod.load_press_conferences
load_speeches           = _mod.load_speeches

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
RAW_DATA_DIR = ROOT / "data" / "raw"
DEFAULT_OUT  = ROOT / "data" / "full_corpus_ordered.json"

# Stable-ID namespace — fixed so IDs are reproducible across runs
_CORPUS_NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

# Per-doc-type cutoff years (include year > cutoff, i.e. cutoff+1 onwards).
# Minutes and press conferences start naturally from 2011 regardless.
_CUTOFF = {
    "statements":             1999,  # from 2000
    "minutes":                1999,  # from 2004 in theory; raw data only has 2011+
    "press_conferences":      1999,  # from 2011 (Bernanke era onwards)
    "speeches":               1999,  # from 2000
}


def make_stable_id(source: str, sentence_idx: int) -> str:
    """Deterministic UUID from document source + sentence position.

    Same source + same position → same ID every run.
    Same text in two different documents → two different IDs (source differs).
    """
    return str(uuid.uuid5(_CORPUS_NS, f"{source}::{sentence_idx}"))


def build_full_corpus(raw_data_dir: Path) -> list[dict]:
    """Load all raw docs and extract sentences in document order."""
    print("Loading raw documents...")
    all_docs = []
    for loader, key in [
        (load_minutes,           "minutes"),
        (load_statements,        "statements"),
        (load_press_conferences, "press_conferences"),
        (load_speeches,          "speeches"),
    ]:
        _mod.CUTOFF_YEAR = _CUTOFF[key]
        all_docs += loader(raw_data_dir)
    print(f"  Total document components: {len(all_docs)}")

    # Sort by date then source for stable, chronological ordering
    all_docs.sort(key=lambda d: (str(d.get("date", "")), str(d.get("source", ""))))

    records = []
    for doc in all_docs:
        clean = clean_text(doc["text"])
        is_pc = doc["doc_type"].startswith("press_conference")
        chunks = build_annotatable_records(clean, is_conversational=is_pc)

        source = doc["source"]
        for idx, chunk in enumerate(chunks):
            records.append({
                "id":               make_stable_id(source, idx),
                "sentence":         chunk["annotatable"],
                "source":           source,
                "doc_type":         doc["doc_type"],
                "date":             str(doc.get("date", "")),
                "context_question": doc.get("context_question"),
                "sentence_idx":     idx,   # position within document for ordering
            })

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--dry-run", action="store_true",
                        help="Count sentences without writing output file")
    args = parser.parse_args()

    records = build_full_corpus(RAW_DATA_DIR)

    # Summary
    doc_type_counts = Counter(r["doc_type"] for r in records)
    date_vals = [r["date"] for r in records if r["date"] not in ("", "unknown")]
    print(f"\nTotal sentences: {len(records)}")
    print(f"Date range: [{min(date_vals)}, {max(date_vals)}]")
    print("By doc_type:")
    for dt, n in sorted(doc_type_counts.items()):
        print(f"  {dt:<35} {n:>6}")

    # Duplicate sentence text check (informational only — duplicates are KEPT)
    texts = [r["sentence"] for r in records]
    n_unique_texts = len(set(texts))
    n_duplicate_texts = len(texts) - n_unique_texts
    if n_duplicate_texts:
        print(f"\nNote: {n_duplicate_texts} sentences share text with another record "
              f"(kept — same text in different docs is intentional).")

    if args.dry_run:
        print("\n--dry-run: no file written.")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
