"""
Fix dependency-starting sentence fragments in press_conference_qa _tolabel files.

Algorithm:
  For each sentence in a _tolabel file that starts with a PC dependency trigger:
    1. Re-run the chunking algorithm on its source Q&A answer.
    2. Find the chunk that CONTAINS this sentence (exact or as a substring).
    3. If the sentence IS the full chunk → already correctly processed, skip.
    4. If the sentence is a FRAGMENT of a larger chunk → replace with the full chunk.
    5. Purge any sentences from other data files whose text is subsumed by the
       merged chunk, to maintain mutual exclusivity.
"""

import re, json
from pathlib import Path
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt_tab', quiet=True)

# ── constants (verbatim from 00_initialize_data.py first-branch) ─────────────

MAX_CONTEXT_CHARS = 1000

DEPENDENCY_STARTS = (
    "therefore", "thus", "consequently", "as a result", "for example",
    "that ",
    "this is ", "this was ",
    "these are ", "these were ",
    "i think they", "i think those", "i think these", "i think that",
    "we think they", "we think those", "we think these", "we think that",
)

_CONJ_BEFORE_DEMO = re.compile(
    r'\b(although|though|even though|while|whilst|whereas|because|since|'
    r'as\b|when\b|if\b|unless|until|before|after|despite|in spite of)\b',
    re.IGNORECASE,
)

PC_DEPENDENCY_STARTS = DEPENDENCY_STARTS + (
    "so ", "but ", "and ", "then ", "and then", "and so",
)

DEMO_NOUN_PATTERN = re.compile(
    r'\b(?:this|that|these|those|such)\s+'
    r'(outcome|goal|objective|development|progress|condition|measure|'
    r'action|policy|purchase|assessment|view|stance|trend|event|effect|'
    r'forecast|projection|risk|imbalance|strain|'
    r'factor|aspect|area|issue|concern|pressure|indicator|approach|'
    r'dynamic|tension|challenge|uncertainty|decision|situation|'
    r'circumstance|environment|signal|pattern|finding|feature|'
    r'consideration|shift|change|move|step|path|framework|'
    r'backdrop|context|regime|episode|period|phase|cycle)s?\b'
)

_BP = re.compile(
    r'^\s*(mr\.|ms\.|mrs\.|dr\.)\s+[a-z]+\s*[:,]'
    r'|^\s*(thank you|thanks very much|good morning|good afternoon|welcome)[,\.\s!]'
    r'|^\s*--'
    r'|^\s*\.\s*$',
    re.IGNORECASE,
)


def is_boilerplate(s):
    return bool(_BP.search(s))


def is_dependent(sent):
    s = sent.lower()
    if any(s.startswith(t) for t in PC_DEPENDENCY_STARTS):
        return True
    for m in DEMO_NOUN_PATTERN.finditer(s):
        if not _CONJ_BEFORE_DEMO.search(s[:m.start()]):
            return True
    return False


def clean_text(text):
    text = text.replace('’', "'").replace('‘', "'")
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('—', ' -- ').replace('–', '-')
    text = re.sub(r'[ \t\xa0]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()


def chunk_answer(text):
    """Replicate build_annotatable_records(text, is_conversational=True)."""
    chunks = []
    for para in [p.strip() for p in text.split('\n\n') if p.strip()]:
        sents = [s.strip() for s in sent_tokenize(para)
                 if s.strip() and not is_boilerplate(s)]
        if not sents:
            continue
        current = [sents[0]]
        for sent in sents[1:]:
            dep = is_dependent(sent)
            projected = sum(len(s) for s in current) + len(current) + len(sent)
            if dep and projected <= MAX_CONTEXT_CHARS:
                current.append(sent)
            else:
                chunks.append(" ".join(current))
                current = [sent]
        chunks.append(" ".join(current))
    return chunks


def parse_flat_transcript(text):
    text = re.sub(r'\bQUESTION\.\s+', 'JOURNALIST QUESTION. ', text)
    sp = re.compile(r'\b([A-Z]{2,}(?:\s+[A-Z][a-zA-Z\-\']*){1,3})\.\s+')
    parts = sp.split(text)
    FED = {"POWELL", "BERNANKE", "YELLEN", "WARSH", "CHAIR", "CHAIRMAN"}
    prepared, pairs, last_q, qa_started = [], [], None, False
    if parts[0].strip():
        prepared.append(parts[0].strip())
    for i in range(1, len(parts), 2):
        speaker = parts[i].strip().upper()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        is_chair = any(o in speaker for o in FED)
        if not is_chair:
            qa_started = True
            last_q = content
        elif not qa_started:
            prepared.append(content)
        elif last_q:
            pairs.append({"question": last_q, "answer": content})
            last_q = None
        else:
            prepared.append(content)
    return "\n\n".join(prepared), pairs


def find_containing_chunk(source_json, context_question, sentence):
    """
    Re-chunk the matching Q&A answer and return the chunk that contains
    `sentence` as a substring (or exact match). Returns None if not found.
    """
    try:
        with open(source_json, encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return None

    raw = data.get("prepared_remarks", "")
    if isinstance(raw, list):
        raw = " ".join(raw)

    _, pairs = parse_flat_transcript(raw)

    cq = (context_question or "")[:80].lower().strip()
    answer = None
    for pair in pairs:
        if pair["question"][:80].lower().strip() == cq:
            answer = pair["answer"]
            break
    if answer is None:
        for pair in pairs:
            if cq[:40] and cq[:40] in pair["question"].lower():
                answer = pair["answer"]
                break

    if answer is None:
        return None

    chunks = chunk_answer(clean_text(answer))
    sent_norm = sentence.strip().lower()

    for ch in chunks:
        ch_norm = ch.strip().lower()
        if ch_norm == sent_norm:
            return ch  # exact match — already fully merged
        if sent_norm in ch_norm:
            return ch  # sentence is a fragment of this larger chunk
    return None


def remove_if_present(path, sentence_text):
    """Remove records matching sentence_text from a JSON list file. Returns count removed."""
    try:
        with open(path, encoding='utf-8') as f:
            records = json.load(f)
    except Exception:
        return 0
    before = len(records)
    records = [r for r in records if r.get("sentence", "").strip() != sentence_text.strip()]
    if len(records) < before:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
    return before - len(records)


def purge_from_all_files(base_dir, sentence_text, skip_file):
    """Remove sentence_text from all seed/unlabelled JSON files except skip_file."""
    removed = []
    for d in [
        base_dir / "data" / "QuantCube_Seed_Batches",
        base_dir / "data" / "QuantCube_Seed_Labelled",
        base_dir / "data" / "all_unlabelled_sentences",
    ]:
        if not d.exists():
            continue
        for path in d.glob("*.json"):
            if path == skip_file:
                continue
            n = remove_if_present(path, sentence_text)
            if n:
                removed.append(path.name)
    return removed


def main():
    base = Path(__file__).parent.parent
    raw_dir = base / "data" / "raw" / "structured_json"
    labelled_dir = base / "data" / "QuantCube_Seed_Labelled"

    tolabel_files = sorted(labelled_dir.glob("seed_press_conference_qa_batch_*_tolabel.json"))
    print(f"Scanning {len(tolabel_files)} _tolabel files...\n")
    total_fixed = 0

    for tfile in tolabel_files:
        with open(tfile, encoding='utf-8') as f:
            records = json.load(f)

        changed = False
        for rec in records:
            sent = rec["sentence"]
            if not is_dependent(sent):
                continue  # no dependency trigger — nothing to check

            src_name = rec.get("source", "")
            src_path = raw_dir / src_name
            if not src_path.exists():
                src_path = raw_dir / re.sub(r'\.pdf$', '.json', src_name, flags=re.IGNORECASE)
            if not src_path.exists():
                print(f"  [SKIP] Source not found: {src_name}")
                continue

            context_q = rec.get("context_question", "")
            full_chunk = find_containing_chunk(src_path, context_q, sent)

            if full_chunk is None:
                print(f"  [MISS] Not found in source: {sent[:70]!r}")
                continue

            if full_chunk.strip().lower() == sent.strip().lower():
                # Sentence IS the full chunk — correctly merged, nothing to do.
                continue

            # Sentence is a fragment of a larger correctly-merged chunk.
            print(f"  [FIX]  {tfile.name}  id={rec['id'][:8]}")
            print(f"         Was     : {sent[:80]!r}")
            print(f"         Correct : {full_chunk[:80]!r}")

            rec["sentence"] = full_chunk
            changed = True
            total_fixed += 1

            # Purge the old fragment from all other data files (mutual exclusivity)
            removed = purge_from_all_files(base, sent, skip_file=tfile)
            if removed:
                print(f"         Purged old fragment from: {', '.join(removed)}")

            # Also purge the full_chunk if it already existed as a different record
            # (prevents duplicates if an old version of the merged text was there)
            removed2 = purge_from_all_files(base, full_chunk, skip_file=tfile)
            if removed2:
                print(f"         Purged existing full-chunk from: {', '.join(removed2)}")

        if changed:
            with open(tfile, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            print(f"  -> Saved {tfile.name}\n")
        else:
            print(f"  {tfile.name}: no fragments found")

    print(f"\nDone. Total fragments corrected: {total_fixed}")


if __name__ == "__main__":
    main()
