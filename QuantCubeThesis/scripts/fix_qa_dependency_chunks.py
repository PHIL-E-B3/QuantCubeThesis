"""
Fix dependency-starting sentence fragments in press_conference_qa _tolabel files.

For each sentence in a _tolabel file that starts with a PC dependency trigger:
  1. Find its Q&A answer in the source document.
  2. Locate the sentence within that answer using sent_tokenize.
  3. Prepend the IMMEDIATELY PRECEDING sentence (one sentence only).
  4. Purge that preceding sentence from all other data files (mutual exclusivity).

The 3-sentence cap from build_annotatable_records applies only to non-conversational
doc types (is_conversational=False), so it is irrelevant here but kept for reference.
"""

import re, json
from pathlib import Path
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt_tab', quiet=True)

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


def find_preceding_sentence(source_json, context_question, fragment):
    """
    Tokenize the matching Q&A answer and return the single sentence that
    immediately precedes `fragment`. Returns None if fragment is first or not found.
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

    sents = [s.strip() for s in sent_tokenize(clean_text(answer)) if s.strip()]
    frag_norm = fragment.strip().lower()

    for i, s in enumerate(sents):
        # Match: the fragment starts with the first 40 chars of the sentence
        if s.lower().strip()[:40] == frag_norm[:40] and i > 0:
            return sents[i - 1]
        # Or the fragment sentence is a substring start of this sentence
        if frag_norm[:40] in s.lower() and i > 0:
            return sents[i - 1]

    return None


def remove_if_present(path, sentence_text):
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
                continue

            src_name = rec.get("source", "")
            src_path = raw_dir / src_name
            if not src_path.exists():
                src_path = raw_dir / re.sub(r'\.pdf$', '.json', src_name, flags=re.IGNORECASE)
            if not src_path.exists():
                print(f"  [SKIP] Source not found: {src_name}")
                continue

            preceding = find_preceding_sentence(src_path, rec.get("context_question", ""), sent)

            if preceding is None:
                # First sentence of the answer — conversational opener, fine as-is
                continue

            merged = preceding + " " + sent
            print(f"  [FIX]  {tfile.name}  id={rec['id'][:8]}")
            print(f"         Preceding : {preceding[:80]!r}")
            print(f"         Fragment  : {sent[:80]!r}")
            print(f"         Merged    : {merged[:100]!r}")

            rec["sentence"] = merged
            changed = True
            total_fixed += 1

            # Purge the preceding sentence from all other data files
            removed = purge_from_all_files(base, preceding, skip_file=tfile)
            if removed:
                print(f"         Purged from: {', '.join(removed)}")
            else:
                print(f"         (preceding not found in other files)")
            print()

        if changed:
            with open(tfile, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Done. Total fragments fixed: {total_fixed}")


if __name__ == "__main__":
    main()
