import os
import re
import json
import uuid
import pandas as pd
from pathlib import Path
from datetime import datetime
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt_tab')
# Nota: Si es la primera vez que usas nltk, descomenta las siguientes dos líneas:
# import nltk
# nltk.download('punkt_tab')  # <-- use 'punkt_tab' for newer NLTK versions

# ── CONFIGURATION ────────────────────────────────────────────────────────────
BASE_DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DATA_DIR = BASE_DATA_DIR / "raw"

UNLABELLED_DIR = BASE_DATA_DIR / "all_unlabelled_sentences"
SEED_DIR = BASE_DATA_DIR / "initial_training_seed_sentences"
AL_ROUNDS_DIR = BASE_DATA_DIR / "active_learning_rounds"

# Constants
SEED_SAMPLE_SIZE = 150
CUTOFF_YEAR = 2006 
MAX_CONTEXT_CHARS = 1000

# Model to use for annotation (Claude Opus 4.7 = most capable, or use claude-opus-4-6)
CLAUDE_MODEL = "claude-opus-4-7"

DEPENDENCY_STARTS = (
    # Formal connectives (all doc types)
    "however", "therefore", "thus", "consequently", "as a result", "for example",
    # Epistemic + demonstrative pronoun (all doc types)
    "i think they", "i think those", "i think these", "i think that",
    "we think they", "we think those", "we think these", "we think that",
)

PC_DEPENDENCY_STARTS = DEPENDENCY_STARTS + (
    # Informal connectives applied only to press conference text
    "so ", "but ", "and ", "then ", "and then", "and so",
)

def setup_directories():
    """Create the folder architecture if it doesn't exist."""
    for directory in [UNLABELLED_DIR, SEED_DIR, AL_ROUNDS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    print("📁 Folder architecture verified.")

# ── TEXT PROCESSING ──────────────────────────────────────────────────────────

def clean_text(text):
    # Normalize Unicode curly quotes/dashes to plain ASCII
    text = text.replace('’', "'").replace('‘', "'")
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('—', ' -- ').replace('–', '-')

    # Fix mojibake encoding bugs found in FOMC text (UTF-8 read as Latin-1)
    text = text.replace('â€™', "'").replace('â€', "-")

    text = re.sub(
        r'An official website of the United States Government.*?Resources for Consumers\s*',
        '', text, flags=re.IGNORECASE | re.DOTALL
    )
    if '[SECTION]' in text:
        sections = text.split('[SECTION]')
        sections = [s.strip() for s in sections if len(s.strip()) > 0]
        text = '\n\n'.join(sections)

    text = re.sub(
        r'For release on delivery\s+.*?'
        r'(?:January|February|March|April|May|June|July|August|'
        r'September|October|November|December)\s+\d{1,2},\s+\d{4}'
        r'.*?'
        r'(?:January|February|March|April|May|June|July|August|'
        r'September|October|November|December)\s+\d{1,2},\s+\d{4}\s+',
        '', text, flags=re.IGNORECASE | re.DOTALL
    )
    text = re.sub(
        r'At the conclusion of the discussion.*?(?=Voting for this action)',
        '', text, flags=re.IGNORECASE | re.DOTALL
    )
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'([a-zA-Z])\.(\d+)\s', r'\1. ', text)
    text = re.sub(r'([a-zA-Z]),(\d+)\s', r'\1, ', text)
    text = re.sub(r'([a-zA-Z])\.(\d+)([A-Z])', r'\1. \3', text)
    text = re.sub(
        r'(?<=\.)\s+\d{1,2}\s+(?:See\s+)?[A-Z][a-z].{10,400}?(?:www\.\S+|\.htm[l]?|\.pdf)\.',
        ' ', text, flags=re.DOTALL
    )
    text = re.sub(
        r'\s+\d+\s+[A-Z][^.]+(?:CBO|Congress|Budget|available at|www\.|p\.\s*\d+)[^.]*\.',
        '', text
    )
    text = re.sub(r'^\d+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'(Figure|Table|Chart|Exhibit)\s+[A-Z0-9\-]+[:\.]?', '', text)
    text = re.sub(r'See\s+[A-Z][a-z]+.*?\(\d{4}\).*?\.', '', text)
    text = re.sub(r'\bShare\b\s+', '', text)

    # Bottom cut: remove closing boilerplate from first end marker
    END_MARKERS = [r'Notation Vote', r'Following the FOMC policy vote']
    earliest = None
    for marker in END_MARKERS:
        m = re.search(marker, text, re.IGNORECASE)
        if m and (earliest is None or m.start() < earliest.start()):
            earliest = m
    if earliest:
        text = text[:earliest.start()].strip()

    # Top cut: keep from earliest substantive trigger
    best = None
    for trigger in [r'\breview\b', r'\bdevelopments\b', r'\bparticipants\b',
                    r'\bstaff\b', r'\boutlook\b', r'\bguidance\b']:
        m = re.search(trigger, text, re.IGNORECASE)
        if m and (best is None or m.start() < best.start()):
            best = m
    if best:
        boundaries = list(re.finditer(r'(?:\.|\?|!)\s+(?=[A-Z])', text[:best.start()]))
        if boundaries:
            text = text[boundaries[-1].end():]
        else:
            text = text[best.start():]

    text = re.sub(r'[ \t\xA0]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

# ── BOILERPLATE DETECTION ────────────────────────────────────────────────────

BOILERPLATE_PATTERNS = [
    r'^\s*[A-Z][a-z]+.*?\(\d{4}\).*?(?:pp\.|vol\.|journal|review|quarterly)',
    r'^\s*"[^"]+,"\s*(speech|remarks|testimony|statement)',
    r'doi\.org', r'rey\s*\(\d{4}\)', r'miranda.agrippino',
    r'^\s*ibid\.', r'^\s*op\.\s*cit\.',
    r'^\s*\d+\s*$', r'^\s*page \d+', r'^\s*\d+\s+see\s+',
    r'^\s*\*+\s', r'^\s*[ivxlcdm]+\.\s', r'^\s*\w+\.\s*$',
    r'^\s*(figure|table|chart|exhibit)\s+\w',
    r'^\s*(note|source|notes):\s',
    r'for release on delivery', r'^\s*class i fomc',
    r'authorized for public release', r'restricted controlled',
    r'^\s*transcript of (chairman|the|vice)',
    r'^\s*statement regarding',
    r'^\s*monetary policy normalization\s*$',
    r'^\s*(mr\.|ms\.|mrs\.|dr\.)\s+[a-z]+\s*[:,]',
    r'^\s*present\s*:', r'^\s*voting\s+(for|against)',
    r'^\s*the vote was', r'^\s*(present|voting|absent|attended)',
    r'^\s*(mr\.|ms\.|mrs\.) [a-z]+ (noted|said|stated|added)',
    r'voted as (?:an )?alternate member',
    r'[a-z\s]+@[a-z]+\.[a-z]+', r'[\w\s]+frb\.gov',
    r'for media inquiries', r'^\s*implementation note',
    r'^\s*attendance',
    r'attended (through|the discussion|opening|wednesday|tuesday)',
    r'return to text', r"secretary's note",
    r'^\s*the meeting adjourned',
    r'board of governors.*voted unanimously to (lower|raise|approve)',
    r'jerome h\. powell, chair',
    r'the fomc directs the desk to',
    r'in taking this action, the board approved requests',
    r'(this |the )?vote also encompassed approval by the board of governors',
    r'_{3,}', r'secretary\s+\d+\.\s*$',
    r'^\s*footnote \d+ has been corrected',
    r'^\s*the elected members and alternate members',
    r'by unanimous vote,', r'^\s*the guidelines for the conduct',
    r'^\s*the federal open market committee authorizes and directs',
    r'^\s*to buy or sell', r'^\s*to hold balances of',
    r'^\s*to purchase and sell the following foreign currencies',
    r'^\s*authorization for (domestic|foreign)',
    r'^\s*foreign currency directive',
    r'^\s*procedural instructions with respect to foreign currency',
    r'reaffirmed january',
    r'^\s*in order to ensure the effective conduct of open market',
    r'^\s*the federal reserve bank of new york shall',
    r'^\s*the federal reserve bank of new york may reject',
    r'^\s*all (transactions|operations|federal reserve banks) (undertaken|shall)',
    r'^\s*the foreign currency subcommittee',
    r'^\s*meetings of the subcommittee',
    r'^\s*with the approval of the committee, to enter',
    r'^\s*to keep the secretary of the treasury',
    r'^\s*staff officers of the committee',
    r'system operations in foreign currencies shall',
    r'^\s*undertake spot and forward', r'^\s*maintain reciprocal currency',
    r'^\s*to adjust system balances',
    r'^\s*to provide means for meeting system',
    r'^\s*system foreign currency operations shall',
    r'^\s*in close and continuous consultation',
    r'^\s*in cooperation, as appropriate',
    r'notation vote', r'it was agreed that the next meeting',
    r'the other members of the subcommittee will include',
    r'the role of the subcommittee will be to',
    r'at the conclusion of the discussion, the committee voted to authorize',
    r'the vote encompassed approval of the statement below',
    r'^\s*chairman,\s+[a-z]',
    r"^\s*i'm [a-z]+ [a-z]+ (from|with|of)",
    r'^\s*[a-z]+ [a-z]+[\s,]*(from|with|of|filing for)\s+(the\s+)?'
    r'(wall street journal|financial times|reuters|associated press|'
    r'bloomberg|cnbc|fox business|washington post|new york times|'
    r'politico|axios|marketplace|american banker|marketwatch|'
    r'bankrate|dow jones newswires|market news international|'
    r'agence france.presse|nikkei|l\.a\.\s*times|los angeles times|npr)',
    r'^\s*[a-z]+ [a-z]+,\s+(the\s+)?'
    r'(wall street journal|financial times|reuters|associated press|'
    r'bloomberg|cnbc|fox business|washington post|new york times|'
    r'politico|axios|marketplace|american banker|marketwatch|'
    r'bankrate|dow jones newswires|market news international|'
    r'agence france.presse|nikkei|l\.a\.\s*times|los angeles times|npr)\.',
    r'^\s*thanks\s+(for being here|very much|mr\.|mr\s+chair)',
    r'^\s*thank you[\s,\.!]*$',
    r'^\s*thank you,?\s*(for taking|for being|very much|mr\.|madam chair)',
    r"^\s*i('m| am) happy to take your questions",
    r"^\s*i('ll| will) (then )?be glad to take your questions",
    r'^\s*hi[\s,\.!]', r'^\s*let me stop there',
    r'a table showing the projections has been distributed',
    r'^\s*i turn now to (the committee\'s|our)',
    r'^\s*i would now like to turn',
    r"^\s*i would be pleased to take your questions",
    r'^\s*good (morning|afternoon|evening)[,\.]',
    r'^\s*welcome[\.,]', r'^\s*welcome to the federal reserve',
    r'^\s*i am sorry that',
    r"^\s*we'll go to \w+ for the (last|next) question",
    r'^\s*hey there[--]', r'^\s*--',
    r'chair powell\.?\s*$', r'chair yellen\.?\s*$',
    r'for the last question',
    r'board of governors of the federal reserve system\s+20th street',
    r'return to text accessible version',
    r'^\s*\.\s*$',
]

def is_boilerplate(sentence):
    s = sentence.lower().strip()
    for pattern in BOILERPLATE_PATTERNS:
        if re.search(pattern, s, re.IGNORECASE):
            return True
    if len(sentence) > 0:
        cap_ratio = sum(1 for c in sentence if c.isupper()) / len(sentence)
        if cap_ratio > 0.4 and len(sentence) < 100:
            return True
    word_count = len(sentence.split())
    num_count = len(re.findall(r'\b\d+(?:\.\d+)?\b', sentence))
    if word_count > 0 and num_count / word_count > 0.4:
        return True
    return False

# ── CHUNKING ENGINE ──────────────────────────────────────────────────────────

def build_annotatable_records(text, is_conversational=False):
    records = []
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    triggers = PC_DEPENDENCY_STARTS if is_conversational else DEPENDENCY_STARTS

    for para in paragraphs:
        all_sents_in_para = sent_tokenize(para)
        valid_sents = [s.strip() for s in all_sents_in_para if s.strip() and not is_boilerplate(s)]
        if not valid_sents:
            continue

        current_chunk_sentences = [valid_sents[0]]

        for i in range(1, len(valid_sents)):
            sent = valid_sents[i]
            s_lower = sent.lower()
            is_dependent = any(s_lower.startswith(trigger) for trigger in triggers)

            if not is_dependent:
                pattern = r'\b(?:this|that|these|those|such)\s+(outcome|goal|objective|development|progress|condition|measure|action|policy|purchase|assessment|view|stance|trend|event|effect|forecast|projection|risk|imbalance|strain)s?\b'
                for match in re.finditer(pattern, s_lower):
                    noun = match.group(1)
                    first_occurrence = s_lower.find(noun)
                    if not (first_occurrence != -1 and first_occurrence < match.start()):
                        is_dependent = True
                        break

            projected_len = sum(len(s) for s in current_chunk_sentences) + len(current_chunk_sentences) + len(sent)
            if is_dependent and projected_len <= MAX_CONTEXT_CHARS:
                current_chunk_sentences.append(sent)
            else:
                records.append({
                    "annotatable": " ".join(current_chunk_sentences),
                    "merge_count": len(current_chunk_sentences)
                })
                current_chunk_sentences = [sent]

        records.append({
            "annotatable": " ".join(current_chunk_sentences),
            "merge_count": len(current_chunk_sentences)
        })
    return records

# ── HELPERS & LOADERS ────────────────────────────────────────────────────────

def parse_year(date_str):
    if not date_str or str(date_str).strip() in ("unknown", ""):
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y%m%d", "%Y"):
        try:
            return datetime.strptime(str(date_str).strip()[:19], fmt).year
        except ValueError:
            continue
    m = re.search(r'\b(19|20)\d{2}\b', str(date_str))
    if m:
        return int(m.group())
    print(f"  ⚠️  Could not parse date: {date_str!r}")
    return None

def load_speeches(base_path):
    docs = []
    speeches_path = Path(base_path) / "structured_json_speeches"
    if not speeches_path.exists():
        return docs
    for f in speeches_path.glob("*.json"):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        text = data.get("contents") or data.get("text", "")
        date = data.get("date", "unknown")
        year = parse_year(date)
        if text and year and year > CUTOFF_YEAR:
            docs.append({
                "text": text,
                "source": data.get("title") or data.get("source", f.name),
                "date": date,
                "doc_type": "speech",
            })
    print(f"  Loaded {len(docs)} speeches post-{CUTOFF_YEAR}")
    return docs

def load_statements(base_path):
    docs = []
    stmt_path = Path(base_path) / "structured_json_statements"
    if not stmt_path.exists():
        return docs
    for f in stmt_path.glob("*.json"):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        text = data.get("text", "")
        date = data.get("date", "unknown")
        year = parse_year(date)
        if text and year and year > CUTOFF_YEAR:
            docs.append({
                "text": text,
                "source": data.get("source", f.name),
                "date": date,
                "doc_type": "statement",
            })
    print(f"  Loaded {len(docs)} statements post-{CUTOFF_YEAR}")
    return docs

def load_minutes(base_path):
    docs = []
    min_path = Path(base_path) / "structured_json_minutes"
    if not min_path.exists():
        return docs
    for f in min_path.glob("*.json"):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        paras = data.get("content", [])
        text = "\n\n".join(paras) if isinstance(paras, list) else paras
        date = data.get("date", "unknown")
        year = parse_year(date)
        if text and year and year > CUTOFF_YEAR:
            docs.append({
                "text": text,
                "source": data.get("source_file", f.name),
                "date": date,
                "doc_type": "minutes",
            })
    print(f"  Loaded {len(docs)} minutes post-{CUTOFF_YEAR}")
    return docs

def parse_flat_transcript(text):
    """
    Split a flat press conference transcript into prepared remarks and Q&A pairs.

    Transcripts are stored as a single unbroken string with speaker turns marked
    by an ALL-CAPS tag followed by a period (e.g. "CHAIRMAN BERNANKE." or
    "YLAN MUI."). We slice on those tags, then classify each block as either
    part of the prepared remarks or one side of a Q&A exchange.

    Returns:
        prepared_text (str): The opening statement by the Chair.
        qa_pairs (list[dict]): Each element has "question" (reporter text)
                               and "answer" (Chair's response).
    """
    # Matches 2-4 words where the first is ALL-CAPS; handles names like "McGRANE"
    speaker_pattern = re.compile(r'\b([A-Z]{2,}(?:\s+[A-Z][a-zA-Z.\-\']*){1,3})\.\s+')
    parts = speaker_pattern.split(text)

    FED_OFFICIALS = {"POWELL", "BERNANKE", "YELLEN", "WARSH", "CHAIR", "CHAIRMAN"}

    prepared_remarks = []
    qa_pairs = []
    last_question_text = None

    # parts[0] is any text before the first speaker tag
    intro = parts[0].strip()
    if intro:
        prepared_remarks.append(intro)

    # Step through (speaker, content) pairs
    for i in range(1, len(parts), 2):
        speaker = parts[i].strip().upper()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""

        is_chair = any(official in speaker for official in FED_OFFICIALS)

        if is_chair:
            if last_question_text:
                qa_pairs.append({"question": last_question_text, "answer": content})
                last_question_text = None
            else:
                prepared_remarks.append(content)
        else:
            # Reporter turn — store as the next question
            last_question_text = content

    return "\n\n".join(prepared_remarks), qa_pairs


def load_press_conferences(base_path):
    """
    Load press conference transcripts, splitting each into:
      - Prepared remarks  (doc_type = "press_conference_prepared", context_question = None)
      - Q&A answers       (doc_type = "press_conference_qa",       context_question = <reporter question>)
    """
    docs = []
    pc_path = Path(base_path) / "structured_json"
    if not pc_path.exists():
        return docs

    for f in pc_path.glob("*.json"):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)

        date = data.get("date", "unknown")
        year = parse_year(date)
        if not year or year <= CUTOFF_YEAR:
            continue

        # The full transcript is stored as a flat string in prepared_remarks
        raw_text = data.get("prepared_remarks", "")
        if isinstance(raw_text, list):
            raw_text = " ".join(raw_text)
        if not raw_text:
            continue

        source = data.get("source", f.name)
        prepared_text, qa_pairs = parse_flat_transcript(raw_text)

        if prepared_text:
            docs.append({
                "text":             prepared_text,
                "context_question": None,
                "source":           source,
                "date":             date,
                "doc_type":         "press_conference_prepared",
            })

        for pair in qa_pairs:
            docs.append({
                "text":             pair["answer"],
                "context_question": pair["question"],
                "source":           source,
                "date":             date,
                "doc_type":         "press_conference_qa",
            })

    print(f"  Loaded {len(docs)} press conference components post-{CUTOFF_YEAR}")
    return docs

def process_docs(docs):
    all_records = []
    for doc in docs:
        clean = clean_text(doc["text"])
        is_pc = doc["doc_type"].startswith("press_conference")
        records = build_annotatable_records(clean, is_conversational=is_pc)
        for r in records:
            all_records.append({
                "sentence":         r["annotatable"],
                "source":           doc["source"],
                "date":             doc["date"],
                "doc_type":         doc["doc_type"],
                "context_question": doc.get("context_question"),
            })
    return all_records

# ── PIPELINE EXECUTION ───────────────────────────────────────────────────────

def format_for_claude(records: list) -> list:
    claude_ready = []
    for r in records:
        claude_ready.append({
            "id":               str(uuid.uuid4()),
            "sentence":         r.get("sentence", ""),
            "source":           r.get("source", "unknown"),
            "doc_type":         r.get("doc_type", "unknown"),
            "date":             str(r.get("date", "unknown")),
            "context_question": r.get("context_question"),  # None for non-QA docs
            # Annotation fields (empty until labelled)
            "top": "", "ten": "", "sen": "", "dir": "",
            "com": "", "hor": "", "con": "", "dom": "",
            "ris": "", "wid": ""
        })
    return claude_ready

def initialize_data_pipeline(all_raw_docs: list, preserved_unlabelled: list = None):
    """
    Process documents, write seed files, and update the unlabelled pool.

    Args:
        all_raw_docs: Documents to process (must NOT include minutes — those
                      are handled separately via preserved_unlabelled).
        preserved_unlabelled: Existing unlabelled records to carry over verbatim
                              (i.e. the minutes records from the old pool).
    """
    if preserved_unlabelled is None:
        preserved_unlabelled = []

    print("\n⚙️  Processing documents into chunks...")
    all_chunks = process_docs(all_raw_docs)

    unique_chunks_dict = {r["sentence"]: r for r in all_chunks}
    unique_chunks = list(unique_chunks_dict.values())
    print(f"✅ Total unique sentences generated: {len(unique_chunks)}")

    master_records = format_for_claude(unique_chunks)
    df_master = pd.DataFrame(master_records)

    # Remove stale seed_press_conference.json (replaced by prepared/qa variants)
    stale_seed = SEED_DIR / "seed_press_conference.json"
    if stale_seed.exists():
        stale_seed.unlink()
        print(f"  -> Removed stale {stale_seed.name}")

    print("\n🌱 Generating Seed Batches...")
    seed_records = []

    for doc_type, group in df_master.groupby("doc_type"):
        sample_size = min(SEED_SAMPLE_SIZE, len(group))
        sampled_group = group.sample(n=sample_size, random_state=42)

        seed_filename = SEED_DIR / f"seed_{doc_type}.json"
        with open(seed_filename, 'w', encoding='utf-8') as f:
            json.dump(sampled_group.to_dict(orient="records"), f, indent=2, ensure_ascii=False)
        print(f"  -> Saved {sample_size} {doc_type} sentences to {seed_filename.name}")
        seed_records.extend(sampled_group.to_dict(orient="records"))

    seed_ids = {r["id"] for r in seed_records}

    print("\n📦 Updating Unlabelled Pool...")
    new_unlabelled = df_master[~df_master["id"].isin(seed_ids)].to_dict(orient="records")

    # Merge: preserved minutes + newly generated non-seed records
    combined_unlabelled = preserved_unlabelled + new_unlabelled

    unlabelled_filename = UNLABELLED_DIR / "master_unlabelled_pool.json"
    with open(unlabelled_filename, 'w', encoding='utf-8') as f:
        json.dump(combined_unlabelled, f, indent=2, ensure_ascii=False)

    print(f"  -> Pool total: {len(combined_unlabelled)} sentences")
    print(f"     ({len(preserved_unlabelled)} preserved minutes + {len(new_unlabelled)} new)")
    print(f"\n🤖 Model configured: {CLAUDE_MODEL}")
    print("🚀 Regeneration Complete.")

if __name__ == "__main__":
    setup_directories()

    # ── Preserve already-labelled minutes data ───────────────────────────────
    # seed_minutes.json has been labelled and must not be touched.
    # We keep the minutes records that are in the unlabelled pool so they
    # are not lost when we rebuild the pool for the other doc types.
    preserved_unlabelled = []
    existing_pool_path = UNLABELLED_DIR / "master_unlabelled_pool.json"
    if existing_pool_path.exists():
        with open(existing_pool_path, encoding="utf-8") as f:
            existing_pool = json.load(f)
        preserved_unlabelled = [r for r in existing_pool if r.get("doc_type") == "minutes"]
        print(f"  Preserved {len(preserved_unlabelled)} minutes records from existing pool.")
    else:
        print("  No existing pool found — minutes unlabelled records will be empty.")

    # ── Regenerate press conference, speech, and statement docs only ─────────
    # Minutes are intentionally excluded: seed_minutes.json is already labelled.
    print("\nLoading documents for regeneration...")
    all_raw_docs = []
    all_raw_docs += load_statements(RAW_DATA_DIR)
    all_raw_docs += load_press_conferences(RAW_DATA_DIR)
    all_raw_docs += load_speeches(RAW_DATA_DIR)

    if all_raw_docs:
        initialize_data_pipeline(all_raw_docs, preserved_unlabelled=preserved_unlabelled)
    else:
        print("❌ No documents found. Check your RAW_DATA_DIR paths.")