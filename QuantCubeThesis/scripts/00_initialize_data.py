import os
import json
import uuid
import random
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
# Define your base paths
BASE_DATA_DIR = Path("data")
RAW_DATA_DIR = BASE_DATA_DIR / "raw"

# Define the new pipeline directories
UNLABELLED_DIR = BASE_DATA_DIR / "all_unlabelled_sentences"
SEED_DIR = BASE_DATA_DIR / "initial_training_seed_sentences"
AL_ROUNDS_DIR = BASE_DATA_DIR / "active_learning_rounds"

# Constants
SEED_SAMPLE_SIZE = 150

def setup_directories():
    """Create the folder architecture if it doesn't exist."""
    for directory in [UNLABELLED_DIR, SEED_DIR, AL_ROUNDS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    print("📁 Folder architecture verified.")

# ==============================================================================
# PASTE YOUR TEXT PROCESSING FUNCTIONS HERE
def clean_text(text):
    text = re.sub(
        r'An official website of the United States Government.*?Resources for Consumers\s*',
        '', text, flags=re.IGNORECASE | re.DOTALL
    )
    if '[SECTION]' in text:
        sections = text.split('[SECTION]')
        sections = [s.strip() for s in sections if len(s.strip()) > 0]
        text = ' '.join(sections)
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
    text = re.sub(
        r'^(January|February|March|April|May|June|July|August|'
        r'September|October|November|December)\s+\d{1,2},\s+\d{4}\s+',
        '', text, flags=re.IGNORECASE
    )
    text = re.sub(
        r'^(January|February|March|April|May|June|July|August|'
        r'September|October|November|December)\s+\d{1,2}[--]\d{1,2},\s*\d{4}',
        '', text, flags=re.IGNORECASE
    )
    text = re.sub(
        r'^(January|February|March|April|May|June|July|August|'
        r'September|October|November|December)\s+\d{1,2},\s+\d{4}\s+'
        r'FOMC\s+\w+\s+\w+\s+',
        '', text, flags=re.IGNORECASE
    )
    text = re.sub(
        r'\s+[A-Z]{2,}(?:\s+[A-Z]{2,})+\.\s*$',
        '', text, flags=re.MULTILINE
    )
    text = re.sub(
        r'(?:'
        r'Developments in Financial Markets and Open Market Operations|'
        r'Developments in Financial Markets|'
        r'Staff Review of the (?:Economic Situation|Financial Situation)|'
        r'Staff Economic Outlook|'
        r"Participants' Views on Current Conditions and the Economic Outlook|"
        r"Participants' Views on Current Conditions|"
        r'Committee Policy Actions?|'
        r'The Standing Repo Facility|'
        r'Balance Sheet Issues|'
        r'Monetary Policy Normalization|'
        r'Structural Unemployment|'
        r'Special Topic:[^A-Z]{0,50}|'
        r'The Economic Outlook|'
        r'Fiscal Policy|'
        r'Financial Markets|'
        r'Labor Market|'
        r'Concluding Remarks|'
        r'Recent Developments|'
        r'The Outlook'
        r')([A-Z])',
        r'\1', text
    )
    text = re.sub(r'If the rate of interest on\s+Under', 'Under', text)
    text = re.sub(
        r'^.*?(?=The manager of the System Open Market Account)',
        '', text, flags=re.IGNORECASE | re.DOTALL
    )
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ══════════════════════════════════════════════════════════════════════════════
# BOILERPLATE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

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
    r'^\s*(january|february|march|april|may|june|july|august|'
     r'september|october|november|december)\s+\d{1,2},?\s+\d{4}\s*$',
    r'^\s*transcript of (chairman|the|vice)',
    r'^\s*statement regarding',
    r'^\s*monetary policy normalization\s*$',
    r'^\s*(mr\.|ms\.|mrs\.|dr\.)\s+[a-z]+\s*[:,]',
    r'^\s*present\s*:', r'^\s*voting\s+(for|against)',
    r'^\s*the vote was', r'^\s*(present|voting|absent|attended)',
    r'^\s*(mr\.|ms\.|mrs\.) [a-z]+ (noted|said|stated|added)',
    r'voted as alternate member',
    r'[a-z\s]+@[a-z]+\.[a-z]+', r'[\w\s]+frb\.gov',
    r'for media inquiries', r'^\s*implementation note',
    r'^\s*attendance',
    r'attended (through|the discussion|opening|wednesday|tuesday)',
    r'return to text', r"secretary's note",
    r'^\s*the meeting adjourned',
    r'board of governors.*voted unanimously to (lower|raise|approve)',
    r'jerome h\. powell, chair',
    r'effective (october|january|february|march|april|may|june|'
     r'july|august|september|november|december)',
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
    num_count  = len(re.findall(r'\b\d+(?:\.\d+)?\b', sentence))
    if word_count > 0 and num_count / word_count > 0.4:
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# CHUNKER
# ══════════════════════════════════════════════════════════════════════════════

SHORT_THRESHOLD = 8

PRONOUN_STARTS = (
    "it's", "it is", "its", "that's", "that is", "they've", "they are",
    "they're", "this is", "this has", "these are", "those are", "he's",
    "she's", "we've", "we're", "there's", "there are", "which is",
    "which has", "that'll", "it'll", "it has",
)
CONJUNCTION_STARTS = (
    "and ", "but ", "so ", "or ", "nor ", "yet ", "because ", "although ",
    "however ", "moreover ", "furthermore ", "still ", "also ", "plus ",
)
DEMONSTRATIVE_STARTS = (
    "this ", "that ", "these ", "those ", "such ", "the former ", "the latter ",
)
FILLER_PATTERNS = (
    r"^(you know[\.,]?|i mean[\.,]?|right[\.,]?|okay[\.,]?|so[\.,]?)\s*$",
    r"^(again[\.,]?|exactly[\.,]?|precisely[\.,]?|absolutely[\.,]?)\s*$",
)

def is_incomplete(sentence, is_press_conf=False):
    s = sentence.strip().lower()
    if any(s.startswith(p) for p in DEMONSTRATIVE_STARTS):
        return True
    if is_press_conf:
        if len(s.split()) < SHORT_THRESHOLD:
            return True
        if any(s.startswith(p) for p in PRONOUN_STARTS):
            return True
        if any(s.startswith(c) for c in CONJUNCTION_STARTS):
            return True
        if any(re.match(p, s) for p in FILLER_PATTERNS):
            return True
    return False

def chunk_sentences(sentences, is_press_conf=False):
    if not sentences:
        return []
    chunks = []
    buffer = sentences[0]
    for curr in sentences[1:]:
        would_be = buffer.rstrip() + " " + curr.strip()
        if is_incomplete(curr, is_press_conf) and len(would_be) <= MAX_CHUNK_CHARS:
            buffer = would_be
        else:
            chunks.append(buffer)
            buffer = curr
    chunks.append(buffer)
    return chunks
# ══════════════════════════════════════════════════════════════════════════════
# DATE PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_year(date_str):
    if not date_str or str(date_str).strip() in ("unknown", ""):
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y%m%d", "%Y"):
        try:
            return datetime.strptime(str(date_str).strip()[:19], fmt).year
        except:
            continue
    m = re.search(r'\b(19|20)\d{2}\b', str(date_str))
    return int(m.group()) if m else None
# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_speeches(base_path):
    docs = []
    with open(PICKLE_PATH, "rb") as f:
        import pickle
        df = pickle.load(f)
    for _, row in df.iterrows():
        text = row.get("contents", "")
        date = str(row.get("date", "unknown"))
        if text and parse_year(date) and parse_year(date) > CUTOFF_YEAR:
            docs.append({
                "text": text,
                "source": str(row.get("title", "unknown")),
                "date": date,
                "doc_type": "speech",
            })
    print(f"  Loaded {len(docs)} speeches post-{CUTOFF_YEAR}")
    return docs

def load_statements(base_path):
    docs = []
    for f in (Path(base_path) / "structured_json_statements").glob("*.json"):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        text = data.get("text", "")
        date = data.get("date", "unknown")
        if text and parse_year(date) and parse_year(date) > CUTOFF_YEAR:
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
    for f in (Path(base_path) / "structured_json_minutes").glob("*.json"):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        paragraphs = data.get("content", [])
        text = " ".join(paragraphs) if isinstance(paragraphs, list) else paragraphs
        date = data.get("date", "unknown")
        if text and parse_year(date) and parse_year(date) > CUTOFF_YEAR:
            docs.append({
                "text": text,
                "source": data.get("source_file", f.name),
                "date": date,
                "doc_type": "minutes",
            })
    print(f"  Loaded {len(docs)} minutes post-{CUTOFF_YEAR}")
    return docs

def load_press_conferences(base_path):
    docs = []
    pc_path = Path(base_path) / "structured_json"
    if not pc_path.exists():
        return docs
    for f in pc_path.glob("*.json"):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        raw = data.get("prepared_remarks", "")
        text = " ".join(raw) if isinstance(raw, list) else raw
        date = data.get("date", "unknown")
        if text and parse_year(date) and parse_year(date) > CUTOFF_YEAR:
            docs.append({
                "text": text,
                "source": data.get("source", f.name),
                "date": date,
                "doc_type": "press_conference",
            })
    print(f"  Loaded {len(docs)} press conferences post-{CUTOFF_YEAR}")
    return docs


# ══════════════════════════════════════════════════════════════════════════════
# PROCESS DOCUMENTS → CHUNKS
# ══════════════════════════════════════════════════════════════════════════════

def process_docs(docs):
    """Clean, split, remove boilerplate, chunk. Returns list of chunk dicts."""
    all_chunks = []
    for doc in docs:
        is_pc = doc["doc_type"] == "press_conference"
        clean  = clean_text(doc["text"])
        sents  = sent_tokenize(clean)
        kept   = [s.strip() for s in sents if s.strip() and not is_boilerplate(s)]
        chunks = chunk_sentences(kept, is_press_conf=is_pc)
        for chunk in chunks:
            all_chunks.append({
                "sentence": chunk,
                "source":   doc["source"],
                "date":     doc["date"],
                "doc_type": doc["doc_type"],
            })
    return all_chunks

# ==============================================================================

def format_for_claude(records: list) -> list:
    """Formats a list of text records into the blank JSON structure Claude expects."""
    claude_ready = []
    for r in records:
        claude_ready.append({
            "id": str(uuid.uuid4()),  # Attach a permanent unique ID
            "sentence": r["sentence"],
            "source": r.get("source", "unknown"),
            "doc_type": r.get("doc_type", "unknown"),
            "date": str(r.get("date", "unknown")),
            # Empty fields for Claude
            "top": "", "ten": "", "sen": "", "dir": "", 
            "com": "", "hor": "", "con": "", "dom": "", 
            "ris": "", "wid": ""
        })
    return claude_ready

def initialize_data_pipeline(all_raw_docs: list):
    """
    1. Processes all documents into clean sentences.
    2. Deduplicates them.
    3. Extracts the seed batch (split by doc type).
    4. Saves the remainder to the unlabelled pool.
    """
    print("\n⚙️ Processing documents into chunks...")
    # Assume process_docs is your existing function that applies the chunking engine
    all_chunks = process_docs(all_raw_docs) 
    
    # Deduplicate identically matched sentences (copy-pasted FOMC boilerplate)
    unique_chunks_dict = {r["sentence"]: r for r in all_chunks}
    unique_chunks = list(unique_chunks_dict.values())
    print(f"✅ Total unique sentences generated: {len(unique_chunks)}")

    # Format everything to Claude's JSON schema with UUIDs
    master_records = format_for_claude(unique_chunks)
    df_master = pd.DataFrame(master_records)

    print("\n🌱 Generating Initial Seed Batches...")
    seed_records = []
    
    # Group by document type to extract the exact sample size for each
    for doc_type, group in df_master.groupby("doc_type"):
        sample_size = min(SEED_SAMPLE_SIZE, len(group))
        
        # Sample the records
        sampled_group = group.sample(n=sample_size, random_state=42)
        
        # Save this specific doc_type to its own JSON in the seed folder
        seed_filename = SEED_DIR / f"seed_{doc_type}.json"
        with open(seed_filename, 'w', encoding='utf-8') as f:
            # Convert back to dict for clean JSON export
            json.dump(sampled_group.to_dict(orient="records"), f, indent=2)
            
        print(f"  -> Saved {sample_size} {doc_type} sentences to {seed_filename.name}")
        seed_records.extend(sampled_group.to_dict(orient="records"))

    # Create a set of the Seed IDs so we can remove them from the unlabelled pool
    seed_ids = set([r["id"] for r in seed_records])

    # Filter out the seed sentences to create the Unlabelled Pool
    print("\n📦 Generating Unlabelled Pool...")
    df_unlabelled = df_master[~df_master["id"].isin(seed_ids)]
    
    unlabelled_filename = UNLABELLED_DIR / "master_unlabelled_pool.json"
    with open(unlabelled_filename, 'w', encoding='utf-8') as f:
        json.dump(df_unlabelled.to_dict(orient="records"), f, indent=2)

    print(f"  -> Quarantined {len(df_unlabelled)} sentences to {unlabelled_filename.name}")
    print("\n🚀 Initialization Complete. Ready for Claude.")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    setup_directories()
    
    print("Loading documents...")
    all_raw_docs = []
    
    # Notice we pass RAW_DATA_DIR to your original loader functions!
    all_raw_docs += load_statements(RAW_DATA_DIR)
    all_raw_docs += load_press_conferences(RAW_DATA_DIR)
    all_raw_docs += load_minutes(RAW_DATA_DIR)
    all_raw_docs += load_speeches(RAW_DATA_DIR) # Make sure to update PICKLE_PATH in your function to point here too!
    
    initialize_data_pipeline(all_raw_docs)