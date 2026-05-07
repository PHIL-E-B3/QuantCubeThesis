"""
FOMC Statement Parser
=====================
Decomposes raw FOMC statements into individual sentences,
with metadata tagging for stratified sampling.
"""

import re
import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

import nltk
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize


# ── Fed Chair eras for stratification ──────────────────────────
CHAIR_ERAS = {
    "volcker":   (1979, 1987),
    "greenspan": (1987, 2006),
    "bernanke":  (2006, 2014),
    "yellen":    (2014, 2018),
    "powell":    (2018, 2030),
}

# ── Policy regime keywords (heuristic pre-labels) ─────────────
REGIME_KEYWORDS = {
    "hiking":  ["increase", "raise", "tighten", "higher", "hike"],
    "cutting": ["reduce", "lower", "ease", "cut", "decrease"],
    "hold":    ["maintain", "unchanged", "steady", "current level"],
    "crisis":  ["extraordinary", "unprecedented", "severe", "emergency",
                "pandemic", "financial crisis"],
}


@dataclass
class FOMCSentence:
    """A single sentence extracted from an FOMC statement."""
    sentence_id: str
    text: str
    statement_date: str
    chair_era: str
    regime_hint: str          # heuristic hint, NOT ground truth
    position_in_doc: float    # 0.0 = first sentence, 1.0 = last
    paragraph_idx: int
    word_count: int


def identify_chair_era(year: int) -> str:
    """Map a year to the sitting Fed chair."""
    for chair, (start, end) in CHAIR_ERAS.items():
        if start <= year < end:
            return chair
    return "unknown"


def detect_regime_hint(text: str) -> str:
    """Simple keyword-based heuristic for regime tagging.
    This is NOT a ground-truth label — just for stratification."""
    text_lower = text.lower()
    scores = {}
    for regime, keywords in REGIME_KEYWORDS.items():
        scores[regime] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "ambiguous"


def parse_statement(
    text: str,
    date: str,
    min_words: int = 5,
) -> List[FOMCSentence]:
    """
    Parse a single FOMC statement into sentences.

    Args:
        text: Raw text of the FOMC statement.
        date: Date string in YYYY-MM-DD format.
        min_words: Minimum word count to keep a sentence (filters boilerplate).

    Returns:
        List of FOMCSentence dataclass instances.
    """
    year = int(date[:4])
    chair = identify_chair_era(year)

    # Split into paragraphs, then sentences
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sentences = []

    for para_idx, paragraph in enumerate(paragraphs):
        para_sents = sent_tokenize(paragraph)
        for sent in para_sents:
            sent = sent.strip()
            wc = len(sent.split())
            if wc < min_words:
                continue
            sentences.append((sent, para_idx, wc))

    total = len(sentences)
    results = []
    for i, (sent, para_idx, wc) in enumerate(sentences):
        sid = f"{date}_{i:03d}"
        results.append(FOMCSentence(
            sentence_id=sid,
            text=sent,
            statement_date=date,
            chair_era=chair,
            regime_hint=detect_regime_hint(sent),
            position_in_doc=i / max(total - 1, 1),
            paragraph_idx=para_idx,
            word_count=wc,
        ))

    return results


def parse_all_statements(
    raw_dir: str,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Parse all FOMC statement files in a directory.

    Expects files named like: YYYY-MM-DD.txt
    (or any file with a date parseable from the filename).

    Returns:
        DataFrame with all parsed sentences.
    """
    raw_path = Path(raw_dir)
    all_sentences = []

    txt_files = sorted(raw_path.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {raw_dir}")
        return pd.DataFrame()

    for fpath in txt_files:
        # Extract date from filename
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", fpath.stem)
        if not date_match:
            print(f"  Skipping {fpath.name} — no date in filename")
            continue

        date_str = date_match.group(1)
        text = fpath.read_text(encoding="utf-8")
        sents = parse_statement(text, date_str)
        all_sentences.extend([asdict(s) for s in sents])
        print(f"  Parsed {fpath.name}: {len(sents)} sentences")

    df = pd.DataFrame(all_sentences)

    if output_path and len(df) > 0:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} sentences to {output_path}")

    return df


# ── Stratified sampling for seed set ──────────────────────────
def create_stratified_seed(
    df: pd.DataFrame,
    n_samples: int = 300,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Create a stratified sample for the seed labelling set.
    Stratifies by chair_era and regime_hint to ensure coverage.
    """
    # Create stratification key
    df["strat_key"] = df["chair_era"] + "_" + df["regime_hint"]

    # Proportional sampling, with a minimum of 2 per stratum
    strat_counts = df["strat_key"].value_counts()
    strat_fracs = strat_counts / len(df)

    samples = []
    remaining = n_samples

    for stratum, frac in strat_fracs.items():
        n = max(2, int(n_samples * frac))
        stratum_df = df[df["strat_key"] == stratum]
        n = min(n, len(stratum_df))
        samples.append(stratum_df.sample(n=n, random_state=random_state))
        remaining -= n

    seed_df = pd.concat(samples).drop_duplicates(subset=["sentence_id"])

    # If we oversampled, trim
    if len(seed_df) > n_samples:
        seed_df = seed_df.sample(n=n_samples, random_state=random_state)

    return seed_df.drop(columns=["strat_key"])


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fomc_parser.py <raw_dir> [output.csv]")
        sys.exit(1)

    raw_dir = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else "data/processed/fomc_sentences.csv"
    df = parse_all_statements(raw_dir, output)
    print(f"\nTotal sentences parsed: {len(df)}")
    print(f"Chair eras: {df['chair_era'].value_counts().to_dict()}")
    print(f"Regime hints: {df['regime_hint'].value_counts().to_dict()}")
