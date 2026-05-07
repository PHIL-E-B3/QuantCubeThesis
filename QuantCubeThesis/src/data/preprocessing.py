"""
Preprocessing Utilities
=======================
Text cleaning and normalization for FOMC statements.
"""

import re
from typing import List


def clean_fomc_text(text: str) -> str:
    """
    Clean a raw FOMC statement.
    - Removes voting records and procedural boilerplate
    - Normalizes whitespace
    - Keeps economic substance
    """
    # Remove voting records (e.g., "Voting for this action: ...")
    text = re.sub(
        r"Voting\s+for\s+(?:this|the)\s+(?:action|FOMC).*?(?=\n\n|\Z)",
        "", text, flags=re.DOTALL | re.IGNORECASE,
    )
    # Remove "For immediate release" headers
    text = re.sub(r"For\s+immediate\s+release\.?", "", text, flags=re.IGNORECASE)
    # Remove date/location headers
    text = re.sub(r"(?:January|February|March|April|May|June|July|August|"
                  r"September|October|November|December)\s+\d{1,2}(?:[-–]\d{1,2})?,?\s*\d{4}",
                  "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove leading/trailing artifacts
    text = re.sub(r"^[\s\-—]+|[\s\-—]+$", "", text)

    return text


def normalize_sentence(sentence: str) -> str:
    """Normalize a single sentence for model input."""
    # Light normalization — don't destroy Fed-speak nuance
    s = sentence.strip()
    # Normalize quotes
    s = s.replace(""", '"').replace(""", '"').replace("'", "'").replace("'", "'")
    # Normalize dashes
    s = s.replace("—", " - ").replace("–", " - ")
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    return s


def filter_boilerplate(sentences: List[str]) -> List[str]:
    """
    Remove common boilerplate sentences that carry no sentiment signal.
    """
    boilerplate_patterns = [
        r"^the\s+federal\s+open\s+market\s+committee\s+decided",
        r"^the\s+committee\s+directs\s+the\s+desk",
        r"^this\s+action\s+is\s+expected\s+to",
        r"^the\s+vote\s+encompassed",
        r"^information\s+received\s+since",  # Keep this — it's substantive preamble
    ]

    filtered = []
    for s in sentences:
        is_boilerplate = any(
            re.search(pat, s, re.IGNORECASE) for pat in boilerplate_patterns[:4]
        )
        if not is_boilerplate:
            filtered.append(s)

    return filtered
