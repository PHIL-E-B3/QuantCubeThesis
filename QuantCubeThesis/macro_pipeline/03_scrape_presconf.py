"""
03_scrape_presconf.py
--------------------
Scrapes missing FOMC press conference transcripts from the Federal Reserve
website and saves them as JSON files in data/raw/structured_json/, matching
the format of the existing files:

    {
        "source":           "FOMCpresconf20200610.pdf",
        "date":             "20200610",
        "prepared_remarks": "<full raw transcript text>",
        "qa_session":       "",
        "word_counts":      {"total": N, "remarks": N, "qa": 0}
    }

The full transcript (prepared remarks + Q&A) is stored in prepared_remarks as
a flat string with ALLCAPS speaker labels (e.g. "CHAIR POWELL.", "REPORTER NAME.")
so that load_press_conferences() / parse_flat_transcript() can split it correctly.

Transcripts are sourced from the Fed's mediacenter PDF archive:
    https://www.federalreserve.gov/mediacenter/files/FOMCpresconf{YYYYMMDD}.pdf

Usage:
    python macro_pipeline/03_scrape_presconf.py            # scrape missing years
    python macro_pipeline/03_scrape_presconf.py --years 2020
    python macro_pipeline/03_scrape_presconf.py --no-skip  # re-scrape all
"""

import argparse
import io
import json
import re
import time
import requests
import pdfplumber
from pathlib import Path

from config import PRESS_CONF_JSON_DIR   # = data/raw/structured_json

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

PDF_URL = "https://www.federalreserve.gov/mediacenter/files/FOMCpresconf{date}.pdf"

# All FOMC meeting dates from 2020 that have press conference PDFs.
# Extend this list if future years are also missing HTML transcripts.
DATES_TO_SCRAPE = [
    # 2020 (all missing from structured_json)
    "20200129", "20200303", "20200315", "20200429",
    "20200610", "20200729", "20200916", "20201105", "20201216",
]


def _fetch_pdf(date: str) -> bytes | None:
    url = PDF_URL.format(date=date)
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200 and len(r.content) > 10_000:
            return r.content
    except requests.RequestException:
        pass
    return None


def _pdf_to_text(pdf_bytes: bytes) -> str:
    """Extract full transcript text from PDF, preserving speaker-turn structure."""
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            # Strip the running header that appears on every page
            # e.g. "June 10, 2020  Chair Powell's Press Conference  FINAL"
            text = re.sub(
                r'^.*?Chair Powell.s Press Conference.*?(?:FINAL|PRELIMINARY)?\s*\n',
                '', text, flags=re.IGNORECASE | re.MULTILINE
            )
            pages.append(text)

    full = "\n".join(pages)
    # Normalise whitespace while preserving paragraph breaks
    full = re.sub(r'[ \t]+', ' ', full)
    full = re.sub(r'\n{3,}', '\n\n', full)
    return full.strip()


def scrape_presconf(date: str) -> dict | None:
    pdf_bytes = _fetch_pdf(date)
    if pdf_bytes is None:
        return None

    text = _pdf_to_text(pdf_bytes)
    if len(text) < 500:
        return None

    word_count = len(text.split())
    fname = f"FOMCpresconf{date}.pdf"
    return {
        "source":           fname,
        "date":             date,
        "prepared_remarks": text,
        "qa_session":       "",
        "word_counts":      {"total": word_count, "remarks": word_count, "qa": 0},
    }


def scrape_all(dates: list[str], output_dir: Path, skip_existing: bool = True):
    output_dir.mkdir(parents=True, exist_ok=True)
    ok = skipped = failed = 0

    for date in dates:
        out_path = output_dir / f"FOMCpresconf{date}.json"
        if skip_existing and out_path.exists():
            skipped += 1
            continue

        record = scrape_presconf(date)
        if record:
            out_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            ok += 1
            print(f"  OK  {date}  ({record['word_counts']['total']:,} words)")
        else:
            failed += 1
            print(f"  --  {date}  not found")

        time.sleep(1.0)

    print(f"\nDone: {ok} saved, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape missing FOMC press conference PDFs")
    parser.add_argument("--years", nargs="+", type=int,
                        help="Restrict to specific years (default: all in DATES_TO_SCRAPE)")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-scrape even if file already exists")
    args = parser.parse_args()

    dates = DATES_TO_SCRAPE
    if args.years:
        dates = [d for d in dates if int(d[:4]) in args.years]

    print(f"Scraping {len(dates)} press conference transcripts ...")
    scrape_all(dates, PRESS_CONF_JSON_DIR, skip_existing=not args.no_skip)
