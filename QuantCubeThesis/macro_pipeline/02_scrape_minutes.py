"""
02_scrape_minutes.py
--------------------
Scrapes FOMC minutes from the Federal Reserve website for 2004–2010 and saves
each as a JSON file in data/raw/structured_json_minutes/, matching the format
of the existing 2011+ files:

    {
        "date":            "20040128",
        "source_file":     "minutes_20040128.html",
        "paragraph_count": N,
        "content":         ["<paragraph 1>", "<paragraph 2>", ...]
    }

Usage:
    python macro_pipeline/02_scrape_minutes.py
    python macro_pipeline/02_scrape_minutes.py --start 2004 --end 2010
    python macro_pipeline/02_scrape_minutes.py --start 2004 --end 2010 --no-skip
"""

import argparse
import io
import json
import re
import time
import requests
import pdfplumber
from pathlib import Path

from config import MINUTES_JSON_DIR

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    )
}

# URL patterns tried in order; first 200 wins.
# The Fed reorganised its URL structure several times — all known patterns covered.
URL_PATTERNS = [
    "https://www.federalreserve.gov/monetarypolicy/fomcminutes{date}.htm",
    "https://www.federalreserve.gov/fomc/minutes/{date}.htm",
    "https://www.federalreserve.gov/boarddocs/minutes/{year}/{date}.htm",
    "https://www.federalreserve.gov/fomc/MINUTES/{date}.htm",
]

# All FOMC meeting dates 2004–2010 for which minutes were published
FOMC_DATES_2004_2010 = [
    # 2004
    "20040128", "20040316", "20040504", "20040630",
    "20040810", "20040921", "20041110", "20041214",
    # 2005
    "20050202", "20050322", "20050503", "20050630",
    "20050809", "20050920", "20051101", "20051213",
    # 2006
    "20060131", "20060328", "20060510", "20060629",
    "20060808", "20060920", "20061025", "20061212",
    # 2007
    "20070131", "20070321", "20070509", "20070628",
    "20070807", "20070918", "20071031", "20071211",
    # 2008
    "20080122", "20080130", "20080318", "20080430",
    "20080625", "20080805", "20080916", "20081008",
    "20081029", "20081216",
    # 2009
    "20090128", "20090318", "20090429", "20090624",
    "20090812", "20090923", "20091104", "20091216",
    # 2010
    "20100127", "20100316", "20100428", "20100623",
    "20100810", "20100921", "20101103", "20101214",
]


def _fetch(url: str, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.text
        except requests.RequestException:
            pass
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    return None


def _html_to_paragraphs(html: str) -> list[str]:
    """Extract substantive paragraphs from minutes HTML.

    Tries <p> tags first (standard Fed layout). Falls back to splitting on
    double newlines after stripping all tags (older pages).
    Returns list of non-empty paragraph strings.
    """
    # Remove script/style blocks entirely
    html = re.sub(r'<(script|style)[^>]*>.*?</\1>', ' ', html,
                  flags=re.IGNORECASE | re.DOTALL)

    # Try <p> tags first
    paras = re.findall(r'<p[^>]*>(.*?)</p>', html,
                       re.IGNORECASE | re.DOTALL)
    if len(paras) < 5:
        # Older pages use <br> separated paragraphs — fall back
        paras = re.split(r'<br\s*/?\s*>\s*<br\s*/?\s*>', html,
                         flags=re.IGNORECASE)

    cleaned = []
    for p in paras:
        # Strip remaining HTML tags
        text = re.sub(r'<[^>]+>', ' ', p)
        # Normalise whitespace
        text = re.sub(r'[ \t\xA0]+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n', text)
        text = text.strip()
        if len(text) > 40:   # skip navigation links, empty cells, etc.
            cleaned.append(text)
    return cleaned


def _pdf_to_paragraphs(pdf_bytes: bytes) -> list[str]:
    """Extract paragraphs from a PDF using pdfplumber."""
    paras = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for block in text.split("\n\n"):
                block = re.sub(r'[ \t]+', ' ', block).strip()
                if len(block) > 40:
                    paras.append(block)
    return paras


def scrape_minutes(date_str: str) -> dict | None:
    """Fetch minutes for one FOMC meeting; return dict or None if not found.

    Tries HTML URL patterns first, then falls back to the PDF at the
    standard /monetarypolicy/files/fomcminutes{date}.pdf path.
    """
    year = date_str[:4]

    # Try HTML patterns first
    for pattern in URL_PATTERNS:
        url = pattern.format(date=date_str, year=year)
        html = _fetch(url)
        if html is None:
            continue
        paras = _html_to_paragraphs(html)
        if len(paras) >= 5:
            return {
                "date":            date_str,
                "source_file":     f"minutes_{date_str}.html",
                "paragraph_count": len(paras),
                "content":         paras,
            }

    # Fallback: try PDF
    pdf_url = f"https://www.federalreserve.gov/monetarypolicy/files/fomcminutes{date_str}.pdf"
    resp = _fetch(pdf_url)
    if resp is not None:
        try:
            pdf_bytes = requests.get(pdf_url, headers=HEADERS, timeout=30).content
            paras = _pdf_to_paragraphs(pdf_bytes)
            if len(paras) >= 5:
                return {
                    "date":            date_str,
                    "source_file":     f"minutes_{date_str}.pdf",
                    "paragraph_count": len(paras),
                    "content":         paras,
                }
        except Exception:
            pass

    return None


def scrape_all(start_year: int = 2004, end_year: int = 2010,
               output_dir: Path = MINUTES_JSON_DIR,
               skip_existing: bool = True):
    output_dir.mkdir(parents=True, exist_ok=True)
    dates = [d for d in FOMC_DATES_2004_2010
             if start_year <= int(d[:4]) <= end_year]

    print(f"Scraping {len(dates)} FOMC minutes ({start_year}–{end_year}) ...")
    ok = skipped = failed = 0

    for date_str in dates:
        out_path = output_dir / f"minutes_{date_str}.json"
        if skip_existing and out_path.exists():
            skipped += 1
            continue

        record = scrape_minutes(date_str)
        if record:
            out_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
            ok += 1
            print(f"  OK  {date_str}  ({record['paragraph_count']} paragraphs)")
        else:
            failed += 1
            print(f"  --  {date_str}  not found (all URL patterns failed)")

        time.sleep(1.0)   # polite crawl rate

    print(f"\nDone: {ok} saved, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape 2004-2010 FOMC minutes")
    parser.add_argument("--start",    type=int, default=2004)
    parser.add_argument("--end",      type=int, default=2010)
    parser.add_argument("--no-skip",  action="store_true",
                        help="Re-scrape even if file already exists")
    args = parser.parse_args()
    scrape_all(args.start, args.end, skip_existing=not args.no_skip)
