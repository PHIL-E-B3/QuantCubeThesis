"""
01_scrape_statements.py
-----------------------
Scrapes FOMC statements from the Federal Reserve website for 1994-2010
and saves each as both .txt and .json in data/raw/structured_json_statements/.

Post-2010 statements are handled by the existing pipeline (already in the
structured_json_statements folder).  Run this once to back-fill the history.

Usage:
    python macro_pipeline/01_scrape_statements.py
    python macro_pipeline/01_scrape_statements.py --start 2005 --end 2010
"""

import argparse
import json
import re
import time
import requests
from pathlib import Path

from config import STATEMENTS_JSON_DIR

# ── URL patterns by era ───────────────────────────────────────────────────────
# Each pattern is tried in order; first that returns HTTP 200 wins.
URL_PATTERNS = [
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary{date}a.htm",
    "https://www.federalreserve.gov/newsevents/press/monetary/{date}a.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/{year}/{date}/",
    "https://www.federalreserve.gov/boarddocs/press/general/{year}/{date}/",
    "https://www.federalreserve.gov/fomc/{date}",
]

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    )
}

# Known FOMC meeting dates 1994–2010 (YYYYMMDD)
FOMC_DATES_1994_2010 = [
    # 1994
    "19940204","19940322","19940517","19940706","19940816","19940927","19941115",
    # 1995
    "19950201","19950328","19950523","19950706","19950822","19950926","19951115",
    "19951219",
    # 1996
    "19960131","19960326","19960521","19960703","19960820","19961224",
    # 1997
    "19970205","19970325","19970520","19970702","19970819","19970930","19971112",
    "19971216",
    # 1998
    "19980204","19980331","19980519","19980701","19980818","19980929","19981015",
    "19981117","19981222",
    # 1999
    "19990203","19990330","19990518","19990630","19990824","19991005","19991116",
    "19991221",
    # 2000
    "20000202","20000321","20000516","20000628","20000822","20001003","20001115",
    "20001219",
    # 2001
    "20010103","20010131","20010320","20010515","20010627","20010821","20010917",
    "20011002","20011106","20011211",
    # 2002
    "20020130","20020319","20020507","20020626","20020813","20020924","20021106",
    "20021210",
    # 2003
    "20030129","20030318","20030506","20030625","20030812","20030916","20031028",
    "20031209",
    # 2004
    "20040128","20040316","20040504","20040630","20040810","20040921","20041110",
    "20041214",
    # 2005
    "20050202","20050322","20050503","20050630","20050809","20050920","20051101",
    "20051213",
    # 2006
    "20060131","20060328","20060510","20060629","20060808","20060920","20061025",
    "20061212",
    # 2007
    "20070131","20070321","20070509","20070628","20070807","20070918","20071031",
    "20071211",
    # 2008
    "20080122","20080130","20080318","20080430","20080625","20080805","20080916",
    "20081008","20081029","20081216",
    # 2009
    "20090128","20090318","20090429","20090624","20090812","20090923","20091104",
    "20091216",
    # 2010
    "20100127","20100316","20100428","20100623","20100810","20100921","20101103",
    "20101214",
]


def _fetch_url(url: str, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                return r.text
        except requests.RequestException:
            pass
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    return None


def _extract_text(html: str) -> str | None:
    # Strategy 1: everything after "For release" header
    m = re.search(r'For release.*?(?=<)', html, re.IGNORECASE | re.DOTALL)
    if m:
        raw = re.sub(r'<[^>]+>', ' ', html[m.start():])
        raw = re.sub(r'\s+', ' ', raw).strip()
        if len(raw) > 200:
            return raw

    # Strategy 2: all <p> tags
    paras = re.findall(r'<p[^>]*>(.*?)</p>', html, re.IGNORECASE | re.DOTALL)
    if paras:
        raw = ' '.join(re.sub(r'<[^>]+>', ' ', p) for p in paras)
        raw = re.sub(r'\s+', ' ', raw).strip()
        if len(raw) > 200:
            return raw

    # Strategy 3: strip all tags
    raw = re.sub(r'<[^>]+>', ' ', html)
    raw = re.sub(r'\s+', ' ', raw).strip()
    return raw if len(raw) > 200 else None


def scrape_statement(date_str: str) -> dict | None:
    """Fetch a single FOMC statement; return dict or None if not found."""
    year = date_str[:4]
    for pattern in URL_PATTERNS:
        url = pattern.format(date=date_str, year=year)
        html = _fetch_url(url)
        if html is None:
            continue
        text = _extract_text(html)
        if text:
            return {
                'date':       f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}',
                'source':     url,
                'word_count': len(text.split()),
                'text':       text,
            }
    return None


def scrape_all(start_year: int = 1994, end_year: int = 2010,
               output_dir: Path = STATEMENTS_JSON_DIR,
               skip_existing: bool = True):
    output_dir.mkdir(parents=True, exist_ok=True)
    dates = [d for d in FOMC_DATES_1994_2010
             if start_year <= int(d[:4]) <= end_year]

    print(f"Scraping {len(dates)} FOMC statements ({start_year}–{end_year}) ...")
    ok = skipped = failed = 0

    for date_str in dates:
        out_json = output_dir / f'statement_{date_str}.json'
        if skip_existing and out_json.exists():
            skipped += 1
            continue

        record = scrape_statement(date_str)
        if record:
            out_json.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
            ok += 1
            print(f'  ✓  {date_str}  ({record["word_count"]} words)')
        else:
            failed += 1
            print(f'  ✗  {date_str}  — not found')

        time.sleep(1.0)  # polite crawl rate

    print(f'\nDone: {ok} saved, {skipped} skipped, {failed} failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scrape pre-2011 FOMC statements')
    parser.add_argument('--start', type=int, default=1994)
    parser.add_argument('--end',   type=int, default=2010)
    parser.add_argument('--no-skip', action='store_true',
                        help='Re-scrape even if file already exists')
    args = parser.parse_args()
    scrape_all(args.start, args.end, skip_existing=not args.no_skip)
