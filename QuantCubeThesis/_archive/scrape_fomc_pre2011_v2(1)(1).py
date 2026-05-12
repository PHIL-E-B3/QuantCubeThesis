"""
scrape_fomc_pre2011.py  (v2 — fixed URL patterns + text extraction)
"""

import os, re, json, time, requests
from bs4 import BeautifulSoup
from pathlib import Path

BASE_DRIVE = '/content/drive/MyDrive/HEC Thesis'
TXT_DIR    = os.path.join(BASE_DRIVE, 'Text Data', 'fomc_statements')
JSON_DIR   = os.path.join(BASE_DRIVE, 'Text Data', 'structured_json_statements')
FED_BASE   = 'https://www.federalreserve.gov'

Path(TXT_DIR).mkdir(parents=True, exist_ok=True)
Path(JSON_DIR).mkdir(parents=True, exist_ok=True)

HEADERS = {'User-Agent': 'Mozilla/5.0 Chrome/120.0.0.0'}

# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def get_page(url, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                return r.text
            elif r.status_code == 404:
                return None
            time.sleep(2)
        except Exception as e:
            print(f"    Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return None


def extract_text(html):
    """
    Extract clean statement body text.
    Strategy: find the substantive paragraph content, strip boilerplate.
    """
    if not html:
        return None

    soup = BeautifulSoup(html, 'html.parser')

    # Remove navigation, scripts, styles, headers, footers entirely
    for tag in soup.find_all(['script', 'style', 'nav', 'footer',
                               'header', 'iframe', 'form']):
        tag.decompose()

    # Try content divs in order of preference
    content = (
        soup.find('div', class_='col-xs-12 col-sm-8 col-md-8') or
        soup.find('div', id='content')                          or
        soup.find('div', class_='article')                      or
        soup.find('div', id='leftText')                         or
        soup.find('td', class_='content')                       or
        soup.find('body')
    )

    if not content:
        return None

    text = re.sub(r'\s+', ' ', content.get_text(separator=' ')).strip()

    # ── Strategy 1: extract between release marker and voting section end ────
    # This captures the full statement including the voting record
    m = re.search(
        r'For (?:immediate )?release\s*(.+?)(?=\s*(?:Last\s+[Uu]pdate|'
        r'Return to top|Accessibility|Home\s*\|\s*News|\Z))',
        text, re.DOTALL | re.IGNORECASE
    )
    if m:
        body = m.group(1).strip()
        if len(body) > 80:
            return body

    # ── Strategy 2: find all <p> tags and join them ───────────────────────────
    # More reliable for legacy pages where regex fails
    paragraphs = content.find_all('p')
    if paragraphs:
        para_texts = []
        for p in paragraphs:
            pt = p.get_text(separator=' ', strip=True)
            pt = re.sub(r'\s+', ' ', pt)
            # Skip boilerplate navigation paragraphs
            if any(skip in pt.lower() for skip in [
                'home |', 'return to top', 'accessibility',
                'last update', 'contact us', 'site map',
                'about the fed', 'monetary policy'
            ]):
                continue
            if len(pt) > 20:
                para_texts.append(pt)
        if para_texts:
            joined = ' '.join(para_texts)
            if len(joined) > 80:
                return joined

    # ── Strategy 3: full text fallback ───────────────────────────────────────
    if len(text) > 80:
        # Strip common footer boilerplate from end
        text = re.sub(
            r'\s*(Home\s*\|.*|Return to top.*|Last [Uu]pdate.*|'
            r'Accessibility.*|Contact Us.*)$',
            '', text, flags=re.DOTALL | re.IGNORECASE
        ).strip()
        return text if len(text) > 80 else None

    return None


def already_exists(date_str):
    # Only protect 2011+ files — re-scrape pre-2011 to fix truncation issues
    if date_str >= '20110101':
        return os.path.exists(os.path.join(JSON_DIR, f"statement_{date_str}.json"))
    return False


def save_statement(date_str, text):
    json_path = os.path.join(JSON_DIR, f"statement_{date_str}.json")
    txt_path  = os.path.join(TXT_DIR,  f"statement_{date_str}.txt")
    record = {
        "source":     f"statement_{date_str}.txt",
        "date":       f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
        "date_raw":   date_str,
        "type":       "fomc_statement",
        "text":       text,
        "word_count": len(text.split()),
    }
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
        f.flush(); os.fsync(f.fileno())
    return record['word_count']


def find_urls_for_year(year):
    """
    Find all statement URLs for a given year from the Fed historical page.
    Handles all URL patterns used across different eras.
    """
    urls = []
    html = get_page(f'{FED_BASE}/monetarypolicy/fomchistorical{year}.htm')
    if not html:
        return urls

    soup = BeautifulSoup(html, 'html.parser')

    for a in soup.find_all('a', href=True):
        href = a['href']
        date_str = None
        url      = None

        # Pattern 1: /newsevents/pressreleases/monetaryYYYYMMDDa.htm  (2011+)
        m = re.search(r'/newsevents/pressreleases/monetary(\d{8})a\.htm', href, re.I)
        if m:
            date_str, url = m.group(1), FED_BASE + href
            
        # Pattern 2: /newsevents/press/monetary/YYYYMMDDa.htm  (2006-2010)
        if not date_str:
            m = re.search(r'/newsevents/press/monetary/(\d{8})a\.htm', href, re.I)
            if m:
                date_str, url = m.group(1), FED_BASE + href

        # Pattern 3: /boarddocs/press/monetary/YYYY/YYYYMMDD/  (2002-2005)
        if not date_str:
            m = re.search(r'/boarddocs/press/monetary/\d{4}/(\d{8})/', href, re.I)
            if m:
                date_str, url = m.group(1), FED_BASE + href

        # Pattern 4: /boarddocs/press/general/YYYY/YYYYMMDD/  (1997-2001)
        if not date_str:
            m = re.search(r'/boarddocs/press/general/\d{4}/(\d{8})/', href, re.I)
            if m:
                date_str, url = m.group(1), FED_BASE + href

        # Pattern 5: /fomc/YYYYMMDD  (very old)
        if not date_str:
            m = re.search(r'/fomc/(\d{8})', href, re.I)
            if m:
                date_str, url = m.group(1), FED_BASE + href

        # Only keep links labelled "Statement" or matching known patterns
        link_text = a.get_text(strip=True).lower()
        if date_str and url:
            if ('statement' in link_text or
                href.endswith('a.htm') or
                href.endswith('default.htm') or
                'monetary' in href.lower()):
                urls.append((date_str, url))

    # Deduplicate
    seen, deduped = set(), []
    for ds, url in sorted(urls):
        if ds not in seen:
            seen.add(ds)
            deduped.append((ds, url))
    return deduped


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

print("Pre-2011 FOMC Statement Scraper (1994–2010)")
print(f"JSON → {JSON_DIR}\n")

scraped, skipped, failed = 0, 0, 0
failed_list = []

for year in range(1994, 2011):
    print(f"\n── {year} ──────────────────────────────")
    year_urls = find_urls_for_year(year)
    print(f"  Found {len(year_urls)} URLs")

    if not year_urls:
        print(f"  ⚠️  No URLs — check manually")
        time.sleep(1)
        continue

    for date_str, url in year_urls:
        if already_exists(date_str):
            print(f"  ⏭  {date_str} exists — skip")
            skipped += 1
            continue

        html = get_page(url)
        text = extract_text(html) if html else None

        if text and len(text.split()) >= 20:
            wc = save_statement(date_str, text)
            if wc:
                print(f"  ✓  {date_str}  [{wc} words]")
                scraped += 1
            else:
                print(f"  ⏭  {date_str} exists — skip")
                skipped += 1
        else:
            print(f"  ✗  {date_str}  FAILED ({len(text.split()) if text else 0} words)  {url}")
            failed += 1
            failed_list.append((date_str, url))

        time.sleep(0.4)
    time.sleep(1.0)

print(f"\n{'═'*55}")
print(f"  ✓ Scraped:  {scraped}")
print(f"  ⏭ Skipped:  {skipped}")
print(f"  ✗ Failed:   {failed}")
print(f"\n  Total JSON files now: {len(os.listdir(JSON_DIR))}")

if failed_list:
    print(f"\n  Failed (check manually):")
    for ds, url in failed_list:
        print(f"    {ds}  {url}")
