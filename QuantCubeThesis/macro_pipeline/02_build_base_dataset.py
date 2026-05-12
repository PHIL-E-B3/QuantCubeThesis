"""
02_build_base_dataset.py
------------------------
Builds the core FOMC macro panel — one row per event date (blackout_date,
meeting_date, minutes_date) — with all macro data reflected as-of that date.

Point-in-time safety: every value in a row reflects only information that was
publicly available on or before that row's date.  Vintage matrices ensure
revised macro series use the vintage actually available at that time.

Outputs:
    data/macro_inputs/Master_Macro.csv   — final stacked panel

Usage:
    python macro_pipeline/02_build_base_dataset.py
    python macro_pipeline/02_build_base_dataset.py --rebuild-calendar
    python macro_pipeline/02_build_base_dataset.py --clear-cache
"""

import argparse
import os
import re
import time
import random
import requests
import warnings
import numpy  as np
import pandas as pd
from io       import BytesIO
from pathlib  import Path
from fredapi  import Fred

warnings.filterwarnings('ignore')

from config import (
    FRED_API_KEY, VINTAGE_START,
    STATEMENTS_JSON_DIR, SPEECHES_JSON_DIR, SPEECH_SUBDIRS,
    BB_FUTURES_FILE, FOMC_CALENDAR_CSV, MASTER_MACRO_CSV,
    CACHE_DIR,
)

CACHE_DIR.mkdir(parents=True, exist_ok=True)
fred = Fred(api_key=FRED_API_KEY)

INFLATION_VARS = {'cpi', 'core_cpi', 'pce', 'core_pce'}

# ── Series definitions ────────────────────────────────────────────────────────

VINTAGE_SERIES = {
    'cpi':               'CPIAUCSL',
    'core_cpi':          'CPILFESL',
    'pce':               'PCEPI',
    'core_pce':          'PCEPILFE',
    'unemployment_rate': 'UNRATE',
    'nonfarm_payroll':   'PAYEMS',
    'gdp':               'GDP',
    'gdp_deflator':      'GDPDEF',
    'nat_unemp_rate':    'NROU',
}

MARKET_SERIES = {
    'fed_funds_rate':      'FEDFUNDS',
    'yield_3mo':           'DTB3',
    'yield_6mo':           'DTB6',
    'yield_2yr':           'DGS2',
    'yield_5yr':           'DGS5',
    'yield_10yr':          'DGS10',
    'vix':                 'VIXCLS',
    'breakeven_10yr':      'T10YIE',
    'term_spread_10_2':    'T10Y2Y',
    'real_rate_5yr':       'DFII5',
    'fed_target_midpoint': 'FEDTARMDLR',
}

SPF_URLS = {
    'spf_gdp':   ('https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/'
                  'survey-of-professional-forecasters/data-files/files/median_rgdp_level.xlsx'),
    'spf_unemp': ('https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/'
                  'survey-of-professional-forecasters/data-files/files/median_unemp_level.xlsx'),
    'spf_cpi':   ('https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/'
                  'survey-of-professional-forecasters/data-files/files/median_cpi_level.xlsx'),
}


# ── FOMC Calendar ─────────────────────────────────────────────────────────────

def build_fomc_calendar(minutes_lag_days: int = 21) -> pd.DataFrame:
    """
    Build FOMC calendar from statement JSON filenames and speech JSON filenames.
    Returns DataFrame with meeting_date, minutes_date, blackout_date columns.
    """
    # Meeting dates from statement filenames (statement_YYYYMMDD.json)
    meeting_dates = sorted([
        pd.to_datetime(m.group(1), format='%Y%m%d')
        for fname in os.listdir(STATEMENTS_JSON_DIR)
        for m in [re.search(r'statement_(\d{8})', fname)]
        if m
    ])

    # Speech dates from JSON files (date field inside each JSON)
    import json
    speech_dates = []
    for subdir in SPEECH_SUBDIRS:
        folder = SPEECHES_JSON_DIR / subdir
        if not folder.exists():
            continue
        for fname in os.listdir(folder):
            if not fname.endswith('.json'):
                continue
            try:
                with open(folder / fname, encoding='utf-8') as f:
                    rec = json.load(f)
                dt = pd.to_datetime(rec.get('date', ''), errors='coerce')
                if pd.notna(dt):
                    speech_dates.append(dt)
            except Exception:
                pass

    all_speech_dates = sorted(set(speech_dates))

    rows = []
    for meeting_dt in meeting_dates:
        minutes_dt  = meeting_dt + pd.Timedelta(days=minutes_lag_days)
        speeches_before = [d for d in all_speech_dates if d < meeting_dt]
        blackout_dt = speeches_before[-1] if speeches_before else pd.NaT
        lag = (meeting_dt - blackout_dt).days if pd.notna(blackout_dt) else None
        rows.append({
            'meeting_date':      meeting_dt,
            'minutes_date':      minutes_dt,
            'blackout_date':     blackout_dt,
            'blackout_lag_days': lag,
        })

    df = pd.DataFrame(rows)
    print(f'  Calendar: {len(df)} meetings  '
          f'({df["meeting_date"].min().date()} – {df["meeting_date"].max().date()})')
    return df


def load_or_build_calendar(rebuild: bool = False) -> pd.DataFrame:
    if FOMC_CALENDAR_CSV.exists() and not rebuild:
        df = pd.read_csv(FOMC_CALENDAR_CSV,
                         parse_dates=['meeting_date','minutes_date','blackout_date'])
        df = df[[c for c in df.columns if not c.startswith('Unnamed')]]
        print(f'  Loaded existing calendar ({len(df)} rows)')
        return df
    print('  Building FOMC calendar from statement/speech JSON files ...')
    df = build_fomc_calendar()
    df.to_csv(FOMC_CALENDAR_CSV, index=False)
    return df


# ── FRED helpers ──────────────────────────────────────────────────────────────

def _cache_path(name: str) -> Path:
    return CACHE_DIR / f'{name}.parquet'


def _fred_retry(fn, *args, max_retries=8, base_delay=2.0, **kwargs):
    """Call fn(*args, **kwargs) with exponential back-off on rate-limit errors."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            err = str(e).lower()
            if 'mismatched tag' in err or 'parseerror' in err:
                return None
            if 'too many requests' in err or '429' in err or 'forbidden' in err or '403' in err:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                delay = max(delay, 30) if ('forbidden' in err or '403' in err) and attempt < 2 else delay
                print(f'    [retry {attempt+1}/{max_retries}] rate-limited — waiting {delay:.0f}s',
                      flush=True)
                time.sleep(delay)
            else:
                print(f'    Warning: {e}')
                return None
    return None


def build_vintage_matrix(series_id: str, name: str,
                         fomc_calendar: pd.DataFrame) -> pd.DataFrame:
    """
    Pull targeted FRED vintages needed for each blackout date.
    Only fetches the vintage immediately on-or-before each blackout date
    (plus the next one as a safety buffer).  Fully cached.
    """
    final_path   = _cache_path(f'vintage_{name}')
    partial_path = _cache_path(f'vintage_{name}_partial')

    if final_path.exists():
        print(f'  [cache hit]  {name}')
        return pd.read_parquet(final_path)

    raw_dates = _fred_retry(fred.get_series_vintage_dates, series_id) or []
    all_vts   = pd.to_datetime([
        d for d in raw_dates if pd.Timestamp(d) >= pd.Timestamp(VINTAGE_START)
    ])
    if len(all_vts) == 0:
        print(f'  ERROR: no vintage dates for {name}')
        return pd.DataFrame()

    blackout_dates  = pd.to_datetime(fomc_calendar['blackout_date'].dropna().unique())
    needed_vintages = set()
    for bd in blackout_dates:
        before = all_vts[all_vts <= bd]
        after  = all_vts[all_vts  > bd]
        if not before.empty: needed_vintages.add(before[-1].strftime('%Y-%m-%d'))
        if not after.empty:  needed_vintages.add(after[0].strftime('%Y-%m-%d'))

    vintage_dates = sorted(needed_vintages)
    print(f'  [fetching]   {name} — {len(vintage_dates)} vintages', flush=True)

    frames = {}
    if partial_path.exists():
        partial = pd.read_parquet(partial_path)
        frames  = {col.strftime('%Y-%m-%d'): partial[col] for col in partial.columns}
        remaining = [v for v in vintage_dates if v not in frames]
        print(f'    Resuming: {len(frames)} done, {len(remaining)} left', flush=True)
    else:
        remaining = vintage_dates

    for i, vdate in enumerate(remaining):
        result = _fred_retry(fred.get_series, series_id,
                             realtime_start=vdate, realtime_end=vdate)
        if result is not None:
            result.name = pd.Timestamp(vdate)
            frames[vdate] = result
        if (i + 1) % 20 == 0:
            print(f'    ... {i+1}/{len(remaining)}', flush=True)
            if frames:
                tmp = pd.concat(frames.values(), axis=1)
                tmp.columns = pd.to_datetime(list(frames.keys()))
                tmp.to_parquet(partial_path)
        time.sleep(0.6)

    if not frames:
        print(f'  ERROR: no data for {name}')
        return pd.DataFrame()

    matrix = pd.concat(frames.values(), axis=1)
    matrix.columns = pd.to_datetime(list(frames.keys()))
    matrix = matrix.sort_index(axis=1)
    matrix.to_parquet(final_path)
    if partial_path.exists():
        partial_path.unlink()
    print(f'  ✓  {name}  ({matrix.shape[1]} vintages × {matrix.shape[0]} obs)')
    return matrix


def compute_yoy_vintage_matrix(level_matrix: pd.DataFrame, name: str) -> pd.DataFrame:
    """YoY (%) derived vintage-consistently from a levels vintage matrix."""
    yoy_path = _cache_path(f'vintage_{name}_yoy')
    if yoy_path.exists():
        return pd.read_parquet(yoy_path)
    if level_matrix.empty:
        return pd.DataFrame()
    yoy = level_matrix.pct_change(periods=12) * 100
    yoy.to_parquet(yoy_path)
    return yoy


def vintage_as_of(matrix: pd.DataFrame, date) -> tuple:
    """Latest non-NaN vintage observation on or before date."""
    if matrix.empty:
        return None, None
    bd   = pd.Timestamp(date)
    cols = matrix.columns[matrix.columns <= bd]
    if cols.empty:
        return None, None
    snap = matrix[cols].ffill(axis=1).iloc[:, -1]
    snap = snap[snap.index <= bd].dropna()
    if snap.empty:
        return None, None
    return snap.iloc[-1], snap.index[-1]


def fetch_market_series(series_id: str, name: str) -> pd.Series:
    path = _cache_path(f'market_{name}')
    if path.exists():
        return pd.read_parquet(path).squeeze()
    print(f'  [fetching market]  {name}')
    s = fred.get_series(series_id)
    s.index = pd.to_datetime(s.index)
    s.name  = name
    s.to_frame().to_parquet(path)
    return s


def market_as_of(series: pd.Series, date) -> tuple:
    series.index = pd.to_datetime(series.index)
    sliced = series[series.index <= pd.Timestamp(date)].dropna()
    return (sliced.iloc[-1], sliced.index[-1]) if not sliced.empty else (None, None)


def fetch_ads() -> pd.Series:
    path = _cache_path('ads_index')
    if path.exists():
        print('  [cache hit]  ads_index')
        return pd.read_parquet(path).squeeze()
    print('  [fetching]   ADS Business Conditions Index ...')
    headers = {'User-Agent': 'Mozilla/5.0 Chrome/120.0.0.0'}
    for url in [
        'https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/ads/ads_index_most_current_vintage.xlsx',
        'https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/ads/ads.xlsx',
    ]:
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 200 and r.content[:2] == b'PK':
            try:
                df = pd.read_excel(BytesIO(r.content), index_col=0,
                                   parse_dates=True, engine='openpyxl')
                s  = df.iloc[:, 0].rename('ads')
                s.to_frame().to_parquet(path)
                return s
            except Exception as e:
                print(f'  Warning: could not parse ADS: {e}')
    raise ValueError('Failed to fetch ADS index')


def fetch_ebp() -> pd.Series:
    path = _cache_path('excess_bond_premium')
    if path.exists():
        print('  [cache hit]  excess_bond_premium')
        return pd.read_parquet(path).squeeze()
    print('  [fetching]   Excess Bond Premium ...')
    url = 'https://www.federalreserve.gov/econresdata/notes/feds-notes/2016/files/ebp_csv.csv'
    r   = requests.get(url, timeout=30,
                       headers={'User-Agent': 'Mozilla/5.0 Chrome/120.0.0.0'})
    if r.status_code != 200:
        raise ValueError(f'Failed to fetch EBP (HTTP {r.status_code})')
    df  = pd.read_csv(BytesIO(r.content), parse_dates=['date'], index_col='date')
    df.to_parquet(path)
    return df['ebp']


def _series_as_of(series: pd.Series, date) -> tuple:
    """Generic as-of lookup handling both Series and DataFrame cache types."""
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    series.index = pd.to_datetime(series.index)
    sliced = series[series.index <= pd.Timestamp(date)].dropna()
    return (sliced.iloc[-1], sliced.index[-1]) if not sliced.empty else (None, None)


def fetch_spf(name: str, url: str) -> pd.DataFrame:
    path = _cache_path(f'spf_{name}')
    if path.exists():
        print(f'  [cache hit]  {name}')
        return pd.read_parquet(path)
    print(f'  [fetching SPF]  {name} ...')
    r  = requests.get(url, timeout=30)
    df = pd.read_excel(BytesIO(r.content), engine='openpyxl')
    df.columns = df.columns.str.strip().str.upper()
    df['survey_date'] = (
        pd.to_datetime(df['YEAR'].astype(int).astype(str) + 'Q' +
                       df['QUARTER'].astype(int).astype(str))
        + pd.offsets.QuarterEnd(0)
    )
    df = df.sort_values('survey_date').set_index('survey_date')
    df.to_parquet(path)
    return df


def spf_as_of(spf_df: pd.DataFrame, date) -> tuple:
    available = spf_df[spf_df.index <= pd.Timestamp(date)]
    return (available.iloc[-1], available.index[-1]) if not available.empty else (None, None)


# ── Implied FFR from Bloomberg futures ───────────────────────────────────────

def build_implied_ffr(fomc_calendar: pd.DataFrame) -> pd.DataFrame:
    """
    Load Bloomberg Fed Funds Futures and compute implied FFR for each
    FOMC event date (blackout_date prices THIS meeting; meeting_date and
    minutes_date price the NEXT meeting).
    """
    print('  Loading Bloomberg futures ...')
    bb = pd.read_excel(BB_FUTURES_FILE, header=0)
    # Columns alternate: date_ffr/target_ffr, date_effr/effr, date_ff1/FF1, ...
    # Standardise by pairing them up
    col_map = {
        0: 'date_ffr',  1: 'target_ffr',
        2: 'date_effr', 3: 'effr',
        4: 'date_ff1',  5: 'FF1',
        6: 'date_ff2',  7: 'FF2',
        8: 'date_ff3',  9: 'FF3',
    }
    bb = bb.rename(columns={bb.columns[k]: v for k, v in col_map.items()
                             if k < len(bb.columns)})

    def parse_col(date_col, val_col):
        df = bb[[date_col, val_col]].dropna()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df[val_col]  = pd.to_numeric(df[val_col], errors='coerce')
        return df.dropna().sort_values(date_col).rename(columns={date_col: 'date'})

    df_effr = parse_col('date_effr', 'effr') if 'date_effr' in bb.columns else pd.DataFrame()
    df_ff1  = parse_col('date_ff1',  'FF1')  if 'date_ff1'  in bb.columns else pd.DataFrame()
    df_ff2  = parse_col('date_ff2',  'FF2')  if 'date_ff2'  in bb.columns else pd.DataFrame()

    # Build one record per FOMC event
    placeholder = fomc_calendar['meeting_date'].max() + pd.DateOffset(days=45)
    fomc_ext = pd.concat([
        fomc_calendar,
        pd.DataFrame([{'meeting_date': placeholder}])
    ]).sort_values('meeting_date').reset_index(drop=True)
    fomc_ext['next_meeting_date'] = fomc_ext['meeting_date'].shift(-1)

    records = []
    for _, row in fomc_ext.iterrows():
        if pd.isna(row.get('next_meeting_date')):
            continue
        for col in ['meeting_date', 'minutes_date']:
            date = row.get(col)
            if pd.notna(date):
                records.append({'rep_date': date, 'next_meeting': row['next_meeting_date']})
        if pd.notna(row.get('blackout_date')):
            records.append({'rep_date': row['blackout_date'], 'next_meeting': row['meeting_date']})

    rep_df = (pd.DataFrame(records)
              .drop_duplicates('rep_date')
              .sort_values('rep_date')
              .reset_index(drop=True))

    def nearest_before(df_series: pd.DataFrame, date, val_col: str):
        if df_series.empty:
            return np.nan
        sub = df_series[df_series['date'] <= pd.Timestamp(date)]
        return sub[val_col].iloc[-1] if not sub.empty else np.nan

    rep_df['effective_rate'] = rep_df['rep_date'].apply(
        lambda d: nearest_before(df_effr, d, 'effr'))
    rep_df['ff1'] = rep_df['rep_date'].apply(
        lambda d: nearest_before(df_ff1, d, 'FF1'))
    rep_df['ff2'] = rep_df['rep_date'].apply(
        lambda d: nearest_before(df_ff2, d, 'FF2'))

    # Implied FFR: price the next meeting using front-month contract
    rep_df['implied_ffr'] = rep_df['ff1'].combine_first(rep_df['ff2'])
    return rep_df[['rep_date', 'effective_rate', 'implied_ffr']].rename(
        columns={'rep_date': 'date'})


# ── Main assembly ─────────────────────────────────────────────────────────────

def assemble(fomc_calendar: pd.DataFrame,
             vintage_matrices: dict,
             yoy_matrices: dict,
             market_cache: dict,
             ads: pd.Series,
             ebp: pd.Series,
             spf_frames: dict) -> pd.DataFrame:
    """Stack three event types per meeting and fill all macro columns."""
    events = (
        pd.melt(fomc_calendar,
                value_vars=['blackout_date','meeting_date','minutes_date'],
                var_name='event_type', value_name='date')
        .dropna(subset=['date'])
        .sort_values('date')
        .reset_index(drop=True)
    )
    SPF_META = {'YEAR','QUARTER','SURVEY_DATE'}
    rows = []
    n = len(events)
    for idx, ev in events.iterrows():
        date, event_type = ev['date'], ev['event_type']
        row = {'date': date, 'event_type': event_type}

        for name, matrix in vintage_matrices.items():
            val, _ = vintage_as_of(matrix, date)
            row[name] = val
            if name in INFLATION_VARS:
                yoy_val, _ = vintage_as_of(yoy_matrices[name], date)
                row[f'{name}_yoy'] = yoy_val

        for name, series in market_cache.items():
            val, _ = market_as_of(series, date)
            row[name] = val

        row['ads'],            _ = _series_as_of(ads, date)
        row['excess_bond_prem'], _ = _series_as_of(ebp, date)

        for spf_name, spf_df in spf_frames.items():
            spf_row, _ = spf_as_of(spf_df, date)
            if spf_row is not None:
                fcol = next((c for c in spf_df.columns if c.upper() not in SPF_META), None)
                row[f'{spf_name}_1q'] = spf_row[fcol] if fcol else None
            else:
                row[f'{spf_name}_1q'] = None

        unemp = row.get('unemployment_rate')
        nat_u = row.get('nat_unemp_rate')
        row['unemployment_gap']       = (unemp - nat_u) if unemp and nat_u else None
        row['inflation_dev_from_target'] = (row.get('core_pce_yoy', 0) or 0) - 2.0
        y10 = row.get('yield_10yr'); y2 = row.get('yield_2yr')
        row['yield_spread_10_2'] = (y10 - y2) if y10 and y2 else None

        rows.append(row)
        if (idx + 1) % 100 == 0:
            print(f'    ... {idx+1}/{n} events assembled', flush=True)

    return pd.DataFrame(rows).sort_values('date').reset_index(drop=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(rebuild_calendar: bool = False, clear_cache: bool = False):
    if clear_cache:
        import shutil
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        CACHE_DIR.mkdir(parents=True)
        print('Cache cleared.')

    print('\n' + '='*65)
    print('STEP 0  —  FOMC Calendar')
    print('='*65)
    fomc_calendar = load_or_build_calendar(rebuild=rebuild_calendar)

    print('\n' + '='*65)
    print('STEP 1/5  —  FRED vintage matrices')
    print('='*65)
    vintage_matrices = {
        name: build_vintage_matrix(sid, name, fomc_calendar)
        for name, sid in VINTAGE_SERIES.items()
    }

    print('\n' + '='*65)
    print('STEP 2/5  —  YoY inflation matrices')
    print('='*65)
    yoy_matrices = {
        name: compute_yoy_vintage_matrix(vintage_matrices[name], name)
        for name in INFLATION_VARS
    }

    print('\n' + '='*65)
    print('STEP 3/5  —  FRED market series')
    print('='*65)
    market_cache = {name: fetch_market_series(sid, name)
                    for name, sid in MARKET_SERIES.items()}

    print('\n' + '='*65)
    print('STEP 4/5  —  ADS index & Excess Bond Premium')
    print('='*65)
    ads = fetch_ads()
    ebp = fetch_ebp()

    print('\n' + '='*65)
    print('STEP 5/5  —  Survey of Professional Forecasters')
    print('='*65)
    spf_frames = {name: fetch_spf(name, url) for name, url in SPF_URLS.items()}

    print('\n' + '='*65)
    print('ASSEMBLING STACKED DATASET')
    print('='*65)
    dataset = assemble(fomc_calendar, vintage_matrices, yoy_matrices,
                       market_cache, ads, ebp, spf_frames)
    print(f'  Assembled: {dataset.shape[0]} rows × {dataset.shape[1]} columns')

    print('\n' + '='*65)
    print('APPENDING IMPLIED FFR (Bloomberg futures)')
    print('='*65)
    if BB_FUTURES_FILE.exists():
        implied = build_implied_ffr(fomc_calendar)
        dataset = dataset.merge(implied, on='date', how='left')
        print(f'  ✓  implied_ffr: {dataset["implied_ffr"].notna().sum()} non-NaN')
        print(f'  ✓  effective_rate: {dataset["effective_rate"].notna().sum()} non-NaN')
    else:
        print(f'  ⚠️  Bloomberg file not found: {BB_FUTURES_FILE}')
        print(f'     Skipping implied_ffr column — add file to data/macro_inputs/')

    MASTER_MACRO_CSV.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(MASTER_MACRO_CSV, index=False)
    print(f'\n✅  Saved → {MASTER_MACRO_CSV}')
    print(f'   {dataset.shape[0]} rows × {dataset.shape[1]} columns')
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build FOMC base macro dataset')
    parser.add_argument('--rebuild-calendar', action='store_true',
                        help='Rebuild FOMC calendar from statement/speech JSON files')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Delete FRED cache and re-fetch everything')
    args = parser.parse_args()
    main(rebuild_calendar=args.rebuild_calendar, clear_cache=args.clear_cache)
