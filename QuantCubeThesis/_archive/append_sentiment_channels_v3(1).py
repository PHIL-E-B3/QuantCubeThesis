"""
append_sentiment_channels.py  (v2 — full date range 1999–2025)
════════════════════════════════════════════════════════════════════════════════
Appends channel regressor columns to Master_Macro.csv.
Works for the full extended sample including pre-2011 meeting dates.

CHANGES FROM v1
─────────────────────────────────────────────────────────────────────────────
- FRED series fetched from 1990-01-01 (not 2010-01-01)
- Statement novelty uses local JSON files (already scraped) instead of
  re-scraping the Fed website
- Dissent data: no longer filtered to post-2011 — TW dataset covers 1936+
- JK shocks: no longer filtered to post-2010
- MASTER_CSV = OUTPUT_CSV (reads and overwrites same file)
"""

import os
import re
import json
import time
import warnings
import requests
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

BASE_DIR      = '/content/drive/MyDrive/HEC Thesis'
MASTER_CSV    = f'{BASE_DIR}/Data/Master_Macro.csv'
OUTPUT_CSV    = f'{BASE_DIR}/Data/Master_Macro.csv'   # overwrite in place
FRED_API_KEY  = 'c5b5ec287d025fecd23b73051ee03c84'
JK_SHOCKS_CSV = f'{BASE_DIR}/Data/fomc_surprises_jk.csv.xlsx'
TW_FILE       = f'{BASE_DIR}/Data/FOMC_Dissents_Data.xlsx'
JSON_STMTS    = f'{BASE_DIR}/Text Data/structured_json_statements'

# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def fred_series(series_id, api_key, start='1990-01-01'):
    url = (f'https://api.stlouisfed.org/fred/series/observations'
           f'?series_id={series_id}&api_key={api_key}'
           f'&observation_start={start}&file_type=json')
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()['observations']
    s = pd.Series(
        {pd.Timestamp(d['date']): float(d['value'])
         for d in data if d['value'] != '.'},
        name=series_id
    )
    return s.sort_index()


def merge_asof_backward(df_master, series, col_name):
    s_df = series.reset_index()
    s_df.columns = ['date', col_name]
    s_df['date'] = pd.to_datetime(s_df['date'])
    tmp = pd.DataFrame({'date': pd.to_datetime(df_master['date'])})
    tmp = tmp.sort_values('date').reset_index()
    merged = pd.merge_asof(tmp, s_df.sort_values('date'),
                           on='date', direction='backward')
    merged = merged.set_index('index').sort_index()
    return merged[col_name]


def print_section(title):
    print(f'\n{"═"*70}\n  {title}\n{"═"*70}')


# ════════════════════════════════════════════════════════════════════════════
# LOAD MASTER
# ════════════════════════════════════════════════════════════════════════════

print_section('Loading Master_Macro.csv')
df = pd.read_csv(MASTER_CSV, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
print(f'  Loaded: {len(df)} rows × {len(df.columns)} columns')
print(f'  Date range: {df["date"].min().date()} → {df["date"].max().date()}')
print(f'  Event types: {df["event_type"].value_counts().to_dict()}')

is_meeting = df['event_type'] == 'meeting_date'

# Drop old channel columns so we rebuild them cleanly
OLD_CHANNEL_COLS = [
    'ffr_path_change',
    'michigan_1yr_inf_exp', 'inf_exp_gap', 'inf_exp_5y5y',
    'nfci', 'sloos_ci_tightening',
    'jk_mp1', 'jk_mp_shock', 'jk_info_shock',
    'stmt_novelty_jaccard', 'stmt_novelty_cosine',
    'fomc_dissent_count', 'fomc_gov_dissent',
    'fomc_pres_dissent', 'fomc_any_dissent',
]
df = df.drop(columns=[c for c in OLD_CHANNEL_COLS if c in df.columns])
print(f'  Dropped old channel columns, rebuilding from scratch.')

# ════════════════════════════════════════════════════════════════════════════
# 1. FORWARD GUIDANCE SHIFT
# ════════════════════════════════════════════════════════════════════════════

print_section('1. Forward Guidance Shift')

meeting_implied = df.loc[is_meeting, ['date', 'implied_ffr']].copy()
meeting_implied = meeting_implied.sort_values('date')
meeting_implied['ffr_path_change'] = meeting_implied['implied_ffr'].diff()

df = df.merge(meeting_implied[['date', 'ffr_path_change']], on='date', how='left')
df['ffr_path_change'] = df['ffr_path_change'].ffill()
print(f'  ✓  ffr_path_change: {df["ffr_path_change"].notna().sum()} non-NaN rows')

# ════════════════════════════════════════════════════════════════════════════
# 2. INFLATION EXPECTATIONS
# ════════════════════════════════════════════════════════════════════════════

print_section('2. Inflation Expectations')

try:
    mich = fred_series('MICH', FRED_API_KEY)
    df['michigan_1yr_inf_exp'] = merge_asof_backward(df, mich, 'michigan_1yr_inf_exp')
    df['inf_exp_gap']          = df['michigan_1yr_inf_exp'] - 2.0
    print(f'  ✓  michigan_1yr_inf_exp: {df["michigan_1yr_inf_exp"].notna().sum()} non-NaN')
    print(f'  ✓  inf_exp_gap:          {df["inf_exp_gap"].notna().sum()} non-NaN')
except Exception as e:
    print(f'  ✗  MICH fetch failed: {e}')
    df['michigan_1yr_inf_exp'] = np.nan
    df['inf_exp_gap']          = np.nan

try:
    t5yifr = fred_series('T5YIFR', FRED_API_KEY)
    df['inf_exp_5y5y'] = merge_asof_backward(df, t5yifr, 'inf_exp_5y5y')
    print(f'  ✓  inf_exp_5y5y: {df["inf_exp_5y5y"].notna().sum()} non-NaN')
except Exception as e:
    print(f'  ✗  T5YIFR fetch failed: {e}')
    df['inf_exp_5y5y'] = np.nan

# ════════════════════════════════════════════════════════════════════════════
# 3. FINANCIAL CONDITIONS
# ════════════════════════════════════════════════════════════════════════════

print_section('3. Financial Conditions')

try:
    nfci = fred_series('NFCI', FRED_API_KEY)
    df['nfci'] = merge_asof_backward(df, nfci, 'nfci')
    print(f'  ✓  nfci: {df["nfci"].notna().sum()} non-NaN')
except Exception as e:
    print(f'  ✗  NFCI fetch failed: {e}')
    df['nfci'] = np.nan

try:
    sloos = fred_series('DRTSCILM', FRED_API_KEY)
    df['sloos_ci_tightening'] = merge_asof_backward(df, sloos, 'sloos_ci_tightening')
    print(f'  ✓  sloos_ci_tightening: {df["sloos_ci_tightening"].notna().sum()} non-NaN')
except Exception as e:
    print(f'  ✗  DRTSCILM fetch failed: {e}')
    df['sloos_ci_tightening'] = np.nan

# ════════════════════════════════════════════════════════════════════════════
# 4. JK SHOCKS (full date range — no start filter)
# ════════════════════════════════════════════════════════════════════════════

print_section('4. MP Surprise / Information Effect (Jarociński-Karadi)')

if os.path.exists(JK_SHOCKS_CSV):
    try:
        jk = pd.read_excel(JK_SHOCKS_CSV)
        date_col = [c for c in jk.columns if c.lower() in ('start', 'date')][0]
        jk['date'] = pd.to_datetime(jk[date_col]).dt.normalize()

        # ── No date filter — use full history ────────────────────────────────
        if 'description' in jk.columns:
            jk = jk[jk['description'].str.contains('Scheduled', case=False, na=False)]

        jk = jk.rename(columns={'MP1': 'jk_mp1'})
        jk['jk_mp1'] = pd.to_numeric(jk['jk_mp1'], errors='coerce')

        sp500_col = next(
            (c for c in jk.columns if c.upper() in ('SP500', 'SP500FUT')), None
        )
        if sp500_col:
            jk[sp500_col] = pd.to_numeric(jk[sp500_col], errors='coerce')
            clean  = jk[['jk_mp1', sp500_col]].dropna()
            ols_jk = sm.OLS(clean['jk_mp1'],
                            sm.add_constant(clean[sp500_col])).fit()
            jk.loc[clean.index, 'jk_mp_shock']  = ols_jk.resid.values
            jk.loc[clean.index, 'jk_info_shock'] = ols_jk.fittedvalues.values
            print(f'  ✓  JK decomposition: {len(clean)} meetings used')
            print(f'     R² of MP1 ~ SP500: {ols_jk.rsquared:.4f}')
        else:
            jk['jk_mp_shock']  = jk['jk_mp1']
            jk['jk_info_shock'] = np.nan
            print(f'  ⚠️  SP500 column not found — raw MP1 only')

        merge_cols = [c for c in ['date','jk_mp1','jk_mp_shock','jk_info_shock']
                      if c in jk.columns]
        df = df.merge(jk[merge_cols], on='date', how='left')
        print(f'  ✓  jk_mp1:        {df["jk_mp1"].notna().sum()} non-NaN rows')
        print(f'  ✓  jk_mp_shock:   {df["jk_mp_shock"].notna().sum()} non-NaN rows')
        print(f'  ✓  jk_info_shock: {df["jk_info_shock"].notna().sum()} non-NaN rows')
    except Exception as e:
        print(f'  ✗  JK shocks failed: {e}')
        df['jk_mp1'] = df['jk_mp_shock'] = df['jk_info_shock'] = np.nan
else:
    print(f'  ⚠️  JK file not found: {JK_SHOCKS_CSV}')
    df['jk_mp1'] = df['jk_mp_shock'] = df['jk_info_shock'] = np.nan

# ════════════════════════════════════════════════════════════════════════════
# 5. STATEMENT NOVELTY — uses local JSON files (no web scraping needed)
#    Reads all statement_YYYYMMDD.json from the structured_json_statements
#    folder (which already has 1994–2025 statements scraped earlier).
# ════════════════════════════════════════════════════════════════════════════

print_section('5. Statement Novelty (from local JSON files)')


def jaccard_similarity(text1, text2):
    set1 = set(re.findall(r'\b[a-z]+\b', text1.lower()))
    set2 = set(re.findall(r'\b[a-z]+\b', text2.lower()))
    if not set1 or not set2:
        return np.nan
    return len(set1 & set2) / len(set1 | set2)


try:
    if not os.path.exists(JSON_STMTS):
        raise FileNotFoundError(f'JSON folder not found: {JSON_STMTS}')

    # Load all statement JSON files
    stmt_texts = {}
    for fname in sorted(os.listdir(JSON_STMTS)):
        if not fname.endswith('.json') or not fname.startswith('statement_'):
            continue
        fpath = os.path.join(JSON_STMTS, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                rec = json.load(f)
            dt   = pd.Timestamp(rec['date'])
            text = rec.get('text', '')
            if text and len(text) > 50:
                stmt_texts[dt] = text
        except Exception:
            pass

    print(f'  Loaded {len(stmt_texts)} statements from JSON files')
    print(f'  Date range: {min(stmt_texts).date()} → {max(stmt_texts).date()}')

    stmt_dates = sorted(stmt_texts.keys())
    novelty_jaccard = {}
    novelty_cosine  = {}

    all_texts = [stmt_texts[d] for d in stmt_dates]
    tfidf     = TfidfVectorizer(stop_words='english', max_features=500)
    tfidf_mat = tfidf.fit_transform(all_texts)

    for i in range(1, len(stmt_dates)):
        dt_curr = stmt_dates[i]
        jac = jaccard_similarity(stmt_texts[dt_curr], stmt_texts[stmt_dates[i-1]])
        novelty_jaccard[dt_curr] = 1 - jac if pd.notna(jac) else np.nan
        cos = sk_cosine(tfidf_mat[i], tfidf_mat[i-1])[0, 0]
        novelty_cosine[dt_curr]  = 1 - cos

    nov_df = pd.DataFrame({
        'date':                 pd.Series(list(novelty_jaccard.keys())),
        'stmt_novelty_jaccard': pd.Series(list(novelty_jaccard.values())),
        'stmt_novelty_cosine':  pd.Series(list(novelty_cosine.values())),
    })
    nov_df['date'] = pd.to_datetime(nov_df['date'])

    df_meeting = df[is_meeting][['date']].copy().sort_values('date')
    nov_merged = pd.merge_asof(
        df_meeting,
        nov_df.sort_values('date'),
        on='date', direction='nearest', tolerance=pd.Timedelta('3d')
    )
    df = df.merge(nov_merged, on='date', how='left')
    df['stmt_novelty_jaccard'] = df['stmt_novelty_jaccard'].ffill()
    df['stmt_novelty_cosine']  = df['stmt_novelty_cosine'].ffill()

    print(f'  ✓  stmt_novelty_jaccard: {df["stmt_novelty_jaccard"].notna().sum()} non-NaN')
    print(f'  ✓  stmt_novelty_cosine:  {df["stmt_novelty_cosine"].notna().sum()} non-NaN')

except Exception as e:
    print(f'  ✗  Statement novelty failed: {e}')
    df['stmt_novelty_jaccard'] = np.nan
    df['stmt_novelty_cosine']  = np.nan

# ════════════════════════════════════════════════════════════════════════════
# 6. FOMC DISAGREEMENT (Thornton-Wheelock — full history, no date filter)
# ════════════════════════════════════════════════════════════════════════════

print_section('6. FOMC Internal Disagreement (Thornton-Wheelock)')

try:
    tw = pd.read_excel(TW_FILE, skiprows=3)
    tw.columns = tw.columns.str.strip()
    tw = tw.rename(columns={'FOMC Meeting': 'date'})
    tw['date'] = pd.to_datetime(tw['date']).dt.normalize()

    # ── No date filter — use full TW history ─────────────────────────────────
    tw = tw.rename(columns={
        'Votes Against Action':         'fomc_dissent_count',
        'Number Governors Dissenting':  'fomc_gov_dissent',
        'Number Presidents Dissenting': 'fomc_pres_dissent',
    })

    tw_cols = ['date','fomc_dissent_count','fomc_gov_dissent','fomc_pres_dissent']
    tw_cols = [c for c in tw_cols if c in tw.columns]

    # Use merge_asof with 2-day tolerance to handle any date discrepancies
    df_mtg = df[is_meeting][['date']].copy().sort_values('date')
    tw_merged = pd.merge_asof(
        df_mtg,
        tw[tw_cols].sort_values('date'),
        on='date', direction='nearest', tolerance=pd.Timedelta('2d')
    )
    df = df.merge(tw_merged, on='date', how='left')

    df['fomc_any_dissent'] = (df['fomc_dissent_count'] > 0).astype(float)
    df.loc[df['fomc_dissent_count'].isna(), 'fomc_any_dissent'] = np.nan

    # Forward-fill to minutes/blackout rows
    for c in ['fomc_dissent_count','fomc_gov_dissent','fomc_pres_dissent','fomc_any_dissent']:
        if c in df.columns:
            df[c] = df[c].ffill()

    print(f'  ✓  fomc_dissent_count: {df["fomc_dissent_count"].notna().sum()} non-NaN')
    dist = df.loc[is_meeting,'fomc_dissent_count'].value_counts().sort_index().to_dict()
    print(f'     Distribution: {dist}')
    print(f'     Meetings with dissent: {(df.loc[is_meeting,"fomc_dissent_count"]>0).sum()}')

except Exception as e:
    print(f'  ✗  Dissent failed: {e}')
    for c in ['fomc_dissent_count','fomc_gov_dissent','fomc_pres_dissent','fomc_any_dissent']:
        df[c] = np.nan

# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════

print_section('Summary')

new_cols = [
    'ffr_path_change',
    'michigan_1yr_inf_exp', 'inf_exp_5y5y', 'inf_exp_gap',
    'nfci', 'sloos_ci_tightening',
    'jk_mp1', 'jk_mp_shock', 'jk_info_shock',
    'stmt_novelty_jaccard', 'stmt_novelty_cosine',
    'fomc_dissent_count', 'fomc_gov_dissent', 'fomc_pres_dissent', 'fomc_any_dissent',
]

print(f'\n  {"Column":<30} {"Non-NaN":>8} {"Min":>10} {"Max":>10}')
print(f'  {"─"*62}')
for col in new_cols:
    if col in df.columns:
        s = df[col].dropna()
        if len(s) > 0:
            print(f'  {col:<30} {len(s):>8} {s.min():>10.3f} {s.max():>10.3f}')
        else:
            print(f'  {col:<30} {"0 — check source":>40}')
    else:
        print(f'  {col:<30} {"NOT IN DATAFRAME":>40}')

# ════════════════════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════════════════════

print_section('Saving')
df.to_csv(OUTPUT_CSV, index=False)
print(f'  ✓  Saved: {OUTPUT_CSV}')
print(f'     {df.shape[0]} rows × {df.shape[1]} columns')
