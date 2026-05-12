"""
03_append_channels.py
---------------------
Appends channel regressor columns to Master_Macro.csv in place.

Channels added (or rebuilt if already present):
    ffr_path_change         — meeting-to-meeting shift in implied FFR
    michigan_1yr_inf_exp    — Michigan 1-year inflation expectations (MICH)
    inf_exp_gap             — michigan_1yr_inf_exp − 2.0
    inf_exp_5y5y            — 5y5y forward breakeven (T5YIFR)
    nfci                    — Chicago Fed National Financial Conditions Index
    sloos_ci_tightening     — SLOOS C&I lending standards (DRTSCILM)
    jk_mp1                  — Jarociński-Karadi raw MP surprise
    jk_mp_shock             — JK monetary policy shock (MP1 residual on SP500)
    jk_info_shock           — JK information effect (MP1 fitted on SP500)
    stmt_novelty_jaccard    — 1 − Jaccard similarity vs. previous statement
    stmt_novelty_cosine     — 1 − TF-IDF cosine similarity vs. previous statement
    fomc_dissent_count      — total votes against (Thornton-Wheelock)
    fomc_gov_dissent        — governor dissents
    fomc_pres_dissent       — president dissents
    fomc_any_dissent        — binary dissent indicator

Usage:
    python macro_pipeline/03_append_channels.py
"""

import json
import os
import re
import warnings

import numpy  as np
import pandas as pd
import requests
import statsmodels.api as sm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise        import cosine_similarity as sk_cosine

warnings.filterwarnings('ignore')

from config import (
    FRED_API_KEY, MASTER_MACRO_CSV,
    JK_SHOCKS_FILE, TW_DISSENTS_FILE, STATEMENTS_JSON_DIR,
)

CHANNEL_COLS = [
    'ffr_path_change',
    'michigan_1yr_inf_exp', 'inf_exp_gap', 'inf_exp_5y5y',
    'nfci', 'sloos_ci_tightening',
    'jk_mp1', 'jk_mp_shock', 'jk_info_shock',
    'stmt_novelty_jaccard', 'stmt_novelty_cosine',
    'fomc_dissent_count', 'fomc_gov_dissent', 'fomc_pres_dissent', 'fomc_any_dissent',
]


# ── FRED helper ───────────────────────────────────────────────────────────────

def fred_get(series_id: str, start: str = '1990-01-01') -> pd.Series:
    url = (f'https://api.stlouisfed.org/fred/series/observations'
           f'?series_id={series_id}&api_key={FRED_API_KEY}'
           f'&observation_start={start}&file_type=json')
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.Series(
        {pd.Timestamp(d['date']): float(d['value'])
         for d in r.json()['observations'] if d['value'] != '.'}
    ).sort_index()


def merge_bwd(df: pd.DataFrame, series: pd.Series, col_name: str) -> pd.Series:
    """Merge a time-series into df using backward as-of lookup."""
    s = series.reset_index()
    s.columns = ['date', col_name]
    s['date'] = pd.to_datetime(s['date'])
    tmp = pd.DataFrame({'date': pd.to_datetime(df['date'])}).sort_values('date').reset_index()
    merged = pd.merge_asof(tmp, s.sort_values('date'), on='date', direction='backward')
    return merged.set_index('index').sort_index()[col_name]


def p(title: str):
    print(f'\n{"═"*65}\n  {title}\n{"═"*65}')


# ── Channel builders ──────────────────────────────────────────────────────────

def add_ffr_path_change(df: pd.DataFrame) -> pd.DataFrame:
    is_meeting = df['event_type'] == 'meeting_date'
    mi = df.loc[is_meeting, ['date', 'implied_ffr']].sort_values('date').copy()
    mi['ffr_path_change'] = mi['implied_ffr'].diff()
    df = df.merge(mi[['date', 'ffr_path_change']], on='date', how='left')
    df['ffr_path_change'] = df['ffr_path_change'].ffill()
    print(f'  ✓  ffr_path_change: {df["ffr_path_change"].notna().sum()} non-NaN')
    return df


def add_fred_channels(df: pd.DataFrame) -> pd.DataFrame:
    for col, sid in [
        ('michigan_1yr_inf_exp', 'MICH'),
        ('inf_exp_5y5y',         'T5YIFR'),
        ('nfci',                 'NFCI'),
        ('sloos_ci_tightening',  'DRTSCILM'),
    ]:
        try:
            s = fred_get(sid)
            df[col] = merge_bwd(df, s, col)
            print(f'  ✓  {col}: {df[col].notna().sum()} non-NaN')
        except Exception as e:
            print(f'  ✗  {col}: {e}')
            df[col] = np.nan
    df['inf_exp_gap'] = df['michigan_1yr_inf_exp'] - 2.0
    print(f'  ✓  inf_exp_gap: {df["inf_exp_gap"].notna().sum()} non-NaN')
    return df


def add_jk_shocks(df: pd.DataFrame) -> pd.DataFrame:
    if not JK_SHOCKS_FILE.exists():
        print(f'  ⚠️  JK file not found: {JK_SHOCKS_FILE}')
        df[['jk_mp1', 'jk_mp_shock', 'jk_info_shock']] = np.nan
        return df
    try:
        jk = pd.read_excel(JK_SHOCKS_FILE)
        date_col = next(c for c in jk.columns if c.lower() in ('start', 'date'))
        jk['date'] = pd.to_datetime(jk[date_col]).dt.normalize()

        if 'description' in jk.columns:
            jk = jk[jk['description'].str.contains('Scheduled', case=False, na=False)]

        jk = jk.rename(columns={'MP1': 'jk_mp1'})
        jk['jk_mp1'] = pd.to_numeric(jk['jk_mp1'], errors='coerce')

        sp_col = next((c for c in jk.columns if c.upper() in ('SP500', 'SP500FUT')), None)
        if sp_col:
            jk[sp_col] = pd.to_numeric(jk[sp_col], errors='coerce')
            clean = jk[['jk_mp1', sp_col]].dropna()
            ols   = sm.OLS(clean['jk_mp1'], sm.add_constant(clean[sp_col])).fit()
            jk.loc[clean.index, 'jk_mp_shock']  = ols.resid.values
            jk.loc[clean.index, 'jk_info_shock'] = ols.fittedvalues.values
            print(f'  ✓  JK decomposition: {len(clean)} meetings  R²={ols.rsquared:.4f}')
        else:
            jk['jk_mp_shock']  = jk['jk_mp1']
            jk['jk_info_shock'] = np.nan
            print('  ⚠️  SP500 column not found — raw MP1 used for jk_mp_shock')

        merge_cols = [c for c in ['date', 'jk_mp1', 'jk_mp_shock', 'jk_info_shock']
                      if c in jk.columns]
        df = df.merge(jk[merge_cols], on='date', how='left')
        print(f'  ✓  jk_mp1:        {df["jk_mp1"].notna().sum()} non-NaN')
        print(f'  ✓  jk_mp_shock:   {df["jk_mp_shock"].notna().sum()} non-NaN')
        print(f'  ✓  jk_info_shock: {df["jk_info_shock"].notna().sum()} non-NaN')
    except Exception as e:
        print(f'  ✗  JK shocks failed: {e}')
        df[['jk_mp1', 'jk_mp_shock', 'jk_info_shock']] = np.nan
    return df


def add_statement_novelty(df: pd.DataFrame) -> pd.DataFrame:
    if not STATEMENTS_JSON_DIR.exists():
        print(f'  ⚠️  Statements folder not found: {STATEMENTS_JSON_DIR}')
        df[['stmt_novelty_jaccard', 'stmt_novelty_cosine']] = np.nan
        return df
    try:
        stmt_texts = {}
        for fname in sorted(os.listdir(STATEMENTS_JSON_DIR)):
            if not (fname.endswith('.json') and fname.startswith('statement_')):
                continue
            try:
                with open(STATEMENTS_JSON_DIR / fname, encoding='utf-8') as f:
                    rec = json.load(f)
                text = rec.get('text', '')
                if text and len(text) > 50:
                    stmt_texts[pd.Timestamp(rec['date'])] = text
            except Exception:
                pass
        print(f'  Loaded {len(stmt_texts)} statements  '
              f'({min(stmt_texts).date()} – {max(stmt_texts).date()})')

        stmt_dates = sorted(stmt_texts.keys())
        tfidf      = TfidfVectorizer(stop_words='english', max_features=500)
        tfidf_mat  = tfidf.fit_transform([stmt_texts[d] for d in stmt_dates])

        def jaccard(t1, t2):
            s1 = set(re.findall(r'\b[a-z]+\b', t1.lower()))
            s2 = set(re.findall(r'\b[a-z]+\b', t2.lower()))
            return len(s1 & s2) / len(s1 | s2) if s1 and s2 else np.nan

        nov = {}
        for i in range(1, len(stmt_dates)):
            d   = stmt_dates[i]
            jac = jaccard(stmt_texts[d], stmt_texts[stmt_dates[i-1]])
            cos = sk_cosine(tfidf_mat[i], tfidf_mat[i-1])[0, 0]
            nov[d] = {'stmt_novelty_jaccard': 1 - jac if pd.notna(jac) else np.nan,
                      'stmt_novelty_cosine':  1 - cos}

        nov_df = (pd.DataFrame(nov).T.reset_index()
                  .rename(columns={'index': 'date'}))
        nov_df['date'] = pd.to_datetime(nov_df['date'])

        is_meeting = df['event_type'] == 'meeting_date'
        df_mtg     = df[is_meeting][['date']].sort_values('date').copy()
        nov_merged = pd.merge_asof(
            df_mtg, nov_df.sort_values('date'),
            on='date', direction='nearest', tolerance=pd.Timedelta('3d')
        )
        df = df.merge(nov_merged, on='date', how='left')
        df['stmt_novelty_jaccard'] = df['stmt_novelty_jaccard'].ffill()
        df['stmt_novelty_cosine']  = df['stmt_novelty_cosine'].ffill()
        print(f'  ✓  stmt_novelty_jaccard: {df["stmt_novelty_jaccard"].notna().sum()} non-NaN')
        print(f'  ✓  stmt_novelty_cosine:  {df["stmt_novelty_cosine"].notna().sum()} non-NaN')
    except Exception as e:
        print(f'  ✗  Statement novelty failed: {e}')
        df[['stmt_novelty_jaccard', 'stmt_novelty_cosine']] = np.nan
    return df


def add_fomc_dissent(df: pd.DataFrame) -> pd.DataFrame:
    if not TW_DISSENTS_FILE.exists():
        print(f'  ⚠️  Dissents file not found: {TW_DISSENTS_FILE}')
        df[['fomc_dissent_count','fomc_gov_dissent','fomc_pres_dissent','fomc_any_dissent']] = np.nan
        return df
    try:
        tw = pd.read_excel(TW_DISSENTS_FILE, skiprows=3)
        tw.columns = tw.columns.str.strip()
        tw = tw.rename(columns={
            'FOMC Meeting':               'date',
            'Votes Against Action':       'fomc_dissent_count',
            'Number Governors Dissenting': 'fomc_gov_dissent',
            'Number Presidents Dissenting':'fomc_pres_dissent',
        })
        tw['date'] = pd.to_datetime(tw['date']).dt.normalize()
        tw = tw.dropna(subset=['date'])

        tw_cols = [c for c in ['date','fomc_dissent_count','fomc_gov_dissent','fomc_pres_dissent']
                   if c in tw.columns]

        is_meeting = df['event_type'] == 'meeting_date'  # recompute after prior merges
        df_mtg     = df[is_meeting][['date']].sort_values('date').copy()
        tw_merged  = pd.merge_asof(
            df_mtg, tw[tw_cols].sort_values('date'),
            on='date', direction='nearest', tolerance=pd.Timedelta('2d')
        )
        df = df.merge(tw_merged, on='date', how='left')

        df['fomc_any_dissent'] = (df['fomc_dissent_count'] > 0).astype(float)
        df.loc[df['fomc_dissent_count'].isna(), 'fomc_any_dissent'] = np.nan

        for c in ['fomc_dissent_count','fomc_gov_dissent','fomc_pres_dissent','fomc_any_dissent']:
            if c in df.columns:
                df[c] = df[c].ffill()

        print(f'  ✓  fomc_dissent_count: {df["fomc_dissent_count"].notna().sum()} non-NaN')
        dist = df.loc[is_meeting, 'fomc_dissent_count'].value_counts().sort_index().to_dict()
        print(f'     Distribution: {dist}')
    except Exception as e:
        print(f'  ✗  Dissent failed: {e}')
        df[['fomc_dissent_count','fomc_gov_dissent','fomc_pres_dissent','fomc_any_dissent']] = np.nan
    return df


# ── Verification ──────────────────────────────────────────────────────────────

def verify(df: pd.DataFrame):
    is_m = df['event_type'] == 'meeting_date'
    print(f'\n  {"Column":<30} {"Meetings":>10}  {"All rows":>10}')
    print(f'  {"─"*55}')
    for col in CHANNEL_COLS:
        if col in df.columns:
            n_m = df.loc[is_m, col].notna().sum()
            n_a = df[col].notna().sum()
            print(f'  {col:<30} {n_m:>6}/{is_m.sum()}    {n_a:>6}/{len(df)}')
        else:
            print(f'  {col:<30} {"NOT IN DATAFRAME":>30}')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p('Loading Master_Macro.csv')
    df = pd.read_csv(MASTER_MACRO_CSV, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f'  {len(df)} rows × {len(df.columns)} columns  '
          f'({df["date"].min().date()} – {df["date"].max().date()})')

    df = df.drop(columns=[c for c in CHANNEL_COLS if c in df.columns])
    print(f'  Dropped old channel columns — rebuilding from scratch.')

    p('1. FFR Path Change')
    df = add_ffr_path_change(df)

    p('2. FRED Channels (inflation expectations, NFCI, SLOOS)')
    df = add_fred_channels(df)

    p('3. JK Monetary Policy Shocks')
    df = add_jk_shocks(df)

    p('4. Statement Novelty')
    df = add_statement_novelty(df)

    p('5. FOMC Internal Disagreement (Thornton-Wheelock)')
    df = add_fomc_dissent(df)

    p('Verification')
    verify(df)

    p('Saving')
    df.to_csv(MASTER_MACRO_CSV, index=False)
    print(f'  ✓  {MASTER_MACRO_CSV}')
    print(f'     {df.shape[0]} rows × {df.shape[1]} columns')


if __name__ == '__main__':
    main()
