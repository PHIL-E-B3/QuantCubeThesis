"""
append_sentiment_channels_final.py
Full date range 1999–2025. Reads and overwrites Master_Macro.csv in place.
Run: exec(open('append_sentiment_channels_final.py').read())
"""

import os, json, re, warnings, requests
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

warnings.filterwarnings('ignore')

BASE_DIR   = '/content/drive/MyDrive/HEC Thesis'
MASTER_CSV = f'{BASE_DIR}/Data/Master_Macro.csv'
FRED_KEY   = 'c5b5ec287d025fecd23b73051ee03c84'
JK_FILE    = f'{BASE_DIR}/Data/fomc_surprises_jk.csv.xlsx'
TW_FILE    = f'{BASE_DIR}/Data/FOMC_Dissents_Data.xlsx'
JSON_STMTS = f'{BASE_DIR}/Text Data/structured_json_statements'

def p(t): print(f'\n{"═"*65}\n  {t}\n{"═"*65}')

# ════════════════════════════════════════════════════════════════════════════
# LOAD
# ════════════════════════════════════════════════════════════════════════════

p('Loading Master_Macro.csv')
df = pd.read_csv(MASTER_CSV, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
print(f'  {len(df)} rows × {len(df.columns)} columns')

# Drop old channel columns to rebuild cleanly
DROP = ['ffr_path_change','michigan_1yr_inf_exp','inf_exp_gap','inf_exp_5y5y',
        'nfci','sloos_ci_tightening','jk_mp1','jk_mp_shock','jk_info_shock',
        'stmt_novelty_jaccard','stmt_novelty_cosine',
        'fomc_dissent_count','fomc_gov_dissent','fomc_pres_dissent','fomc_any_dissent']
df = df.drop(columns=[c for c in DROP if c in df.columns])
print(f'  Dropped old channel columns, rebuilding.')

# ════════════════════════════════════════════════════════════════════════════
# 1. STATEMENT NOVELTY (from local JSON files)
# ════════════════════════════════════════════════════════════════════════════

p('1. Statement Novelty')

stmt_texts = {}
for fname in sorted(os.listdir(JSON_STMTS)):
    if not fname.endswith('.json'): continue
    with open(os.path.join(JSON_STMTS, fname)) as f:
        rec = json.load(f)
    if rec.get('text') and len(rec['text']) > 50:
        stmt_texts[pd.Timestamp(rec['date'])] = rec['text']

print(f'  Loaded {len(stmt_texts)} statement texts')
stmt_dates = sorted(stmt_texts.keys())
tfidf      = TfidfVectorizer(stop_words='english', max_features=500)
tfidf_mat  = tfidf.fit_transform([stmt_texts[d] for d in stmt_dates])

def jaccard(t1, t2):
    s1 = set(re.findall(r'\b[a-z]+\b', t1.lower()))
    s2 = set(re.findall(r'\b[a-z]+\b', t2.lower()))
    return len(s1&s2)/len(s1|s2) if s1 and s2 else np.nan

nov = {}
for i in range(1, len(stmt_dates)):
    d   = stmt_dates[i]
    jac = jaccard(stmt_texts[d], stmt_texts[stmt_dates[i-1]])
    cos = sk_cosine(tfidf_mat[i], tfidf_mat[i-1])[0, 0]
    nov[d] = {'stmt_novelty_jaccard': 1-jac if pd.notna(jac) else np.nan,
              'stmt_novelty_cosine':  1-cos}

nov_df = (pd.DataFrame(nov).T
            .reset_index()
            .rename(columns={'index': 'date'}))
nov_df['date'] = pd.to_datetime(nov_df['date'])

is_meeting = df['event_type'] == 'meeting_date'
df_mtg     = df[is_meeting][['date']].sort_values('date').copy()

nov_merged = pd.merge_asof(df_mtg, nov_df.sort_values('date'),
                            on='date', direction='nearest',
                            tolerance=pd.Timedelta('3d'))
df = df.merge(nov_merged, on='date', how='left')
df['stmt_novelty_jaccard'] = df['stmt_novelty_jaccard'].ffill()
df['stmt_novelty_cosine']  = df['stmt_novelty_cosine'].ffill()
print(f'  ✓  stmt_novelty_jaccard: {df["stmt_novelty_jaccard"].notna().sum()} non-NaN')
print(f'  ✓  stmt_novelty_cosine:  {df["stmt_novelty_cosine"].notna().sum()} non-NaN')

# ════════════════════════════════════════════════════════════════════════════
# 2. DISSENT (Thornton-Wheelock — full history)
# ════════════════════════════════════════════════════════════════════════════

p('2. FOMC Disagreement (Thornton-Wheelock)')

try:
    tw = pd.read_excel(TW_FILE, skiprows=3)
    tw.columns = tw.columns.str.strip()
    tw = tw.rename(columns={
        'FOMC Meeting':              'date',
        'Votes Against Action':      'fomc_dissent_count',
        'Number Governors Dissenting':'fomc_gov_dissent',
        'Number Presidents Dissenting':'fomc_pres_dissent',
    })
    tw['date'] = pd.to_datetime(tw['date']).dt.normalize()
    tw = tw.dropna(subset=['date'])

    tw_cols = [c for c in ['date','fomc_dissent_count',
                            'fomc_gov_dissent','fomc_pres_dissent']
               if c in tw.columns]

    is_meeting = df['event_type'] == 'meeting_date'
    df_mtg     = df[is_meeting][['date']].sort_values('date').copy()

    tw_merged = pd.merge_asof(df_mtg,
                               tw[tw_cols].sort_values('date'),
                               on='date', direction='nearest',
                               tolerance=pd.Timedelta('2d'))
    df = df.merge(tw_merged, on='date', how='left')

    df['fomc_any_dissent'] = (df['fomc_dissent_count'] > 0).astype(float)
    df.loc[df['fomc_dissent_count'].isna(), 'fomc_any_dissent'] = np.nan

    for c in ['fomc_dissent_count','fomc_gov_dissent',
              'fomc_pres_dissent','fomc_any_dissent']:
        if c in df.columns:
            df[c] = df[c].ffill()

    print(f'  ✓  fomc_dissent_count: {df["fomc_dissent_count"].notna().sum()} non-NaN')
except Exception as e:
    print(f'  ✗  Dissent failed: {e}')
    for c in ['fomc_dissent_count','fomc_gov_dissent',
              'fomc_pres_dissent','fomc_any_dissent']:
        df[c] = np.nan

# ════════════════════════════════════════════════════════════════════════════
# 3. FRED CHANNELS
# ════════════════════════════════════════════════════════════════════════════

p('3. FRED Channels')

def fred_get(sid):
    url = (f'https://api.stlouisfed.org/fred/series/observations'
           f'?series_id={sid}&api_key={FRED_KEY}'
           f'&observation_start=1990-01-01&file_type=json')
    r = requests.get(url, timeout=30); r.raise_for_status()
    return pd.Series(
        {pd.Timestamp(d['date']): float(d['value'])
         for d in r.json()['observations'] if d['value'] != '.'}
    ).sort_index()

def merge_bwd(series, col):
    s = series.reset_index(); s.columns = ['date', col]
    s['date'] = pd.to_datetime(s['date'])
    tmp = pd.DataFrame({'date': df['date']}).sort_values('date').reset_index()
    m = pd.merge_asof(tmp, s.sort_values('date'), on='date', direction='backward')
    return m.set_index('index').sort_index()[col]

for col, sid in [('michigan_1yr_inf_exp', 'MICH'),
                 ('inf_exp_5y5y',         'T5YIFR'),
                 ('nfci',                 'NFCI'),
                 ('sloos_ci_tightening',  'DRTSCILM')]:
    try:
        s = fred_get(sid)
        df[col] = merge_bwd(s, col)
        print(f'  ✓  {col}: {df[col].notna().sum()} non-NaN')
    except Exception as e:
        print(f'  ✗  {col}: {e}')
        df[col] = np.nan

df['inf_exp_gap'] = df['michigan_1yr_inf_exp'] - 2.0
print(f'  ✓  inf_exp_gap: {df["inf_exp_gap"].notna().sum()} non-NaN')

# ════════════════════════════════════════════════════════════════════════════
# 4. FFR PATH CHANGE
# ════════════════════════════════════════════════════════════════════════════

p('4. Forward Guidance Shift')

is_meeting = df['event_type'] == 'meeting_date'   # recompute after merges
mi = df.loc[is_meeting, ['date','implied_ffr']].sort_values('date').copy()
mi['ffr_path_change'] = mi['implied_ffr'].diff()
df = df.merge(mi[['date','ffr_path_change']], on='date', how='left')
df['ffr_path_change'] = df['ffr_path_change'].ffill()
print(f'  ✓  ffr_path_change: {df["ffr_path_change"].notna().sum()} non-NaN')

# ════════════════════════════════════════════════════════════════════════════
# 5. JK SHOCKS
# ════════════════════════════════════════════════════════════════════════════

p('5. JK Shocks')

try:
    jk = pd.read_excel(JK_FILE)
    dc = [c for c in jk.columns if c.lower() in ('start','date')][0]
    jk['date'] = pd.to_datetime(jk[dc]).dt.normalize()
    if 'description' in jk.columns:
        jk = jk[jk['description'].str.contains('Scheduled', case=False, na=False)]
    jk = jk.rename(columns={'MP1': 'jk_mp1'})
    jk['jk_mp1'] = pd.to_numeric(jk['jk_mp1'], errors='coerce')
    sp = next((c for c in jk.columns if c.upper() in ('SP500','SP500FUT')), None)
    if sp:
        jk[sp] = pd.to_numeric(jk[sp], errors='coerce')
        clean = jk[['jk_mp1', sp]].dropna()
        ols   = sm.OLS(clean['jk_mp1'], sm.add_constant(clean[sp])).fit()
        jk.loc[clean.index, 'jk_mp_shock']  = ols.resid.values
        jk.loc[clean.index, 'jk_info_shock'] = ols.fittedvalues.values
        print(f'  ✓  JK decomposition: {len(clean)} meetings  R²={ols.rsquared:.4f}')
    else:
        jk['jk_mp_shock'] = jk['jk_mp1']
        jk['jk_info_shock'] = np.nan

    mc = [c for c in ['date','jk_mp1','jk_mp_shock','jk_info_shock'] if c in jk.columns]
    df = df.merge(jk[mc], on='date', how='left')
    print(f'  ✓  jk_mp_shock:   {df["jk_mp_shock"].notna().sum()} non-NaN')
    print(f'  ✓  jk_info_shock: {df["jk_info_shock"].notna().sum()} non-NaN')
except Exception as e:
    print(f'  ✗  JK failed: {e}')
    df['jk_mp1'] = df['jk_mp_shock'] = df['jk_info_shock'] = np.nan

# ════════════════════════════════════════════════════════════════════════════
# SAVE + VERIFY
# ════════════════════════════════════════════════════════════════════════════

p('Saving')
df.to_csv(MASTER_CSV, index=False)
print(f'  ✓  Saved: {MASTER_CSV}')
print(f'     {df.shape[0]} rows × {df.shape[1]} columns')

# Read back to verify
p('Verification (read back from CSV)')
df2 = pd.read_csv(MASTER_CSV, parse_dates=['date'])
is_m = df2['event_type'] == 'meeting_date'
check = ['stmt_novelty_cosine','stmt_novelty_jaccard',
         'fomc_dissent_count','fomc_any_dissent',
         'michigan_1yr_inf_exp','inf_exp_5y5y','nfci','sloos_ci_tightening',
         'ffr_path_change','jk_mp_shock','jk_info_shock']
print(f'\n  {"Column":<30} {"Meeting":>10} {"All rows":>10}')
print(f'  {"─"*52}')
for c in check:
    if c in df2.columns:
        n_mtg = df2.loc[is_m, c].notna().sum()
        n_all = df2[c].notna().sum()
        print(f'  {c:<30} {n_mtg:>10}/{is_m.sum()} {n_all:>10}/{len(df2)}')
    else:
        print(f'  {c:<30} {"NOT FOUND":>22}')
