"""
05_append_sep_data.py
---------------------
Appends FOMC Summary of Economic Projections (SEP) data to Master_Macro.csv,
writing the result to Master_Macro_with_SEP.csv (Master_Macro.csv is not
overwritten so the base dataset stays clean).

Data sources:
  1. FRED API — SEP central tendency, range, and median series for:
       - Real GDP growth (GDPC1*)
       - Unemployment rate (UNRATE*)
       - PCE inflation (PCPILFE*)
  2. FRED API — FFR dot plot dispersion (FEDTAR*)

Usage:
    python macro_pipeline/05_append_sep_data.py
    python macro_pipeline/05_append_sep_data.py --output data/macro_inputs/Master_Macro_with_SEP.csv
"""

import argparse
import warnings
import requests
import numpy  as np
import pandas as pd

warnings.filterwarnings('ignore')

from config import FRED_API_KEY, MASTER_MACRO_CSV, MASTER_SEP_CSV

FRED_BASE = 'https://api.stlouisfed.org/fred/series/observations'

# ── SEP FRED series catalogue ─────────────────────────────────────────────────
# Each entry: (series_id, variable, horizon, metric)
SEP_SERIES = [
    # Real GDP growth
    ('GDPC1MD',    'gdp', 'projection', 'median'),
    ('GDPC1CTM',   'gdp', 'projection', 'ct_mid'),
    ('GDPC1CTL',   'gdp', 'projection', 'ct_low'),
    ('GDPC1CTH',   'gdp', 'projection', 'ct_high'),
    ('GDPC1RL',    'gdp', 'projection', 'range_low'),
    ('GDPC1RH',    'gdp', 'projection', 'range_high'),
    ('GDPC1MDLR',  'gdp', 'longer_run', 'median'),
    ('GDPC1CTMLR', 'gdp', 'longer_run', 'ct_mid'),
    ('GDPC1CTLLR', 'gdp', 'longer_run', 'ct_low'),
    ('GDPC1CTHLR', 'gdp', 'longer_run', 'ct_high'),
    # Unemployment rate
    ('UNRATEMD',   'unemp', 'projection', 'median'),
    ('UNRATECTM',  'unemp', 'projection', 'ct_mid'),
    ('UNRATECTL',  'unemp', 'projection', 'ct_low'),
    ('UNRATECTH',  'unemp', 'projection', 'ct_high'),
    ('UNRATERL',   'unemp', 'projection', 'range_low'),
    ('UNRATERH',   'unemp', 'projection', 'range_high'),
    ('UNRATEMDLR', 'unemp', 'longer_run', 'median'),
    ('UNRATECTMLR','unemp', 'longer_run', 'ct_mid'),
    ('UNRATECTLLR','unemp', 'longer_run', 'ct_low'),
    ('UNRACTCTHLR','unemp', 'longer_run', 'ct_high'),
    # PCE inflation
    ('PCPILFEMD',  'pce', 'projection', 'median'),
    ('PCPILFECTM', 'pce', 'projection', 'ct_mid'),
    ('PCPILFECTL', 'pce', 'projection', 'ct_low'),
    ('PCPILFECTH', 'pce', 'projection', 'ct_high'),
    ('PCPILFERLR', 'pce', 'projection', 'range_low'),
    ('PCPILFERHR', 'pce', 'projection', 'range_high'),
    ('PCPILFEMDLR','pce', 'longer_run', 'median'),
    ('PCPILFECTMLR','pce','longer_run', 'ct_mid'),
    ('PCPILFECTLLR','pce','longer_run', 'ct_low'),
    ('PCPILFECTHLR','pce','longer_run', 'ct_high'),
]

# FFR dot plot dispersion
FFR_SERIES = {
    'sep_ffr_median_0y':   'FEDTARMD0',
    'sep_ffr_median_1y':   'FEDTARMD1',
    'sep_ffr_median_2y':   'FEDTARMD2',
    'sep_ffr_median_3y':   'FEDTARMD3',
    'sep_ffr_median_lr':   'FEDTARMD',
    'sep_ffr_ct_high':     'FEDTARCTH',
    'sep_ffr_ct_low':      'FEDTARCTL',
    'sep_ffr_range_high':  'FEDTARRH',
    'sep_ffr_range_low':   'FEDTARRL',
}


def fred_fetch(series_id: str) -> pd.Series | None:
    try:
        r = requests.get(FRED_BASE, params={
            'series_id':    series_id,
            'api_key':      FRED_API_KEY,
            'file_type':    'json',
        }, timeout=20)
        r.raise_for_status()
        data = r.json().get('observations', [])
        s = pd.Series(
            {pd.Timestamp(d['date']): float(d['value'])
             for d in data if d['value'] != '.'},
            name=series_id
        )
        return s.sort_index() if not s.empty else None
    except Exception as e:
        print(f'  ✗  {series_id}: {e}')
        return None


def merge_asof_bwd(df: pd.DataFrame, series: pd.Series, col_name: str) -> pd.Series:
    s = series.reset_index()
    s.columns = ['date', col_name]
    s['date'] = pd.to_datetime(s['date'])
    tmp = pd.DataFrame({'date': df['date']}).sort_values('date').reset_index()
    merged = pd.merge_asof(tmp, s.sort_values('date'), on='date', direction='backward')
    return merged.set_index('index').sort_index()[col_name]


def p(title: str):
    print(f'\n{"═"*65}\n  {title}\n{"═"*65}')


def main(output_path=None):
    out_path = output_path or MASTER_SEP_CSV

    p('Loading Master_Macro.csv')
    df = pd.read_csv(MASTER_MACRO_CSV, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f'  {len(df)} rows × {len(df.columns)} columns')

    p('Fetching SEP GDP / Unemployment / PCE series from FRED')
    fetched = 0
    for series_id, variable, horizon, metric in SEP_SERIES:
        col_name = f'sep_{variable}_{horizon}_{metric}'
        s = fred_fetch(series_id)
        if s is not None and not s.empty:
            df[col_name] = merge_asof_bwd(df, s, col_name)
            n = df[col_name].notna().sum()
            print(f'  ✓  {col_name:<40} {n} non-NaN')
            fetched += 1
        else:
            df[col_name] = np.nan

    p('Fetching FFR dot-plot dispersion from FRED')
    for col_name, series_id in FFR_SERIES.items():
        s = fred_fetch(series_id)
        if s is not None and not s.empty:
            df[col_name] = merge_asof_bwd(df, s, col_name)
            n = df[col_name].notna().sum()
            print(f'  ✓  {col_name:<40} {n} non-NaN')
        else:
            df[col_name] = np.nan

    p(f'Saving to {out_path}')
    out_path = type(MASTER_SEP_CSV)(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'  ✓  {out_path}')
    print(f'     {df.shape[0]} rows × {df.shape[1]} columns')
    print(f'     ({fetched} SEP series fetched successfully)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Append SEP data to Master_Macro')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: Master_Macro_with_SEP.csv)')
    args = parser.parse_args()
    main(output_path=args.output)
