"""
04_append_spf_dispersion.py
---------------------------
Appends SPF cross-sectional dispersion columns to Master_Macro.csv.

Sources (Philadelphia Fed):
    Dispersion_1.xlsx  — D1: P75−P25 of level forecasts
                         Sheets: NGDP, UNEMP, CPI, PCE
    Dispersion_2.xlsx  — D2: P75−P25 of Q/Q growth forecasts
                         Sheets: NGDP, RGDP, RCONSUM
    Dispersion_3.xlsx  — D3: 100×log(P75/P25) of level forecasts
                         Sheets: NGDP, UNEMP, RGDP, RCONSUM

Each FOMC date is matched to the most recent survey available on or before
that date (merge_asof backward), adding ~240 columns at horizons T through T+4.

Usage:
    python macro_pipeline/04_append_spf_dispersion.py
"""

import warnings
import numpy  as np
import pandas as pd

warnings.filterwarnings('ignore')

from config import (
    MASTER_MACRO_CSV,
    SPF_DISP_D1_FILE, SPF_DISP_D2_FILE, SPF_DISP_D3_FILE,
)

DISP_FILES = {
    'D1': SPF_DISP_D1_FILE,
    'D2': SPF_DISP_D2_FILE,
    'D3': SPF_DISP_D3_FILE,
}
SHEETS_TO_USE = {
    'D1': ['NGDP', 'UNEMP', 'CPI', 'PCE'],
    'D2': ['NGDP', 'RGDP', 'RCONSUM'],
    'D3': ['NGDP', 'UNEMP', 'RGDP', 'RCONSUM'],
}


def p(title: str):
    print(f'\n{"═"*65}\n  {title}\n{"═"*65}')


def parse_survey_date(val) -> pd.Timestamp:
    try:
        return pd.Period(str(val).strip(), freq='Q').to_timestamp('Q')
    except Exception:
        return pd.NaT


def load_dispersion_sheet(filepath, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(filepath, sheet_name=sheet_name, skiprows=9, header=0)
    raw.columns = [str(c).strip() for c in raw.columns]
    date_col = raw.columns[0]
    raw = raw.dropna(subset=[date_col])
    raw['survey_date'] = raw[date_col].apply(parse_survey_date)
    raw = raw.dropna(subset=['survey_date']).drop(columns=[date_col])
    raw = raw.set_index('survey_date').sort_index()
    for col in raw.columns:
        raw[col] = pd.to_numeric(raw[col], errors='coerce')
    return raw


def main():
    p('Loading Master_Macro.csv')
    df = pd.read_csv(MASTER_MACRO_CSV, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f'  {len(df)} rows × {len(df.columns)} columns  '
          f'({df["date"].min().date()} – {df["date"].max().date()})')

    p('Loading Dispersion files')
    all_disp = []
    for disp_key, filepath in DISP_FILES.items():
        if not filepath.exists():
            print(f'  ⚠️  Not found: {filepath}')
            continue
        for sheet in SHEETS_TO_USE[disp_key]:
            try:
                sheet_df = load_dispersion_sheet(filepath, sheet)
                print(f'  ✓  {disp_key}/{sheet:<10}  '
                      f'{len(sheet_df)} quarters  '
                      f'{len(sheet_df.columns)} cols  '
                      f'({sheet_df.index.min().date()} – {sheet_df.index.max().date()})')
                all_disp.append(sheet_df)
            except Exception as e:
                print(f'  ✗  {disp_key}/{sheet}: {e}')

    if not all_disp:
        print('  No dispersion data loaded — check file paths in config.py')
        return

    disp_combined = pd.concat(all_disp, axis=1)
    disp_combined = disp_combined.loc[:, ~disp_combined.columns.duplicated()].sort_index()
    print(f'\n  Combined: {len(disp_combined)} quarters × {len(disp_combined.columns)} columns')

    p('Merging to Master_Macro')
    all_disp_cols = list(disp_combined.columns)
    df = df.drop(columns=[c for c in all_disp_cols if c in df.columns])

    disp_reset = disp_combined.reset_index().rename(columns={'survey_date': 'survey_date'})
    disp_reset['survey_date'] = pd.to_datetime(disp_reset['survey_date'])

    tmp    = pd.DataFrame({'date': df['date']}).sort_values('date').reset_index()
    merged = pd.merge_asof(tmp, disp_reset, left_on='date', right_on='survey_date',
                           direction='backward')
    merged = merged.set_index('index').sort_index()
    for col in all_disp_cols:
        if col in merged.columns:
            df[col] = merged[col].values

    sample_cols = [c for c in all_disp_cols if str(c).endswith('(T)')][:8]
    print(f'  Coverage check (T-horizon columns):')
    for c in sample_cols:
        print(f'    {str(c):<35}: {df[c].notna().sum()}/{len(df)} non-NaN')

    p('Saving')
    df.to_csv(MASTER_MACRO_CSV, index=False)
    print(f'  ✓  {MASTER_MACRO_CSV}')
    print(f'     {df.shape[0]} rows × {df.shape[1]} columns')


if __name__ == '__main__':
    main()
