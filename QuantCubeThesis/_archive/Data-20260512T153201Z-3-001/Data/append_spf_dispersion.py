"""
append_spf_dispersion.py
════════════════════════════════════════════════════════════════════════════════
Appends SPF cross-sectional dispersion columns to Master_Macro.csv.
Uses the three Dispersion_1/2/3.xlsx files from the Philadelphia Fed.

SHEETS USED (matching existing Master_Macro column names):
  Dispersion_1 (D1 = P75 - P25 of level forecasts):
    NGDP, UNEMP, CPI, PCE
  Dispersion_2 (D2 = P75 - P25 of Q/Q growth forecasts):
    NGDP, RGDP, RCONSUM
  Dispersion_3 (D3 = 100*log(P75/P25) of level forecasts):
    NGDP, UNEMP, RGDP, RCONSUM

MERGE LOGIC:
  Survey dates are quarterly (e.g. 1999Q1). Each FOMC meeting row is matched
  to the most recent survey available on or before that meeting date.
  Results are forward-filled to minutes_date and blackout_date rows.

USAGE:
  Upload Dispersion_1.xlsx, Dispersion_2.xlsx, Dispersion_3.xlsx to:
    /content/drive/MyDrive/HEC Thesis/Data/
  Then run: exec(open('append_spf_dispersion.py').read())
"""

import os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE_DIR   = '/content/drive/MyDrive/HEC Thesis'
MASTER_CSV = f'{BASE_DIR}/Data/Master_Macro.csv'
DISP_FILES = {
    'D1': f'{BASE_DIR}/Data/Dispersion_1.xlsx',
    'D2': f'{BASE_DIR}/Data/Dispersion_2.xlsx',
    'D3': f'{BASE_DIR}/Data/Dispersion_3.xlsx',
}

# Sheets to extract per dispersion file
# Format: {dispersion_key: [sheet_names]}
SHEETS_TO_USE = {
    'D1': ['NGDP', 'UNEMP', 'CPI', 'PCE'],
    'D2': ['NGDP', 'RGDP', 'RCONSUM'],
    'D3': ['NGDP', 'UNEMP', 'RGDP', 'RCONSUM'],
}

def p(t): print(f'\n{"═"*65}\n  {t}\n{"═"*65}')

# ════════════════════════════════════════════════════════════════════════════
# LOAD MASTER
# ════════════════════════════════════════════════════════════════════════════

p('Loading Master_Macro.csv')
df = pd.read_csv(MASTER_CSV, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
print(f'  {len(df)} rows × {len(df.columns)} columns')
print(f'  Date range: {df["date"].min().date()} → {df["date"].max().date()}')

# ════════════════════════════════════════════════════════════════════════════
# LOAD AND PARSE DISPERSION FILES
# ════════════════════════════════════════════════════════════════════════════

p('Loading Dispersion files')

def parse_survey_date(val):
    """Convert '1999Q1' to end-of-quarter Timestamp."""
    try:
        s = str(val).strip()
        return pd.Period(s, freq='Q').to_timestamp('Q')
    except Exception:
        return pd.NaT


def load_dispersion_sheet(filepath, sheet_name):
    """
    Load one sheet from a Dispersion file.
    Returns a DataFrame indexed by survey_date (end of quarter).
    """
    raw = pd.read_excel(filepath, sheet_name=sheet_name, skiprows=9, header=0)
    raw.columns = [str(c).strip() for c in raw.columns]

    # First column is Survey_Date(T)
    date_col = raw.columns[0]
    raw = raw.dropna(subset=[date_col])
    raw['survey_date'] = raw[date_col].apply(parse_survey_date)
    raw = raw.dropna(subset=['survey_date'])
    raw = raw.drop(columns=[date_col])
    raw = raw.set_index('survey_date').sort_index()

    # Convert all data columns to numeric
    for col in raw.columns:
        raw[col] = pd.to_numeric(raw[col], errors='coerce')

    return raw


# Load all relevant sheets into one combined DataFrame indexed by survey_date
all_disp = []

for disp_key, filepath in DISP_FILES.items():
    if not os.path.exists(filepath):
        print(f'  ⚠️  Not found: {filepath}')
        print(f'       Upload Dispersion_{disp_key[-1]}.xlsx to {BASE_DIR}/Data/')
        continue

    sheets = SHEETS_TO_USE.get(disp_key, [])
    for sheet in sheets:
        try:
            df_sheet = load_dispersion_sheet(filepath, sheet)
            print(f'  ✓  {disp_key} / {sheet:<10}  '
                  f'{len(df_sheet)} quarters  '
                  f'{len(df_sheet.columns)} cols  '
                  f'{df_sheet.index.min().date()} → {df_sheet.index.max().date()}')
            all_disp.append(df_sheet)
        except Exception as e:
            print(f'  ✗  {disp_key} / {sheet}: {e}')

if not all_disp:
    print('  No dispersion data loaded. Check file paths.')
    raise SystemExit(1)

# Combine all sheets (they share the survey_date index)
disp_combined = pd.concat(all_disp, axis=1)
# Remove duplicate columns if any sheet appears in multiple files
disp_combined = disp_combined.loc[:, ~disp_combined.columns.duplicated()]
disp_combined = disp_combined.sort_index()

print(f'\n  Combined dispersion data: {len(disp_combined)} quarters × '
      f'{len(disp_combined.columns)} columns')

# ════════════════════════════════════════════════════════════════════════════
# IDENTIFY WHICH COLUMNS TO UPDATE
# ════════════════════════════════════════════════════════════════════════════

p('Matching columns to Master_Macro')

# Find columns in dispersion data that exist in Master_Macro
master_disp_cols = [c for c in disp_combined.columns if c in df.columns]
new_disp_cols    = [c for c in disp_combined.columns if c not in df.columns]

print(f'  Columns already in Master_Macro: {len(master_disp_cols)}')
print(f'  New columns to add:              {len(new_disp_cols)}')

if new_disp_cols:
    print(f'  New cols (first 10): {new_disp_cols[:10]}')

# Work on union — update existing + add new
all_disp_cols = master_disp_cols + new_disp_cols

# ════════════════════════════════════════════════════════════════════════════
# MERGE TO MASTER
# ════════════════════════════════════════════════════════════════════════════

p('Merging dispersion data to Master_Macro')

# Reset dispersion index for merge
disp_reset = disp_combined[all_disp_cols].reset_index()
disp_reset.columns = ['survey_date'] + all_disp_cols
disp_reset['survey_date'] = pd.to_datetime(disp_reset['survey_date'])
disp_reset = disp_reset.sort_values('survey_date')

# Drop existing dispersion columns from df to rebuild cleanly
df = df.drop(columns=[c for c in all_disp_cols if c in df.columns])

# For each row in master, find most recent survey on or before that date
tmp = pd.DataFrame({'date': df['date']}).sort_values('date').reset_index()
merged = pd.merge_asof(
    tmp,
    disp_reset,
    left_on='date',
    right_on='survey_date',
    direction='backward'
)
merged = merged.set_index('index').sort_index()

# Add dispersion columns back to df
for col in all_disp_cols:
    df[col] = merged[col].values

# Forward-fill within each meeting group (blackout/minutes inherit from meeting)
# Already handled by merge_asof since all three dates per meeting are in df
# and merge picks the most recent survey for each date independently

print(f'  Merged. Checking coverage:')
sample_cols = [c for c in all_disp_cols if c.endswith('(T)')][:8]
for c in sample_cols:
    n = df[c].notna().sum()
    print(f'    {c:<35}: {n}/{len(df)}')

# ════════════════════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════════════════════

p('Saving')
df.to_csv(MASTER_CSV, index=False)
print(f'  ✓  Saved: {MASTER_CSV}')
print(f'     {df.shape[0]} rows × {df.shape[1]} columns')

# Verify
p('Verification')
df2 = pd.read_csv(MASTER_CSV, parse_dates=['date'])
is_m = df2['event_type'] == 'meeting_date'
pre2011 = df2[(df2['event_type']=='meeting_date') & (df2['date']<'2011-01-01')]
print(f'  Total meeting dates:     {is_m.sum()}')
print(f'  Pre-2011 meeting dates:  {len(pre2011)}')
print(f'\n  Coverage check (T horizon, meeting dates only):')
for c in sample_cols:
    if c in df2.columns:
        n_all  = df2.loc[is_m, c].notna().sum()
        n_pre  = pre2011[c].notna().sum() if len(pre2011) > 0 else 0
        print(f'    {c:<35}: {n_all}/{is_m.sum()} total  '
              f'| {n_pre}/{len(pre2011)} pre-2011')
