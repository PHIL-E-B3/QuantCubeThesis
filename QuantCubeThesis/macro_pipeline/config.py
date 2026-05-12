"""
config.py — single source of truth for paths and API credentials.
All other scripts in macro_pipeline/ import from here.
"""

from pathlib import Path

# ── Project root (two levels up from this file) ───────────────────────────────
ROOT = Path(__file__).parent.parent

# ── FRED API ──────────────────────────────────────────────────────────────────
FRED_API_KEY  = 'c5b5ec287d025fecd23b73051ee03c84'
VINTAGE_START = '2010-01-01'   # earliest vintage to pull (pre-2010 not needed)

# ── Raw text data ─────────────────────────────────────────────────────────────
STATEMENTS_JSON_DIR = ROOT / 'data' / 'raw' / 'structured_json_statements'
SPEECHES_JSON_DIR   = ROOT / 'data' / 'raw' / 'structured_json_speeches'
MINUTES_JSON_DIR    = ROOT / 'data' / 'raw' / 'structured_json_minutes'

# ── External input files ──────────────────────────────────────────────────────
INPUTS_DIR         = ROOT / 'data' / 'macro_inputs'
JK_SHOCKS_FILE     = INPUTS_DIR / 'fomc_surprises_jk.csv.xlsx'
TW_DISSENTS_FILE   = INPUTS_DIR / 'FOMC_Dissents_Data.xlsx'
BB_FUTURES_FILE    = INPUTS_DIR / 'FedFundsFutures.xlsx'
SPF_DISP_D1_FILE   = INPUTS_DIR / 'Dispersion_1.xlsx'
SPF_DISP_D2_FILE   = INPUTS_DIR / 'Dispersion_2.xlsx'
SPF_DISP_D3_FILE   = INPUTS_DIR / 'Dispersion_3.xlsx'

# ── FRED cache (parquet files, safe to delete and re-fetch) ───────────────────
CACHE_DIR = ROOT / 'data' / 'cache'

# ── Pipeline outputs ──────────────────────────────────────────────────────────
FOMC_CALENDAR_CSV = ROOT / 'data' / 'macro_inputs' / 'fomc_calendar.csv'
MASTER_MACRO_CSV  = ROOT / 'data' / 'macro_inputs' / 'Master_Macro.csv'
MASTER_SEP_CSV    = ROOT / 'data' / 'macro_inputs' / 'Master_Macro_with_SEP.csv'

# ── Speech subdirectories (used by calendar builder) ─────────────────────────
SPEECH_SUBDIRS = ['chair', 'vice_chair', 'others']
