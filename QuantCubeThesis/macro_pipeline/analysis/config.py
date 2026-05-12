"""
analysis/config.py — paths and constants for the FOMC analysis pipeline.
"""

from pathlib import Path
import numpy as np

ROOT        = Path(__file__).parent.parent.parent
TAYLOR_DIR  = ROOT / 'Taylor Rule'
OUTPUTS_DIR = TAYLOR_DIR / 'outputs'
INTER_DIR   = OUTPUTS_DIR / 'intermediate'
NLP_DIR     = TAYLOR_DIR / 'nlp_output'

MASTER_CSV  = ROOT / 'data' / 'macro_inputs' / 'Master_Macro.csv'

STATEMENTS_DIR  = ROOT / 'data' / 'raw' / 'structured_json_statements'
MINUTES_DIR     = ROOT / 'data' / 'raw' / 'structured_json_minutes'
SPEECHES_DIR    = ROOT / 'data' / 'raw' / 'structured_json_speeches'
PRESS_CONF_DIR  = ROOT / 'data' / 'raw' / 'structured_json'

SPEECH_SUBDIRS  = {'chair': 'chair', 'vice_chair': 'vice_chair', 'others': 'others'}

WARNINGS_LOG    = OUTPUTS_DIR / 'warnings.log'
DATE_ALIGN_LOG  = OUTPUTS_DIR / 'date_alignment_warnings.txt'

RANDOM_SEED     = 42
FIG_DPI         = 150
OOS_MIN_FRAC    = 0.60    # minimum fraction of data for initial OOS window
PCA_VAR_MACRO   = 0.90    # cumulative variance threshold for macro PCA
PCA_VAR_SENT    = 0.85    # cumulative variance threshold for sentiment PCA
MIN_MASK_OBS    = 30      # minimum observations to run a masked regression

MACRO_REGRESSORS = [
    'inflation_dev_from_target',
    'unemployment_gap',
    'implied_ffr',
    'effective_rate',
    'gdp',
    'vix',
]
TARGET_COL  = 'effective_rate'
TARGET_NEXT = 'target_next'   # column name after forward shift

TOPICS = [
    'monetary_policy', 'inflation', 'unemployment',
    'economic_activity', 'financial_conditions', 'macro',
]
EXCLUDE_TOPICS = {'no_topic', 'boilerplate'}

COLOUR_PALETTE = [
    '#2196F3', '#F44336', '#4CAF50', '#FF9800',
    '#9C27B0', '#00BCD4', '#795548', '#607D8B',
]

# Ensure output dirs exist at import time
for d in [OUTPUTS_DIR, INTER_DIR, NLP_DIR]:
    d.mkdir(parents=True, exist_ok=True)
