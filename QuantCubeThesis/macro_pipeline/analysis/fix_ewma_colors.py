"""
fix_ewma_colors.py
Regenerate ewma_*.png with colours that match the combined_*.png plots:
  LLM      -> red   (#d62728)
  Gardner  -> green (#2ca02c)
  Sharpe   -> cyan  (#17becf)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

ROOT      = Path(__file__).parent.parent.parent
INTER_CSV = ROOT / 'Taylor Rule' / 'outputs' / 'intermediate' / 'step0_merged_to_macro.csv'
NLP_DIR   = ROOT / 'Taylor Rule' / 'nlp_output'
OUT_DIR   = ROOT / 'Taylor Rule' / 'outputs'

# ── Method colours (must match combined_*.png) ────────────────────────────────
COLOURS = {
    'LLM':     '#d62728',   # red
    'Gardner': '#2ca02c',   # green
    'Sharpe':  '#17becf',   # cyan
}

plt.rcParams.update({
    'font.family':     'serif',
    'font.serif':      ['Times New Roman', 'DejaVu Serif'],
    'font.size':       10,
    'axes.titlesize':  11,
    'axes.labelsize':  10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':       True,
    'grid.alpha':      0.3,
    'grid.linewidth':  0.5,
})


def zscore_series(s: pd.Series) -> pd.Series:
    vals = s.dropna().values.reshape(-1, 1)
    sc = StandardScaler()
    z = sc.fit_transform(vals).ravel()
    out = s.copy()
    out[s.notna()] = z
    return out


def load_dict_total(dict_name: str, col: str) -> pd.Series:
    """Load all-docs wordcount NLP output, aggregate by meeting_date, return total col."""
    fname = NLP_DIR / f'{dict_name}_statements_minutes_speeches_press_conferences_wordcount_nlp.csv'
    df = pd.read_csv(fname, parse_dates=['meeting_date'])
    agg = df.groupby('meeting_date')[col].sum().reset_index()
    agg.columns = ['date', col]
    return agg.set_index('date')[col]


def plot_ewma(dates, raw, ewma3, ewma5, title, colour, out_path):
    fig, ax = plt.subplots(figsize=(14, 4))

    light = matplotlib.colors.to_rgba(colour, alpha=0.30)

    # Raw — very thin, semi-transparent
    ax.plot(dates, raw, color=colour, linewidth=0.6, alpha=0.35, label='Raw')

    # Shaded band between ewma3 and ewma5
    ax.fill_between(dates, ewma3, ewma5, color=colour, alpha=0.15)

    # EWMA lines
    ax.plot(dates, ewma3, color=colour, linewidth=1.2, linestyle='--',
            label='EWMA span=3')
    ax.plot(dates, ewma5, color=colour, linewidth=2.2, linestyle='-',
            label='EWMA span=5')

    ax.axhline(0, color='#888', linewidth=0.7, linestyle=':')
    ax.set_xlabel('Date')
    ax.set_ylabel('Z-score')
    ax.set_title(title)
    ax.legend(loc='upper left', framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {out_path.name}')


# ── Load base data ────────────────────────────────────────────────────────────
print('Loading merged panel ...')
df = pd.read_csv(INTER_CSV, parse_dates=['date'])
df = df[df['date'] >= '1999-01-01'].sort_values('date').reset_index(drop=True)

# ── LLM sentiment ─────────────────────────────────────────────────────────────
print('Processing LLM ...')
llm_raw  = zscore_series(df.set_index('date')['sent_total'])
llm_raw  = llm_raw.reindex(df['date'].values)
llm_raw.index = df['date']

llm_e3   = llm_raw.ewm(span=3, adjust=False).mean()
llm_e5   = llm_raw.ewm(span=5, adjust=False).mean()

plot_ewma(
    dates   = df['date'],
    raw     = llm_raw.values,
    ewma3   = llm_e3.values,
    ewma5   = llm_e5.values,
    title   = 'LLM Sentiment (sent_total) — Raw vs EWMA (span=3 and span=5)',
    colour  = COLOURS['LLM'],
    out_path= OUT_DIR / 'ewma_sent_total.png',
)

# ── Gardner ────────────────────────────────────────────────────────────────────
print('Processing Gardner ...')
gard_raw_s = load_dict_total('Gardner', 'gardner_total')
gard_merged = pd.merge_asof(
    df[['date']].sort_values('date'),
    gard_raw_s.reset_index().rename(columns={'date': 'date', 'gardner_total': 'g'}),
    on='date', direction='nearest', tolerance=pd.Timedelta('45d')
)
gard_raw = zscore_series(gard_merged.set_index('date')['g'])
gard_raw.index = df['date']

gard_e3 = gard_raw.ewm(span=3, adjust=False).mean()
gard_e5 = gard_raw.ewm(span=5, adjust=False).mean()

plot_ewma(
    dates   = df['date'],
    raw     = gard_raw.values,
    ewma3   = gard_e3.values,
    ewma5   = gard_e5.values,
    title   = 'Gardner Dictionary Sentiment — Raw vs EWMA (span=3 and span=5)',
    colour  = COLOURS['Gardner'],
    out_path= OUT_DIR / 'ewma_Gardner_total.png',
)

# ── Sharpe ─────────────────────────────────────────────────────────────────────
print('Processing Sharpe ...')
shp_raw_s = load_dict_total('Sharpe', 'sharpe_net')
shp_merged = pd.merge_asof(
    df[['date']].sort_values('date'),
    shp_raw_s.reset_index().rename(columns={'date': 'date', 'sharpe_net': 's'}),
    on='date', direction='nearest', tolerance=pd.Timedelta('45d')
)
shp_raw = zscore_series(shp_merged.set_index('date')['s'])
shp_raw.index = df['date']

shp_e3 = shp_raw.ewm(span=3, adjust=False).mean()
shp_e5 = shp_raw.ewm(span=5, adjust=False).mean()

plot_ewma(
    dates   = df['date'],
    raw     = shp_raw.values,
    ewma3   = shp_e3.values,
    ewma5   = shp_e5.values,
    title   = 'Sharpe Dictionary Sentiment — Raw vs EWMA (span=3 and span=5)',
    colour  = COLOURS['Sharpe'],
    out_path= OUT_DIR / 'ewma_Sharpe_net.png',
)

# ── Combined EWMA plots (macro background + EWMA-smoothed sentiment) ──────────

# Macro background lines — same style as the original combined_*.png
MACRO_LINES = [
    ('effective_rate', 'Effective FFR',  '#1f77b4', '-',  1.2),
    ('unemployment_rate', 'Unemployment', '#1f77b4', '--', 0.9),
    ('core_pce_yoy',  'Core PCE YoY',   '#1f77b4', ':',  0.9),
    ('cpi_yoy',       'CPI YoY',        '#9467bd', '--', 0.8),
    ('vix',           'VIX',            '#ff7f0e', '--', 0.9),
]


def plot_combined_ewma(dates, macro_df, sent_ewma5, sent_label, colour, title, out_path):
    fig, ax = plt.subplots(figsize=(14, 4.5))

    n = len(dates)
    for col, label, c, ls, lw in MACRO_LINES:
        if col not in macro_df.columns:
            continue
        s = zscore_series(macro_df[col])
        ax.plot(dates, s.values, color=c, linestyle=ls, linewidth=lw,
                alpha=0.55, label=label)

    # Sentiment EWMA5 — bold, method colour
    ax.fill_between(dates, sent_ewma5, 0, color=colour, alpha=0.12)
    ax.plot(dates, sent_ewma5, color=colour, linewidth=2.4, label=sent_label, zorder=5)

    ax.axhline(0, color='#888', linewidth=0.6, linestyle=':')
    ax.set_xlabel('Date')
    ax.set_ylabel('Z-score')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {out_path.name}')


macro_cols = [c for c, *_ in MACRO_LINES]
macro_sub = df.set_index('date')[macro_cols]

print('\nGenerating combined EWMA plots ...')

plot_combined_ewma(
    dates      = df['date'],
    macro_df   = macro_sub,
    sent_ewma5 = llm_e5.values,
    sent_label = 'LLM Sentiment (EWMA span=5)',
    colour     = COLOURS['LLM'],
    title      = 'Macro Variables + LLM Sentiment (EWMA span=5)  (standardised, 2000-2025, N=207)',
    out_path   = OUT_DIR / 'combined_ewma_sent_total.png',
)

plot_combined_ewma(
    dates      = df['date'],
    macro_df   = macro_sub,
    sent_ewma5 = gard_e5.values,
    sent_label = 'Gardner Dict Sentiment (EWMA span=5)',
    colour     = COLOURS['Gardner'],
    title      = 'Macro Variables + Gardner Dict Sentiment (EWMA span=5)  (standardised, 2000-2025, N=207)',
    out_path   = OUT_DIR / 'combined_ewma_Gardner_total.png',
)

plot_combined_ewma(
    dates      = df['date'],
    macro_df   = macro_sub,
    sent_ewma5 = shp_e5.values,
    sent_label = 'Sharpe Dict Sentiment (EWMA span=5)',
    colour     = COLOURS['Sharpe'],
    title      = 'Macro Variables + Sharpe Dict Sentiment (EWMA span=5)  (standardised, 2000-2025, N=207)',
    out_path   = OUT_DIR / 'combined_ewma_Sharpe_net.png',
)

print('\nDone.')
