"""
plot_results.py -- Publication-quality figures for dictionary FAVAR results.
OOS metric: oos_rmse_80 (expanding window, 80% training / 20% OOS).

Reads:
    Taylor Rule/outputs/dict_favar_summary_oos80.csv   (level, best per combo, 80% init)
    Taylor Rule/outputs/dict_favar_models.csv          (level, all specs)
    Taylor Rule/outputs/dict_favar_models_delta.csv    (change, all specs)

Outputs (Taylor Rule/outputs/figures/):
    fig1_oos_rmse_level.pdf
    fig2_spec_comparison.pdf
    fig3_is_oos_scatter.pdf
    fig4_level_vs_change.pdf
    fig5_cw_significance.pdf
    fig6_oos_horizon.pdf

Usage:
    python plot_results.py
    python plot_results.py --fig 1 3
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent.parent
TAYLOR_DIR = ROOT / 'Taylor Rule'
OUT_DIR    = TAYLOR_DIR / 'outputs'
FIG_DIR    = OUT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── OOS constants ─────────────────────────────────────────────────────────────
OOS_COL        = 'oos_rmse_80'          # primary metric column in models CSVs
BASELINE_LEVEL = 0.469204               # 4b_sent_sd at oos_rmse_80, level
BASELINE_CHANGE= 0.348585               # 4b_sent_sd at oos_rmse_80, change
BASELINE_R2    = 0.9716                 # baseline adj_r2 (level)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif'],
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   9,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linewidth':    0.5,
})

C = {
    'sharpe_wc':  '#1f77b4',
    'sharpe_zs':  '#aec7e8',
    'gardner_wc': '#d62728',
    'gardner_zs': '#ff9896',
    'baseline':   '#2ca02c',
}

DOC_LABELS = {
    'statements':     'Statements',
    'minutes':        'Minutes',
    'speeches':       'Speeches',
    'press_conferences': 'Press Conf.',
    'statements+minutes+speeches+press_conferences': 'All Docs',
}

SPEC_LABELS = {
    '4a_total_sent':                     '4a: Total sent.',
    '4b_sent_sd':                        '4b: Dispersion',
    '4b_consensus':                      '4b: Consensus',
    '4b_consensus_matched_interactions': '4b: Cons. x macro',
    '4c_all_topics':                     '4c: All topics',
    '4d_topic_pca':                      '4d: Topic PCA',
    '4e_total_interactions':             '4e: Total x macro',
    '4f_matched_interactions':           '4f: Matched x macro',
    '4g_novelty':                        '4g: Novelty',
}

DPI = 300


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_summary_oos80() -> pd.DataFrame:
    p = OUT_DIR / 'dict_favar_summary_oos80.csv'
    if not p.exists():
        print('  [WARN] dict_favar_summary_oos80.csv not found.')
        return pd.DataFrame()
    return pd.read_csv(p)


def _read_models_csv(fname: str) -> pd.DataFrame:
    """Read a models CSV that may have 14 or 16 columns per row."""
    p = OUT_DIR / fname
    if not p.exists():
        print(f'  [WARN] {fname} not found.')
        return pd.DataFrame()
    with open(p, encoding='utf-8') as fh:
        header_line = fh.readline().strip()
    base_cols = header_line.split(',')
    extra     = ['spec', 'extreme_val']
    full_cols = base_cols + [c for c in extra if c not in base_cols]
    df = pd.read_csv(p, names=full_cols, skiprows=1,
                     on_bad_lines='skip', engine='python')
    # Ensure spec is populated
    if 'model_label' in df.columns:
        extracted = df['model_label'].str.extract(r'__(\w+)_ev')[0]
        if 'spec' not in df.columns or df['spec'].isna().all():
            df['spec'] = extracted
        else:
            df['spec'] = df['spec'].fillna(extracted)
    # Numeric coercion
    for col in ['n_obs', 'r2', 'adj_r2', 'oos_rmse_60', 'oos_rmse_70',
                'oos_rmse_80', 'oos_rmse_90', 'cw_tstat', 'cw_pval', 'extreme_val']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def load_level_models() -> pd.DataFrame:
    return _read_models_csv('dict_favar_models.csv')


def load_change_models() -> pd.DataFrame:
    return _read_models_csv('dict_favar_models_delta.csv')


def best_per_combo(models: pd.DataFrame, oos_col: str = OOS_COL,
                   excl_specs=None) -> pd.DataFrame:
    """Return the best-spec row per combo, ranked by oos_col ascending."""
    if models.empty:
        return pd.DataFrame()
    if excl_specs is None:
        excl_specs = ['4b_sent_sd', '4b_sent_var', '4b_sd_var_joint']
    df = models[~models['spec'].isin(excl_specs)].dropna(subset=[oos_col]).copy()
    df['combo'] = df['model_label'].str.extract(r'^(.+?)__')[0]
    return (df.sort_values(oos_col)
              .groupby('combo', sort=False)
              .first()
              .reset_index()
              .sort_values(oos_col))


def row_colour(row) -> str:
    d = str(row.get('dict', '')).lower()
    n = str(row.get('norm', ''))[:2]
    key = f'{d}_{n}'
    return C.get(key, '#888888')


def combo_colour(model_label: str) -> str:
    lbl = str(model_label)
    if 'Sharpe' in lbl and 'zscore' in lbl:   return C['sharpe_zs']
    if 'Sharpe' in lbl:                         return C['sharpe_wc']
    if 'Gardner' in lbl and 'zscore' in lbl:   return C['gardner_zs']
    return C['gardner_wc']


# ── Fig 1: OOS RMSE bar chart (level) ─────────────────────────────────────────

def fig1_oos_rmse_level():
    print('  Figure 1: OOS RMSE level regression (80% init)')
    summary = load_summary_oos80()
    if summary.empty:
        return

    df = summary.sort_values('best_oos_rmse').reset_index(drop=True)
    colours = [row_colour(r) for _, r in df.iterrows()]

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df['best_oos_rmse'], color=colours,
            edgecolor='white', linewidth=0.5, height=0.72)

    ax.axvline(BASELINE_LEVEL, color=C['baseline'], linewidth=1.6,
               linestyle='--', zorder=5,
               label=f'Baseline (no sentiment) = {BASELINE_LEVEL:.3f}')

    for i, (_, row) in enumerate(df.iterrows()):
        rmse = row['best_oos_rmse']
        pct  = (rmse - BASELINE_LEVEL) / BASELINE_LEVEL * 100
        pct_s = f'{pct:+.1f}%'
        sig   = ''
        p = row.get('best_cw_pval', np.nan)
        if pd.notna(p):
            if p < 0.001: sig = '***'
            elif p < 0.01: sig = '**'
            elif p < 0.05: sig = '*'
            elif p < 0.10: sig = '+'
        colour_txt = '#222222' if rmse <= BASELINE_LEVEL else '#888888'
        ax.text(rmse + 0.004, i,
                f'{rmse:.3f}  {pct_s}{sig}',
                va='center', ha='left', fontsize=8, color=colour_txt)

    # Y labels
    labels = []
    for _, r in df.iterrows():
        doc = DOC_LABELS.get(r['doc_types'], r['doc_types'])
        labels.append(f"{r['dict']} {r['norm']}\n({doc})")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()

    ax.set_xlabel('OOS RMSE  (expanding window, 80% training / 20% OOS)', fontsize=10)
    ax.set_title('Dictionary FAVAR -- OOS RMSE by Combo\n'
                 'Level Regression: Effective Rate (t+1)',
                 fontsize=11, fontweight='bold', pad=12)

    legend_elements = [
        mpatches.Patch(color=C['sharpe_wc'],  label='Sharpe -- wordcount'),
        mpatches.Patch(color=C['sharpe_zs'],  label='Sharpe -- zscore'),
        mpatches.Patch(color=C['gardner_wc'], label='Gardner -- wordcount'),
        mpatches.Patch(color=C['gardner_zs'], label='Gardner -- zscore'),
        Line2D([0], [0], color=C['baseline'], linestyle='--', linewidth=1.6,
               label=f'Baseline = {BASELINE_LEVEL:.3f}'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8.5,
              framealpha=0.92, edgecolor='#cccccc')
    ax.set_xlim(0, df['best_oos_rmse'].max() * 1.22)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig1_oos_rmse_level.pdf', dpi=DPI)
    plt.close(fig)
    print('    Saved: fig1_oos_rmse_level.pdf')


# ── Fig 2: Spec breakdown (top 4 combos, level) ───────────────────────────────

def fig2_spec_comparison():
    print('  Figure 2: Spec-level OOS RMSE (80% init)')
    models = load_level_models()
    if models.empty:
        return

    combos_to_show = [
        ('Gardner', 'statements', 'wordcount', C['gardner_wc'], 'Gardner / Stmt / wc'),
        ('Gardner', 'statements_minutes_speeches_press_conferences', 'wordcount',
         '#8B0000', 'Gardner / AllDocs / wc'),
        ('Sharpe',  'statements', 'wordcount', C['sharpe_wc'],  'Sharpe / Stmt / wc'),
        ('Gardner', 'statements', 'zscore',    C['gardner_zs'], 'Gardner / Stmt / zs'),
    ]

    SPECS = ['4a_total_sent', '4b_consensus', '4c_all_topics',
             '4d_topic_pca', '4e_total_interactions',
             '4f_matched_interactions', '4g_novelty']

    fig, ax = plt.subplots(figsize=(10, 5))
    n_specs  = len(SPECS)
    n_combos = len(combos_to_show)
    width    = 0.18
    x        = np.arange(n_specs)

    for ci, (dict_name, doc, norm, colour, label) in enumerate(combos_to_show):
        prefix = f'{dict_name}_{doc}_{norm}__'
        sub = models[models['model_label'].str.startswith(prefix, na=False)].copy()
        if sub.empty:
            continue
        if 'extreme_val' in sub.columns:
            sub = sub[sub['extreme_val'] == 1.0]
        rmses = []
        for spec in SPECS:
            row = sub[sub['spec'] == spec]
            rmses.append(float(row[OOS_COL].iloc[0]) if not row.empty else np.nan)
        offset = (ci - n_combos / 2 + 0.5) * width
        ax.bar(x + offset, rmses, width=width * 0.9,
               color=colour, edgecolor='white', linewidth=0.4, label=label)

    ax.axhline(BASELINE_LEVEL, color=C['baseline'], linewidth=1.6,
               linestyle='--', label=f'Baseline = {BASELINE_LEVEL:.3f}', zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([SPEC_LABELS.get(s, s) for s in SPECS],
                       rotation=22, ha='right', fontsize=8.5)
    ax.set_ylabel(f'OOS RMSE (80% init)', fontsize=10)
    ax.set_title('OOS RMSE by Specification -- Top Combos\n'
                 'Level Regression',
                 fontsize=11, fontweight='bold', pad=12)
    ax.legend(fontsize=8.5, framealpha=0.92, edgecolor='#cccccc', loc='upper right')
    ax.set_ylim(0.3, None)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig2_spec_comparison.pdf', dpi=DPI)
    plt.close(fig)
    print('    Saved: fig2_spec_comparison.pdf')


# ── Fig 3: IS adj-R2 vs OOS RMSE scatter ──────────────────────────────────────

def fig3_is_oos_scatter():
    print('  Figure 3: IS adj-R2 vs OOS RMSE scatter (80% init)')
    models = load_level_models()
    if models.empty:
        return

    excl = ['4b_sent_sd', '4b_sent_var', '4b_sd_var_joint']
    if 'extreme_val' in models.columns:
        df = models[(models['extreme_val'] == 1.0) & ~models['spec'].isin(excl)]
    else:
        df = models[~models['spec'].isin(excl)]
    df = df.dropna(subset=['adj_r2', OOS_COL])

    colours = df['model_label'].apply(combo_colour)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df['adj_r2'], df[OOS_COL],
               c=colours, alpha=0.5, s=28, edgecolors='none', zorder=3)

    ax.axhline(BASELINE_LEVEL, color=C['baseline'], linewidth=1.2, linestyle='--',
               alpha=0.7, label=f'Baseline RMSE = {BASELINE_LEVEL:.3f}')
    ax.axvline(BASELINE_R2,   color=C['baseline'], linewidth=1.2, linestyle=':',
               alpha=0.7, label=f'Baseline adj R2 = {BASELINE_R2:.3f}')

    # Label best and worst
    for idx, ha, va in [(df[OOS_COL].idxmin(), 'left', 'top'),
                        (df[OOS_COL].idxmax(), 'right', 'bottom')]:
        row = df.loc[idx]
        lbl = SPEC_LABELS.get(row.get('spec', ''), str(row.get('spec', '')))
        ax.annotate(lbl, xy=(row['adj_r2'], row[OOS_COL]),
                    xytext=(5 if ha == 'left' else -5, -5 if va == 'top' else 5),
                    textcoords='offset points', fontsize=7.5, ha=ha, va=va,
                    arrowprops=dict(arrowstyle='->', color='#444', lw=0.8))

    ax.set_xlabel('In-Sample Adjusted R2', fontsize=10)
    ax.set_ylabel('OOS RMSE (80% init)  [lower = better]', fontsize=10)
    ax.set_title('In-Sample Fit vs Out-of-Sample Accuracy\n'
                 'Level Regression -- all specs, ev=1.0',
                 fontsize=11, fontweight='bold', pad=12)

    # Shade better-than-baseline quadrant
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    ax.fill_betweenx([ylim[0], BASELINE_LEVEL], BASELINE_R2, xlim[1],
                     alpha=0.06, color='green', zorder=0)
    ax.text((BASELINE_R2 + xlim[1]) * 0.5, (ylim[0] + BASELINE_LEVEL) * 0.5,
            'Better than\nbaseline', ha='center', va='center',
            fontsize=7.5, color='green', alpha=0.5, style='italic')

    legend_elements = [
        mpatches.Patch(color=C['sharpe_wc'],  label='Sharpe -- wordcount'),
        mpatches.Patch(color=C['sharpe_zs'],  label='Sharpe -- zscore'),
        mpatches.Patch(color=C['gardner_wc'], label='Gardner -- wordcount'),
        mpatches.Patch(color=C['gardner_zs'], label='Gardner -- zscore'),
        Line2D([0], [0], color=C['baseline'], linestyle='--', label='Baseline RMSE'),
        Line2D([0], [0], color=C['baseline'], linestyle=':', label='Baseline adj R2'),
    ]
    ax.legend(handles=legend_elements, fontsize=8.5, loc='upper left',
              framealpha=0.92, edgecolor='#cccccc')

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig3_is_oos_scatter.pdf', dpi=DPI)
    plt.close(fig)
    print('    Saved: fig3_is_oos_scatter.pdf')


# ── Fig 4: Level vs change comparison (80% init) ──────────────────────────────

def fig4_level_vs_change():
    print('  Figure 4: Level vs change OOS improvement (80% init)')
    sum_lvl = load_summary_oos80()
    dlt     = load_change_models()
    if sum_lvl.empty or dlt.empty:
        return

    # Level: pct improvement per combo
    sum_lvl = sum_lvl.copy()
    sum_lvl['pct_lvl'] = (BASELINE_LEVEL - sum_lvl['best_oos_rmse']) / BASELINE_LEVEL * 100
    sum_lvl['combo_key'] = (sum_lvl['dict'] + '_' +
                             sum_lvl['doc_types'].str.replace('+', '_').str.replace(',', '_') +
                             '_' + sum_lvl['norm'])

    # Change: best per combo at oos_rmse_80
    chg_best = best_per_combo(dlt, OOS_COL)
    chg_best['pct_chg'] = (BASELINE_CHANGE - chg_best[OOS_COL]) / BASELINE_CHANGE * 100

    # Merge
    merged = sum_lvl[['combo_key', 'pct_lvl', 'dict', 'norm']].merge(
        chg_best[['combo', 'pct_chg']].rename(columns={'combo': 'combo_key'}),
        on='combo_key', how='inner'
    )
    if merged.empty:
        print('    No matched combos — skipping.')
        return

    colours = [row_colour(r) for _, r in merged.iterrows()]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(merged['pct_lvl'], merged['pct_chg'],
               c=colours, s=60, alpha=0.85, edgecolors='white', linewidth=0.5, zorder=3)

    ax.axhline(0, color='#555', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axvline(0, color='#555', linewidth=0.8, linestyle='--', alpha=0.5)

    ax.set_xlabel('Level regression: % RMSE improvement vs baseline (positive = better)',
                  fontsize=10)
    ax.set_ylabel('Change regression: % RMSE improvement vs baseline (positive = better)',
                  fontsize=10)
    ax.set_title('Level vs Change OOS Improvement (80% init)\n'
                 'Each point = one dictionary x doc-type x norm combo',
                 fontsize=11, fontweight='bold', pad=12)

    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    qk = dict(ha='center', va='center', fontsize=7.5, alpha=0.4, style='italic')
    ax.text(xlim[1]*0.55, ylim[1]*0.75, 'Helps both',  color='green',  **qk)
    ax.text(xlim[0]*0.55, ylim[0]*0.75, 'Hurts both',  color='red',    **qk)
    ax.text(xlim[1]*0.55, ylim[0]*0.75, 'Level only',  color='#cc8800',**qk)
    ax.text(xlim[0]*0.55, ylim[1]*0.75, 'Change only', color='#0055cc',**qk)

    legend_elements = [
        mpatches.Patch(color=C['sharpe_wc'],  label='Sharpe -- wordcount'),
        mpatches.Patch(color=C['sharpe_zs'],  label='Sharpe -- zscore'),
        mpatches.Patch(color=C['gardner_wc'], label='Gardner -- wordcount'),
        mpatches.Patch(color=C['gardner_zs'], label='Gardner -- zscore'),
    ]
    ax.legend(handles=legend_elements, fontsize=8.5, loc='upper right',
              framealpha=0.92, edgecolor='#cccccc')

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig4_level_vs_change.pdf', dpi=DPI)
    plt.close(fig)
    print('    Saved: fig4_level_vs_change.pdf')


# ── Fig 5: CW significance bar chart (level, 80% init) ────────────────────────

def fig5_cw_significance():
    print('  Figure 5: Clark-West significance (80% init)')
    summary = load_summary_oos80()
    if summary.empty:
        return

    df = summary.sort_values('best_oos_rmse').reset_index(drop=True)

    cmap = {-1: '#ff6b6b', 0: '#e0e0e0', 1: '#b3d9ff', 2: '#66b2ff',
            3: '#1a7fd4', 4: '#003d80'}
    sig_labels = {-1: '(worse)', 0: 'n.s.', 1: '+', 2: '*', 3: '**', 4: '***'}

    def sig(row):
        p = row.get('best_cw_pval', np.nan)
        t = row.get('best_cw_tstat', np.nan)
        if pd.isna(p):
            return -1 if (pd.notna(t) and t < 0) else 0
        if p < 0.001: return 4
        if p < 0.01:  return 3
        if p < 0.05:  return 2
        if p < 0.10:  return 1
        return 0

    df['sig'] = df.apply(sig, axis=1)
    bar_colours = [cmap[s] for s in df['sig']]

    fig, ax = plt.subplots(figsize=(9, 6))
    n = len(df)
    ax.barh(range(n), df['best_oos_rmse'], color=bar_colours,
            edgecolor='white', linewidth=0.5, height=0.75)
    ax.axvline(BASELINE_LEVEL, color=C['baseline'], linewidth=1.6, linestyle='--',
               zorder=5, label=f'Baseline = {BASELINE_LEVEL:.3f}')

    for i, (_, row) in enumerate(df.iterrows()):
        rmse = row['best_oos_rmse']
        s    = row['sig']
        sl   = sig_labels[s]
        t_s  = (f"t={row['best_cw_tstat']:.2f}"
                if pd.notna(row.get('best_cw_tstat')) else '')
        txt_col = '#cc0000' if s == -1 else '#222222'
        ax.text(rmse + 0.004, i,
                f"{rmse:.3f}  {sl}  {t_s}",
                va='center', ha='left', fontsize=7.5, color=txt_col)

    labels = []
    for _, r in df.iterrows():
        doc = DOC_LABELS.get(r['doc_types'], r['doc_types'])
        labels.append(f"{r['dict']} {r['norm']}\n({doc})")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel('OOS RMSE (80% init)', fontsize=10)
    ax.set_title('Clark-West Significance -- Level Regression (80% init)\n'
                 'Sorted by OOS RMSE (best to worst)',
                 fontsize=11, fontweight='bold', pad=12)

    legend_elements = [
        mpatches.Patch(color=cmap[4],  label='p < 0.001 (***)'),
        mpatches.Patch(color=cmap[3],  label='p < 0.01  (**)'),
        mpatches.Patch(color=cmap[2],  label='p < 0.05  (*)'),
        mpatches.Patch(color=cmap[1],  label='p < 0.10  (+)'),
        mpatches.Patch(color=cmap[0],  label='Not significant'),
        mpatches.Patch(color=cmap[-1], label='Worse than baseline'),
        Line2D([0], [0], color=C['baseline'], linestyle='--', linewidth=1.6,
               label=f'Baseline = {BASELINE_LEVEL:.3f}'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8,
              framealpha=0.92, edgecolor='#cccccc')
    ax.set_xlim(0, df['best_oos_rmse'].max() * 1.28)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig5_cw_significance.pdf', dpi=DPI)
    plt.close(fig)
    print('    Saved: fig5_cw_significance.pdf')


# ── Fig 6: OOS RMSE across horizons (level, top combos) ───────────────────────

def fig6_oos_horizon():
    print('  Figure 6: OOS RMSE across training thresholds (level)')
    models = load_level_models()
    if models.empty:
        return

    top_combos = [
        ('Gardner_statements_minutes_speeches_press_conferences_wordcount',
         '#8B0000', 'Gardner / All Docs / wc  (best OOS)'),
        ('Gardner_statements_wordcount',   C['gardner_wc'], 'Gardner / Stmt / wc'),
        ('Sharpe_statements_wordcount',    C['sharpe_wc'],  'Sharpe / Stmt / wc'),
        ('Sharpe_statements_zscore',       C['sharpe_zs'],  'Sharpe / Stmt / zs'),
    ]
    BEST_SPEC = '4e_total_interactions'
    horizons  = [60, 70, 80, 90]

    # Baseline across horizons from 4b_sent_sd
    baselines = {}
    for h in horizons:
        col  = f'oos_rmse_{h}'
        mask = models['model_label'].str.contains('4b_sent_sd', na=False)
        vals = models.loc[mask, col].dropna() if col in models.columns else pd.Series()
        baselines[h] = float(vals.iloc[0]) if not vals.empty else np.nan

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.array(horizons)

    for prefix, colour, label in top_combos:
        combo_mask = models['model_label'].str.startswith(prefix + '__', na=False)
        spec_mask  = models['spec'] == BEST_SPEC
        if 'extreme_val' in models.columns:
            ev_mask = models['extreme_val'] == 1.0
        else:
            ev_mask = pd.Series(True, index=models.index)
        sub = models[combo_mask & spec_mask & ev_mask]
        if sub.empty:
            continue
        row = sub.iloc[0]
        y = [row.get(f'oos_rmse_{h}', np.nan) for h in horizons]
        ax.plot(x, y, color=colour, linewidth=1.8, marker='o', markersize=5,
                label=label, zorder=3)

    y_base = [baselines.get(h, np.nan) for h in horizons]
    ax.plot(x, y_base, color=C['baseline'], linewidth=1.5, linestyle='--',
            marker='s', markersize=4, label='Baseline (no sentiment)', zorder=2)

    # Shade the 80% threshold
    ax.axvline(80, color='#888', linewidth=0.8, linestyle=':', alpha=0.7)
    ax.text(80.5, ax.get_ylim()[1]*0.98 if ax.get_ylim()[1] else 0.7,
            'Primary\nhorizon', fontsize=7.5, color='#555', va='top')

    ax.set_xlabel('Training fraction (% of sample)', fontsize=10)
    ax.set_ylabel('OOS RMSE', fontsize=10)
    ax.set_title('OOS RMSE by Training-Fraction Threshold -- Spec 4e\n'
                 'Level Regression (top 4 combos)',
                 fontsize=11, fontweight='bold', pad=12)
    ax.set_xticks(horizons)
    ax.set_xticklabels([f'{h}%' for h in horizons])
    ax.legend(fontsize=8.5, framealpha=0.92, edgecolor='#cccccc')
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig6_oos_horizon.pdf', dpi=DPI)
    plt.close(fig)
    print('    Saved: fig6_oos_horizon.pdf')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Plot dictionary FAVAR results (80% OOS init)')
    parser.add_argument('--fig', nargs='*', type=int, default=None,
                        help='Figures to produce, e.g. --fig 1 3 5. Default: all.')
    args = parser.parse_args()

    fig_fns = {
        1: fig1_oos_rmse_level,
        2: fig2_spec_comparison,
        3: fig3_is_oos_scatter,
        4: fig4_level_vs_change,
        5: fig5_cw_significance,
        6: fig6_oos_horizon,
    }
    to_run = sorted(args.fig) if args.fig else sorted(fig_fns.keys())
    print(f'\nProducing {len(to_run)} figure(s) -> {FIG_DIR}\n')
    for k in to_run:
        if k in fig_fns:
            fig_fns[k]()
        else:
            print(f'  [WARN] Figure {k} not defined.')
    print(f'\nDone. Figures saved to: {FIG_DIR}')


if __name__ == '__main__':
    main()
