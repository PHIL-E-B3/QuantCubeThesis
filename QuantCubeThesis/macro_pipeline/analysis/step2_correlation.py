"""
step2_correlation.py — Correlation of LLM sentiment vs Gardner/Sharpe.
"""

import sys
import warnings
from pathlib import Path

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUTS_DIR, INTER_DIR, NLP_DIR, COLOUR_PALETTE
from utils  import save_fig

warnings.filterwarnings('ignore')


def _load_dict_scores(dict_name: str, doc_types: str, norm: str) -> pd.DataFrame:
    fname = NLP_DIR / f'{dict_name}_{doc_types}_{norm}_nlp.csv'
    if not fname.exists():
        print(f'  ⚠️  Dictionary file not found: {fname}')
        return pd.DataFrame()
    df = pd.read_csv(fname, parse_dates=['meeting_date'])
    return df


def _corr_pair(s1: pd.Series, s2: pd.Series, label: str) -> dict:
    df = pd.concat([s1, s2], axis=1).dropna()
    if len(df) < 10:
        return {}
    a, b = df.iloc[:,0], df.iloc[:,1]
    pr, pp = pearsonr(a, b)
    sr, sp = spearmanr(a, b)
    return {
        'pair':      label,
        'n':         len(df),
        'pearson_r': round(pr, 4),
        'pearson_p': round(pp, 4),
        'spearman_r':round(sr, 4),
        'spearman_p':round(sp, 4),
    }


def run_correlation(df_macro: pd.DataFrame,
                    dict_doc_type: str = 'statements',
                    norm_variants: list = None) -> pd.DataFrame:
    """
    Correlate LLM sent_total / per-topic columns against Gardner and Sharpe
    dictionary scores across normalization variants.

    Parameters
    ----------
    df_macro      : merged macro + LLM sentiment DataFrame from Step 0
    dict_doc_type : which doc_type combo to use for dictionary scores
    norm_variants : list of normalizations to compare; default ['wordcount','zscore']
    """
    if norm_variants is None:
        norm_variants = ['wordcount', 'zscore']

    df = df_macro.copy()
    df['date'] = pd.to_datetime(df['date'])

    results = []

    for norm in norm_variants:
        gard = _load_dict_scores('Gardner', dict_doc_type, norm)
        shp  = _load_dict_scores('Sharpe',  dict_doc_type, norm)

        if not gard.empty:
            gard = gard.groupby('meeting_date')[
                [c for c in gard.columns if c.startswith('gardner_')]
            ].sum().reset_index().rename(columns={'meeting_date':'date'})
            df = pd.merge_asof(
                df.sort_values('date'), gard.sort_values('date'),
                on='date', direction='backward',
                suffixes=('', f'_gard_{norm}')
            )

        if not shp.empty:
            shp = shp.groupby('meeting_date')[
                [c for c in shp.columns if c.startswith('sharpe_')]
            ].sum().reset_index().rename(columns={'meeting_date':'date'})
            df = pd.merge_asof(
                df.sort_values('date'), shp.sort_values('date'),
                on='date', direction='backward',
                suffixes=('', f'_shp_{norm}')
            )

        # Correlation pairs
        sent_total = df.get('sent_total', pd.Series(dtype=float))
        for g_col in [c for c in df.columns if 'gardner_total' in c]:
            r = _corr_pair(sent_total, df[g_col],
                           f'sent_total vs {g_col} [{norm}]')
            if r: results.append(r)
        for s_col in [c for c in df.columns if 'sharpe_net' in c]:
            r = _corr_pair(sent_total, df[s_col],
                           f'sent_total vs {s_col} [{norm}]')
            if r: results.append(r)

        # Gardner vs Sharpe benchmark
        g_cols = [c for c in df.columns if 'gardner_total' in c]
        s_cols = [c for c in df.columns if 'sharpe_net' in c]
        if g_cols and s_cols:
            r = _corr_pair(df[g_cols[0]], df[s_cols[0]],
                           f'Gardner vs Sharpe [{norm}]')
            if r: results.append(r)

    summary = pd.DataFrame(results)
    summary.to_csv(OUTPUTS_DIR / 'correlation_summary.csv', index=False)
    print('\n  Correlation summary:')
    print(summary[['pair','pearson_r','pearson_p','spearman_r']].to_string(index=False))

    # ── Heatmap of all numeric columns ────────────────────────────────────────
    heat_cols = (
        [c for c in df.columns if c.startswith('sent_')] +
        [c for c in df.columns if 'gardner_' in c or 'sharpe_' in c]
    )
    heat_cols = [c for c in heat_cols if df[c].dtype in [np.float64, np.int64]]
    corr_mat  = df[heat_cols].corr(method='pearson')

    fig, ax = plt.subplots(figsize=(max(8, len(heat_cols)*0.5),
                                    max(6, len(heat_cols)*0.4)))
    sns.heatmap(corr_mat, ax=ax, cmap='coolwarm', center=0,
                annot=len(heat_cols) <= 15, fmt='.2f', linewidths=0.5)
    ax.set_title('Pairwise Pearson Correlations: LLM sentiment vs Dictionaries')
    ax.tick_params(labelsize=7)
    save_fig(fig, OUTPUTS_DIR / 'correlation_heatmap.png')

    # ── Scatter plots ──────────────────────────────────────────────────────────
    scatter_pairs = []
    for norm in norm_variants:
        g_col = next((c for c in df.columns if f'gardner_total' in c), None)
        s_col = next((c for c in df.columns if f'sharpe_net'    in c), None)
        if g_col: scatter_pairs.append(('sent_total', g_col, f'LLM vs Gardner [{norm}]'))
        if s_col: scatter_pairs.append(('sent_total', s_col, f'LLM vs Sharpe [{norm}]'))
        if g_col and s_col:
            scatter_pairs.append((g_col, s_col, f'Gardner vs Sharpe [{norm}]'))

    if scatter_pairs:
        n = len(scatter_pairs)
        fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
        if n == 1:
            axes = [axes]
        for ax, (c1, c2, lbl) in zip(axes, scatter_pairs):
            sub = df[[c1, c2]].dropna()
            if sub.empty:
                continue
            ax.scatter(sub[c1], sub[c2], alpha=0.5, s=15, color=COLOUR_PALETTE[0])
            m, b = np.polyfit(sub[c1], sub[c2], 1)
            xr = np.linspace(sub[c1].min(), sub[c1].max(), 50)
            ax.plot(xr, m*xr+b, color=COLOUR_PALETTE[1], linewidth=1.5)
            ax.set_xlabel(c1, fontsize=8)
            ax.set_ylabel(c2, fontsize=8)
            ax.set_title(lbl, fontsize=9)
        save_fig(fig, OUTPUTS_DIR / 'correlation_scatter.png')

    return summary
