"""
step1_stationarity.py — ADF + KPSS stationarity checks on all sent_ columns.
"""

import sys
import warnings
from pathlib import Path

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUTS_DIR, INTER_DIR, COLOUR_PALETTE
from utils  import save_fig

warnings.filterwarnings('ignore')


def run_stationarity(df: pd.DataFrame) -> pd.DataFrame:
    """
    ADF + KPSS tests on all sent_ columns.
    Returns summary DataFrame; saves CSV and plots.
    """
    sent_cols = [c for c in df.columns if c.startswith('sent_')]
    results   = []

    for col in sent_cols:
        series = df[col].dropna()
        if len(series) < 20:
            continue

        # ADF
        try:
            adf_res  = adfuller(series, regression='c', autolag='AIC')
            adf_stat = adf_res[0]
            adf_pval = adf_res[1]
            adf_stat_flag = adf_pval < 0.05
        except Exception:
            adf_stat = adf_pval = np.nan
            adf_stat_flag = None

        # KPSS
        try:
            kpss_res  = kpss(series, regression='c', nlags='auto')
            kpss_stat = kpss_res[0]
            kpss_pval = kpss_res[1]
            kpss_stat_flag = kpss_pval > 0.05  # fail to reject null of stationarity
        except Exception:
            kpss_stat = kpss_pval = np.nan
            kpss_stat_flag = None

        # Conclusion
        if adf_stat_flag is True and kpss_stat_flag is True:
            conclusion = 'stationary'
        elif adf_stat_flag is False and kpss_stat_flag is False:
            conclusion = 'non-stationary'
        else:
            conclusion = 'conflicting'

        results.append({
            'series':          col,
            'adf_stat':        round(adf_stat, 4) if pd.notna(adf_stat) else np.nan,
            'adf_pvalue':      round(adf_pval, 4) if pd.notna(adf_pval) else np.nan,
            'adf_stationary':  adf_stat_flag,
            'kpss_stat':       round(kpss_stat, 4) if pd.notna(kpss_stat) else np.nan,
            'kpss_pvalue':     round(kpss_pval, 4) if pd.notna(kpss_pval) else np.nan,
            'kpss_stationary': kpss_stat_flag,
            'conclusion':      conclusion,
        })

    summary = pd.DataFrame(results)
    summary.to_csv(OUTPUTS_DIR / 'stationarity_results.csv', index=False)
    print(f'\n  Stationarity: {len(summary)} series tested')
    print(summary['conclusion'].value_counts().to_string())

    # ── Plots ─────────────────────────────────────────────────────────────────
    n_per_fig = 6
    chunks    = [sent_cols[i:i+n_per_fig] for i in range(0, len(sent_cols), n_per_fig)]
    for fig_idx, chunk in enumerate(chunks):
        fig, axes = plt.subplots(
            (len(chunk)+1) // 2, 2,
            figsize=(14, 3 * ((len(chunk)+1)//2))
        )
        axes = axes.flatten()
        for i, col in enumerate(chunk):
            is_stat = summary.loc[summary['series']==col, 'conclusion'].values
            is_stat = is_stat[0] if len(is_stat) else 'unknown'
            colour  = COLOUR_PALETTE[1] if is_stat == 'non-stationary' else COLOUR_PALETTE[0]
            series  = df[col].dropna()
            axes[i].plot(series.index, series.values, color=colour, linewidth=0.8)
            axes[i].set_title(col, fontsize=8)
            axes[i].tick_params(labelsize=7)
        for j in range(len(chunk), len(axes)):
            axes[j].set_visible(False)
        save_fig(fig, OUTPUTS_DIR / f'stationarity_plots_{fig_idx+1}.png')

    return summary
