"""
step6_scaling.py — Scaling & normalisation exploration (6a, 6b, 6c).
"""

import sys
import warnings
from pathlib import Path

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUTS_DIR, INTER_DIR, NLP_DIR, TARGET_NEXT, COLOUR_PALETTE
from utils  import fit_ols, save_model_row, print_metrics, log_warn, save_fig

warnings.filterwarnings('ignore')


def run_scaling(df: pd.DataFrame, df_sentences,
                best_result: dict) -> dict:
    """
    6a: extreme_val sweep | 6b: aggregation function | 6c: norm strategy.
    Returns dict of results.
    """
    from step0_aggregate import aggregate_to_document, rescale_sen
    from step0_aggregate import merge_to_macro

    factor_df = best_result.get('_factor_df', pd.DataFrame())
    results   = {}

    # ── 6a: Extreme val sweep ─────────────────────────────────────────────────
    print('\n  6a. Extreme sentiment rescaling')
    extreme_vals  = [1.0, 1.5, 2.0, 2.5, 3.0]
    ev_adj_r2     = []

    for ev in extreme_vals:
        doc_ev   = aggregate_to_document(df_sentences, agg_func='sum',
                                         rescale_extreme=ev)
        df_ev    = merge_to_macro(doc_ev)
        target   = df_ev[TARGET_NEXT]
        fdf_sub  = factor_df.loc[factor_df.index.intersection(df_ev.index)]
        extra    = [c for c in ['sent_total'] if c in df_ev.columns]
        X        = fdf_sub.join(df_ev[extra], how='left').dropna()
        y        = target.loc[X.index].dropna()
        X        = X.loc[y.index]
        r = fit_ols(y, X, label=f'6a_ev{ev}')
        if r:
            print_metrics(r)
            save_model_row(r)
            ev_adj_r2.append((ev, r['adj_r2']))
            results[f'6a_ev{ev}'] = r

    if ev_adj_r2:
        fig, ax = plt.subplots(figsize=(6, 4))
        evs, adjs = zip(*ev_adj_r2)
        ax.plot(evs, adjs, marker='o', color=COLOUR_PALETTE[0], linewidth=2)
        ax.set_xlabel('extreme_val (|2| mapped to)')
        ax.set_ylabel('Adjusted R²')
        ax.set_title('Adj R² vs extreme_val for |sen|=2 rescaling')
        ax.axvline(2.0, linestyle='--', color='grey', linewidth=0.8,
                   label='baseline (ev=2)')
        ax.legend(fontsize=8)
        save_fig(fig, OUTPUTS_DIR / 'scaling_extreme_val.png')

    # ── 6b: Aggregation function ──────────────────────────────────────────────
    print('\n  6b. Aggregation function comparison')
    agg_rows = []
    for agg_func in ['sum', 'power', 'log']:
        doc_agg = aggregate_to_document(df_sentences, agg_func=agg_func)
        df_agg  = merge_to_macro(doc_agg)
        target  = df_agg[TARGET_NEXT]
        fdf_sub = factor_df.loc[factor_df.index.intersection(df_agg.index)]
        extra   = [c for c in ['sent_total'] if c in df_agg.columns]
        X       = fdf_sub.join(df_agg[extra], how='left').dropna()
        y       = target.loc[X.index].dropna()
        X       = X.loc[y.index]
        r = fit_ols(y, X, label=f'6b_agg_{agg_func}')
        if r:
            print_metrics(r)
            save_model_row(r)
            agg_rows.append({'agg_func': agg_func, 'adj_r2': r['adj_r2'],
                              'aic': r['aic'], 'bic': r['bic']})
            results[f'6b_{agg_func}'] = r

    if agg_rows:
        pd.DataFrame(agg_rows).to_csv(
            OUTPUTS_DIR / 'aggregation_func_comparison.csv', index=False
        )

    # ── 6c: Normalisation strategy ────────────────────────────────────────────
    print('\n  6c. Normalisation strategy comparison')
    norm_rows = []
    for dict_name in ['Gardner', 'Sharpe']:
        for norm in ['wordcount', 'zscore', 'none']:
            fname = NLP_DIR / f'{dict_name}_statements_{norm}_nlp.csv'
            if not fname.exists():
                print(f'  ⚠️  Missing: {fname.name} — skipping')
                continue
            dict_df = pd.read_csv(fname, parse_dates=['meeting_date'])
            score_col = 'gardner_total' if dict_name == 'Gardner' else 'sharpe_net'
            if score_col not in dict_df.columns:
                continue
            dict_agg = (dict_df.groupby('meeting_date')[[score_col]]
                        .sum().reset_index()
                        .rename(columns={'meeting_date': 'date',
                                         score_col: f'{score_col}_{norm}'}))
            df_n = pd.merge_asof(
                df.sort_values('date'), dict_agg.sort_values('date'),
                on='date', direction='backward'
            )
            new_col = f'{score_col}_{norm}'
            if new_col not in df_n.columns:
                continue
            target  = df_n[TARGET_NEXT]
            fdf_sub = factor_df.loc[factor_df.index.intersection(df_n.index)]
            X       = fdf_sub.join(df_n[[new_col]], how='left').dropna()
            y       = target.loc[X.index].dropna()
            X       = X.loc[y.index]
            r = fit_ols(y, X, label=f'6c_{dict_name}_{norm}')
            if r:
                print_metrics(r)
                save_model_row(r)
                norm_rows.append({'dictionary': dict_name, 'norm': norm,
                                   'adj_r2': r['adj_r2'],
                                   'aic': r['aic'], 'bic': r['bic']})
                results[f'6c_{dict_name}_{norm}'] = r

    if norm_rows:
        pd.DataFrame(norm_rows).to_csv(
            OUTPUTS_DIR / 'normalization_comparison.csv', index=False
        )

    return results
