"""
step4h_joint_pca.py — Joint PCA FAVAR sweep across all methods and interaction combos.

For each method (LLM, Gardner, Sharpe) and each sentiment variant we run PCA on:
    macro_vars + sentiment_cols + [interaction_terms]
and then OLS on the resulting joint factors → same OOS/CW infrastructure as step4.

Variants per method:
  base        — macro only (baseline, for reference)
  total       — macro + total score
  total_inter — macro + total + total×macro interactions
  topics      — macro + topic scores (no total)
  topics_inter— macro + topic scores + matched topic×macro interactions
  all_inter   — macro + total + topics + all interactions

Usage:
    python step4h_joint_pca.py                    # all methods
    python step4h_joint_pca.py --method LLM
    python step4h_joint_pca.py --method Gardner --start-date 2007-01-01
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings('ignore')

from config import (
    TARGET_NEXT, MACRO_REGRESSORS, PCA_VAR_MACRO,
    OUTPUTS_DIR, INTER_DIR,
)
from utils import run_pca, fit_ols, clark_west_test, save_model_row, print_metrics
from step0_aggregate import load_macro
from step3_baseline_favar import run_baseline


# ── Macro interaction partners (used for all methods) ─────────────────────────
MACRO_INTER = ['vix', 'gdp', 'unemployment_gap', 'inflation_dev_from_target',
               'implied_ffr', 'cfnai']

# ── Matched sentiment→macro pairs (topic-level interactions) ──────────────────
LLM_MATCHED = {
    'sent_total':                 'implied_ffr',
    'sent_inflation':             'inflation_dev_from_target',
    'sent_labor_market':          'unemployment_gap',
    'sent_economic_activity':     'gdp',
    'sent_financial_conditions':  'vix',
    'sent_monetary_policy':       'implied_ffr',
    'sent_macro':                 'cfnai',
}
GARDNER_MATCHED = {
    'g_total':     'implied_ffr',
    'g_inflation': 'inflation_dev_from_target',
    'g_labor_mkt': 'unemployment_gap',
    'g_econ_act':  'gdp',
    'g_fin_cond':  'vix',
    'g_mon_pol':   'implied_ffr',
}
SHARPE_MATCHED = {
    's_net':       'implied_ffr',
    's_positive':  'implied_ffr',
    's_negative':  'implied_ffr',
    's_consensus': 'implied_ffr',
}


# ── Core runner ───────────────────────────────────────────────────────────────

def run_joint_pca_spec(df, pca_input_cols, label, baseline_preds=None,
                       model_csv=None, var_threshold=PCA_VAR_MACRO):
    """
    Run joint PCA on pca_input_cols, then OLS on factors → target.
    Returns result dict (same schema as step4 specs).
    """
    avail = [c for c in pca_input_cols if c in df.columns]
    sub = df[avail + [TARGET_NEXT]].dropna()
    if len(sub) < 30:
        return None

    factor_df, _, _, n_comp = run_pca(sub, avail, var_threshold=var_threshold)
    factor_df.columns = [f'JPC{i+1}' for i in range(n_comp)]
    target = sub[TARGET_NEXT].loc[factor_df.index]

    result = fit_ols(target, factor_df, label=label)
    if not result:
        return None

    if baseline_preds:
        cw_t, cw_p = clark_west_test(baseline_preds, result.get('_oos_preds', []))
        result['cw_tstat'] = cw_t
        result['cw_pval']  = cw_p

    result['n_pca_inputs'] = len(avail)
    result['n_components'] = n_comp
    print_metrics(result)
    save_model_row(result, csv_path=model_csv)
    return result


def add_interactions(df, sent_cols, macro_cols, matched_map=None):
    """
    Add interaction columns to df in-place.
    Returns list of new column names added.
    """
    inter_cols = []
    avail_macro = [c for c in macro_cols if c in df.columns]

    # total × all macro (broadcast interactions)
    for sc in sent_cols:
        if sc not in df.columns:
            continue
        for mc in avail_macro:
            if mc not in df.columns:
                continue
            iname = f'inter_{sc}_x_{mc}'
            df[iname] = df[sc] * df[mc]
            inter_cols.append(iname)

    # matched topic × paired macro (only if map provided)
    if matched_map:
        for sc, mc in matched_map.items():
            if sc in df.columns and mc in df.columns:
                iname = f'matched_{sc}_x_{mc}'
                if iname not in df.columns:
                    df[iname] = df[sc] * df[mc]
                    inter_cols.append(iname)

    return inter_cols


# ── Per-method dataset builders ───────────────────────────────────────────────

def build_llm_panel(start_date, root):
    df = pd.read_csv(
        root + r'\Taylor Rule\outputs\intermediate\step0_merged_to_macro.csv',
        parse_dates=['date'])
    if start_date:
        df = df[df['date'] >= pd.Timestamp(start_date)]
    return df.reset_index(drop=True)


def build_dict_panel(dict_name, remap, root, start_date):
    nlp_dir = root + r'\Taylor Rule\nlp_output'
    fname = (f'{nlp_dir}\\{dict_name}_statements_minutes_speeches'
             f'_press_conferences_zscore_nlp.csv')
    raw = pd.read_csv(fname, parse_dates=['meeting_date'])
    score_cols = [c for c in raw.columns
                  if c.startswith(dict_name.lower() + '_') and 'consensus' not in c]
    agg = (raw.groupby('meeting_date')[score_cols].sum()
           .reset_index()
           .rename(columns={**{'meeting_date': 'date'}, **remap}))
    macro = load_macro()
    df = pd.merge_asof(macro.sort_values('date'), agg.sort_values('date'),
                       on='date', direction='backward')
    if start_date:
        df = df[df['date'] >= pd.Timestamp(start_date)]
    return df.reset_index(drop=True)


GARDNER_REMAP = {
    'gardner_total': 'g_total', 'gardner_inf': 'g_inflation',
    'gardner_labor': 'g_labor_mkt', 'gardner_out': 'g_econ_act',
    'gardner_fin':   'g_fin_cond',  'gardner_mp':  'g_mon_pol',
}
SHARPE_REMAP = {
    'sharpe_net': 's_net', 'sharpe_positive': 's_positive',
    'sharpe_negative': 's_negative', 'sharpe_consensus': 's_consensus',
}


# ── Main sweep ────────────────────────────────────────────────────────────────

def run_sweep(method='LLM', start_date='2007-01-01',
              root=None, model_csv=None):

    if root is None:
        root = str(Path(__file__).parent.parent.parent)

    mc = [c for c in MACRO_REGRESSORS]  # filtered per-df below

    # Load panel and define sentiment columns
    if method == 'LLM':
        df = build_llm_panel(start_date, root)
        total_cols  = ['sent_total']
        topic_cols  = ['sent_inflation', 'sent_labor_market', 'sent_economic_activity',
                       'sent_financial_conditions', 'sent_monetary_policy', 'sent_macro']
        matched_map = LLM_MATCHED
        prefix      = 'llm'

    elif method == 'Gardner':
        df = build_dict_panel('Gardner', GARDNER_REMAP, root, start_date)
        total_cols  = ['g_total']
        topic_cols  = ['g_inflation', 'g_labor_mkt', 'g_econ_act', 'g_fin_cond', 'g_mon_pol']
        matched_map = GARDNER_MATCHED
        prefix      = 'gard'

    elif method == 'Sharpe':
        df = build_dict_panel('Sharpe', SHARPE_REMAP, root, start_date)
        total_cols  = ['s_net']
        topic_cols  = ['s_positive', 's_negative', 's_consensus']
        matched_map = SHARPE_MATCHED
        prefix      = 'sharpe'

    else:
        raise ValueError(f'Unknown method: {method}')

    # Filter macro cols to those available
    mc_avail = [c for c in mc if c in df.columns]
    topic_avail = [c for c in topic_cols if c in df.columns]
    total_avail = [c for c in total_cols if c in df.columns]
    all_sent = total_avail + topic_avail

    # Add interaction columns to df
    inter_total  = add_interactions(df, total_avail,  MACRO_INTER)
    inter_topics = add_interactions(df, topic_avail,  MACRO_INTER, matched_map)
    inter_all    = list(dict.fromkeys(inter_total + inter_topics))  # deduplicated

    # Baseline (macro only) — for CW comparison
    print(f'\n{"="*70}')
    print(f'  {method} — Joint PCA FAVAR sweep  [start={start_date}]')
    print(f'{"="*70}')

    baseline = run_baseline(df)
    baseline_preds = baseline.get('_oos_preds', [])

    if model_csv is None:
        tag = f'_{start_date.replace("-","")[:6]}' if start_date else ''
        model_csv = OUTPUTS_DIR / f'4h_{prefix}_joint_pca{tag}.csv'
    if Path(model_csv).exists():
        Path(model_csv).unlink()

    results = []

    variants = [
        # label suffix         pca input cols
        ('base',               mc_avail),
        ('total',              mc_avail + total_avail),
        ('total_inter',        mc_avail + total_avail + inter_total),
        ('topics',             mc_avail + topic_avail),
        ('topics_matched',     mc_avail + topic_avail + [c for c in inter_topics
                                                         if 'matched_' in c]),
        ('topics_inter',       mc_avail + topic_avail + inter_topics),
        ('all_sent',           mc_avail + all_sent),
        ('all_sent_inter',     mc_avail + all_sent + inter_all),
    ]

    for var_name, pca_cols in variants:
        label = f'4h_{prefix}_{var_name}'
        print(f'\n  -- {label}')
        r = run_joint_pca_spec(df, pca_cols, label,
                               baseline_preds=baseline_preds,
                               model_csv=model_csv)
        if r:
            r['method']  = method
            r['variant'] = var_name
            results.append(r)

    return results


# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary(all_results):
    rows = [{
        'method':      r['method'],
        'variant':     r['variant'],
        'n_inputs':    r.get('n_pca_inputs', '?'),
        'n_comp':      r.get('n_components', '?'),
        'adj_r2':      r.get('adj_r2'),
        'oos_rmse_80': r.get('oos_rmse_80'),
        'cw_tstat':    r.get('cw_tstat'),
        'cw_pval':     r.get('cw_pval'),
    } for r in all_results if r]

    df = pd.DataFrame(rows)
    print('\n' + '='*85)
    print('  SUMMARY — Joint PCA FAVAR (all methods & variants)')
    print('='*85)
    print(f'  {"Method":<10} {"Variant":<22} {"Inputs":>7} {"Comps":>6} '
          f'{"AdjR2":>8} {"OOS80":>8} {"CW-t":>8} {"CW-p":>8}')
    print('  ' + '-'*85)
    for _, row in df.sort_values(['method', 'oos_rmse_80']).iterrows():
        cw_t = f'{row["cw_tstat"]:>8.3f}' if pd.notna(row['cw_tstat']) else '     n/a'
        cw_p = f'{row["cw_pval"]:>8.4f}' if pd.notna(row['cw_pval']) else '     n/a'
        print(f'  {row["method"]:<10} {row["variant"]:<22} {row["n_inputs"]:>7} '
              f'{row["n_comp"]:>6} {row["adj_r2"]:>8.4f} {row["oos_rmse_80"]:>8.4f}'
              f'{cw_t}{cw_p}')

    print()
    print('  === BEST PER METHOD ===')
    for method, grp in df.groupby('method'):
        best = grp.loc[grp['oos_rmse_80'].idxmin()]
        print(f'  {method:<10} {best["variant"]:<22} OOS80={best["oos_rmse_80"]:.4f}  '
              f'AdjR2={best["adj_r2"]:.4f}  CW-t={best["cw_tstat"]:.3f}')

    tag = all_results[0].get('model_label', '').split('_')[2] if all_results else ''
    out = OUTPUTS_DIR / '4h_joint_pca_summary.csv'
    df.to_csv(out, index=False)
    print(f'\n  Summary -> {out}')


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default=None,
                        choices=['LLM', 'Gardner', 'Sharpe'],
                        help='Single method to run (default: all three)')
    parser.add_argument('--start-date', default='2007-01-01')
    args = parser.parse_args()

    root = str(Path(__file__).parent.parent.parent)
    methods = [args.method] if args.method else ['LLM', 'Gardner', 'Sharpe']

    all_results = []
    for method in methods:
        results = run_sweep(method=method, start_date=args.start_date, root=root)
        all_results.extend(results)

    print_summary(all_results)
