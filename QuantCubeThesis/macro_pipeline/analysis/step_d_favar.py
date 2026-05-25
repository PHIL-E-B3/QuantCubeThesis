"""
step_d_favar.py — FAVAR regressions using dictionary sentiment scores (no LLM labels).

Reads the CSVs produced by Step D (step_d_dictionary_runner.py), renames score
columns to the sent_* schema consumed by step3/step4, aggregates to meeting level,
merges onto Master_Macro, then runs baseline + NLP-augmented FAVARs.

Gardner maps per-topic sub-scores to sent_* columns, so specs 4c/4d/4f (topic
PCA, matched interactions) run in full. Sharpe produces sent_total only, so
those specs are silently skipped.

Step 4b (dispersion) uses consensus scores (agreement among scoring events)
as a proxy for dispersion — available for both Gardner and Sharpe.
Step 5 (masks) is not run — it requires LLM-only fields.

Usage:
    python macro_pipeline/analysis/step_d_favar.py              # all 20 combos
    python macro_pipeline/analysis/step_d_favar.py --dict Gardner
    python macro_pipeline/analysis/step_d_favar.py --dict Sharpe --norm zscore
"""

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import NLP_DIR, OUTPUTS_DIR, INTER_DIR
from step0_aggregate import load_macro
from utils import log_warn

warnings.filterwarnings('ignore')

# ── Column remaps ─────────────────────────────────────────────────────────────

GARDNER_REMAP = {
    'gardner_total': 'sent_total',
    'gardner_inf':   'sent_inflation',
    'gardner_labor': 'sent_labor_market',
    'gardner_out':   'sent_economic_activity',
    'gardner_fin':   'sent_financial_conditions',
    'gardner_mp':    'sent_monetary_policy',
    # Consensus columns — used as dispersion proxy in spec 4b
    'gardner_inf_consensus':   'sent_inflation_consensus',
    'gardner_labor_consensus': 'sent_labor_market_consensus',
    'gardner_out_consensus':   'sent_economic_activity_consensus',
    'gardner_fin_consensus':   'sent_financial_conditions_consensus',
    'gardner_mp_consensus':    'sent_monetary_policy_consensus',
}
SHARPE_REMAP = {
    'sharpe_net':       'sent_total',
    'sharpe_consensus': 'sent_consensus',
}

# ── All 20 combos matching step_d_dictionary_runner ───────────────────────────

ALL_COMBOS = [
    ('Gardner', ['statements'],                                                  'wordcount'),
    ('Gardner', ['statements'],                                                  'zscore'),
    ('Gardner', ['minutes'],                                                     'wordcount'),
    ('Gardner', ['minutes'],                                                     'zscore'),
    ('Gardner', ['speeches'],                                                    'wordcount'),
    ('Gardner', ['speeches'],                                                    'zscore'),
    ('Gardner', ['press_conferences'],                                           'wordcount'),
    ('Gardner', ['press_conferences'],                                           'zscore'),
    ('Gardner', ['statements', 'minutes', 'speeches', 'press_conferences'],     'wordcount'),
    ('Gardner', ['statements', 'minutes', 'speeches', 'press_conferences'],     'zscore'),
    ('Sharpe',  ['statements'],                                                  'wordcount'),
    ('Sharpe',  ['statements'],                                                  'zscore'),
    ('Sharpe',  ['minutes'],                                                     'wordcount'),
    ('Sharpe',  ['minutes'],                                                     'zscore'),
    ('Sharpe',  ['speeches'],                                                    'wordcount'),
    ('Sharpe',  ['speeches'],                                                    'zscore'),
    ('Sharpe',  ['press_conferences'],                                           'wordcount'),
    ('Sharpe',  ['press_conferences'],                                           'zscore'),
    ('Sharpe',  ['statements', 'minutes', 'speeches', 'press_conferences'],     'wordcount'),
    ('Sharpe',  ['statements', 'minutes', 'speeches', 'press_conferences'],     'zscore'),
]


# ── Bridge: dictionary CSV → df_macro ─────────────────────────────────────────

def build_dict_macro(dict_name: str, doc_types: list, norm: str) -> pd.DataFrame:
    """
    Load one dictionary CSV, rename score columns to sent_* schema,
    sum across doc_types per meeting_date, and merge_asof onto Master_Macro.
    Returns df_macro ready for step3/step4, or empty DataFrame if file missing.
    """
    fname = NLP_DIR / f'{dict_name}_{"_".join(doc_types)}_{norm}_nlp.csv'
    if not fname.exists():
        log_warn(f'Dictionary file not found: {fname}')
        return pd.DataFrame()

    df = pd.read_csv(fname, parse_dates=['meeting_date', 'date'])

    remap = GARDNER_REMAP if dict_name == 'Gardner' else SHARPE_REMAP
    df = df.rename(columns={k: v for k, v in remap.items() if k in df.columns})

    sent_cols = [c for c in df.columns if c.startswith('sent_')]
    if not sent_cols:
        log_warn(f'No sent_* columns after remap for {fname.name}')
        return pd.DataFrame()

    # Sum across doc_types per meeting (relevant for combined combos)
    agg = df.groupby('meeting_date')[sent_cols].sum().reset_index()
    agg = agg.rename(columns={'meeting_date': 'date'})

    macro = load_macro()
    df_macro = pd.merge_asof(
        macro.sort_values('date'),
        agg.sort_values('date'),
        on='date',
        direction='backward',
    )
    return df_macro.sort_values('date').reset_index(drop=True)


# ── Main runner ───────────────────────────────────────────────────────────────

def run_dict_favars(dict_filter: str = None, norm_filter: str = None,
                    doc_filter: str = None, target_col: str = None,
                    oos_frac: float = None, start_date: str = None,
                    ewma_span: int = None):
    """
    Run baseline + NLP-augmented FAVARs for all (or filtered) dictionary combos.

    target_col: override the regression target. Must be a column in df_macro.
                Defaults to TARGET_NEXT ('target_next' = level of next period rate).
                Pass 'delta_rate_next' for change-in-rate regressions.
    oos_frac:   override the OOS training fraction (e.g. 0.80 → 80% train / 20% OOS).
                Defaults to 0.60 (config.OOS_FRACS minimum).
    ewma_span:  apply causal EWMA smoothing to all sent_* columns before regression.
    """
    import utils as _utils_mod
    from config import TARGET_NEXT, OOS_FRACS
    from step3_baseline_favar import run_baseline
    from step4_nlp_tests import run_nlp_tests

    # Patch OOS_FRACS so oos_rmse() uses the requested init fraction
    if oos_frac is not None:
        _utils_mod.OOS_FRACS = [oos_frac]
        primary_key = f'oos_rmse_{int(oos_frac * 100)}'
        oos_tag = f'_oos{int(oos_frac * 100)}'
    else:
        _utils_mod.OOS_FRACS = OOS_FRACS
        primary_key = 'oos_rmse_60'
        oos_tag = ''

    use_target = target_col or TARGET_NEXT
    start_tag = f'_from{start_date.replace("-","")[:6]}' if start_date else ''
    ewma_tag  = f'_ewma{ewma_span}' if ewma_span else ''
    tag = ('_delta' if use_target == 'delta_rate_next' else '') + oos_tag + start_tag + ewma_tag

    # Dedicated output CSV so dict results stay separate from LLM model_comparison.csv
    dict_models_csv = OUTPUTS_DIR / f'dict_favar_models{tag}.csv'
    if dict_models_csv.exists():
        dict_models_csv.unlink()

    baseline_result = None
    summary_rows = []

    combos_to_run = [
        (d, dt, n) for d, dt, n in ALL_COMBOS
        if (not dict_filter or d.lower() == dict_filter.lower())
        and (not norm_filter or n == norm_filter)
        and (not doc_filter or dt == [doc_filter])
    ]

    if not combos_to_run:
        print('  No combos match the given filters.')
        return

    for dict_name, doc_types, norm in combos_to_run:
        combo_label = f'{dict_name}_{"_".join(doc_types)}_{norm}'
        print(f'\n{"="*65}')
        print(f'  Dictionary FAVAR: {combo_label}  [target={use_target}]')
        print(f'{"="*65}')

        df_macro = build_dict_macro(dict_name, doc_types, norm)
        if df_macro.empty:
            print('  Skipped (no data - run Step D first)')
            continue
        if start_date:
            df_macro = df_macro[df_macro['date'] >= pd.Timestamp(start_date)].reset_index(drop=True)

        if ewma_span:
            sent_cols = [c for c in df_macro.columns if c.startswith('sent_')]
            for c in sent_cols:
                df_macro[c] = df_macro[c].ewm(span=ewma_span, adjust=False).mean()

        # Swap target column in-place so step3/step4 see it as TARGET_NEXT
        if use_target != TARGET_NEXT:
            if use_target not in df_macro.columns:
                print(f'  Skipped: column {use_target!r} not in df_macro')
                continue
            df_macro = df_macro.copy()
            df_macro[TARGET_NEXT] = df_macro[use_target]

        # Baseline is macro-only — identical across all combos, compute once
        if baseline_result is None:
            print('\n  Running baseline FAVAR (macro-only)...')
            baseline_result = run_baseline(df_macro)

        # NLP-augmented FAVAR — label_prefix keeps combo rows identifiable
        step4_best = run_nlp_tests(
            df_macro,
            baseline_result,
            label_prefix=f'{combo_label}__',
            model_csv=dict_models_csv,
        )

        summary_rows.append({
            'dict':              dict_name,
            'doc_types':         '+'.join(doc_types),
            'norm':              norm,
            'combo_label':       combo_label,
            'target':            use_target,
            'oos_init_frac':     oos_frac or min(OOS_FRACS),
            'best_spec':         step4_best.get('spec')             if step4_best else None,
            'best_adj_r2':       step4_best.get('adj_r2')           if step4_best else None,
            'best_oos_rmse':     step4_best.get(primary_key)        if step4_best else None,
            'best_cw_tstat':     step4_best.get('cw_tstat')         if step4_best else None,
            'best_cw_pval':      step4_best.get('cw_pval')          if step4_best else None,
        })

    if not summary_rows:
        print('\n  No dictionary files found — run Step D first '
              '(python macro_pipeline/analysis/step_d_dictionary_runner.py).')
        return

    summary = pd.DataFrame(summary_rows).sort_values('best_adj_r2', ascending=False)
    out = OUTPUTS_DIR / f'dict_favar_summary{tag}.csv'
    summary.to_csv(out, index=False)

    print(f'\n{"="*65}')
    print('  Dictionary FAVAR - top results by Adj R2:')
    print(f'{"="*65}')
    print(summary[['combo_label', 'best_spec', 'best_adj_r2',
                   'best_oos_rmse', 'best_cw_tstat', 'best_cw_pval']].to_string(index=False))
    print(f'\n  Summary    -> {out}')
    print(f'  All models -> {dict_models_csv}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dictionary-based FAVAR regressions')
    parser.add_argument('--dict', type=str, default=None,
                        choices=['Gardner', 'Sharpe'],
                        help='Run only this dictionary (default: both)')
    parser.add_argument('--norm', type=str, default=None,
                        choices=['wordcount', 'zscore'],
                        help='Run only this normalization (default: both)')
    parser.add_argument('--doc', type=str, default=None,
                        help='Run only combos whose doc_types list matches exactly '
                             '(e.g. "statements", "minutes")')
    parser.add_argument('--target', type=str, default=None,
                        choices=['target_next', 'delta_rate_next'],
                        help='Regression target: level (target_next) or change (delta_rate_next)')
    parser.add_argument('--oos-frac', type=float, default=None,
                        help='OOS training fraction, e.g. 0.80 for 80%% train / 20%% OOS (default: 0.60)')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Drop rows before this date, e.g. 2007-01-01')
    parser.add_argument('--ewma-span', type=int, default=None,
                        help='Apply causal EWMA smoothing to all sent_* columns, e.g. 10')
    args = parser.parse_args()
    run_dict_favars(dict_filter=args.dict, norm_filter=args.norm,
                    doc_filter=args.doc, target_col=args.target,
                    oos_frac=args.oos_frac, start_date=args.start_date,
                    ewma_span=args.ewma_span)
