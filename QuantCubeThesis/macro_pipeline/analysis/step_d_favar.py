"""
step_d_favar.py — FAVAR regressions using dictionary sentiment scores (no LLM labels).

Reads the CSVs produced by Step D (step_d_dictionary_runner.py), renames score
columns to the sent_* schema consumed by step3/step4, aggregates to meeting level,
merges onto Master_Macro, then runs baseline + NLP-augmented FAVARs.

Gardner maps per-topic sub-scores to sent_* columns, so specs 4c/4d/4f (topic
PCA, matched interactions) run in full. Sharpe produces sent_total only, so
those specs are silently skipped.

Step 4b (dispersion) is skipped for both — dictionary has no sentence-level
variance. Step 5 (masks) is not run — it requires LLM-only fields.

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
}
SHARPE_REMAP = {
    'sharpe_net': 'sent_total',
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

def run_dict_favars(dict_filter: str = None, norm_filter: str = None):
    """
    Run baseline + NLP-augmented FAVARs for all (or filtered) dictionary combos.
    Results go to outputs/dict_favar_summary.csv and outputs/dict_favar_models.csv.
    """
    from step3_baseline_favar import run_baseline
    from step4_nlp_tests import run_nlp_tests

    # Dedicated output CSV so dict results stay separate from LLM model_comparison.csv
    dict_models_csv = OUTPUTS_DIR / 'dict_favar_models.csv'
    if dict_models_csv.exists():
        dict_models_csv.unlink()

    baseline_result = None
    summary_rows = []

    combos_to_run = [
        (d, dt, n) for d, dt, n in ALL_COMBOS
        if (not dict_filter or d.lower() == dict_filter.lower())
        and (not norm_filter or n == norm_filter)
    ]

    if not combos_to_run:
        print('  No combos match the given filters.')
        return

    for dict_name, doc_types, norm in combos_to_run:
        combo_label = f'{dict_name}_{"_".join(doc_types)}_{norm}'
        print(f'\n{"="*65}')
        print(f'  Dictionary FAVAR: {combo_label}')
        print(f'{"="*65}')

        df_macro = build_dict_macro(dict_name, doc_types, norm)
        if df_macro.empty:
            print('  Skipped (no data — run Step D first)')
            continue

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
            'dict':          dict_name,
            'doc_types':     '+'.join(doc_types),
            'norm':          norm,
            'combo_label':   combo_label,
            'best_spec':     step4_best.get('spec')     if step4_best else None,
            'best_adj_r2':   step4_best.get('adj_r2')   if step4_best else None,
            'best_oos_rmse': step4_best.get('oos_rmse') if step4_best else None,
        })

    if not summary_rows:
        print('\n  No dictionary files found — run Step D first '
              '(python macro_pipeline/analysis/step_d_dictionary_runner.py).')
        return

    summary = pd.DataFrame(summary_rows).sort_values('best_adj_r2', ascending=False)
    out = OUTPUTS_DIR / 'dict_favar_summary.csv'
    summary.to_csv(out, index=False)

    print(f'\n{"="*65}')
    print('  Dictionary FAVAR — top results by Adj R²:')
    print(f'{"="*65}')
    print(summary[['combo_label', 'best_spec', 'best_adj_r2', 'best_oos_rmse']].to_string(index=False))
    print(f'\n  Summary   → {out}')
    print(f'  All models → {dict_models_csv}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dictionary-based FAVAR regressions')
    parser.add_argument('--dict', type=str, default=None,
                        choices=['Gardner', 'Sharpe'],
                        help='Run only this dictionary (default: both)')
    parser.add_argument('--norm', type=str, default=None,
                        choices=['wordcount', 'zscore'],
                        help='Run only this normalization (default: both)')
    args = parser.parse_args()
    run_dict_favars(dict_filter=args.dict, norm_filter=args.norm)
