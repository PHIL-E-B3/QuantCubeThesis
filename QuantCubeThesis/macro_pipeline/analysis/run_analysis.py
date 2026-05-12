"""
run_analysis.py — Master runner for the FOMC NLP sentiment analysis pipeline.

Steps:
    D   Re-compute dictionary scores (Gardner + Sharpe, wordcount + zscore)
    0   Aggregate sentence labels → document level + merge to macro panel
    1   Stationarity checks (ADF + KPSS)
    2   Correlation with dictionary approaches
    3   Baseline FAVAR
    4   NLP regressor tests (4a–4g)
    5   Mask-based investigations (5a–5f, including speaker mask)
    6   Scaling & normalisation exploration (6a–6c)
    7   Robustness checks (7a–7d)
    R   Generate RESULTS_SUMMARY.md

Usage:
    # Full pipeline (after LLM inference is done):
    python macro_pipeline/analysis/run_analysis.py --sentences data/llm_labels.json

    # Skip dictionary re-computation (already done):
    python macro_pipeline/analysis/run_analysis.py --sentences data/llm_labels.json --steps 0 1 2 3 4 5 6 7 R

    # Run only dictionary computation (no LLM labels needed):
    python macro_pipeline/analysis/run_analysis.py --steps D

    # Run specific steps:
    python macro_pipeline/analysis/run_analysis.py --sentences data/llm_labels.json --steps 3 4 5

NOTE: Steps 1–7 and R all require --sentences (LLM output).
      Step D only requires access to raw FOMC text in data/raw/.
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import OUTPUTS_DIR, INTER_DIR, RANDOM_SEED
import numpy as np
np.random.seed(RANDOM_SEED)
warnings.filterwarnings('ignore')


def _banner(step: str):
    print(f'\n{"="*65}')
    print(f'  STEP {step}')
    print(f'{"="*65}')


def run(sentences_path=None, steps=None):
    steps = steps or ['D','0','1','2','3','4','5','6','7','R']
    steps = [str(s).upper() for s in steps]

    state = {}  # carries results between steps

    t_total = time.time()

    # ── Step D: Dictionary re-computation ─────────────────────────────────────
    if 'D' in steps:
        _banner('D — Dictionary Re-computation')
        from step_d_dictionary_runner import main as run_dict
        run_dict()

    # ── Steps requiring LLM labels ────────────────────────────────────────────
    needs_labels = set('0123456789R') & set(steps)
    if needs_labels and not sentences_path:
        print('\n  ⚠️  Steps 0–7 and R require --sentences <path>. '
              'Skipping those steps.\n  Re-run with --sentences once LLM '
              'inference is complete.')
        return

    if needs_labels:
        # ── Step 0 ─────────────────────────────────────────────────────────────
        if '0' in steps:
            _banner('0 — Sentence-to-Document Aggregation')
            from step0_aggregate import load_sentences, aggregate_to_document, merge_to_macro
            df_sentences = load_sentences(sentences_path)
            print(f'  Loaded {len(df_sentences):,} sentences from {sentences_path}')
            doc_df      = aggregate_to_document(df_sentences)
            df_macro    = merge_to_macro(doc_df)
            state['df_sentences'] = df_sentences
            state['doc_df']       = doc_df
            state['df_macro']     = df_macro
            print(f'  Macro panel: {df_macro.shape[0]} rows × {df_macro.shape[1]} cols')

        elif any(s in steps for s in ['1','2','3','4','5','6','7','R']):
            # Try to reload from cache
            f0 = INTER_DIR / 'step0_merged_to_macro.csv'
            if f0.exists():
                print(f'  Loading Step 0 cache: {f0}')
                state['df_macro'] = pd.read_csv(f0, parse_dates=['date'])
                from step0_aggregate import load_sentences
                state['df_sentences'] = load_sentences(sentences_path)
            else:
                print('  Step 0 output not found — run Step 0 first.')
                return

        df_macro    = state.get('df_macro', pd.DataFrame())
        df_sentences = state.get('df_sentences', pd.DataFrame())

        # ── Step 1 ─────────────────────────────────────────────────────────────
        if '1' in steps:
            _banner('1 — Stationarity Checks')
            from step1_stationarity import run_stationarity
            state['stationarity'] = run_stationarity(df_macro)

        # ── Step 2 ─────────────────────────────────────────────────────────────
        if '2' in steps:
            _banner('2 — Correlation with Dictionary Approaches')
            from step2_correlation import run_correlation
            state['correlation'] = run_correlation(df_macro)

        # ── Step 3 ─────────────────────────────────────────────────────────────
        if '3' in steps:
            _banner('3 — Baseline FAVAR')
            from step3_baseline_favar import run_baseline
            state['baseline'] = run_baseline(df_macro)

        elif any(s in steps for s in ['4','5','6','7']):
            # Try to rebuild factors from cache
            f3 = INTER_DIR / 'step3_baseline_factors.csv'
            if f3.exists() and 'baseline' not in state:
                print('  Reloading baseline factors from cache ...')
                factors = pd.read_csv(f3, index_col=0)
                state['baseline'] = {'_factor_df': factors}

        # ── Step 4 ─────────────────────────────────────────────────────────────
        if '4' in steps:
            _banner('4 — NLP Regressor Tests (4a–4g)')
            from step4_nlp_tests import run_nlp_tests
            state['best_step4'] = run_nlp_tests(df_macro, state.get('baseline', {}))

        # ── Step 5 ─────────────────────────────────────────────────────────────
        if '5' in steps:
            _banner('5 — Mask-Based Investigations (5a–5f)')
            from step5_masks import run_masks
            best = state.get('best_step4') or state.get('baseline', {})
            state['masks'] = run_masks(df_macro, best, df_sentences=df_sentences)

        # ── Step 6 ─────────────────────────────────────────────────────────────
        if '6' in steps:
            _banner('6 — Scaling & Normalisation Exploration')
            from step6_scaling import run_scaling
            best = state.get('best_step4') or state.get('baseline', {})
            state['scaling'] = run_scaling(df_macro, df_sentences, best)

        # ── Step 7 ─────────────────────────────────────────────────────────────
        if '7' in steps:
            _banner('7 — Robustness Checks')
            from step7_robustness import run_robustness
            best = state.get('best_step4') or state.get('baseline', {})
            state['robustness'] = run_robustness(df_macro, best)

        # ── Results summary ────────────────────────────────────────────────────
        if 'R' in steps:
            _banner('R — Results Summary')
            _write_summary(state)

    elapsed = time.time() - t_total
    print(f'\n{"="*65}')
    print(f'  Pipeline complete in {elapsed/60:.1f} min')
    print(f'  Outputs → {OUTPUTS_DIR}')
    print(f'{"="*65}')


def _write_summary(state: dict):
    lines = ['# FOMC NLP Sentiment Analysis — Results Summary\n']

    # Model comparison table
    mc = OUTPUTS_DIR / 'model_comparison.csv'
    if mc.exists():
        df_mc = pd.read_csv(mc)
        lines.append('## All Model Specifications\n')
        lines.append(df_mc.to_markdown(index=False))
        lines.append('\n')

        best_row = df_mc.loc[df_mc['adj_r2'].idxmax()]
        lines.append(f'## Best Model Overall\n')
        lines.append(f'**{best_row["model_label"]}** — '
                     f'Adj R² = {best_row["adj_r2"]:.4f}, '
                     f'OOS RMSE = {best_row.get("oos_rmse","n/a")}\n')

    # Mask results
    mf = INTER_DIR / 'step5_mask_results.csv'
    if mf.exists():
        df_m = pd.read_csv(mf)
        if 'adj_r2' in df_m.columns and not df_m.empty:
            best_mask = df_m.loc[df_m['adj_r2'].idxmax()]
            lines.append(f'\n## Best Mask\n')
            lines.append(f'**{best_mask["model_label"]}** — '
                         f'Adj R² = {best_mask["adj_r2"]:.4f}\n')

    # Scaling
    sf = OUTPUTS_DIR / 'aggregation_func_comparison.csv'
    if sf.exists():
        df_s = pd.read_csv(sf)
        best_s = df_s.loc[df_s['adj_r2'].idxmax()]
        lines.append(f'\n## Optimal Aggregation Function\n')
        lines.append(f'**{best_s["agg_func"]}** — Adj R² = {best_s["adj_r2"]:.4f}\n')

    nf = OUTPUTS_DIR / 'normalization_comparison.csv'
    if nf.exists():
        df_n = pd.read_csv(nf)
        best_n = df_n.loc[df_n['adj_r2'].idxmax()]
        lines.append(f'\n## Optimal Normalisation Strategy\n')
        lines.append(f'**{best_n["dictionary"]} / {best_n["norm"]}** — '
                     f'Adj R² = {best_n["adj_r2"]:.4f}\n')

    # Speaker mask finding
    spk = INTER_DIR / 'step5f_speaker_mask_results.csv'
    if spk.exists():
        df_spk = pd.read_csv(spk)
        lines.append(f'\n## Speaker Mask (5f)\n')
        lines.append(df_spk[['mask','adj_r2','oos_rmse']].to_markdown(index=False))
        lines.append('\n')

    out = OUTPUTS_DIR / 'RESULTS_SUMMARY.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f'  Summary written → {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FOMC NLP Analysis Pipeline')
    parser.add_argument('--sentences', type=str, default=None,
                        help='Path to LLM-labelled sentences JSON (single file or dir)')
    parser.add_argument('--steps', nargs='+', default=None,
                        help='Steps to run, e.g. --steps D 0 1 2 3 4 5 6 7 R')
    args = parser.parse_args()
    run(sentences_path=args.sentences, steps=args.steps)
