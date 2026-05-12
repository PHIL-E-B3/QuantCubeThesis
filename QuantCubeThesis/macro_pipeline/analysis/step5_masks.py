"""
step5_masks.py — Mask-based investigations (5a–5f).
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
from config import (
    OUTPUTS_DIR, INTER_DIR, TARGET_NEXT, TARGET_COL,
    MIN_MASK_OBS, COLOUR_PALETTE,
)
from utils import (
    fit_ols, save_model_row, print_metrics, log_warn,
    shadow_rate_direction_barplot, new_fig, save_fig,
)

warnings.filterwarnings('ignore')


def _rerun_best(df_masked: pd.DataFrame, best_result: dict,
                label: str) -> dict:
    """Re-fit the best Step 4 model on a masked subset."""
    if len(df_masked) < MIN_MASK_OBS:
        log_warn(f'Mask [{label}]: only {len(df_masked)} obs — skipped.')
        return {}
    factor_df = best_result.get('_factor_df')
    if factor_df is None:
        return {}
    factor_sub = factor_df.loc[factor_df.index.intersection(df_masked.index)]
    target     = df_masked.loc[factor_sub.index, TARGET_NEXT]

    # Add sentiment columns that were in the best model
    best_model = best_result.get('_model')
    if best_model is not None:
        extra = [c for c in best_model.model.exog_names
                 if c != 'const' and c not in factor_sub.columns]
        for c in extra:
            if c in df_masked.columns:
                factor_sub = factor_sub.join(df_masked[[c]], how='left')

    r = fit_ols(target, factor_sub, label=label)
    if r:
        print_metrics(r)
        save_model_row(r)
    return r


def _direction_plot(df: pd.DataFrame, label: str, ax):
    shadow_rate_direction_barplot(df, TARGET_COL, title=label, ax=ax)


def apply_mask(df: pd.DataFrame, condition: pd.Series,
               label: str, best_result: dict) -> dict:
    masked = df[condition]
    return _rerun_best(masked, best_result, label)


def run_masks(df: pd.DataFrame, best_result: dict,
              df_sentences=None) -> dict:
    """
    Run all mask analyses (5a–5f).
    df_sentences: original sentence-level DataFrame (needed for 5a/5b/5c
                  option-a masks that re-aggregate).
    Returns dict of {mask_label: result_dict}.
    """
    from step0_aggregate import aggregate_to_document, merge_to_macro

    results = {}
    fig_dir = OUTPUTS_DIR

    # ── 5a: Uncertainty mask ──────────────────────────────────────────────────
    print('\n  5a. Uncertainty mask')
    fig, axes = new_fig(3, cols=3, figsize_per=(5, 3.5))

    # Mask A: re-aggregate with only elevated-wid sentences
    if df_sentences is not None:
        df_elev = df_sentences[df_sentences['wid'] == 'elevated'].copy()
        if len(df_elev) > MIN_MASK_OBS:
            doc_elev = aggregate_to_document(df_elev)
            df_elev_m = merge_to_macro(doc_elev)
            r = _rerun_best(df_elev_m, best_result, '5a_uncertainty_elevated_sentences')
            results['5a_uncertainty_A'] = r
            _direction_plot(df_elev_m, '5a-A elevated wid (sentences)', axes[0])

    # Mask B: flag_elevated_wid > median
    if 'flag_elevated_wid' in df.columns:
        med  = df['flag_elevated_wid'].median()
        cond = df['flag_elevated_wid'] > med
        r = apply_mask(df, cond, '5a_uncertainty_flag_gt_median', best_result)
        results['5a_uncertainty_B'] = r
        _direction_plot(df[cond], '5a-B flag > median', axes[1])

    _direction_plot(df, '5a full sample', axes[2])
    save_fig(fig, fig_dir / 'step5a_uncertainty.png')

    # ── 5b: Tail risk mask ────────────────────────────────────────────────────
    print('\n  5b. Tail risk mask')
    for mask_label, cond_fn in [
        ('5b_skew_up',   lambda d: d.get('flag_skew_up', pd.Series(0, index=df.index)) > 0),
        ('5b_skew_down', lambda d: d.get('flag_skew_down', pd.Series(0, index=df.index)) > 0),
    ]:
        cond = cond_fn(df)
        r = apply_mask(df, cond, mask_label, best_result)
        results[mask_label] = r

    fig2, ax2 = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (lbl, col) in zip(ax2, [
        ('5b skew up',   'flag_skew_up'),
        ('5b skew down', 'flag_skew_down'),
        ('5b full',      None),
    ]):
        sub = df[df[col] > 0] if col else df
        _direction_plot(sub, lbl, ax)
    save_fig(fig2, fig_dir / 'step5b_tail_risk.png')

    # ── 5c: Tense mask ────────────────────────────────────────────────────────
    print('\n  5c. Tense mask')
    if df_sentences is not None:
        for mask_label, tenses in [
            ('5c_forward_only',      ['forward']),
            ('5c_forward_present',   ['forward', 'present']),
        ]:
            df_t = df_sentences[df_sentences['ten'].isin(tenses)].copy()
            if len(df_t) > MIN_MASK_OBS:
                doc_t  = aggregate_to_document(df_t)
                df_t_m = merge_to_macro(doc_t)
                r = _rerun_best(df_t_m, best_result, mask_label)
                results[mask_label] = r

    # ── 5d: Commitment / Odyssean guidance mask ───────────────────────────────
    print('\n  5d. Commitment mask')
    doc_type_sets = {
        'all':      None,
        'minutes':  ['minutes'],
        'external': ['statements', 'press_conference_prepared'],
    }
    for dt_label, dt_filter in doc_type_sets.items():
        sub = df if dt_filter is None else df[df['doc_type'].isin(dt_filter)]
        for mask_label, col, cond_fn in [
            ('odyssean',  'flag_unconditional_forward', lambda d: d == 1),
            ('delphic',   'flag_unconditional_forward', lambda d: d == 0),
            ('conditional', 'flag_conditional',         lambda d: d > 0),
        ]:
            if col in sub.columns:
                cond = cond_fn(sub[col])
                full_label = f'5d_{mask_label}_{dt_label}'
                r = apply_mask(sub, cond, full_label, best_result)
                results[full_label] = r

    # ── 5e: Shadow rate direction mask (LOOKAHEAD) ────────────────────────────
    print('\n  5e. Shadow rate direction mask [LOOKAHEAD]')
    if TARGET_COL in df.columns:
        chg = df[TARGET_COL].diff()
        for mlabel, cond in [
            ('5e_up_LOOKAHEAD',   chg > 0.001),
            ('5e_down_LOOKAHEAD', chg < -0.001),
        ]:
            r = apply_mask(df, cond, mlabel, best_result)
            if r:
                r['lookahead'] = True
            results[mlabel] = r

    # ── 5f: Speaker mask (speeches only) ─────────────────────────────────────
    print('\n  5f. Speaker mask (speeches)')
    if df_sentences is not None and 'speaker_type' in df_sentences.columns:
        speaker_masks = {
            '5f_chair_only':        ['chair'],
            '5f_chair_vice_chair':  ['chair', 'vice_chair'],
            '5f_all_speakers':      ['chair', 'vice_chair', 'others'],
        }
        coef_rows = []
        for mlabel, speakers in speaker_masks.items():
            df_spk = df_sentences[
                (df_sentences['doc_type'] == 'speech') &
                (df_sentences['speaker_type'].isin(speakers))
            ].copy()
            if len(df_spk) < MIN_MASK_OBS:
                log_warn(f'{mlabel}: < {MIN_MASK_OBS} observations — skipped.')
                continue
            doc_spk  = aggregate_to_document(df_spk)
            df_spk_m = merge_to_macro(doc_spk)
            r = _rerun_best(df_spk_m, best_result, mlabel)
            results[mlabel] = r
            if r and r.get('_model'):
                m = r['_model']
                sent_params = {k: v for k, v in m.params.items()
                               if 'sent' in k.lower()}
                coef_rows.append({
                    'mask': mlabel,
                    'adj_r2': r['adj_r2'],
                    'oos_rmse': r['oos_rmse'],
                    **sent_params,
                })

        # Plot coefficient of sent_total across speaker masks
        if coef_rows:
            coef_df = pd.DataFrame(coef_rows)
            sent_col_name = next(
                (c for c in coef_df.columns if 'sent_total' in c), None
            )
            if sent_col_name:
                fig_s, ax_s = plt.subplots(figsize=(6, 4))
                masks  = coef_df['mask'].tolist()
                coefs  = coef_df[sent_col_name].tolist()
                # Approximate 95% CI from model
                ci_lo, ci_hi = [], []
                for mlabel in masks:
                    r = results.get(mlabel, {})
                    m = r.get('_model') if r else None
                    if m and sent_col_name in m.conf_int().index:
                        ci_lo.append(m.conf_int().loc[sent_col_name, 0])
                        ci_hi.append(m.conf_int().loc[sent_col_name, 1])
                    else:
                        ci_lo.append(np.nan)
                        ci_hi.append(np.nan)
                x = range(len(masks))
                ax_s.bar(x, coefs, color=COLOUR_PALETTE[0], alpha=0.7)
                for i in range(len(masks)):
                    if not np.isnan(ci_lo[i]):
                        ax_s.errorbar(i, coefs[i],
                                      yerr=[[coefs[i]-ci_lo[i]],[ci_hi[i]-coefs[i]]],
                                      fmt='none', color='black', capsize=5)
                ax_s.set_xticks(list(x))
                ax_s.set_xticklabels([m.replace('5f_','') for m in masks],
                                     fontsize=8)
                ax_s.axhline(0, color='black', linewidth=0.8, linestyle='--')
                ax_s.set_title('sent_total coefficient by speaker mask (95% CI)')
                ax_s.set_ylabel('Coefficient')
                save_fig(fig_s, OUTPUTS_DIR / 'step5f_speaker_mask.png')

            coef_df.to_csv(INTER_DIR / 'step5f_speaker_mask_results.csv', index=False)

    # Consolidate
    mask_summary = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith('_')}
        for r in results.values() if r
    ])
    mask_summary.to_csv(INTER_DIR / 'step5_mask_results.csv', index=False)
    return results
