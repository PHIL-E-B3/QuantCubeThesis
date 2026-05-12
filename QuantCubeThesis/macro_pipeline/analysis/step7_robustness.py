"""
step7_robustness.py — Robustness checks (7a–7d).
"""

import sys
import warnings
from pathlib import Path

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    OUTPUTS_DIR, INTER_DIR, TARGET_NEXT, TARGET_COL,
    RANDOM_SEED, COLOUR_PALETTE, MIN_MASK_OBS,
)
from utils import fit_ols, save_model_row, print_metrics, log_warn, save_fig

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_SEED)


def run_robustness(df: pd.DataFrame, best_result: dict) -> dict:
    """Run all robustness checks (7a–7d). Returns dict of results."""
    results   = {}
    factor_df = best_result.get('_factor_df', pd.DataFrame())
    best_model = best_result.get('_model')

    # ── 7a: Swap DV to fed_funds_rate ─────────────────────────────────────────
    print('\n  7a. Swap DV to fed_funds_rate(t+1)')
    if 'fed_funds_rate' in df.columns:
        df_7a = df.copy()
        df_7a['ffr_next'] = df_7a['fed_funds_rate'].shift(-1)
        target_7a = df_7a['ffr_next']
        fdf_sub   = factor_df.loc[factor_df.index.intersection(df_7a.index)]
        extra     = _get_best_extra_cols(best_result, df_7a)
        X_7a      = fdf_sub.join(df_7a[extra], how='left').dropna()
        y_7a      = target_7a.loc[X_7a.index].dropna()
        X_7a      = X_7a.loc[y_7a.index]
        r = fit_ols(y_7a, X_7a, label='7a_ffr_dv')
        if r:
            print_metrics(r)
            save_model_row(r)
        results['7a'] = r

    # ── 7b: Rolling window stability ──────────────────────────────────────────
    print('\n  7b. Rolling window stability (5-year window)')
    window    = 20  # ~5 years at 4 meetings/year
    target    = df[TARGET_NEXT]
    extra     = _get_best_extra_cols(best_result, df)
    fdf_aug   = factor_df.join(df[extra], how='left').dropna()
    y_all     = target.loc[fdf_aug.index].dropna()
    X_all     = sm.add_constant(fdf_aug.loc[y_all.index])

    coef_rows = []
    sent_cols = [c for c in X_all.columns if 'sent' in c.lower()]
    if not sent_cols:
        sent_cols = [X_all.columns[-1]]  # fallback

    for t in range(window, len(X_all)):
        X_w = X_all.iloc[t-window:t]
        y_w = y_all.iloc[t-window:t]
        try:
            m = sm.OLS(y_w, X_w).fit(cov_type='HC3')
            row = {'t': X_all.index[t]}
            for sc in sent_cols:
                if sc in m.params:
                    row[f'coef_{sc}']   = m.params[sc]
                    row[f'ci_lo_{sc}']  = m.conf_int().loc[sc, 0]
                    row[f'ci_hi_{sc}']  = m.conf_int().loc[sc, 1]
            coef_rows.append(row)
        except Exception:
            pass

    if coef_rows:
        coef_df = pd.DataFrame(coef_rows).set_index('t')
        coef_df.to_csv(INTER_DIR / 'step7b_rolling_coefficients.csv')

        n_sc = len(sent_cols)
        fig, axes = plt.subplots(n_sc, 1, figsize=(12, 4*n_sc))
        if n_sc == 1:
            axes = [axes]
        for ax, sc in zip(axes, sent_cols):
            c_col  = f'coef_{sc}'
            lo_col = f'ci_lo_{sc}'
            hi_col = f'ci_hi_{sc}'
            if c_col not in coef_df:
                continue
            ax.plot(coef_df.index, coef_df[c_col],
                    color=COLOUR_PALETTE[0], linewidth=1.5, label=sc)
            if lo_col in coef_df:
                ax.fill_between(coef_df.index, coef_df[lo_col], coef_df[hi_col],
                                alpha=0.2, color=COLOUR_PALETTE[0])
            ax.axhline(0, linestyle='--', color='grey', linewidth=0.8)
            ax.set_title(f'Rolling coefficient: {sc} (5-yr window)')
            ax.set_ylabel('Coefficient')
            ax.legend(fontsize=8)
        save_fig(fig, OUTPUTS_DIR / 'rolling_coefficients.png')

    # ── 7c: Leave-one-out by doc_type ─────────────────────────────────────────
    print('\n  7c. Leave-one-out by doc_type')
    if 'doc_type' in df.columns:
        loo_rows = []
        doc_types = df['doc_type'].dropna().unique()
        for dt in doc_types:
            df_loo   = df[df['doc_type'] != dt]
            fdf_sub  = factor_df.loc[factor_df.index.intersection(df_loo.index)]
            extra_loo = _get_best_extra_cols(best_result, df_loo)
            X_loo    = fdf_sub.join(df_loo[extra_loo], how='left').dropna()
            y_loo    = target.loc[X_loo.index].dropna()
            X_loo    = X_loo.loc[y_loo.index]
            r = fit_ols(y_loo, X_loo, label=f'7c_excl_{dt}')
            if r:
                loo_rows.append({'excl_doc_type': dt, 'adj_r2': r['adj_r2'],
                                  'n_obs': r['n_obs']})
                save_model_row(r)
        if loo_rows:
            loo_df = pd.DataFrame(loo_rows)
            loo_df.to_csv(INTER_DIR / 'step7c_loo_results.csv', index=False)
            print(loo_df.to_string(index=False))
        results['7c'] = loo_rows

    # ── 7d: Placebo test ──────────────────────────────────────────────────────
    print('\n  7d. Placebo test (500 shuffles, seed=42)')
    N_BOOT = 500
    rng    = np.random.default_rng(RANDOM_SEED)

    extra_p   = _get_best_extra_cols(best_result, df)
    fdf_p_aug = factor_df.join(df[extra_p], how='left').dropna()
    y_p       = target.loc[fdf_p_aug.index].dropna()
    X_p       = fdf_p_aug.loc[y_p.index]

    if 'sent_total' in X_p.columns:
        real_adj_r2 = fit_ols(y_p, X_p, label='7d_observed').get('adj_r2', np.nan)
        null_adj_r2 = []
        for _ in range(N_BOOT):
            X_sh = X_p.copy()
            X_sh['sent_total'] = rng.permutation(X_sh['sent_total'].values)
            r_sh = fit_ols(y_p, X_sh, label='_placebo')
            if r_sh:
                null_adj_r2.append(r_sh['adj_r2'])

        if null_adj_r2:
            emp_pval = np.mean(np.array(null_adj_r2) >= real_adj_r2)
            print(f'  Observed Adj R² = {real_adj_r2:.4f}  '
                  f'| Empirical p-value = {emp_pval:.4f}')

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(null_adj_r2, bins=40, color=COLOUR_PALETTE[7], alpha=0.7,
                    edgecolor='white', label=f'Null ({N_BOOT} shuffles)')
            ax.axvline(real_adj_r2, color=COLOUR_PALETTE[1], linewidth=2,
                       label=f'Observed = {real_adj_r2:.4f} (p={emp_pval:.3f})')
            ax.set_xlabel('Adjusted R²')
            ax.set_ylabel('Frequency')
            ax.set_title('Placebo test: shuffled sent_total')
            ax.legend()
            save_fig(fig, OUTPUTS_DIR / 'placebo_test.png')

            pd.DataFrame({'null_adj_r2': null_adj_r2}).to_csv(
                INTER_DIR / 'step7d_placebo_null.csv', index=False
            )
            results['7d'] = {
                'observed_adj_r2': real_adj_r2,
                'empirical_pvalue': emp_pval,
                'n_boot': N_BOOT,
            }

    return results


def _get_best_extra_cols(best_result: dict, df: pd.DataFrame) -> list:
    """Extract the extra (non-PC) column names used in the best model."""
    m = best_result.get('_model')
    if m is None:
        return [c for c in ['sent_total'] if c in df.columns]
    factor_names = set(best_result.get('_factor_df', pd.DataFrame()).columns)
    return [c for c in m.model.exog_names
            if c != 'const' and c not in factor_names and c in df.columns]
