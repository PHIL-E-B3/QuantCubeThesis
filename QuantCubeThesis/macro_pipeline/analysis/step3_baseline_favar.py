"""
step3_baseline_favar.py — Baseline FAVAR (macro PCA → OLS → target_next).
"""

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import MACRO_REGRESSORS, TARGET_NEXT, PCA_VAR_MACRO, INTER_DIR
from utils  import run_pca, fit_ols, save_model_row, print_metrics, log_warn

warnings.filterwarnings('ignore')


def run_baseline(df: pd.DataFrame) -> dict:
    """
    Fit baseline FAVAR: PCA on macro regressors → OLS on effective_rate(t+1).
    Returns result dict (includes _model and _factor_df keys).
    """
    avail = [c for c in MACRO_REGRESSORS if c in df.columns]
    if len(avail) < 2:
        log_warn('Baseline FAVAR: fewer than 2 macro regressors available.')
        return {}

    factor_df, pca, scaler, n_comp = run_pca(df, avail, var_threshold=PCA_VAR_MACRO)
    print(f'  Baseline PCA: {n_comp} components explain '
          f'{pca.explained_variance_ratio_[:n_comp].sum():.1%} variance '
          f'(threshold {PCA_VAR_MACRO:.0%})')

    target = df.loc[factor_df.index, TARGET_NEXT]
    result = fit_ols(target, factor_df, label='baseline_favar')
    if not result:
        return {}

    result['_factor_df'] = factor_df
    result['_pca']       = pca
    result['_scaler']    = scaler
    result['_avail_regressors'] = avail

    print_metrics(result)
    save_model_row(result)
    factor_df.to_csv(INTER_DIR / 'step3_baseline_factors.csv')
    return result
