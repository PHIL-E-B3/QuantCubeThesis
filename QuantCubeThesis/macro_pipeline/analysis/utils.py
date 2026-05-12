"""
analysis/utils.py — shared helpers: model fitting, metrics, OOS CV, PCA, plots.
"""

import warnings
import logging
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing   import StandardScaler
from sklearn.decomposition   import PCA
from scipy.stats             import chi2

from config import (
    RANDOM_SEED, FIG_DPI, OOS_MIN_FRAC, COLOUR_PALETTE,
    WARNINGS_LOG, MIN_MASK_OBS, OUTPUTS_DIR,
)

np.random.seed(RANDOM_SEED)

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=str(WARNINGS_LOG),
    level=logging.WARNING,
    format='%(asctime)s  %(levelname)s  %(message)s',
)

def log_warn(msg: str):
    logging.warning(msg)
    warnings.warn(msg)


# ── PCA ───────────────────────────────────────────────────────────────────────

def run_pca(df: pd.DataFrame, cols: list, var_threshold: float = 0.90) -> tuple:
    """
    Standardise `cols` in df, fit PCA, retain components explaining ≥ var_threshold
    cumulative variance.
    Returns (factors_df, pca_obj, scaler_obj, n_components).
    """
    X = df[cols].dropna()
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca     = PCA(random_state=RANDOM_SEED)
    pca.fit(X_scaled)
    cum_var  = np.cumsum(pca.explained_variance_ratio_)
    n_comp   = int(np.searchsorted(cum_var, var_threshold)) + 1
    factors  = pca.transform(X_scaled)[:, :n_comp]
    factor_df = pd.DataFrame(
        factors,
        index=X.index,
        columns=[f'PC{i+1}' for i in range(n_comp)]
    )
    return factor_df, pca, scaler, n_comp


# ── OLS with HC3 ──────────────────────────────────────────────────────────────

def fit_ols(y: pd.Series, X: pd.DataFrame,
            label: str = '', add_const: bool = True) -> dict:
    """
    Fit OLS with HC3-robust standard errors.
    Returns dict of metrics + fitted model.
    """
    df_clean = pd.concat([y, X], axis=1).dropna()
    if len(df_clean) < MIN_MASK_OBS:
        log_warn(f'[{label}] Only {len(df_clean)} obs — skipping.')
        return {}
    y_c = df_clean.iloc[:, 0]
    X_c = df_clean.iloc[:, 1:]
    if add_const:
        X_c = sm.add_constant(X_c)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = sm.OLS(y_c, X_c).fit(cov_type='HC3')
    return {
        'model_label':   label,
        'n_obs':         int(model.nobs),
        'r2':            round(model.rsquared,     4),
        'adj_r2':        round(model.rsquared_adj, 4),
        'aic':           round(model.aic,          2),
        'bic':           round(model.bic,          2),
        'f_stat':        round(model.fvalue,       4) if model.fvalue else np.nan,
        'f_pvalue':      round(model.f_pvalue,     6) if model.f_pvalue else np.nan,
        'oos_rmse':      oos_rmse(y_c, X_c, label),
        '_model':        model,
    }


def lrt(model_restricted, model_full) -> tuple:
    """Likelihood Ratio Test: returns (statistic, pvalue)."""
    lr_stat = 2 * (model_full.llf - model_restricted.llf)
    df_diff = model_full.df_model - model_restricted.df_model
    pval    = chi2.sf(lr_stat, df_diff) if df_diff > 0 else np.nan
    return round(lr_stat, 4), round(pval, 6)


def vif(X: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factors for each column in X."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    Xc = sm.add_constant(X.dropna())
    data = []
    for i, col in enumerate(Xc.columns):
        if col == 'const':
            continue
        try:
            v = variance_inflation_factor(Xc.values, i)
        except Exception:
            v = np.nan
        data.append({'feature': col, 'VIF': round(v, 2)})
    return pd.DataFrame(data)


# ── OOS expanding-window RMSE ─────────────────────────────────────────────────

def oos_rmse(y: pd.Series, X: pd.DataFrame, label: str = '') -> float:
    n = len(y)
    init = max(int(n * OOS_MIN_FRAC), 10)
    preds = []
    for t in range(init, n):
        y_train, X_train = y.iloc[:t], X.iloc[:t]
        y_test,  X_test  = y.iloc[t],  X.iloc[[t]]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m = sm.OLS(y_train, X_train).fit()
            preds.append((y_test - m.predict(X_test)[0]) ** 2)
        except Exception:
            pass
    if not preds:
        return np.nan
    return round(np.sqrt(np.mean(preds)), 6)


# ── Model comparison table ────────────────────────────────────────────────────

def save_model_row(result: dict, csv_path=None):
    """Append one model result dict to model_comparison.csv."""
    if not result:
        return
    csv_path = csv_path or OUTPUTS_DIR / 'model_comparison.csv'
    row = {k: v for k, v in result.items() if not k.startswith('_')}
    df  = pd.DataFrame([row])
    if csv_path.exists():
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


def print_metrics(result: dict):
    if not result:
        return
    keys = ['model_label','n_obs','r2','adj_r2','aic','bic','f_stat','f_pvalue','oos_rmse']
    row  = {k: result.get(k,'') for k in keys}
    print(f"\n  {'Model':<45} {'N':>5}  {'R²':>6}  {'AdjR²':>6}  "
          f"{'AIC':>8}  {'F':>8}  {'Fp':>8}  {'OOS RMSE':>10}")
    print(f"  {'─'*105}")
    print(f"  {row['model_label']:<45} {row['n_obs']:>5}  {row['r2']:>6.4f}  "
          f"{row['adj_r2']:>6.4f}  {row['aic']:>8.1f}  "
          f"{row['f_stat']:>8.3f}  {row['f_pvalue']:>8.4f}  {row['oos_rmse']:>10.6f}")


# ── Standard plot helpers ──────────────────────────────────────────────────────

def new_fig(n_subplots: int = 1, cols: int = 2, figsize_per: tuple = (7, 3.5)):
    rows   = (n_subplots + cols - 1) // cols
    figsize = (figsize_per[0] * cols, figsize_per[1] * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_subplots == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()
    return fig, axes


def save_fig(fig, path, tight: bool = True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    print(f'  Figure saved: {path}')


def shadow_rate_direction_barplot(df: pd.DataFrame, target_col: str,
                                  title: str, ax=None):
    """Bar chart of up/flat/down distribution of the target variable."""
    chg = df[target_col].diff()
    labels = pd.cut(chg, bins=[-np.inf, -0.001, 0.001, np.inf],
                    labels=['Down', 'Flat', 'Up'])
    counts = labels.value_counts().reindex(['Down','Flat','Up'], fill_value=0)
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))
    ax.bar(counts.index, counts.values,
           color=[COLOUR_PALETTE[1], COLOUR_PALETTE[7], COLOUR_PALETTE[2]])
    ax.set_title(title, fontsize=9)
    ax.set_ylabel('# meetings')
    return ax
