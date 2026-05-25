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
    RANDOM_SEED, FIG_DPI, OOS_FRACS, COLOUR_PALETTE,
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
    oos = oos_rmse(y_c, X_c, label)
    result = {
        'model_label':   label,
        'n_obs':         int(model.nobs),
        'r2':            round(model.rsquared,     4),
        'adj_r2':        round(model.rsquared_adj, 4),
        'aic':           round(model.aic,          2),
        'bic':           round(model.bic,          2),
        'f_stat':        round(model.fvalue,       4) if model.fvalue else np.nan,
        'f_pvalue':      round(model.f_pvalue,     6) if model.f_pvalue else np.nan,
        '_model':        model,
        '_oos_preds':    oos.pop('_preds', []),
    }
    for frac, val in oos.items():
        result[f'oos_rmse_{frac}'] = val
    return result


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

def oos_rmse(y: pd.Series, X: pd.DataFrame, label: str = '') -> dict:
    """
    Expanding-window OOS RMSE at multiple training-size thresholds (OOS_FRACS).

    One expanding-window pass starting at min(OOS_FRACS) collects all
    out-of-sample squared errors.  Each threshold fraction f then reports
    RMSE over the tail of the OOS period that begins at row int(n*f):

        oos_rmse_60  →  RMSE over all predictions (training starts at 60%)
        oos_rmse_70  →  RMSE over the last 30% of predictions
        oos_rmse_80  →  RMSE over the last 20% of predictions
        oos_rmse_90  →  RMSE over the last 10% of predictions

    This allows checking whether predictive accuracy is stable or deteriorates
    in later sub-periods of the OOS window.

    Returns dict keyed by integer threshold (e.g. {60: rmse, 70: rmse, ...}).
    HC3 does not affect predictions (only SEs), included for consistency.
    """
    n = len(y)
    init = max(int(n * min(OOS_FRACS)), 10)
    errors = []    # list of (t, sq_error)
    preds  = []    # list of (t, y_actual, y_hat)  — used by Clark-West test
    n_skipped = 0
    for t in range(init, n):
        y_train, X_train = y.iloc[:t], X.iloc[:t]
        y_test,  X_test  = y.iloc[t],  X.iloc[[t]]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m = sm.OLS(y_train, X_train).fit(cov_type='HC3')
            y_hat = m.predict(X_test).iloc[0]
            errors.append((t, (y_test - y_hat) ** 2))
            preds.append((t, float(y_test), float(y_hat)))
        except Exception as exc:
            n_skipped += 1
            log_warn(f'[{label}] OOS window t={t} skipped: {exc}')
    if n_skipped:
        log_warn(f'[{label}] {n_skipped}/{n - init} OOS windows skipped — RMSE based on {len(errors)} predictions.')
    result = {'_preds': preds}
    for frac in OOS_FRACS:
        key = int(frac * 100)
        cutoff = int(n * frac)
        subset = [e for t, e in errors if t >= cutoff]
        result[key] = round(np.sqrt(np.mean(subset)), 6) if subset else np.nan
    return result


# ── Clark-West (2007) OOS significance test ───────────────────────────────────

def diebold_mariano_test(baseline_preds: list, model_preds: list) -> tuple:
    """
    Diebold-Mariano (1995) test for equal predictive accuracy.

    Valid for both nested and non-nested models.  Use this instead of
    clark_west_test when the model and baseline are non-nested (e.g. joint
    PCA vs macro-only PCA, where you cannot recover the baseline by zeroing
    coefficients in the augmented model).

    H0: equal MSPE.
    H1: augmented model has lower MSPE (one-sided).

    d_t = e_baseline_t² - e_model_t²
    DM  = mean(d) / HAC_se(d)  ~  N(0,1) under H0.

    Unlike CW there is no (ŷ_b - ŷ_m)² adjustment term, so the test is
    unbiased but potentially conservative when models are in fact nested.

    Returns
    -------
    (t_stat, p_value) — p_value is one-sided; NaN when model is worse on MSPE.
    """
    from scipy.stats import norm as _norm

    base_dict  = {t: (y, yh) for t, y, yh in baseline_preds}
    model_dict = {t: (y, yh) for t, y, yh in model_preds}
    common_t   = sorted(set(base_dict) & set(model_dict))

    if len(common_t) < 10:
        return np.nan, np.nan

    d = []
    for t in common_t:
        y,  yh_b = base_dict[t]
        _y, yh_m = model_dict[t]
        e_b = y - yh_b
        e_m = y - yh_m
        d.append(e_b**2 - e_m**2)

    d = np.array(d)
    T = len(d)
    model_beats_base = np.mean(d) > 0

    bw = max(1, int(np.ceil(T ** 0.25)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = sm.OLS(d, np.ones(T)).fit(cov_type='HAC',
                                         cov_kwds={'maxlags': bw, 'use_correction': True})
    t_stat = float(res.tvalues[0])
    if not model_beats_base:
        return round(t_stat, 4), np.nan

    p_val = float(_norm.sf(t_stat))
    return round(t_stat, 4), round(p_val, 4)


def clark_west_test(baseline_preds: list, model_preds: list) -> tuple:
    """
    Clark-West (2007) test for equal predictive accuracy of nested models.

    Appropriate when the augmented model nests the baseline (i.e. baseline is
    a restricted version of the augmented model).  The standard Diebold-Mariano
    test is biased in this case because under H0 the unrestricted model adds
    estimation noise; CW corrects for this via the adjustment term.

    H0: augmented model does NOT predict better than baseline (equal MSPE).
    H1: augmented model predicts better  (one-sided, lower MSPE).

    f_t = e_baseline_t² - [e_augmented_t² - (ŷ_baseline_t - ŷ_augmented_t)²]

    t_CW = mean(f) / HAC_se(f)  ~  N(0,1) under H0 (one-sided rejection for t_CW > 0).

    Parameters
    ----------
    baseline_preds, model_preds : list of (t, y_actual, y_hat) from oos_rmse().

    Returns
    -------
    (t_stat, p_value) — p_value is one-sided (small = augmented model is better).
    """
    from scipy.stats import norm as _norm

    base_dict  = {t: (y, yh) for t, y, yh in baseline_preds}
    model_dict = {t: (y, yh) for t, y, yh in model_preds}
    common_t   = sorted(set(base_dict) & set(model_dict))

    if len(common_t) < 10:
        return np.nan, np.nan

    f = []
    for t in common_t:
        y,   yh_b = base_dict[t]
        _y,  yh_m = model_dict[t]
        e_b = y - yh_b
        e_m = y - yh_m
        f.append(e_b**2 - e_m**2 + (yh_b - yh_m)**2)

    f = np.array(f)
    T = len(f)

    # Sanity check: if the augmented model has higher raw MSPE than the baseline,
    # the CW correction term can still produce a positive t-stat (known inflation
    # when prediction differences are large).  Flag these cases so callers can
    # treat the result with caution.
    e_base_sq = np.array([(base_dict[t][0] - base_dict[t][1])**2 for t in common_t])
    e_mod_sq  = np.array([(model_dict[t][0] - model_dict[t][1])**2 for t in common_t])
    model_beats_base = np.mean(e_mod_sq) < np.mean(e_base_sq)

    # HAC (Newey-West) standard error via OLS of f on a constant
    ones = np.ones(T)
    bw   = max(1, int(np.ceil(T ** 0.25)))   # bandwidth ~ T^(1/4)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = sm.OLS(f, ones).fit(cov_type='HAC',
                                  cov_kwds={'maxlags': bw, 'use_correction': True})
    t_stat = float(res.tvalues[0])
    # Return NaN p-value when model is actually worse — CW result is not meaningful
    if not model_beats_base:
        return round(t_stat, 4), np.nan

    p_val = float(_norm.sf(t_stat))   # one-sided: P(Z > t_stat)
    return round(t_stat, 4), round(p_val, 4)


# ── Model comparison table ────────────────────────────────────────────────────

_MODEL_CSV_COLS = (
    ['model_label', 'n_obs', 'r2', 'adj_r2', 'aic', 'bic', 'f_stat', 'f_pvalue']
    + [f'oos_rmse_{int(f*100)}' for f in OOS_FRACS]
    + ['cw_tstat', 'cw_pval']
)

def save_model_row(result: dict, csv_path=None):
    """Append one model result dict to model_comparison.csv (fixed schema)."""
    if not result:
        return
    csv_path = csv_path or OUTPUTS_DIR / 'model_comparison.csv'
    # Enforce fixed column set — extra keys (e.g. 'lookahead') are dropped,
    # missing ones get NaN, so every row has the same width.
    row = {col: result.get(col, np.nan) for col in _MODEL_CSV_COLS}
    df  = pd.DataFrame([row])
    if csv_path.exists():
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


def print_metrics(result: dict):
    if not result:
        return
    oos_keys = [f'oos_rmse_{int(f*100)}' for f in OOS_FRACS]
    oos_hdrs = [f'OOS{int(f*100)}' for f in OOS_FRACS]
    oos_vals = [result.get(k, float('nan')) for k in oos_keys]
    oos_header = '  '.join(f'{h:>9}' for h in oos_hdrs)
    oos_row    = '  '.join(
        f'{v:>9.6f}' if not (isinstance(v, float) and np.isnan(v)) else f'{"nan":>9}'
        for v in oos_vals
    )
    print(f"\n  {'Model':<45} {'N':>5}  {'R2':>6}  {'AdjR2':>6}  "
          f"{'AIC':>8}  {'BIC':>8}  {'F':>8}  {'Fp':>8}  {oos_header}")
    print(f"  {'-'*135}")
    print(f"  {result.get('model_label',''):<45} {result.get('n_obs',0):>5}  "
          f"{result.get('r2',0):>6.4f}  {result.get('adj_r2',0):>6.4f}  "
          f"{result.get('aic',0):>8.1f}  {result.get('bic',0):>8.1f}  "
          f"{result.get('f_stat',0):>8.3f}  {result.get('f_pvalue',0):>8.4f}  {oos_row}")


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
