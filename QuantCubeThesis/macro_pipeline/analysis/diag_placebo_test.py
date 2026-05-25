"""
Placebo / permutation test.

Shuffle the sentiment scores randomly across meetings (breaking temporal alignment)
and re-run the best joint-PCA model per method 500 times.

Null hypothesis: the temporal ordering of sentiment is irrelevant.
If the real CW-t/OOS-RMSE are extreme vs the null distribution → signal is genuine.

Reports:
  - Real OOS RMSE and CW-t
  - Permutation distribution (mean, std, p5, p95)
  - Empirical p-value: fraction of permutations with RMSE <= real (or CW-t >= real)
"""
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from config import TARGET_NEXT, MACRO_REGRESSORS, PCA_VAR_MACRO
from utils import run_pca, clark_west_test
from step0_aggregate import load_macro

root = r'C:\Users\Javier\OneDrive - HEC Paris\Documentos\QuantCubeThesis\QuantCubeThesis'

N_PERM   = 500
SEED     = 42
TRAIN_FRAC = 0.60   # expanding window starts here
OOS_CUTOFF = 0.80   # report RMSE over the last 20%

rng = np.random.default_rng(SEED)


# ── Expanding-window RMSE + CW predictions ───────────────────────────────────

def expanding_oos(df, pca_cols, target_col, train_frac=TRAIN_FRAC):
    """
    Full-sample PCA then expanding-window OLS.
    Returns (oos_rmse_80, preds) where preds = list of (t, y_actual, y_hat).
    """
    avail = [c for c in pca_cols if c in df.columns]
    sub   = df[avail + [target_col]].dropna().reset_index(drop=True)
    n     = len(sub)
    init  = int(n * train_frac)
    if init < 10 or init >= n:
        return np.nan, []

    factor_df, _, _, n_comp = run_pca(sub, avail, var_threshold=PCA_VAR_MACRO)
    factor_df.columns = [f'JPC{i+1}' for i in range(n_comp)]
    y = sub[target_col].values
    X = sm.add_constant(factor_df.values)

    errors, preds = [], []
    for t in range(init, n):
        m     = sm.OLS(y[:t], X[:t]).fit()
        y_hat = float(m.predict(X[t:t+1])[0])
        errors.append((t, (y[t] - y_hat) ** 2))
        preds.append((t, float(y[t]), y_hat))

    cutoff = int(n * OOS_CUTOFF)
    subset = [e for t, e in errors if t >= cutoff]
    rmse   = np.sqrt(np.mean(subset)) if subset else np.nan
    return rmse, preds


# ── Build baseline predictions once (shared across all methods) ──────────────

def get_baseline_preds(df):
    mc_avail = [c for c in MACRO_REGRESSORS if c in df.columns]
    _, preds = expanding_oos(df, mc_avail, TARGET_NEXT)
    return preds


# ── Panel builders ────────────────────────────────────────────────────────────

def build_panel(method):
    nlp_dir = root + r'\Taylor Rule\nlp_output'
    if method in ('LLM', 'Baseline'):
        df = pd.read_csv(
            root + r'\Taylor Rule\outputs\intermediate\step0_merged_to_macro.csv',
            parse_dates=['date'])
    elif method == 'Gardner':
        remap = {'gardner_total':'g_total','gardner_inf':'g_inflation',
                 'gardner_labor':'g_labor_mkt','gardner_out':'g_econ_act',
                 'gardner_fin':'g_fin_cond','gardner_mp':'g_mon_pol'}
        raw = pd.read_csv(
            nlp_dir + r'\Gardner_statements_minutes_speeches_press_conferences_zscore_nlp.csv',
            parse_dates=['meeting_date'])
        gcols = [c for c in raw.columns if c.startswith('gardner_') and 'consensus' not in c]
        agg = raw.groupby('meeting_date')[gcols].sum().reset_index().rename(
            columns={**{'meeting_date':'date'}, **remap})
        df = pd.merge_asof(load_macro().sort_values('date'),
                           agg.sort_values('date'), on='date', direction='backward')
    elif method == 'Sharpe':
        remap = {'sharpe_net':'s_net','sharpe_positive':'s_positive',
                 'sharpe_negative':'s_negative','sharpe_consensus':'s_consensus'}
        raw = pd.read_csv(
            nlp_dir + r'\Sharpe_statements_minutes_speeches_press_conferences_zscore_nlp.csv',
            parse_dates=['meeting_date'])
        scols = [c for c in raw.columns if c.startswith('sharpe_')]
        agg = raw.groupby('meeting_date')[scols].sum().reset_index().rename(
            columns={**{'meeting_date':'date'}, **remap})
        df = pd.merge_asof(load_macro().sort_values('date'),
                           agg.sort_values('date'), on='date', direction='backward')
        for mc_col in ['vix','gdp','unemployment_gap','inflation_dev_from_target',
                       'implied_ffr','cfnai']:
            if mc_col in df.columns:
                df[f's_net_x_{mc_col}'] = df['s_net'] * df[mc_col]
    return df[df['date'] >= '2007-01-01'].reset_index(drop=True)


def get_sentiment_cols(method):
    """Return the sentiment column(s) that get shuffled (NOT interactions — those
    are recomputed after shuffling the base sentiment)."""
    if method == 'LLM':
        return ['sent_total']
    elif method == 'Gardner':
        return ['g_total']
    elif method == 'Sharpe':
        return ['s_net']


def get_model_pca_cols(method, df):
    """Return the full PCA input columns for the best spec."""
    mc = [c for c in MACRO_REGRESSORS if c in df.columns]
    if method == 'LLM':
        return mc + ['sent_total']
    elif method == 'Gardner':
        return mc + ['g_total']
    elif method == 'Sharpe':
        inter = [f's_net_x_{c}' for c in
                 ['vix','gdp','unemployment_gap','inflation_dev_from_target',
                  'implied_ffr','cfnai'] if f's_net_x_{c}' in df.columns]
        return mc + ['s_net'] + inter


def rebuild_interactions(df, method):
    """After shuffling s_net, recompute interaction columns in-place."""
    if method != 'Sharpe':
        return
    for mc_col in ['vix','gdp','unemployment_gap','inflation_dev_from_target',
                   'implied_ffr','cfnai']:
        if mc_col in df.columns:
            df[f's_net_x_{mc_col}'] = df['s_net'] * df[mc_col]


# ── Single permutation run ────────────────────────────────────────────────────

def run_one_permutation(df_orig, sent_cols, method, baseline_preds):
    df = df_orig.copy()
    # Shuffle the sentiment column(s) — break temporal ordering
    for col in sent_cols:
        if col in df.columns:
            df[col] = rng.permutation(df[col].values)
    rebuild_interactions(df, method)
    pca_cols = get_model_pca_cols(method, df)
    rmse, preds = expanding_oos(df, pca_cols, TARGET_NEXT)
    cw_t, _ = clark_west_test(baseline_preds, preds)
    return rmse, cw_t


# ── Main ──────────────────────────────────────────────────────────────────────

methods = ['LLM', 'Gardner', 'Sharpe']

print('=' * 85)
print('  PLACEBO / PERMUTATION TEST')
print(f'  N_permutations={N_PERM}  |  expanding window from {int(TRAIN_FRAC*100)}%'
      f'  |  OOS RMSE reported at {int(OOS_CUTOFF*100)}% cutoff')
print('=' * 85)

all_rows = []

for method in methods:
    print(f'\n  [{method}] Loading panel…', end='', flush=True)
    df_real = build_panel(method)
    sent_cols  = get_sentiment_cols(method)
    pca_cols   = get_model_pca_cols(method, df_real)
    baseline_preds = get_baseline_preds(df_real)

    # Real model
    real_rmse, real_preds = expanding_oos(df_real, pca_cols, TARGET_NEXT)
    real_cw, _            = clark_west_test(baseline_preds, real_preds)

    # Permutation distribution
    perm_rmses, perm_cws = [], []
    print(f'  running {N_PERM} permutations…', end='', flush=True)
    for i in range(N_PERM):
        r, c = run_one_permutation(df_real, sent_cols, method, baseline_preds)
        perm_rmses.append(r)
        perm_cws.append(c)
        if (i + 1) % 100 == 0:
            print(f'{i+1}…', end='', flush=True)
    print(' done')

    perm_rmses = np.array([v for v in perm_rmses if not np.isnan(v)])
    perm_cws   = np.array([v for v in perm_cws   if not np.isnan(v)])

    # p-values
    # RMSE: fraction of permutations with RMSE <= real (model no better than random)
    p_rmse = np.mean(perm_rmses <= real_rmse)
    # CW-t: fraction of permutations with CW-t >= real (real t is unusually high)
    p_cw   = np.mean(perm_cws   >= real_cw)

    print(f'\n  --- {method} ---')
    print(f'  Real model:  OOS_RMSE_80={real_rmse:.4f}   CW-t={real_cw:.3f}')
    print(f'  Null dist (N={len(perm_rmses)}):')
    print(f'    RMSE   mean={np.mean(perm_rmses):.4f}  std={np.std(perm_rmses):.4f}'
          f'  p5={np.percentile(perm_rmses,5):.4f}  p95={np.percentile(perm_rmses,95):.4f}')
    print(f'    CW-t   mean={np.nanmean(perm_cws):.3f}  std={np.nanstd(perm_cws):.3f}'
          f'  p5={np.nanpercentile(perm_cws,5):.3f}  p95={np.nanpercentile(perm_cws,95):.3f}')
    print(f'  Empirical p-values:')
    print(f'    P(perm RMSE <= real)  = {p_rmse:.4f}  '
          f'{"<-- signal is NOT genuine (random is as good)" if p_rmse > 0.10 else "<-- genuine signal confirmed"}')
    print(f'    P(perm CW-t >= real)  = {p_cw:.4f}  '
          f'{"<-- signal is NOT genuine" if p_cw > 0.10 else "<-- real CW-t is extreme vs null"}')

    all_rows.append({
        'method': method,
        'real_oos_rmse_80': real_rmse,
        'real_cw_t': real_cw,
        'perm_rmse_mean': np.mean(perm_rmses),
        'perm_rmse_std': np.std(perm_rmses),
        'perm_rmse_p5': np.percentile(perm_rmses, 5),
        'perm_rmse_p95': np.percentile(perm_rmses, 95),
        'perm_cw_mean': np.nanmean(perm_cws),
        'perm_cw_std': np.nanstd(perm_cws),
        'perm_cw_p5': np.nanpercentile(perm_cws, 5),
        'perm_cw_p95': np.nanpercentile(perm_cws, 95),
        'p_val_rmse': p_rmse,
        'p_val_cw': p_cw,
        'n_perm': len(perm_rmses),
    })

# ── Summary table ─────────────────────────────────────────────────────────────
print()
print('=' * 85)
print('  SUMMARY')
print('=' * 85)
print(f'  {"Method":<10} {"Real RMSE":>10} {"Null mean":>10} {"Null p95":>10}'
      f' {"p-val(RMSE)":>12} {"Real CW-t":>10} {"p-val(CW)":>11}')
print('  ' + '-' * 80)
for r in all_rows:
    print(f'  {r["method"]:<10} {r["real_oos_rmse_80"]:>10.4f} {r["perm_rmse_mean"]:>10.4f}'
          f' {r["perm_rmse_p95"]:>10.4f} {r["p_val_rmse"]:>12.4f}'
          f' {r["real_cw_t"]:>10.3f} {r["p_val_cw"]:>11.4f}')

out = root + r'\Taylor Rule\outputs\placebo_test.csv'
pd.DataFrame(all_rows).to_csv(out, index=False)
print(f'\n  Saved -> {out}')
