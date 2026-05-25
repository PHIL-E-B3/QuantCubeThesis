"""
Robustness: OOS performance of best model per method across train/test splits.

Two approaches:
  A) Expanding-window RMSE at different cutoffs (60/70/80/90) — already computed
     by the pipeline. oos_rmse_X = RMSE over predictions from t>=X% onwards,
     where the OLS is re-fitted at each step using data up to t.
     This tests: "does the model stay good as we shorten the test window?"

  B) Fixed single-split: train on first X%, fit ONE model, predict remaining.
     No re-fitting. Tests whether the model generalises from a fixed training set.
     This is harsher and more traditional.
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

SPLITS = [0.60, 0.70, 0.80, 0.90]


# ── Helper: fixed single-split RMSE ──────────────────────────────────────────

def fixed_split_rmse(df, pca_cols, target_col, train_frac):
    """Train on first train_frac of data, predict the rest. Single model."""
    avail = [c for c in pca_cols if c in df.columns]
    sub = df[avail + [target_col]].dropna().reset_index(drop=True)
    n_train = int(len(sub) * train_frac)
    if n_train < 10 or n_train >= len(sub):
        return np.nan, np.nan

    # Full-sample PCA (standard FAVAR practice)
    factor_df, _, _, n_comp = run_pca(sub, avail, var_threshold=PCA_VAR_MACRO)
    factor_df.columns = [f'JPC{i+1}' for i in range(n_comp)]

    y     = sub[target_col].values
    X_reg = sm.add_constant(factor_df.values)

    # Train on first n_train rows
    model = sm.OLS(y[:n_train], X_reg[:n_train]).fit()
    y_hat = model.predict(X_reg[n_train:])
    y_act = y[n_train:]

    rmse = np.sqrt(np.mean((y_act - y_hat) ** 2))

    # CW test vs baseline (macro only)
    mc_avail = [c for c in MACRO_REGRESSORS if c in sub.columns]
    fac_base, _, _, nb = run_pca(sub, mc_avail, var_threshold=PCA_VAR_MACRO)
    fac_base.columns = [f'PC{i+1}' for i in range(nb)]
    X_base = sm.add_constant(fac_base.values)
    m_base  = sm.OLS(y[:n_train], X_base[:n_train]).fit()
    y_base  = m_base.predict(X_base[n_train:])

    # CW preds format: list of (t, y_actual, y_hat)
    base_preds  = [(t, float(y_act[t]), float(y_base[t]))  for t in range(len(y_act))]
    model_preds = [(t, float(y_act[t]), float(y_hat[t]))   for t in range(len(y_act))]
    cw_t, cw_p  = clark_west_test(base_preds, model_preds)

    return rmse, cw_t


# ── Best-model definitions ────────────────────────────────────────────────────

def get_best_model_cols(method):
    mc = [c for c in MACRO_REGRESSORS]
    if method == 'LLM':
        return mc + ['sent_total']   # 4h_llm_total (OOS80=0.255)
    elif method == 'Gardner':
        return mc + ['g_total']      # 4h_gard_total (OOS80=0.403)
    elif method == 'Sharpe':
        # 4h_sharpe_total_inter (OOS80=0.372)
        return mc + ['s_net'] + [f's_net_x_{c}' for c in
                                  ['vix','gdp','unemployment_gap',
                                   'inflation_dev_from_target','implied_ffr','cfnai']]
    elif method == 'Baseline':
        return mc


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
        raw = pd.read_csv(nlp_dir + r'\Gardner_statements_minutes_speeches_press_conferences_zscore_nlp.csv',
                          parse_dates=['meeting_date'])
        gcols = [c for c in raw.columns if c.startswith('gardner_') and 'consensus' not in c]
        agg = raw.groupby('meeting_date')[gcols].sum().reset_index().rename(
            columns={**{'meeting_date':'date'},**remap})
        df = pd.merge_asof(load_macro().sort_values('date'),
                           agg.sort_values('date'), on='date', direction='backward')
    elif method == 'Sharpe':
        remap = {'sharpe_net':'s_net','sharpe_positive':'s_positive',
                 'sharpe_negative':'s_negative','sharpe_consensus':'s_consensus'}
        raw = pd.read_csv(nlp_dir + r'\Sharpe_statements_minutes_speeches_press_conferences_zscore_nlp.csv',
                          parse_dates=['meeting_date'])
        scols = [c for c in raw.columns if c.startswith('sharpe_')]
        agg = raw.groupby('meeting_date')[scols].sum().reset_index().rename(
            columns={**{'meeting_date':'date'},**remap})
        df = pd.merge_asof(load_macro().sort_values('date'),
                           agg.sort_values('date'), on='date', direction='backward')
        # Add Sharpe interactions
        for mc_col in ['vix','gdp','unemployment_gap','inflation_dev_from_target',
                       'implied_ffr','cfnai']:
            if mc_col in df.columns:
                df[f's_net_x_{mc_col}'] = df['s_net'] * df[mc_col]

    return df[df['date'] >= '2007-01-01'].reset_index(drop=True)


# ── Pull expanding-window results already in CSV ──────────────────────────────

def get_expanding_results(method):
    """Read already-computed OOS fracs from saved CSVs."""
    tag = '_200701'
    prefix_map = {'LLM':'llm','Gardner':'gard','Sharpe':'sharpe','Baseline':'base'}
    p = prefix_map[method]
    best_variant = {'LLM':'total','Gardner':'total','Sharpe':'total_inter','Baseline':'base'}[method]
    label = f'4h_{p}_{best_variant}'

    path = root + fr'\Taylor Rule\outputs\4h_{p}_joint_pca{tag}.csv'
    try:
        df = pd.read_csv(path)
        row = df[df['model_label'] == label]
        if row.empty:
            return None
        row = row.iloc[0]
        return {f: row.get(f'oos_rmse_{int(f*100)}', np.nan) for f in SPLITS}
    except Exception:
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

methods = ['Baseline', 'LLM', 'Gardner', 'Sharpe']

print('=' * 85)
print('  ROBUSTNESS: OOS RMSE across train/test splits')
print('  N=151  (post-2007)')
print('=' * 85)

# ── A: Expanding-window (already computed) ────────────────────────────────────
print()
print('  A. Expanding-window OOS (re-fit OLS at each step, PCA fixed full-sample)')
print('     oos_rmse_X = RMSE over predictions starting from X% of sample')
print()
print(f'  {"Method":<12} {"Best spec":<28}', end='')
for f in SPLITS:
    print(f'  OOS{int(f*100):>2}', end='')
print()
print('  ' + '-' * 75)

exp_results = {}
for method in methods:
    r = get_expanding_results(method)
    best_spec = {'LLM':'joint_pca_total','Gardner':'joint_pca_total',
                 'Sharpe':'joint_pca_total_inter','Baseline':'macro_only'}[method]
    if r:
        exp_results[method] = r
        vals = ''.join(f'  {r[f]:>6.3f}' for f in SPLITS)
        print(f'  {method:<12} {best_spec:<28}{vals}')
    else:
        print(f'  {method:<12} {best_spec:<28}  (not found in CSV — re-running)')

# ── B: Fixed single-split ─────────────────────────────────────────────────────
print()
print('  B. Fixed single-split OOS (one model trained on first X%, predicts rest)')
print('     Harsher test — no re-fitting as new data arrives')
print()
print(f'  {"Method":<12} {"Best spec":<28}', end='')
for f in SPLITS:
    print(f'  OOS{int(f*100):>2}', end='')
print(f'  {"CW-t@80":>9}')
print('  ' + '-' * 85)

fixed_results = {}
for method in methods:
    df = build_panel(method)
    pca_cols = get_best_model_cols(method)
    best_spec = {'LLM':'joint_pca_total','Gardner':'joint_pca_total',
                 'Sharpe':'joint_pca_total_inter','Baseline':'macro_only'}[method]
    row_rmses = []
    cw80 = np.nan
    for f in SPLITS:
        rmse, cw_t = fixed_split_rmse(df, pca_cols, TARGET_NEXT, f)
        row_rmses.append(rmse)
        if f == 0.80:
            cw80 = cw_t
    fixed_results[method] = dict(zip(SPLITS, row_rmses))
    vals = ''.join(f'  {v:>6.3f}' if not np.isnan(v) else '     n/a' for v in row_rmses)
    print(f'  {method:<12} {best_spec:<28}{vals}  {cw80:>9.3f}')

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print('  C. Improvement of best LLM over Baseline (fixed split, % RMSE reduction)')
print()
print(f'  {"Split":<8}', end='')
for m in ['LLM','Gardner','Sharpe']:
    print(f'  {m:>12}', end='')
print()
print('  ' + '-' * 50)
for f in SPLITS:
    base_rmse = fixed_results['Baseline'][f]
    print(f'  OOS{int(f*100):<5}', end='')
    for m in ['LLM','Gardner','Sharpe']:
        imp = (base_rmse - fixed_results[m][f]) / base_rmse * 100
        print(f'  {imp:>+11.1f}%', end='')
    print()

# Save
rows = []
for method in methods:
    for f in SPLITS:
        rows.append({'method': method, 'split': int(f*100),
                     'expanding_oos': exp_results.get(method, {}).get(f, np.nan),
                     'fixed_oos': fixed_results.get(method, {}).get(f, np.nan)})
out = root + r'\Taylor Rule\outputs\robustness_splits.csv'
pd.DataFrame(rows).to_csv(out, index=False)
print(f'\n  Saved -> {out}')
