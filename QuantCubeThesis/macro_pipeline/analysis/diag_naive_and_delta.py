"""
Two analyses:

A) Level target (target_next): re-run CW tests vs naive no-change baseline
   (naive = predict effective_rate stays the same next meeting)

B) Delta target (delta_rate_next): full FAVAR sweep
   - Best baseline (naive-zero vs macro FAVAR)
   - Best NLP model per method (joint PCA, same specs as 4h)
   - CW tests vs naive-zero
   - 60/70/80/90 expanding + fixed-split RMSE
   - Permutation test (500 shuffles)
"""
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from config import MACRO_REGRESSORS, PCA_VAR_MACRO
from utils import run_pca, clark_west_test
from step0_aggregate import load_macro

root = r'C:\Users\Javier\OneDrive - HEC Paris\Documentos\QuantCubeThesis\QuantCubeThesis'
nlp_dir = root + r'\Taylor Rule\nlp_output'

TARGET_LEVEL = 'target_next'
TARGET_DELTA = 'delta_rate_next'

N_PERM     = 500
TRAIN_FRAC = 0.60
OOS_SPLITS = [0.60, 0.70, 0.80, 0.90]
SEED       = 42
rng        = np.random.default_rng(SEED)


# ── Panel builders ────────────────────────────────────────────────────────────

def build_panel(method):
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


def get_nlp_cols(method, df):
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


def get_sent_cols(method):
    return {'LLM': ['sent_total'], 'Gardner': ['g_total'], 'Sharpe': ['s_net']}[method]


# ── Core OOS engine ───────────────────────────────────────────────────────────

def expanding_oos(df, pca_cols, target_col, train_frac=TRAIN_FRAC):
    """Joint-PCA FAVAR expanding window. Returns (rmse_dict, preds_list)."""
    avail = [c for c in pca_cols if c in df.columns]
    sub   = df[avail + [target_col]].dropna().reset_index(drop=True)
    n     = len(sub)
    init  = int(n * train_frac)
    if init < 10 or init >= n:
        return {}, []

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

    rmse = {}
    for frac in OOS_SPLITS:
        cutoff = int(n * frac)
        sub_e  = [e for t, e in errors if t >= cutoff]
        rmse[frac] = round(np.sqrt(np.mean(sub_e)), 4) if sub_e else np.nan
    return rmse, preds


def fixed_split_oos(df, pca_cols, target_col, train_frac):
    """Single-model fixed split. Returns (rmse, cw_t) vs naive baseline."""
    avail = [c for c in pca_cols if c in df.columns]
    sub   = df[avail + [target_col]].dropna().reset_index(drop=True)
    n     = len(sub)
    n_tr  = int(n * train_frac)
    if n_tr < 10 or n_tr >= n:
        return np.nan, np.nan

    factor_df, _, _, n_comp = run_pca(sub, avail, var_threshold=PCA_VAR_MACRO)
    factor_df.columns = [f'JPC{i+1}' for i in range(n_comp)]
    y = sub[target_col].values
    X = sm.add_constant(factor_df.values)

    model = sm.OLS(y[:n_tr], X[:n_tr]).fit()
    y_hat = model.predict(X[n_tr:])
    y_act = y[n_tr:]

    rmse = np.sqrt(np.mean((y_act - y_hat) ** 2))

    # CW vs naive-zero (for delta) or naive-no-change (for level)
    if target_col == TARGET_DELTA:
        naive_hat = np.zeros(len(y_act))
    else:
        naive_hat = sub['effective_rate'].values[n_tr:]  # last known level

    base_preds  = [(t, float(y_act[t]), float(naive_hat[t])) for t in range(len(y_act))]
    model_preds = [(t, float(y_act[t]), float(y_hat[t]))     for t in range(len(y_act))]
    cw_t, _     = clark_west_test(base_preds, model_preds)
    return rmse, cw_t


# ── Naive baselines ───────────────────────────────────────────────────────────

def naive_preds_level(df, train_frac=TRAIN_FRAC):
    """Predict target_next = current effective_rate (no change)."""
    sub  = df[['effective_rate', TARGET_LEVEL]].dropna().reset_index(drop=True)
    n    = len(sub)
    init = int(n * train_frac)
    preds = [(t, float(sub[TARGET_LEVEL].iloc[t]),
              float(sub['effective_rate'].iloc[t]))
             for t in range(init, n)]
    errors = [(t, (y - yh)**2) for t, y, yh in preds]
    rmse = {}
    for frac in OOS_SPLITS:
        cutoff = int(n * frac)
        sub_e  = [e for t, e in errors if t >= cutoff]
        rmse[frac] = round(np.sqrt(np.mean(sub_e)), 4) if sub_e else np.nan
    return rmse, preds


def naive_preds_delta(df, train_frac=TRAIN_FRAC):
    """Predict delta_rate_next = 0 (no change)."""
    sub  = df[['effective_rate', TARGET_DELTA]].dropna().reset_index(drop=True)
    n    = len(sub)
    init = int(n * train_frac)
    preds = [(t, float(sub[TARGET_DELTA].iloc[t]), 0.0) for t in range(init, n)]
    errors = [(t, (y - yh)**2) for t, y, yh in preds]
    rmse = {}
    for frac in OOS_SPLITS:
        cutoff = int(n * frac)
        sub_e  = [e for t, e in errors if t >= cutoff]
        rmse[frac] = round(np.sqrt(np.mean(sub_e)), 4) if sub_e else np.nan
    return rmse, preds


def naive_fixed_split(df, target_col, train_frac):
    sub  = df[['effective_rate', target_col]].dropna().reset_index(drop=True)
    n    = len(sub)
    n_tr = int(n * train_frac)
    y_act = sub[target_col].values[n_tr:]
    if target_col == TARGET_DELTA:
        y_hat = np.zeros(len(y_act))
    else:
        y_hat = sub['effective_rate'].values[n_tr:]
    return np.sqrt(np.mean((y_act - y_hat)**2))


# ── Permutation test helper ───────────────────────────────────────────────────

def run_permutation(df_orig, sent_cols, method, target_col, naive_preds_list):
    df = df_orig.copy()
    for col in sent_cols:
        if col in df.columns:
            df[col] = rng.permutation(df[col].values)
    if method == 'Sharpe':
        for mc_col in ['vix','gdp','unemployment_gap','inflation_dev_from_target',
                       'implied_ffr','cfnai']:
            if mc_col in df.columns:
                df[f's_net_x_{mc_col}'] = df['s_net'] * df[mc_col]
    pca_cols = get_nlp_cols(method, df)
    rmse, preds = expanding_oos(df, pca_cols, target_col)
    cw_t, _     = clark_west_test(naive_preds_list, preds)
    return rmse.get(0.80, np.nan), cw_t


# ─────────────────────────────────────────────────────────────────────────────
# SECTION A: LEVEL TARGET — CW vs naive no-change
# ─────────────────────────────────────────────────────────────────────────────

print('=' * 85)
print('  SECTION A: Level target (target_next) — CW test vs NAIVE NO-CHANGE')
print('  Naive = predict effective_rate stays same next meeting')
print('=' * 85)

methods = ['Baseline', 'LLM', 'Gardner', 'Sharpe']
df_llm = build_panel('LLM')

naive_rmse_L, naive_preds_L = naive_preds_level(df_llm)
print(f'\n  Naive no-change OOS RMSE: '
      + '  '.join(f'OOS{int(f*100)}={naive_rmse_L[f]:.4f}' for f in OOS_SPLITS))

print(f'\n  {"Method":<12} {"OOS60":>8} {"OOS70":>8} {"OOS80":>8} {"OOS90":>8}'
      f' {"CW-t vs naive":>14} {"CW-p":>8}')
print('  ' + '-' * 75)

level_results = {}
for method in methods:
    df = build_panel(method) if method != 'Baseline' else df_llm
    mc = [c for c in MACRO_REGRESSORS if c in df.columns]
    pca_cols = get_nlp_cols(method, df) if method != 'Baseline' else mc
    rmse, preds = expanding_oos(df, pca_cols, TARGET_LEVEL)

    cw_t, cw_p = clark_west_test(naive_preds_L, preds)
    level_results[method] = {'rmse': rmse, 'preds': preds, 'cw_t': cw_t, 'cw_p': cw_p}

    rmse_str = '  '.join(f'{rmse.get(f, np.nan):>8.4f}' for f in OOS_SPLITS)
    cw_p_str = f'{cw_p:>8.4f}' if not np.isnan(cw_p) else '     n/a'
    cw_t_str = f'{cw_t:>14.3f}' if not np.isnan(cw_t) else '           n/a'
    print(f'  {method:<12} {rmse_str}{cw_t_str}{cw_p_str}')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION B: DELTA TARGET — full analysis
# ─────────────────────────────────────────────────────────────────────────────

print()
print('=' * 85)
print('  SECTION B: Delta target (delta_rate_next = eff_rate[t+1] - eff_rate[t])')
print('=' * 85)

# B1: Naive-zero baseline
naive_rmse_D, naive_preds_D = naive_preds_delta(df_llm)
print(f'\n  Naive predict-zero OOS RMSE: '
      + '  '.join(f'OOS{int(f*100)}={naive_rmse_D[f]:.4f}' for f in OOS_SPLITS))

# B2: Expanding window results
print(f'\n  B1. Expanding-window OOS RMSE (joint PCA, delta target)')
print(f'      CW test vs naive-zero (predict delta = 0)\n')
print(f'  {"Model":<12} {"OOS60":>8} {"OOS70":>8} {"OOS80":>8} {"OOS90":>8}'
      f' {"CW-t vs naive":>14} {"CW-p":>8}')
print('  ' + '-' * 75)

delta_exp_results = {}
for method in methods:
    df = build_panel(method) if method != 'Baseline' else df_llm
    mc = [c for c in MACRO_REGRESSORS if c in df.columns]
    pca_cols = get_nlp_cols(method, df) if method != 'Baseline' else mc
    rmse, preds = expanding_oos(df, pca_cols, TARGET_DELTA)

    cw_t, cw_p = clark_west_test(naive_preds_D, preds)
    delta_exp_results[method] = {'rmse': rmse, 'preds': preds, 'cw_t': cw_t, 'cw_p': cw_p}

    rmse_str = '  '.join(f'{rmse.get(f, np.nan):>8.4f}' for f in OOS_SPLITS)
    cw_p_str = f'{cw_p:>8.4f}' if not np.isnan(cw_p) else '     n/a'
    cw_t_str = f'{cw_t:>14.3f}' if not np.isnan(cw_t) else '           n/a'
    print(f'  {method:<12} {rmse_str}{cw_t_str}{cw_p_str}')

# B3: Fixed single-split RMSE
print()
print(f'  B2. Fixed single-split OOS RMSE (delta target)')
print(f'  {"Model":<12}', end='')
for f in OOS_SPLITS:
    print(f'  {"OOS"+str(int(f*100)):>8}', end='')
print(f'  {"CW-t@80":>10}')
print('  ' + '-' * 70)

delta_fixed = {}
for method in methods:
    df = build_panel(method) if method != 'Baseline' else df_llm
    mc = [c for c in MACRO_REGRESSORS if c in df.columns]
    pca_cols = get_nlp_cols(method, df) if method != 'Baseline' else mc
    row, cw80 = [], np.nan
    for f in OOS_SPLITS:
        rmse, cw_t = fixed_split_oos(df, pca_cols, TARGET_DELTA, f)
        row.append(rmse)
        if f == 0.80:
            cw80 = cw_t
    delta_fixed[method] = row
    rmse_str = ''.join(f'  {v:>8.4f}' if not np.isnan(v) else '       n/a' for v in row)
    print(f'  {method:<12}{rmse_str}  {cw80:>10.3f}')

# Naive fixed-split for reference
print(f'  {"Naive-zero":<12}', end='')
for f in OOS_SPLITS:
    r = naive_fixed_split(df_llm, TARGET_DELTA, f)
    print(f'  {r:>8.4f}', end='')
print(f'  {"—":>10}')

# B4: % improvement over naive
print()
print('  B3. % RMSE improvement over naive-zero (fixed split)')
print(f'  {"Split":<8}', end='')
for m in ['Baseline', 'LLM', 'Gardner', 'Sharpe']:
    print(f'  {m:>12}', end='')
print()
print('  ' + '-' * 60)
for i, f in enumerate(OOS_SPLITS):
    naive_r = naive_fixed_split(df_llm, TARGET_DELTA, f)
    print(f'  OOS{int(f*100):<5}', end='')
    for m in ['Baseline', 'LLM', 'Gardner', 'Sharpe']:
        imp = (naive_r - delta_fixed[m][i]) / naive_r * 100
        print(f'  {imp:>+11.1f}%', end='')
    print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION C: PERMUTATION TEST on delta target
# ─────────────────────────────────────────────────────────────────────────────

print()
print('=' * 85)
print(f'  SECTION C: Permutation test on DELTA target (N={N_PERM} shuffles)')
print('  Null: temporal ordering of sentiment is irrelevant')
print('=' * 85)

perm_nlp_methods = ['LLM', 'Gardner', 'Sharpe']
perm_summary = []

for method in perm_nlp_methods:
    df_real  = build_panel(method)
    sent_cols = get_sent_cols(method)
    pca_cols  = get_nlp_cols(method, df_real)

    real_rmse, real_preds = expanding_oos(df_real, pca_cols, TARGET_DELTA)
    real_r80 = real_rmse.get(0.80, np.nan)
    real_cw, _ = clark_west_test(naive_preds_D, real_preds)

    print(f'\n  [{method}] running {N_PERM} permutations...', end='', flush=True)
    perm_r, perm_c = [], []
    for i in range(N_PERM):
        r, c = run_permutation(df_real, sent_cols, method, TARGET_DELTA, naive_preds_D)
        perm_r.append(r)
        perm_c.append(c)
        if (i + 1) % 100 == 0:
            print(f'{i+1}...', end='', flush=True)
    print(' done')

    perm_r = np.array([v for v in perm_r if not np.isnan(v)])
    perm_c = np.array([v for v in perm_c if not np.isnan(v)])

    p_rmse = np.mean(perm_r <= real_r80)
    p_cw   = np.mean(perm_c >= real_cw)

    print(f'  Real:  OOS_RMSE_80={real_r80:.4f}   CW-t={real_cw:.3f}')
    print(f'  Null:  RMSE mean={np.mean(perm_r):.4f}  p95={np.percentile(perm_r,95):.4f}'
          f'   CW-t mean={np.nanmean(perm_c):.3f}  p95={np.nanpercentile(perm_c,95):.3f}')
    print(f'  p-val (RMSE): {p_rmse:.4f}   p-val (CW-t): {p_cw:.4f}')

    perm_summary.append({
        'method': method, 'target': 'delta',
        'real_oos80': real_r80, 'real_cw': real_cw,
        'null_rmse_mean': np.mean(perm_r), 'null_rmse_p95': np.percentile(perm_r, 95),
        'null_cw_mean': np.nanmean(perm_c), 'null_cw_p95': np.nanpercentile(perm_c, 95),
        'p_rmse': p_rmse, 'p_cw': p_cw,
    })

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print()
print('=' * 85)
print('  FINAL SUMMARY')
print('=' * 85)

print('\n  A. Level target — OOS80 and CW vs naive no-change')
print(f'  {"Model":<12} {"OOS80(lvl)":>12} {"CW-t vs naive":>14} {"CW-p":>8}')
print('  ' + '-' * 52)
naive_r80_L = naive_rmse_L[0.80]
print(f'  {"Naive"::<12} {naive_r80_L:>12.4f} {"—":>14} {"—":>8}')
for method in methods:
    r = level_results[method]
    cw_p = f'{r["cw_p"]:>8.4f}' if not np.isnan(r["cw_p"]) else '     n/a'
    cw_t = f'{r["cw_t"]:>14.3f}' if not np.isnan(r["cw_t"]) else '           n/a'
    print(f'  {method:<12} {r["rmse"].get(0.80,np.nan):>12.4f}{cw_t}{cw_p}')

print('\n  B. Delta target — OOS80 and CW vs naive-zero')
print(f'  {"Model":<12} {"OOS80(dlt)":>12} {"CW-t vs naive":>14} {"CW-p":>8}')
print('  ' + '-' * 52)
naive_r80_D = naive_rmse_D[0.80]
print(f'  {"Naive-zero":<12} {naive_r80_D:>12.4f} {"—":>14} {"—":>8}')
for method in methods:
    r = delta_exp_results[method]
    cw_p = f'{r["cw_p"]:>8.4f}' if not np.isnan(r["cw_p"]) else '     n/a'
    cw_t = f'{r["cw_t"]:>14.3f}' if not np.isnan(r["cw_t"]) else '           n/a'
    print(f'  {method:<12} {r["rmse"].get(0.80,np.nan):>12.4f}{cw_t}{cw_p}')

print('\n  C. Permutation test — delta target')
print(f'  {"Method":<10} {"Real OOS80":>10} {"Null mean":>10} {"p-val(RMSE)":>12}'
      f' {"Real CW-t":>10} {"p-val(CW)":>10}')
print('  ' + '-' * 65)
for r in perm_summary:
    print(f'  {r["method"]:<10} {r["real_oos80"]:>10.4f} {r["null_rmse_mean"]:>10.4f}'
          f' {r["p_rmse"]:>12.4f} {r["real_cw"]:>10.3f} {r["p_cw"]:>10.4f}')

# Save
rows = []
for method in methods:
    for f in OOS_SPLITS:
        rows.append({
            'method': method, 'split': int(f*100),
            'level_exp_oos':  level_results[method]['rmse'].get(f, np.nan),
            'level_cw_naive': level_results[method]['cw_t'],
            'delta_exp_oos':  delta_exp_results[method]['rmse'].get(f, np.nan),
            'delta_cw_naive': delta_exp_results[method]['cw_t'],
        })
out = root + r'\Taylor Rule\outputs\naive_and_delta_results.csv'
pd.DataFrame(rows).to_csv(out, index=False)
pd.DataFrame(perm_summary).to_csv(
    root + r'\Taylor Rule\outputs\permutation_delta.csv', index=False)
print(f'\n  Saved -> {out}')
