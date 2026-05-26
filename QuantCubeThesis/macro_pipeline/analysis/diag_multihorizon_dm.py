"""
diag_multihorizon_dm.py

Multi-horizon DM test for best 4h LLM models vs macro-only baseline.

Target at horizon h: cumulative delta = effective_rate[t+h] - effective_rate[t]
  h=1  : change at next meeting          (= current delta_rate_next)
  h=5  : cumulative change over 5 meetings
  h=10 : cumulative change over 10 meetings
  ...

Expanding OLS: train on first 50%, retrain at each step, predict h meetings ahead.
DM test over all OOS predictions from t >= init.

Question answered: at what horizon h does sentiment add statistically
significant predictive power over the macro-only baseline?
"""
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from config import MACRO_REGRESSORS, PCA_VAR_MACRO
from utils import run_pca, diebold_mariano_test
from step0_aggregate import load_macro

root      = r'C:\Users\Javier\OneDrive - HEC Paris\Documentos\QuantCubeThesis\QuantCubeThesis'
INTER_CSV = root + r'\Taylor Rule\outputs\intermediate\step0_merged_to_macro.csv'

TRAIN_FRAC = 0.50
HORIZONS   = [1, 5, 10, 15, 20, 25, 30]

TOPIC_COLS = [
    'sent_inflation', 'sent_labor_market', 'sent_economic_activity',
    'sent_financial_conditions', 'sent_monetary_policy', 'sent_macro',
]


# ── Core functions ─────────────────────────────────────────────────────────────

def expanding_ols(y, X, init):
    n = len(y); preds = []
    for t in range(init, n):
        try:
            m = sm.OLS(y[:t], X[:t]).fit()
            preds.append((t, float(y[t]), float(m.predict(X[t:t+1])[0])))
        except Exception:
            pass
    return preds


def rmse(preds):
    return np.sqrt(np.mean([(y - yh)**2 for _, y, yh in preds]))


def sig_star(p):
    if p is None or (isinstance(p, float) and np.isnan(p)): return ''
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''


# ── Model runner ───────────────────────────────────────────────────────────────

def run_horizon(df, pca_input_cols, h):
    """
    4h joint-PCA model vs macro-only baseline, predicting cumulative delta at horizon h.
    Returns (model_preds, base_preds, n_full, n_pred, init).
    """
    mc_avail = [c for c in MACRO_REGRESSORS  if c in df.columns]
    avail    = [c for c in pca_input_cols    if c in df.columns]

    # Cumulative delta target: rate h meetings ahead minus current rate
    df = df.copy()
    target_col = f'cum_delta_h{h}'
    df[target_col] = df['effective_rate'].shift(-h) - df['effective_rate']

    all_need = list(dict.fromkeys(avail + mc_avail + [target_col]))
    sub = df[all_need].dropna().reset_index(drop=True)
    n   = len(sub)
    if n < 20:
        return None, None, n, 0, 0

    init = int(n * TRAIN_FRAC)

    # Joint PCA (model)
    avail_s = [c for c in avail if c in sub.columns]
    fac_m, _, _, nc_m = run_pca(sub, avail_s, var_threshold=PCA_VAR_MACRO)
    fac_m.columns = [f'JPC{i+1}' for i in range(nc_m)]

    # Macro-only PCA (baseline)
    mc_s = [c for c in mc_avail if c in sub.columns]
    fac_b, _, _, nc_b = run_pca(sub, mc_s, var_threshold=PCA_VAR_MACRO)
    fac_b.columns = [f'BPC{i+1}' for i in range(nc_b)]

    y       = sub[target_col].values
    X_model = sm.add_constant(fac_m.values)
    X_base  = sm.add_constant(fac_b.values)

    m_preds = expanding_ols(y, X_model, init)
    b_preds = expanding_ols(y, X_base,  init)

    return m_preds, b_preds, n, len(m_preds), init


# ── Load panel ─────────────────────────────────────────────────────────────────

print('  Loading panel...', flush=True)
df_llm = pd.read_csv(INTER_CSV, parse_dates=['date'])
df_llm = df_llm[df_llm['date'] >= '2007-01-01'].reset_index(drop=True)

# Ensure effective_rate is present; fall back to loading from macro if needed
if 'effective_rate' not in df_llm.columns:
    macro = load_macro()[['date', 'effective_rate']]
    df_llm = pd.merge_asof(df_llm.sort_values('date'), macro.sort_values('date'),
                           on='date', direction='backward')

mc_llm   = [c for c in MACRO_REGRESSORS if c in df_llm.columns]
sent_tot = [c for c in ['sent_total']   if c in df_llm.columns]
sent_top = [c for c in TOPIC_COLS       if c in df_llm.columns]

# 4h LLM variants (best two by OOS80 delta from summary table)
MODELS = {
    '4h_llm_all_sent': mc_llm + sent_tot + sent_top,
    '4h_llm_topics'  : mc_llm + sent_top,
    '4h_llm_total'   : mc_llm + sent_tot,
}


# ── Run and print ──────────────────────────────────────────────────────────────

print('\n' + '=' * 90)
print('  MULTI-HORIZON DM TEST  |  4h LLM vs macro-only baseline')
print('  Target: cumulative delta = effective_rate[t+h] - effective_rate[t]')
print('  Training: expanding from 50% of N  |  DM test on all OOS predictions')
print('  Significance: * p<0.10  ** p<0.05  *** p<0.01')
print('=' * 90)

all_results = []

HDR = (f'  {"h":>3}  {"N":>4}  {"n_pred":>6}  '
       f'{"mdl_RMSE":>8}  {"base_RMSE":>9}  {"improv%":>7}  '
       f'{"DM-t":>6}  {"DM-p":>7}  {"sig":>3}')

for model_name, pca_cols in MODELS.items():
    print(f'\n  Model: {model_name}')
    print(HDR)
    print('  ' + '-' * 72)

    for h in HORIZONS:
        m_preds, b_preds, n, n_pred, init = run_horizon(df_llm, pca_cols, h)

        if m_preds is None or n_pred < 5:
            print(f'  {h:>3}  {n:>4}  {"--":>6}  (insufficient predictions)')
            continue

        m_r   = rmse(m_preds)
        b_r   = rmse(b_preds)
        improv = (b_r - m_r) / b_r * 100
        dm_t, dm_p = diebold_mariano_test(b_preds, m_preds)
        star = sig_star(dm_p)

        t_s = f'{dm_t:>6.3f}' if not np.isnan(dm_t) else '   n/a'
        p_s = f'{dm_p:.4f}'   if not (isinstance(dm_p, float) and np.isnan(dm_p)) else '   n/a'

        print(f'  {h:>3}  {n:>4}  {n_pred:>6}  '
              f'{m_r:>8.4f}  {b_r:>9.4f}  {improv:>7.1f}%  '
              f'{t_s}  {p_s}  {star:>3}')

        all_results.append({
            'model': model_name, 'h': h, 'n': n, 'n_pred': n_pred,
            'model_rmse': round(m_r, 4), 'base_rmse': round(b_r, 4),
            'improv_pct': round(improv, 2),
            'dm_t': round(dm_t, 4) if not np.isnan(dm_t) else np.nan,
            'dm_p': round(dm_p, 4) if not (isinstance(dm_p, float) and np.isnan(dm_p)) else np.nan,
            'sig': star,
        })

# ── Summary: first significant horizon per model ───────────────────────────────
print('\n\n  SUMMARY: first horizon at which DM is significant (p < 0.10)')
print(f'  {"Model":<25}  {"h=1":>6}  {"first_sig_h":>11}  {"best_h_by_improv":>16}')
print('  ' + '-' * 65)

df_res = pd.DataFrame(all_results)
for model_name in MODELS:
    sub = df_res[df_res['model'] == model_name]
    if sub.empty:
        continue
    h1_row  = sub[sub['h'] == 1]
    h1_sig  = h1_row['sig'].values[0] if not h1_row.empty else 'n/a'
    sig_rows = sub[sub['dm_p'] <= 0.10]
    first_sig = int(sig_rows['h'].min()) if not sig_rows.empty else 'none'
    best_improv_h = int(sub.loc[sub['improv_pct'].idxmax(), 'h']) if not sub.empty else 'n/a'
    print(f'  {model_name:<25}  {h1_sig:>6}  {str(first_sig):>11}  {str(best_improv_h):>16}')

# ── Save ───────────────────────────────────────────────────────────────────────
out = root + r'\Taylor Rule\outputs\multihorizon_dm.csv'
df_res.to_csv(out, index=False)
print(f'\n  Saved -> multihorizon_dm.csv')
