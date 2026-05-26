"""
diag_significance_check.py

Investigate why G4 (Gardner, CW) appears more significant than LLM (DM)
despite LLM having lower OOS RMSE.

For each model at OOS80, prints:
  - Model RMSE
  - Baseline RMSE actually used in the test (sub-sample, not full-sample)
  - RMSE improvement over that baseline
  - n_predictions in the OOS80 window
  - DM t-stat
  - CW t-stat
  - Both p-values side-by-side

This isolates whether the difference is due to:
  (a) CW adjustment term inflating t-stats vs DM
  (b) Different baselines (N=118 vs N=151)
  (c) Different OOS windows / time periods
"""
import sys, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from config import TARGET_NEXT, MACRO_REGRESSORS, PCA_VAR_MACRO
from utils import run_pca, clark_west_test, diebold_mariano_test
from step0_aggregate import load_macro

root      = r'C:\Users\Javier\OneDrive - HEC Paris\Documentos\QuantCubeThesis\QuantCubeThesis'
INTER_CSV = root + r'\Taylor Rule\outputs\intermediate\step0_merged_to_macro.csv'
NLP_DIR   = root + r'\Taylor Rule\nlp_output'

TRAIN_FRAC = 0.60
CUTOFF     = 0.80   # focus on OOS80

MACRO_INTER_COLS = [
    'vix', 'gdp', 'unemployment_gap',
    'inflation_dev_from_target', 'implied_ffr', 'cfnai',
]
MATCHED_CONSENSUS = {
    'sent_inflation_consensus':            'inflation_dev_from_target',
    'sent_labor_market_consensus':         'unemployment_gap',
    'sent_economic_activity_consensus':    'gdp',
    'sent_financial_conditions_consensus': 'vix',
    'sent_monetary_policy_consensus':      'implied_ffr',
}
GARDNER_REMAP = {
    'gardner_total': 'sent_total',
    'gardner_inf':   'sent_inflation',
    'gardner_labor': 'sent_labor_market',
    'gardner_out':   'sent_economic_activity',
    'gardner_fin':   'sent_financial_conditions',
    'gardner_mp':    'sent_monetary_policy',
    'gardner_inf_consensus':            'sent_inflation_consensus',
    'gardner_labor_consensus':          'sent_labor_market_consensus',
    'gardner_out_consensus':            'sent_economic_activity_consensus',
    'gardner_fin_consensus':            'sent_financial_conditions_consensus',
    'gardner_mp_consensus':             'sent_monetary_policy_consensus',
}


def expanding_ols(y, X):
    n = len(y); init = int(n * TRAIN_FRAC)
    preds = []
    for t in range(init, n):
        try:
            m = sm.OLS(y[:t], X[:t]).fit()
            preds.append((t, float(y[t]), float(m.predict(X[t:t+1])[0])))
        except Exception:
            pass
    return preds


def sub_preds(preds, n, frac):
    ct = int(n * frac)
    return [(t, y, yh) for t, y, yh in preds if t >= ct]


def rmse(sub):
    return np.sqrt(np.mean([(y - yh)**2 for _, y, yh in sub]))


def diagnose(label, m_preds, b_preds, n, dates=None):
    ct   = int(n * CUTOFF)
    m80  = sub_preds(m_preds, n, CUTOFF)
    b80  = sub_preds(b_preds, n, CUTOFF)
    if not m80:
        print(f'  {label}: no predictions'); return

    model_rmse = rmse(m80)
    base_rmse  = rmse(b80)
    improvement = base_rmse - model_rmse
    n_pred = len(m80)

    dm_t,  dm_p  = diebold_mariano_test(b80, m80)
    cw_t,  cw_p  = clark_west_test(b80, m80)

    # Date range of OOS80 predictions
    if dates is not None:
        t_indices = [t for t, _, _ in m80]
        date_start = dates.iloc[min(t_indices)] if len(dates) > min(t_indices) else 'n/a'
        date_end   = dates.iloc[max(t_indices)] if len(dates) > max(t_indices) else 'n/a'
        date_str = f'{date_start}  to  {date_end}'
    else:
        date_str = f't=[{ct}..{n-1}]'

    def fmt_p(p):
        if p is None or (isinstance(p, float) and np.isnan(p)): return '  n/a  '
        s = '***' if p < 0.01 else ('** ' if p < 0.05 else ('*  ' if p < 0.10 else '   '))
        return f'{p:.4f}{s}'

    print(f'\n  {"-"*70}')
    print(f'  Model        : {label}')
    print(f'  N total      : {n}   |   N OOS80 predictions: {n_pred}   (t>={ct})')
    print(f'  Period       : {date_str}')
    print(f'  Model RMSE   : {model_rmse:.4f}')
    print(f'  Baseline RMSE: {base_rmse:.4f}   (sub-sample baseline, same window)')
    print(f'  Improvement  : {improvement:.4f}   ({improvement/base_rmse*100:.1f}% reduction)')
    print(f'  DM  t={dm_t:>6.3f}   p={fmt_p(dm_p)}   [pure loss differential]')
    print(f'  CW  t={cw_t:>6.3f}   p={fmt_p(cw_p)}   [+adjustment term for nested]')
    print(f'  CW - DM t-stat inflation: {(cw_t - dm_t):.3f}')


# ── Load and run ───────────────────────────────────────────────────────────────

DELTA_COL = 'delta_rate_next'

MATCHED_TOPIC = {
    'sent_inflation':            'inflation_dev_from_target',
    'sent_labor_market':         'unemployment_gap',
    'sent_economic_activity':    'gdp',
    'sent_financial_conditions': 'vix',
    'sent_monetary_policy':      'implied_ffr',
}

print('=' * 75)
print('  SIGNIFICANCE INVESTIGATION: LLM (DM) vs Gardner G4 (CW) at OOS80')
print('  DELTA target = change in effective rate at next FOMC meeting')
print('  Both models N=151 (same sample) -- so only DM vs CW matters')
print('=' * 75)

# ── LLM: 4h_llm_all_sent on DELTA ─────────────────────────────────────────────
print('\n  [1] LLM 4h_llm_all_sent  (joint PCA, DM test -- non-nested)')
df_llm = pd.read_csv(INTER_CSV, parse_dates=['date'])
df_llm = df_llm[df_llm['date'] >= '2007-01-01'].reset_index(drop=True)

mc_llm   = [c for c in MACRO_REGRESSORS if c in df_llm.columns]
TOPIC_COLS = ['sent_inflation', 'sent_labor_market', 'sent_economic_activity',
              'sent_financial_conditions', 'sent_monetary_policy', 'sent_macro']
sent_all = ['sent_total'] + [c for c in TOPIC_COLS if c in df_llm.columns]
all_need = list(dict.fromkeys(mc_llm + sent_all + [DELTA_COL]))
sub = df_llm[all_need].dropna().reset_index(drop=True)
n_llm = len(sub)

fac_m, _, _, nc_m = run_pca(sub, mc_llm + sent_all, var_threshold=PCA_VAR_MACRO)
fac_m.columns = [f'JPC{i+1}' for i in range(nc_m)]
fac_b, _, _, nc_b = run_pca(sub, mc_llm, var_threshold=PCA_VAR_MACRO)
fac_b.columns = [f'BPC{i+1}' for i in range(nc_b)]

y_llm = sub[DELTA_COL].values
m_preds_llm = expanding_ols(y_llm, sm.add_constant(fac_m.values))
b_preds_llm = expanding_ols(y_llm, sm.add_constant(fac_b.values))
diagnose('4h_llm_all_sent  [N=151, delta]', m_preds_llm, b_preds_llm, n_llm)

# ── Gardner G4: speeches_zscore 4f_matched on DELTA ───────────────────────────
print('\n  [2] Gardner G4: speeches_zs_4f_matched  (standard FAVAR, CW test -- nested)')
raw = pd.read_csv(NLP_DIR + '\\Gardner_speeches_zscore_nlp.csv', parse_dates=['meeting_date'])
raw = raw.rename(columns={k: v for k, v in GARDNER_REMAP.items() if k in raw.columns})
sent_cols_g = [c for c in raw.columns if c.startswith('sent_')]
agg = raw.groupby('meeting_date')[sent_cols_g].sum().reset_index()
agg = agg.rename(columns={'meeting_date': 'date'})
macro = load_macro()
df_g4 = pd.merge_asof(macro.sort_values('date'), agg.sort_values('date'),
                      on='date', direction='backward')
df_g4 = df_g4[df_g4['date'] >= '2007-01-01'].reset_index(drop=True)

# Build matched interactions
for sent_col, mc in MATCHED_TOPIC.items():
    if sent_col in df_g4.columns and mc in df_g4.columns:
        df_g4[f'inter_{sent_col}_x_{mc}'] = df_g4[sent_col] * df_g4[mc]
inter_cols = [f'inter_{s}_x_{m}' for s, m in MATCHED_TOPIC.items()
              if f'inter_{s}_x_{m}' in df_g4.columns]

mc_avail = [c for c in MACRO_REGRESSORS if c in df_g4.columns]
fac_g, _, _, nc_g = run_pca(df_g4.dropna(subset=mc_avail), mc_avail, var_threshold=PCA_VAR_MACRO)
fac_g.columns = [f'PC{i+1}' for i in range(nc_g)]

ex_avail = [c for c in inter_cols if c in df_g4.columns]
X_aug = fac_g.join(df_g4[ex_avail], how='left').dropna()
y_srs = df_g4.loc[X_aug.index, DELTA_COL].dropna()
X_aug = X_aug.loc[y_srs.index]
n_g4  = len(X_aug)
y_g4  = y_srs.values
pc_cols = [c for c in X_aug.columns if c.startswith('PC')]
m_preds_g4 = expanding_ols(y_g4, sm.add_constant(X_aug.values))
b_preds_g4 = expanding_ols(y_g4, sm.add_constant(X_aug[pc_cols].values))
diagnose('G4_speeches_zs_4f_matched  [N=151, delta]', m_preds_g4, b_preds_g4, n_g4)

# ── Cross-check: apply BOTH tests to both models ──────────────────────────────
print('\n\n  == CROSS-CHECK: apply DM and CW to BOTH models at OOS80 ==')

m80_llm = sub_preds(m_preds_llm, n_llm, CUTOFF)
b80_llm = sub_preds(b_preds_llm, n_llm, CUTOFF)
m80_g4  = sub_preds(m_preds_g4,  n_g4,  CUTOFF)
b80_g4  = sub_preds(b_preds_g4,  n_g4,  CUTOFF)

dm_t_llm, dm_p_llm = diebold_mariano_test(b80_llm, m80_llm)
cw_t_llm, cw_p_llm = clark_west_test(b80_llm, m80_llm)
dm_t_g4,  dm_p_g4  = diebold_mariano_test(b80_g4,  m80_g4)
cw_t_g4,  cw_p_g4  = clark_west_test(b80_g4,  m80_g4)

def ps(p):
    if p is None or (isinstance(p, float) and np.isnan(p)): return 'n/a    '
    s = '***' if p < 0.01 else ('** ' if p < 0.05 else ('*  ' if p < 0.10 else '   '))
    return f'{p:.4f} {s}'

print(f'\n  {"Model":<42}  {"n_pred":>6}  {"mdl_rmse":>8}  {"base_rmse":>9}  {"DM-t":>6}  {"DM-p":>12}  {"CW-t":>6}  {"CW-p":>12}')
print('  ' + '-' * 110)
print(f'  {"4h_llm_all_sent (N=151)":<42}  {len(m80_llm):>6}  {rmse(m80_llm):>8.4f}  {rmse(b80_llm):>9.4f}  {dm_t_llm:>6.3f}  {ps(dm_p_llm):>12}  {cw_t_llm:>6.3f}  {ps(cw_p_llm):>12}')
print(f'  {"G4_speeches_zs_4f_matched (N=151)":<42}  {len(m80_g4):>6}  {rmse(m80_g4):>8.4f}  {rmse(b80_g4):>9.4f}  {dm_t_g4:>6.3f}  {ps(dm_p_g4):>12}  {cw_t_g4:>6.3f}  {ps(cw_p_g4):>12}')

print('\n  CW - DM t-stat inflation:')
print(f'    LLM : CW-t={cw_t_llm:.3f}  DM-t={dm_t_llm:.3f}  diff=+{cw_t_llm - dm_t_llm:.3f}')
print(f'    G4  : CW-t={cw_t_g4:.3f}  DM-t={dm_t_g4:.3f}  diff=+{cw_t_g4 - dm_t_g4:.3f}')
print(f'\n  % RMSE improvement over baseline (same N=151):')
print(f'    LLM: {rmse(b80_llm):.4f} -> {rmse(m80_llm):.4f}  ({(rmse(b80_llm)-rmse(m80_llm))/rmse(b80_llm)*100:.1f}% reduction)')
print(f'    G4:  {rmse(b80_g4):.4f} -> {rmse(m80_g4):.4f}  ({(rmse(b80_g4)-rmse(m80_g4))/rmse(b80_g4)*100:.1f}% reduction)')
