"""
diag_robustness_h1.py

Robustness check for the h=1 DM result:
  4h_llm_all_sent beats macro FAVAR at *** (DM-t=2.42, p=0.008), 76 OOS predictions.

Checks:
  1. Sub-period breakdown — RMSE improvement and DM p-value within each regime
  2. Leave-one-regime-out — DM significance when each regime is excluded
  3. Cumulative loss differential — where in time gains accumulate

Economic regimes (OOS period ~ 2019-2025):
  A. Pre-COVID  : 2019-01 to 2020-02  (mild cuts, normal cycle)
  B. COVID shock: 2020-03 to 2020-06  (emergency -150bps, ZLB entry)
  C. ZLB        : 2020-07 to 2022-02  (near-zero, stable hold)
  D. Hiking     : 2022-03 to 2023-07  (aggressive tightening +500bps)
  E. Peak+Cuts  : 2023-08 to 2025-12  (hold then easing)
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
H          = 1

TOPIC_COLS = [
    'sent_inflation', 'sent_labor_market', 'sent_economic_activity',
    'sent_financial_conditions', 'sent_monetary_policy', 'sent_macro',
]

REGIMES = {
    'A_pre_covid' : ('2019-01-01', '2020-02-28'),
    'B_covid_shock': ('2020-03-01', '2020-06-30'),
    'C_zlb'       : ('2020-07-01', '2022-02-28'),
    'D_hiking'    : ('2022-03-01', '2023-07-31'),
    'E_peak_cuts' : ('2023-08-01', '2025-12-31'),
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def expanding_ols(y, X, init):
    n = len(y); preds = []
    for t in range(init, n):
        try:
            m = sm.OLS(y[:t], X[:t]).fit()
            preds.append((t, float(y[t]), float(m.predict(X[t:t+1])[0])))
        except Exception:
            pass
    return preds


def rmse_of(triples):
    if not triples: return np.nan
    return np.sqrt(np.mean([(y - yh) ** 2 for _, y, yh in triples]))


def sig_star(p):
    if p is None or (isinstance(p, float) and np.isnan(p)): return '    '
    if p < 0.01: return '*** '
    if p < 0.05: return '**  '
    if p < 0.10: return '*   '
    return '    '


def dm_summary(b_sub, m_sub, label=''):
    if len(m_sub) < 5:
        return np.nan, np.nan, np.nan, np.nan
    m_r = rmse_of(m_sub)
    b_r = rmse_of(b_sub)
    dm_t, dm_p = diebold_mariano_test(b_sub, m_sub)
    return m_r, b_r, dm_t, dm_p


# ── Build full prediction series with dates ────────────────────────────────────

print('  Loading panel and building predictions...', flush=True)
df_llm = pd.read_csv(INTER_CSV, parse_dates=['date'])
df_llm = df_llm[df_llm['date'] >= '2007-01-01'].reset_index(drop=True)

if 'effective_rate' not in df_llm.columns:
    macro = load_macro()[['date', 'effective_rate', 'implied_ffr']]
    df_llm = pd.merge_asof(df_llm.sort_values('date'), macro.sort_values('date'),
                           on='date', direction='backward')

mc_llm   = [c for c in MACRO_REGRESSORS if c in df_llm.columns]
sent_tot = [c for c in ['sent_total']   if c in df_llm.columns]
sent_top = [c for c in TOPIC_COLS       if c in df_llm.columns]
pca_cols = mc_llm + sent_tot + sent_top


def build_preds(df, pca_input_cols, h=1):
    """Run model and FAVAR baseline; return preds with dates."""
    mc_avail = [c for c in MACRO_REGRESSORS if c in df.columns]
    avail    = [c for c in pca_input_cols   if c in df.columns]

    df = df.copy()
    target_col = f'cum_delta_h{h}'
    df[target_col] = df['effective_rate'].shift(-h) - df['effective_rate']

    need = list(dict.fromkeys(['date'] + avail + mc_avail + [target_col]))
    sub  = df[need].dropna().reset_index(drop=True)
    n    = len(sub)
    init = int(n * TRAIN_FRAC)

    # Joint PCA (model)
    avail_s = [c for c in avail if c in sub.columns]
    fac_m, _, _, nc_m = run_pca(sub, avail_s, var_threshold=PCA_VAR_MACRO)
    fac_m.columns = [f'JPC{i+1}' for i in range(nc_m)]

    # Macro FAVAR (baseline)
    mc_s = [c for c in mc_avail if c in sub.columns]
    fac_b, _, _, nc_b = run_pca(sub, mc_s, var_threshold=PCA_VAR_MACRO)
    fac_b.columns = [f'BPC{i+1}' for i in range(nc_b)]

    y       = sub[target_col].values
    X_model = sm.add_constant(fac_m.values)
    X_base  = sm.add_constant(fac_b.values)

    dates   = sub['date'].values

    raw_m, raw_b = [], []
    for t in range(init, n):
        try:
            mm = sm.OLS(y[:t], X_model[:t]).fit()
            mb = sm.OLS(y[:t], X_base[:t]).fit()
            raw_m.append({'t': t, 'date': dates[t],
                          'y': float(y[t]), 'yh': float(mm.predict(X_model[t:t+1])[0])})
            raw_b.append({'t': t, 'date': dates[t],
                          'y': float(y[t]), 'yh': float(mb.predict(X_base[t:t+1])[0])})
        except Exception:
            pass

    df_m = pd.DataFrame(raw_m)
    df_b = pd.DataFrame(raw_b)
    return df_m, df_b, n, init


df_m, df_b, n, init = build_preds(df_llm, pca_cols, h=H)
df_m['date'] = pd.to_datetime(df_m['date'])
df_b['date'] = pd.to_datetime(df_b['date'])

print(f'  OOS predictions: {len(df_m)}  '
      f'({df_m["date"].min().date()} to {df_m["date"].max().date()})')
print(f'  Overall RMSE: model={rmse_of(list(zip(df_m.t,df_m.y,df_m.yh))):.4f}  '
      f'base={rmse_of(list(zip(df_b.t,df_b.y,df_b.yh))):.4f}')


# ── Helper: filter preds by date range ────────────────────────────────────────

def filter_by_date(df_preds, start, end):
    mask = (df_preds['date'] >= start) & (df_preds['date'] <= end)
    sub  = df_preds[mask]
    return list(zip(sub['t'], sub['y'], sub['yh']))


# ── 1. Sub-period breakdown ────────────────────────────────────────────────────

print('\n' + '=' * 90)
print('  1. SUB-PERIOD BREAKDOWN  (DM within each economic regime)')
print('  Model: 4h_llm_all_sent  |  Baseline: Macro FAVAR')
print('=' * 90)
print(f'\n  {"Regime":<18} {"N":>4}  {"mdl_RMSE":>8}  {"base_RMSE":>9}  '
      f'{"improv%":>7}  {"DM-t":>6}  {"DM-p":>8}  {"sig":>4}')
print('  ' + '-' * 78)

for regime, (start, end) in REGIMES.items():
    m_sub = filter_by_date(df_m, start, end)
    b_sub = filter_by_date(df_b, start, end)
    nr    = len(m_sub)
    if nr == 0:
        print(f'  {regime:<18} {"0":>4}  (no OOS predictions in this period)')
        continue
    m_r, b_r, dm_t, dm_p = dm_summary(b_sub, m_sub)
    improv = (b_r - m_r) / b_r * 100 if b_r > 0 else np.nan
    t_s = f'{dm_t:>6.3f}' if not np.isnan(dm_t) else '   n/a'
    p_s = f'{dm_p:.4f}'   if not (isinstance(dm_p, float) and np.isnan(dm_p)) else '   n/a'
    print(f'  {regime:<18} {nr:>4}  {m_r:>8.4f}  {b_r:>9.4f}  '
          f'{improv:>7.1f}%  {t_s}  {p_s}  {sig_star(dm_p)}')


# ── 2. Leave-one-regime-out ────────────────────────────────────────────────────

print('\n' + '=' * 90)
print('  2. LEAVE-ONE-REGIME-OUT  (DM on all OOS predictions except one regime)')
print('  If *** holds in every row: result is NOT driven by any single period.')
print('=' * 90)
print(f'\n  {"Excluded regime":<20} {"N":>4}  {"mdl_RMSE":>8}  {"base_RMSE":>9}  '
      f'{"improv%":>7}  {"DM-t":>6}  {"DM-p":>8}  {"sig":>4}')
print('  ' + '-' * 78)

# Full result first
all_m = list(zip(df_m['t'], df_m['y'], df_m['yh']))
all_b = list(zip(df_b['t'], df_b['y'], df_b['yh']))
m_r, b_r, dm_t, dm_p = dm_summary(all_b, all_m)
improv = (b_r - m_r) / b_r * 100
print(f'  {"[All included]":<20} {len(all_m):>4}  {m_r:>8.4f}  {b_r:>9.4f}  '
      f'{improv:>7.1f}%  {dm_t:>6.3f}  {dm_p:.4f}  {sig_star(dm_p)}')
print('  ' + '-' * 78)

for regime, (start, end) in REGIMES.items():
    # Exclude this regime
    mask = ~((df_m['date'] >= start) & (df_m['date'] <= end))
    sub_m = df_m[mask]; sub_b = df_b[mask]
    m_sub = list(zip(sub_m['t'], sub_m['y'], sub_m['yh']))
    b_sub = list(zip(sub_b['t'], sub_b['y'], sub_b['yh']))
    nr = len(m_sub)
    if nr < 5:
        print(f'  {("excl. "+regime):<20} {nr:>4}  (too few)')
        continue
    m_r, b_r, dm_t, dm_p = dm_summary(b_sub, m_sub)
    improv = (b_r - m_r) / b_r * 100
    t_s = f'{dm_t:>6.3f}' if not np.isnan(dm_t) else '   n/a'
    p_s = f'{dm_p:.4f}'   if not (isinstance(dm_p, float) and np.isnan(dm_p)) else '   n/a'
    print(f'  {("excl. "+regime):<20} {nr:>4}  {m_r:>8.4f}  {b_r:>9.4f}  '
          f'{improv:>7.1f}%  {t_s}  {p_s}  {sig_star(dm_p)}')


# ── 3. Cumulative loss differential ───────────────────────────────────────────

print('\n' + '=' * 90)
print('  3. CUMULATIVE LOSS DIFFERENTIAL  (base_loss - model_loss, per prediction)')
print('  Positive and rising = model consistently outperforming over time.')
print('  If driven by one spike: line rises sharply only in that period.')
print('=' * 90)

merged = df_m.merge(df_b, on=['t', 'date', 'y'], suffixes=('_m', '_b'))
merged = merged.sort_values('date').reset_index(drop=True)
merged['loss_diff'] = (merged['y'] - merged['yh_b'])**2 - (merged['y'] - merged['yh_m'])**2
merged['cum_loss_diff'] = merged['loss_diff'].cumsum()

print(f'\n  {"Date":<12}  {"Regime":<18}  {"loss_diff":>9}  {"cum_diff":>9}  {"model_err":>9}  {"base_err":>9}')
print('  ' + '-' * 75)

def get_regime(dt):
    for name, (s, e) in REGIMES.items():
        if pd.Timestamp(s) <= dt <= pd.Timestamp(e):
            return name
    return 'pre_OOS'

for _, row in merged.iterrows():
    regime = get_regime(row['date'])
    print(f'  {str(row["date"].date()):<12}  {regime:<18}  '
          f'{row["loss_diff"]:>9.4f}  {row["cum_loss_diff"]:>9.4f}  '
          f'{row["y"]-row["yh_m"]:>9.4f}  '
          f'{row["y"]-row["yh_b"]:>9.4f}')

# Summary by regime
print(f'\n  Cumulative loss differential by regime:')
print(f'  {"Regime":<20}  {"n":>4}  {"avg_loss_diff":>13}  {"cum_contrib":>11}')
print('  ' + '-' * 55)
merged['regime'] = merged['date'].apply(get_regime)
for regime in list(REGIMES.keys()) + ['pre_OOS']:
    sub = merged[merged['regime'] == regime]
    if sub.empty: continue
    print(f'  {regime:<20}  {len(sub):>4}  '
          f'{sub["loss_diff"].mean():>13.4f}  '
          f'{sub["loss_diff"].sum():>11.4f}')

# Save
out = root + r'\Taylor Rule\outputs\robustness_h1.csv'
merged.to_csv(out, index=False)
print(f'\n  Saved -> robustness_h1.csv')
