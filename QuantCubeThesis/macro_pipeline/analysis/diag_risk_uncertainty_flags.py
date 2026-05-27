"""
diag_risk_uncertainty_flags.py

Tests whether the risk and uncertainty sentence flags carry predictive
information about the next FOMC policy decision.

Hypotheses:
  (U) Elevated uncertainty -> asymmetric FOMC reaction function -> model
      should be MORE predictive in high-uncertainty periods (risk management
      motive dominates; the Fed's response becomes more systematic)

  (R-down) High downside risk -> more likely rate cut (or hold)
  (R-up)   High upside risk   -> more likely rate hike

Aggregation rule (determined by checking distribution):
  All three flags have >0 counts at almost every meeting (flag_elevated_wid
  and flag_skew_down: 0% zero; flag_skew_up: 5% zero). Flags are continuous
  and spread, so we use MEDIAN SPLIT: flag > median = "high".

Three outputs:
  1. Flag distributions and correlation with actual rate changes
  2. Directional accuracy: in high-downside-risk / high-upside-risk periods,
     what fraction of actual rate changes are in the predicted direction?
  3. Subsample DM at h=1 (50% training): does model outperform in high-
     uncertainty vs low-uncertainty periods?
"""
import sys
import warnings
import numpy  as np
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

FLAG_COLS = ['flag_elevated_wid', 'flag_skew_up', 'flag_skew_down']


# ── Load panel ─────────────────────────────────────────────────────────────────

df = pd.read_csv(INTER_CSV, parse_dates=['date'])
df = df[df['date'] >= '2007-01-01'].reset_index(drop=True)

if 'effective_rate' not in df.columns:
    macro = load_macro()[['date', 'effective_rate', 'implied_ffr']]
    df = pd.merge_asof(df.sort_values('date'), macro.sort_values('date'),
                       on='date', direction='backward')

df['cum_delta_h1'] = df['effective_rate'].shift(-1) - df['effective_rate']

# ── 1. Flag distributions ──────────────────────────────────────────────────────

print('=' * 80)
print('  1. FLAG DISTRIBUTIONS  (N=151 FOMC meetings, 2007–2025)')
print('=' * 80)

print(f'\n  {"Flag":<22}  {"min":>5}  {"p25":>5}  {"med":>5}  {"p75":>5}  '
      f'{"max":>5}  {"pct=0":>6}  {"pct>med":>8}')
print('  ' + '-' * 65)

thresholds = {}
for col in FLAG_COLS:
    s = df[col].dropna()
    med = s.median()
    thresholds[col] = med
    print(f'  {col:<22}  {s.min():>5.0f}  {s.quantile(.25):>5.0f}  '
          f'{med:>5.0f}  {s.quantile(.75):>5.0f}  {s.max():>5.0f}  '
          f'{(s==0).mean()*100:>5.0f}%  '
          f'{(s>med).mean()*100:>7.0f}%')

print(f'\n  Thresholds for "high" flag (median split):')
for col, thresh in thresholds.items():
    print(f'    {col}: > {thresh:.0f}')

# Build binary regime columns on the OOS-eligible sub-panel
df_oos = df.dropna(subset=['cum_delta_h1']).copy()
for col in FLAG_COLS:
    df_oos[f'hi_{col}'] = df_oos[col] > thresholds[col]


# ── 2. Directional accuracy ────────────────────────────────────────────────────

print('\n\n' + '=' * 80)
print('  2. DIRECTIONAL ACCURACY: do risk flags predict the direction of the move?')
print('  High-downside-risk -> rate cut hypothesis  (delta < 0)')
print('  High-upside-risk   -> rate hike hypothesis  (delta > 0)')
print('=' * 80)

def directional_stats(sub, label):
    delta = sub['cum_delta_h1'].dropna()
    n     = len(delta)
    if n < 3:
        return
    n_cut  = (delta < -0.01).sum()
    n_hold = (delta.abs() <= 0.01).sum()
    n_hike = (delta >  0.01).sum()
    mean_d = delta.mean()
    print(f'\n  {label}  (n={n})')
    print(f'    Cut  (<0): {n_cut:>3}  ({n_cut/n*100:>5.1f}%)')
    print(f'    Hold (~0): {n_hold:>3}  ({n_hold/n*100:>5.1f}%)')
    print(f'    Hike (>0): {n_hike:>3}  ({n_hike/n*100:>5.1f}%)')
    print(f'    Mean delta: {mean_d:>+.4f}')

print('\n  --- Full sample ---')
directional_stats(df_oos, 'All meetings')

print('\n  --- Uncertainty (flag_elevated_wid) ---')
directional_stats(df_oos[df_oos['hi_flag_elevated_wid']],  'High uncertainty (>median)')
directional_stats(df_oos[~df_oos['hi_flag_elevated_wid']], 'Low  uncertainty (<=median)')

print('\n  --- Downside risk (flag_skew_down) ---')
directional_stats(df_oos[df_oos['hi_flag_skew_down']],  'High downside risk (>median)')
directional_stats(df_oos[~df_oos['hi_flag_skew_down']], 'Low  downside risk (<=median)')

print('\n  --- Upside risk (flag_skew_up) ---')
directional_stats(df_oos[df_oos['hi_flag_skew_up']],  'High upside risk (>median)')
directional_stats(df_oos[~df_oos['hi_flag_skew_up']], 'Low  upside risk (<=median)')


# ── 3. Correlation between flags and actual delta ─────────────────────────────

print('\n\n' + '=' * 80)
print('  3. CORRELATION: flags vs actual cum_delta_h1')
print('=' * 80)

for col in FLAG_COLS:
    corr = df_oos[[col, 'cum_delta_h1']].dropna().corr().iloc[0, 1]
    print(f'  {col:<25}  corr with delta_h1 = {corr:>+.4f}')


# ── 4. Subsample DM test ───────────────────────────────────────────────────────

print('\n\n' + '=' * 80)
print('  4. SUBSAMPLE DM TEST  |  h=1, 50% training, model vs Macro FAVAR')
print('  Split meetings by high/low flag; run DM within each half.')
print('  If model outperforms MORE in high-uncertainty: asymmetric reaction confirmed.')
print('=' * 80)

mc_avail   = [c for c in MACRO_REGRESSORS if c in df.columns]
sent_tot   = [c for c in ['sent_total']   if c in df.columns]
sent_top   = [c for c in TOPIC_COLS       if c in df.columns]
pca_input  = mc_avail + sent_tot + sent_top

# Build full expanding-window predictions with dates
target_col = 'cum_delta_h1'
need = list(dict.fromkeys(['date'] + pca_input + [target_col]))
sub  = df[need].dropna().reset_index(drop=True)
n    = len(sub)
init = int(n * TRAIN_FRAC)

y     = sub[target_col].values
dates = sub['date'].values

fac_m, _, _, nc_m = run_pca(sub, pca_input, var_threshold=PCA_VAR_MACRO)
fac_m.columns = [f'JPC{i+1}' for i in range(nc_m)]
X_model = sm.add_constant(fac_m.values)

fac_b, _, _, nc_b = run_pca(sub, mc_avail, var_threshold=PCA_VAR_MACRO)
fac_b.columns = [f'BPC{i+1}' for i in range(nc_b)]
X_base = sm.add_constant(fac_b.values)

raw_m, raw_b = [], []
for t in range(init, n):
    try:
        mm = sm.OLS(y[:t], X_model[:t]).fit()
        mb = sm.OLS(y[:t], X_base[:t]).fit()
        raw_m.append({'t': t, 'date': dates[t], 'y': float(y[t]),
                      'yh': float(mm.predict(X_model[t:t+1])[0])})
        raw_b.append({'t': t, 'date': dates[t], 'y': float(y[t]),
                      'yh': float(mb.predict(X_base[t:t+1])[0])})
    except Exception:
        pass

df_m = pd.DataFrame(raw_m); df_m['date'] = pd.to_datetime(df_m['date'])
df_b = pd.DataFrame(raw_b); df_b['date'] = pd.to_datetime(df_b['date'])

print(f'\n  OOS predictions: {len(df_m)}  '
      f'({df_m["date"].min().date()} to {df_m["date"].max().date()})')

# Merge flag columns onto prediction rows
preds = df_m.merge(df_b[['date', 'yh']].rename(columns={'yh': 'yh_b'}),
                   on='date', how='inner')
preds = preds.merge(df_oos[['date'] + [f'hi_{c}' for c in FLAG_COLS]],
                    on='date', how='left')


def dm_on_subset(mask, label):
    sub_m = [(r.t, r.y, r.yh)   for _, r in preds[mask].iterrows()]
    sub_b = [(r.t, r.y, r.yh_b) for _, r in preds[mask].iterrows()]
    nr = len(sub_m)
    if nr < 5:
        print(f'  {label:<40}  n={nr:>3}  (too few)')
        return
    rmse_m = np.sqrt(np.mean([(y - yh)**2 for _, y, yh in sub_m]))
    rmse_b = np.sqrt(np.mean([(y - yh)**2 for _, y, yh in sub_b]))
    improv = (rmse_b - rmse_m) / rmse_b * 100
    dm_t, dm_p = diebold_mariano_test(sub_b, sub_m)
    t_s = f'{dm_t:>6.3f}' if not np.isnan(dm_t) else '   n/a'
    p_s = f'{dm_p:.4f}' if not (isinstance(dm_p, float) and np.isnan(dm_p)) else '   n/a'
    def star(p):
        if p is None or (isinstance(p, float) and np.isnan(p)): return ''
        return '***' if p < .01 else ('**' if p < .05 else ('*' if p < .10 else ''))
    print(f'  {label:<40}  n={nr:>3}  '
          f'mdl={rmse_m:.4f}  base={rmse_b:.4f}  '
          f'imp={improv:>+6.1f}%  DM-t={t_s}  p={p_s}  {star(dm_p)}')


print(f'\n  {"Subsample":<40}  {"n":>3}  '
      f'{"mdl RMSE":>9}  {"base RMSE":>9}  {"improv":>8}  '
      f'{"DM-t":>8}  {"p":>8}')
print('  ' + '-' * 85)

# Full sample (reference)
all_mask = pd.Series([True] * len(preds), index=preds.index)
dm_on_subset(all_mask, '[Full sample]')
print('  ' + '-' * 85)

for col in FLAG_COLS:
    hi_col = f'hi_{col}'
    if hi_col not in preds.columns:
        continue
    mask_hi = preds[hi_col] == True
    mask_lo = preds[hi_col] == False
    print()
    dm_on_subset(mask_hi, f'High {col} (>median)')
    dm_on_subset(mask_lo, f'Low  {col} (<=median)')


# ── 5. Save ────────────────────────────────────────────────────────────────────

out_df = preds.copy()
for col in FLAG_COLS:
    out_df[f'loss_diff_{col}'] = (out_df['y'] - out_df['yh_b'])**2 - \
                                  (out_df['y'] - out_df['yh'])**2
out_path = root + r'\Taylor Rule\outputs\risk_uncertainty_dm.csv'
out_df.to_csv(out_path, index=False)
print(f'\n  Saved -> risk_uncertainty_dm.csv')
