"""
diag_ffr_baseline_comparison.py

Compare 3 no-sentiment baselines using delta_fed_funds_rate as DV:
  (1) Naive-zero   : always predict 0
  (2) Market       : expanding OLS of delta_ffr ~ (implied_ffr - fed_funds_rate)
  (3) Macro FAVAR  : expanding OLS of delta_ffr ~ PCA(macro vars, 90% threshold)

Expanding window: train from first 50%, retrain at each step, predict next.
DM test: Market vs Naive, FAVAR vs Naive, FAVAR vs Market.
Sample: post-2007 (aligns with LLM comparison window).
"""
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings('ignore')

sys.path.insert(0, r'C:\Users\Javier\OneDrive - HEC Paris\Documentos\QuantCubeThesis\QuantCubeThesis\macro_pipeline\analysis')
from config import MACRO_REGRESSORS, PCA_VAR_MACRO
from utils import run_pca
from step0_aggregate import load_macro

root = r'C:\Users\Javier\OneDrive - HEC Paris\Documentos\QuantCubeThesis\QuantCubeThesis'
INTER_CSV = root + r'\Taylor Rule\outputs\intermediate\step0_merged_to_macro.csv'

TRAIN_FRAC = 0.50
OOS_SPLITS = [0.60, 0.70, 0.80, 0.90]


# ── Load data ─────────────────────────────────────────────────────────────────

# Load the fully merged macro+sentiment file (has fed_funds_rate + implied_ffr + effective_rate)
df_raw = pd.read_csv(INTER_CSV, parse_dates=['date'])
df = df_raw[df_raw['date'] >= '2007-01-01'].reset_index(drop=True)

# Also bring in any macro cols that may be missing from merged CSV via load_macro()
macro = load_macro()
for col in MACRO_REGRESSORS:
    if col not in df.columns and col in macro.columns:
        tmp = macro[['date', col]].sort_values('date')
        df = pd.merge_asof(df.sort_values('date'), tmp, on='date', direction='backward')
df = df.sort_values('date').reset_index(drop=True)

# Build delta_ffr_next = fed_funds_rate[t+1] - fed_funds_rate[t]
df['delta_ffr_next'] = df['fed_funds_rate'].shift(-1) - df['fed_funds_rate']

# Market predictor: implied_ffr - fed_funds_rate (spread at time t)
df['mkt_spread'] = df['implied_ffr'] - df['fed_funds_rate']

TARGET = 'delta_ffr_next'
df_clean = df.dropna(subset=[TARGET, 'mkt_spread', 'effective_rate', 'implied_ffr']).reset_index(drop=True)

N    = len(df_clean)
INIT = int(N * TRAIN_FRAC)

print(f'\nSample: {df_clean["date"].iloc[0].date()} to {df_clean["date"].iloc[-2].date()}')
print(f'N={N}, init={INIT}, n_oos={N - INIT}')
print(f'DV: delta_fed_funds_rate | mean={df_clean[TARGET].mean():.4f}, std={df_clean[TARGET].std():.4f}\n')


# ── Expanding-window OLS helpers ───────────────────────────────────────────────

def expanding_ols_1d(y, x, init):
    """Single predictor + constant."""
    preds = []
    for t in range(init, len(y)):
        X_tr = sm.add_constant(x[:t].reshape(-1, 1))
        X_te = sm.add_constant(x[t:t+1].reshape(-1, 1))
        try:
            m = sm.OLS(y[:t], X_tr).fit()
            preds.append((t, float(y[t]), float(m.predict(X_te)[0])))
        except Exception:
            pass
    return preds


def expanding_ols_favar(df_sub, macro_cols, target, init):
    """FAVAR: PCA on full sample, expanding OLS."""
    avail = [c for c in macro_cols if c in df_sub.columns]
    sub   = df_sub[avail + [target]].dropna().reset_index(drop=True)
    n     = len(sub)
    fac, _, _, n_comp = run_pca(sub, avail, var_threshold=PCA_VAR_MACRO)
    fac.columns = [f'PC{i+1}' for i in range(n_comp)]
    y = sub[target].values
    X = sm.add_constant(fac.values)
    preds = []
    for t in range(init, n):
        try:
            m = sm.OLS(y[:t], X[:t]).fit()
            preds.append((t, float(y[t]), float(m.predict(X[t:t+1])[0])))
        except Exception:
            pass
    return preds, n


def rmse(preds):
    if not preds:
        return np.nan
    return np.sqrt(np.mean([(y - yh)**2 for _, y, yh in preds]))


def rmse_from(preds, n, frac):
    ct  = int(n * frac)
    sub = [(t, y, yh) for t, y, yh in preds if t >= ct]
    return np.sqrt(np.mean([(y - yh)**2 for _, y, yh in sub])) if sub else np.nan


def dm_test(preds_a, preds_b):
    """DM test: H0 equal forecast accuracy. Positive t => a beats b."""
    common = {t: (ya, yha) for t, ya, yha in preds_a}
    d = []
    for t, yb, yhb in preds_b:
        if t in common:
            ya, yha = common[t]
            d.append((ya - yha)**2 - (yb - yhb)**2)
    if len(d) < 4:
        return np.nan, np.nan
    d = np.array(d)
    dbar = d.mean()
    h = max(1, int(len(d)**0.25))
    nw = sum((1 - abs(k)/(h+1)) * np.mean(d[abs(k):] * d[:len(d)-abs(k)])
             for k in range(-h, h+1) if abs(k) <= len(d)-1)
    se = np.sqrt(max(nw, 1e-12) / len(d))
    t_stat = dbar / se
    from scipy import stats
    p = stats.norm.sf(abs(t_stat))
    return round(t_stat, 4), round(p, 4)


# ── Run 3 models ──────────────────────────────────────────────────────────────

y_arr   = df_clean[TARGET].values
mkt_arr = df_clean['mkt_spread'].values

# (1) Naive-zero
naive_preds = [(t, float(y_arr[t]), 0.0) for t in range(INIT, N)]

# (2) Market: use spread directly as forecast (no OLS wrapper needed —
#   implied_ffr - fed_funds_rate IS the market's expected next-meeting change)
mkt_preds = [(t, float(y_arr[t]), float(mkt_arr[t])) for t in range(INIT, N)
             if not np.isnan(mkt_arr[t])]

# (3) FAVAR
favar_preds, n_fav = expanding_ols_favar(df_clean, MACRO_REGRESSORS, TARGET, INIT)


# ── Results ───────────────────────────────────────────────────────────────────

print('=' * 70)
print('  EXPANDING WINDOW OOS  |  DV = d(fed_funds_rate)  |  init=50%')
print('=' * 70)

models = [
    ('Naive-zero',  naive_preds, N),
    ('Market',      mkt_preds,   N),
    ('Macro FAVAR', favar_preds, n_fav),
]

print(f'\n  {"Model":<14} {"N_oos":>5}  {"RMSE_exp":>9}  '
      + '  '.join(f'OOS{int(f*100):02d}' for f in OOS_SPLITS))
print('  ' + '-' * 65)

rmse_dict = {}
for label, preds, n in models:
    r_exp = rmse(preds)
    r_cut = [rmse_from(preds, n, f) for f in OOS_SPLITS]
    rmse_dict[label] = {'exp': r_exp, 'cuts': r_cut, 'preds': preds}
    cuts_str = '  '.join(f'{v:>7.4f}' for v in r_cut)
    print(f'  {label:<14} {len(preds):>5}  {r_exp:>9.4f}  {cuts_str}')

# DM tests
print(f'\n  DM TESTS (positive t = row model beats column)')
print(f'  {"":20} {"vs Naive":>12} {"vs Market":>12} {"vs FAVAR":>12}')
print('  ' + '-' * 60)

pairs = [
    ('Naive-zero',  'Naive-zero'),
    ('Market',      'Naive-zero'),
    ('Market',      'Market'),
    ('Macro FAVAR', 'Naive-zero'),
    ('Macro FAVAR', 'Market'),
    ('Macro FAVAR', 'Macro FAVAR'),
]

row_labels = ['Market', 'Macro FAVAR']
col_labels = ['Naive-zero', 'Market', 'Macro FAVAR']

for row in row_labels:
    row_preds = rmse_dict[row]['preds']
    cells = []
    for col in col_labels:
        if row == col:
            cells.append('    —   ')
        else:
            col_preds = rmse_dict[col]['preds']
            t, p = dm_test(row_preds, col_preds)
            sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
            cells.append(f'{t:+.3f}{sig:3s}')
    print(f'  {row:<20} {"  ".join(cells)}')

# % improvement over naive
print(f'\n  % RMSE IMPROVEMENT OVER NAIVE-ZERO (expanding window)')
naive_r = rmse_dict['Naive-zero']['exp']
for label in ['Market', 'Macro FAVAR']:
    r = rmse_dict[label]['exp']
    imp = (naive_r - r) / naive_r * 100
    print(f'  {label:<14}  {imp:+.1f}%  (RMSE {r:.4f} vs naive {naive_r:.4f})')

# Save
rows = []
for label, preds, n in models:
    r = rmse_dict[label]
    row = {'model': label, 'n_oos': len(preds), 'rmse_exp': r['exp']}
    for f, rv in zip(OOS_SPLITS, r['cuts']):
        row[f'rmse_oos{int(f*100)}'] = rv
    rows.append(row)

out = root + r'\Taylor Rule\outputs\ffr_baseline_comparison.csv'
pd.DataFrame(rows).to_csv(out, index=False)
print(f'\n  Saved -> {out}')
