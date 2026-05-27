"""
diag_block_bootstrap.py

Moving block bootstrap (MBB) DM test for the h=1 result.

Permutation tests are invalid for time series: shuffling breaks the temporal
ordering that expanding-window OLS depends on. The MBB (Kunsch 1989) resamples
contiguous blocks of the loss differential series d_t = e_b^2 - e_m^2,
preserving local autocorrelation while building an empirical null distribution.

Procedure:
  1. Compute d_t for the 76 OOS predictions (same setup as diag_multihorizon_dm)
  2. Observed DM stat: d_bar / se(d_bar)
  3. Bootstrap:
     - Center d_t by subtracting d_bar  (impose H0: E[d_t] = 0)
     - Draw B=5000 bootstrap samples by resampling non-overlapping blocks
     - For each, compute bootstrap DM stat
  4. p-value = fraction of bootstrap stats >= observed stat  (one-sided, H1: model better)
  5. Repeat for block lengths L = 3, 4, 5, 8, T^(1/3) ≈ 4, T^(1/2) ≈ 9

Also reports the Harvey-Leybourne-Newbold (1997) small-sample corrected DM as comparison.
"""
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from config import MACRO_REGRESSORS, PCA_VAR_MACRO
from utils import run_pca, diebold_mariano_test
from step0_aggregate import load_macro

root      = r'C:\Users\Javier\OneDrive - HEC Paris\Documentos\QuantCubeThesis\QuantCubeThesis'
INTER_CSV = root + r'\Taylor Rule\outputs\intermediate\step0_merged_to_macro.csv'

TRAIN_FRAC = 0.50
H          = 1
B          = 5000
SEED       = 42

TOPIC_COLS = [
    'sent_inflation', 'sent_labor_market', 'sent_economic_activity',
    'sent_financial_conditions', 'sent_monetary_policy', 'sent_macro',
]


# ── Build predictions (identical to diag_robustness_h1) ───────────────────────

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
    mc_avail = [c for c in MACRO_REGRESSORS if c in df.columns]
    avail    = [c for c in pca_input_cols   if c in df.columns]

    df = df.copy()
    target_col = f'cum_delta_h{h}'
    df[target_col] = df['effective_rate'].shift(-h) - df['effective_rate']

    need = list(dict.fromkeys(['date'] + avail + mc_avail + [target_col]))
    sub  = df[need].dropna().reset_index(drop=True)
    n    = len(sub)
    init = int(n * TRAIN_FRAC)

    avail_s = [c for c in avail if c in sub.columns]
    fac_m, _, _, nc_m = run_pca(sub, avail_s, var_threshold=PCA_VAR_MACRO)
    fac_m.columns = [f'JPC{i+1}' for i in range(nc_m)]

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

    return pd.DataFrame(raw_m), pd.DataFrame(raw_b), n, init


df_m, df_b, n_total, init = build_preds(df_llm, pca_cols, h=H)
df_m['date'] = pd.to_datetime(df_m['date'])
df_b['date'] = pd.to_datetime(df_b['date'])

T = len(df_m)
print(f'  OOS predictions: T={T}  '
      f'({df_m["date"].min().date()} to {df_m["date"].max().date()})')

# ── Loss differential series ──────────────────────────────────────────────────

e_m = df_m['y'].values - df_m['yh'].values   # model errors
e_b = df_b['y'].values - df_b['yh'].values   # baseline errors
d   = e_b**2 - e_m**2                        # positive = model better

d_bar  = d.mean()
d_std  = d.std(ddof=1)
dm_t   = d_bar / (d_std / np.sqrt(T))
dm_p_param = 1 - scipy_stats.norm.cdf(dm_t)  # one-sided, same as diebold_mariano_test

# Harvey-Leybourne-Newbold (1997) small-sample correction at h=1
hln_factor = np.sqrt((T + 1 - 2 * H + H * (H - 1) / T) / T)
dm_t_hln   = dm_t * hln_factor
dm_p_hln   = 1 - scipy_stats.t.cdf(dm_t_hln, df=T - 1)


# ── Moving block bootstrap ────────────────────────────────────────────────────

def moving_block_bootstrap(d, L, B, seed=42):
    """
    Moving block bootstrap (Kunsch 1989) for E[d_t] under H0: E[d_t] = 0.

    Centers d_t to impose H0, resamples blocks of length L, builds B bootstrap
    DM statistics. One-sided p-value = P(DM* >= DM_obs | H0).
    """
    rng    = np.random.default_rng(seed)
    T      = len(d)
    d_bar  = d.mean()
    d_std  = d.std(ddof=1)
    dm_obs = d_bar / (d_std / np.sqrt(T))

    # Center to impose H0
    d0 = d - d_bar

    # All blocks of length L (non-circular, so starts in [0, T-L])
    n_blocks_needed = int(np.ceil(T / L))
    starts = np.arange(T - L + 1)

    boot_dm = np.empty(B)
    for b in range(B):
        idx    = rng.choice(starts, size=n_blocks_needed, replace=True)
        d_boot = np.concatenate([d0[i:i + L] for i in idx])[:T]
        boot_dm[b] = d_boot.mean() / (d_boot.std(ddof=1) / np.sqrt(T))

    p_val = np.mean(boot_dm >= dm_obs)
    return boot_dm, p_val


def sig_star(p):
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''


# ── Run ────────────────────────────────────────────────────────────────────────

print('\n' + '=' * 72)
print('  MOVING BLOCK BOOTSTRAP DM TEST  (Kunsch 1989)')
print('  Model: 4h_llm_all_sent  |  Baseline: Macro FAVAR  |  h=1')
print(f'  T={T} OOS predictions  |  B={B} replications  |  seed={SEED}')
print('  H0: E[d_t]=0 (equal accuracy)   H1: E[d_t]>0 (model better)')
print('=' * 72)

print(f'\n  Observed DM stat  : t = {dm_t:.4f}')
print(f'  Parametric p-val  : p = {dm_p_param:.4f} {sig_star(dm_p_param)}  '
      f'(one-sided, standard normal)')
print(f'  HLN-corrected     : t = {dm_t_hln:.4f}  p = {dm_p_hln:.4f} '
      f'{sig_star(dm_p_hln)}  (Harvey et al. 1997, t_{T-1})')

# Block lengths to test
block_spec = {}
for L in [3, 4, 5, 8]:
    block_spec[L] = f'fixed {L}'
L_cbrt = max(2, int(round(T ** (1 / 3))))
L_sqrt = max(2, int(round(T ** (1 / 2))))
block_spec[L_cbrt] = f'T^(1/3) = {L_cbrt}'
block_spec[L_sqrt] = f'T^(1/2) = {L_sqrt}'

print(f'\n  {"Block L":>8}  {"Rationale":<18}  {"Boot p-val":>10}  {"sig":>4}  '
      f'{"% boot >= obs":>14}')
print('  ' + '-' * 62)

boot_results = []
seen_L = set()
for L in sorted(block_spec):
    if L in seen_L:
        continue
    seen_L.add(L)
    label = block_spec[L]
    boot_dm, p = moving_block_bootstrap(d, L=L, B=B, seed=SEED)
    pct = p * 100
    print(f'  {L:>8}  {label:<18}  {p:>10.4f}  {sig_star(p):>4}  {pct:>13.1f}%')
    boot_results.append({'block_L': L, 'rationale': label, 'boot_p': round(p, 4),
                         'sig': sig_star(p)})

print(f'\n  Parametric DM (reference):')
print(f'    Standard normal  : t={dm_t:.4f}  p={dm_p_param:.4f}  {sig_star(dm_p_param)}')
print(f'    HLN small-sample : t={dm_t_hln:.4f}  p={dm_p_hln:.4f}  {sig_star(dm_p_hln)}')

print('\n  Note: p-values are one-sided (H1: model outperforms).')
print('  Block bootstrap is robust to autocorrelation in d_t without assuming')
print('  a parametric error distribution.')

# ── Save ───────────────────────────────────────────────────────────────────────
df_boot = pd.DataFrame(boot_results)
df_boot['dm_t_param']  = round(dm_t, 4)
df_boot['dm_p_param']  = round(dm_p_param, 4)
df_boot['dm_t_hln']    = round(dm_t_hln, 4)
df_boot['dm_p_hln']    = round(dm_p_hln, 4)
df_boot['T']           = T
df_boot['B']           = B

out = root + r'\Taylor Rule\outputs\block_bootstrap_dm.csv'
df_boot.to_csv(out, index=False)
print(f'\n  Saved -> block_bootstrap_dm.csv')
