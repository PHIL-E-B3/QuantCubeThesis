"""
diag_multihorizon_dm.py

Multi-horizon DM test for best 4h LLM models.

Target at horizon h: cumulative delta = effective_rate[t+h] - effective_rate[t]
  h=1 : next meeting rate change
  h=5 : cumulative change over 5 meetings  ... etc.

Expanding OLS: train from first 50%, retrain at each step, predict h steps ahead.

Three baselines evaluated at every horizon -- DM is run against the BEST one:
  (1) Naive        : predict zero change always (random walk)
  (2) Market       : OLS of cum_delta_h ~ (implied_ffr - effective_rate), expanding
  (3) Macro FAVAR  : OLS of cum_delta_h ~ PCA(7 macro vars), expanding  [current]

The strongest no-sentiment baseline is used so that DM significance is
a conservative claim.

Code checks:
  - effective_rate and implied_ffr are verified present before use
  - Targets are built BEFORE dropna so shift-NaN rows are correctly removed
  - PCA is full-sample (standard FAVAR convention, no expanding PCA)
  - No future data enters X[t] (mkt_delta = implied_ffr[t] - effective_rate[t]
    uses post-meeting futures rate at t, predicting meeting t+h; no look-ahead)
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def expanding_ols(y, X, init):
    """Expanding-window OLS from init. Returns [(t, y_actual, y_hat), ...]."""
    n = len(y)
    assert init >= 2, 'init must be >= 2 to fit OLS'
    preds = []
    for t in range(init, n):
        try:
            m = sm.OLS(y[:t], X[:t]).fit()
            preds.append((t, float(y[t]), float(m.predict(X[t:t+1])[0])))
        except Exception:
            pass
    return preds


def naive_preds(y, init):
    """Baseline: always predict zero change."""
    return [(t, float(y[t]), 0.0) for t in range(init, len(y))]


def rmse_of(preds):
    if not preds:
        return np.nan
    return np.sqrt(np.mean([(y - yh) ** 2 for _, y, yh in preds]))


def sig_star(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ''
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''


# ── Per-horizon runner ─────────────────────────────────────────────────────────

def run_horizon(df, pca_input_cols, h):
    """
    Build target, run model and all baselines.
    Returns dict with preds lists and metadata.
    """
    mc_avail = [c for c in MACRO_REGRESSORS if c in df.columns]
    avail    = [c for c in pca_input_cols   if c in df.columns]

    # ── Build target BEFORE dropna ─────────────────────────────────────────────
    df = df.copy()
    assert 'effective_rate' in df.columns, 'effective_rate missing from panel'
    assert 'implied_ffr'    in df.columns, 'implied_ffr missing from panel'

    target_col  = f'cum_delta_h{h}'
    mkt_col     = 'mkt_delta'
    df[target_col] = df['effective_rate'].shift(-h) - df['effective_rate']
    df[mkt_col]    = df['implied_ffr'] - df['effective_rate']

    # Drop rows where any feature or target is NaN
    need_cols = list(dict.fromkeys(avail + mc_avail + [target_col, mkt_col]))
    sub = df[need_cols].dropna().reset_index(drop=True)
    n   = len(sub)
    if n < 20:
        return None

    init = int(n * TRAIN_FRAC)
    assert init >= 5, f'init={init} too small at h={h}'

    y = sub[target_col].values

    # ── Baseline 1: Naive (zero change) ───────────────────────────────────────
    b_naive = naive_preds(y, init)

    # ── Baseline 2: Market — OLS on mkt_delta only (expanding) ────────────────
    X_mkt = sm.add_constant(sub[[mkt_col]].values)
    b_mkt  = expanding_ols(y, X_mkt, init)

    # ── Baseline 3: Macro FAVAR — PCA on 7 macro vars (expanding OLS) ─────────
    mc_s = [c for c in mc_avail if c in sub.columns]
    fac_b, _, _, nc_b = run_pca(sub, mc_s, var_threshold=PCA_VAR_MACRO)
    fac_b.columns = [f'BPC{i+1}' for i in range(nc_b)]
    X_favar = sm.add_constant(fac_b.values)
    b_favar = expanding_ols(y, X_favar, init)

    # ── Model: Joint PCA (macro + sentiment) ──────────────────────────────────
    avail_s = [c for c in avail if c in sub.columns]
    fac_m, _, _, nc_m = run_pca(sub, avail_s, var_threshold=PCA_VAR_MACRO)
    fac_m.columns = [f'JPC{i+1}' for i in range(nc_m)]
    X_model = sm.add_constant(fac_m.values)
    m_preds = expanding_ols(y, X_model, init)

    # ── Pick best baseline (lowest RMSE) ──────────────────────────────────────
    rmse_naive = rmse_of(b_naive)
    rmse_mkt   = rmse_of(b_mkt)
    rmse_favar = rmse_of(b_favar)

    candidates = {
        'naive': (b_naive, rmse_naive),
        'market': (b_mkt,  rmse_mkt),
        'favar': (b_favar, rmse_favar),
    }
    best_name = min(candidates, key=lambda k: candidates[k][1]
                    if not np.isnan(candidates[k][1]) else np.inf)
    best_preds, best_rmse = candidates[best_name]

    return {
        'n': n, 'init': init, 'n_pred': len(m_preds),
        'nc_m': nc_m, 'nc_b': nc_b,
        'm_preds': m_preds,
        'best_base_name': best_name,
        'best_base_preds': best_preds,
        'rmse_naive': rmse_naive,
        'rmse_mkt': rmse_mkt,
        'rmse_favar': rmse_favar,
        'best_rmse': best_rmse,
    }


# ── Load panel ─────────────────────────────────────────────────────────────────

print('  Loading panel...', flush=True)
df_llm = pd.read_csv(INTER_CSV, parse_dates=['date'])
df_llm = df_llm[df_llm['date'] >= '2007-01-01'].reset_index(drop=True)

# Ensure effective_rate present (fallback: merge from load_macro)
if 'effective_rate' not in df_llm.columns:
    macro = load_macro()[['date', 'effective_rate', 'implied_ffr']]
    df_llm = pd.merge_asof(df_llm.sort_values('date'),
                           macro.sort_values('date'),
                           on='date', direction='backward')

mc_llm   = [c for c in MACRO_REGRESSORS if c in df_llm.columns]
sent_tot = [c for c in ['sent_total']   if c in df_llm.columns]
sent_top = [c for c in TOPIC_COLS       if c in df_llm.columns]

print(f'  Panel shape: {df_llm.shape}  |  N(2007+): {len(df_llm)}')
print(f'  MACRO_REGRESSORS in panel: {mc_llm}')
print(f'  Sentiment cols in panel:   sent_total={("sent_total" in df_llm.columns)}  '
      f'topics={len(sent_top)}')
mkt_check = df_llm['implied_ffr'] - df_llm['effective_rate']
print(f'  mkt_delta check: mean={mkt_check.mean():.4f}  std={mkt_check.std():.4f}  '
      f'NaN={mkt_check.isna().sum()}')

MODELS = {
    '4h_llm_all_sent': mc_llm + sent_tot + sent_top,
    '4h_llm_topics'  : mc_llm + sent_top,
}


# ── Run ────────────────────────────────────────────────────────────────────────

print('\n' + '=' * 105)
print('  MULTI-HORIZON DM TEST  |  4h LLM vs best no-sentiment baseline')
print('  Target: cum_delta_h = effective_rate[t+h] - effective_rate[t]')
print('  Baselines compared at each h: (1) Naive  (2) Market OLS  (3) Macro FAVAR')
print('  DM test uses whichever baseline has the lowest RMSE (most conservative)')
print('  Training: expanding from 50% of N')
print('=' * 105)

# First print baseline comparison (run once on 4h_llm_all_sent features to get baselines)
print('\n  BASELINE RMSE COMPARISON  (independent of model choice)')
print(f'  {"h":>3}  {"N":>4}  {"n_pred":>6}  {"naive":>7}  {"market":>8}  '
      f'{"favar":>7}  {"BEST":>8}')
print('  ' + '-' * 58)

baseline_cache = {}  # cache results keyed by h for the main model run
for h in HORIZONS:
    r = run_horizon(df_llm, MODELS['4h_llm_all_sent'], h)
    if r is None:
        continue
    baseline_cache[h] = r
    winner = f'{r["best_base_name"].upper()}'
    print(f'  {h:>3}  {r["n"]:>4}  {r["n_pred"]:>6}  '
          f'{r["rmse_naive"]:>7.4f}  {r["rmse_mkt"]:>8.4f}  '
          f'{r["rmse_favar"]:>7.4f}  {winner:>8}')

# Main DM results per model
all_results = []

HDR = (f'\n  {"h":>3}  {"N":>4}  {"n_pred":>6}  {"JPC":>4}  '
       f'{"mdl_RMSE":>8}  {"best_base":>9}  {"base_type":>10}  '
       f'{"improv%":>7}  {"DM-t":>6}  {"DM-p":>8}  {"sig":>3}')

for model_name, pca_cols in MODELS.items():
    print(f'\n\n  Model: {model_name}')
    print(HDR)
    print('  ' + '-' * 90)

    for h in HORIZONS:
        r = run_horizon(df_llm, pca_cols, h)
        if r is None or r['n_pred'] < 5:
            print(f'  {h:>3}  (insufficient data)')
            continue

        m_r    = rmse_of(r['m_preds'])
        b_r    = r['best_rmse']
        improv = (b_r - m_r) / b_r * 100
        dm_t, dm_p = diebold_mariano_test(r['best_base_preds'], r['m_preds'])
        star = sig_star(dm_p)

        t_s = f'{dm_t:>6.3f}' if not np.isnan(dm_t) else '   n/a'
        p_s = f'{dm_p:.4f}'   if not (isinstance(dm_p, float) and np.isnan(dm_p)) else '    n/a'

        print(f'  {h:>3}  {r["n"]:>4}  {r["n_pred"]:>6}  {r["nc_m"]:>4}  '
              f'{m_r:>8.4f}  {b_r:>9.4f}  {r["best_base_name"]:>10}  '
              f'{improv:>7.1f}%  {t_s}  {p_s}  {star:>3}')

        all_results.append({
            'model': model_name, 'h': h,
            'n': r['n'], 'n_pred': r['n_pred'], 'n_jpc': r['nc_m'],
            'model_rmse': round(m_r, 4),
            'base_rmse': round(b_r, 4),
            'base_type': r['best_base_name'],
            'rmse_naive': round(r['rmse_naive'], 4),
            'rmse_mkt':   round(r['rmse_mkt'],   4),
            'rmse_favar': round(r['rmse_favar'],  4),
            'improv_pct': round(improv, 2),
            'dm_t': round(dm_t, 4) if not np.isnan(dm_t) else np.nan,
            'dm_p': round(dm_p, 4) if not (isinstance(dm_p, float) and np.isnan(dm_p)) else np.nan,
            'sig': star,
        })

# ── Summary ────────────────────────────────────────────────────────────────────
print('\n\n  SUMMARY: first h with DM significance (p<0.10)')
print(f'  {"Model":<25}  {"h=1 sig":>7}  {"first sig h":>11}  {"base used at h=1":>16}')
print('  ' + '-' * 68)
df_res = pd.DataFrame(all_results)
for model_name in MODELS:
    sub = df_res[df_res['model'] == model_name]
    if sub.empty: continue
    h1  = sub[sub['h'] == 1]
    h1_sig  = h1['sig'].values[0]  if not h1.empty else 'n/a'
    h1_base = h1['base_type'].values[0] if not h1.empty else 'n/a'
    sig_rows = sub[sub['dm_p'].notna() & (sub['dm_p'] <= 0.10)]
    first_sig = int(sig_rows['h'].min()) if not sig_rows.empty else 'none'
    print(f'  {model_name:<25}  {h1_sig:>7}  {str(first_sig):>11}  {h1_base:>16}')

# ── Save ───────────────────────────────────────────────────────────────────────
out = root + r'\Taylor Rule\outputs\multihorizon_dm.csv'
df_res.to_csv(out, index=False)
print(f'\n  Saved -> multihorizon_dm.csv')
