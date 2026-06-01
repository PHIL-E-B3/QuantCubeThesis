"""
diag_ewma_delta.py

EWMA-smoothed LLM sentiment vs raw: comparison on the delta target (cum_delta_h1, h=1).

Takes the best-performing raw spec (LLM_all_sent: joint PCA, sent_total + 6 topics)
and runs identical models with EWMA-smoothed sentiment columns at span=5 and span=10.

EWMA is causal (uses only past observations) so applying it to the full series
before OOS evaluation introduces no look-ahead.

Does NOT modify unified_delta_table.csv — results saved to unified_delta_ewma.csv.
"""
import sys
import warnings
import numpy  as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from config import MACRO_REGRESSORS, PCA_VAR_MACRO
from utils import run_pca, diebold_mariano_test, clark_west_test
from step0_aggregate import load_macro

root      = r'C:\Users\Javier\OneDrive - HEC Paris\Documentos\QuantCubeThesis\QuantCubeThesis'
INTER_CSV = root + r'\Taylor Rule\outputs\intermediate\step0_merged_to_macro.csv'

H         = 1
INIT_FRAC = 0.50
START_DATE = '2007-01-01'

LLM_TOTAL  = ['sent_total']
LLM_TOPICS = ['sent_inflation', 'sent_labor_market', 'sent_economic_activity',
               'sent_financial_conditions', 'sent_monetary_policy', 'sent_macro']
LLM_BEST   = LLM_TOTAL + LLM_TOPICS   # the all_sent spec


# ── Shared helpers (copied from diag_unified_delta_table.py) ─────────────────

def sig_star(p):
    if p is None or (isinstance(p, float) and np.isnan(p)): return ''
    return '***' if p < .01 else ('**' if p < .05 else ('*' if p < .10 else ''))

def rmse(y, yh):
    return np.sqrt(np.mean((np.array(y) - np.array(yh)) ** 2))

def run_expanding(y, X_m, X_b, init):
    n = len(y)
    t_all, y_all, yh_m_all, yh_b_all = [], [], [], []
    for t in range(init, n):
        try:
            mm = sm.OLS(y[:t], X_m[:t]).fit()
            mb = sm.OLS(y[:t], X_b[:t]).fit()
            t_all.append(t)
            y_all.append(float(y[t]))
            yh_m_all.append(float(mm.predict(X_m[t:t+1])[0]))
            yh_b_all.append(float(mb.predict(X_b[t:t+1])[0]))
        except Exception:
            pass
    return (np.array(t_all), np.array(y_all),
            np.array(yh_m_all), np.array(yh_b_all))

def evaluate_model(label, arch, sent_cols, mc_avail, df_full):
    is_nested = (arch == 'nested')
    sc = [c for c in sent_cols if c in df_full.columns]
    mc_s = [c for c in mc_avail if c in df_full.columns]
    if not sc:
        return None

    if arch == 'joint_pca':
        avail = mc_s + sc
        need  = list(dict.fromkeys(avail + ['cum_delta_h1']))
        need  = [c for c in need if c in df_full.columns]
        s     = df_full[need].dropna().reset_index(drop=True)
        ns    = len(s); init_s = int(ns * INIT_FRAC)
        if ns < 30: return None

        fac_m, _, _, nc_m = run_pca(s, avail, var_threshold=PCA_VAR_MACRO)
        fac_m.columns = [f'JPC{i+1}' for i in range(nc_m)]
        fac_b, _, _, nc_b = run_pca(s, mc_s,  var_threshold=PCA_VAR_MACRO)
        fac_b.columns = [f'BPC{i+1}' for i in range(nc_b)]

        y   = s['cum_delta_h1'].values
        X_m = sm.add_constant(fac_m.values)
        X_b = sm.add_constant(fac_b.values)

    elif arch == 'nested':
        need = list(dict.fromkeys(mc_s + sc + ['cum_delta_h1']))
        need = [c for c in need if c in df_full.columns]
        s    = df_full[need].dropna().reset_index(drop=True)
        ns   = len(s); init_s = int(ns * INIT_FRAC)
        if ns < 30: return None

        fac_b, _, _, nc_b = run_pca(s, mc_s, var_threshold=PCA_VAR_MACRO)
        fac_b.columns = [f'MPC{i+1}' for i in range(nc_b)]

        y   = s['cum_delta_h1'].values
        S   = s[sc].values
        X_m = sm.add_constant(np.hstack([fac_b.values, S]))
        X_b = sm.add_constant(fac_b.values)
    else:
        raise ValueError(arch)

    t_all, y_all, yh_m_all, yh_b_all = run_expanding(y, X_m, X_b, init_s)
    if len(t_all) < 5:
        return None

    test_label = 'CW' if is_nested else 'DM'
    r_exp = rmse(y_all, yh_m_all)
    r_bas = rmse(y_all, yh_b_all)
    imp   = (r_bas - r_exp) / r_bas * 100

    triples_m = [(int(t_all[i]), float(y_all[i]), float(yh_m_all[i])) for i in range(len(t_all))]
    triples_b = [(int(t_all[i]), float(y_all[i]), float(yh_b_all[i])) for i in range(len(t_all))]
    if is_nested:
        dm_t, dm_p = clark_west_test(triples_b, triples_m)
    else:
        dm_t, dm_p = diebold_mariano_test(triples_b, triples_m)

    # sub-window RMSE at 60 and 80
    def sw_rmse(frac):
        cutoff = int(ns * frac)
        mask   = np.array(t_all) >= cutoff
        ys, yhms, yhbs = y_all[mask], yh_m_all[mask], yh_b_all[mask]
        return rmse(ys, yhms) if mask.sum() >= 5 else np.nan

    row = {
        'model':     label,
        'arch':      arch,
        'test':      test_label,
        'n_obs':     ns,
        'n_oos':     len(t_all),
        'rmse':      round(r_exp, 4),
        'base_rmse': round(r_bas, 4),
        'imp_pct':   round(imp, 1),
        'dm_t':      round(dm_t, 4) if not np.isnan(dm_t) else np.nan,
        'dm_p':      round(dm_p, 4) if not np.isnan(dm_p) else np.nan,
        'sig':       sig_star(dm_p),
        'rmse_60':   round(sw_rmse(0.60), 4),
        'rmse_80':   round(sw_rmse(0.80), 4),
    }

    t_s = f'{dm_t:>6.3f}' if not np.isnan(dm_t) else '   n/a'
    p_s = f'{dm_p:.4f}'   if not np.isnan(dm_p) else '   n/a'
    print(f'  {label:<42}  n={ns:>3}  oos={len(t_all):>2}  '
          f'rmse={r_exp:.4f}  base={r_bas:.4f}  imp={imp:>+6.1f}%  '
          f'{test_label}-t={t_s}  p={p_s}  {sig_star(dm_p)}')

    return row


# ── Load base panel ───────────────────────────────────────────────────────────

print('Loading base panel...', flush=True)
df_base = pd.read_csv(INTER_CSV, parse_dates=['date'])
df_base = df_base[df_base['date'] >= START_DATE].sort_values('date').reset_index(drop=True)
if 'effective_rate' not in df_base.columns:
    macro = load_macro()[['date', 'effective_rate', 'implied_ffr']]
    df_base = pd.merge_asof(df_base, macro, on='date', direction='backward')
df_base['cum_delta_h1'] = df_base['effective_rate'].shift(-H) - df_base['effective_rate']

mc_avail = [c for c in MACRO_REGRESSORS if c in df_base.columns]

# Verify raw sentiment columns exist
raw_cols_present = [c for c in LLM_BEST if c in df_base.columns]
print(f'  Raw sentiment cols present: {len(raw_cols_present)}/{len(LLM_BEST)}')


# ── Build EWMA-smoothed columns ───────────────────────────────────────────────
# ewm().mean() is causal: value at t uses only observations <=t, no look-ahead.

for span in [5, 10]:
    for col in LLM_BEST:
        if col in df_base.columns:
            df_base[f'{col}_ewma{span}'] = (
                df_base[col].ewm(span=span, adjust=False).mean()
            )

print(f'  EWMA columns created: ewma5 and ewma10 for {len(raw_cols_present)} sentiment cols')


# ── Run models ────────────────────────────────────────────────────────────────

results = []

for arch in ['joint_pca', 'nested']:
    test_name  = 'DM' if arch == 'joint_pca' else 'CW'
    arch_label = 'Joint PCA' if arch == 'joint_pca' else 'Nested FAVAR'

    print('\n' + '=' * 100)
    print(f'  {arch_label.upper()}  |  {test_name} test  |  delta h=1  |  50% init')
    print('=' * 100)

    # Raw baseline (the existing best result, reproduced for direct comparison)
    raw_cols = LLM_BEST
    row = evaluate_model(f'LLM_all_sent_raw ({arch})', arch, raw_cols, mc_avail, df_base)
    if row:
        row['signal'] = 'raw'; results.append(row)

    # EWMA variants
    for span in [5, 10]:
        ewma_cols = [f'{c}_ewma{span}' for c in LLM_BEST]
        row = evaluate_model(f'LLM_all_sent_ewma{span} ({arch})', arch, ewma_cols, mc_avail, df_base)
        if row:
            row['signal'] = f'ewma{span}'; results.append(row)


# ── Save ──────────────────────────────────────────────────────────────────────

out_df   = pd.DataFrame(results)
out_path = root + r'\Taylor Rule\outputs\unified_delta_ewma.csv'
out_df.to_csv(out_path, index=False)
print(f'\n  Saved -> unified_delta_ewma.csv  ({len(out_df)} rows)')

print('\n' + '=' * 80)
print('  SUMMARY: Raw vs EWMA per architecture')
print('=' * 80)
print(f'  {"Model":<42}  {"Arch":<10}  {"RMSE":>7}  {"Imp%":>6}  {"t":>6}  {"p":>7}  {"sig"}')
print('  ' + '-' * 80)
for _, r in out_df.iterrows():
    p_s = f'{r["dm_p"]:.4f}' if not (isinstance(r["dm_p"], float) and np.isnan(r["dm_p"])) else '    n/a'
    t_s = f'{r["dm_t"]:>6.3f}' if not (isinstance(r["dm_t"], float) and np.isnan(r["dm_t"])) else '   n/a'
    print(f'  {r["model"]:<42}  {r["arch"]:<10}  {r["rmse"]:>7.4f}  '
          f'{r["imp_pct"]:>+6.1f}%  {t_s}  {p_s}  {r["sig"]}')
