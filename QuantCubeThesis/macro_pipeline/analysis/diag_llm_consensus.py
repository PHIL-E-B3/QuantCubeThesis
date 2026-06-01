"""
diag_llm_consensus.py

Runs 4b2 consensus regressions for LLM sentiment.

Consensus = |sent_total| or |sent_<topic>|  (absolute value of mean sentiment),
analogous to the dictionary consensus = |sum(si)| / n.

Tests three variants against the macro FAVAR baseline (delta h=1, expanding 50%):
  - consensus_total  : |sent_total|
  - consensus_topics : |sent_<topic>| for all 6 topics
  - consensus_all    : total + all topic consensus columns

Both joint_pca and nested FAVAR architectures, DM and CW tests respectively.
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

root      = r'C:\Users\Javier\OneDrive - HEC Paris\Documentos\QuantCubeThesis\QuantCubeThesis'
INTER_CSV = root + r'\Taylor Rule\outputs\intermediate\step0_merged_to_macro.csv'
OUT_CSV   = root + r'\Taylor Rule\outputs\llm_consensus_results.csv'

H         = 1
INIT_FRAC = 0.50
START_DATE = '2007-01-01'

TOPICS = ['inflation', 'labor_market', 'economic_activity',
          'financial_conditions', 'monetary_policy', 'macro']


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
    sc  = [c for c in sent_cols if c in df_full.columns]
    mc_s = [c for c in mc_avail if c in df_full.columns]
    if not sc:
        print(f'  SKIP {label} — no columns found: {sent_cols}')
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

    r_exp = rmse(y_all, yh_m_all)
    r_bas = rmse(y_all, yh_b_all)
    imp   = (r_bas - r_exp) / r_bas * 100
    test_label = 'CW' if is_nested else 'DM'

    triples_m = [(int(t_all[i]), float(y_all[i]), float(yh_m_all[i])) for i in range(len(t_all))]
    triples_b = [(int(t_all[i]), float(y_all[i]), float(yh_b_all[i])) for i in range(len(t_all))]
    if is_nested:
        dm_t, dm_p = clark_west_test(triples_b, triples_m)
    else:
        dm_t, dm_p = diebold_mariano_test(triples_b, triples_m)

    def sw_rmse(frac):
        cutoff = int(ns * frac)
        mask   = np.array(t_all) >= cutoff
        return rmse(y_all[mask], yh_m_all[mask]) if mask.sum() >= 5 else np.nan

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
    print(f'  {label:<45}  arch={arch:<10}  n={ns:>3}  oos={len(t_all):>2}  '
          f'rmse={r_exp:.4f}  base={r_bas:.4f}  imp={imp:>+6.1f}%  '
          f'{test_label}-t={t_s}  p={p_s}  {sig_star(dm_p)}')
    return row


# ── Load & prepare ────────────────────────────────────────────────────────────
print('Loading base panel ...', flush=True)
df = pd.read_csv(INTER_CSV, parse_dates=['date'])
df = df[df['date'] >= START_DATE].sort_values('date').reset_index(drop=True)
df['cum_delta_h1'] = df['effective_rate'].shift(-H) - df['effective_rate']

mc_avail = [c for c in MACRO_REGRESSORS if c in df.columns]

# ── Build consensus columns: |sent_x| ────────────────────────────────────────
df['cons_total'] = df['sent_total'].abs()
for t in TOPICS:
    col = f'sent_{t}'
    if col in df.columns:
        df[f'cons_{t}'] = df[col].abs()

cons_total_cols  = ['cons_total']
cons_topic_cols  = [f'cons_{t}' for t in TOPICS if f'cons_{t}' in df.columns]
cons_all_cols    = cons_total_cols + cons_topic_cols

# Also test: signed total + consensus total together (does sign + magnitude both help?)
signed_plus_cons = ['sent_total', 'cons_total']

print(f'\nConsensus columns created: {cons_all_cols}')
print(f'Base macro regressors:     {mc_avail}')

# ── Run regressions ───────────────────────────────────────────────────────────
results = []

for arch in ['joint_pca', 'nested']:
    test = 'DM' if arch == 'joint_pca' else 'CW'
    print(f'\n{"="*100}')
    print(f'  {arch.upper()}  |  {test} test  |  delta h=1  |  50% init')
    print('='*100)

    # Reference: raw 4a total sent (best existing result)
    r = evaluate_model('4a_total_sent (reference)', arch,
                       ['sent_total'], mc_avail, df)
    if r: results.append(r)

    # 4b2 consensus variants
    r = evaluate_model('4b2_cons_total', arch,
                       cons_total_cols, mc_avail, df)
    if r: results.append(r)

    r = evaluate_model('4b2_cons_topics', arch,
                       cons_topic_cols, mc_avail, df)
    if r: results.append(r)

    r = evaluate_model('4b2_cons_all', arch,
                       cons_all_cols, mc_avail, df)
    if r: results.append(r)

    # Combination: signed + consensus together (does adding magnitude to sign help?)
    r = evaluate_model('4b2_signed_plus_cons', arch,
                       signed_plus_cons, mc_avail, df)
    if r: results.append(r)

# ── Save & print summary ──────────────────────────────────────────────────────
out_df = pd.DataFrame(results)
out_df.to_csv(OUT_CSV, index=False)
print(f'\n  Saved -> {OUT_CSV}')

print('\n' + '='*100)
print('  SUMMARY')
print('='*100)
print(f'  {"Model":<45}  {"Arch":<10}  {"RMSE":>7}  {"Base":>7}  '
      f'{"Imp%":>6}  {"t":>7}  {"p":>7}  {"sig"}')
print('  ' + '-'*100)
for _, r in out_df.iterrows():
    p_s = f'{r["dm_p"]:.4f}' if pd.notna(r["dm_p"]) else '    n/a'
    t_s = f'{r["dm_t"]:>7.3f}' if pd.notna(r["dm_t"]) else '    n/a'
    print(f'  {r["model"]:<45}  {r["arch"]:<10}  {r["rmse"]:>7.4f}  '
          f'{r["base_rmse"]:>7.4f}  {r["imp_pct"]:>+6.1f}%  {t_s}  {p_s}  {r["sig"]}')
