"""
diag_doctype_comparison.py

Per-document-type comparison on the delta target (cum_delta_h1, h=1).

For each method (LLM, Gardner, Sharpe), runs the best all-sent spec separately
for each document type:
  LLM      : minutes | statement | speech | press_conf_prepared | press_conf_qa | all_docs
  Gardner  : minutes | statements | speeches | press_conferences | all_docs
  Sharpe   : same as Gardner

Two architectures:
  Panel A - Joint PCA (non-nested, DM test)
  Panel B - Nested FAVAR (nested, CW test):
            macro PCA factors + raw sentiment scores as OLS regressors

Protocol: 50% init expanding window vs Macro FAVAR baseline.
"""
import sys, warnings
import numpy  as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from config import MACRO_REGRESSORS, PCA_VAR_MACRO
from utils  import run_pca, diebold_mariano_test, clark_west_test
from step0_aggregate import load_macro

root      = r'C:\Users\Javier\OneDrive - HEC Paris\Documentos\QuantCubeThesis\QuantCubeThesis'
INTER_CSV = root + r'\Taylor Rule\outputs\intermediate\step0_merged_to_macro.csv'
NLP_DIR   = root + r'\Taylor Rule\nlp_output'

H          = 1
INIT_FRAC  = 0.50
START_DATE = '2007-01-01'

TOPICS = ['inflation', 'labor_market', 'economic_activity',
          'financial_conditions', 'monetary_policy', 'macro']

GARD_REMAP = {'gardner_total': 'g_total', 'gardner_inf': 'g_inflation',
              'gardner_labor': 'g_labor_mkt', 'gardner_out': 'g_econ_act',
              'gardner_fin':   'g_fin_cond',  'gardner_mp':  'g_mon_pol'}
SHAR_REMAP = {'sharpe_net': 's_net', 'sharpe_positive': 's_positive',
              'sharpe_negative': 's_negative', 'sharpe_consensus': 's_consensus'}


# ── Helpers ───────────────────────────────────────────────────────────────────

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
            t_all.append(t); y_all.append(float(y[t]))
            yh_m_all.append(float(mm.predict(X_m[t:t+1])[0]))
            yh_b_all.append(float(mb.predict(X_b[t:t+1])[0]))
        except Exception:
            pass
    return (np.array(t_all), np.array(y_all),
            np.array(yh_m_all), np.array(yh_b_all))

def evaluate(label, df_full, sent_cols, mc_avail, arch='joint_pca'):
    """Joint PCA (DM) or Nested FAVAR (CW) vs macro FAVAR baseline."""
    sc    = [c for c in sent_cols if c in df_full.columns]
    mc_s  = [c for c in mc_avail  if c in df_full.columns]
    if not sc:
        return None

    if arch == 'joint_pca':
        avail = [c for c in mc_s + sc if c in df_full.columns]
        need  = list(dict.fromkeys(avail + ['cum_delta_h1']))
        need  = [c for c in need if c in df_full.columns]
        s     = df_full[need].dropna().reset_index(drop=True)
        ns    = len(s); init_s = int(ns * INIT_FRAC)
        if ns < 30: return None

        fac_m, _, _, nc_m = run_pca(s, avail, var_threshold=PCA_VAR_MACRO)
        fac_m.columns = [f'JPC{i+1}' for i in range(nc_m)]
        fac_b, _, _, nc_b = run_pca(s, mc_s, var_threshold=PCA_VAR_MACRO)
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

    r_m = rmse(y_all, yh_m_all)
    r_b = rmse(y_all, yh_b_all)
    imp = (r_b - r_m) / r_b * 100

    triples_m = [(int(t_all[i]), float(y_all[i]), float(yh_m_all[i])) for i in range(len(t_all))]
    triples_b = [(int(t_all[i]), float(y_all[i]), float(yh_b_all[i])) for i in range(len(t_all))]

    if arch == 'nested':
        dm_t, dm_p = clark_west_test(triples_b, triples_m)
        test_label = 'CW'
    else:
        dm_t, dm_p = diebold_mariano_test(triples_b, triples_m)
        test_label = 'DM'

    t_s = f'{dm_t:>6.3f}' if not np.isnan(dm_t) else '   n/a'
    p_s = f'{dm_p:.4f}' if not (isinstance(dm_p, float) and np.isnan(dm_p)) else '   n/a'
    sig = sig_star(dm_p)

    print(f'  {label:<45}  n={ns:>3} oos={len(t_all):>2}  '
          f'mdl={r_m:.4f}  base={r_b:.4f}  '
          f'imp={imp:>+6.1f}%  {test_label}-t={t_s}  p={p_s}  {sig}')

    return {'model': label, 'arch': arch, 'test': test_label,
            'n_obs': ns, 'n_oos': len(t_all),
            'rmse': round(r_m, 4), 'base_rmse': round(r_b, 4),
            'imp_pct': round(imp, 1),
            'dm_t': round(dm_t, 4) if not np.isnan(dm_t) else np.nan,
            'dm_p': round(dm_p, 4) if not np.isnan(dm_p) else np.nan,
            'sig': sig}


# ── Load base panel ───────────────────────────────────────────────────────────

print('Loading base panel...', flush=True)
df_base = pd.read_csv(INTER_CSV, parse_dates=['date'])
df_base = df_base[df_base['date'] >= START_DATE].reset_index(drop=True)
if 'effective_rate' not in df_base.columns:
    macro = load_macro()[['date', 'effective_rate', 'implied_ffr']]
    df_base = pd.merge_asof(df_base.sort_values('date'), macro.sort_values('date'),
                            on='date', direction='backward')
df_base['cum_delta_h1'] = df_base['effective_rate'].shift(-H) - df_base['effective_rate']
mc_avail = [c for c in MACRO_REGRESSORS if c in df_base.columns]


# ── Helper: load Gardner/Sharpe NLP file and merge ───────────────────────────

def load_dict_scores(filename, remap):
    path    = NLP_DIR + '\\' + filename
    raw     = pd.read_csv(path, parse_dates=['meeting_date'])
    src_cols = [c for c in raw.columns if c in remap]
    mtg     = (raw.groupby('meeting_date')[src_cols].sum()
               .reset_index().rename(columns={**{'meeting_date': 'date'}, **remap}))
    return mtg


# ── Doc-type lists (reused for both architectures) ───────────────────────────

LLM_DOC_TYPES = [
    ('all_docs',        '',                           'All documents (combined)'),
    ('minutes',         '_minutes',                   'Minutes'),
    ('statement',       '_statement',                 'Statements'),
    ('speech',          '_speech',                    'Speeches'),
    ('press_conf_prep', '_press_conference_prepared', 'Press conf. (prepared)'),
    ('press_conf_qa',   '_press_conference_qa',       'Press conf. (Q&A)'),
]

GARD_FILES = [
    ('all_docs',    'Gardner_statements_minutes_speeches_press_conferences_zscore_nlp.csv', 'All documents'),
    ('minutes',     'Gardner_minutes_zscore_nlp.csv',           'Minutes'),
    ('statements',  'Gardner_statements_zscore_nlp.csv',        'Statements'),
    ('speeches',    'Gardner_speeches_zscore_nlp.csv',          'Speeches'),
    ('press_conf',  'Gardner_press_conferences_zscore_nlp.csv', 'Press conferences'),
]

SHAR_FILES = [
    ('all_docs',    'Sharpe_statements_minutes_speeches_press_conferences_zscore_nlp.csv', 'All documents'),
    ('minutes',     'Sharpe_minutes_zscore_nlp.csv',           'Minutes'),
    ('statements',  'Sharpe_statements_zscore_nlp.csv',        'Statements'),
    ('speeches',    'Sharpe_speeches_zscore_nlp.csv',          'Speeches'),
    ('press_conf',  'Sharpe_press_conferences_zscore_nlp.csv', 'Press conferences'),
]


def run_llm_block(arch):
    for key, suffix, desc in LLM_DOC_TYPES:
        cols = [f'sent_total{suffix}'] + [f'sent_{t}{suffix}' for t in TOPICS]
        cols = [c for c in cols if c in df_base.columns]
        if not cols:
            continue
        row = evaluate(f'LLM_{key}', df_base, cols, mc_avail, arch=arch)
        if row:
            row['method'] = 'LLM'; row['doc_type'] = desc
            results.append(row)

def run_gard_block(arch):
    for key, fname, desc in GARD_FILES:
        mtg  = load_dict_scores(fname, GARD_REMAP)
        df_g = pd.merge_asof(df_base.sort_values('date'), mtg.sort_values('date'),
                             on='date', direction='backward').copy()
        gard_cols = [c for c in GARD_REMAP.values() if c in df_g.columns]
        row = evaluate(f'Gardner_{key}', df_g, gard_cols, mc_avail, arch=arch)
        if row:
            row['method'] = 'Gardner'; row['doc_type'] = desc
            results.append(row)

def run_shar_block(arch):
    for key, fname, desc in SHAR_FILES:
        mtg  = load_dict_scores(fname, SHAR_REMAP)
        df_s = pd.merge_asof(df_base.sort_values('date'), mtg.sort_values('date'),
                             on='date', direction='backward').copy()
        shar_cols = [c for c in SHAR_REMAP.values() if c in df_s.columns]
        row = evaluate(f'Sharpe_{key}', df_s, shar_cols, mc_avail, arch=arch)
        if row:
            row['method'] = 'Sharpe'; row['doc_type'] = desc
            results.append(row)


# ── Run all models ────────────────────────────────────────────────────────────

results = []

for arch, test_name in [('joint_pca', 'DM'), ('nested', 'CW')]:
    arch_label = 'Joint PCA' if arch == 'joint_pca' else 'Nested FAVAR'
    print('\n' + '=' * 110)
    print(f'  PANEL: {arch_label.upper()}  |  {test_name} test  |  delta h=1  |  50% init')
    print('=' * 110)

    print('\n  --- LLM SENTIMENT ---')
    run_llm_block(arch)

    print('\n  --- GARDNER (2022) ---')
    run_gard_block(arch)

    print('\n  --- SHARPE (2023) ---')
    run_shar_block(arch)


# ── Save ──────────────────────────────────────────────────────────────────────

out_df   = pd.DataFrame(results)
out_path = root + r'\Taylor Rule\outputs\doctype_comparison_delta.csv'
out_df.to_csv(out_path, index=False)
print(f'\n  Saved -> doctype_comparison_delta.csv  ({len(out_df)} rows)')

print('\n' + '=' * 90)
print('  SUMMARY: Best doc type per method x architecture')
print('=' * 90)
for arch in ['joint_pca', 'nested']:
    test = 'DM' if arch == 'joint_pca' else 'CW'
    for method in ['LLM', 'Gardner', 'Sharpe']:
        sub = out_df[(out_df['method'] == method) & (out_df['arch'] == arch)].copy()
        if sub.empty: continue
        best = sub.loc[sub['rmse'].idxmin()]
        p    = best['dm_p']
        p_s  = f'{p:.4f}' if not (isinstance(p, float) and np.isnan(p)) else 'n/a'
        print(f'  {arch:<10} {method:<10} best: {best["doc_type"]:<30}  '
              f'RMSE={best["rmse"]:.4f}  imp={best["imp_pct"]:+.1f}%  '
              f'{test}-t={best["dm_t"]:.3f}  p={p_s}  {best["sig"]}')
