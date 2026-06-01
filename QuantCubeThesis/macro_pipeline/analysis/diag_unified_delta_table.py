"""
diag_unified_delta_table.py

Definitive unified comparison table for the delta target (cum_delta_h1, h=1).

Covers ALL model architectures and ALL methods with a CONSISTENT OOS protocol:
  - Expanding window, 50% initial training (76 OOS predictions for n=151)
  - Sub-window RMSE reported at t>=60%, 70%, 80%, 90% of n
    (same logic as utils.oos_rmse but starting at 50% not 60%)
  - DM test for non-nested models (joint PCA)
  - CW test for nested models (standard FAVAR: macro factors + sentiment)

Architectures included
  A. Joint PCA (non-nested, DM):
     LLM  × 8 variants: base/total/total_inter/topics/topics_matched/
                         topics_inter/all_sent/all_sent_inter
     Gardner × 8 variants (same scheme, g_* cols)
     Sharpe  × 8 variants (same scheme, s_* cols)

  B. Nested FAVAR (nested, CW):
     For each of LLM / Gardner / Sharpe, using all-doc zscore files:
     nest_total    : macro_pca_factors + total score
     nest_topics   : macro_pca_factors + all topic scores
     nest_matched  : macro_pca_factors + matched topic×macro interactions
     nest_total_ix : macro_pca_factors + total × all macro interactions

Issues flagged vs prior files
  - summary_table_delta.csv used 60% init → rmse_60 there ≠ RMSE_EXP here
  - delta_full_sweep.csv used fixed one-shot splits for oos60/70/80/90 (now corrected)
  - old 4h_gard/sharpe_joint_pca_200701.csv used CW for joint PCA (wrong, level target)
  - Gardner_topics_matched includes matched_g_total interaction w/o g_total main effect
    (minor quirk, consistent with step4h_joint_pca.py original)
  - PCA fitted on full sample (look-ahead, standard FAVAR practice per Bernanke et al. 2005)
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
GARD_CSV  = root + r'\Taylor Rule\nlp_output\Gardner_statements_minutes_speeches_press_conferences_zscore_nlp.csv'
SHAR_CSV  = root + r'\Taylor Rule\nlp_output\Sharpe_statements_minutes_speeches_press_conferences_zscore_nlp.csv'

H          = 1
INIT_FRAC  = 0.50
START_DATE = '2007-01-01'

MACRO_INTER = ['vix', 'gdp', 'unemployment_gap',
               'inflation_dev_from_target', 'implied_ffr', 'cfnai']
LLM_MATCHED = {
    'sent_total':                'implied_ffr',
    'sent_inflation':            'inflation_dev_from_target',
    'sent_labor_market':         'unemployment_gap',
    'sent_economic_activity':    'gdp',
    'sent_financial_conditions': 'vix',
    'sent_monetary_policy':      'implied_ffr',
    'sent_macro':                'cfnai',
}
GARDNER_MATCHED = {
    'g_total':     'implied_ffr',
    'g_inflation': 'inflation_dev_from_target',
    'g_labor_mkt': 'unemployment_gap',
    'g_econ_act':  'gdp',
    'g_fin_cond':  'vix',
    'g_mon_pol':   'implied_ffr',
}
SHARPE_MATCHED = {
    's_net':       'implied_ffr',
    's_positive':  'implied_ffr',
    's_negative':  'implied_ffr',
    's_consensus': 'implied_ffr',
}


# ── Shared helpers ────────────────────────────────────────────────────────────

def add_interactions(df, sent_cols, macro_cols, matched_map=None):
    inter_cols = []
    avail_macro = [c for c in macro_cols if c in df.columns]
    for sc in sent_cols:
        if sc not in df.columns:
            continue
        for mc in avail_macro:
            iname = f'inter_{sc}_x_{mc}'
            df[iname] = df[sc] * df[mc]
            inter_cols.append(iname)
    if matched_map:
        for sc, mc in matched_map.items():
            if sc in df.columns and mc in df.columns:
                iname = f'matched_{sc}_x_{mc}'
                if iname not in df.columns:
                    df[iname] = df[sc] * df[mc]
                    inter_cols.append(iname)
    return inter_cols


def sig_star(p):
    if p is None or (isinstance(p, float) and np.isnan(p)): return ''
    return '***' if p < .01 else ('**' if p < .05 else ('*' if p < .10 else ''))


def rmse(y, yh): return np.sqrt(np.mean((np.array(y) - np.array(yh)) ** 2))


def subwindow_rmse_dm(y_all, yh_m_all, yh_b_all, t_all, n, frac, is_nested):
    """RMSE and DM/CW stat for predictions where t >= int(n*frac)."""
    cutoff = int(n * frac)
    mask   = np.array(t_all) >= cutoff
    y_s, yh_m_s, yh_b_s = y_all[mask], yh_m_all[mask], yh_b_all[mask]
    ns = mask.sum()
    if ns < 5:
        return np.nan, np.nan, np.nan, ''
    r_m  = rmse(y_s, yh_m_s)
    triples_m = [(int(t_all[mask][i]), float(y_s[i]), float(yh_m_s[i])) for i in range(ns)]
    triples_b = [(int(t_all[mask][i]), float(y_s[i]), float(yh_b_s[i])) for i in range(ns)]
    if is_nested:
        dm_t, dm_p = clark_west_test(triples_b, triples_m)
    else:
        dm_t, dm_p = diebold_mariano_test(triples_b, triples_m)
    return r_m, dm_t, dm_p, sig_star(dm_p)


# ── Expanding OLS core ───────────────────────────────────────────────────────

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


def evaluate_model(label, arch, pca_cols_or_sent_cols, mc_avail, df_full, is_nested):
    """
    arch == 'joint_pca': run_pca on (pca_cols_or_sent_cols), OLS on joint factors.
                         Baseline = run_pca on (mc_avail), OLS on macro factors.
    arch == 'nested':    Fresh PCA on mc_avail for the exact sub-sample used.
                         X_model = macro_pca_factors + raw sentiment cols.
                         X_base  = macro_pca_factors only.
                         CW test.

    Returns row dict.
    """

    if arch == 'joint_pca':
        avail = [c for c in pca_cols_or_sent_cols if c in df_full.columns]
        mc_s  = [c for c in mc_avail              if c in df_full.columns]
        need  = list(dict.fromkeys(avail + mc_s + ['cum_delta_h1']))
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
        t_all, y_all, yh_m_all, yh_b_all = run_expanding(y, X_m, X_b, init_s)
        n_eval = ns

    elif arch == 'nested':
        sent_cols = [c for c in pca_cols_or_sent_cols if c in df_full.columns]
        if not sent_cols:
            return None
        need = list(dict.fromkeys(mc_avail + sent_cols + ['cum_delta_h1']))
        need = [c for c in need if c in df_full.columns]
        s    = df_full[need].dropna().reset_index(drop=True)
        ns   = len(s); init_s = int(ns * INIT_FRAC)
        if ns < 30: return None

        # Fresh PCA on the exact rows used (consistent baseline for this sub-sample)
        fac_b, _, _, nc_b = run_pca(s, mc_avail, var_threshold=PCA_VAR_MACRO)
        fac_b.columns = [f'MPC{i+1}' for i in range(nc_b)]

        y   = s['cum_delta_h1'].values
        S   = s[sent_cols].values
        X_m = sm.add_constant(np.hstack([fac_b.values, S]))
        X_b = sm.add_constant(fac_b.values)
        t_all, y_all, yh_m_all, yh_b_all = run_expanding(y, X_m, X_b, init_s)
        n_eval = ns

    else:
        raise ValueError(f'Unknown arch: {arch}')

    if len(t_all) < 5:
        return None

    test_label = 'CW' if is_nested else 'DM'
    r_exp = rmse(y_all, yh_m_all)
    r_bas = rmse(y_all, yh_b_all)
    imp   = (r_bas - r_exp) / r_bas * 100

    triples_m = [(int(t_all[i]), float(y_all[i]), float(yh_m_all[i])) for i in range(len(t_all))]
    triples_b = [(int(t_all[i]), float(y_all[i]), float(yh_b_all[i])) for i in range(len(t_all))]
    if is_nested:
        dm_t_exp, dm_p_exp = clark_west_test(triples_b, triples_m)
    else:
        dm_t_exp, dm_p_exp = diebold_mariano_test(triples_b, triples_m)

    row = {
        'model': label, 'arch': arch, 'test': test_label,
        'n_obs': n_eval, 'n_oos_exp': len(t_all),
        'rmse_exp':    round(r_exp, 4),
        'base_rmse':   round(r_bas, 4),
        'imp_exp':     round(imp, 1),
        'dm_t_exp':    round(dm_t_exp, 4) if not np.isnan(dm_t_exp) else np.nan,
        'dm_p_exp':    round(dm_p_exp, 4) if not np.isnan(dm_p_exp) else np.nan,
        'sig_exp':     sig_star(dm_p_exp),
    }

    # Sub-window RMSE at 60/70/80/90 thresholds
    for frac in [0.60, 0.70, 0.80, 0.90]:
        tag  = f'oos{int(frac*100)}'
        r_sw, t_sw, p_sw, s_sw = subwindow_rmse_dm(
            y_all, yh_m_all, yh_b_all, t_all, n_eval, frac, is_nested)
        row[f'rmse_{tag}']  = round(r_sw, 4) if not np.isnan(r_sw) else np.nan
        row[f'dm_t_{tag}']  = round(t_sw, 4) if not np.isnan(t_sw) else np.nan
        row[f'dm_p_{tag}']  = round(p_sw, 4) if not np.isnan(p_sw) else np.nan
        row[f'sig_{tag}']   = s_sw

    return row


# ── Load and prepare panel ────────────────────────────────────────────────────

print('  Loading data...', flush=True)

df_base = pd.read_csv(INTER_CSV, parse_dates=['date'])
df_base = df_base[df_base['date'] >= START_DATE].reset_index(drop=True)
if 'effective_rate' not in df_base.columns:
    macro = load_macro()[['date', 'effective_rate', 'implied_ffr']]
    df_base = pd.merge_asof(df_base.sort_values('date'), macro.sort_values('date'),
                            on='date', direction='backward')
df_base['cum_delta_h1'] = df_base['effective_rate'].shift(-H) - df_base['effective_rate']

GARDNER_REMAP = {
    'gardner_total': 'g_total', 'gardner_inf': 'g_inflation',
    'gardner_labor': 'g_labor_mkt', 'gardner_out': 'g_econ_act',
    'gardner_fin':   'g_fin_cond',  'gardner_mp':  'g_mon_pol',
}
SHARPE_REMAP = {
    'sharpe_net': 's_net', 'sharpe_positive': 's_positive',
    'sharpe_negative': 's_negative', 'sharpe_consensus': 's_consensus',
}

gard_raw = pd.read_csv(GARD_CSV, parse_dates=['meeting_date'])
gard_s   = [c for c in gard_raw.columns if c.startswith('gardner_') and 'consensus' not in c]
gard_mtg = (gard_raw.groupby('meeting_date')[gard_s].sum().reset_index()
            .rename(columns={**{'meeting_date': 'date'}, **GARDNER_REMAP}))

shar_raw = pd.read_csv(SHAR_CSV, parse_dates=['meeting_date'])
shar_s   = [c for c in shar_raw.columns if c.startswith('sharpe_')]
shar_mtg = (shar_raw.groupby('meeting_date')[shar_s].sum().reset_index()
            .rename(columns={**{'meeting_date': 'date'}, **SHARPE_REMAP}))

df_all = pd.merge_asof(df_base.sort_values('date'), gard_mtg.sort_values('date'),
                       on='date', direction='backward')
df_all = pd.merge_asof(df_all.sort_values('date'),  shar_mtg.sort_values('date'),
                       on='date', direction='backward').copy()

mc_avail = [c for c in MACRO_REGRESSORS if c in df_all.columns]

# Add all interaction columns once
LLM_TOTAL  = ['sent_total']
LLM_TOPICS = ['sent_inflation', 'sent_labor_market', 'sent_economic_activity',
               'sent_financial_conditions', 'sent_monetary_policy', 'sent_macro']
GARD_TOTAL  = ['g_total']
GARD_TOPICS = ['g_inflation', 'g_labor_mkt', 'g_econ_act', 'g_fin_cond', 'g_mon_pol']
SHAR_TOTAL  = ['s_net']
SHAR_TOPICS = ['s_positive', 's_negative', 's_consensus']

inter_llm_total   = add_interactions(df_all, LLM_TOTAL,  MACRO_INTER)
inter_llm_topics  = add_interactions(df_all, LLM_TOPICS, MACRO_INTER, LLM_MATCHED)
inter_llm_all     = list(dict.fromkeys(inter_llm_total + inter_llm_topics))
matched_llm       = [c for c in inter_llm_topics  if 'matched_' in c]

inter_gard_total  = add_interactions(df_all, GARD_TOTAL,  MACRO_INTER)
inter_gard_topics = add_interactions(df_all, GARD_TOPICS, MACRO_INTER, GARDNER_MATCHED)
inter_gard_all    = list(dict.fromkeys(inter_gard_total + inter_gard_topics))
matched_gard      = [c for c in inter_gard_topics if 'matched_' in c]

inter_shar_total  = add_interactions(df_all, SHAR_TOTAL,  MACRO_INTER)
inter_shar_topics = add_interactions(df_all, SHAR_TOPICS, MACRO_INTER, SHARPE_MATCHED)
inter_shar_all    = list(dict.fromkeys(inter_shar_total + inter_shar_topics))
matched_shar      = [c for c in inter_shar_topics if 'matched_' in c]

n_clean = df_all[mc_avail + ['cum_delta_h1']].dropna().shape[0]
print(f'  Panel: {len(df_all)} meetings | clean for regression: {n_clean}')
print(f'  INIT_FRAC={INIT_FRAC}')

# ── Define all model specs ────────────────────────────────────────────────────

def joint_variants(prefix, total_cols, topic_cols, inter_t, matched, inter_top, inter_all):
    all_s = total_cols + topic_cols
    return [
        (f'{prefix}_base',            'joint_pca', mc_avail,                             False),
        (f'{prefix}_total',           'joint_pca', mc_avail + total_cols,                False),
        (f'{prefix}_total_inter',     'joint_pca', mc_avail + total_cols + inter_t,      False),
        (f'{prefix}_topics',          'joint_pca', mc_avail + topic_cols,                False),
        (f'{prefix}_topics_matched',  'joint_pca', mc_avail + topic_cols + matched,      False),
        (f'{prefix}_topics_inter',    'joint_pca', mc_avail + topic_cols + inter_top,    False),
        (f'{prefix}_all_sent',        'joint_pca', mc_avail + all_s,                     False),
        (f'{prefix}_all_sent_inter',  'joint_pca', mc_avail + all_s + inter_all,         False),
    ]

def nested_variants(prefix, total_cols, topic_cols, inter_t, matched_inter, inter_top):
    all_s = total_cols + topic_cols
    matched_no_total = [c for c in matched_inter if not any(t in c for t in total_cols)]
    return [
        (f'{prefix}_nest_total',      'nested', total_cols,                               True),
        (f'{prefix}_nest_topics',     'nested', topic_cols,                               True),
        (f'{prefix}_nest_matched',    'nested', matched_no_total or matched_inter,        True),
        (f'{prefix}_nest_total_ix',   'nested', inter_t,                                  True),
    ]

SPECS = (
    joint_variants('LLM', LLM_TOTAL, LLM_TOPICS,
                   inter_llm_total, matched_llm, inter_llm_topics, inter_llm_all) +
    nested_variants('LLM', LLM_TOTAL, LLM_TOPICS,
                    inter_llm_total, matched_llm, inter_llm_topics) +
    joint_variants('Gardner', GARD_TOTAL, GARD_TOPICS,
                   inter_gard_total, matched_gard, inter_gard_topics, inter_gard_all) +
    nested_variants('Gardner', GARD_TOTAL, GARD_TOPICS,
                    inter_gard_total, matched_gard, inter_gard_topics) +
    joint_variants('Sharpe', SHAR_TOTAL, SHAR_TOPICS,
                   inter_shar_total, matched_shar, inter_shar_topics, inter_shar_all) +
    nested_variants('Sharpe', SHAR_TOTAL, SHAR_TOPICS,
                    inter_shar_total, matched_shar, inter_shar_topics)
)

# ── Run all specs ─────────────────────────────────────────────────────────────

print('\n' + '=' * 120)
print('  UNIFIED DELTA TABLE  |  h=1  |  2007-2025  |  50% init expanding window')
print('  Architecture: joint_pca=non-nested(DM)  nested=nested-FAVAR(CW)')
print('=' * 120)

hdr = (f'  {"Model":<32}  {"Arch":<10}  {"Tst":<3}  {"N":>3}  '
       f'{"RMSE_exp":>8}  {"t_exp":>7}  {"p_exp":>6}  {"sig":>3}  '
       f'{"RMSE60":>7}  {"t60":>6}  '
       f'{"RMSE70":>7}  {"t70":>6}  '
       f'{"RMSE80":>7}  {"t80":>6}  '
       f'{"RMSE90":>7}  {"t90":>6}')
print(hdr)
print('  ' + '-' * 120)

results = []
prev_block = None
for label, arch, cols, is_nested in SPECS:
    block = label.split('_')[0]
    if block != prev_block and prev_block is not None:
        print('  ' + '-' * 120)
    prev_block = block

    row = evaluate_model(label, arch, cols, mc_avail, df_all, is_nested)
    if row is None:
        print(f'  {label:<32}  (skipped)')
        continue
    results.append(row)

    def f(v, w=7, d=4):
        return f'{v:{w}.{d}f}' if (isinstance(v, float) and not np.isnan(v)) else f'{"n/a":>{w}}'
    def ft(v):
        return f'{v:>6.3f}' if (isinstance(v, float) and not np.isnan(v)) else '   n/a'

    print(f'  {label:<32}  {arch:<10}  {row["test"]:<3}  {row["n_oos_exp"]:>3}  '
          f'{f(row.get("rmse_exp"), 8)}  {ft(row.get("dm_t_exp"))}  '
          f'{f(row.get("dm_p_exp"), 6, 4)}{row.get("sig_exp",""):>3}  '
          f'{f(row.get("rmse_oos60"))}  {ft(row.get("dm_t_oos60"))}  '
          f'{f(row.get("rmse_oos70"))}  {ft(row.get("dm_t_oos70"))}  '
          f'{f(row.get("rmse_oos80"))}  {ft(row.get("dm_t_oos80"))}  '
          f'{f(row.get("rmse_oos90"))}  {ft(row.get("dm_t_oos90"))}')


# ── Save and summarise ────────────────────────────────────────────────────────

out_df   = pd.DataFrame(results)
out_path = root + r'\Taylor Rule\outputs\unified_delta_table.csv'
out_df.to_csv(out_path, index=False)
print(f'\n  Saved -> unified_delta_table.csv  ({len(out_df)} rows)')

print('\n' + '=' * 80)
print('  BEST PER METHOD × ARCHITECTURE  (by expanding-window RMSE, excl. base)')
print('=' * 80)
for block in ['LLM', 'Gardner', 'Sharpe']:
    for arch in ['joint_pca', 'nested']:
        sub = out_df[(out_df['model'].str.startswith(block)) &
                     (out_df['arch'] == arch) &
                     ~out_df['model'].str.contains('_base')].copy()
        if sub.empty or sub['rmse_exp'].isna().all(): continue
        best = sub.loc[sub['rmse_exp'].idxmin()]
        p    = best.get('dm_p_exp', np.nan)
        print(f'  {best["model"]:<38}  RMSE={best["rmse_exp"]:.4f}'
              f'  base={best["base_rmse"]:.4f}'
              f'  imp={best["imp_exp"]:+.1f}%'
              f'  {best["test"]}-t={best["dm_t_exp"]:.3f}'
              f'  p={p:.4f}  {best["sig_exp"]}' if not np.isnan(p) else
              f'  {best["model"]:<38}  RMSE={best["rmse_exp"]:.4f}'
              f'  base={best["base_rmse"]:.4f}  imp={best["imp_exp"]:+.1f}%'
              f'  {best["test"]}-t={best["dm_t_exp"]:.3f}  p=n/a')

print('\n  NOTES:')
print('  1. All models: 50% init expanding window, retrain at each step.')
print('  2. Sub-windows 60/70/80/90: RMSE for predictions where t >= frac*n.')
print('  3. PCA fitted on full sample (standard FAVAR, Bernanke 2005) — mild look-ahead.')
print('  4. Joint PCA: non-nested vs macro FAVAR -> DM test.')
print('  5. Nested FAVAR: macro PCA factors + raw sentiment -> CW test (nested).')
print('  6. Gardner_topics_matched includes matched_g_total*implied_ffr without')
print('     g_total as main effect (consistent with step4h_joint_pca.py original).')
