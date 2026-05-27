"""
diag_delta_full_sweep.py

Full sweep: all 8 variants × 3 methods on the DELTA target (cum_delta_h1).

For each model:
  - Expanding window (50% initial training, retrain each step)  → RMSE + DM
  - Fixed OOS splits 60 / 70 / 80 / 90 %                       → RMSE + DM

Test: DM (Diebold-Mariano, one-sided) throughout.
Joint PCA makes all specs non-nested vs macro FAVAR → DM is correct.

Variants (matching step4h_joint_pca.py):
  base            macro only                     (= baseline)
  total           macro + total score
  total_inter     macro + total + total×macro interactions
  topics          macro + topic scores (no total)
  topics_matched  macro + topics + matched topic×macro interactions
  topics_inter    macro + topics + all topic×macro interactions
  all_sent        macro + total + all topics
  all_sent_inter  macro + total + topics + all interactions
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
GARD_CSV  = root + r'\Taylor Rule\nlp_output\Gardner_statements_minutes_speeches_press_conferences_zscore_nlp.csv'
SHAR_CSV  = root + r'\Taylor Rule\nlp_output\Sharpe_statements_minutes_speeches_press_conferences_zscore_nlp.csv'

H          = 1
TRAIN_FRAC = 0.50          # expanding window initial fraction
OOS_FRACS  = [0.60, 0.70, 0.80, 0.90]   # fixed-split training fractions
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def add_interactions(df, sent_cols, macro_cols, matched_map=None):
    inter_cols = []
    avail_macro = [c for c in macro_cols if c in df.columns]
    for sc in sent_cols:
        if sc not in df.columns:
            continue
        for mc in avail_macro:
            if mc not in df.columns:
                continue
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
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ''
    return '***' if p < .01 else ('**' if p < .05 else ('*' if p < .10 else ''))


def run_dm(y, yh_m, yh_b):
    """DM test: returns (t, p). p=NaN if model worse (one-sided H1: model better)."""
    n = len(y)
    triples_m = [(i, float(y[i]), float(yh_m[i])) for i in range(n)]
    triples_b = [(i, float(y[i]), float(yh_b[i])) for i in range(n)]
    return diebold_mariano_test(triples_b, triples_m)


def rmse(y, yh):
    return np.sqrt(np.mean((np.array(y) - np.array(yh)) ** 2))


# ── Expanding window evaluation ───────────────────────────────────────────────

def expanding_eval(y, X_m, X_b, init):
    """Expanding OLS from init; returns (yh_model, yh_base) arrays."""
    n = len(y)
    yh_m, yh_b = np.full(n, np.nan), np.full(n, np.nan)
    for t in range(init, n):
        try:
            mm = sm.OLS(y[:t], X_m[:t]).fit()
            mb = sm.OLS(y[:t], X_b[:t]).fit()
            yh_m[t] = mm.predict(X_m[t:t+1])[0]
            yh_b[t] = mb.predict(X_b[t:t+1])[0]
        except Exception:
            pass
    mask = ~np.isnan(yh_m) & ~np.isnan(yh_b)
    return y[mask], yh_m[mask], yh_b[mask]


# ── Fixed-split evaluation ────────────────────────────────────────────────────

def fixed_split_eval(y, X_m, X_b, train_frac):
    """Train on first train_frac, predict on rest (one-shot OLS)."""
    n       = len(y)
    n_train = int(n * train_frac)
    if n - n_train < 5:
        return None, None, None
    try:
        mm = sm.OLS(y[:n_train], X_m[:n_train]).fit()
        mb = sm.OLS(y[:n_train], X_b[:n_train]).fit()
        y_test   = y[n_train:]
        yh_m_oos = mm.predict(X_m[n_train:])
        yh_b_oos = mb.predict(X_b[n_train:])
        return y_test, yh_m_oos, yh_b_oos
    except Exception:
        return None, None, None


# ── Build factors from PCA ────────────────────────────────────────────────────

def build_factors(sub, pca_cols, mc_cols):
    avail_m = [c for c in pca_cols if c in sub.columns]
    avail_b = [c for c in mc_cols  if c in sub.columns]
    fac_m, _, _, nc_m = run_pca(sub, avail_m, var_threshold=PCA_VAR_MACRO)
    fac_b, _, _, nc_b = run_pca(sub, avail_b, var_threshold=PCA_VAR_MACRO)
    fac_m.columns = [f'JPC{i+1}' for i in range(nc_m)]
    fac_b.columns = [f'BPC{i+1}' for i in range(nc_b)]
    X_m = sm.add_constant(fac_m.values)
    X_b = sm.add_constant(fac_b.values)
    return X_m, X_b, nc_m, nc_b


# ── Full evaluation for one variant ──────────────────────────────────────────

def evaluate_variant(sub, pca_cols, mc_cols, label):
    """Returns dict with all RMSE and DM results."""
    need = list(dict.fromkeys(pca_cols + mc_cols + ['cum_delta_h1']))
    need = [c for c in need if c in sub.columns]
    s    = sub[need].dropna().reset_index(drop=True)
    n    = len(s)
    if n < 30:
        return None

    y    = s['cum_delta_h1'].values
    X_m, X_b, nc_m, nc_b = build_factors(s, pca_cols, mc_cols)
    init = int(n * TRAIN_FRAC)

    row = {'model': label, 'n_obs': n, 'n_jpc': nc_m, 'n_bpc': nc_b}

    # --- Expanding window (50% init) ---
    y_e, yh_m_e, yh_b_e = expanding_eval(y, X_m, X_b, init)
    n_oos = len(y_e)
    if n_oos >= 5:
        r_m  = rmse(y_e, yh_m_e)
        r_b  = rmse(y_e, yh_b_e)
        imp  = (r_b - r_m) / r_b * 100
        dm_t, dm_p = run_dm(y_e, yh_m_e, yh_b_e)
        row.update({'n_oos_exp': n_oos,
                    'rmse_exp': round(r_m, 4), 'base_rmse_exp': round(r_b, 4),
                    'imp_exp': round(imp, 1),
                    'dm_t_exp': round(dm_t, 4) if not np.isnan(dm_t) else np.nan,
                    'dm_p_exp': round(dm_p, 4) if not np.isnan(dm_p) else np.nan,
                    'sig_exp': sig_star(dm_p)})
    else:
        row.update({'n_oos_exp': n_oos, 'rmse_exp': np.nan, 'base_rmse_exp': np.nan,
                    'imp_exp': np.nan, 'dm_t_exp': np.nan, 'dm_p_exp': np.nan, 'sig_exp': ''})

    # --- Fixed splits ---
    for frac in OOS_FRACS:
        tag = f'oos{int(frac*100)}'
        y_f, yh_m_f, yh_b_f = fixed_split_eval(y, X_m, X_b, frac)
        if y_f is None or len(y_f) < 5:
            row.update({f'rmse_{tag}': np.nan, f'base_{tag}': np.nan,
                        f'imp_{tag}': np.nan, f'dm_t_{tag}': np.nan,
                        f'dm_p_{tag}': np.nan, f'sig_{tag}': ''})
            continue
        r_m  = rmse(y_f, yh_m_f)
        r_b  = rmse(y_f, yh_b_f)
        imp  = (r_b - r_m) / r_b * 100
        dm_t, dm_p = run_dm(y_f, yh_m_f, yh_b_f)
        row.update({f'rmse_{tag}':   round(r_m, 4),
                    f'base_{tag}':   round(r_b, 4),
                    f'imp_{tag}':    round(imp, 1),
                    f'dm_t_{tag}':   round(dm_t, 4) if not np.isnan(dm_t) else np.nan,
                    f'dm_p_{tag}':   round(dm_p, 4) if not np.isnan(dm_p) else np.nan,
                    f'sig_{tag}':    sig_star(dm_p)})

    return row


# ── Load panels ───────────────────────────────────────────────────────────────

print('  Loading panels...', flush=True)

# Base LLM panel
df_llm = pd.read_csv(INTER_CSV, parse_dates=['date'])
df_llm = df_llm[df_llm['date'] >= START_DATE].reset_index(drop=True)
if 'effective_rate' not in df_llm.columns:
    macro = load_macro()[['date', 'effective_rate', 'implied_ffr']]
    df_llm = pd.merge_asof(df_llm.sort_values('date'), macro.sort_values('date'),
                           on='date', direction='backward')
df_llm['cum_delta_h1'] = df_llm['effective_rate'].shift(-H) - df_llm['effective_rate']

# Gardner scores → rename to g_xxx
GARDNER_REMAP = {
    'gardner_total': 'g_total', 'gardner_inf': 'g_inflation',
    'gardner_labor': 'g_labor_mkt', 'gardner_out': 'g_econ_act',
    'gardner_fin':   'g_fin_cond',  'gardner_mp':  'g_mon_pol',
}
gard_raw = pd.read_csv(GARD_CSV, parse_dates=['meeting_date'])
gard_score_cols = [c for c in gard_raw.columns
                   if c.startswith('gardner_') and 'consensus' not in c]
gard_mtg = (gard_raw.groupby('meeting_date')[gard_score_cols]
            .sum().reset_index()
            .rename(columns={**{'meeting_date': 'date'}, **GARDNER_REMAP}))

# Sharpe scores → rename to s_xxx
SHARPE_REMAP = {
    'sharpe_net': 's_net', 'sharpe_positive': 's_positive',
    'sharpe_negative': 's_negative', 'sharpe_consensus': 's_consensus',
}
shar_raw = pd.read_csv(SHAR_CSV, parse_dates=['meeting_date'])
shar_score_cols = [c for c in shar_raw.columns if c.startswith('sharpe_')]
shar_mtg = (shar_raw.groupby('meeting_date')[shar_score_cols]
            .sum().reset_index()
            .rename(columns={**{'meeting_date': 'date'}, **SHARPE_REMAP}))

# Merge onto base panel
df_all = pd.merge_asof(df_llm.sort_values('date'),
                       gard_mtg.sort_values('date'), on='date', direction='backward')
df_all = pd.merge_asof(df_all.sort_values('date'),
                       shar_mtg.sort_values('date'), on='date', direction='backward')
df_all = df_all.copy()  # avoid SettingWithCopyWarning when adding interactions

mc_avail = [c for c in MACRO_REGRESSORS if c in df_all.columns]

# ── Define sentiment column sets ──────────────────────────────────────────────

LLM_TOTAL  = ['sent_total']
LLM_TOPICS = ['sent_inflation', 'sent_labor_market', 'sent_economic_activity',
               'sent_financial_conditions', 'sent_monetary_policy', 'sent_macro']
GARD_TOTAL  = ['g_total']
GARD_TOPICS = ['g_inflation', 'g_labor_mkt', 'g_econ_act', 'g_fin_cond', 'g_mon_pol']
SHAR_TOTAL  = ['s_net']
SHAR_TOPICS = ['s_positive', 's_negative', 's_consensus']

# Add all interaction columns to df_all (all columns live in the same df)
inter_llm_total   = add_interactions(df_all, LLM_TOTAL,  MACRO_INTER)
inter_llm_topics  = add_interactions(df_all, LLM_TOPICS, MACRO_INTER, LLM_MATCHED)
inter_llm_all     = list(dict.fromkeys(inter_llm_total + inter_llm_topics))

inter_gard_total  = add_interactions(df_all, GARD_TOTAL,  MACRO_INTER)
inter_gard_topics = add_interactions(df_all, GARD_TOPICS, MACRO_INTER, GARDNER_MATCHED)
inter_gard_all    = list(dict.fromkeys(inter_gard_total + inter_gard_topics))

inter_shar_total  = add_interactions(df_all, SHAR_TOTAL,  MACRO_INTER)
inter_shar_topics = add_interactions(df_all, SHAR_TOPICS, MACRO_INTER, SHARPE_MATCHED)
inter_shar_all    = list(dict.fromkeys(inter_shar_total + inter_shar_topics))

matched_llm_inter   = [c for c in inter_llm_topics   if 'matched_' in c]
matched_gard_inter  = [c for c in inter_gard_topics  if 'matched_' in c]
matched_shar_inter  = [c for c in inter_shar_topics  if 'matched_' in c]

# ── Define all variants ───────────────────────────────────────────────────────

def make_variants(method, total_cols, topic_cols,
                  inter_total, matched_inter, inter_topics, inter_all):
    all_sent = total_cols + topic_cols
    return [
        (f'{method}_base',            mc_avail),
        (f'{method}_total',           mc_avail + total_cols),
        (f'{method}_total_inter',     mc_avail + total_cols + inter_total),
        (f'{method}_topics',          mc_avail + topic_cols),
        (f'{method}_topics_matched',  mc_avail + topic_cols + matched_inter),
        (f'{method}_topics_inter',    mc_avail + topic_cols + inter_topics),
        (f'{method}_all_sent',        mc_avail + all_sent),
        (f'{method}_all_sent_inter',  mc_avail + all_sent + inter_all),
    ]

VARIANTS = (
    make_variants('LLM',     LLM_TOTAL,  LLM_TOPICS,
                  inter_llm_total,  matched_llm_inter,  inter_llm_topics,  inter_llm_all) +
    make_variants('Gardner', GARD_TOTAL, GARD_TOPICS,
                  inter_gard_total, matched_gard_inter, inter_gard_topics, inter_gard_all) +
    make_variants('Sharpe',  SHAR_TOTAL, SHAR_TOPICS,
                  inter_shar_total, matched_shar_inter, inter_shar_topics, inter_shar_all)
)


# ── Run all variants ──────────────────────────────────────────────────────────

print('\n' + '=' * 110)
print('  DELTA TARGET FULL SWEEP  |  h=1  |  2007-2025  |  DM test (one-sided)')
print('  Expanding window (50% init) + Fixed OOS splits 60/70/80/90')
print('=' * 110)

hdr  = (f'  {"Model":<28}  {"N":>3}  '
        f'{"RMSE_exp":>8}  {"DM_exp":>7}  {"p_exp":>6}  '
        f'{"RMSE60":>7}  {"DM60":>6}  {"p60":>6}  '
        f'{"RMSE70":>7}  {"DM70":>6}  {"p70":>6}  '
        f'{"RMSE80":>7}  {"DM80":>6}  {"p80":>6}  '
        f'{"RMSE90":>7}  {"DM90":>6}  {"p90":>6}')
print(hdr)
print('  ' + '-' * 110)

results = []
prev_method = None
for label, pca_cols in VARIANTS:
    method = label.split('_')[0]
    if method != prev_method and prev_method is not None:
        print('  ' + '-' * 110)
    prev_method = method

    row = evaluate_variant(df_all, pca_cols, mc_avail, label)
    if row is None:
        print(f'  {label:<28}  (skipped — insufficient data)')
        continue
    results.append(row)

    def fmt(v, w=7, d=4):
        return f'{v:{w}.{d}f}' if pd.notna(v) else f'{"n/a":>{w}}'
    def fmt_t(v):
        return f'{v:>6.3f}' if pd.notna(v) else '   n/a'
    def fmt_p(v, s):
        return f'{v:>6.4f}{s:>3}' if pd.notna(v) else '    n/a   '

    print(f'  {label:<28}  {row["n_obs"]:>3}  '
          f'{fmt(row.get("rmse_exp"), 8)}  '
          f'{fmt_t(row.get("dm_t_exp"))}  {fmt(row.get("dm_p_exp"), 6, 4)}{row.get("sig_exp",""):>3}  '
          f'{fmt(row.get("rmse_oos60"))}  {fmt_t(row.get("dm_t_oos60"))}  {fmt(row.get("dm_p_oos60"), 6, 4)}{row.get("sig_oos60",""):>3}  '
          f'{fmt(row.get("rmse_oos70"))}  {fmt_t(row.get("dm_t_oos70"))}  {fmt(row.get("dm_p_oos70"), 6, 4)}{row.get("sig_oos70",""):>3}  '
          f'{fmt(row.get("rmse_oos80"))}  {fmt_t(row.get("dm_t_oos80"))}  {fmt(row.get("dm_p_oos80"), 6, 4)}{row.get("sig_oos80",""):>3}  '
          f'{fmt(row.get("rmse_oos90"))}  {fmt_t(row.get("dm_t_oos90"))}  {fmt(row.get("dm_p_oos90"), 6, 4)}{row.get("sig_oos90",""):>3}')


# ── Save ──────────────────────────────────────────────────────────────────────

out_df   = pd.DataFrame(results)
out_path = root + r'\Taylor Rule\outputs\delta_full_sweep.csv'
out_df.to_csv(out_path, index=False)
print(f'\n  Saved -> delta_full_sweep.csv  ({len(out_df)} rows)')


# ── Best-per-method summary ───────────────────────────────────────────────────

print('\n' + '=' * 80)
print('  BEST PER METHOD (by expanding-window RMSE)')
print('=' * 80)
for method in ['LLM', 'Gardner', 'Sharpe']:
    sub = out_df[out_df['model'].str.startswith(method) &
                 ~out_df['model'].str.contains('base')].copy()
    if sub.empty or sub['rmse_exp'].isna().all():
        continue
    best = sub.loc[sub['rmse_exp'].idxmin()]
    print(f'  {best["model"]:<30}  RMSE_exp={best["rmse_exp"]:.4f}  '
          f'DM-t={best["dm_t_exp"]:.3f}  p={best["dm_p_exp"]:.4f}  {best["sig_exp"]}')
