"""
diag_top3_cw_splits.py

Show the 3 best models per method with OOS RMSE and CW-t at each cutoff
(60 / 70 / 80 / 90 percent).

9 models:
  LLM (joint PCA, all-doc zscore via step0_merged_to_macro.csv):
    1. 4h_llm_total       OOS80=0.255  macro + sent_total
    2. 4h_llm_topics      OOS80=0.266  macro + all topic scores
    3. 4h_llm_all_sent    OOS80=0.267  macro + total + all topics

  Gardner (standard FAVAR + EWMA-10, press_conf only):
    4. press_conf_wc   4b_consensus                     OOS80=0.230
    5. press_conf_zs   4f_matched_interactions          OOS80=0.245
    6. press_conf_zs   4b_consensus_matched_interactions OOS80=0.254

  Sharpe (standard FAVAR):
    7. statements_zs   4e_total_interactions            OOS80=0.296
    8. speeches_wc  EWMA-10  4e_total_interactions      OOS80=0.323
    9. 4h_sharpe_total_inter (joint PCA, all-doc zscore) OOS80=0.372

CW test at each cutoff = CW restricted to predictions from t >= cutoff*n.
Baseline is always macro-only expanding FAVAR on the same sub-sample.
"""
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from config import TARGET_NEXT, MACRO_REGRESSORS, PCA_VAR_MACRO
from utils import run_pca, clark_west_test, diebold_mariano_test
from step0_aggregate import load_macro

root     = r'C:\Users\Javier\OneDrive - HEC Paris\Documentos\QuantCubeThesis\QuantCubeThesis'
INTER_CSV = root + r'\Taylor Rule\outputs\intermediate\step0_merged_to_macro.csv'
NLP_DIR   = root + r'\Taylor Rule\nlp_output'

TRAIN_FRAC = 0.60
CUTOFFS    = [0.60, 0.70, 0.80, 0.90]

MACRO_INTER_COLS = [
    'vix', 'gdp', 'unemployment_gap',
    'inflation_dev_from_target', 'implied_ffr', 'cfnai',
]

GARDNER_REMAP = {
    'gardner_total': 'sent_total',
    'gardner_inf':   'sent_inflation',
    'gardner_labor': 'sent_labor_market',
    'gardner_out':   'sent_economic_activity',
    'gardner_fin':   'sent_financial_conditions',
    'gardner_mp':    'sent_monetary_policy',
    'gardner_inf_consensus':   'sent_inflation_consensus',
    'gardner_labor_consensus': 'sent_labor_market_consensus',
    'gardner_out_consensus':   'sent_economic_activity_consensus',
    'gardner_fin_consensus':   'sent_financial_conditions_consensus',
    'gardner_mp_consensus':    'sent_monetary_policy_consensus',
}
SHARPE_REMAP = {
    'sharpe_net':       'sent_total',
    'sharpe_consensus': 'sent_consensus',
}
SHARPE_4H_REMAP = {
    'sharpe_net':       's_net',
    'sharpe_positive':  's_positive',
    'sharpe_negative':  's_negative',
    'sharpe_consensus': 's_consensus',
}

# Matched pairs for 4f and 4b_consensus_matched
MATCHED_TOPIC = {
    'sent_inflation':            'inflation_dev_from_target',
    'sent_labor_market':         'unemployment_gap',
    'sent_economic_activity':    'gdp',
    'sent_financial_conditions': 'vix',
    'sent_monetary_policy':      'implied_ffr',
}
MATCHED_CONSENSUS = {
    'sent_inflation_consensus':            'inflation_dev_from_target',
    'sent_labor_market_consensus':         'unemployment_gap',
    'sent_economic_activity_consensus':    'gdp',
    'sent_financial_conditions_consensus': 'vix',
    'sent_monetary_policy_consensus':      'implied_ffr',
}


# ── Core functions ─────────────────────────────────────────────────────────────

def expanding_ols(y, X, train_frac=TRAIN_FRAC):
    """Fixed-design expanding-window OLS from train_frac.
    Returns list of (t, y_actual, y_hat)."""
    n    = len(y)
    init = int(n * train_frac)
    preds = []
    for t in range(init, n):
        try:
            m     = sm.OLS(y[:t], X[:t]).fit()
            y_hat = float(m.predict(X[t:t+1])[0])
            preds.append((t, float(y[t]), y_hat))
        except Exception:
            pass
    return preds


def compute_metrics(model_preds, base_preds, n, test='cw'):
    """RMSE and test-t at each cutoff.

    test='cw'  Clark-West (2007)  — for nested models (standard FAVAR specs)
    test='dm'  Diebold-Mariano (1995) — for non-nested models (joint PCA specs)
    """
    stat_fn = clark_west_test if test == 'cw' else diebold_mariano_test
    out = {'_test': test}
    for frac in CUTOFFS:
        ct  = int(n * frac)
        key = int(frac * 100)
        m_sub = [(t, y, yh) for t, y, yh in model_preds if t >= ct]
        b_sub = [(t, y, yh) for t, y, yh in base_preds  if t >= ct]
        if m_sub:
            out[f'rmse_{key}'] = round(
                np.sqrt(np.mean([(y - yh)**2 for _, y, yh in m_sub])), 4)
            t_stat, t_p = stat_fn(b_sub, m_sub)
            out[f'cw_t_{key}'] = t_stat
            out[f'cw_p_{key}'] = t_p
        else:
            out[f'rmse_{key}'] = np.nan
            out[f'cw_t_{key}'] = np.nan
            out[f'cw_p_{key}'] = np.nan
    return out


# ── Panel builders ─────────────────────────────────────────────────────────────

def load_nlp_panel(nlp_fname, remap, ewma_span=None):
    """Load NLP CSV, remap cols, aggregate by meeting, merge with macro,
    filter 2007+, optionally apply EWMA."""
    raw = pd.read_csv(NLP_DIR + '\\' + nlp_fname, parse_dates=['meeting_date'])
    raw = raw.rename(columns={k: v for k, v in remap.items() if k in raw.columns})
    sent_cols = [c for c in raw.columns if c.startswith('sent_') or c.startswith('s_')]
    agg = raw.groupby('meeting_date')[sent_cols].sum().reset_index()
    agg = agg.rename(columns={'meeting_date': 'date'})

    macro = load_macro()
    df = pd.merge_asof(macro.sort_values('date'), agg.sort_values('date'),
                       on='date', direction='backward')
    df = df[df['date'] >= '2007-01-01'].reset_index(drop=True)

    if ewma_span:
        for c in df.columns:
            if c.startswith('sent_') or c.startswith('s_'):
                df[c] = df[c].ewm(span=ewma_span, adjust=False).mean()
    return df


# ── Model runners ──────────────────────────────────────────────────────────────

def run_4h_model(df, pca_input_cols):
    """Joint PCA on pca_input_cols + expanding OLS.
    Returns (model_preds, base_preds, n)."""
    mc_avail = [c for c in MACRO_REGRESSORS if c in df.columns]
    avail    = [c for c in pca_input_cols   if c in df.columns]
    all_need = list(dict.fromkeys(avail + mc_avail + [TARGET_NEXT]))
    sub = df[all_need].dropna().reset_index(drop=True)
    n   = len(sub)

    # Joint PCA (model)
    avail_s = [c for c in avail if c in sub.columns]
    fac_m, _, _, nc_m = run_pca(sub, avail_s, var_threshold=PCA_VAR_MACRO)
    fac_m.columns = [f'JPC{i+1}' for i in range(nc_m)]

    # Macro-only PCA (baseline on same sub)
    mc_s = [c for c in mc_avail if c in sub.columns]
    fac_b, _, _, nc_b = run_pca(sub, mc_s, var_threshold=PCA_VAR_MACRO)
    fac_b.columns = [f'BPC{i+1}' for i in range(nc_b)]

    y       = sub[TARGET_NEXT].values
    X_model = sm.add_constant(fac_m.values)
    X_base  = sm.add_constant(fac_b.values)

    return expanding_ols(y, X_model), expanding_ols(y, X_base), n


def run_standard_model(df, extra_cols):
    """Macro-only PCA factors (full-sample) + extra_cols + expanding OLS.
    PCA is fit on ALL rows with valid macro data, then augmented with
    extra_cols (dropping rows with missing sentiment). This matches the
    pipeline's approach where baseline factors use the full macro sample.
    Returns (model_preds, base_preds, n)."""
    mc_avail = [c for c in MACRO_REGRESSORS if c in df.columns]
    ex_avail = [c for c in extra_cols       if c in df.columns]

    # Full-sample macro PCA (not restricted to rows with sentiment)
    fac, _, _, nc = run_pca(df.dropna(subset=mc_avail), mc_avail,
                            var_threshold=PCA_VAR_MACRO)
    fac.columns = [f'PC{i+1}' for i in range(nc)]

    # Augment: keeps only rows where both macro factors AND extra_cols are valid
    X_aug = fac.join(df[ex_avail], how='left').dropna()
    y_srs = df.loc[X_aug.index, TARGET_NEXT].dropna()
    X_aug = X_aug.loc[y_srs.index]
    n = len(X_aug)

    y      = y_srs.values
    pc_cols = [c for c in X_aug.columns if c.startswith('PC')]
    X_model = sm.add_constant(X_aug.values)
    X_base  = sm.add_constant(X_aug[pc_cols].values)

    return expanding_ols(y, X_model), expanding_ols(y, X_base), n


# ── Interaction builders ───────────────────────────────────────────────────────

def add_4e_interactions(df, sent_col='sent_total'):
    """Add sent_col x each macro interaction. Returns updated df and col list."""
    df = df.copy()
    cols = []
    for mc in MACRO_INTER_COLS:
        if mc in df.columns and sent_col in df.columns:
            iname = f'inter_{sent_col}_x_{mc}'
            df[iname] = df[sent_col] * df[mc]
            cols.append(iname)
    return df, cols


def add_matched_interactions(df, matched_dict):
    """Add col_a x col_b interactions from a {sent_col: macro_col} dict.
    Returns updated df and col list."""
    df = df.copy()
    cols = []
    for sent_col, mc in matched_dict.items():
        if sent_col in df.columns and mc in df.columns:
            iname = f'inter_{sent_col}_x_{mc}'
            df[iname] = df[sent_col] * df[mc]
            cols.append(iname)
    return df, cols


# ── Main ──────────────────────────────────────────────────────────────────────

all_rows = []

print('=' * 100)
print('  TOP-3 MODELS PER METHOD: OOS RMSE AND CW-t AT EACH CUTOFF')
print('  Baseline = macro-only expanding FAVAR on same sub-sample')
print('=' * 100)


# ──────────────────────────────────────────────────────────────────────────────
# LLM (joint PCA, all-doc zscore)
# ──────────────────────────────────────────────────────────────────────────────

print('\n  [LLM] Loading panel...', end='', flush=True)
df_llm = pd.read_csv(INTER_CSV, parse_dates=['date'])
df_llm = df_llm[df_llm['date'] >= '2007-01-01'].reset_index(drop=True)

mc_llm    = [c for c in MACRO_REGRESSORS if c in df_llm.columns]
sent_tot  = ['sent_total']
# Aggregate topic scores only (no per-doc-type breakdowns, no _sd/_var)
TOPIC_COLS = ['sent_inflation', 'sent_labor_market', 'sent_economic_activity',
              'sent_financial_conditions', 'sent_monetary_policy', 'sent_macro']
sent_top  = [c for c in TOPIC_COLS if c in df_llm.columns]

llm_variants = [
    ('4h_llm_total',    mc_llm + sent_tot),
    ('4h_llm_topics',   mc_llm + sent_top),
    ('4h_llm_all_sent', mc_llm + sent_tot + sent_top),
]

for label, pca_cols in llm_variants:
    print(f'\n  [{label}]', end='', flush=True)
    m_preds, b_preds, n = run_4h_model(df_llm, pca_cols)
    metrics = compute_metrics(m_preds, b_preds, n, test='dm')   # non-nested: use DM
    metrics.update({'model': label, 'method': 'LLM', 'n': n})
    all_rows.append(metrics)
    print(f'  N={n}  OOS80={metrics["rmse_80"]:.4f}', end='')

# ──────────────────────────────────────────────────────────────────────────────
# Gardner (standard FAVAR + EWMA-10, press_conf only)
# ──────────────────────────────────────────────────────────────────────────────

print('\n\n  [Gardner] Loading press_conf panels...')

# Model 4: press_conf_wordcount EWMA-10, 4b_consensus
print('  [G4: press_conf_wc_EWMA10_4b_consensus]', end='', flush=True)
df_g_wc = load_nlp_panel('Gardner_press_conferences_wordcount_nlp.csv',
                          GARDNER_REMAP, ewma_span=10)
consensus_cols_wc = [c for c in df_g_wc.columns if c.endswith('_consensus')]
m_preds, b_preds, n = run_standard_model(df_g_wc, consensus_cols_wc)
metrics = compute_metrics(m_preds, b_preds, n)
metrics.update({'model': 'G4_press_conf_wc_EWMA10_4b_consensus', 'method': 'Gardner', 'n': n})
all_rows.append(metrics)
print(f'  N={n}  OOS80={metrics["rmse_80"]:.4f}', end='')

# Model 5: press_conf_zscore EWMA-10, 4f_matched_interactions
print('\n  [G5: press_conf_zs_EWMA10_4f_matched]', end='', flush=True)
df_g_zs = load_nlp_panel('Gardner_press_conferences_zscore_nlp.csv',
                          GARDNER_REMAP, ewma_span=10)
df_g_zs_4f, inter_4f = add_matched_interactions(df_g_zs, MATCHED_TOPIC)
m_preds, b_preds, n = run_standard_model(df_g_zs_4f, inter_4f)
metrics = compute_metrics(m_preds, b_preds, n)
metrics.update({'model': 'G5_press_conf_zs_EWMA10_4f_matched', 'method': 'Gardner', 'n': n})
all_rows.append(metrics)
print(f'  N={n}  OOS80={metrics["rmse_80"]:.4f}', end='')

# Model 6: press_conf_zscore EWMA-10, 4b_consensus_matched_interactions
print('\n  [G6: press_conf_zs_EWMA10_4b_consensus_matched]', end='', flush=True)
df_g_zs_con, inter_con = add_matched_interactions(df_g_zs, MATCHED_CONSENSUS)
m_preds, b_preds, n = run_standard_model(df_g_zs_con, inter_con)
metrics = compute_metrics(m_preds, b_preds, n)
metrics.update({'model': 'G6_press_conf_zs_EWMA10_4b_consensus_matched', 'method': 'Gardner', 'n': n})
all_rows.append(metrics)
print(f'  N={n}  OOS80={metrics["rmse_80"]:.4f}', end='')

# ──────────────────────────────────────────────────────────────────────────────
# Sharpe
# ──────────────────────────────────────────────────────────────────────────────

print('\n\n  [Sharpe] Loading panels...')

# Model 7: statements_zscore 4e_total_interactions (no EWMA)
print('  [S7: statements_zs_4e]', end='', flush=True)
df_s_stmt = load_nlp_panel('Sharpe_statements_zscore_nlp.csv', SHARPE_REMAP)
df_s_stmt_4e, inter_s_4e = add_4e_interactions(df_s_stmt, sent_col='sent_total')
m_preds, b_preds, n = run_standard_model(df_s_stmt_4e, inter_s_4e)
metrics = compute_metrics(m_preds, b_preds, n)
metrics.update({'model': 'S7_statements_zs_4e', 'method': 'Sharpe', 'n': n})
all_rows.append(metrics)
print(f'  N={n}  OOS80={metrics["rmse_80"]:.4f}', end='')

# Model 8: speeches_wordcount EWMA-10, 4e_total_interactions
print('\n  [S8: speeches_wc_EWMA10_4e]', end='', flush=True)
df_s_sp = load_nlp_panel('Sharpe_speeches_wordcount_nlp.csv', SHARPE_REMAP, ewma_span=10)
df_s_sp_4e, inter_sp_4e = add_4e_interactions(df_s_sp, sent_col='sent_total')
m_preds, b_preds, n = run_standard_model(df_s_sp_4e, inter_sp_4e)
metrics = compute_metrics(m_preds, b_preds, n)
metrics.update({'model': 'S8_speeches_wc_EWMA10_4e', 'method': 'Sharpe', 'n': n})
all_rows.append(metrics)
print(f'  N={n}  OOS80={metrics["rmse_80"]:.4f}', end='')

# Model 9: 4h_sharpe_total_inter (joint PCA, all-docs zscore)
# Variant: macro + s_net + inter_s_net_x_each_macro
print('\n  [S9: 4h_sharpe_total_inter]', end='', flush=True)
df_s_all = load_nlp_panel(
    'Sharpe_statements_minutes_speeches_press_conferences_zscore_nlp.csv',
    SHARPE_4H_REMAP)
df_s_all, inter_s_all = add_4e_interactions(df_s_all, sent_col='s_net')
mc_avail_s = [c for c in MACRO_REGRESSORS if c in df_s_all.columns]
pca_cols_s9 = mc_avail_s + ['s_net'] + [c for c in inter_s_all if c in df_s_all.columns]
m_preds, b_preds, n = run_4h_model(df_s_all, pca_cols_s9)
metrics = compute_metrics(m_preds, b_preds, n, test='dm')   # non-nested: use DM
metrics.update({'model': 'S9_4h_sharpe_total_inter', 'method': 'Sharpe', 'n': n})
all_rows.append(metrics)
print(f'  N={n}  OOS80={metrics["rmse_80"]:.4f}', end='')


# ── Print results ──────────────────────────────────────────────────────────────

print('\n\n')
print('=' * 100)
print('  RESULTS [LEVEL]: OOS RMSE and t-stat at each cutoff (vs macro-only baseline, same sample)')
print('  t-stat: DM = Diebold-Mariano (1995) for joint PCA models (LLM, S9) [non-nested]')
print('          CW = Clark-West (2007) for standard FAVAR models (Gardner, Sharpe S7-S8) [nested]')
print('  p-value: * p<0.10  ** p<0.05  *** p<0.01  (blank = model has higher MSE than baseline)')
print('=' * 100)

def sig_star(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ''
    if p < 0.01:  return '***'
    if p < 0.05:  return '** '
    if p < 0.10:  return '*  '
    return '   '

hdr = (f'  {"Model":<45} {"N":>5} {"test":>4}'
       + ''.join(f'  {"OOS"+str(int(f*100)):>5} {"t-stat":>6}{"p":>5}'
                 for f in CUTOFFS))
print(hdr)
print('  ' + '-' * 100)

for row in all_rows:
    name  = row['model']
    n     = row['n']
    test  = row.get('_test', 'cw').upper()
    line  = f'  {name:<45} {n:>5} {test:>4}'
    for frac in CUTOFFS:
        key   = int(frac * 100)
        rmse  = row.get(f'rmse_{key}', np.nan)
        cw_t  = row.get(f'cw_t_{key}', np.nan)
        cw_p  = row.get(f'cw_p_{key}', np.nan)
        rmse_s = f'{rmse:>5.3f}' if not (isinstance(rmse, float) and np.isnan(rmse)) else '  n/a'
        cwt_s  = f'{cw_t:>6.2f}' if not (isinstance(cw_t, float) and np.isnan(cw_t)) else '   n/a'
        star   = sig_star(cw_p)
        line  += f'  {rmse_s} {cwt_s}{star}'
    print(line)

print()
print('  Blank p-value: model has higher MSE than baseline over that sub-period')

# ── Save CSV ───────────────────────────────────────────────────────────────────
out = root + r'\Taylor Rule\outputs\top3_cw_splits.csv'
pd.DataFrame(all_rows).to_csv(out, index=False)
print(f'\n  Saved -> {out}')

# ── Clean up temp files ────────────────────────────────────────────────────────
import os
for tmp in ['tmp_src.txt', 'cw_src.txt', 'run_pca_src.txt']:
    try:
        os.remove(tmp)
    except FileNotFoundError:
        pass
