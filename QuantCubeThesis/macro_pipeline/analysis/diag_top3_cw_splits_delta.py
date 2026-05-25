"""
diag_top3_cw_splits_delta.py

Same analysis as diag_top3_cw_splits.py but for the DELTA target
(delta_rate_next = change in effective rate at next FOMC meeting).

9 models (selected by broad scan + CW check on current 2007-2025 data,
N=151, all on the same full sample for fair comparison):

  LLM (joint PCA, all-doc zscore via step0_merged_to_macro.csv):
    1-3. Best 3 of 4h variants on delta_rate_next (run fresh)

  Gardner (verified by diag_delta_broad_scan.py + CW check):
    4. speeches_zscore    4f_matched_inter      OOS80=0.240 CW**  at OOS60/70/80
    5. minutes_zscore_ewma10  4c_all_topics      OOS80=0.223 CW*   at OOS80  CW** at OOS70
    6. minutes_zscore     4f_matched_inter       OOS80=0.255 CW**  at OOS80

  Sharpe (verified by broad scan; all specs weak for delta):
    7. all_docs_wc_ewma10   4b_consensus         OOS80=0.242
    8. statements_wc_ewma10 4b_consensus         OOS80=0.243
    9. minutes_zscore       4e_total_inter       OOS80=0.253

Note: press_conf specs (N=118) have low absolute RMSE (0.157) but their
sub-sample macro baseline is ALSO 0.157 — sentiment adds nothing on the
2011-2025 sub-sample. All models here use the full N=151 (2007-2025) sample.

CW test at each cutoff = CW restricted to predictions from t >= cutoff*n.
Baseline = macro-only expanding FAVAR on delta_rate_next, same sub-sample.
"""
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from config import TARGET_NEXT, MACRO_REGRESSORS, PCA_VAR_MACRO
from utils import run_pca, clark_west_test
from step0_aggregate import load_macro

root      = r'C:\Users\Javier\OneDrive - HEC Paris\Documentos\QuantCubeThesis\QuantCubeThesis'
INTER_CSV = root + r'\Taylor Rule\outputs\intermediate\step0_merged_to_macro.csv'
NLP_DIR   = root + r'\Taylor Rule\nlp_output'

DELTA_COL  = 'delta_rate_next'
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

MATCHED_TOPIC = {
    'sent_inflation':            'inflation_dev_from_target',
    'sent_labor_market':         'unemployment_gap',
    'sent_economic_activity':    'gdp',
    'sent_financial_conditions': 'vix',
    'sent_monetary_policy':      'implied_ffr',
}

TOPIC_COLS = ['sent_inflation', 'sent_labor_market', 'sent_economic_activity',
              'sent_financial_conditions', 'sent_monetary_policy', 'sent_macro']


# ── Core functions ─────────────────────────────────────────────────────────────

def expanding_ols(y, X, train_frac=TRAIN_FRAC):
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


def compute_metrics(model_preds, base_preds, n):
    out = {}
    for frac in CUTOFFS:
        ct  = int(n * frac)
        key = int(frac * 100)
        m_sub = [(t, y, yh) for t, y, yh in model_preds if t >= ct]
        b_sub = [(t, y, yh) for t, y, yh in base_preds  if t >= ct]
        if m_sub:
            out[f'rmse_{key}'] = round(
                np.sqrt(np.mean([(y - yh)**2 for _, y, yh in m_sub])), 4)
            cw_t, cw_p = clark_west_test(b_sub, m_sub)
            out[f'cw_t_{key}'] = cw_t
            out[f'cw_p_{key}'] = cw_p
        else:
            out[f'rmse_{key}'] = np.nan
            out[f'cw_t_{key}'] = np.nan
            out[f'cw_p_{key}'] = np.nan
    return out


# ── Panel builders ─────────────────────────────────────────────────────────────

def load_nlp_panel(nlp_fname, remap, ewma_span=None):
    raw = pd.read_csv(NLP_DIR + '\\' + nlp_fname, parse_dates=['meeting_date'])
    raw = raw.rename(columns={k: v for k, v in remap.items() if k in raw.columns})
    sent_cols = [c for c in raw.columns if c.startswith('sent_')]
    agg = raw.groupby('meeting_date')[sent_cols].sum().reset_index()
    agg = agg.rename(columns={'meeting_date': 'date'})
    macro = load_macro()
    df = pd.merge_asof(macro.sort_values('date'), agg.sort_values('date'),
                       on='date', direction='backward')
    df = df[df['date'] >= '2007-01-01'].reset_index(drop=True)
    if ewma_span:
        for c in df.columns:
            if c.startswith('sent_'):
                df[c] = df[c].ewm(span=ewma_span, adjust=False).mean()
    return df


# ── Model runners ──────────────────────────────────────────────────────────────

def run_4h_delta(df, pca_input_cols):
    """Joint PCA on pca_input_cols, expanding OLS on delta_rate_next."""
    mc_avail = [c for c in MACRO_REGRESSORS if c in df.columns]
    avail    = [c for c in pca_input_cols   if c in df.columns]
    all_need = list(dict.fromkeys(avail + mc_avail + [DELTA_COL]))
    sub = df[all_need].dropna().reset_index(drop=True)
    n   = len(sub)

    avail_s = [c for c in avail if c in sub.columns]
    fac_m, _, _, nc_m = run_pca(sub, avail_s, var_threshold=PCA_VAR_MACRO)
    fac_m.columns = [f'JPC{i+1}' for i in range(nc_m)]

    mc_s = [c for c in mc_avail if c in sub.columns]
    fac_b, _, _, nc_b = run_pca(sub, mc_s, var_threshold=PCA_VAR_MACRO)
    fac_b.columns = [f'BPC{i+1}' for i in range(nc_b)]

    y       = sub[DELTA_COL].values
    X_model = sm.add_constant(fac_m.values)
    X_base  = sm.add_constant(fac_b.values)

    return expanding_ols(y, X_model), expanding_ols(y, X_base), n


def run_standard_delta(df, extra_cols):
    """Full-sample macro PCA + extra_cols, expanding OLS on delta_rate_next.

    Macro PCA is fit on ALL rows with valid macro (not sentiment-filtered subset)
    so the full 5 components are available.
    """
    mc_avail = [c for c in MACRO_REGRESSORS if c in df.columns]
    ex_avail = [c for c in extra_cols       if c in df.columns]

    fac, _, _, nc = run_pca(df.dropna(subset=mc_avail), mc_avail,
                            var_threshold=PCA_VAR_MACRO)
    fac.columns = [f'PC{i+1}' for i in range(nc)]

    X_aug = fac.join(df[ex_avail], how='left').dropna()
    y_srs = df.loc[X_aug.index, DELTA_COL].dropna()
    X_aug = X_aug.loc[y_srs.index]
    n = len(X_aug)

    y      = y_srs.values
    pc_cols = [c for c in X_aug.columns if c.startswith('PC')]
    X_model = sm.add_constant(X_aug.values)
    X_base  = sm.add_constant(X_aug[pc_cols].values)

    return expanding_ols(y, X_model), expanding_ols(y, X_base), n


# ── Interaction builders ───────────────────────────────────────────────────────

def add_4e_interactions(df, sent_col='sent_total'):
    df = df.copy()
    cols = []
    for mc in MACRO_INTER_COLS:
        if mc in df.columns and sent_col in df.columns:
            iname = f'inter_{sent_col}_x_{mc}'
            df[iname] = df[sent_col] * df[mc]
            cols.append(iname)
    return df, cols


def add_matched_interactions(df, matched_dict):
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
print('  TOP-3 MODELS PER METHOD: OOS RMSE AND CW-t AT EACH CUTOFF  [TARGET: delta_rate_next]')
print('  Baseline = macro-only expanding FAVAR on delta_rate_next, same sub-sample')
print('  All models: N=151 (2007-2025) for fair comparison')
print('=' * 100)


# ──────────────────────────────────────────────────────────────────────────────
# LLM 4h variants on delta (run fresh — no stored results)
# ──────────────────────────────────────────────────────────────────────────────

print('\n  [LLM] Loading panel and running 4h variants on delta...', flush=True)
df_llm = pd.read_csv(INTER_CSV, parse_dates=['date'])
df_llm = df_llm[df_llm['date'] >= '2007-01-01'].reset_index(drop=True)

mc_llm   = [c for c in MACRO_REGRESSORS if c in df_llm.columns]
sent_tot = ['sent_total']
sent_top = [c for c in TOPIC_COLS if c in df_llm.columns]

# Build total_inter cols for LLM
df_llm_inter = df_llm.copy()
inter_llm = []
for mc in MACRO_INTER_COLS:
    if mc in df_llm_inter.columns and 'sent_total' in df_llm_inter.columns:
        iname = f'inter_sent_total_x_{mc}'
        df_llm_inter[iname] = df_llm_inter['sent_total'] * df_llm_inter[mc]
        inter_llm.append(iname)

llm_variants = [
    ('4h_llm_total',      df_llm,       mc_llm + sent_tot),
    ('4h_llm_topics',     df_llm,       mc_llm + sent_top),
    ('4h_llm_all_sent',   df_llm,       mc_llm + sent_tot + sent_top),
    ('4h_llm_total_inter',df_llm_inter, mc_llm + sent_tot + inter_llm),
]

llm_results = []
for label, df_v, pca_cols in llm_variants:
    print(f'  [{label}]', end='', flush=True)
    m_preds, b_preds, n = run_4h_delta(df_v, pca_cols)
    metrics = compute_metrics(m_preds, b_preds, n)
    metrics.update({'model': label, 'method': 'LLM', 'n': n})
    llm_results.append(metrics)
    print(f'  N={n}  OOS80={metrics["rmse_80"]:.4f}', end='')

# Pick best 3 by OOS80
llm_results.sort(key=lambda r: r['rmse_80'] if not np.isnan(r['rmse_80']) else 999)
all_rows.extend(llm_results[:3])
print(f'\n  -> LLM best 3: {[r["model"] for r in llm_results[:3]]}')


# ──────────────────────────────────────────────────────────────────────────────
# Gardner (verified best specs for delta on current N=151 data)
# ──────────────────────────────────────────────────────────────────────────────

print('\n\n  [Gardner] Loading panels...', flush=True)

# G4: speeches_zscore 4f_matched — strong CW at OOS60/70/80
print('  [G4: speeches_zs_4f_matched]', end='', flush=True)
df_g4 = load_nlp_panel('Gardner_speeches_zscore_nlp.csv', GARDNER_REMAP)
df_g4, inter_g4 = add_matched_interactions(df_g4, MATCHED_TOPIC)
m_preds, b_preds, n = run_standard_delta(df_g4, inter_g4)
metrics = compute_metrics(m_preds, b_preds, n)
metrics.update({'model': 'G4_speeches_zs_4f_matched', 'method': 'Gardner', 'n': n})
all_rows.append(metrics)
print(f'  N={n}  OOS80={metrics["rmse_80"]:.4f}', end='')

# G5: minutes_zscore EWMA=10  4c_all_topics — best OOS80 for Gardner overall
print('\n  [G5: minutes_zs_ewma10_4c_topics]', end='', flush=True)
df_g5 = load_nlp_panel('Gardner_minutes_zscore_nlp.csv', GARDNER_REMAP, ewma_span=10)
top_avail_g5 = [c for c in TOPIC_COLS if c in df_g5.columns and c != 'sent_macro']
m_preds, b_preds, n = run_standard_delta(df_g5, top_avail_g5)
metrics = compute_metrics(m_preds, b_preds, n)
metrics.update({'model': 'G5_minutes_zs_ewma10_4c_topics', 'method': 'Gardner', 'n': n})
all_rows.append(metrics)
print(f'  N={n}  OOS80={metrics["rmse_80"]:.4f}', end='')

# G6: minutes_zscore (no EWMA)  4f_matched — stable, verified CW** at OOS80
print('\n  [G6: minutes_zs_4f_matched]', end='', flush=True)
df_g6 = load_nlp_panel('Gardner_minutes_zscore_nlp.csv', GARDNER_REMAP)
df_g6, inter_g6 = add_matched_interactions(df_g6, MATCHED_TOPIC)
m_preds, b_preds, n = run_standard_delta(df_g6, inter_g6)
metrics = compute_metrics(m_preds, b_preds, n)
metrics.update({'model': 'G6_minutes_zs_4f_matched', 'method': 'Gardner', 'n': n})
all_rows.append(metrics)
print(f'  N={n}  OOS80={metrics["rmse_80"]:.4f}', end='')


# ──────────────────────────────────────────────────────────────────────────────
# Sharpe (all specs weak for delta; showing best 3 by OOS80 on N=151)
# ──────────────────────────────────────────────────────────────────────────────

print('\n\n  [Sharpe] Loading panels...', flush=True)

# S7: all-docs wordcount EWMA=10  4b_consensus — OOS80=0.242
print('  [S7: all_docs_wc_ewma10_4b_cons]', end='', flush=True)
df_s7 = load_nlp_panel(
    'Sharpe_statements_minutes_speeches_press_conferences_wordcount_nlp.csv',
    SHARPE_REMAP, ewma_span=10)
cons_s7 = [c for c in df_s7.columns if c.endswith('_consensus')]
m_preds, b_preds, n = run_standard_delta(df_s7, cons_s7)
metrics = compute_metrics(m_preds, b_preds, n)
metrics.update({'model': 'S7_all_docs_wc_ewma10_4b_cons', 'method': 'Sharpe', 'n': n})
all_rows.append(metrics)
print(f'  N={n}  OOS80={metrics["rmse_80"]:.4f}', end='')

# S8: statements wordcount EWMA=10  4b_consensus — OOS80=0.243
print('\n  [S8: statements_wc_ewma10_4b_cons]', end='', flush=True)
df_s8 = load_nlp_panel('Sharpe_statements_wordcount_nlp.csv', SHARPE_REMAP, ewma_span=10)
cons_s8 = [c for c in df_s8.columns if c.endswith('_consensus')]
m_preds, b_preds, n = run_standard_delta(df_s8, cons_s8)
metrics = compute_metrics(m_preds, b_preds, n)
metrics.update({'model': 'S8_statements_wc_ewma10_4b_cons', 'method': 'Sharpe', 'n': n})
all_rows.append(metrics)
print(f'  N={n}  OOS80={metrics["rmse_80"]:.4f}', end='')

# S9: minutes zscore  4e_total_inter — OOS80=0.253
print('\n  [S9: minutes_zs_4e_inter]', end='', flush=True)
df_s9 = load_nlp_panel('Sharpe_minutes_zscore_nlp.csv', SHARPE_REMAP)
df_s9, inter_s9 = add_4e_interactions(df_s9, 'sent_total')
m_preds, b_preds, n = run_standard_delta(df_s9, inter_s9)
metrics = compute_metrics(m_preds, b_preds, n)
metrics.update({'model': 'S9_minutes_zs_4e_inter', 'method': 'Sharpe', 'n': n})
all_rows.append(metrics)
print(f'  N={n}  OOS80={metrics["rmse_80"]:.4f}', end='')


# ── Print results ──────────────────────────────────────────────────────────────

print('\n\n')
print('=' * 100)
print('  RESULTS [DELTA]: OOS RMSE and CW-t at each cutoff (vs macro-only baseline, same sample)')
print('  Macro baseline delta: OOS60=0.314  OOS70=0.313  OOS80=0.265  OOS90=0.168  (N=151, 2007-2025)')
print('  CW-t > 1.65 => model beats baseline at ~5% one-sided')
print('  p-value: * p<0.10  ** p<0.05  *** p<0.01  (NaN p = model has higher MSE than baseline)')
print('=' * 100)


def sig_star(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return '   '
    if p < 0.01:  return '***'
    if p < 0.05:  return '** '
    if p < 0.10:  return '*  '
    return '   '


hdr = (f'  {"Model":<50} {"N":>5}'
       + ''.join(f'  {"OOS"+str(int(f*100)):>5} {"CW-t":>6}{"p":>4}'
                 for f in CUTOFFS))
print(hdr)
print('  ' + '-' * 100)

for row in all_rows:
    name = row['model']
    n    = row['n']
    line = f'  {name:<50} {n:>5}'
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
print('  Note: NaN CW p-value = model has higher MSE than baseline OR HAC test fails')
print('  Note: Gardner speeches covers full 2007-2025 sample (same as LLM)')
print('  Note: Sharpe specs weakly significant — sentiment less useful for delta than level')

# ── Save ───────────────────────────────────────────────────────────────────────
out = root + r'\Taylor Rule\outputs\top3_cw_splits_delta.csv'
pd.DataFrame(all_rows).to_csv(out, index=False)
print(f'\n  Saved -> {out}')
