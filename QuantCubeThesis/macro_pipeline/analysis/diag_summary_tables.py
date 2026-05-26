"""
diag_summary_tables.py

Print two summary tables (Level and Delta):
  - Best 3 models per method (LLM, Gardner, Sharpe)
  - Columns: Adj-R² | OOS60 | OOS70 | OOS80 | OOS90
  - OOS RMSE with significance stars from CW (nested) or DM (non-nested) test
  - Adj-R²: full-sample OLS on all N observations (in-sample fit quality)

Run from macro_pipeline/analysis/:
    python diag_summary_tables.py
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

root      = r'C:\Users\Javier\OneDrive - HEC Paris\Documentos\QuantCubeThesis\QuantCubeThesis'
INTER_CSV = root + r'\Taylor Rule\outputs\intermediate\step0_merged_to_macro.csv'
NLP_DIR   = root + r'\Taylor Rule\nlp_output'
DELTA_COL = 'delta_rate_next'

TRAIN_FRAC = 0.60
CUTOFFS    = [0.60, 0.70, 0.80, 0.90]

MACRO_INTER_COLS = [
    'vix', 'gdp', 'unemployment_gap',
    'inflation_dev_from_target', 'implied_ffr', 'cfnai',
]
TOPIC_COLS = [
    'sent_inflation', 'sent_labor_market', 'sent_economic_activity',
    'sent_financial_conditions', 'sent_monetary_policy', 'sent_macro',
]
GARDNER_REMAP = {
    'gardner_total': 'sent_total',
    'gardner_inf':   'sent_inflation',
    'gardner_labor': 'sent_labor_market',
    'gardner_out':   'sent_economic_activity',
    'gardner_fin':   'sent_financial_conditions',
    'gardner_mp':    'sent_monetary_policy',
    'gardner_inf_consensus':            'sent_inflation_consensus',
    'gardner_labor_consensus':          'sent_labor_market_consensus',
    'gardner_out_consensus':            'sent_economic_activity_consensus',
    'gardner_fin_consensus':            'sent_financial_conditions_consensus',
    'gardner_mp_consensus':             'sent_monetary_policy_consensus',
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def expanding_ols(y, X, train_frac=TRAIN_FRAC):
    n = len(y); init = int(n * train_frac)
    preds = []
    for t in range(init, n):
        try:
            m = sm.OLS(y[:t], X[:t]).fit()
            preds.append((t, float(y[t]), float(m.predict(X[t:t+1])[0])))
        except Exception:
            pass
    return preds


def oos_rmse(preds, n, frac):
    ct  = int(n * frac)
    sub = [(t, y, yh) for t, y, yh in preds if t >= ct]
    return round(np.sqrt(np.mean([(y - yh)**2 for _, y, yh in sub])), 4) if sub else np.nan


def compute_metrics(model_preds, base_preds, n, test='cw'):
    stat_fn = clark_west_test if test == 'cw' else diebold_mariano_test
    out = {'_test': test}
    for frac in CUTOFFS:
        ct  = int(n * frac); key = int(frac * 100)
        m_sub = [(t, y, yh) for t, y, yh in model_preds if t >= ct]
        b_sub = [(t, y, yh) for t, y, yh in base_preds  if t >= ct]
        if m_sub:
            out[f'rmse_{key}'] = oos_rmse(model_preds, n, frac)
            t_stat, t_p = stat_fn(b_sub, m_sub)
            out[f'cw_t_{key}'] = t_stat; out[f'cw_p_{key}'] = t_p
        else:
            out[f'rmse_{key}'] = out[f'cw_t_{key}'] = out[f'cw_p_{key}'] = np.nan
    return out


def sig_star(p):
    if p is None or (isinstance(p, float) and np.isnan(p)): return ''
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''


def load_nlp_panel(fname, remap, ewma_span=None):
    raw = pd.read_csv(NLP_DIR + '\\' + fname, parse_dates=['meeting_date'])
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


def add_interactions(df, sent_col, macro_cols=None):
    df = df.copy(); cols = []
    targets = macro_cols if macro_cols else {mc: mc for mc in MACRO_INTER_COLS if mc in df.columns}
    if isinstance(targets, dict):
        items = targets.items()
    else:
        items = [(mc, mc) for mc in targets]
    for mc_key, mc_val in items:
        if mc_key in df.columns and mc_val in df.columns:
            iname = f'inter_{mc_key}_x_{mc_val}'
            df[iname] = df[mc_key] * df[mc_val]; cols.append(iname)
        elif sent_col in df.columns and mc_key in df.columns:
            iname = f'inter_{sent_col}_x_{mc_key}'
            df[iname] = df[sent_col] * df[mc_key]; cols.append(iname)
    return df, cols


def add_4e_interactions(df, sent_col='sent_total'):
    df = df.copy(); cols = []
    for mc in MACRO_INTER_COLS:
        if mc in df.columns and sent_col in df.columns:
            iname = f'inter_{sent_col}_x_{mc}'
            df[iname] = df[sent_col] * df[mc]; cols.append(iname)
    return df, cols


def add_matched_interactions(df, matched_dict):
    df = df.copy(); cols = []
    for sent_col, mc in matched_dict.items():
        if sent_col in df.columns and mc in df.columns:
            iname = f'inter_{sent_col}_x_{mc}'
            df[iname] = df[sent_col] * df[mc]; cols.append(iname)
    return df, cols


# ── Model runners with adj-R² ──────────────────────────────────────────────────

def run_standard(df, extra_cols, target_col):
    """Standard FAVAR (full-sample macro PCA + sentiment extras). Returns (m_preds, b_preds, n, adj_r2)."""
    mc_avail = [c for c in MACRO_REGRESSORS if c in df.columns]
    ex_avail = [c for c in extra_cols       if c in df.columns]
    if not ex_avail:
        return None, None, 0, np.nan

    fac, _, _, nc = run_pca(df.dropna(subset=mc_avail), mc_avail, var_threshold=PCA_VAR_MACRO)
    fac.columns   = [f'PC{i+1}' for i in range(nc)]

    X_aug = fac.join(df[ex_avail], how='left').dropna()
    y_srs = df.loc[X_aug.index, target_col].dropna()
    X_aug = X_aug.loc[y_srs.index]
    n     = len(X_aug)
    if n < 30:
        return None, None, n, np.nan

    y       = y_srs.values
    pc_cols = [c for c in X_aug.columns if c.startswith('PC')]
    X_model = sm.add_constant(X_aug.values)
    X_base  = sm.add_constant(X_aug[pc_cols].values)

    m_preds = expanding_ols(y, X_model)
    b_preds = expanding_ols(y, X_base)
    adj_r2  = round(sm.OLS(y, X_model).fit().rsquared_adj, 4)
    return m_preds, b_preds, n, adj_r2


def run_4h(df, pca_input_cols, target_col):
    """Joint PCA model. Returns (m_preds, b_preds, n, adj_r2)."""
    mc_avail = [c for c in MACRO_REGRESSORS if c in df.columns]
    avail    = [c for c in pca_input_cols   if c in df.columns]
    all_need = list(dict.fromkeys(avail + mc_avail + [target_col]))
    sub = df[all_need].dropna().reset_index(drop=True)
    n   = len(sub)
    if n < 30:
        return None, None, n, np.nan

    avail_s = [c for c in avail if c in sub.columns]
    fac_m, _, _, nc_m = run_pca(sub, avail_s, var_threshold=PCA_VAR_MACRO)
    fac_m.columns = [f'JPC{i+1}' for i in range(nc_m)]

    mc_s = [c for c in mc_avail if c in sub.columns]
    fac_b, _, _, nc_b = run_pca(sub, mc_s, var_threshold=PCA_VAR_MACRO)
    fac_b.columns = [f'BPC{i+1}' for i in range(nc_b)]

    y       = sub[target_col].values
    X_model = sm.add_constant(fac_m.values)
    X_base  = sm.add_constant(fac_b.values)

    m_preds = expanding_ols(y, X_model)
    b_preds = expanding_ols(y, X_base)
    adj_r2  = round(sm.OLS(y, X_model).fit().rsquared_adj, 4)
    return m_preds, b_preds, n, adj_r2


# ── Run all models ─────────────────────────────────────────────────────────────

def collect_rows(target_col, label_suffix=''):
    rows = []
    tag  = f'  [{label_suffix}]' if label_suffix else ''

    # ── LLM ───────────────────────────────────────────────────────────────────
    print(f'{tag} LLM: loading panel...', flush=True)
    df_llm = pd.read_csv(INTER_CSV, parse_dates=['date'])
    df_llm = df_llm[df_llm['date'] >= '2007-01-01'].reset_index(drop=True)

    mc_llm   = [c for c in MACRO_REGRESSORS if c in df_llm.columns]
    sent_tot = ['sent_total']
    sent_top = [c for c in TOPIC_COLS if c in df_llm.columns]

    df_llm_i = df_llm.copy()
    inter_llm = []
    for mc in MACRO_INTER_COLS:
        if mc in df_llm_i.columns and 'sent_total' in df_llm_i.columns:
            iname = f'inter_sent_total_x_{mc}'
            df_llm_i[iname] = df_llm_i['sent_total'] * df_llm_i[mc]
            inter_llm.append(iname)

    llm_variants = [
        ('4h_llm_total',       df_llm,   mc_llm + sent_tot),
        ('4h_llm_topics',      df_llm,   mc_llm + sent_top),
        ('4h_llm_all_sent',    df_llm,   mc_llm + sent_tot + sent_top),
        ('4h_llm_total_inter', df_llm_i, mc_llm + sent_tot + inter_llm),
    ]
    llm_rows = []
    for label, df_v, pca_cols in llm_variants:
        print(f'    {label}...', end='', flush=True)
        m_preds, b_preds, n, adj_r2 = run_4h(df_v, pca_cols, target_col)
        if m_preds is None:
            print(' skipped'); continue
        metrics = compute_metrics(m_preds, b_preds, n, test='dm')
        metrics.update({'model': label, 'method': 'LLM', 'n': n, 'adj_r2': adj_r2})
        llm_rows.append(metrics)
        print(f'  N={n}  OOS80={metrics["rmse_80"]:.4f}  adj-R²={adj_r2:.3f}')

    llm_rows.sort(key=lambda r: r['rmse_80'] if not np.isnan(r['rmse_80']) else 999)
    rows.extend(llm_rows[:3])

    # ── Gardner ───────────────────────────────────────────────────────────────
    print(f'{tag} Gardner: loading panels...', flush=True)

    if target_col == TARGET_NEXT:
        # Level: press_conf specs (verified for level)
        specs = [
            ('G4_pc_wc_EWMA10_4b_cons',
             'Gardner_press_conferences_wordcount_nlp.csv', GARDNER_REMAP, 10,
             'consensus', None),
            ('G5_pc_zs_EWMA10_4f_matched',
             'Gardner_press_conferences_zscore_nlp.csv', GARDNER_REMAP, 10,
             'matched_topic', None),
            ('G6_pc_zs_EWMA10_4b_cons_matched',
             'Gardner_press_conferences_zscore_nlp.csv', GARDNER_REMAP, 10,
             'matched_consensus', None),
        ]
    else:
        # Delta: speeches + minutes specs (full N=151)
        specs = [
            ('G4_speeches_zs_4f_matched',
             'Gardner_speeches_zscore_nlp.csv', GARDNER_REMAP, None,
             'matched_topic', None),
            ('G5_minutes_zs_ewma10_4c_topics',
             'Gardner_minutes_zscore_nlp.csv', GARDNER_REMAP, 10,
             'topics', None),
            ('G6_minutes_zs_4f_matched',
             'Gardner_minutes_zscore_nlp.csv', GARDNER_REMAP, None,
             'matched_topic', None),
        ]

    for label, fname, remap, ewma, spec_type, _ in specs:
        print(f'    {label}...', end='', flush=True)
        df_g = load_nlp_panel(fname, remap, ewma_span=ewma)
        if spec_type == 'consensus':
            extra = [c for c in df_g.columns if c.endswith('_consensus')]
        elif spec_type == 'matched_topic':
            df_g, extra = add_matched_interactions(df_g, MATCHED_TOPIC)
        elif spec_type == 'matched_consensus':
            df_g, extra = add_matched_interactions(df_g, MATCHED_CONSENSUS)
        elif spec_type == 'topics':
            extra = [c for c in TOPIC_COLS if c in df_g.columns and c != 'sent_macro']
        else:
            extra = []
        m_preds, b_preds, n, adj_r2 = run_standard(df_g, extra, target_col)
        if m_preds is None:
            print(' skipped'); continue
        metrics = compute_metrics(m_preds, b_preds, n)
        metrics.update({'model': label, 'method': 'Gardner', 'n': n, 'adj_r2': adj_r2})
        rows.append(metrics)
        print(f'  N={n}  OOS80={metrics["rmse_80"]:.4f}  adj-R²={adj_r2:.3f}')

    # ── Sharpe ────────────────────────────────────────────────────────────────
    print(f'{tag} Sharpe: loading panels...', flush=True)

    if target_col == TARGET_NEXT:
        # Level: statements_zs 4e, speeches_wc EWMA10 4e, 4h all-docs
        sharpe_level = [
            ('S7_statements_zs_4e',
             'Sharpe_statements_zscore_nlp.csv', SHARPE_REMAP, None, '4e', None),
            ('S8_speeches_wc_EWMA10_4e',
             'Sharpe_speeches_wordcount_nlp.csv', SHARPE_REMAP, 10,  '4e', None),
        ]
        for label, fname, remap, ewma, spec_type, _ in sharpe_level:
            print(f'    {label}...', end='', flush=True)
            df_s = load_nlp_panel(fname, remap, ewma_span=ewma)
            df_s, extra = add_4e_interactions(df_s, 'sent_total')
            m_preds, b_preds, n, adj_r2 = run_standard(df_s, extra, target_col)
            if m_preds is None:
                print(' skipped'); continue
            metrics = compute_metrics(m_preds, b_preds, n)
            metrics.update({'model': label, 'method': 'Sharpe', 'n': n, 'adj_r2': adj_r2})
            rows.append(metrics)
            print(f'  N={n}  OOS80={metrics["rmse_80"]:.4f}  adj-R²={adj_r2:.3f}')

        # S9: 4h sharpe all-docs
        print(f'    S9_4h_sharpe_total_inter...', end='', flush=True)
        df_s9 = load_nlp_panel(
            'Sharpe_statements_minutes_speeches_press_conferences_zscore_nlp.csv',
            SHARPE_4H_REMAP)
        df_s9, inter_s9 = add_4e_interactions(df_s9, 's_net')
        mc_s9  = [c for c in MACRO_REGRESSORS if c in df_s9.columns]
        pca_s9 = mc_s9 + ['s_net'] + [c for c in inter_s9 if c in df_s9.columns]
        m_preds, b_preds, n, adj_r2 = run_4h(df_s9, pca_s9, target_col)
        if m_preds is not None:
            metrics = compute_metrics(m_preds, b_preds, n, test='dm')
            metrics.update({'model': 'S9_4h_sharpe_total_inter', 'method': 'Sharpe', 'n': n, 'adj_r2': adj_r2})
            rows.append(metrics)
            print(f'  N={n}  OOS80={metrics["rmse_80"]:.4f}  adj-R²={adj_r2:.3f}')
    else:
        # Delta: all-docs wc EWMA10 4b_cons, statements wc EWMA10 4b_cons, minutes zs 4e
        delta_sharpe = [
            ('S7_all_docs_wc_ewma10_4b_cons',
             'Sharpe_statements_minutes_speeches_press_conferences_wordcount_nlp.csv',
             SHARPE_REMAP, 10, 'consensus'),
            ('S8_statements_wc_ewma10_4b_cons',
             'Sharpe_statements_wordcount_nlp.csv',
             SHARPE_REMAP, 10, 'consensus'),
            ('S9_minutes_zs_4e_inter',
             'Sharpe_minutes_zscore_nlp.csv',
             SHARPE_REMAP, None, '4e'),
        ]
        for label, fname, remap, ewma, spec_type in delta_sharpe:
            print(f'    {label}...', end='', flush=True)
            df_s = load_nlp_panel(fname, remap, ewma_span=ewma)
            if spec_type == 'consensus':
                extra = [c for c in df_s.columns if c.endswith('_consensus')]
            elif spec_type == '4e':
                df_s, extra = add_4e_interactions(df_s, 'sent_total')
            else:
                extra = []
            m_preds, b_preds, n, adj_r2 = run_standard(df_s, extra, target_col)
            if m_preds is None:
                print(' skipped'); continue
            metrics = compute_metrics(m_preds, b_preds, n)
            metrics.update({'model': label, 'method': 'Sharpe', 'n': n, 'adj_r2': adj_r2})
            rows.append(metrics)
            print(f'  N={n}  OOS80={metrics["rmse_80"]:.4f}  adj-R²={adj_r2:.3f}')

    return rows


# ── Table printer ──────────────────────────────────────────────────────────────

def print_table(rows, title, baseline_rmse, baseline_adj_r2, baseline_n):
    W_NAME = 42
    print()
    print('=' * 100)
    print(f'  {title}')
    print('  DM = Diebold-Mariano [non-nested joint-PCA]  |  CW = Clark-West [nested FAVAR]')
    print('  Significance: * p<0.10  ** p<0.05  *** p<0.01  (blank = model RMSE > baseline)')
    print('=' * 100)

    hdr = (f'  {"Method":<9} {"Model":<{W_NAME}} {"N":>5}  {"Adj-R2":>6}'
           + ''.join(f'  {f"OOS{int(f*100)}":>10}' for f in CUTOFFS))
    print(hdr)
    print('  ' + '=' * 98)

    # Baseline row
    adj_s = f'{baseline_adj_r2:.3f}' if not np.isnan(baseline_adj_r2) else '  n/a'
    bl_line = f'  {"Baseline":<9} {"Macro-only FAVAR (no sentiment)":<{W_NAME}} {baseline_n:>5}  {adj_s:>6}'
    for v in baseline_rmse:
        cell = f'{v:.3f}   ' if not np.isnan(v) else '   n/a'
        bl_line += f'  {cell:>10}'
    print(bl_line)
    print('  ' + '=' * 98)

    prev_method = None
    for row in rows:
        method = row['method']
        if prev_method and method != prev_method:
            print('  ' + '-' * 98)
        prev_method = method

        name   = row['model'][:W_NAME]
        n      = row['n']
        adj_r2 = row.get('adj_r2', np.nan)
        test   = row.get('_test', 'cw').upper()
        adj_s  = f'{adj_r2:.3f}' if not np.isnan(adj_r2) else '  n/a'

        line = f'  {method:<9} {name:<{W_NAME}} {n:>5}  {adj_s:>6}'
        for frac in CUTOFFS:
            key  = int(frac * 100)
            rmse = row.get(f'rmse_{key}', np.nan)
            p    = row.get(f'cw_p_{key}', np.nan)
            star = sig_star(p)
            if np.isnan(rmse):
                cell = f'{"n/a":>10}'
            else:
                cell = f'{rmse:.3f}{star:<3}'
                cell = f'{cell:>10}'
            line += f'  {cell}'
        line += f'  [{test}]'
        print(line)
    print()


# ── Macro baseline ─────────────────────────────────────────────────────────────

def compute_macro_baseline(target_col):
    """Returns (rmse_list, adj_r2, n, preds) for the macro-only FAVAR baseline."""
    macro = load_macro()
    df = macro[macro['date'] >= '2007-01-01'].reset_index(drop=True)
    mc_avail = [c for c in MACRO_REGRESSORS if c in df.columns]
    fac, _, _, nc = run_pca(df.dropna(subset=mc_avail), mc_avail, var_threshold=PCA_VAR_MACRO)
    fac.columns = [f'PC{i+1}' for i in range(nc)]
    y_srs = df.loc[fac.index, target_col].dropna()
    fac   = fac.loc[y_srs.index]
    y     = y_srs.values
    X     = sm.add_constant(fac.values)
    n     = len(y)
    preds = expanding_ols(y, X)
    rmse_list = [oos_rmse(preds, n, f) for f in CUTOFFS]
    adj_r2    = round(sm.OLS(y, X).fit().rsquared_adj, 4)
    return rmse_list, adj_r2, n


# ── Main ──────────────────────────────────────────────────────────────────────

print('\n  Computing macro baselines...', flush=True)
bl_level_rmse, bl_level_r2, bl_level_n = compute_macro_baseline(TARGET_NEXT)
bl_delta_rmse, bl_delta_r2, bl_delta_n = compute_macro_baseline(DELTA_COL)
print(f'  Baseline LEVEL: {[round(v,3) for v in bl_level_rmse]}  adj-R2={bl_level_r2:.3f}')
print(f'  Baseline DELTA: {[round(v,3) for v in bl_delta_rmse]}  adj-R2={bl_delta_r2:.3f}')

print('\n  Running LEVEL models...', flush=True)
rows_level = collect_rows(TARGET_NEXT, 'LEVEL')

print('\n  Running DELTA models...', flush=True)
rows_delta = collect_rows(DELTA_COL,  'DELTA')

print_table(rows_level,
            'TABLE 1: LEVEL TARGET  (effective rate at next FOMC meeting)',
            bl_level_rmse, bl_level_r2, bl_level_n)

print_table(rows_delta,
            'TABLE 2: DELTA TARGET  (change in effective rate at next FOMC meeting)',
            bl_delta_rmse, bl_delta_r2, bl_delta_n)

# Build baseline rows for CSV
def make_baseline_row(rmse_list, adj_r2, n):
    row = {'model': 'Macro-only FAVAR (no sentiment)', 'method': 'Baseline', 'n': n, 'adj_r2': adj_r2}
    for frac, v in zip(CUTOFFS, rmse_list):
        key = int(frac * 100)
        row[f'rmse_{key}'] = v
        row[f'cw_t_{key}'] = np.nan
        row[f'cw_p_{key}'] = np.nan
    return row

# Save (baseline row first)
out_level = root + r'\Taylor Rule\outputs\summary_table_level.csv'
out_delta = root + r'\Taylor Rule\outputs\summary_table_delta.csv'
pd.DataFrame([make_baseline_row(bl_level_rmse, bl_level_r2, bl_level_n)] + rows_level).to_csv(out_level, index=False)
pd.DataFrame([make_baseline_row(bl_delta_rmse, bl_delta_r2, bl_delta_n)] + rows_delta).to_csv(out_delta, index=False)
print(f'  Saved -> summary_table_level.csv  |  summary_table_delta.csv')
