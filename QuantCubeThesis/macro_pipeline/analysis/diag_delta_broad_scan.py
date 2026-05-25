"""
diag_delta_broad_scan.py

Broad scan: test all Gardner and Sharpe doc-type files × {4b, 4c, 4e, 4f} specs
on delta_rate_next (N=151, 2007-2025).

Goal: verify whether the minutes-based Gardner/Sharpe specs used in
diag_top3_cw_splits_delta.py are genuinely among the best for delta on current data.

Prints a ranked table by OOS80 for each method.
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

root    = r'C:\Users\Javier\OneDrive - HEC Paris\Documentos\QuantCubeThesis\QuantCubeThesis'
NLP_DIR = root + r'\Taylor Rule\nlp_output'

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
MATCHED_CONSENSUS = {
    'sent_inflation_consensus':            'inflation_dev_from_target',
    'sent_labor_market_consensus':         'unemployment_gap',
    'sent_economic_activity_consensus':    'gdp',
    'sent_financial_conditions_consensus': 'vix',
    'sent_monetary_policy_consensus':      'implied_ffr',
}

TOPIC_COLS = ['sent_inflation', 'sent_labor_market', 'sent_economic_activity',
              'sent_financial_conditions', 'sent_monetary_policy']


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


def oos_rmse(preds, n, frac):
    ct  = int(n * frac)
    sub = [(t, y, yh) for t, y, yh in preds if t >= ct]
    if not sub:
        return np.nan
    return np.sqrt(np.mean([(y - yh)**2 for _, y, yh in sub]))


def run_standard_delta(df, extra_cols):
    mc_avail = [c for c in MACRO_REGRESSORS if c in df.columns]
    ex_avail = [c for c in extra_cols       if c in df.columns]
    if not ex_avail:
        return None, None, 0

    fac, _, _, nc = run_pca(df.dropna(subset=mc_avail), mc_avail,
                            var_threshold=PCA_VAR_MACRO)
    fac.columns = [f'PC{i+1}' for i in range(nc)]

    X_aug = fac.join(df[ex_avail], how='left').dropna()
    y_srs = df.loc[X_aug.index, DELTA_COL].dropna()
    X_aug = X_aug.loc[y_srs.index]
    n = len(X_aug)
    if n < 30:
        return None, None, n

    y      = y_srs.values
    pc_cols = [c for c in X_aug.columns if c.startswith('PC')]
    X_model = sm.add_constant(X_aug.values)
    X_base  = sm.add_constant(X_aug[pc_cols].values)

    return expanding_ols(y, X_model), expanding_ols(y, X_base), n


def load_panel(fname, remap, ewma_span=None):
    raw = pd.read_csv(NLP_DIR + '\\' + fname, parse_dates=['meeting_date'])
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


def add_4e_interactions(df, sent_col='sent_total'):
    df = df.copy()
    cols = []
    for mc in MACRO_INTER_COLS:
        if mc in df.columns and sent_col in df.columns:
            iname = f'inter_{sent_col}_x_{mc}'
            df[iname] = df[sent_col] * df[mc]
            cols.append(iname)
    return df, cols


def run_all_specs(df, label):
    """Run 4b/4c/4e/4f specs on a preloaded panel; return list of result dicts."""
    results = []

    # 4b_consensus
    cons_cols = [c for c in df.columns if c.endswith('_consensus')]
    if cons_cols:
        m, b, n = run_standard_delta(df, cons_cols)
        if m:
            results.append({'label': f'{label}_4b_cons', 'n': n,
                            'oos80': round(oos_rmse(m, n, 0.80), 4),
                            'oos60': round(oos_rmse(m, n, 0.60), 4),
                            'oos70': round(oos_rmse(m, n, 0.70), 4),
                            'oos90': round(oos_rmse(m, n, 0.90), 4)})

    # 4c_all_topics
    top_avail = [c for c in TOPIC_COLS if c in df.columns]
    if top_avail:
        m, b, n = run_standard_delta(df, top_avail)
        if m:
            results.append({'label': f'{label}_4c_topics', 'n': n,
                            'oos80': round(oos_rmse(m, n, 0.80), 4),
                            'oos60': round(oos_rmse(m, n, 0.60), 4),
                            'oos70': round(oos_rmse(m, n, 0.70), 4),
                            'oos90': round(oos_rmse(m, n, 0.90), 4)})

    # 4e_total_inter
    if 'sent_total' in df.columns:
        df_4e, inter_4e = add_4e_interactions(df, 'sent_total')
        m, b, n = run_standard_delta(df_4e, inter_4e)
        if m:
            results.append({'label': f'{label}_4e_inter', 'n': n,
                            'oos80': round(oos_rmse(m, n, 0.80), 4),
                            'oos60': round(oos_rmse(m, n, 0.60), 4),
                            'oos70': round(oos_rmse(m, n, 0.70), 4),
                            'oos90': round(oos_rmse(m, n, 0.90), 4)})

    # 4f_matched (topic × paired macro)
    match_avail = {k: v for k, v in MATCHED_TOPIC.items() if k in df.columns}
    if match_avail:
        df_4f = df.copy()
        inter_4f = []
        for sent_col, mc in match_avail.items():
            if mc in df_4f.columns:
                iname = f'inter_{sent_col}_x_{mc}'
                df_4f[iname] = df_4f[sent_col] * df_4f[mc]
                inter_4f.append(iname)
        if inter_4f:
            m, b, n = run_standard_delta(df_4f, inter_4f)
            if m:
                results.append({'label': f'{label}_4f_matched', 'n': n,
                                'oos80': round(oos_rmse(m, n, 0.80), 4),
                                'oos60': round(oos_rmse(m, n, 0.60), 4),
                                'oos70': round(oos_rmse(m, n, 0.70), 4),
                                'oos90': round(oos_rmse(m, n, 0.90), 4)})

    # 4f_matched_consensus
    match_cons = {k: v for k, v in MATCHED_CONSENSUS.items() if k in df.columns}
    if match_cons:
        df_4fc = df.copy()
        inter_4fc = []
        for sent_col, mc in match_cons.items():
            if mc in df_4fc.columns:
                iname = f'inter_{sent_col}_x_{mc}'
                df_4fc[iname] = df_4fc[sent_col] * df_4fc[mc]
                inter_4fc.append(iname)
        if inter_4fc:
            m, b, n = run_standard_delta(df_4fc, inter_4fc)
            if m:
                results.append({'label': f'{label}_4f_cons_matched', 'n': n,
                                'oos80': round(oos_rmse(m, n, 0.80), 4),
                                'oos60': round(oos_rmse(m, n, 0.60), 4),
                                'oos70': round(oos_rmse(m, n, 0.70), 4),
                                'oos90': round(oos_rmse(m, n, 0.90), 4)})
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

print('=' * 90)
print('  BROAD SCAN: DELTA  (all doc types × 4b/4c/4e/4f specs)')
print('  Macro baseline OOS80 ~ 0.265  |  lower = better')
print('=' * 90)

GARDNER_FILES = [
    'Gardner_statements_wordcount_nlp.csv',
    'Gardner_statements_zscore_nlp.csv',
    'Gardner_minutes_wordcount_nlp.csv',
    'Gardner_minutes_zscore_nlp.csv',
    'Gardner_speeches_wordcount_nlp.csv',
    'Gardner_speeches_zscore_nlp.csv',
    'Gardner_press_conferences_wordcount_nlp.csv',
    'Gardner_press_conferences_zscore_nlp.csv',
    'Gardner_statements_minutes_speeches_press_conferences_wordcount_nlp.csv',
    'Gardner_statements_minutes_speeches_press_conferences_zscore_nlp.csv',
]

SHARPE_FILES = [
    'Sharpe_statements_wordcount_nlp.csv',
    'Sharpe_statements_zscore_nlp.csv',
    'Sharpe_minutes_wordcount_nlp.csv',
    'Sharpe_minutes_zscore_nlp.csv',
    'Sharpe_speeches_wordcount_nlp.csv',
    'Sharpe_speeches_zscore_nlp.csv',
    'Sharpe_press_conferences_wordcount_nlp.csv',
    'Sharpe_press_conferences_zscore_nlp.csv',
    'Sharpe_statements_minutes_speeches_press_conferences_wordcount_nlp.csv',
    'Sharpe_statements_minutes_speeches_press_conferences_zscore_nlp.csv',
]

gardner_all = []
sharpe_all  = []

print('\n  [Gardner] scanning...', flush=True)
for fname in GARDNER_FILES:
    short = fname.replace('Gardner_', '').replace('_nlp.csv', '')
    print(f'  {short:55s}', end='', flush=True)
    df = load_panel(fname, GARDNER_REMAP)
    res = run_all_specs(df, short)
    gardner_all.extend(res)
    best = min(res, key=lambda r: r['oos80']) if res else None
    print(f'  best_OOS80={best["oos80"]:.4f} ({best["label"].split("_")[-2]+"_"+best["label"].split("_")[-1]})' if best else '  (no results)', flush=True)

# Also test EWMA=10 for Gardner (key for press_conf)
print('\n  [Gardner EWMA=10] scanning...', flush=True)
for fname in GARDNER_FILES:
    short = fname.replace('Gardner_', '').replace('_nlp.csv', '') + '_ewma10'
    print(f'  {short:55s}', end='', flush=True)
    df = load_panel(fname, GARDNER_REMAP, ewma_span=10)
    res = run_all_specs(df, short)
    gardner_all.extend(res)
    best = min(res, key=lambda r: r['oos80']) if res else None
    print(f'  best_OOS80={best["oos80"]:.4f} ({best["label"].split("_")[-2]+"_"+best["label"].split("_")[-1]})' if best else '  (no results)', flush=True)

print('\n  [Sharpe] scanning...', flush=True)
for fname in SHARPE_FILES:
    short = fname.replace('Sharpe_', '').replace('_nlp.csv', '')
    print(f'  {short:55s}', end='', flush=True)
    df = load_panel(fname, SHARPE_REMAP)
    res = run_all_specs(df, short)
    sharpe_all.extend(res)
    best = min(res, key=lambda r: r['oos80']) if res else None
    print(f'  best_OOS80={best["oos80"]:.4f} ({best["label"].split("_")[-2]+"_"+best["label"].split("_")[-1]})' if best else '  (no results)', flush=True)

print('\n  [Sharpe EWMA=10] scanning...', flush=True)
for fname in SHARPE_FILES:
    short = fname.replace('Sharpe_', '').replace('_nlp.csv', '') + '_ewma10'
    print(f'  {short:55s}', end='', flush=True)
    df = load_panel(fname, SHARPE_REMAP, ewma_span=10)
    res = run_all_specs(df, short)
    sharpe_all.extend(res)
    best = min(res, key=lambda r: r['oos80']) if res else None
    print(f'  best_OOS80={best["oos80"]:.4f} ({best["label"].split("_")[-2]+"_"+best["label"].split("_")[-1]})' if best else '  (no results)', flush=True)

# ── Print ranked tables ────────────────────────────────────────────────────────

def print_ranked(results, method, top_n=10):
    ranked = sorted([r for r in results if not np.isnan(r['oos80'])],
                    key=lambda r: r['oos80'])
    print(f'\n  {method} TOP {top_n} BY OOS80 (macro baseline OOS80=0.265)')
    print(f'  {"Spec":<65} {"N":>5}  {"OOS60":>6}  {"OOS70":>6}  {"OOS80":>6}  {"OOS90":>6}')
    print('  ' + '-' * 95)
    for r in ranked[:top_n]:
        print(f'  {r["label"]:<65} {r["n"]:>5}  {r["oos60"]:>6.4f}  {r["oos70"]:>6.4f}  {r["oos80"]:>6.4f}  {r["oos90"]:>6.4f}')

print('\n\n')
print('=' * 90)
print_ranked(gardner_all, 'GARDNER', top_n=10)
print_ranked(sharpe_all,  'SHARPE',  top_n=10)

# Save
pd.DataFrame(gardner_all).sort_values('oos80').to_csv(
    root + r'\Taylor Rule\outputs\broad_scan_gardner_delta.csv', index=False)
pd.DataFrame(sharpe_all).sort_values('oos80').to_csv(
    root + r'\Taylor Rule\outputs\broad_scan_sharpe_delta.csv', index=False)
print('\n  Saved -> broad_scan_gardner_delta.csv / broad_scan_sharpe_delta.csv')
