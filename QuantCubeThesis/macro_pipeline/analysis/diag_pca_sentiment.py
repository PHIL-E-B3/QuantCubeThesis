"""Diagnostic: PCA with sentiment included alongside macro variables — LLM, Gardner, Sharpe."""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

root = r'C:\Users\Javier\OneDrive - HEC Paris\Documentos\QuantCubeThesis\QuantCubeThesis'
nlp_dir = root + r'\Taylor Rule\nlp_output'

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from config import TARGET_NEXT, MACRO_REGRESSORS
from step0_aggregate import load_macro

macro_cols = MACRO_REGRESSORS  # used as fixed list; filtered per-dataset below


# ── Core function ─────────────────────────────────────────────────────────────

def run_pca_favar(data, pca_cols, target, label, variance_threshold=0.90, oos_frac=0.80):
    avail = [c for c in pca_cols if c in data.columns]
    sub = data[avail + [target]].dropna()
    if len(sub) < 30:
        return None

    X = sub[avail].values
    X_sc = StandardScaler().fit_transform(X)

    cumvar = np.cumsum(PCA().fit(X_sc).explained_variance_ratio_)
    n_comp = int(np.searchsorted(cumvar, variance_threshold)) + 1

    factors = PCA(n_components=n_comp).fit_transform(X_sc)
    var_exp = cumvar[n_comp - 1] * 100

    y = sub[target].values
    X_reg = sm.add_constant(factors)
    model = sm.OLS(y, X_reg).fit()

    n_train = int(len(y) * oos_frac)
    preds = [sm.OLS(y[:i], X_reg[:i]).fit().predict(X_reg[i])[0]
             for i in range(n_train, len(y))]
    oos_rmse = np.sqrt(np.mean((y[n_train:] - np.array(preds)) ** 2))

    return {'label': label, 'n': len(sub), 'n_comp': n_comp,
            'var_exp': var_exp, 'adj_r2': model.rsquared_adj, 'oos_rmse_80': oos_rmse}


def print_block(title, rows, base_n):
    print()
    print(f'{"=" * 72}')
    print(f'  {title}')
    print(f'{"=" * 72}')
    print(f'  {"Spec":<38} {"N":>4} {"Comps":>6} {"Var%":>6} {"Adj R2":>9} {"OOS80":>9}')
    print('  ' + '-' * 72)
    for r in rows:
        if r is None:
            continue
        extra = f'+{r["n_comp"] - base_n}' if r["n_comp"] != base_n else '  '
        print(f'  {r["label"]:<38} {r["n"]:>4} {r["n_comp"]:>5}{extra} '
              f'{r["var_exp"]:>6.1f} {r["adj_r2"]:>9.4f} {r["oos_rmse_80"]:>9.4f}')


# ── Helper: build Gardner/Sharpe macro panel ──────────────────────────────────

def build_dict_panel(dict_name, remap):
    fname = (f'{nlp_dir}\\{dict_name}_statements_minutes_speeches'
             f'_press_conferences_zscore_nlp.csv')
    raw = pd.read_csv(fname, parse_dates=['meeting_date'])
    score_cols = [c for c in raw.columns
                  if c.startswith(dict_name.lower() + '_') and 'consensus' not in c]
    agg = raw.groupby('meeting_date')[score_cols].sum().reset_index()
    agg = agg.rename(columns={'meeting_date': 'date', **remap})
    macro = load_macro()
    df = pd.merge_asof(macro.sort_values('date'), agg.sort_values('date'),
                       on='date', direction='backward')
    return df[df['date'] >= '2007-01-01'].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LLM
# ══════════════════════════════════════════════════════════════════════════════

df_llm = pd.read_csv(root + r'\Taylor Rule\outputs\intermediate\step0_merged_to_macro.csv',
                     parse_dates=['date'])
df_llm = df_llm[df_llm['date'] >= '2007-01-01'].reset_index(drop=True)

mc = [c for c in macro_cols if c in df_llm.columns]
llm_topics = ['sent_inflation', 'sent_labor_market', 'sent_economic_activity',
              'sent_financial_conditions', 'sent_monetary_policy']
llm_topics = [c for c in llm_topics if c in df_llm.columns]

llm_specs = [
    (mc,                                          'Macro only (baseline)'),
    (mc + ['sent_total'],                         'Macro + sent_total'),
    (mc + ['sent_inflation', 'sent_monetary_policy'], 'Macro + inf + monpol'),
    (mc + ['sent_labor_market'],                  'Macro + labor_mkt'),
    (mc + llm_topics,                             'Macro + all topics (no total)'),
    (mc + ['sent_total'] + llm_topics,            'Macro + total + all topics'),
]

llm_rows = [run_pca_favar(df_llm, cols, TARGET_NEXT, lbl) for cols, lbl in llm_specs]
base_n_llm = llm_rows[0]['n_comp']
print_block('LLM — Joint PCA FAVAR (post-2007, N=151)', llm_rows, base_n_llm)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Gardner
# ══════════════════════════════════════════════════════════════════════════════

GARDNER_REMAP = {
    'gardner_total': 'g_total', 'gardner_inf': 'g_inflation',
    'gardner_labor': 'g_labor_mkt', 'gardner_out': 'g_econ_act',
    'gardner_fin':   'g_fin_cond',  'gardner_mp':  'g_mon_pol',
}
df_g = build_dict_panel('Gardner', GARDNER_REMAP)
mc_g = [c for c in macro_cols if c in df_g.columns]
g_topics = [v for v in GARDNER_REMAP.values() if v != 'g_total' and v in df_g.columns]

g_specs = [
    (mc_g,                              'Macro only (baseline)'),
    (mc_g + ['g_total'],                'Macro + g_total'),
    (mc_g + ['g_inflation', 'g_mon_pol'], 'Macro + g_inf + g_monpol'),
    (mc_g + ['g_labor_mkt'],            'Macro + g_labor_mkt'),
    (mc_g + g_topics,                   'Macro + all topics (no total)'),
    (mc_g + ['g_total'] + g_topics,     'Macro + total + all topics'),
]

g_rows = [run_pca_favar(df_g, cols, TARGET_NEXT, lbl) for cols, lbl in g_specs]
base_n_g = g_rows[0]['n_comp']
print_block('Gardner — Joint PCA FAVAR (post-2007, all doc types, zscore)', g_rows, base_n_g)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Sharpe
# ══════════════════════════════════════════════════════════════════════════════

SHARPE_REMAP = {
    'sharpe_net': 's_net', 'sharpe_positive': 's_positive',
    'sharpe_negative': 's_negative', 'sharpe_consensus': 's_consensus',
}
df_s = build_dict_panel('Sharpe', SHARPE_REMAP)
mc_s = [c for c in macro_cols if c in df_s.columns]

s_specs = [
    (mc_s,                              'Macro only (baseline)'),
    (mc_s + ['s_net'],                  'Macro + s_net'),
    (mc_s + ['s_positive', 's_negative'], 'Macro + s_pos + s_neg'),
    (mc_s + ['s_consensus'],            'Macro + s_consensus'),
    (mc_s + ['s_net', 's_consensus'],   'Macro + s_net + s_consensus'),
]

s_rows = [run_pca_favar(df_s, cols, TARGET_NEXT, lbl) for cols, lbl in s_specs]
base_n_s = s_rows[0]['n_comp']
print_block('Sharpe — Joint PCA FAVAR (post-2007, all doc types, zscore)', s_rows, base_n_s)


# ══════════════════════════════════════════════════════════════════════════════
# Summary: best per method
# ══════════════════════════════════════════════════════════════════════════════

print()
print('=' * 72)
print('  BEST PER METHOD (joint PCA FAVAR, post-2007, OOS 80/20)')
print('=' * 72)
print(f'  {"Method":<20} {"Best spec":<40} {"Adj R2":>9} {"OOS80":>9}')
print('  ' + '-' * 72)

for method, rows in [('LLM', llm_rows), ('Gardner', g_rows), ('Sharpe', s_rows)]:
    valid = [r for r in rows if r is not None]
    best = min(valid, key=lambda r: r['oos_rmse_80'])
    print(f'  {method:<20} {best["label"]:<40} {best["adj_r2"]:>9.4f} {best["oos_rmse_80"]:>9.4f}')
