"""
diag_4h_method_comparison.py

Apples-to-apples comparison of LLM, Gardner (2022), and Sharpe (2023) dictionary
scores in the joint-PCA FAVAR architecture on the RATE CHANGE (delta) target.

All prior results for Gardner/Sharpe used the LEVEL target and CW test.
This script is the first to run the delta target with the DM test (correct for
non-nested joint PCA models) across all three methods at h=1.

Architecture for every model:
  - Joint PCA on [macro_regressors + sentiment_cols] -> factors
  - Expanding OLS (50% training) on cum_delta_h1
  - DM test vs Macro FAVAR baseline (PCA on macro_regressors only)

Methods and variants:
  LLM      : macro + sent_total + 6 topic scores (best LLM spec = 4h_llm_all_sent)
  Gardner  : macro + all 6 Gardner scores  (g_total, g_inf, g_labor, g_out, g_fin, g_mp)
  Gardner-T: macro + gardner_total only
  Sharpe   : macro + all 4 Sharpe scores   (s_net, s_pos, s_neg, s_consensus)
  Sharpe-N : macro + sharpe_net only

Output: side-by-side DM table comparing all variants.
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

TRAIN_FRAC = 0.50
H          = 1
START_DATE = '2007-01-01'

TOPIC_COLS = [
    'sent_inflation', 'sent_labor_market', 'sent_economic_activity',
    'sent_financial_conditions', 'sent_monetary_policy', 'sent_macro',
]
GARD_ALL_COLS  = ['gardner_total', 'gardner_inf', 'gardner_labor',
                  'gardner_out',   'gardner_fin',  'gardner_mp']
GARD_TOT_COLS  = ['gardner_total']
SHAR_ALL_COLS  = ['sharpe_net', 'sharpe_positive', 'sharpe_negative', 'sharpe_consensus']
SHAR_NET_COLS  = ['sharpe_net']


# ── Load and aggregate Gardner and Sharpe to meeting level ────────────────────

print('  Loading NLP output files...', flush=True)

gard_raw = pd.read_csv(GARD_CSV, parse_dates=['meeting_date'])
gard_mtg = (gard_raw
            .groupby('meeting_date')[GARD_ALL_COLS]
            .sum()
            .reset_index()
            .rename(columns={'meeting_date': 'date'}))

shar_raw = pd.read_csv(SHAR_CSV, parse_dates=['meeting_date'])
shar_mtg = (shar_raw
            .groupby('meeting_date')[SHAR_ALL_COLS]
            .sum()
            .reset_index()
            .rename(columns={'meeting_date': 'date'}))


# ── Build base macro panel ────────────────────────────────────────────────────

print('  Building macro panel...', flush=True)

df_base = pd.read_csv(INTER_CSV, parse_dates=['date'])
df_base = df_base[df_base['date'] >= START_DATE].reset_index(drop=True)

if 'effective_rate' not in df_base.columns:
    macro = load_macro()[['date', 'effective_rate', 'implied_ffr']]
    df_base = pd.merge_asof(df_base.sort_values('date'), macro.sort_values('date'),
                            on='date', direction='backward')

df_base['cum_delta_h1'] = df_base['effective_rate'].shift(-H) - df_base['effective_rate']

# Merge Gardner and Sharpe scores onto base panel
df_base = pd.merge_asof(df_base.sort_values('date'),
                        gard_mtg.sort_values('date'),
                        on='date', direction='backward')
df_base = pd.merge_asof(df_base.sort_values('date'),
                        shar_mtg.sort_values('date'),
                        on='date', direction='backward')

mc_avail = [c for c in MACRO_REGRESSORS if c in df_base.columns]
print(f'  Macro regressors available: {mc_avail}')


# ── Core function: build expanding OLS predictions ───────────────────────────

def build_preds_for_cols(df, sentiment_cols, target_col='cum_delta_h1'):
    """
    Run joint PCA model and macro FAVAR baseline on df with the given sentiment_cols.
    Returns DataFrames of (t, date, y, yh) for model and baseline.
    """
    avail = [c for c in mc_avail + sentiment_cols if c in df.columns]
    mc_s  = [c for c in mc_avail if c in df.columns]

    need = list(dict.fromkeys(['date'] + avail + [target_col]))
    sub  = df[need].dropna().reset_index(drop=True)
    n    = len(sub)
    init = int(n * TRAIN_FRAC)

    fac_m, _, _, nc_m = run_pca(sub, avail, var_threshold=PCA_VAR_MACRO)
    fac_m.columns = [f'JPC{i+1}' for i in range(nc_m)]

    fac_b, _, _, nc_b = run_pca(sub, mc_s, var_threshold=PCA_VAR_MACRO)
    fac_b.columns = [f'BPC{i+1}' for i in range(nc_b)]

    y       = sub[target_col].values
    X_model = sm.add_constant(fac_m.values)
    X_base  = sm.add_constant(fac_b.values)
    dates   = sub['date'].values

    raw_m, raw_b = [], []
    for t in range(init, n):
        try:
            mm = sm.OLS(y[:t], X_model[:t]).fit()
            mb = sm.OLS(y[:t], X_base[:t]).fit()
            raw_m.append({'t': t, 'date': dates[t],
                          'y': float(y[t]), 'yh': float(mm.predict(X_model[t:t+1])[0])})
            raw_b.append({'t': t, 'date': dates[t],
                          'y': float(y[t]), 'yh': float(mb.predict(X_base[t:t+1])[0])})
        except Exception:
            pass

    return (pd.DataFrame(raw_m).assign(date=lambda d: pd.to_datetime(d['date'])),
            pd.DataFrame(raw_b).assign(date=lambda d: pd.to_datetime(d['date'])),
            init, n)


def summarize(df_m, df_b, label, n_pca_factors='?'):
    """Print one-line DM summary; return dict of results."""
    m_triples = list(zip(df_m['t'], df_m['y'], df_m['yh']))
    b_triples = list(zip(df_b['t'], df_b['y'], df_b['yh']))
    T = len(m_triples)
    if T < 5:
        print(f'  {label:<30}  n={T}  (too few)')
        return {}

    rmse_m = np.sqrt(np.mean([(y - yh)**2 for _, y, yh in m_triples]))
    rmse_b = np.sqrt(np.mean([(y - yh)**2 for _, y, yh in b_triples]))
    improv  = (rmse_b - rmse_m) / rmse_b * 100
    dm_t, dm_p = diebold_mariano_test(b_triples, m_triples)

    def star(p):
        if p is None or (isinstance(p, float) and np.isnan(p)): return ''
        return '***' if p < .01 else ('**' if p < .05 else ('*' if p < .10 else ''))

    t_s = f'{dm_t:>6.3f}' if not np.isnan(dm_t) else '   n/a'
    p_s = f'{dm_p:.4f}' if not (isinstance(dm_p, float) and np.isnan(dm_p)) else '   n/a'

    print(f'  {label:<30}  n={T:>3}  '
          f'mdl={rmse_m:.4f}  base={rmse_b:.4f}  '
          f'imp={improv:>+6.1f}%  DM-t={t_s}  p={p_s}  {star(dm_p)}')

    return {'method': label, 'n_oos': T, 'n_pca': n_pca_factors,
            'rmse_model': round(rmse_m, 4), 'rmse_base': round(rmse_b, 4),
            'improv_pct': round(improv, 1), 'dm_t': round(dm_t, 4),
            'dm_p': round(dm_p, 4), 'sig': star(dm_p)}


# ── Define variants ───────────────────────────────────────────────────────────

llm_sent_cols  = [c for c in ['sent_total'] + TOPIC_COLS if c in df_base.columns]
gard_all_avail = [c for c in GARD_ALL_COLS  if c in df_base.columns]
gard_tot_avail = [c for c in GARD_TOT_COLS  if c in df_base.columns]
shar_all_avail = [c for c in SHAR_ALL_COLS  if c in df_base.columns]
shar_net_avail = [c for c in SHAR_NET_COLS  if c in df_base.columns]

VARIANTS = [
    ('LLM_all_sent',  llm_sent_cols),
    ('Gardner_all',   gard_all_avail),
    ('Gardner_total', gard_tot_avail),
    ('Sharpe_all',    shar_all_avail),
    ('Sharpe_net',    shar_net_avail),
]


# ── Run all variants ──────────────────────────────────────────────────────────

print('\n' + '=' * 90)
print('  METHOD COMPARISON: delta target (cum_delta_h1), h=1, 50% training, 2007+')
print('  Architecture: joint PCA(macro + sentiment) vs macro FAVAR baseline')
print('  Test: DM (Diebold-Mariano, one-sided, H1: model better)')
print('=' * 90)
print(f'\n  {"Method":<30}  {"n":>3}  '
      f'{"mdl RMSE":>9}  {"base RMSE":>9}  {"improv":>7}  '
      f'{"DM-t":>7}  {"p":>7}  {"sig":>3}')
print('  ' + '-' * 82)

results = []
for label, sent_cols in VARIANTS:
    if not sent_cols:
        print(f'  {label:<30}  (no sentiment columns available — skipping)')
        continue
    df_m, df_b, init, n_total = build_preds_for_cols(df_base, sent_cols)
    n_pca = len([c for c in mc_avail + sent_cols if c in df_base.columns])
    row = summarize(df_m, df_b, label, n_pca_factors=n_pca)
    if row:
        results.append(row)

# Baseline-only row for reference (macro FAVAR vs random walk / no-change)
print('  ' + '-' * 82)
print('\n  Note: "base" column is Macro FAVAR (PCA on 7 macro regressors).')
print('  All models use the same expanding-window OLS setup with 50% training.')
print('  DM test is one-sided (H1: model outperforms baseline in RMSE).')
print('  Joint PCA => non-nested vs baseline => DM is the correct test (not CW).')


# ── Save results ──────────────────────────────────────────────────────────────

out_df  = pd.DataFrame(results)
out_path = root + r'\Taylor Rule\outputs\method_comparison_delta_h1.csv'
out_df.to_csv(out_path, index=False)
print(f'\n  Saved -> method_comparison_delta_h1.csv')
print(f'\n  Full results:\n{out_df.to_string(index=False)}')
