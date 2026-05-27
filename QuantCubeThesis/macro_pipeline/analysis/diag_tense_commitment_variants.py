"""
diag_tense_commitment_variants.py

Two sentence-level filtering variants of the 4h_llm_all_sent model:

  Variant A — Interpretive only:
    Re-aggregate sentiment using ONLY sentences labelled tense='interpretive'.
    Drops all backward-looking descriptive sentences; retains only sentences
    that project, signal or interpret the forward outlook.

  Variant B — Exclude documents with any unconditional language:
    For each document (source, date, doc_type), if ANY sentence has
    commitment=True, the entire document's sentiment is excluded from
    the meeting-level aggregate. This tests whether the predictive
    signal survives when 'Odyssean' forward-guidance documents are removed.

Both variants are compared against the same Macro FAVAR baseline at h=1
(50% training, expanding OLS, DM test). Results are printed side-by-side
with the original 4h_llm_all_sent for comparison.

Data flow:
  nlp_labels.json
    → filter sentences
    → aggregate_to_document()    (overwrites step0_document_sentiment.csv — fine)
    → in-memory merge to macro   (does NOT touch step0_merged_to_macro.csv)
    → expanding OLS + DM
"""
import sys
import warnings
import numpy  as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from config import MACRO_REGRESSORS, PCA_VAR_MACRO, TOPICS
from utils import run_pca, diebold_mariano_test
from step0_aggregate import load_sentences, aggregate_to_document, load_macro

root        = r'C:\Users\Javier\OneDrive - HEC Paris\Documentos\QuantCubeThesis\QuantCubeThesis'
SENTENCES   = root + r'\Taylor Rule\nlp_output\nlp_labels.json'
INTER_CSV   = root + r'\Taylor Rule\outputs\intermediate\step0_merged_to_macro.csv'

TRAIN_FRAC  = 0.50
H           = 1
TOPIC_COLS  = [f'sent_{t}' for t in TOPICS if t != 'no_topic']


# ── Sentence → meeting-date merge (in-memory, does not write to disk) ─────────

def sentences_to_meeting_panel(df_sent_filtered: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate filtered sentences to meeting-date level and merge onto the
    existing macro panel (replacing only the sent_* / flag_* columns).

    Returns the full panel with new sentiment columns. Does NOT write to any
    intermediate CSV.
    """
    # Aggregate to document level (writes step0_document_sentiment.csv as side
    # effect — that intermediate file is fine to overwrite)
    doc_df = aggregate_to_document(df_sent_filtered)

    sent_cols = [c for c in doc_df.columns if c.startswith('sent_')]

    # Pool all doc_types: sum sentiment per meeting date (backward as-of merge)
    all_sent = (
        doc_df[['date'] + sent_cols]
        .groupby('date').sum()
        .reset_index()
        .sort_values('date')
    )

    # Load existing macro panel → keep everything EXCEPT old sent_* / flag_*
    df_base = pd.read_csv(INTER_CSV, parse_dates=['date'])
    drop    = [c for c in df_base.columns if c.startswith('sent_') or c.startswith('flag_')]
    df_base = df_base.drop(columns=drop).sort_values('date')

    # Backward as-of merge: each meeting date gets the most recent doc-level sentiment
    merged = pd.merge_asof(df_base, all_sent, on='date', direction='backward')
    return merged.sort_values('date').reset_index(drop=True)


# ── OOS DM test (same setup as diag_multihorizon_dm) ──────────────────────────

def run_dm_h1(df_panel: pd.DataFrame, label: str):
    """
    Expanding-window OLS at h=1 against Macro FAVAR baseline.
    Returns dict with RMSE and DM result.
    """
    df = df_panel.copy()
    df = df[df['date'] >= '2007-01-01'].reset_index(drop=True)

    # Ensure effective_rate and implied_ffr are present
    if 'effective_rate' not in df.columns or df['effective_rate'].isna().all():
        macro = load_macro()[['date', 'effective_rate', 'implied_ffr']]
        df = pd.merge_asof(df.sort_values('date'), macro.sort_values('date'),
                           on='date', direction='backward')

    mc_avail   = [c for c in MACRO_REGRESSORS if c in df.columns]
    sent_tot   = [c for c in ['sent_total']   if c in df.columns]
    sent_top   = [c for c in TOPIC_COLS       if c in df.columns]
    pca_input  = mc_avail + sent_tot + sent_top

    target_col = f'cum_delta_h{H}'
    df[target_col] = df['effective_rate'].shift(-H) - df['effective_rate']

    need = list(dict.fromkeys(['date'] + pca_input + [target_col]))
    sub  = df[need].dropna().reset_index(drop=True)
    n    = len(sub)
    init = int(n * TRAIN_FRAC)

    if n < 20 or init < 5:
        print(f'  {label}: insufficient data (n={n})')
        return None

    y       = sub[target_col].values
    dates   = sub['date'].values

    # Model: joint PCA (macro + sentiment)
    avail_s  = [c for c in pca_input if c in sub.columns]
    fac_m, _, _, nc_m = run_pca(sub, avail_s, var_threshold=PCA_VAR_MACRO)
    fac_m.columns = [f'JPC{i+1}' for i in range(nc_m)]
    X_model = sm.add_constant(fac_m.values)

    # Baseline: macro FAVAR (macro regressors only)
    fac_b, _, _, nc_b = run_pca(sub, mc_avail, var_threshold=PCA_VAR_MACRO)
    fac_b.columns = [f'BPC{i+1}' for i in range(nc_b)]
    X_base  = sm.add_constant(fac_b.values)

    raw_m, raw_b = [], []
    for t in range(init, n):
        try:
            mm = sm.OLS(y[:t], X_model[:t]).fit()
            mb = sm.OLS(y[:t], X_base[:t]).fit()
            raw_m.append((t, float(y[t]), float(mm.predict(X_model[t:t+1])[0])))
            raw_b.append((t, float(y[t]), float(mb.predict(X_base[t:t+1])[0])))
        except Exception:
            pass

    if len(raw_m) < 5:
        print(f'  {label}: too few predictions ({len(raw_m)})')
        return None

    rmse_m  = np.sqrt(np.mean([(y - yh)**2 for _, y, yh in raw_m]))
    rmse_b  = np.sqrt(np.mean([(y - yh)**2 for _, y, yh in raw_b]))
    improv  = (rmse_b - rmse_m) / rmse_b * 100
    dm_t, dm_p = diebold_mariano_test(raw_b, raw_m)

    oos_start = pd.Timestamp(dates[init]).date()
    oos_end   = pd.Timestamp(dates[-1]).date()

    return {
        'label'    : label,
        'n_total'  : n,
        'n_pred'   : len(raw_m),
        'n_jpc'    : nc_m,
        'n_sent'   : len([c for c in sent_top + sent_tot if c in sub.columns]),
        'n_sent_sentences': None,  # filled by caller
        'oos_start': oos_start,
        'oos_end'  : oos_end,
        'rmse_m'   : rmse_m,
        'rmse_b'   : rmse_b,
        'improv'   : improv,
        'dm_t'     : dm_t,
        'dm_p'     : dm_p,
    }


def sig_star(p):
    if p is None or (isinstance(p, float) and np.isnan(p)): return ''
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''


# ── Load raw sentences ─────────────────────────────────────────────────────────

print('  Loading raw sentences...', flush=True)
df_all = load_sentences(SENTENCES)
print(f'  Total sentences: {len(df_all):,}')
print(f'  tense distribution:\n'
      f'    interpretive : {(df_all["tense"]=="interpretive").sum():,}\n'
      f'    descriptive  : {(df_all["tense"]=="descriptive").sum():,}')
commitment_true = df_all['commitment'].apply(
    lambda v: v is True or str(v).lower() == 'unconditional'
)
print(f'  commitment=True (unconditional): {commitment_true.sum():,}')


# ── Variant A: interpretive sentences only ────────────────────────────────────

print('\n  [A] Filtering to interpretive sentences only...', flush=True)
df_interp  = df_all[df_all['tense'] == 'interpretive'].copy()
print(f'  Sentences after filter: {len(df_interp):,}  '
      f'({len(df_interp)/len(df_all)*100:.1f}% of total)')
panel_a    = sentences_to_meeting_panel(df_interp)


# ── Variant B: exclude documents with any unconditional sentences ──────────────

print('\n  [B] Excluding documents with any unconditional sentences...', flush=True)

# Identify which (source, date, doc_type) tuples have any commitment=True sentence
df_all['_firm'] = df_all['commitment'].apply(
    lambda v: v is True or str(v).lower() == 'unconditional'
)
firm_docs = (
    df_all[df_all['_firm']]
    .groupby(['source', 'date', 'doc_type'])
    .size()
    .reset_index(name='n_firm')
    [['source', 'date', 'doc_type']]
)
firm_docs['_exclude'] = True

df_no_unconditional = df_all.merge(firm_docs, on=['source', 'date', 'doc_type'],
                                   how='left')
df_no_unconditional = df_no_unconditional[
    df_no_unconditional['_exclude'].isna()
].drop(columns=['_firm', '_exclude']).copy()

n_docs_excluded = len(firm_docs)
n_sent_excluded = len(df_all) - len(df_no_unconditional)
print(f'  Documents excluded: {n_docs_excluded}')
print(f'  Sentences excluded: {n_sent_excluded:,}  '
      f'(from excluded docs, not all were unconditional)')
print(f'  Sentences remaining: {len(df_no_unconditional):,}  '
      f'({len(df_no_unconditional)/len(df_all)*100:.1f}% of total)')

panel_b = sentences_to_meeting_panel(df_no_unconditional)


# ── Variant B2: drop unconditional sentences (not whole documents) ─────────────
# More surgical: only the 2,602 commitment=True sentences are excluded;
# the rest of each document's sentences still contribute to the aggregate.

print('\n  [B2] Dropping only unconditional sentences (sentence-level)...', flush=True)
df_no_firm_sents = df_all[~df_all['_firm']].drop(columns=['_firm']).copy()
n_firm_sents = df_all['_firm'].sum()
print(f'  Sentences dropped: {n_firm_sents:,}  '
      f'({n_firm_sents/len(df_all)*100:.1f}% of total)')
print(f'  Sentences remaining: {len(df_no_firm_sents):,}')
panel_b2 = sentences_to_meeting_panel(df_no_firm_sents)


# ── Baseline: original 4h_llm_all_sent (from existing merged CSV) ─────────────

print('\n  [Orig] Loading original merged panel...', flush=True)
panel_orig = pd.read_csv(INTER_CSV, parse_dates=['date'])


# ── Run DM tests ───────────────────────────────────────────────────────────────

print('\n  Running DM tests...', flush=True)
r_orig = run_dm_h1(panel_orig, 'Original 4h_llm_all_sent')
r_a    = run_dm_h1(panel_a,    'Variant A: interpretive only')
r_b    = run_dm_h1(panel_b,    'Variant B: excl unconditional docs')
r_b2   = run_dm_h1(panel_b2,   'Variant B2: drop unconditional sents')

if r_a:  r_a['n_sent_sentences']  = len(df_interp)
if r_b:  r_b['n_sent_sentences']  = len(df_no_unconditional)
if r_b2: r_b2['n_sent_sentences'] = len(df_no_firm_sents)

# ── Print results ──────────────────────────────────────────────────────────────

print('\n' + '=' * 100)
print('  SENTENCE FILTER VARIANTS  |  h=1 expanding DM test vs Macro FAVAR')
print('  Target: cum_delta_h1 = effective_rate[t+1] - effective_rate[t]')
print('  Training: expanding from 50%  |  Model: joint PCA (macro + sentiment)')
print('=' * 100)

hdr = (f'\n  {"Model":<36}  {"N_sent":>8}  {"n_pred":>6}  {"JPC":>4}  '
       f'{"mdl_RMSE":>9}  {"base_RMSE":>10}  {"improv%":>8}  {"DM-t":>7}  {"DM-p":>8}  {"sig":>3}')
print(hdr)
print('  ' + '-' * 96)

for r in [r_orig, r_a, r_b, r_b2]:
    if r is None:
        continue
    n_s  = f'{r["n_sent_sentences"]:,}' if r.get('n_sent_sentences') else 'original'
    dm_t = f'{r["dm_t"]:>7.3f}' if r['dm_t'] is not None and not np.isnan(r['dm_t']) else '    n/a'
    dm_p = f'{r["dm_p"]:.4f}'  if r['dm_p'] is not None and not np.isnan(r['dm_p']) else '   n/a'
    print(f'  {r["label"]:<36}  {n_s:>8}  {r["n_pred"]:>6}  {r["n_jpc"]:>4}  '
          f'{r["rmse_m"]:>9.4f}  {r["rmse_b"]:>10.4f}  {r["improv"]:>7.1f}%  '
          f'{dm_t}  {dm_p}  {sig_star(r["dm_p"]):>3}')

print('\n  Notes:')
print(f'  - Variant A uses only the {len(df_interp):,} interpretive sentences '
      f'({len(df_interp)/len(df_all)*100:.0f}% of corpus)')
print(f'  - Variant B excludes {n_docs_excluded} documents containing any unconditional sentence;')
print(f'    {n_sent_excluded:,} total sentences removed ({n_sent_excluded/len(df_all)*100:.1f}% of corpus)')
print(f'  - Variant B2 drops only the {n_firm_sents:,} unconditional sentences themselves;')
print(f'    the rest of each document still contributes to the aggregate.')
print(f'  - Variant B doc-level exclusion rate by type:')
print(f'    minutes=94%, statements=85%, press_conf_prepared=90%, speech=25%')
print(f'  - base_RMSE may differ slightly because fewer sentiment cols shift PCA')

# ── Save ───────────────────────────────────────────────────────────────────────
rows = [r for r in [r_orig, r_a, r_b, r_b2] if r is not None]
df_out = pd.DataFrame([{k: v for k, v in r.items() if k != 'n_sent_sentences'}
                        for r in rows])
out = root + r'\Taylor Rule\outputs\tense_commitment_variants.csv'
df_out.to_csv(out, index=False)
print(f'\n  Saved -> tense_commitment_variants.csv')
