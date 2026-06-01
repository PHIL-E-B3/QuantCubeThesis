"""
diag_pca_compression.py

Checks whether LLM sentiment variance sits in retained vs discarded PCA components
in the joint macro+sentiment PCA used by the Joint PCA FAVAR architecture.

Outputs:
  1. Variance explained per PC, retention threshold marked
  2. Sentiment loading magnitudes per PC (retained vs discarded)
  3. % of total sentiment "loading energy" captured by retained vs discarded PCs
  4. Same analysis for the statements doc-type (the key comparison case)
"""
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from config import MACRO_REGRESSORS, PCA_VAR_MACRO

root      = r'C:\Users\Javier\OneDrive - HEC Paris\Documentos\QuantCubeThesis\QuantCubeThesis'
INTER_CSV = root + r'\Taylor Rule\outputs\intermediate\step0_merged_to_macro.csv'

H          = 1
START_DATE = '2007-01-01'
TOPICS     = ['inflation', 'labor_market', 'economic_activity',
              'financial_conditions', 'monetary_policy', 'macro']


def pca_compression_report(df_full, sent_cols, label):
    """
    Run joint PCA on [macro + sentiment] and report where sentiment variance lands.
    """
    mc_cols  = [c for c in MACRO_REGRESSORS if c in df_full.columns]
    sc       = [c for c in sent_cols if c in df_full.columns]
    all_cols = mc_cols + sc

    need = list(dict.fromkeys(all_cols + ['cum_delta_h1']))
    need = [c for c in need if c in df_full.columns]
    s    = df_full[need].dropna().reset_index(drop=True)

    if len(s) < 30:
        print(f'  [{label}] Too few observations ({len(s)}), skipping.')
        return

    n_mac  = len(mc_cols)
    n_sent = len(sc)
    n_all  = len(all_cols)

    X      = s[all_cols].values
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    pca = PCA(random_state=42)
    pca.fit(X_sc)

    ev_ratio = pca.explained_variance_ratio_
    cum_var  = np.cumsum(ev_ratio)
    n_retain = int(np.searchsorted(cum_var, PCA_VAR_MACRO)) + 1

    # loadings: shape (n_components, n_features)
    # components_[k, j] = loading of feature j on PC k
    loadings = pca.components_   # (n_all, n_all)

    # sentiment loading magnitudes per PC
    sent_idx = list(range(n_mac, n_all))  # indices of sentiment cols in all_cols
    sent_load_sq = loadings[:, sent_idx] ** 2  # squared loadings for each PC

    # sum of squared loadings attributable to sentiment per PC
    sent_energy_per_pc = sent_load_sq.sum(axis=1)   # shape (n_all,)
    total_energy_per_pc = (loadings ** 2).sum(axis=1)  # = 1 for each PC (unit vectors)

    print(f'\n{"="*80}')
    print(f'  {label}  |  n={len(s)}  |  {n_mac} macro + {n_sent} sentiment cols  |  retain={n_retain} PCs @ {PCA_VAR_MACRO*100:.0f}%')
    print(f'{"="*80}')
    print(f'\n  {"PC":>4}  {"VarExpl":>8}  {"CumVar":>8}  {"SentLoadSq":>12}  {"SentShare%":>12}  {"Retained"}')
    print(f'  {"-"*70}')
    for k in range(n_all):
        retained = '<-- retained' if k < n_retain else ''
        print(f'  {k+1:>4}  {ev_ratio[k]:>8.4f}  {cum_var[k]:>8.4f}  '
              f'{sent_energy_per_pc[k]:>12.4f}  '
              f'{sent_energy_per_pc[k]/total_energy_per_pc[k]*100:>12.1f}%  {retained}')

    # Summary: % of total sentiment "energy" in retained vs discarded
    total_sent_energy = sent_energy_per_pc.sum()
    retained_sent_energy   = sent_energy_per_pc[:n_retain].sum()
    discarded_sent_energy  = sent_energy_per_pc[n_retain:].sum()

    print(f'\n  Sentiment loading energy summary:')
    print(f'    Retained  ({n_retain} PCs): {retained_sent_energy/total_sent_energy*100:>6.1f}%')
    print(f'    Discarded ({n_all-n_retain} PCs): {discarded_sent_energy/total_sent_energy*100:>6.1f}%')

    # Top 3 PCs by sentiment loading for context
    top3 = np.argsort(sent_energy_per_pc)[::-1][:3]
    print(f'\n  Top 3 PCs by sentiment loading:')
    for k in top3:
        tag = 'RETAINED' if k < n_retain else 'discarded'
        print(f'    PC{k+1:>2}  sentLoadSq={sent_energy_per_pc[k]:.4f}  '
              f'cumVar@this={cum_var[k]:.4f}  [{tag}]')

    # Per-sentiment-column breakdown on top PCs
    print(f'\n  Per-column loadings on first {min(n_retain+2, n_all)} PCs:')
    header = f'  {"Column":<35}' + ''.join(f'  PC{k+1:>2}' for k in range(min(n_retain+2, n_all)))
    print(header)
    print(f'  {"-"*len(header)}')
    for j, col in enumerate(sc):
        row = f'  {col:<35}'
        for k in range(min(n_retain+2, n_all)):
            row += f'  {loadings[k, n_mac+j]:>5.2f}'
        print(row)

    return {
        'n_retain': n_retain,
        'retained_sent_pct': round(retained_sent_energy/total_sent_energy*100, 1),
        'discarded_sent_pct': round(discarded_sent_energy/total_sent_energy*100, 1),
    }


# ── Load data ─────────────────────────────────────────────────────────────────

print('Loading base panel...', flush=True)
df_base = pd.read_csv(INTER_CSV, parse_dates=['date'])
df_base = df_base[df_base['date'] >= START_DATE].reset_index(drop=True)
df_base['cum_delta_h1'] = df_base['effective_rate'].shift(-H) - df_base['effective_rate']


# ── Case 1: All-docs LLM sentiment (the best-performing spec) ─────────────────

all_docs_cols = [f'sent_total'] + [f'sent_{t}' for t in TOPICS]
pca_compression_report(df_base, all_docs_cols, 'LLM — All docs (total + 6 topics)')


# ── Case 2: Statements only ───────────────────────────────────────────────────

stmt_cols = [f'sent_total_statement'] + [f'sent_{t}_statement' for t in TOPICS]
pca_compression_report(df_base, stmt_cols, 'LLM — Statements only')


# ── Case 3: Minutes only ──────────────────────────────────────────────────────

min_cols = [f'sent_total_minutes'] + [f'sent_{t}_minutes' for t in TOPICS]
pca_compression_report(df_base, min_cols, 'LLM — Minutes only')


print('\nDone.')
