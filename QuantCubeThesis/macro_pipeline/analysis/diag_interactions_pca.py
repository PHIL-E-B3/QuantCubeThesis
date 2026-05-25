"""Why do interaction terms hurt the joint PCA?"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm

root = r'C:\Users\Javier\OneDrive - HEC Paris\Documentos\QuantCubeThesis\QuantCubeThesis'

import sys; sys.path.insert(0, '.')
from config import TARGET_NEXT, MACRO_REGRESSORS

df = pd.read_csv(root + r'\Taylor Rule\outputs\intermediate\step0_merged_to_macro.csv',
                 parse_dates=['date'])
df = df[df['date'] >= '2007-01-01'].reset_index(drop=True)

mc = [c for c in MACRO_REGRESSORS if c in df.columns]
MACRO_INTER = ['vix', 'gdp', 'unemployment_gap', 'inflation_dev_from_target',
               'implied_ffr', 'cfnai']

# Build the three input sets
df2 = df.copy()
inter_cols = []
for mc_col in MACRO_INTER:
    if mc_col in df2.columns:
        iname = f'sent_total_x_{mc_col}'
        df2[iname] = df2['sent_total'] * df2[mc_col]
        inter_cols.append(iname)

sets = {
    'macro_only':   mc,
    'macro+total':  mc + ['sent_total'],
    'macro+total+inter': mc + ['sent_total'] + inter_cols,
}

print('=' * 70)
print('1. CORRELATION: interaction terms vs. variables already in PCA')
print('=' * 70)
sub = df2[mc + ['sent_total'] + inter_cols + [TARGET_NEXT]].dropna()
print(f'\n   N = {len(sub)}')
print(f'\n   {"Column":<35} {"r with sent_total":>18} {"r with its macro":>18} {"r with PC1*":>12}')
print('   ' + '-' * 85)

# Fit PCA on macro only to get PC1
X_mc = StandardScaler().fit_transform(sub[mc])
pc1 = PCA(n_components=1).fit_transform(X_mc)[:, 0]

for ic in inter_cols:
    mc_name = ic.replace('sent_total_x_', '')
    r_sent = sub['sent_total'].corr(sub[ic])
    r_mc   = sub[mc_name].corr(sub[ic]) if mc_name in sub.columns else np.nan
    r_pc1  = np.corrcoef(pc1, sub[ic].values)[0, 1]
    print(f'   {ic:<35} {r_sent:>18.3f} {r_mc:>18.3f} {r_pc1:>12.3f}')

print()
print('=' * 70)
print('2. VARIANCE EXPLAINED: how much NEW variance do interactions add?')
print('=' * 70)
for name, cols in sets.items():
    avail = [c for c in cols if c in sub.columns]
    X = StandardScaler().fit_transform(sub[avail])
    pca = PCA().fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n5 = cumvar[4] * 100   # variance in first 5 components
    n_90 = int(np.searchsorted(cumvar, 0.90)) + 1
    print(f'   {name:<25}  inputs={len(avail):>3}  '
          f'var@5comps={n5:.1f}%  comps@90%={n_90}')

print()
print('=' * 70)
print('3. CORRELATION: interaction terms vs. target residual')
print('   (after removing macro-only PCA factors)')
print('=' * 70)
X_mc_full = StandardScaler().fit_transform(sub[mc])
factors_mc = PCA(n_components=5).fit_transform(X_mc_full)
y = sub[TARGET_NEXT].values
X_reg = sm.add_constant(factors_mc)
resid_y = y - sm.OLS(y, X_reg).fit().predict(X_reg)
print(f'\n   Baseline R2 = {1 - resid_y.var()/y.var():.4f}   residual std = {resid_y.std():.4f}')
print(f'\n   {"Column":<35} {"r with target residual":>22}')
print('   ' + '-' * 60)
r_sent = np.corrcoef(sub['sent_total'].values, resid_y)[0, 1]
print(f'   {"sent_total":<35} {r_sent:>22.3f}')
for ic in inter_cols:
    r = np.corrcoef(sub[ic].values, resid_y)[0, 1]
    print(f'   {ic:<35} {r:>22.3f}')

print()
print('=' * 70)
print('4. OOS INSTABILITY: walk-forward component count vs training size')
print('=' * 70)
for name, cols in [('macro+total', mc + ['sent_total']),
                   ('macro+total+inter', mc + ['sent_total'] + inter_cols)]:
    avail = [c for c in cols if c in sub.columns]
    n_comp_history = []
    for t in range(100, len(sub), 5):
        Xs = StandardScaler().fit_transform(sub[avail].iloc[:t])
        cv = np.cumsum(PCA().fit(Xs).explained_variance_ratio_)
        n_comp_history.append(int(np.searchsorted(cv, 0.90)) + 1)
    print(f'   {name:<25}  comps@90% range: '
          f'min={min(n_comp_history)}  max={max(n_comp_history)}  '
          f'std={np.std(n_comp_history):.2f}')
