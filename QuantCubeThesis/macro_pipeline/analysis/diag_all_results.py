import pandas as pd

base = r'C:\Users\Javier\OneDrive - HEC Paris\Documentos\QuantCubeThesis\QuantCubeThesis\Taylor Rule\outputs'

results = []

# LLM raw post-2007
df = pd.read_csv(base + r'\..\outputs_from200701\model_comparison.csv')
df['spec'] = df['model_label'].str.replace(r'_ev\d+\.\d+$', '', regex=True)
best = df.sort_values('oos_rmse_80').drop_duplicates('spec').iloc[0]
results.append(('LLM raw (standard pipeline)', best['model_label'], best['adj_r2'], best['oos_rmse_80']))

# LLM EWMA-10 post-2007
df = pd.read_csv(base + r'\..\outputs_from200701_ewma10\model_comparison.csv')
df['spec'] = df['model_label'].str.replace(r'_ev\d+\.\d+$', '', regex=True)
best = df.sort_values('oos_rmse_80').drop_duplicates('spec').iloc[0]
results.append(('LLM EWMA-10 (standard pipeline)', best['model_label'], best['adj_r2'], best['oos_rmse_80']))

# Dict raw post-2007
df = pd.read_csv(base + r'\dict_favar_summary_oos80_from200701.csv')
best = df.sort_values('best_oos_rmse').iloc[0]
results.append(('Dict raw (standard pipeline)', best['combo_label'] + ' / ' + best['best_spec'], best['best_adj_r2'], best['best_oos_rmse']))

# Dict EWMA-10 post-2007
df = pd.read_csv(base + r'\dict_favar_summary_oos80_from200701_ewma10.csv')
best = df.sort_values('best_oos_rmse').iloc[0]
results.append(('Dict EWMA-10 (standard pipeline)', best['combo_label'] + ' / ' + best['best_spec'], best['best_adj_r2'], best['best_oos_rmse']))

# Joint PCA FAVAR (from diagnostic)
results.append(('LLM joint PCA FAVAR',     'Macro + sent_total',   0.9591, 0.2548))
results.append(('Gardner joint PCA FAVAR', 'Macro + g_labor_mkt',  0.9473, 0.3504))
results.append(('Sharpe joint PCA FAVAR',  'Macro + s_net',        0.9669, 0.3964))
results.append(('Baseline (macro only)',   'PCA factors only',     0.9669, 0.5299))

print()
print(f'  {"Method":<35} {"Adj R2":>9} {"OOS80":>9}')
print('  ' + '-' * 55)
for method, spec, r2, oos in sorted(results, key=lambda x: x[3]):
    marker = '  <-- BEST' if oos == min(r[3] for r in results) else ''
    print(f'  {method:<35} {r2:>9.4f} {oos:>9.4f}{marker}')
