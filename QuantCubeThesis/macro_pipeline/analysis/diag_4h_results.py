import pandas as pd

path = r'C:\Users\Javier\OneDrive - HEC Paris\Documentos\QuantCubeThesis\QuantCubeThesis\Taylor Rule\outputs_from200701\model_comparison.csv'
df = pd.read_csv(path)
print('Columns:', df.columns.tolist())

h = df[df['model_label'].str.startswith('4h_')].copy()
rest = df[~df['model_label'].str.startswith('4h_')].copy()
rest['spec'] = rest['model_label'].str.replace(r'_ev\d+\.\d+$', '', regex=True)
best_rest = rest.sort_values('oos_rmse_80').drop_duplicates('spec').head(5)

oos_cols = [c for c in df.columns if 'oos_rmse' in c]
cw_cols  = [c for c in df.columns if 'cw' in c]
show = ['model_label', 'n_obs', 'adj_r2'] + oos_cols + cw_cols

print()
print('=== 4h Joint PCA FAVAR specs ===')
print(h[show].to_string(index=False))

print()
print('=== Top 5 non-4h specs (by OOS80) ===')
print(best_rest[[c for c in show if c in best_rest.columns]].to_string(index=False))

print()
print('=== Baseline ===')
base = df[df['model_label'] == 'baseline_favar']
print(base[[c for c in show if c in base.columns]].to_string(index=False))
