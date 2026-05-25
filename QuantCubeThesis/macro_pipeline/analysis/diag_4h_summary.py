import pandas as pd

df = pd.read_csv(r'C:\Users\Javier\OneDrive - HEC Paris\Documentos\QuantCubeThesis\QuantCubeThesis\Taylor Rule\outputs\4h_joint_pca_summary.csv')
df = df.sort_values(['method', 'oos_rmse_80'])

cols = ['method', 'variant', 'n_inputs', 'n_comp', 'adj_r2', 'oos_rmse_80', 'cw_tstat', 'cw_pval']
print(df[cols].to_string(index=False))

print()
print('=== BEST PER METHOD ===')
for method, grp in df.groupby('method'):
    best = grp.iloc[0]
    cw_t = f'{best["cw_tstat"]:.3f}' if pd.notna(best['cw_tstat']) else 'n/a'
    cw_p = f'{best["cw_pval"]:.4f}' if pd.notna(best['cw_pval']) else 'n/a'
    print(f'  {method:<10} {best["variant"]:<22}  OOS80={best["oos_rmse_80"]:.4f}'
          f'  AdjR2={best["adj_r2"]:.4f}  CW-t={cw_t}  CW-p={cw_p}')
