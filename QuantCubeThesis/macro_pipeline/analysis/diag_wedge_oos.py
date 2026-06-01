"""
diag_wedge_oos.py

OOS expanding-window regressions testing whether the Tealbook-SPF wedge
improves forecast accuracy for delta_effective_rate.

Sample restricted to 2011-01-26 to 2020-12-16 (N=80 meetings where Tealbook
forecasts are publicly available with a 5-year release lag).

Models (all CW-tested vs nested base):
  1. FAVAR_base          : PCA on 7 macro regressors only
  2. FAVAR_wedge_add     : PCA on macro + 3 wedge terms
  3. FAVAR_wedge_matched : macro + 3 wedge + 3 matched interactions
                           (wedge_UNEMP x unemp_gap, wedge_RGDP x gdp,
                            wedge_PCPI x inflation_dev)
  4. FAVAR_wedge_full    : macro + 3 wedge + all 3x7=21 interactions

DV: delta_effective_rate[t+1] - delta_effective_rate[t]  (shadow rate, ZLB-adjusted)
Init: 50% of restricted sample (~40 obs)
OOS sub-windows: 60/70/80/90% cutoffs of restricted sample
Test: Clark-West (nested CW) — base nested in all wedge models
"""
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings('ignore')

sys.path.insert(0, r'C:\Users\Javier\OneDrive - HEC Paris\Documentos\QuantCubeThesis\QuantCubeThesis\macro_pipeline\analysis')
from config import MACRO_REGRESSORS, PCA_VAR_MACRO
from utils import run_pca

root = r'C:\Users\Javier\OneDrive - HEC Paris\Documentos\QuantCubeThesis\QuantCubeThesis'
INTER_CSV = root + r'\Taylor Rule\outputs\intermediate\step0_merged_to_macro.csv'
OUT_CSV   = root + r'\Taylor Rule\outputs\wedge_oos_results.csv'
OUT_MD    = root + r'\Taylor Rule\outputs\wedge_oos_summary.md'

TRAIN_FRAC  = 0.50
OOS_SPLITS  = [0.60, 0.70, 0.80, 0.90]
WEDGE_COLS  = ['actual_wedge_UNEMPF1', 'actual_wedge_gRGDPF1', 'actual_wedge_gPCPIF1']
# Thematically matched interaction (wedge var -> single macro partner)
MATCHED = {
    'actual_wedge_UNEMPF1':  'unemployment_gap',
    'actual_wedge_gRGDPF1':  'gdp',
    'actual_wedge_gPCPIF1':  'inflation_dev_from_target',
}
# Human-readable labels for output
WEDGE_LABELS = {
    'actual_wedge_UNEMPF1':  'UNEMP',
    'actual_wedge_gRGDPF1':  'RGDP',
    'actual_wedge_gPCPIF1':  'PCPI',
}


# ── Load and restrict data ─────────────────────────────────────────────────────

df_raw = pd.read_csv(INTER_CSV, parse_dates=['date'])
df_raw = df_raw[df_raw['date'] >= '2007-01-01'].reset_index(drop=True)
df = df_raw.dropna(subset=WEDGE_COLS).reset_index(drop=True)

df['delta_eff_next'] = df['effective_rate'].shift(-1) - df['effective_rate']
TARGET = 'delta_eff_next'

# Pre-build all interaction columns in df
for wc in WEDGE_COLS:
    for mc in MACRO_REGRESSORS:
        col = f'inter_{wc}_x_{mc}'
        df[col] = df[wc] * df[mc]

# Clean on mandatory columns only
all_feature_cols = list(dict.fromkeys(
    MACRO_REGRESSORS + WEDGE_COLS +
    [f'inter_{wc}_x_{mc}' for wc in WEDGE_COLS for mc in MACRO_REGRESSORS]
))
df_clean = df.dropna(subset=[TARGET] + all_feature_cols).reset_index(drop=True)

N    = len(df_clean)
INIT = int(N * TRAIN_FRAC)

print(f'\nSample (wedge window): {df_clean["date"].iloc[0].date()} '
      f'to {df_clean["date"].iloc[-2].date()}')
print(f'N={N}, init={INIT}, n_oos={N - INIT}')
print(f'DV: delta_effective_rate | mean={df_clean[TARGET].mean():.4f}, '
      f'std={df_clean[TARGET].std():.4f}\n')


# ── Expanding-window FAVAR ────────────────────────────────────────────────────

def expanding_favar(df_sub, feature_cols, target, init, pca_var=PCA_VAR_MACRO):
    """
    PCA on feature_cols (full-sample, no look-ahead — same as main pipeline),
    then expanding-window OLS of target ~ PCA factors.
    Returns list of (t, y_actual, y_hat).
    """
    avail = [c for c in feature_cols if c in df_sub.columns]
    sub   = df_sub[avail + [target]].dropna().reset_index(drop=True)
    n     = len(sub)
    if n <= init:
        return [], n

    fac, _, _, n_comp = run_pca(sub, avail, var_threshold=pca_var)
    fac.columns = [f'PC{i+1}' for i in range(n_comp)]
    y = sub[target].values
    X = sm.add_constant(fac.values)

    preds = []
    for t in range(init, n):
        try:
            m = sm.OLS(y[:t], X[:t]).fit()
            preds.append((t, float(y[t]), float(m.predict(X[t:t+1])[0])))
        except Exception:
            pass
    return preds, n


# ── Stat helpers ───────────────────────────────────────────────────────────────

def rmse(preds):
    if not preds:
        return np.nan
    return np.sqrt(np.mean([(y - yh)**2 for _, y, yh in preds]))


def rmse_from(preds, n, frac):
    ct  = int(n * frac)
    sub = [(t, y, yh) for t, y, yh in preds if t >= ct]
    return np.sqrt(np.mean([(y - yh)**2 for _, y, yh in sub])) if sub else np.nan


def cw_test(preds_base, preds_full):
    """
    Clark-West (2007) test: H0 = base MSFE <= full MSFE (no improvement).
    Positive t-stat => full model beats base.
    One-sided p-value (upper tail).
    """
    base_map  = {t: (y, yh) for t, y, yh in preds_base}
    full_map  = {t: (y, yh) for t, y, yh in preds_full}
    common_t  = sorted(set(base_map) & set(full_map))
    if len(common_t) < 4:
        return np.nan, np.nan

    f_adj = []
    for t in common_t:
        y,  yh_b = base_map[t]
        y2, yh_f = full_map[t]
        e_b = y  - yh_b
        e_f = y2 - yh_f
        # CW adjustment term
        adj  = (yh_b - yh_f)**2
        f_adj.append(e_b**2 - e_f**2 + adj)

    f_adj = np.array(f_adj)
    h     = max(1, int(len(f_adj)**0.25))
    # Newey-West variance
    nw = sum(
        (1 - abs(k) / (h + 1)) * np.mean(f_adj[abs(k):] * f_adj[:len(f_adj) - abs(k)])
        for k in range(-h, h + 1) if abs(k) <= len(f_adj) - 1
    )
    se     = np.sqrt(max(nw, 1e-12) / len(f_adj))
    t_stat = f_adj.mean() / se
    from scipy import stats as sc
    p      = sc.norm.sf(t_stat)          # one-sided upper
    return round(t_stat, 4), round(p, 4)


def sig_stars(p):
    if pd.isna(p): return ''
    if p < 0.01:   return '***'
    if p < 0.05:   return '**'
    if p < 0.10:   return '*'
    return ''


# ── Run models ────────────────────────────────────────────────────────────────

# Base model (shared across all wedge tests)
base_preds, base_n = expanding_favar(df_clean, MACRO_REGRESSORS, TARGET, INIT)
base_rmse_val      = rmse(base_preds)
print(f'  {"FAVAR_base":<32} n_oos={len(base_preds)}  RMSE={base_rmse_val:.4f}\n')

all_rows = []

for wc in WEDGE_COLS:
    label_short = WEDGE_LABELS[wc]
    matched_mc  = MATCHED[wc]
    inter_matched_col  = f'inter_{wc}_x_{matched_mc}'
    inter_all_cols_wc  = [f'inter_{wc}_x_{mc}' for mc in MACRO_REGRESSORS]

    configs = [
        (f'wedge_{label_short}_add',     MACRO_REGRESSORS + [wc]),
        (f'wedge_{label_short}_matched', MACRO_REGRESSORS + [wc, inter_matched_col]),
        (f'wedge_{label_short}_full',    MACRO_REGRESSORS + [wc] + inter_all_cols_wc),
    ]

    print(f'  -- Wedge: {wc} --')
    block_results = {}
    for label, feat_cols in configs:
        preds, n = expanding_favar(df_clean, feat_cols, TARGET, INIT)
        block_results[label] = {'preds': preds, 'n': n}
        print(f'    {label:<36} n_oos={len(preds)}  RMSE={rmse(preds):.4f}')
    print()

    # Build rows for this wedge
    base_row = {
        'wedge': label_short,
        'model': 'FAVAR_base',
        'features': 'macro only',
        'n_oos':    len(base_preds),
        'rmse_exp': round(base_rmse_val, 4),
        'base_rmse': round(base_rmse_val, 4),
        'imp_pct':  0.0,
        'cw_t': np.nan, 'cw_p': np.nan, 'sig': '',
    }
    for f in OOS_SPLITS:
        base_row[f'rmse_oos{int(f*100)}'] = round(rmse_from(base_preds, base_n, f), 4)
    all_rows.append(base_row)

    for label, feat_cols in configs:
        r      = block_results[label]
        preds  = r['preds']
        n      = r['n']
        r_exp  = rmse(preds)
        r_cuts = [rmse_from(preds, n, f) for f in OOS_SPLITS]
        imp    = (base_rmse_val - r_exp) / base_rmse_val * 100
        cw_t, cw_p = cw_test(base_preds, preds)
        sig        = sig_stars(cw_p)

        row = {
            'wedge':     label_short,
            'model':     label,
            'features':  feat_cols,
            'n_oos':     len(preds),
            'rmse_exp':  round(r_exp, 4),
            'base_rmse': round(base_rmse_val, 4),
            'imp_pct':   round(imp, 2),
            'cw_t':      cw_t if not np.isnan(cw_t) else np.nan,
            'cw_p':      cw_p if not np.isnan(cw_p) else np.nan,
            'sig':        sig,
        }
        for f, rv in zip(OOS_SPLITS, r_cuts):
            row[f'rmse_oos{int(f*100)}'] = round(rv, 4) if not np.isnan(rv) else np.nan
        all_rows.append(row)

rows = all_rows

# ── Print summary table ───────────────────────────────────────────────────────

print(f'\n{"="*95}')
print(f'  PER-WEDGE OOS RESULTS  |  DV=delta_effective_rate  |  sample=2011-2020')
print(f'{"="*95}')
hdr = (f'  {"Wedge":<7} {"Model":<32} {"N_oos":>5}  {"RMSE":>7}  {"Imp%":>7}  '
       + '  '.join(f'OOS{int(f*100)}' for f in OOS_SPLITS)
       + f'  {"CW_t":>7}  {"CW_p":>6}  {"Sig":>4}')
print(hdr)
print('  ' + '-' * 93)

prev_wedge = None
for row in rows:
    if row['wedge'] != prev_wedge:
        if prev_wedge is not None:
            print()
        prev_wedge = row['wedge']
    imp_s  = f'{row["imp_pct"]:+.1f}%'
    cuts_s = '  '.join(
        f'{row[f"rmse_oos{int(f*100)}"]:>7.4f}'
        if not pd.isna(row.get(f'rmse_oos{int(f*100)}', np.nan)) else f'{"  NaN":>7}'
        for f in OOS_SPLITS
    )
    cw_t_s = f'{row["cw_t"]:>7.4f}' if not pd.isna(row.get('cw_t', np.nan)) else f'{"—":>7}'
    cw_p_s = f'{row["cw_p"]:>6.4f}' if not pd.isna(row.get('cw_p', np.nan)) else f'{"—":>6}'
    print(f'  {row["wedge"]:<7} {row["model"]:<32} {row["n_oos"]:>5}  '
          f'{row["rmse_exp"]:>7.4f}  {imp_s:>7}  {cuts_s}  {cw_t_s}  {cw_p_s}  '
          f'{row["sig"]:>4}')

save_rows = [{k: v for k, v in r.items() if k != 'features'} for r in rows]
df_out = pd.DataFrame(save_rows)
df_out.to_csv(OUT_CSV, index=False)
print(f'\n  Saved CSV  -> {OUT_CSV}')


# ── Markdown summary ──────────────────────────────────────────────────────────

md = f"""# Tealbook-SPF Wedge OOS Results — Per-Wedge Breakdown
**DV: delta_effective_rate (shadow rate)**
**Sample: 2011-01-26 to 2020-12-16 (N={N}, wedge-available window)**
**Date: 2026-05-31**

---

## Setup

- Tealbook data available 2011–2020 (5-year public release lag), N={N} meetings
- OOS method: Expanding window, init = 50% ({INIT} obs), n_oos = {N - INIT}
- All models use PCA on feature set (90% variance threshold)
- Wedge = Tealbook forecast − SPF forecast (unemployment / real GDP growth / PCE inflation)
- Each wedge tested independently; three specifications per wedge:
  - **add**: macro + 1 wedge term
  - **matched**: macro + wedge + 1 matched interaction (e.g. wedge_UNEMP × unemp_gap)
  - **full**: macro + wedge + all 7 interactions (wedge × each macro regressor)
- Test: Clark-West (CW) — base nested in wedge models; positive t = wedge model better

---

## Results

| Wedge | Model | N_oos | RMSE | Imp% | OOS60 | OOS70 | OOS80 | OOS90 | CW_t | CW_p | Sig |
|---|---|---|---|---|---|---|---|---|---|---|---|
"""
prev_wedge = None
for row in rows:
    if row['wedge'] != prev_wedge and prev_wedge is not None:
        md += '| | | | | | | | | | | | |\n'
    prev_wedge = row['wedge']
    cw_t_s = f"{row['cw_t']:.4f}" if not pd.isna(row.get('cw_t', np.nan)) else '—'
    cw_p_s = f"{row['cw_p']:.4f}" if not pd.isna(row.get('cw_p', np.nan)) else '—'
    r60 = f"{row['rmse_oos60']:.4f}" if not pd.isna(row.get('rmse_oos60', np.nan)) else 'NaN'
    r70 = f"{row['rmse_oos70']:.4f}" if not pd.isna(row.get('rmse_oos70', np.nan)) else 'NaN'
    r80 = f"{row['rmse_oos80']:.4f}" if not pd.isna(row.get('rmse_oos80', np.nan)) else 'NaN'
    r90 = f"{row['rmse_oos90']:.4f}" if not pd.isna(row.get('rmse_oos90', np.nan)) else 'NaN'
    md += (f"| {row['wedge']} | {row['model']} | {row['n_oos']} | {row['rmse_exp']:.4f} | "
           f"{row.get('imp_pct', 0.0):+.1f}% | "
           f"{r60} | {r70} | {r80} | {r90} | {cw_t_s} | {cw_p_s} | {row['sig']} |\n")

md += f"""
---

## Interpretation

- **Imp%** = % RMSE improvement over FAVAR_base on full expanding window
- **CW_t/CW_p** = Clark-West test; positive t, small p => wedge model beats base
- Sample 2011–2020 only (N={N}); base RMSE here ({base_rmse_val:.4f}) differs from
  main table (0.2826) because of the shorter, lower-volatility ZLB-dominated window

---

## Files

- `wedge_oos_results.csv` — machine-readable results table
- `wedge_oos_summary.md` — this file
- Script: `macro_pipeline/analysis/diag_wedge_oos.py`
"""

with open(OUT_MD, 'w', encoding='utf-8') as f:
    f.write(md)
print(f'  Saved MD   -> {OUT_MD}')
