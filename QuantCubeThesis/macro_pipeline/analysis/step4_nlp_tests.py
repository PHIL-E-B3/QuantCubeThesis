"""
step4_nlp_tests.py — NLP regressor tests (4a–4g).
Returns best_result dict (by Adj R²) for use in Steps 5–7.
"""

import sys
import warnings
from pathlib import Path

import numpy  as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    TOPICS, TARGET_NEXT, PCA_VAR_SENT, INTER_DIR, OUTPUTS_DIR,
)
from utils import (
    run_pca, fit_ols, vif, save_model_row, print_metrics, log_warn,
    clark_west_test,
)

warnings.filterwarnings('ignore')


def _augment(factor_df: pd.DataFrame, df: pd.DataFrame,
             extra_cols: list) -> pd.DataFrame:
    """Join factor_df with extra_cols from df, aligned on index."""
    avail = [c for c in extra_cols if c in df.columns]
    if not avail:
        return factor_df
    return factor_df.join(df[avail], how='left')


def _run_spec(label: str, df: pd.DataFrame, target: pd.Series,
              factor_df: pd.DataFrame, extra_cols: list,
              extreme_val: float = 2.0, label_prefix: str = '',
              model_csv=None, baseline_preds: list = None) -> dict:
    X = _augment(factor_df, df, extra_cols).dropna()
    y = target.loc[X.index].dropna()
    X = X.loc[y.index]
    result = fit_ols(y, X, label=f'{label_prefix}{label}_ev{extreme_val}')
    if result:
        if baseline_preds:
            cw_t, cw_p = clark_west_test(baseline_preds, result.get('_oos_preds', []))
            result['cw_tstat'] = cw_t
            result['cw_pval']  = cw_p
        print_metrics(result)
        save_model_row(result, csv_path=model_csv)
        result['spec'] = label
        result['extreme_val'] = extreme_val
    return result


def run_nlp_tests(df: pd.DataFrame, baseline_result: dict,
                  extreme_vals: list = None,
                  label_prefix: str = '',
                  model_csv=None) -> dict:
    """
    Run specifications 4a–4g across all extreme_val variants.
    Returns best_result (highest Adj R²).
    """
    if extreme_vals is None:
        extreme_vals = [1.0, 1.5, 2.0, 2.5, 3.0]

    if not baseline_result:
        log_warn('Step 4: no baseline result — cannot run NLP tests.')
        return {}

    if model_csv is None:
        model_csv = OUTPUTS_DIR / 'model_comparison.csv'

    factor_df     = baseline_result['_factor_df']
    baseline_preds = baseline_result.get('_oos_preds', [])
    target        = df[TARGET_NEXT]
    sent_cols     = [c for c in df.columns if c.startswith('sent_')]
    topic_sent    = [f'sent_{t}' for t in TOPICS if f'sent_{t}' in df.columns]

    all_results = []
    best = None

    for ev in extreme_vals:
        df_ev = df.copy()
        # Proportional rescale of pre-aggregated sent values
        scale = ev / 2.0
        for c in sent_cols:
            df_ev[c] = df_ev[c] * scale

        kw = dict(label_prefix=label_prefix, model_csv=model_csv,
                  baseline_preds=baseline_preds)

        # 4a: total sentiment
        r = _run_spec('4a_total_sent', df_ev, target, factor_df, ['sent_total'], ev, **kw)
        all_results.append(r)

        # 4b: dispersion — sent_sd/sent_var from LLM; consensus cols from dictionary runs
        r = _run_spec('4b_sent_sd',      df_ev, target, factor_df, ['sent_sd'],            ev, **kw)
        all_results.append(r)
        r = _run_spec('4b_sent_var',     df_ev, target, factor_df, ['sent_var'],            ev, **kw)
        all_results.append(r)
        r = _run_spec('4b_sd_var_joint', df_ev, target, factor_df, ['sent_sd', 'sent_var'], ev, **kw)
        all_results.append(r)
        consensus_cols = [c for c in df_ev.columns if c.endswith('_consensus')]
        if consensus_cols:
            r = _run_spec('4b_consensus', df_ev, target, factor_df, consensus_cols, ev, **kw)
            all_results.append(r)

        # 4b interactions: total dispersion × all macro (mirrors 4e structure)
        macro_raw = ['vix', 'gdp', 'unemployment_gap',
                     'inflation_dev_from_target', 'implied_ffr']
        inter_4b_sd = []
        if 'sent_sd' in df_ev.columns:
            for mc in macro_raw:
                if mc in df_ev.columns:
                    iname = f'inter_sd_x_{mc}'
                    df_ev[iname] = df_ev['sent_sd'] * df_ev[mc]
                    inter_4b_sd.append(iname)
        if inter_4b_sd:
            r = _run_spec('4b_sd_macro_interactions', df_ev, target, factor_df, inter_4b_sd, ev, **kw)
            all_results.append(r)

        # 4b matched topic-sd × macro interactions (mirrors 4f but with dispersion)
        # sent_macro_sd is excluded until economic sentiment indicator is added to Master_Macro
        matched_topic_sd = {
            'sent_inflation_sd':            'inflation_dev_from_target',
            'sent_labor_market_sd':         'unemployment_gap',
            'sent_economic_activity_sd':    'gdp',
            'sent_financial_conditions_sd': 'vix',
            'sent_monetary_policy_sd':      'implied_ffr',
            'sent_macro_sd':                'cfnai',
        }
        inter_4b_topic_sd = []
        for sd_col, mc in matched_topic_sd.items():
            if sd_col in df_ev.columns and mc in df_ev.columns:
                iname = f'inter_{sd_col}_x_{mc}'
                df_ev[iname] = df_ev[sd_col] * df_ev[mc]
                inter_4b_topic_sd.append(iname)
        if inter_4b_topic_sd:
            r = _run_spec('4b_topic_sd_matched_interactions', df_ev, target, factor_df, inter_4b_topic_sd, ev, **kw)
            all_results.append(r)

        # 4b consensus matched interactions: topic consensus × its paired macro (mirrors 4f structure)
        matched_consensus = {
            'sent_inflation_consensus':            'inflation_dev_from_target',
            'sent_labor_market_consensus':         'unemployment_gap',
            'sent_economic_activity_consensus':    'gdp',
            'sent_financial_conditions_consensus': 'vix',
            'sent_monetary_policy_consensus':      'implied_ffr',
        }
        inter_4b_con = []
        for con_col, mc in matched_consensus.items():
            if con_col in df_ev.columns and mc in df_ev.columns:
                iname = f'inter_{con_col}_x_{mc}'
                df_ev[iname] = df_ev[con_col] * df_ev[mc]
                inter_4b_con.append(iname)
        if inter_4b_con:
            r = _run_spec('4b_consensus_matched_interactions', df_ev, target, factor_df, inter_4b_con, ev, **kw)
            all_results.append(r)

        # 4c: all topic sentiments + VIF report
        if topic_sent:
            X_vif = _augment(factor_df, df_ev, topic_sent).dropna()
            vif_df = vif(X_vif[topic_sent])
            vif_df.to_csv(INTER_DIR / f'step4c_vif_ev{ev}.csv', index=False)
            flagged = vif_df[vif_df['VIF'] > 10]['feature'].tolist()
            if flagged:
                log_warn(f'4c (ev={ev}): VIF > 10 for {flagged}')
            r = _run_spec('4c_all_topics', df_ev, target, factor_df, topic_sent, ev, **kw)
            all_results.append(r)

        # 4d: PCA on topic sentiments
        ts_avail = [c for c in topic_sent if c in df_ev.columns]
        if len(ts_avail) >= 2:
            topic_factors, _, _, nc = run_pca(
                df_ev.dropna(subset=ts_avail), ts_avail,
                var_threshold=PCA_VAR_SENT
            )
            topic_factors.columns = [f'SPC{i+1}' for i in range(nc)]
            fdf_aug = factor_df.join(topic_factors, how='inner')
            y_4d    = target.loc[fdf_aug.index].dropna()
            X_4d    = fdf_aug.loc[y_4d.index]
            r = fit_ols(y_4d, X_4d, label=f'{label_prefix}4d_topic_pca_ev{ev}')
            if r:
                if baseline_preds:
                    cw_t, cw_p = clark_west_test(baseline_preds, r.get('_oos_preds', []))
                    r['cw_tstat'] = cw_t
                    r['cw_pval']  = cw_p
                r['spec'] = '4d'
                r['extreme_val'] = ev
                print_metrics(r)
                save_model_row(r, csv_path=model_csv)
                all_results.append(r)

        # 4e: total × macro interactions
        macro_raw = ['vix', 'gdp', 'unemployment_gap',
                     'inflation_dev_from_target', 'implied_ffr']
        inter_4e = {}
        for mc in macro_raw:
            if mc in df_ev.columns and 'sent_total' in df_ev.columns:
                df_ev[f'inter_total_x_{mc}'] = df_ev['sent_total'] * df_ev[mc]
                inter_4e[f'inter_total_x_{mc}'] = True
        r = _run_spec('4e_total_interactions', df_ev, target, factor_df,
                      list(inter_4e.keys()), ev, **kw)
        all_results.append(r)

        # 4f: matched topic × macro interactions.
        # sent_macro is paired with cfnai (broad activity index) to avoid
        # duplicating the gdp pairing already used for sent_economic_activity.
        matched = {
            'sent_economic_activity':  'gdp',
            'sent_financial_conditions': 'vix',
            'sent_labor_market':       'unemployment_gap',
            'sent_monetary_policy':    'implied_ffr',
            'sent_inflation':          'inflation_dev_from_target',
            'sent_macro':              'cfnai',
        }
        inter_4f = []
        for sent_col, macro_col in matched.items():
            if sent_col in df_ev.columns and macro_col in df_ev.columns:
                iname = f'inter_{sent_col}_x_{macro_col}'
                df_ev[iname] = df_ev[sent_col] * df_ev[macro_col]
                inter_4f.append(iname)
        r = _run_spec('4f_matched_interactions', df_ev, target, factor_df, inter_4f, ev, **kw)
        all_results.append(r)
        if r:
            df_ev[[c for c in inter_4f if c in df_ev.columns]].to_csv(
                INTER_DIR / f'step4f_matched_interactions_ev{ev}.csv'
            )

        # 4g: novelty (first difference of sentiment).
        # Deltas are computed from df (pre-rescaling) so the ev sweep doesn't
        # trivially rescale the novelty signal — each ev tests the same level
        # change in raw sentiment units.
        delta_cols = []
        for c in topic_sent + ['sent_total']:
            if c in df.columns:
                dname = f'delta_{c}'
                df_ev[dname] = df[c].diff()
                delta_cols.append(dname)
        r = _run_spec('4g_novelty', df_ev, target, factor_df, delta_cols, ev, **kw)
        all_results.append(r)

    # Find best model
    valid = [r for r in all_results if r and r.get('adj_r2') is not None]
    if valid:
        best = max(valid, key=lambda r: r.get('adj_r2', -np.inf))
        print(f'\n  Best Step 4 model: {best["model_label"]}  '
              f'Adj R2 = {best["adj_r2"]:.4f}')
        inter_csv = INTER_DIR / (
            f'step4_all_results_{label_prefix.rstrip("_")}.csv'
            if label_prefix else 'step4_all_results.csv'
        )
        pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')}
                      for r in valid]).to_csv(inter_csv, index=False)
    return best or {}
