"""
step0_aggregate.py — Sentence-to-document aggregation (pipeline foundation).

load_sentences(path)         → raw sentence DataFrame from LLM output
rescale_sen(df, extreme_val) → replace ±2 with ±extreme_val
aggregate_to_document(df_sentences, agg_func, rescale_extreme) → doc-level df
load_macro()                 → Master_Macro meeting-date rows
merge_to_macro(doc_df)       → doc-level merged onto macro panel
"""

import json
import sys
import warnings
from pathlib import Path

import numpy  as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MASTER_CSV, TOPICS, EXCLUDE_TOPICS,
    TARGET_COL, TARGET_NEXT, INTER_DIR, DATE_ALIGN_LOG,
)
from utils import log_warn

warnings.filterwarnings('ignore')


# ── Sentence loader ───────────────────────────────────────────────────────────

def load_sentences(path) -> pd.DataFrame:
    """
    Load LLM-labelled sentences from:
      - A single merged JSON file (list of records), or
      - A directory of per-batch JSON files.
    Returns DataFrame with all label fields.
    """
    path = Path(path)
    if path.is_dir():
        records = []
        for f in sorted(path.glob('*.json')):
            with open(f, encoding='utf-8') as fh:
                data = json.load(fh)
            records.extend(data if isinstance(data, list) else [data])
    else:
        with open(path, encoding='utf-8') as f:
            records = json.load(f)

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # sen should be numeric
    df['sen']  = pd.to_numeric(df['sen'], errors='coerce')
    return df


# ── Rescaling ─────────────────────────────────────────────────────────────────

def rescale_sen(df: pd.DataFrame, extreme_val: float = 2.0) -> pd.DataFrame:
    """Replace sen == ±2 with ±extreme_val."""
    df = df.copy()
    df['sen'] = df['sen'].where(df['sen'] != 2, extreme_val)
    df['sen'] = df['sen'].where(df['sen'] != -2, -extreme_val)
    return df


# ── Aggregation function variants ─────────────────────────────────────────────

def _agg_transform(x: float, func: str) -> float:
    if func == 'sum' or func == 'linear':
        return x
    elif func == 'power':
        return np.sign(x) * (abs(x) ** 1.1)
    elif func == 'log':
        return np.sign(x) * np.log1p(abs(x))
    return x


# ── Main aggregation ──────────────────────────────────────────────────────────

def aggregate_to_document(df_sentences: pd.DataFrame,
                          agg_func: str = 'sum',
                          rescale_extreme: float = 2.0) -> pd.DataFrame:
    """
    Aggregate sentence-level labels to document level.

    Parameters
    ----------
    df_sentences : DataFrame with LLM labels
    agg_func     : 'sum' | 'power' | 'log'
    rescale_extreme : replace |sen|==2 with this value before aggregation

    Returns
    -------
    Document-level DataFrame indexed by (source, date, doc_type).
    """
    df = rescale_sen(df_sentences, rescale_extreme)

    # Filter boilerplate / no_topic
    def _keep(row):
        top = row.get('top', [])
        if isinstance(top, str):
            import ast
            try:
                top = ast.literal_eval(top)
            except Exception:
                top = [top]
        return not set(top).issubset(EXCLUDE_TOPICS)

    df = df[df.apply(_keep, axis=1)].copy()

    # Normalise top to a Python list
    def _to_list(val):
        if isinstance(val, list):
            return val
        import ast
        try:
            return ast.literal_eval(val)
        except Exception:
            return [str(val)]

    df['top_list'] = df['top'].apply(_to_list)

    rows = []
    for (source, date, doc_type), grp in df.groupby(['source', 'date', 'doc_type']):
        row = {'source': source, 'date': date, 'doc_type': doc_type}

        # Per-topic sentiment
        for topic in TOPICS:
            mask      = grp['top_list'].apply(lambda t: topic in t)
            raw_sum   = grp.loc[mask, 'sen'].sum()
            row[f'sent_{topic}'] = _agg_transform(raw_sum, agg_func)

        # Aggregate sentiment
        raw_total = grp['sen'].sum()
        row['sent_total'] = _agg_transform(raw_total, agg_func)
        row['sent_sd']    = grp['sen'].std()
        row['sent_var']   = grp['sen'].var()

        # Flags
        row['flag_elevated_wid']        = (grp['wid'] == 'elevated').sum()
        row['flag_skew_up']             = (grp['ris'] == 'skewed_upside').sum()
        row['flag_skew_down']           = (grp['ris'] == 'skewed_downside').sum()
        row['flag_unconditional_forward'] = int(
            ((grp['com'] == 'unconditional') & (grp['ten'] == 'forward')).any()
        )
        row['flag_conditional']         = (grp['com'] == 'conditional').sum()

        # Speaker types (relevant for speech doc_type)
        spk_col = 'speaker_type' if 'speaker_type' in grp.columns else None
        row['speaker_types_present'] = (
            set(grp[spk_col].dropna().tolist()) if spk_col else set()
        )

        rows.append(row)

    doc_df = pd.DataFrame(rows)
    doc_df['date'] = pd.to_datetime(doc_df['date'])

    # Aggregate speeches across meeting windows
    doc_df = _aggregate_speech_windows(doc_df)

    # Save intermediate
    doc_df.to_csv(INTER_DIR / 'step0_document_sentiment.csv', index=False)
    print(f'  Step 0: {len(doc_df)} document-level observations  '
          f'(agg={agg_func}, extreme_val={rescale_extreme})')
    return doc_df


def _aggregate_speech_windows(doc_df: pd.DataFrame) -> pd.DataFrame:
    """
    For speeches: pool all docs between consecutive FOMC meetings into one
    observation per meeting window, aligned to blackout_date.
    Non-speech doc_types pass through unchanged.
    """
    macro = _load_macro_dates()
    meetings = sorted(macro['meeting_date'].tolist())

    speech_mask = doc_df['doc_type'].isin(['speech'])
    speech_df   = doc_df[speech_mask].copy()
    other_df    = doc_df[~speech_mask].copy()

    if speech_df.empty:
        return doc_df

    sent_cols  = [c for c in doc_df.columns if c.startswith('sent_')]
    flag_cols  = [c for c in doc_df.columns if c.startswith('flag_')]
    agg_dict   = {c: 'sum' for c in sent_cols + flag_cols}
    agg_dict.update({'sent_sd': 'mean', 'sent_var': 'mean'})

    # Assign each speech row to a meeting window
    unmatched = []
    assigned  = []
    for _, row in speech_df.iterrows():
        dt = row['date']
        mt = None
        for i, meeting in enumerate(meetings):
            prev = meetings[i-1] if i > 0 else pd.Timestamp.min
            if prev < dt <= meeting:
                mt = meeting
                break
        if mt is None:
            unmatched.append(str(dt.date()))
        else:
            r = row.copy()
            r['meeting_window'] = mt
            assigned.append(r)

    if unmatched:
        with open(DATE_ALIGN_LOG, 'a', encoding='utf-8') as f:
            for d in unmatched:
                f.write(f'Speech {d} not in any FOMC window\n')

    if not assigned:
        return pd.concat([other_df], ignore_index=True)

    sp = pd.DataFrame(assigned)

    # Aggregate by meeting window
    spk_sets = sp.groupby('meeting_window')['speaker_types_present'].agg(
        lambda x: set.union(*x) if len(x) > 0 else set()
    )

    num_agg = sp.groupby('meeting_window')[
        [c for c in agg_dict if c in sp.columns]
    ].agg(agg_dict)
    num_agg['doc_type'] = 'speech'
    num_agg['speaker_types_present'] = spk_sets
    num_agg = num_agg.reset_index().rename(columns={'meeting_window': 'date'})
    num_agg['source'] = 'speeches_aggregated'

    return pd.concat([other_df, num_agg], ignore_index=True).sort_values('date')


def _load_macro_dates() -> pd.DataFrame:
    df = pd.read_csv(MASTER_CSV, parse_dates=['date'])
    return df[df['event_type'] == 'meeting_date'][
        ['date']
    ].rename(columns={'date': 'meeting_date'}).sort_values('meeting_date')


# ── Macro loader ──────────────────────────────────────────────────────────────

def load_macro() -> pd.DataFrame:
    """Load Master_Macro, keep meeting_date rows, build target t+1."""
    df = pd.read_csv(MASTER_CSV, parse_dates=['date'])
    df = df[[c for c in df.columns if not c.startswith('Unnamed')]]
    df = df[df['event_type'] == 'meeting_date'].copy()
    df = df.sort_values('date').reset_index(drop=True)
    df[TARGET_NEXT] = df[TARGET_COL].shift(-1)
    return df


# ── Merge to macro panel ──────────────────────────────────────────────────────

def merge_to_macro(doc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge document-level sentiment onto Master_Macro meeting-date rows
    using backward as-of: each document aligned to the nearest meeting
    on or after its date.
    """
    macro  = load_macro()
    doc_df = doc_df.copy()
    doc_df['date'] = pd.to_datetime(doc_df['date'])

    sent_cols = [c for c in doc_df.columns
                 if c.startswith('sent_') or c.startswith('flag_')]

    # For each doc_type, aggregate to meeting_date level
    merged_pieces = []
    for doc_type, grp in doc_df.groupby('doc_type'):
        grp_s = grp[['date'] + sent_cols].sort_values('date')
        tmp   = pd.merge_asof(
            macro[['date']].sort_values('date').reset_index(),
            grp_s,
            on='date',
            direction='backward',
        )
        tmp = tmp.set_index('index').sort_index()
        tmp.columns = [f'{c}_{doc_type}' if c in sent_cols else c
                       for c in tmp.columns]
        merged_pieces.append(tmp[[c for c in tmp.columns if c != 'date']])

    result = macro.copy()
    for piece in merged_pieces:
        result = result.join(piece, how='left')

    # Also add combined (all doc_types pooled) sent_ columns
    all_sent = doc_df[['date'] + sent_cols].groupby('date').sum().reset_index()
    result = pd.merge_asof(
        result.sort_values('date'),
        all_sent.sort_values('date'),
        on='date', direction='backward',
        suffixes=('', '_combined')
    )
    result = result.sort_values('date').reset_index(drop=True)
    result.to_csv(INTER_DIR / 'step0_merged_to_macro.csv', index=False)
    return result
