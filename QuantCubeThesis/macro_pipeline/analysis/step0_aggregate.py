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
    MASTER_CSV, SHADOW_RATE_CSV, CFNAI_CSV, KUTTNER_IMPLIED_CSV,
    TOPICS, EXCLUDE_TOPICS, TARGET_COL, TARGET_NEXT, INTER_DIR, DATE_ALIGN_LOG,
)
from utils import log_warn

warnings.filterwarnings('ignore')

SENTIMENT_TO_NUMERIC = {
    'strongly_hawkish':  2,
    'hawkish':           1,
    'neutral':           0,
    'dovish':           -1,
    'strongly_dovish':  -2,
    'na':                0,
}

# Old abbreviated field names → full names (for legacy JSON files)
_FIELD_REMAP = {
    'sen': 'sentiment', 'top': 'topic', 'ten': 'tense',
    'hor': 'horizon',   'com': 'commitment', 'ris': 'risk', 'wid': 'width',
}


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
    # Rename abbreviated field names to full names if present
    df = df.rename(columns={k: v for k, v in _FIELD_REMAP.items() if k in df.columns})
    # Normalise compact YYYYMMDD strings (e.g. '20040128') to ISO before parsing
    df['date'] = df['date'].apply(
        lambda v: f'{str(v)[:4]}-{str(v)[4:6]}-{str(v)[6:8]}'
        if isinstance(v, str) and len(v) == 8 and str(v).isdigit() else v
    )
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # Convert semantic sentiment strings to numeric (also handles already-numeric values)
    if 'sentiment' in df.columns:
        df['sentiment'] = df['sentiment'].map(
            lambda v: SENTIMENT_TO_NUMERIC.get(str(v).lower().strip(), 0)
            if not isinstance(v, (int, float)) else v
        )
    return df


# ── Rescaling ─────────────────────────────────────────────────────────────────

def rescale_sen(df: pd.DataFrame, extreme_val: float = 2.0) -> pd.DataFrame:
    """Replace sentiment == ±2 with ±extreme_val."""
    df = df.copy()
    df['sentiment'] = df['sentiment'].where(df['sentiment'] != 2, extreme_val)
    df['sentiment'] = df['sentiment'].where(df['sentiment'] != -2, -extreme_val)
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
        top = row.get('topic', [])
        if isinstance(top, str):
            import ast
            try:
                top = ast.literal_eval(top)
            except Exception:
                top = [top]
        return not set(top).issubset(EXCLUDE_TOPICS)

    df = df[df.apply(_keep, axis=1)].copy()

    # Normalise topic to a Python list
    def _to_list(val):
        if isinstance(val, list):
            return val
        import ast
        try:
            return ast.literal_eval(val)
        except Exception:
            return [str(val)]

    df['topic_list'] = df['topic'].apply(_to_list)

    rows = []
    for (source, date, doc_type), grp in df.groupby(['source', 'date', 'doc_type']):
        row = {'source': source, 'date': date, 'doc_type': doc_type}

        # Per-topic sentiment + within-topic dispersion
        for topic in TOPICS:
            mask      = grp['topic_list'].apply(lambda t: topic in t)
            topic_grp = grp.loc[mask, 'sentiment']
            raw_sum   = topic_grp.sum()
            row[f'sent_{topic}']    = _agg_transform(raw_sum, agg_func)
            row[f'sent_{topic}_sd'] = topic_grp.std() if mask.sum() > 1 else 0.0

        # Aggregate sentiment
        raw_total = grp['sentiment'].sum()
        row['sent_total'] = _agg_transform(raw_total, agg_func)
        row['sent_sd']    = grp['sentiment'].std()
        row['sent_var']   = grp['sentiment'].var()

        # Flags
        row['flag_elevated_wid']        = (grp['width'] == 'elevated').sum()
        row['flag_skew_up']             = (grp['risk'] == 'skewed_upside').sum()
        row['flag_skew_down']           = (grp['risk'] == 'skewed_downside').sum()
        # commitment is bool (new LLM output: True=firm) or str (legacy: 'unconditional')
        _firm = grp['commitment'].apply(
            lambda v: v is True or str(v).lower() == 'unconditional'
        )
        row['flag_unconditional_forward'] = int(
            (_firm & (grp['tense'] == 'interpretive')).any()
        )
        _cond = grp['commitment'].apply(lambda v: str(v).lower() == 'conditional')
        row['flag_conditional']         = _cond.sum()

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


# ── Shadow rate loader ────────────────────────────────────────────────────────

def _load_shadow_rate() -> pd.DataFrame:
    """
    Load Wu-Xia shadow rate CSV. Returns DataFrame with columns [date, shadow_rate].
    Dates are end-of-month; we forward-fill onto FOMC meeting dates via merge_asof.
    """
    if not SHADOW_RATE_CSV.exists():
        return pd.DataFrame(columns=['date', 'shadow_rate'])
    df = pd.read_csv(SHADOW_RATE_CSV, sep=';', header=0,
                     usecols=[0, 2], names=['date', 'shadow_rate'], skiprows=1)
    df = df.dropna(subset=['shadow_rate'])
    # Parse "Jan-60" format; Python's %y treats 00-68 as 2000-2068, so fix post-2030
    df['date'] = pd.to_datetime(df['date'] + '-01', format='%b-%y-%d', errors='coerce')
    df['date'] = df['date'].apply(
        lambda d: d.replace(year=d.year - 100) if pd.notna(d) and d.year > 2030 else d
    )
    return df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)


# ── Macro loader ──────────────────────────────────────────────────────────────

def load_macro() -> pd.DataFrame:
    """Load Master_Macro, keep meeting_date rows, build effective_rate and target t+1."""
    df = pd.read_csv(MASTER_CSV, parse_dates=['date'])
    df = df[[c for c in df.columns if not c.startswith('Unnamed')]]
    df = df[df['event_type'] == 'meeting_date'].copy()
    n_before = len(df)
    df = df.drop_duplicates(subset=['date'])
    if len(df) < n_before:
        log_warn(f'load_macro: dropped {n_before - len(df)} duplicate meeting-date rows '
                 f'(kept {len(df)} unique dates). Check Master_Macro.csv for cartesian-join artefacts.')
    df = df.sort_values('date').reset_index(drop=True)

    # Build effective_rate: Wu-Xia shadow rate during ZLB (fed_funds_rate <= 0.25),
    # fed_funds_rate otherwise.  combine_first would silently use shadow_rate for
    # the entire sample; the explicit condition restricts substitution to the ZLB.
    shadow = _load_shadow_rate()
    if not shadow.empty:
        df = pd.merge_asof(df, shadow, on='date', direction='backward')
        zlb_mask = df['fed_funds_rate'] <= 0.25
        df['effective_rate'] = np.where(
            zlb_mask & df['shadow_rate'].notna(),
            df['shadow_rate'],
            df['fed_funds_rate'],
        )
        df = df.drop(columns=['shadow_rate'])
    else:
        df['effective_rate'] = df['fed_funds_rate']

    # Replace implied_ffr with Kuttner-computed level from fed funds futures.
    # The Master_Macro column is corrupt for meeting rows (contains surprise
    # shocks, not rate levels). kuttner_implied_ffr_v2.csv has the correct
    # post-meeting implied rate computed via the Kuttner (2001) formula.
    if KUTTNER_IMPLIED_CSV.exists():
        kut = pd.read_csv(KUTTNER_IMPLIED_CSV, parse_dates=['date'])
        kut = kut[kut['event_type'] == 'meeting_date'][['date', 'implied_ffr']]
        # Drop the corrupt column first, then merge the clean one
        if 'implied_ffr' in df.columns:
            df = df.drop(columns=[c for c in df.columns
                                   if 'implied_ffr' in c])
        df = pd.merge_asof(df.sort_values('date'),
                           kut.sort_values('date'),
                           on='date', direction='backward')
    else:
        log_warn('kuttner_implied_ffr_v2.csv not found — implied_ffr will be missing. '
                 'Run macro_pipeline/compute_kuttner_implied_ffr.py first.')

    # Merge CFNAI if available (used as macro variable for sent_macro interactions)
    if CFNAI_CSV.exists():
        cfnai = pd.read_excel(CFNAI_CSV, parse_dates=['observation_date'])
        cfnai = cfnai.rename(columns={'observation_date': 'date', 'CFNAI': 'cfnai'})
        cfnai = cfnai[['date', 'cfnai']].sort_values('date')
        df = pd.merge_asof(df, cfnai, on='date', direction='backward')

    df[TARGET_NEXT] = df[TARGET_COL].shift(-1)
    df['delta_rate_next'] = df[TARGET_COL].shift(-1) - df[TARGET_COL]
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
