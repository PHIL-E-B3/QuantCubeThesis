"""
step_d_dictionary_runner.py
----------------------------
Re-computes Gardner and Sharpe dictionary sentiment scores for all FOMC
document types with two normalization strategies: 'wordcount' and 'zscore'.

Outputs one CSV per (dictionary, doc_type_combo, normalization) in:
    Taylor Rule/nlp_output/{Dictionary}_{doc_types}_{normalization}_nlp.csv

Each CSV columns:
    date, meeting_date, <dict_specific_score_columns>

Usage:
    python macro_pipeline/analysis/step_d_dictionary_runner.py
    python macro_pipeline/analysis/step_d_dictionary_runner.py --dict gardner --norm wordcount
"""

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path

import numpy  as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.Gardner import NLP as gardner_nlp
from src.training.Sharpe  import NLP as sharpe_nlp

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MASTER_CSV, STATEMENTS_DIR, MINUTES_DIR,
    SPEECHES_DIR, PRESS_CONF_DIR, SPEECH_SUBDIRS,
    NLP_DIR, DATE_ALIGN_LOG, WARNINGS_LOG,
)
from utils import log_warn

warnings.filterwarnings('ignore')


# ── Text loaders ──────────────────────────────────────────────────────────────

def _load_json_dir(folder: Path, date_field: str = 'date',
                   text_field: str = 'text') -> dict:
    """Return {date: text} from all JSON files in a folder."""
    out = {}
    if not folder.exists():
        return out
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith('.json'):
            continue
        try:
            with open(folder / fname, encoding='utf-8') as f:
                rec = json.load(f)
            dt   = pd.Timestamp(rec[date_field])
            text = rec.get(text_field, '')
            if not text and 'content' in rec:
                content = rec['content']
                text = '\n\n'.join(content) if isinstance(content, list) else content
            if text and len(text) > 50:
                out[dt] = text
        except Exception:
            pass
    return out


def load_statements() -> dict:
    return _load_json_dir(STATEMENTS_DIR, text_field='text')


def load_minutes() -> dict:
    out = {}
    for fname in sorted(os.listdir(MINUTES_DIR)):
        if not fname.endswith('.json'):
            continue
        try:
            with open(MINUTES_DIR / fname, encoding='utf-8') as f:
                rec = json.load(f)
            dt = pd.Timestamp(rec['date'])
            paras = rec.get('content', [])
            text  = '\n\n'.join(paras) if isinstance(paras, list) else paras
            if text and len(text) > 50:
                out[dt] = text
        except Exception:
            pass
    return out


def load_speeches() -> dict:
    """Return {date: (text, speaker_type)} for all speeches."""
    out = {}
    for subdir, speaker_type in SPEECH_SUBDIRS.items():
        folder = SPEECHES_DIR / subdir
        if not folder.exists():
            continue
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith('.json'):
                continue
            try:
                with open(folder / fname, encoding='utf-8') as f:
                    rec = json.load(f)
                dt   = pd.Timestamp(rec.get('date', ''))
                text = rec.get('contents') or rec.get('text', '')
                if pd.notna(dt) and text and len(text) > 50:
                    out[dt] = (text, speaker_type)
            except Exception:
                pass
    return out


def load_press_conferences() -> dict:
    out = {}
    if not PRESS_CONF_DIR.exists():
        return out
    for fname in sorted(os.listdir(PRESS_CONF_DIR)):
        if not fname.endswith('.json'):
            continue
        try:
            with open(PRESS_CONF_DIR / fname, encoding='utf-8') as f:
                rec = json.load(f)
            dt  = pd.Timestamp(rec['date'])
            raw = rec.get('prepared_remarks', '')
            text = ' '.join(raw) if isinstance(raw, list) else raw
            if text and len(text) > 50:
                out[dt] = text
        except Exception:
            pass
    return out


# ── FOMC calendar helpers ─────────────────────────────────────────────────────

def load_fomc_calendar() -> pd.DataFrame:
    df = pd.read_csv(MASTER_CSV, parse_dates=['date'])
    df = df[[c for c in df.columns if not c.startswith('Unnamed')]]
    return df[df['event_type'] == 'meeting_date'][
        ['date', 'event_type']
    ].rename(columns={'date': 'meeting_date'}).sort_values('meeting_date').reset_index(drop=True)


def assign_to_meeting(doc_date: pd.Timestamp,
                      calendar: pd.DataFrame) -> pd.Timestamp | None:
    """
    Return the meeting_date that a document should align to.
    - Statements/conferences/minutes: nearest meeting on-or-after doc_date.
    - Used for speeches separately (window-based in aggregate_speeches).
    """
    future = calendar[calendar['meeting_date'] >= doc_date]
    return future['meeting_date'].iloc[0] if not future.empty else None


def aggregate_speeches_by_window(speeches: dict,
                                 calendar: pd.DataFrame) -> dict:
    """
    Pool all speeches between two consecutive FOMC meeting dates.
    Returns {meeting_date: {'text': joined_text, 'speaker_types': set}}.
    Logs alignment warnings for speeches outside all windows.
    """
    meetings  = sorted(calendar['meeting_date'].tolist())
    windows   = {}   # meeting_date -> {'texts': [], 'speakers': set()}
    for mt in meetings:
        windows[mt] = {'texts': [], 'speakers': set()}

    unmatched = []
    for dt, (text, spk) in sorted(speeches.items()):
        matched = False
        for i, mt in enumerate(meetings):
            prev_mt = meetings[i-1] if i > 0 else pd.Timestamp.min
            if prev_mt < dt <= mt:
                windows[mt]['texts'].append(text)
                windows[mt]['speakers'].add(spk)
                matched = True
                break
        if not matched:
            unmatched.append(str(dt.date()))

    if unmatched:
        with open(DATE_ALIGN_LOG, 'a', encoding='utf-8') as f:
            for d in unmatched:
                f.write(f'Speech date {d} does not fall within any FOMC window\n')

    return {
        mt: {
            'text':          '\n\n'.join(v['texts']),
            'speaker_types': v['speakers'],
        }
        for mt, v in windows.items()
        if v['texts']
    }


# ── Normalization ─────────────────────────────────────────────────────────────

def normalize_df(df: pd.DataFrame, score_cols: list,
                 norm: str) -> pd.DataFrame:
    """
    Apply normalization to `score_cols` in `df`.
    norm='wordcount': divide by word_count (must be a column in df).
    norm='zscore':    wordcount first, then expanding z-score.
    norm='none':      no change.
    """
    df = df.copy()
    if norm == 'none':
        return df
    if norm in ('wordcount', 'zscore'):
        if 'word_count' in df.columns and df['word_count'].notna().any():
            for c in score_cols:
                df[c] = df[c] / df['word_count'].replace(0, np.nan)
        if norm == 'zscore':
            for c in score_cols:
                expanding_mean = df[c].expanding(min_periods=3).mean().shift(1)
                expanding_std  = df[c].expanding(min_periods=3).std().shift(1)
                df[c] = (df[c] - expanding_mean) / expanding_std.replace(0, np.nan)
    return df


# ── Apply one dictionary to a {date: text} mapping ───────────────────────────

def apply_dict(texts: dict, nlp_fn, meeting_map: dict,
               minutes_lag: bool = False) -> pd.DataFrame:
    """
    Run `nlp_fn` on each text; return DataFrame with date + score columns.
    meeting_map: {doc_date: meeting_date}.
    """
    rows = []
    for doc_date, text in sorted(texts.items()):
        if not text:
            continue
        try:
            scores = nlp_fn(text)
        except Exception as e:
            log_warn(f'NLP failed for {doc_date}: {e}')
            continue
        if isinstance(scores, pd.Series):
            scores = scores.to_dict()
        row = {'date': doc_date,
               'meeting_date': meeting_map.get(doc_date),
               'word_count':   len(text.split()),
               **scores}
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df['date']         = pd.to_datetime(df['date'])
    df['meeting_date'] = pd.to_datetime(df['meeting_date'])
    return df.sort_values('date').reset_index(drop=True)


# ── Main runner ───────────────────────────────────────────────────────────────

DICTIONARIES = {
    'Gardner': {'fn': gardner_nlp, 'score_cols': ['gardner_inf','gardner_labor',
                                                    'gardner_out','gardner_fin',
                                                    'gardner_mp','gardner_total']},
    'Sharpe':  {'fn': sharpe_nlp,  'score_cols': ['sharpe_positive','sharpe_negative',
                                                    'sharpe_net']},
}

DOC_SOURCES = {
    'statements':       {'loader': load_statements,       'lag': False},
    'minutes':          {'loader': load_minutes,           'lag': True},
    'speeches':         {'loader': None,                   'lag': False},
    'press_conferences':{'loader': load_press_conferences, 'lag': False},
}


def run_dictionary(dict_name: str, doc_types: list, norm: str):
    print(f'\n  [{dict_name}] {"+".join(doc_types)} — norm={norm}')
    d      = DICTIONARIES[dict_name]
    nlp_fn = d['fn']
    s_cols = d['score_cols']

    calendar = load_fomc_calendar()
    meetings = sorted(calendar['meeting_date'].tolist())

    all_rows = []

    for doc_type in doc_types:
        if doc_type == 'speeches':
            raw_speeches = load_speeches()
            agg = aggregate_speeches_by_window(raw_speeches, calendar)
            for mt, v in agg.items():
                text = v['text']
                if not text:
                    continue
                try:
                    scores = nlp_fn(text)
                except Exception as e:
                    log_warn(f'NLP failed for speech at {mt}: {e}')
                    continue
                if isinstance(scores, pd.Series):
                    scores = scores.to_dict()
                all_rows.append({'date': mt, 'meeting_date': mt,
                                  'word_count': len(text.split()),
                                  'doc_type': 'speech',
                                  'speaker_types': str(v['speaker_types']),
                                  **scores})
        else:
            src = DOC_SOURCES[doc_type]
            texts = src['loader']()
            meeting_map = {}
            for dt in texts:
                mt = assign_to_meeting(dt, calendar)
                if mt is not None:
                    if src['lag']:
                        # minutes: assign to the NEXT meeting row
                        idx = meetings.index(mt) + 1 if mt in meetings else None
                        mt  = meetings[idx] if idx and idx < len(meetings) else mt
                    meeting_map[dt] = mt
            rows_df = apply_dict(texts, nlp_fn, meeting_map, minutes_lag=src['lag'])
            rows_df['doc_type'] = doc_type
            all_rows.extend(rows_df.to_dict('records'))

    if not all_rows:
        print(f'  No data produced for {doc_types}')
        return

    df = pd.DataFrame(all_rows)
    df['date']         = pd.to_datetime(df['date'])
    df['meeting_date'] = pd.to_datetime(df['meeting_date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = normalize_df(df, s_cols, norm)

    out_name = f'{dict_name}_{"_".join(doc_types)}_{norm}_nlp.csv'
    out_path = NLP_DIR / out_name
    keep_cols = ['date','meeting_date','doc_type'] + \
                (['speaker_types'] if 'speeches' in doc_types else []) + \
                [c for c in s_cols if c in df.columns]
    df[keep_cols].to_csv(out_path, index=False)
    n_meetings = df['meeting_date'].nunique()
    print(f'  ✓  {out_name}  ({n_meetings} meetings × {len(s_cols)} scores)')


def main(dict_filter=None, norm_filter=None):
    combos = [
        ('Gardner', ['statements'],                     'wordcount'),
        ('Gardner', ['statements'],                     'zscore'),
        ('Gardner', ['minutes'],                        'wordcount'),
        ('Gardner', ['minutes'],                        'zscore'),
        ('Gardner', ['speeches'],                       'wordcount'),
        ('Gardner', ['speeches'],                       'zscore'),
        ('Gardner', ['press_conferences'],              'wordcount'),
        ('Gardner', ['press_conferences'],              'zscore'),
        ('Gardner', ['statements','minutes','speeches','press_conferences'], 'wordcount'),
        ('Gardner', ['statements','minutes','speeches','press_conferences'], 'zscore'),
        ('Sharpe',  ['statements'],                     'wordcount'),
        ('Sharpe',  ['statements'],                     'zscore'),
        ('Sharpe',  ['minutes'],                        'wordcount'),
        ('Sharpe',  ['minutes'],                        'zscore'),
        ('Sharpe',  ['speeches'],                       'wordcount'),
        ('Sharpe',  ['speeches'],                       'zscore'),
        ('Sharpe',  ['press_conferences'],              'wordcount'),
        ('Sharpe',  ['press_conferences'],              'zscore'),
        ('Sharpe',  ['statements','minutes','speeches','press_conferences'], 'wordcount'),
        ('Sharpe',  ['statements','minutes','speeches','press_conferences'], 'zscore'),
    ]

    for dict_name, doc_types, norm in combos:
        if dict_filter and dict_name.lower() != dict_filter.lower():
            continue
        if norm_filter and norm != norm_filter:
            continue
        run_dictionary(dict_name, doc_types, norm)

    print(f'\n✅  Dictionary outputs saved to {NLP_DIR}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run FOMC dictionary sentiment')
    parser.add_argument('--dict', type=str, default=None,
                        choices=['Gardner','Sharpe'],
                        help='Run only this dictionary')
    parser.add_argument('--norm', type=str, default=None,
                        choices=['wordcount','zscore','none'],
                        help='Run only this normalization')
    args = parser.parse_args()
    main(dict_filter=args.dict, norm_filter=args.norm)
