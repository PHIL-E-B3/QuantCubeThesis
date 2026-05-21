"""
compute_kuttner_implied_ffr.py
-------------------------------
Computes the Kuttner (2001) post-meeting implied FFR level for every FOMC
event date (meeting, blackout, minutes) using daily FF1/FF2/FF3 futures prices.

Kuttner formula for meeting dates
----------------------------------
The front-month futures price F1 prices the expected average FFR for the
entire current month. Given a meeting on day d of a D-day month and the
known pre-meeting effective rate r_pre:

    f1      = 100 − F1_price           (monthly average implied rate)
    r_post  = (D × f1 − d × r_pre) / (D − d)   (post-meeting implied rate)

For meetings in the last 5 days of a month (D − d < 5), the denominator is
too small and FF2 (next month) is used directly: r_post = 100 − F2_price.

For blackout and minutes dates (no rate decision), the pre-meeting expected
rate is approximated as 100 − F1_price directly.

Output
------
    data/macro_inputs/kuttner_implied_ffr_v2.csv
    columns: date, event_type, effr, f1_price, f2_price,
             f1_rate, f2_rate, implied_ffr, kuttner_method
"""

import calendar as cal
import sys
import warnings
from pathlib import Path

import numpy  as np
import pandas as pd

warnings.filterwarnings('ignore')

ROOT         = Path(__file__).parent.parent
FFF_FILE     = ROOT / 'Fed Funds Futures.xlsx'
MASTER_CSV   = ROOT / 'data' / 'macro_inputs' / 'Master_Macro.csv'
OUTPUT_CSV   = ROOT / 'data' / 'macro_inputs' / 'kuttner_implied_ffr_v2.csv'


# ── Load and parse futures data ───────────────────────────────────────────────

def parse_futures_sheet(xl: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    raw = xl.parse(sheet, header=None)
    header_row = next(
        i for i, row in raw.iterrows()
        if row.astype(str).str.contains('Date', case=False).any()
    )
    df = xl.parse(sheet, header=header_row)
    df.columns = [str(c).strip() for c in df.columns]
    date_col  = next(c for c in df.columns if 'date'  in c.lower())
    price_col = next(c for c in df.columns if 'last'  in c.lower()
                     or 'price' in c.lower())
    df = df[[date_col, price_col]].rename(
        columns={date_col: 'date', price_col: 'price'})
    df['date']  = pd.to_datetime(df['date'],  errors='coerce')
    df['price'] = pd.to_numeric(df['price'],  errors='coerce')
    return df.dropna().sort_values('date').reset_index(drop=True)


def nearest_before(series: pd.DataFrame, date: pd.Timestamp, col='price'):
    """Most recent value on or before `date`."""
    sub = series[series['date'] <= date]
    return float(sub[col].iloc[-1]) if not sub.empty else np.nan


# ── Kuttner unscrambling ──────────────────────────────────────────────────────

def kuttner_post_meeting(meeting_date: pd.Timestamp,
                         f1_price: float,
                         f2_price: float,
                         effr_pre: float,
                         late_month_threshold: int = 5) -> tuple:
    """
    Return (implied_ffr, method_label).

    method_label:
      'kuttner_f1'  — unscrambled from FF1
      'kuttner_f2'  — FF2 used directly (late-month meeting)
      'f1_raw'      — fallback: raw 100−F1 (when EFFR pre unavailable)
    """
    d = meeting_date.day
    D = cal.monthrange(meeting_date.year, meeting_date.month)[1]

    f1_rate = 100 - f1_price if not np.isnan(f1_price) else np.nan
    f2_rate = 100 - f2_price if not np.isnan(f2_price) else np.nan

    # Late-month meeting: FF2 is cleaner
    if (D - d) < late_month_threshold:
        if not np.isnan(f2_rate):
            return f2_rate, 'kuttner_f2'
        return f1_rate, 'f1_raw'

    # Standard Kuttner unscrambling
    if not np.isnan(f1_rate) and not np.isnan(effr_pre):
        denom = D - d
        if denom > 0:
            r_post = (D * f1_rate - d * effr_pre) / denom
            # Sanity-check: implied rate should be within ±3pp of f1_rate
            if abs(r_post - f1_rate) < 3.0:
                return r_post, 'kuttner_f1'

    # Fallback
    if not np.isnan(f1_rate):
        return f1_rate, 'f1_raw'
    return np.nan, 'missing'


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('Loading futures data ...')
    xl  = pd.ExcelFile(FFF_FILE)
    ff1 = parse_futures_sheet(xl, 'FF1')
    ff2 = parse_futures_sheet(xl, 'FF2')
    ff3 = parse_futures_sheet(xl, 'FF3')
    print(f'  FF1: {len(ff1)} rows  '
          f'{ff1["date"].min().date()} – {ff1["date"].max().date()}')
    print(f'  FF2: {len(ff2)} rows  '
          f'{ff2["date"].min().date()} – {ff2["date"].max().date()}')

    print('Loading FOMC calendar from Master_Macro ...')
    master = pd.read_csv(MASTER_CSV, parse_dates=['date'])
    master = master[[c for c in master.columns
                     if not c.startswith('Unnamed')]]
    events = (master[master['event_type'].isin(
                  ['meeting_date', 'blackout_date', 'minutes_date'])]
              [['date', 'event_type', 'fed_funds_rate']]
              .drop_duplicates('date')
              .sort_values('date')
              .reset_index(drop=True))
    print(f'  {len(events)} event rows  '
          f'({events["date"].min().date()} – {events["date"].max().date()})')

    records = []
    for _, ev in events.iterrows():
        date       = ev['date']
        event_type = ev['event_type']
        effr       = ev['fed_funds_rate']   # pre-event effective rate

        f1p = nearest_before(ff1, date)
        f2p = nearest_before(ff2, date)
        f3p = nearest_before(ff3, date)

        f1_rate = 100 - f1p if not np.isnan(f1p) else np.nan
        f2_rate = 100 - f2p if not np.isnan(f2p) else np.nan

        if event_type == 'meeting_date':
            implied, method = kuttner_post_meeting(date, f1p, f2p, effr)
        else:
            # Blackout / minutes: use 100−FF1 directly (pre-meeting expectation)
            implied = f1_rate
            method  = 'f1_raw'

        records.append({
            'date':           date,
            'event_type':     event_type,
            'effr':           round(effr,   4) if not np.isnan(effr)    else np.nan,
            'f1_price':       round(f1p,    4) if not np.isnan(f1p)    else np.nan,
            'f2_price':       round(f2p,    4) if not np.isnan(f2p)    else np.nan,
            'f1_rate':        round(f1_rate, 4) if not np.isnan(f1_rate) else np.nan,
            'f2_rate':        round(f2_rate, 4) if not np.isnan(f2_rate) else np.nan,
            'implied_ffr':    round(implied, 4) if not np.isnan(implied) else np.nan,
            'kuttner_method': method,
        })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f'\nSaved: {OUTPUT_CSV}')
    print(f'  {len(df)} rows  |  NaN implied_ffr: {df["implied_ffr"].isna().sum()}')
    print(f'  Range: {df["implied_ffr"].min():.3f} – {df["implied_ffr"].max():.3f}')
    print(f'  Negative values: {(df["implied_ffr"] < 0).sum()}')
    print()
    print('Method breakdown:')
    print(df['kuttner_method'].value_counts().to_string())
    print()
    print('Sample (meeting dates):')
    mtg = df[df['event_type'] == 'meeting_date']
    print(mtg[['date','effr','f1_rate','implied_ffr','kuttner_method']].iloc[::30].to_string(index=False))


if __name__ == '__main__':
    main()
