"""
run_pipeline.py
---------------
Master runner for the Master_Macro.csv construction pipeline.

Steps:
    01  Scrape pre-2011 FOMC statements (optional — run once)
    02  Build base macro dataset (FRED vintages, market data, SPF, implied FFR)
    03  Append channel regressors (FFR path, inf exp, NFCI, JK shocks, novelty, dissent)
    04  Append SPF cross-sectional dispersion (~240 columns)
    05  Append SEP projections → Master_Macro_with_SEP.csv (optional)

Usage:
    # Full pipeline (02 → 03 → 04)
    python macro_pipeline/run_pipeline.py

    # Including pre-2011 scraping and SEP data
    python macro_pipeline/run_pipeline.py --all

    # Individual steps
    python macro_pipeline/run_pipeline.py --steps 03 04

    # Rebuild base dataset from scratch (clears cache)
    python macro_pipeline/run_pipeline.py --steps 02 --clear-cache
"""

import argparse
import sys
import time
from pathlib import Path

# Make sure imports resolve from project root
sys.path.insert(0, str(Path(__file__).parent))


def run_step(label: str, fn, **kwargs):
    sep = '=' * 65
    print(f'\n{sep}')
    print(f'  PIPELINE STEP: {label}')
    print(f'{sep}')
    t0 = time.time()
    fn(**kwargs)
    elapsed = time.time() - t0
    print(f'\n  ✓  {label} completed in {elapsed/60:.1f} min')


def main():
    parser = argparse.ArgumentParser(description='Run Master_Macro pipeline')
    parser.add_argument('--steps', nargs='+', default=['02','03','04'],
                        choices=['01','02','03','04','05'],
                        help='Which steps to run (default: 02 03 04)')
    parser.add_argument('--all', action='store_true',
                        help='Run all steps including 01 (scraping) and 05 (SEP)')
    parser.add_argument('--rebuild-calendar', action='store_true',
                        help='Rebuild FOMC calendar (passed to step 02)')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear FRED cache before step 02')
    args = parser.parse_args()

    steps = ['01','02','03','04','05'] if args.all else args.steps
    steps = sorted(set(steps))
    print(f'Running steps: {", ".join(steps)}')

    t_total = time.time()

    if '01' in steps:
        from importlib import import_module
        m = import_module('01_scrape_statements')
        run_step('01 — Scrape pre-2011 statements', m.scrape_all)

    if '02' in steps:
        from importlib import import_module
        import importlib, importlib.util
        spec = importlib.util.spec_from_file_location(
            'build_base', Path(__file__).parent / '02_build_base_dataset.py')
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        run_step('02 — Build base dataset',
                 m.main,
                 rebuild_calendar=args.rebuild_calendar,
                 clear_cache=args.clear_cache)

    if '03' in steps:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'append_channels', Path(__file__).parent / '03_append_channels.py')
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        run_step('03 — Append channel regressors', m.main)

    if '04' in steps:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'spf_disp', Path(__file__).parent / '04_append_spf_dispersion.py')
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        run_step('04 — Append SPF dispersion', m.main)

    if '05' in steps:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'sep', Path(__file__).parent / '05_append_sep_data.py')
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        run_step('05 — Append SEP data', m.main)

    total = time.time() - t_total
    print(f'\n{"="*65}')
    print(f'  Pipeline complete in {total/60:.1f} min')
    print(f'{"="*65}')


if __name__ == '__main__':
    main()
