"""
apply_seed_corrections.py
--------------------------
Applies all corrections from 'Initial seed corrections.txt' to
eval_labelled_merged.json and final_extreme_seed.json.

Handles:
  - Prose corrections mapped by 8-char ID prefix
  - JSON-line corrections (full or prefix IDs)
  - Special array ops: _add_to_top, _remove_from_top
"""

import json
from pathlib import Path

ROOT     = Path(__file__).parent.parent / 'data'
CORR_TXT = Path('C:/Users/Javier/Downloads/Initial seed corrections.txt')

# Collect all JSON files that might contain labelled records
import os as _os
def _collect_files():
    candidates = []
    search_dirs = [
        ROOT / 'QuantCube_Seed_Labelled',
        ROOT / 'QuantCube_Seed_Batches',
        ROOT / 'QuantCube_Evaluation_Batches',
        ROOT / 'QuantCube_Evaluation_Pseudolabels',
    ]
    always_include = [
        ROOT / 'eval_labelled_merged.json',
        ROOT / 'eval_labelled_merged_corrected.json',
    ]
    seen = set()
    for p in always_include:
        if p.exists() and str(p) not in seen:
            candidates.append(p)
            seen.add(str(p))
    for d in search_dirs:
        if not d.exists():
            continue
        for fname in sorted(_os.listdir(d)):
            if fname.endswith('.json') and not fname.startswith('.'):
                p = d / fname
                if str(p) not in seen:
                    candidates.append(p)
                    seen.add(str(p))
    return candidates

FILES = _collect_files()
METADATA = {'r', 'c'}

# ── Prose corrections extracted from the text section ─────────────────────────
PROSE = {
    # Batch 1 - wid_elevated
    'a58a93c9': {'wid': 'elevated'},
    '6e78f365': {'hor': True},
    'fc2abf74': {'ten': 'descriptive'},
    '8d153860': {'top': ['no_topic']},
    'd86c1a7c': {'ten': 'interpretive', 'top': ['inflation', 'labor_market']},
    # Batch 2 - wid_contested
    '49724d32': {'wid': 'none'},
    '053aa189': {'wid': 'contested'},
    'a460e423': {'sen': 'hawkish', 'wid': 'none'},
    # Batch 3 - statement_batch_08
    '3356bf8c': {'sen': 'dovish', 'hor': True},
    '9dd272d5': {'top': ['monetary_policy', 'inflation', 'macro'], 'sen': 'dovish',
                 'com': 'conditional', 'ten': 'interpretive'},
    '43d66a97': {'ten': 'interpretive', 'ris': 'na'},
    '201d9b61': {'hor': False, 'sen': 'dovish'},
    '7fea3193': {'top': ['no_topic'], 'sen': 'neutral'},
    '7e8b2524': {'com': 'none'},
    # Batch 4 - statement_batch_09
    'c3082c32': {'sen': 'dovish'},
    'cf1b310f': {'com': 'none', 'hor': False},
    'df621765': {'sen': 'hawkish'},
    '3fce9480': {'com': 'none', 'hor': False},
    '997a032e': {'top': ['no_topic'], 'sen': 'neutral'},
    'd790d3a5': {'hor': True, 'ten': 'interpretive'},
    '77912829': {'top': ['no_topic'], 'sen': 'neutral'},
    '1772c6d9': {'top': ['no_topic'], 'sen': 'neutral'},
    # Batch 5 - statement_batch_06
    '9aa31a12': {'top': ['economic_activity', 'labor_market']},
    '24f5b9a3': {'top': ['no_topic'], 'ten': 'descriptive', 'ris': 'na'},
    '980c0523': {'com': 'none', 'top': ['no_topic']},
    '64b23d60': {'top': ['no_topic'], 'sen': 'neutral'},
    '75f4a0df': {'com': 'none'},
    'c9e4c84a': {'top': ['no_topic'], 'sen': 'neutral'},
    '969b3ae2': {'com': 'none'},
    # Batch 6 - statement_batch_07
    '9368bfde': {'sen': 'dovish'},
    'fbe438bb': {'com': 'none'},
    'edf8a9c3': {'hor': True, 'ten': 'interpretive'},
    '42af920e': {'hor': True},
    '9ec375e9': {'com': 'none'},
    'b43a0bf1': {'top': ['no_topic'], 'sen': 'neutral'},
    '75fbd371': {'top': ['monetary_policy', 'inflation', 'macro'],
                 'sen': 'dovish', 'ten': 'interpretive'},
    'ea32b905': {'com': 'none'},
    '0e930a35': {'top': ['no_topic'], 'sen': 'neutral'},
    '750d7dd6': {'com': 'none'},
    # Speech batches
    '110ba493': {'com': 'conditional'},
    'f111622a': {'sen': 'strongly_dovish'},
    'd3659adc': {'ris': 'na'},
    'd40a5502': {'_remove_from_top': 'monetary_policy'},
    '018080c3': {'sen': 'neutral'},
    '6447855e': {'sen': 'neutral'},
    '3972a577': {'sen': 'strongly_hawkish'},
    '47767d45': {'sen': 'dovish'},
    'cb79c820': {'_add_to_top': 'financial_conditions'},
    '7b6a5868': {'sen': 'strongly_hawkish'},   # CRITICAL INVERSION
    '57729d20': {'sen': 'strongly_hawkish'},   # CRITICAL INVERSION
    '478ecebe': {'sen': 'dovish'},
    'd8551fb5': {'top': ['no_topic'], 'sen': 'neutral'},
    'bd53329c': {'sen': 'neutral'},
    'fc71d536': {'sen': 'neutral'},
    'b09eae52': {'wid': 'elevated'},
    'a0c4d49d': {'sen': 'neutral'},
    'e9a87b7d': {'sen': 'neutral'},
    '337a412f': {'sen': 'dovish'},             # CRITICAL INVERSION
    'd9990757': {'sen': 'hawkish'},
    '9c4a6326': {'top': ['no_topic'], 'sen': 'neutral'},
    '69c6f26a': {'sen': 'neutral'},
    '916ee7e9': {'sen': 'dovish'},
    'c6d48736': {'sen': 'neutral'},
    # Final extreme seed
    '27dc1dc9': {'wid': 'none'},
    '58fc6995': {'sen': 'strongly_dovish'},
}


def parse_jsonl_corrections(txt_path: Path) -> dict:
    out = {}
    with open(txt_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('{'):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            fid = obj.get('id', '')
            if not fid:
                continue
            correction = {k: v for k, v in obj.items()
                          if k not in METADATA and k != 'id'}
            if not correction:
                continue
            key = fid[:8] if len(fid) < 30 else fid
            out.setdefault(key, {}).update(correction)
    return out


def apply_correction(rec: dict, corrections: dict) -> bool:
    changed = False
    for field, new_val in corrections.items():
        if field == '_remove_from_top':
            top = rec.get('top', [])
            if isinstance(top, list) and new_val in top:
                rec['top'] = [t for t in top if t != new_val]
                changed = True
        elif field == '_add_to_top':
            top = rec.get('top', [])
            if isinstance(top, list) and new_val not in top:
                rec['top'] = top + [new_val]
                changed = True
            elif isinstance(top, str) and new_val != top:
                rec['top'] = [top, new_val]
                changed = True
        else:
            if rec.get(field) != new_val:
                rec[field] = new_val
                changed = True
    return changed


def main():
    # Load all records
    file_records = {}   # filename -> list (shared object refs)
    all_records  = {}   # id -> record object
    skipped = 0
    for p in FILES:
        try:
            with open(p, encoding='utf-8', errors='replace') as f:
                recs = json.load(f)
            if not isinstance(recs, list):
                continue
            file_records[str(p)] = recs
            for r in recs:
                if isinstance(r, dict) and 'id' in r:
                    all_records[r['id']] = r
        except Exception:
            skipped += 1
    print(f'Files loaded: {len(file_records)}  (skipped {skipped})')
    print(f'Records loaded: {len(all_records)}')

    # Build prefix -> full-id map
    prefix_map = {fid[:8]: fid for fid in all_records}

    # Merge prose + jsonl corrections
    jsonl = parse_jsonl_corrections(CORR_TXT)
    print(f'Prose corrections: {len(PROSE)} IDs')
    print(f'JSON-line corrections: {len(jsonl)} IDs')

    merged_corrections = dict(PROSE)
    for key, corr in jsonl.items():
        merged_corrections.setdefault(key, {}).update(corr)
    print(f'Total unique correction targets: {len(merged_corrections)}')

    applied   = []
    not_found = []

    for key, corr in merged_corrections.items():
        if key in all_records:
            full_id = key
        elif key in prefix_map:
            full_id = prefix_map[key]
        else:
            not_found.append(key)
            continue

        rec = all_records[full_id]
        if apply_correction(rec, corr):
            applied.append(full_id)

    print(f'\nRecords changed: {len(applied)}')
    if not_found:
        print(f'Not found ({len(not_found)}): {not_found}')

    # Save
    saved = 0
    for p_str, recs in file_records.items():
        p = Path(p_str)
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(recs, f, indent=2, ensure_ascii=False)
        saved += 1
    print(f'Saved {saved} files')

    print('\nDone.')


if __name__ == '__main__':
    main()
