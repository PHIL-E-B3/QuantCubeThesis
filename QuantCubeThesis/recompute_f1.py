import json
from sklearn.metrics import f1_score, accuracy_score
from pathlib import Path

ROOT = Path(__file__).parent
GT_FIELD_MAP = {'top':'topic','ten':'tense','sen':'sentiment','com':'commitment','ris':'risk','wid':'width'}
LABEL_SCHEMA = {
    'topic':      'multi',
    'tense':      'single',
    'sentiment':  'single',
    'commitment': 'single',
    'risk':       'single',
    'width':      'single',
}
PRIMARY = ['topic', 'sentiment', 'risk', 'width']

eval_path = ROOT / 'data' / 'eval_merged_labelled_corrected_3-class_com_con.json'
gt_all = {s['id']: s for s in json.load(open(eval_path, encoding='utf-8'))}

def normalize_gt(rec):
    rec = {GT_FIELD_MAP.get(k, k): v for k, v in rec.items()}
    out = {}
    for f, ftype in LABEL_SCHEMA.items():
        val = rec.get(f)
        if val is None or val == '':
            out[f] = ['na'] if ftype == 'multi' else 'na'
        elif ftype == 'multi':
            out[f] = [str(v).lower().strip() for v in val] if isinstance(val, list) else [str(val).lower().strip()]
        else:
            out[f] = str(val).lower().strip()
    return out

def normalize_pred(rec):
    out = {}
    for f, ftype in LABEL_SCHEMA.items():
        val = rec.get(f)
        if val is None or val == '':
            out[f] = ['na'] if ftype == 'multi' else 'na'
        elif ftype == 'multi':
            out[f] = [str(v).lower().strip() for v in val] if isinstance(val, list) else [str(val).lower().strip()]
        else:
            out[f] = str(val).lower().strip()
    return out

def compute(raw_path):
    raw = json.load(open(raw_path, encoding='utf-8'))
    preds, gts, parse_fails = [], [], 0
    for r in raw:
        sid = r['id']
        if sid not in gt_all:
            continue
        parsed = r.get('parsed') or {}
        if not parsed:
            parse_fails += 1
        preds.append(normalize_pred(parsed))
        gts.append(normalize_gt(gt_all[sid]))

    results = {}
    for f, ftype in LABEL_SCHEMA.items():
        if ftype == 'single':
            yt = [gt[f] for gt in gts]
            yp = [p[f]  for p  in preds]
            results[f] = {
                'accuracy': accuracy_score(yt, yp),
                'f1_macro': f1_score(yt, yp, average='macro', zero_division=0),
                'n': len(yt),
            }
        else:
            yts = [set(gt[f]) for gt in gts]
            yps = [set(p[f])  for p  in preds]
            all_labels = sorted({l for s in yts + yps for l in s})
            ybt = [[1 if l in s else 0 for l in all_labels] for s in yts]
            ybp = [[1 if l in s else 0 for l in all_labels] for s in yps]
            results[f] = {
                'accuracy': sum(1 for t, p in zip(yts, yps) if t == p) / len(yts),
                'f1_macro': f1_score(ybt, ybp, average='macro', zero_division=0),
                'n': len(yts),
            }
    summary = sum(results[f]['f1_macro'] for f in PRIMARY) / len(PRIMARY)
    return results, summary, parse_fails, len(preds)

for tag, fname in [('BASE', 'base_P5_v41b_raw.json'), ('SFT', 'sft_P5_v41b_raw.json')]:
    path = ROOT / 'models/best/eval_results/prompt_compare' / fname
    res, summary, fails, n = compute(path)
    print(f'\n=== {tag} (n={n}, parse_fails={fails}) ===')
    print(f'  Summary F1 (topic+sentiment+risk+width): {summary:.4f}')
    for f in LABEL_SCHEMA:
        marker = ' *' if f in PRIMARY else ''
        acc = res[f]['accuracy']
        f1  = res[f]['f1_macro']
        nn  = res[f]['n']
        print(f'  {f:<12} acc={acc:.3f}  F1={f1:.3f}  (n={nn}){marker}')

print()
