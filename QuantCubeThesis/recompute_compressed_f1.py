"""
Recompute F1 with label compression + forcing rules, mirroring how
the user previously achieved higher scores.

Compression (applied to BOTH pred and GT):
  sentiment:  strongly_hawkish -> hawkish, strongly_dovish -> dovish
  width:      contested -> none
  commitment: conditional -> none

Forcing (applied to PREDICTIONS only, as post-processing):
  - topic has no 'monetary_policy' -> commitment = none
  - topic is no_topic only         -> sentiment  = neutral
"""
import json
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score

ROOT = Path(__file__).parent
GT_FIELD_MAP = {'top':'topic','ten':'tense','sen':'sentiment',
                'com':'commitment','ris':'risk','wid':'width','hor':'horizon'}
PRIMARY = ['topic', 'sentiment', 'commitment', 'risk', 'width']
LABEL_SCHEMA = {'topic':'multi','tense':'single','sentiment':'single',
                'commitment':'single','risk':'single','width':'single'}


def compress(rec: dict) -> dict:
    r = dict(rec)
    SENT_MAP = {'strongly_hawkish': 'hawkish', 'strongly_dovish': 'dovish'}
    if isinstance(r.get('sentiment'), list):
        r['sentiment'] = [SENT_MAP.get(v, v) for v in r['sentiment']]
    elif r.get('sentiment'):
        r['sentiment'] = SENT_MAP.get(r['sentiment'], r['sentiment'])

    if r.get('width') == 'contested':
        r['width'] = 'none'
    if r.get('commitment') == 'conditional':
        r['commitment'] = 'none'
    return r


def force_pred(rec: dict) -> dict:
    r = dict(rec)
    topics = r.get('topic', [])
    if isinstance(topics, str):
        topics = [topics]

    if 'monetary_policy' not in topics:
        r['commitment'] = 'none'

    if topics == ['no_topic'] or topics == ['na']:
        r['sentiment'] = 'neutral'
    return r


def normalize_gt(rec: dict) -> dict:
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


def normalize_pred(rec: dict) -> dict:
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


def compute_metrics(preds, gts):
    results = {}
    for f, ftype in LABEL_SCHEMA.items():
        if ftype == 'single':
            yt = [gt[f] for gt in gts]
            yp = [p[f]  for p  in preds]
            # Exclude 'na' from sentiment — true 3-class eval (hawkish/neutral/dovish)
            if f == 'sentiment':
                pairs = [(t, p) for t, p in zip(yt, yp) if t != 'na']
                yt, yp = zip(*pairs) if pairs else ([], [])
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
    return results, summary


def run(raw_path, gt_lookup, label):
    raw = json.load(open(raw_path, encoding='utf-8'))
    preds, gts = [], []
    for r in raw:
        sid = r['id']
        if sid not in gt_lookup:
            continue
        parsed = r.get('parsed') or {}
        # normalize pred, then compress, then force
        pred = normalize_pred(parsed)
        pred = compress(pred)
        pred = force_pred(pred)
        # normalize GT, then compress
        gt = normalize_gt(gt_lookup[sid])
        gt = compress(gt)
        preds.append(pred)
        gts.append(gt)

    metrics, summary = compute_metrics(preds, gts)
    print(f'\n=== {label} (n={len(preds)}) — compressed + forced ===')
    print(f'  Summary F1: {summary:.4f}')
    for f in LABEL_SCHEMA:
        marker = ' *' if f in PRIMARY else ''
        print(f'  {f:<12} acc={metrics[f]["accuracy"]:.3f}  F1={metrics[f]["f1_macro"]:.3f}{marker}')
    return summary


# ── Same GT for both: 3-class_com_con (n=711, all IDs present in both raw files)
gt = {s['id']: s for s in json.load(open(ROOT / 'data' / 'eval_merged_labelled_corrected_3-class_com_con.json', encoding='utf-8'))}

run(ROOT / 'models/best/eval_results/sft_best_raw.json',
    gt, 'Original adapter (checkpoint-258) | 3-class_com_con GT')

run(ROOT / 'models/best/eval_results/prompt_compare/sft_P5_v41b_raw.json',
    gt, 'New adapter | 3-class_com_con GT')
