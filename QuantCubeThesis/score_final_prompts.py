import json, sys
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

GOLD_PATH = 'data/eval_labelled_merged_corrected.json'
RAW_DIR   = Path('models/prompt_eval')

PROMPTS = [
    'P2_medium_3shot_final',
    'P2_medium_3shot_final_v3',
    'P3_medium_5shot_final',
    'P3_medium_5shot_final_v3',
    'P5_high_3shot_final',
    'P5_high_3shot_final_v3',
    'P7_high_5shot_final',
    'P7_high_5shot_final_v3',
    'P8_high_12shot_final_v4',
]

# Field name map: final prompt output → gold label keys
FIELD_MAP = {
    'topic':      'top',
    'tense':      'ten',
    'sentiment':  'sen',
    'horizon':    'hor',
    'commitment': 'com',
    'risk':       'ris',
    'width':      'wid',
}

ALL_TOPICS = ['inflation', 'labor_market', 'economic_activity', 'macro',
              'financial_conditions', 'monetary_policy', 'boilerplate', 'no_topic']

with open(GOLD_PATH, encoding='utf-8') as f:
    gold_list = json.load(f)
gold = {r['id']: r for r in gold_list}

def compute_topic_f1(y_true_lists, y_pred_lists):
    mlb = MultiLabelBinarizer(classes=ALL_TOPICS)
    Y_true = mlb.fit_transform(y_true_lists)
    Y_pred = mlb.transform(y_pred_lists)
    return f1_score(Y_true, Y_pred, average='macro', zero_division=0)

def compute_field_f1(y_true, y_pred, labels=None):
    return f1_score(y_true, y_pred, average='macro', zero_division=0, labels=labels)

print(f"\n{'Prompt':<35} {'N':>4}  {'SumF1':>6}  {'top':>6}  {'sen':>6}  {'ris':>6}  {'wid':>6}  {'Fail':>5}")
print(f"  {'-'*80}")

results = {}
for prompt in PROMPTS:
    raw_path = RAW_DIR / f'{prompt}_raw.json'
    if not raw_path.exists():
        print(f"  {prompt:<35}  MISSING")
        continue

    with open(raw_path, encoding='utf-8') as f:
        preds = json.load(f)

    top_true, top_pred = [], []
    sen_true, sen_pred = [], []
    ris_true, ris_pred = [], []
    wid_true, wid_pred = [], []
    n_matched = 0
    n_parse_fail = 0

    for rec in preds:
        rid = rec['id']
        if rid not in gold:
            continue
        g = gold[rid]
        p = rec.get('parsed') or {}

        n_matched += 1
        if not p:
            n_parse_fail += 1

        # topic (multi-label)
        top_true.append(g.get('top', []))
        raw_topic = p.get('topic', [])
        if isinstance(raw_topic, list):
            top_pred.append(raw_topic)
        else:
            top_pred.append([raw_topic] if raw_topic else [])

        # sentiment
        sen_true.append(str(g.get('sen', 'na')))
        sen_pred.append(str(p.get('sentiment', 'na')))

        # risk
        ris_true.append(str(g.get('ris', 'na')))
        ris_pred.append(str(p.get('risk', 'na')))

        # width
        wid_true.append(str(g.get('wid', 'none')))
        wid_pred.append(str(p.get('width', 'none')))

    top_f1 = compute_topic_f1(top_true, top_pred)
    sen_f1 = compute_field_f1(sen_true, sen_pred)
    ris_f1 = compute_field_f1(ris_true, ris_pred)
    wid_f1 = compute_field_f1(wid_true, wid_pred)
    summary_f1 = (top_f1 + sen_f1 + ris_f1 + wid_f1) / 4
    fail_rate = n_parse_fail / n_matched if n_matched else 0

    results[prompt] = dict(n=n_matched, summary_f1=summary_f1,
                           top=top_f1, sen=sen_f1, ris=ris_f1, wid=wid_f1,
                           fail=fail_rate)

    print(f"  {prompt:<35} {n_matched:>4}  {summary_f1:>6.4f}  "
          f"{top_f1:>6.3f}  {sen_f1:>6.3f}  {ris_f1:>6.3f}  {wid_f1:>6.3f}  "
          f"{fail_rate:>4.1%}")

print()
