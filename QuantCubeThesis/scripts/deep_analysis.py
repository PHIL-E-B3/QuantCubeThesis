"""Deep analysis of P7, P7v3, P3 on full 498-sentence eval set."""
import json, os, sys
from collections import defaultdict
from pathlib import Path

results_dir = Path(__file__).parent.parent / "models" / "prompt_eval"
eval_path = Path(__file__).parent.parent / "data" / "eval_labelled_merged.json"

EXAMPLE_IDS = {
    "e5418507-7ab1-476e-8db9-fc28796c584f",
    "99122de6-aab0-49fc-9183-d4bd3fc33e27",
    "c7346ec3-ca84-449c-a104-7e0c8d3543ba",
    "cd96b673-cdc4-4ab3-a428-3880ba0bd1dd",
    "995fddd0-966b-4ea0-9eaf-879c5f7fbeed",
}

with open(eval_path, encoding="utf-8") as f:
    eval_all = json.load(f)
eval_data = {s["id"]: s for s in eval_all if s["id"] not in EXAMPLE_IDS}

prompts = ["P7_high_5shot", "P7v3_high_5shot", "P3_medium_5shot"]
raw = {}
for p in prompts:
    with open(results_dir / f"{p}_raw.json", encoding="utf-8") as f:
        raw[p] = {r["id"]: r for r in json.load(f)}

sent_ids = list(raw["P7_high_5shot"].keys())

# ── 1. SEN ERROR DIRECTION ──
print("=" * 70)
print("1. SEN ERROR DIRECTION (498 sentences)")
print("=" * 70)
for p in prompts:
    h = d = o = t = 0
    for sid in sent_ids:
        truth = eval_data[sid]
        true_sen = truth.get("sen")
        pred = raw[p].get(sid, {}).get("parsed", {})
        if not pred: continue
        pred_sen = pred.get("sen")
        try:
            tn = int(true_sen) if true_sen != "na" else None
            pn = int(pred_sen) if pred_sen != "na" else None
        except: continue
        if tn is None and pn is None: continue
        if tn is None or pn is None: t += 1; o += 1; continue
        if pn == tn: continue
        t += 1
        if pn > tn: h += 1
        else: d += 1
    pct = h / t * 100 if t else 0
    print(f"  {p}: hawk={h} dove={d} other={o} total_err={t} hawk%={pct:.0f}%")

# ── 2. SEN ACCURACY BY TOPIC ──
print("\n" + "=" * 70)
print("2. SEN ACCURACY BY TRUE TOPIC")
print("=" * 70)
for p in prompts:
    topic_correct = defaultdict(int)
    topic_total = defaultdict(int)
    for sid in sent_ids:
        truth = eval_data[sid]
        true_top = truth.get("top", [])
        if isinstance(true_top, str): true_top = [true_top]
        pred = raw[p].get(sid, {}).get("parsed", {})
        if not pred: continue
        correct = str(pred.get("sen")) == str(truth.get("sen"))
        for t in true_top:
            topic_total[t] += 1
            if correct: topic_correct[t] += 1
    print(f"\n  {p}:")
    for t in sorted(topic_total.keys(), key=lambda x: topic_correct.get(x,0)/max(topic_total.get(x,1),1)):
        acc = topic_correct[t] / topic_total[t] if topic_total[t] else 0
        print(f"    {t:<25} {topic_correct[t]:>3}/{topic_total[t]:<3} ({acc:.0%})")

# ── 3. SEN CONFUSION MATRIX ──
print("\n" + "=" * 70)
print("3. SEN CONFUSION MATRIX (P7 and P7v3)")
print("=" * 70)
sen_vals = ["-2", "-1", "0", "1", "2", "na"]
for p in ["P7_high_5shot", "P7v3_high_5shot"]:
    cm = defaultdict(lambda: defaultdict(int))
    for sid in sent_ids:
        truth = eval_data[sid]
        pred = raw[p].get(sid, {}).get("parsed", {})
        if not pred: continue
        tv = str(truth.get("sen"))
        pv = str(pred.get("sen"))
        cm[tv][pv] += 1
    print(f"\n  {p} (rows=truth, cols=pred):")
    header = f"  {'':>6}" + "".join(f"{v:>6}" for v in sen_vals)
    print(header)
    for tv in sen_vals:
        row = f"  {tv:>6}"
        for pv in sen_vals:
            row += f"{cm[tv][pv]:>6}"
        print(row)

# ── 4. FINANCIAL CONDITIONS DETAIL ──
print("\n" + "=" * 70)
print("4. FINANCIAL CONDITIONS SENTENCES (all)")
print("=" * 70)
fc_count = 0
fc_correct = defaultdict(int)
for sid in sent_ids:
    truth = eval_data[sid]
    true_top = truth.get("top", [])
    if isinstance(true_top, str): true_top = [true_top]
    if "financial_conditions" not in true_top: continue
    fc_count += 1
    true_sen = truth.get("sen")
    sent = truth["sentence"][:80]
    print(f"\n  \"{sent}\"")
    print(f"    truth_sen={true_sen}", end="")
    for p in prompts:
        pred = raw[p].get(sid, {}).get("parsed", {})
        ps = pred.get("sen", "FAIL") if pred else "FAIL"
        m = " ✓" if str(ps) == str(true_sen) else ""
        print(f"  {p[:4]}={ps}{m}", end="")
        if str(ps) == str(true_sen): fc_correct[p] += 1
    print()
print(f"\n  TOTAL fin_cond sentences: {fc_count}")
for p in prompts:
    print(f"    {p}: {fc_correct[p]}/{fc_count} correct ({fc_correct[p]/fc_count*100:.0f}%)")

# ── 5. TOP ACCURACY DETAIL ──
print("\n" + "=" * 70)
print("5. TOP CLASSIFICATION ERRORS (common patterns)")
print("=" * 70)
for p in prompts:
    error_patterns = defaultdict(int)
    for sid in sent_ids:
        truth = eval_data[sid]
        true_top = truth.get("top", [])
        if isinstance(true_top, list): true_top_s = ",".join(sorted(true_top))
        else: true_top_s = true_top
        pred = raw[p].get(sid, {}).get("parsed", {})
        if not pred: continue
        pred_top = pred.get("top", [])
        if isinstance(pred_top, list): pred_top_s = ",".join(sorted(pred_top))
        else: pred_top_s = pred_top
        if pred_top_s != true_top_s:
            error_patterns[(true_top_s, pred_top_s)] += 1
    print(f"\n  {p} (top 10 error patterns, truth -> pred):")
    for (t, pr), cnt in sorted(error_patterns.items(), key=lambda x: -x[1])[:10]:
        print(f"    {t:<45} -> {pr:<45} ({cnt}x)")

# ── 6. SENTENCE DIFFICULTY ──
print("\n" + "=" * 70)
print("6. SENTENCE DIFFICULTY (across P7 and P7v3)")
print("=" * 70)
two_prompts = ["P7_high_5shot", "P7v3_high_5shot"]
for field in ["sen", "top", "ten", "hor", "com", "ris", "wid"]:
    both_right = both_wrong = disagree = 0
    for sid in sent_ids:
        truth = eval_data[sid]
        tv = truth.get(field)
        if isinstance(tv, list): tv = ",".join(sorted(tv))
        if isinstance(tv, bool): tv = str(tv)
        results = []
        for p in two_prompts:
            pred = raw[p].get(sid, {}).get("parsed", {})
            if not pred: results.append(None); continue
            pv = pred.get(field)
            if isinstance(pv, list): pv = ",".join(sorted(pv))
            if isinstance(pv, bool): pv = str(pv)
            results.append(str(pv) == str(tv))
        if all(r is True for r in results): both_right += 1
        elif all(r is False for r in results if r is not None): both_wrong += 1
        else: disagree += 1
    print(f"  {field}: both_right={both_right} both_wrong={both_wrong} disagree={disagree}")

# ── 7. SEN ERRORS WHERE P7 AND P7v3 DISAGREE ──
print("\n" + "=" * 70)
print("7. SEN: WHERE P7 AND P7v3 DISAGREE (one right, one wrong)")
print("=" * 70)
p7_only = p7v3_only = 0
for sid in sent_ids:
    truth = eval_data[sid]
    tv = str(truth.get("sen"))
    p7_pred = str(raw["P7_high_5shot"].get(sid, {}).get("parsed", {}).get("sen", "FAIL"))
    p7v3_pred = str(raw["P7v3_high_5shot"].get(sid, {}).get("parsed", {}).get("sen", "FAIL"))
    p7_right = p7_pred == tv
    p7v3_right = p7v3_pred == tv
    if p7_right and not p7v3_right:
        p7_only += 1
    elif p7v3_right and not p7_right:
        p7v3_only += 1
print(f"  P7 right, P7v3 wrong: {p7_only}")
print(f"  P7v3 right, P7 wrong: {p7v3_only}")
print(f"  Net advantage: {'P7' if p7_only > p7v3_only else 'P7v3'} by {abs(p7_only - p7v3_only)}")

# Show some examples
print("\n  Examples where P7v3 wins:")
count = 0
for sid in sent_ids:
    truth = eval_data[sid]
    tv = str(truth.get("sen"))
    p7_pred = str(raw["P7_high_5shot"].get(sid, {}).get("parsed", {}).get("sen", "FAIL"))
    p7v3_pred = str(raw["P7v3_high_5shot"].get(sid, {}).get("parsed", {}).get("sen", "FAIL"))
    if not (p7v3_pred == tv and p7_pred != tv): continue
    top = truth.get("top", [])
    if isinstance(top, list): top = ",".join(top)
    print(f"    [{top}] \"{truth['sentence'][:70]}\" truth={tv} P7={p7_pred} P7v3={p7v3_pred}")
    count += 1
    if count >= 10: break

print("\n  Examples where P7 wins:")
count = 0
for sid in sent_ids:
    truth = eval_data[sid]
    tv = str(truth.get("sen"))
    p7_pred = str(raw["P7_high_5shot"].get(sid, {}).get("parsed", {}).get("sen", "FAIL"))
    p7v3_pred = str(raw["P7v3_high_5shot"].get(sid, {}).get("parsed", {}).get("sen", "FAIL"))
    if not (p7_pred == tv and p7v3_pred != tv): continue
    top = truth.get("top", [])
    if isinstance(top, list): top = ",".join(top)
    print(f"    [{top}] \"{truth['sentence'][:70]}\" truth={tv} P7={p7_pred} P7v3={p7v3_pred}")
    count += 1
    if count >= 10: break

print("\nDone.")
