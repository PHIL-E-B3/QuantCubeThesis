"""
Gardner Baseline Evaluation
============================
Applies the Gardner et al. (2022) dictionary approach to the eval set
and computes F1 scores for sentiment and topic, comparable to our LLM.

Usage:
    python scripts/gardner_eval.py \
        --eval-set data/eval_labelled_merged_corrected.json
"""
import argparse
import json
import sys
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training"))
from Gardner import NLP, topic_pattern, gardner_map, sorted_topics

# Map Gardner categories → eval topic labels
GARDNER_TO_TOPIC = {
    "inf":   "inflation",
    "labor": "labor_market",
    "out":   "economic_activity",
    "fin":   "financial_conditions",
    "mp":    "monetary_policy",
}

SENTIMENT_5 = ["strongly_dovish", "dovish", "neutral", "hawkish", "strongly_hawkish"]
SENTIMENT_3 = ["dovish", "neutral", "hawkish"]

def collapse_sentiment(s):
    if s in ("strongly_dovish",): return "dovish"
    if s in ("strongly_hawkish",): return "hawkish"
    return s


def gardner_to_sentiment(total, strong_thresh=0.05):
    """Map gardner_total score to a sentiment label."""
    if total > strong_thresh:   return "strongly_hawkish"
    if total > 0:               return "hawkish"
    if total < -strong_thresh:  return "strongly_dovish"
    if total < 0:               return "dovish"
    return "neutral"


def gardner_to_topics(scores):
    """Return list of topic labels for non-zero Gardner category scores."""
    topics = [label for cat, label in GARDNER_TO_TOPIC.items()
               if scores.get(f"gardner_{cat}", 0) != 0]
    return topics if topics else ["no_topic"]


def gardner_to_topics_keyword(sentence):
    """
    Detect topics by keyword presence alone — no modifier required.
    Uses Gardner's own compiled topic regex, just without requiring a sentiment score.
    """
    import re
    clean = sentence.lower()
    clean = re.sub(r'-', ' ', clean)
    clean = re.sub(r'[^\w\s]', '', clean)

    found_cats = set()
    for match in topic_pattern.finditer(clean):
        word = match.group(0).lower()
        info = gardner_map.get(word)
        if info is None:
            info = next(
                (item for item in sorted_topics
                 if item["type"] == "stem" and word.startswith(item["keyword"].lower())),
                None
            )
        if info:
            found_cats.add(info["category"])

    topics = [GARDNER_TO_TOPIC[cat] for cat in found_cats if cat in GARDNER_TO_TOPIC]
    return topics if topics else ["no_topic"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default="data/eval_labelled_merged_corrected.json")
    parser.add_argument("--strong-thresh", type=float, default=0.05,
                        help="|gardner_total| threshold for strongly_hawkish/dovish")
    args = parser.parse_args()

    eval_path = PROJECT_ROOT / args.eval_set
    eval_data = json.load(open(eval_path, encoding="utf-8"))
    print(f"Loaded {len(eval_data)} sentences")

    results = []
    for r in eval_data:
        sentence = r.get("sentence") or ""
        scores = NLP(sentence).to_dict()

        sen_gt = str(r.get("sen", "na")).lower().strip()
        top_gt = [t.lower() for t in (r.get("top") or [])]

        sen_pred = gardner_to_sentiment(scores["gardner_total"], args.strong_thresh)
        top_pred_score   = gardner_to_topics(scores)
        top_pred_keyword = gardner_to_topics_keyword(sentence)

        results.append({
            "id": r["id"],
            "sen_gt": sen_gt,
            "top_gt": top_gt,
            "sen_pred": sen_pred,
            "top_pred_score":   top_pred_score,
            "top_pred_keyword": top_pred_keyword,
            "gardner_total": scores["gardner_total"],
        })

    # ── Sentiment F1 ──────────────────────────────────────────────────────────
    # Filter out 'na'
    sen_rows = [r for r in results if r["sen_gt"] not in ("na", "")]

    # 5-class
    y_true_5 = [r["sen_gt"]  for r in sen_rows]
    y_pred_5 = [r["sen_pred"] for r in sen_rows]
    f1_5 = f1_score(y_true_5, y_pred_5, average="macro",
                    labels=SENTIMENT_5, zero_division=0)

    # 3-class (merge strongly variants)
    y_true_3 = [collapse_sentiment(r["sen_gt"])  for r in sen_rows]
    y_pred_3 = [collapse_sentiment(r["sen_pred"]) for r in sen_rows]
    f1_3 = f1_score(y_true_3, y_pred_3, average="macro",
                    labels=SENTIMENT_3, zero_division=0)

    print(f"\n{'='*65}")
    print("SENTIMENT — Gardner Baseline")
    print(f"{'='*65}")
    print(f"  5-class macro F1: {f1_5:.4f}  (n={len(sen_rows)})")
    print(f"  3-class macro F1: {f1_3:.4f}  (strongly_* merged)")
    print()
    print("  5-class breakdown:")
    print(classification_report(y_true_5, y_pred_5, labels=SENTIMENT_5,
                                zero_division=0, digits=3))
    print("  3-class breakdown:")
    print(classification_report(y_true_3, y_pred_3, labels=SENTIMENT_3,
                                zero_division=0, digits=3))

    # Coverage: how many sentences get a non-neutral prediction?
    non_neutral = sum(1 for r in sen_rows if r["sen_pred"] != "neutral")
    print(f"  Coverage (non-neutral predictions): {non_neutral}/{len(sen_rows)} "
          f"({100*non_neutral/len(sen_rows):.1f}%)")
    print(f"  Prediction distribution: {Counter(r['sen_pred'] for r in sen_rows)}")

    # ── Topic F1 (multi-label Jaccard + macro F1) ─────────────────────────────
    top_rows = [r for r in results if r["top_gt"]]

    ALL_TOPICS = ["inflation", "monetary_policy", "labor_market",
                  "economic_activity", "financial_conditions", "no_topic"]

    def eval_topics(pred_key, label):
        mlb = MultiLabelBinarizer(classes=ALL_TOPICS)
        Y_true = mlb.fit_transform([r["top_gt"]       for r in top_rows])
        Y_pred = mlb.transform(    [r[pred_key]        for r in top_rows])

        jaccard_scores = []
        for yt, yp in zip(Y_true, Y_pred):
            inter = (yt & yp).sum()
            union = (yt | yp).sum()
            jaccard_scores.append(inter / union if union > 0 else 0.0)

        print(f"\n{'='*65}")
        print(f"TOPIC — {label}")
        print(f"{'='*65}")
        print(f"  Jaccard (mean):  {np.mean(jaccard_scores):.4f}")
        print(f"  Macro F1:        {f1_score(Y_true, Y_pred, average='macro', zero_division=0):.4f}")
        print(f"  Micro F1:        {f1_score(Y_true, Y_pred, average='micro', zero_division=0):.4f}  (n={len(top_rows)})")
        print()
        for i, lbl in enumerate(ALL_TOPICS):
            p  = Y_pred[:, i].sum()
            t  = Y_true[:, i].sum()
            tp = (Y_true[:, i] & Y_pred[:, i]).sum()
            prec = tp / p if p else 0
            rec  = tp / t if t else 0
            f1_l = 2*prec*rec/(prec+rec) if (prec+rec) else 0
            print(f"  {lbl:<25} P={prec:.3f}  R={rec:.3f}  F1={f1_l:.3f}  "
                  f"pred={int(p)}  true={int(t)}")
        return float(np.mean(jaccard_scores)), f1_score(Y_true, Y_pred, average="macro", zero_division=0)

    jaccard_score, f1_topic_macro       = eval_topics("top_pred_score",   "Gardner score-based (requires modifier)")
    jaccard_kw,    f1_topic_macro_kw    = eval_topics("top_pred_keyword", "Gardner keyword-based (presence only)")

    # ── LLM 3-class sentiment ─────────────────────────────────────────────────
    gt_by_id = {r["id"]: str(r.get("sen", "na")).lower().strip() for r in eval_data}

    def llm_f1_3class(raw_path):
        raw = json.load(open(raw_path, encoding="utf-8"))
        y_true, y_pred = [], []
        for r in raw:
            gt = gt_by_id.get(r["id"], "na")
            if gt in ("na", ""):
                continue
            pred = str(r["parsed"].get("sentiment", "na")).lower().strip()
            y_true.append(collapse_sentiment(gt))
            y_pred.append(collapse_sentiment(pred))
        return f1_score(y_true, y_pred, average="macro", labels=SENTIMENT_3, zero_division=0)

    res_dir = PROJECT_ROOT / "models/best/eval_results"
    base_f1_3 = llm_f1_3class(res_dir / "base_best_raw.json")
    sft_f1_3  = llm_f1_3class(res_dir / "sft_best_raw.json")

    # ── Summary comparison table ───────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("SUMMARY — Gardner vs LLM (from comparison.json)")
    print(f"{'='*65}")
    cmp_path = PROJECT_ROOT / "models/best/eval_results/comparison.json"
    if cmp_path.exists():
        cmp = json.load(open(cmp_path))
        sft = cmp["sft"]["field_metrics"]
        base = cmp["base"]["field_metrics"]
        print(f"  {'Metric':<35} {'Gardner(score)':>14} {'Gardner(kw)':>12} {'Base LLM':>10} {'SFT LLM':>10}")
        print(f"  {'-'*83}")
        print(f"  {'Sentiment macro F1 (5-class)':<35} {f1_5:>14.4f} {'':>12} "
              f"{base['sentiment']['f1_macro']:>10.4f} {sft['sentiment']['f1_macro']:>10.4f}")
        print(f"  {'Sentiment macro F1 (3-class)':<35} {f1_3:>14.4f} {'':>12} "
              f"{base_f1_3:>10.4f} {sft_f1_3:>10.4f}")
        print(f"  {'Topic Jaccard mean':<35} {jaccard_score:>14.4f} {jaccard_kw:>12.4f} "
              f"{base['topic']['jaccard_mean']:>10.4f} {sft['topic']['jaccard_mean']:>10.4f}")
        print(f"  {'Topic macro F1':<35} {f1_topic_macro:>14.4f} {f1_topic_macro_kw:>12.4f} "
              f"{base['topic']['f1_macro']:>10.4f} {sft['topic']['f1_macro']:>10.4f}")


if __name__ == "__main__":
    main()
