"""
generate_eval_set.py
────────────────────
Pseudo-label every sentence in the unlabelled pool, then draw a stratified
500-sentence evaluation set for manual annotation.

Pseudo-label sources
────────────────────
  top  Multi-label  Keyword presence across 6 economic domains + monetary_policy
                     (Gardner topic lexicon extended with FOMC-specific terms)
  dom  Single       Dominant topic — domain with most keyword hits
  sen  Single       Gardner total sentiment score → binned to {-2,-1,0,1,2}
  dir  Single       Gardner A9 monetary-policy score + hawkish/dovish lexicon
                     → {very hawkish, hawkish, neutral, dovish, very dovish}
  ten  Single       Tense-marker priority (hypothetical > forward > backward >
                     present > none)
  com  Single       Commitment level via modality keywords
                     → {unconditional, conditional, none}
  hor  Single       Temporal-scope keywords
                     → {short-term, long-term, na}
  con  Multi-label  Condition-referenced domains (same 5 as top minus monetary)
  ris  Single       Risk-direction keywords
                     → {skewed_upside, symmetric, skewed_downside, na}
  wid  Single       Uncertainty / contestedness keywords
                     → {elevated, contested, none}

Memory strategy
───────────────
  Pass 1 — stream full pool with ijson, compute a fast strata key per record,
            store only (id, strata_key) — tiny footprint even for 100k+ records.
  Sample   — proportional stratified sample of 500 from populated strata cells.
  Pass 2 — stream pool again, run full pseudo-labelling on the 500 selected
            records only (Gardner NLP + all dict rules), write output.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import ijson

ROOT      = Path(__file__).parent.parent
POOL_PATH = ROOT / "data" / "all_unlabelled_sentences" / "master_unlabelled_pool.json"
OUT_PATH  = ROOT / "data" / "eval_500_pseudolabelled.json"
N_PER_DOC = 100   # samples per doc type  →  5 × 100 = 500 total
SEED      = 42

sys.path.insert(0, str(ROOT))
from src.training.Gardner import NLP as gardner_nlp, score_monetary_policy


# ── TOPIC KEYWORD SETS ────────────────────────────────────────────────────────
# Used for both `top` (multi-label) and `dom` (dominant single label).
# Keywords are checked as substrings after lowercasing the sentence.

TOPIC_KEYWORDS = {
    "inflation": [
        "inflation", "inflationary", "price level", "price stability",
        "price pressure", "pce", "cpi", "deflation", "disinflation",
        "price index", "consumer price", "core inflation",
    ],
    "employment": [
        "employment", "unemployment", "labor market", "labour market",
        "payroll", "job gain", "job loss", "job opening", "hiring",
        "labor force", "workforce", "worker", "wage", "jobless",
        "nonfarm", "jolts",
    ],
    "economic_activity": [
        "gdp", "output", "economic growth", "economic activity",
        "consumption", "spending", "demand", "production", "investment",
        "business conditions", "business outlook", "retail sales",
        "industrial production", "housing", "export", "import",
        "econom", "recession", "expansion", "recovery",
    ],
    "financial_conditions": [
        "financial condition", "financial market", "financial stability",
        "credit", "spread", "yield curve", "equity", "stock market",
        "bank lending", "liquidity", "funding", "repo", "mortgage",
        "asset price", "leverage", "financial stress", "tightening condition",
        "easing condition",
    ],
    "monetary_policy": [
        "federal funds rate", "fed funds", "policy rate", "interest rate",
        "monetary policy", "accommodation", "taper", "asset purchase",
        "balance sheet", "qe", "quantitative easing", "forward guidance",
        "liftoff", "rate hike", "rate cut", "policy stance", "ioer", "ior",
        "on rrp", "overnight reverse", "discount rate",
    ],
    "macro": [
        "global", "international", "foreign", "fiscal", "government spending",
        "federal budget", "deficit", "national debt", "trade balance",
        "geopolit", "europe", "china", "emerging market", "imf",
        "world economy", "cross-border",
    ],
}


def pseudolabel_topic(sentence: str) -> list:
    s = sentence.lower()
    hits = [
        domain
        for domain, kws in TOPIC_KEYWORDS.items()
        if any(kw in s for kw in kws)
    ]
    return hits if hits else ["no_topic"]


def _dominant_topic(sentence: str) -> str:
    """Internal helper for stratification — not an output label."""
    s = sentence.lower()
    counts = {
        domain: sum(1 for kw in kws if kw in s)
        for domain, kws in TOPIC_KEYWORDS.items()
    }
    best_domain = max(counts, key=counts.get)
    return best_domain if counts[best_domain] > 0 else "no_topic"


# ── TENSE ─────────────────────────────────────────────────────────────────────
# Priority: hypothetical > forward > backward > present > none

_HYP = re.compile(
    r'\b(if\b|were to|should (?:inflation|employment|growth|conditions|'
    r'the economy|it|they|we)|in the event (?:that|of)|contingent|'
    r'assuming that|provided that|suppose|what if|hypothetically)\b',
    re.I,
)
_FWD = re.compile(
    r'\b(will|would|shall|going to|expect\w*|anticipat\w*|project\w*|'
    r'forecast\w*|predict\w*|likely to|expected to|anticipated to|'
    r'poised to|set to|plan to|intend to|may\b|might\b|could\b|'
    r'going forward|ahead|outlook|upcoming|in coming|coming months|'
    r'coming quarters|next (?:year|quarter|month)|by (?:year.end|mid.year)|'
    r'over the (?:next|coming))\b',
    re.I,
)
_BWD = re.compile(
    r'\b(was|were|had\b|rose|fell|declined|increased|decreased|grew|'
    r'contracted|expanded|improved|deteriorated|strengthened|weakened|'
    r'last year|last quarter|last month|in \d{4}|previously|historically|'
    r'in recent (?:months|quarters|years)|over the past|has (?:been|risen|'
    r'fallen|declined|increased|grown|improved)|have (?:been|risen|fallen))\b',
    re.I,
)
_PRS = re.compile(
    r'\b(is\b|are\b|has\b|have\b|continues|remains|stands|represents|'
    r'currently|at present|today|now\b|ongoing|remain\w*)\b',
    re.I,
)


def pseudolabel_tense(sentence: str) -> str:
    if _HYP.search(sentence): return "forward"   # hypothetical is forward-looking
    if _FWD.search(sentence): return "forward"
    if _BWD.search(sentence): return "backward"
    if _PRS.search(sentence): return "present"
    return "na"                                   # boilerplate / indeterminate


# ── DIRECTION (hawkish / dovish) ──────────────────────────────────────────────

_HAWK_STRONG = re.compile(
    r'(significantly restrict\w*|well above (?:neutral|2 percent|target)|'
    r'substantially (?:above|tighten|restrict)\w*|aggressive\w* tighten|'
    r'very restrict\w*|highly restrict\w*|deeply restrict\w*)',
    re.I,
)
_HAWK_MILD = re.compile(
    r'\b(tighten\w*|restrict\w*|normaliz\w*|above neutral|'
    r'above (?:2|two) percent|rate (?:hike|increase|rise|rises)|'
    r'hike\w*|taper\w*|reduc\w* (?:purchase|asset purchase|balance sheet)|'
    r'remov\w* accommodation|end (?:purchase|QE)|'
    r'balance sheet (?:reduction|runoff|normaliz\w*)|'
    r'quantitative tighten|QT\b|lift.?off|liftoff)\b',
    re.I,
)
_DOVE_STRONG = re.compile(
    r'(significant accommodation|substantial accommodation|'
    r'emergency (?:rate )?cut|deeply accommodativ\w*|'
    r'well below (?:neutral|target|2 percent)|lower for longer|'
    r'maximum policy accommodation|unprecedented stimulus)',
    re.I,
)
_DOVE_MILD = re.compile(
    r'\b(accommodat\w*|stimulus\b|eas\w+ (?:monetar|policy)|'
    r'maintain (?:low|accommodat\w*|current) (?:rate|stance|policy)|'
    r'cut (?:rate|interest rate)|lower (?:rate|interest rate)|'
    r'reduc\w* (?:rate|interest rate|fed fund)|zero lower bound|ZLB\b|'
    r'asset purchas\w*|quantitative eas\w*|\bQE\b|'
    r'forward guidance|remain\w* patient|ample accommodation|'
    r'policy support\b|supportive (?:monetary|financial) (?:policy|condition))\b',
    re.I,
)

_SEN_LABELS = ["very dovish", "dovish", "neutral", "hawkish", "very hawkish"]


def pseudolabel_sentiment(sentence: str) -> str:
    """
    Unified hawkish/dovish sentiment for ALL topics.
    Combines Gardner A9 monetary-policy score with the extended hawkish/dovish
    lexicon — this works for MP sentences — and falls back to the Gardner total
    economic sentiment score for non-MP sentences.
    """
    # A9 + extended lexicon (strong signal for MP sentences)
    mp_score = score_monetary_policy(sentence)
    if _HAWK_STRONG.search(sentence): mp_score += 2
    elif _HAWK_MILD.search(sentence):  mp_score += 1
    if _DOVE_STRONG.search(sentence):  mp_score -= 2
    elif _DOVE_MILD.search(sentence):  mp_score -= 1

    # Gardner economic total (signal for non-MP sentences)
    gardner_score = gardner_nlp(sentence)["gardner_total"]
    gardner_bin = (
        -2 if gardner_score <= -0.35 else
        -1 if gardner_score <= -0.10 else
         0 if gardner_score <=  0.10 else
         1 if gardner_score <=  0.35 else 2
    )

    # Use whichever gives a stronger non-zero signal; default to gardner_bin
    score = mp_score if abs(mp_score) >= abs(gardner_bin) else gardner_bin
    score = max(-2, min(2, score))
    return _SEN_LABELS[score + 2]


# ── COMMITMENT ────────────────────────────────────────────────────────────────

_UNCONDITIONAL = re.compile(
    r'\b(will\b|shall\b|commit\w*|pledge\w*|resolv\w* to|determin\w* to|'
    r'will not\b|will maintain|will continue|we are prepared to|'
    r'stand ready|ready to|prepared to)\b',
    re.I,
)
_CONDITIONAL = re.compile(
    r'\b(if\b|provided that|as long as|subject to|contingent|'
    r'in the event|when (?:inflation|employment|conditions|growth)|'
    r'should (?:inflation|conditions|growth|the economy)|'
    r'depending on|based on data|data.?dependent|data.?driven|'
    r'adjust (?:as|based)|monitor\w* and|assess\w* (?:and|whether)|'
    r'conditional on|as appropriate)\b',
    re.I,
)


def pseudolabel_commitment(sentence: str, topics: list | None = None) -> str:
    """Commitment only meaningful for monetary_policy sentences; always 'none' otherwise."""
    if topics is not None and "monetary_policy" not in topics:
        return "none"
    has_uncond = bool(_UNCONDITIONAL.search(sentence))
    has_cond   = bool(_CONDITIONAL.search(sentence))
    if has_uncond and not has_cond: return "unconditional"
    if has_cond:                    return "conditional"
    return "none"


# ── HORIZON ───────────────────────────────────────────────────────────────────

_SHORT_TERM = re.compile(
    r'\b(near.?term|short.?term|next (?:few |coming )?(?:months?|quarters?|weeks?)|'
    r'this (?:year|quarter|month)|over (?:the )?(?:next|coming) (?:few )?'
    r'(?:months?|quarters?)|in (?:the )?(?:near|immediate) future|'
    r'upcoming|imminent|by (?:year.?end|mid.?year|end of the year)|'
    r'in the (?:next|coming) (?:few )?(?:months?|quarters?))\b',
    re.I,
)
_LONG_TERM = re.compile(
    r'\b(long.?(?:er|run|term)|over (?:the )?(?:medium|longer).?(?:term|run)|'
    r'structural|secular|in the (?:long|longer) run|over time\b|'
    r'sustained\w*|persist\w*|decade|years ahead|long.?standing|'
    r'neutral (?:rate|interest rate)|longer.?run (?:rate|level|normal|objective|goal)|'
    r'trend growth|potential output)\b',
    re.I,
)


def pseudolabel_horizon(sentence: str) -> str:
    has_short = bool(_SHORT_TERM.search(sentence))
    has_long  = bool(_LONG_TERM.search(sentence))
    if has_short and not has_long: return "near_term"
    if has_long  and not has_short: return "long_term"
    if has_short and has_long:
        m_s = _SHORT_TERM.search(sentence)
        m_l = _LONG_TERM.search(sentence)
        return "near_term" if m_s.start() < m_l.start() else "long_term"
    return "na"


# ── CONDITION REFERENCED (multi-label) ────────────────────────────────────────

_CON_PATTERNS = {
    "inflation": re.compile(
        r'\b(inflation|price (?:level|stability|target)|pce|cpi|deflation|'
        r'inflationary|disinflation\w*|price pressure|2 percent (?:goal|target|objective))\b',
        re.I,
    ),
    "employment": re.compile(
        r'\b(employment|unemployment|labor market|payroll|job\b|hiring|'
        r'labor force|workforce|workers?|maximum employment)\b',
        re.I,
    ),
    "economic_activity": re.compile(
        r'\b(gdp|output gap|growth|spending|consumption|production|'
        r'economic activity|economic conditions|recession|expansion|recovery)\b',
        re.I,
    ),
    "macro": re.compile(
        r'\b(global|international|foreign|fiscal|government spending|'
        r'deficit|debt|trade|geopolit\w*)\b',
        re.I,
    ),
    "financial_conditions": re.compile(
        r'\b(financial (?:condition|market|stability)|credit condition|'
        r'interest rate spread|yield|equity market|stock market|bank|'
        r'financial stress|financial tighten|financial eas)\b',
        re.I,
    ),
}


def pseudolabel_condition(sentence: str) -> list:
    return [
        domain
        for domain, pattern in _CON_PATTERNS.items()
        if pattern.search(sentence)
    ]


# ── RISK BALANCE ──────────────────────────────────────────────────────────────

_RISK_DOWN = re.compile(
    r'\b(downside risk|risks? (?:are )?(?:weighted )?(?:to the )?downside|'
    r'skewed (?:to the )?downside|weighted (?:to the )?downside|'
    r'headwind\w*|vulnerabilit\w*|deteriorat\w* (?:outlook|conditions|risks?)|'
    r'risks? (?:remain )?elevat\w*|tail risk|significant downside)\b',
    re.I,
)
_RISK_SYM = re.compile(
    r'\b(balanced risks?|symmetric(?:al)?|two.?sided|'
    r'equal\w* (?:balance|probability)|roughly balanced|'
    r'broadly balanced|risks? (?:are )?balanced|'
    r'upside and downside|both (?:upside and downside|directions))\b',
    re.I,
)
_RISK_UP = re.compile(
    r'\b(upside risk|risks? (?:are )?(?:weighted )?(?:to the )?upside|'
    r'skewed (?:to the )?upside|weighted (?:to the )?upside|'
    r'overheat\w*|inflationary pressure|above.target|significant upside)\b',
    re.I,
)


def pseudolabel_risk(sentence: str) -> str:
    has_down = bool(_RISK_DOWN.search(sentence))
    has_sym  = bool(_RISK_SYM.search(sentence))
    has_up   = bool(_RISK_UP.search(sentence))
    if has_sym:  return "symmetric"
    if has_down and not has_up: return "skewed_downside"
    if has_up   and not has_down: return "skewed_upside"
    return "na"


# ── WIDTH / UNCERTAINTY ───────────────────────────────────────────────────────

_ELEVATED_UNC = re.compile(
    r'\b(uncertain\w*|volatil\w*|unpredictabl\w*|'
    r'wide range|significant(?:ly)? uncert\w*|'
    r'hard to (?:predict|assess|know)|difficult to (?:assess|predict|gauge)|'
    r'range of (?:views|outcomes|estimates)|considerable uncertainty|'
    r'marked(?:ly)? uncert\w*|substantial uncertainty|'
    r'unusual(?:ly)? (?:high|large|elevated) uncertainty)\b',
    re.I,
)
_CONTESTED = re.compile(
    r'\b(however\b|on the other hand|but\b|although\b|despite\b|yet\b|'
    r'while\b|notwithstanding|even so|nevertheless|nonetheless|'
    r'mixed\b|conflicting|divergent|debate\b|disagree\w*|'
    r'on balance|in contrast|conversely|that said|having said)\b',
    re.I,
)


def pseudolabel_width(sentence: str) -> str:
    has_unc      = bool(_ELEVATED_UNC.search(sentence))
    has_contested = bool(_CONTESTED.search(sentence))
    if has_unc:       return "elevated"
    if has_contested: return "contested"
    return "none"


# ── FULL PSEUDO-LABELLER ──────────────────────────────────────────────────────

def pseudolabel_all(sentence: str) -> dict:
    top = pseudolabel_topic(sentence)
    raw_hor = pseudolabel_horizon(sentence)
    raw_ten = pseudolabel_tense(sentence)
    return {
        "top": top,
        "sen": pseudolabel_sentiment(sentence),
        "ten": "interpretive" if raw_ten == "forward" else "descriptive",
        "hor": raw_hor == "long_term",   # boolean: True if long_term
        "com": pseudolabel_commitment(sentence, topics=top),
        "ris": pseudolabel_risk(sentence),
        "wid": pseudolabel_width(sentence),
    }


# ── FAST STRATA KEY (pass-1 only) ─────────────────────────────────────────────
# Avoids running Gardner NLP on all 100k+ records in pass 1.

_FAST_POS = re.compile(
    r'\b(improv\w*|strength\w*|solid|robust|strong|gains?|expand\w*|'
    r'elevat\w*|pick.?up|rebound\w*|rose\b|risen\b|higher|increas\w*)\b',
    re.I,
)
_FAST_NEG = re.compile(
    r'\b(declin\w*|weak\w*|slow\w*|below\b|soft\b|modest\b|'
    r'moderat\w*|concern\w*|risk\b|risks\b|deteriorat\w*|fell\b|'
    r'fallen\b|decreas\w*|contract\w*|restrain\w*)\b',
    re.I,
)


def fast_strata_key(sentence: str) -> str:
    dom = _dominant_topic(sentence)  # internal helper, not an output label
    s   = sentence.lower()
    pos = len(_FAST_POS.findall(s))
    neg = len(_FAST_NEG.findall(s))
    diff = pos - neg
    if   diff >= 2:  sen = "2"
    elif diff == 1:  sen = "1"
    elif diff == 0:  sen = "0"
    elif diff == -1: sen = "-1"
    else:            sen = "-2"
    return f"{sen}|{dom}"


# ── STRATIFIED SAMPLING ───────────────────────────────────────────────────────

def stratified_sample(records: list, n: int, seed: int = SEED) -> list:
    """
    Proportional stratified sample of n records from a list of
    (id, strata_key) dicts.  Ensures at least 1 record per non-empty cell.
    """
    import math
    import random as rnd
    rnd.seed(seed)

    by_strata = defaultdict(list)
    for r in records:
        by_strata[r["strata"]].append(r["id"])

    n_strata = len(by_strata)
    total    = len(records)
    allocated = {}
    remainder  = {}

    # Proportional allocation with floor
    raw_alloc = {k: n * len(v) / total for k, v in by_strata.items()}
    for k, v_list in by_strata.items():
        alloc = max(1, int(raw_alloc[k]))
        alloc = min(alloc, len(v_list))
        allocated[k] = alloc
        remainder[k] = raw_alloc[k] - alloc

    # Distribute remaining slots to strata with largest remainders
    current_total = sum(allocated.values())
    slots_left = n - current_total
    if slots_left > 0:
        sorted_strata = sorted(remainder, key=remainder.get, reverse=True)
        for k in sorted_strata[:slots_left]:
            extra = min(1, len(by_strata[k]) - allocated[k])
            allocated[k] += extra

    selected_ids = []
    for k, count in allocated.items():
        ids = by_strata[k]
        chosen = rnd.sample(ids, min(count, len(ids)))
        selected_ids.extend(chosen)

    rnd.shuffle(selected_ids)
    return selected_ids[:n]


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Pass 1: compute fast strata keys, grouped by doc_type ─────────────────
    print("Pass 1: streaming pool to compute strata keys ...")
    by_doc: dict[str, list] = defaultdict(list)
    with open(POOL_PATH, "rb") as f:
        for rec in ijson.items(f, "item"):
            by_doc[rec.get("doc_type", "unknown")].append({
                "id":     rec["id"],
                "strata": fast_strata_key(rec.get("sentence", "")),
            })

    total_pool = sum(len(v) for v in by_doc.values())
    print(f"  Total records in pool: {total_pool:,}")
    for dt, recs in sorted(by_doc.items()):
        print(f"    {dt:<35} {len(recs):>7,}")

    # ── Stratified sampling: 100 per doc type ─────────────────────────────────
    print(f"\nStratified sampling {N_PER_DOC} records per doc type ...")
    selected_ids: set[str] = set()
    for dt, recs in sorted(by_doc.items()):
        n = min(N_PER_DOC, len(recs))
        chosen = stratified_sample(recs, n)
        selected_ids.update(chosen)
        print(f"  {dt:<35} → {len(chosen)} selected (pool={len(recs):,})")
    print(f"  Total selected: {len(selected_ids)}")

    # ── Pass 2: collect + pseudo-label selected records ────────────────────────
    print("\nPass 2: collecting selected records and applying pseudo-labels ...")
    eval_records = []
    with open(POOL_PATH, "rb") as f:
        for rec in ijson.items(f, "item"):
            if rec["id"] not in selected_ids:
                continue
            sentence = rec.get("sentence", "")
            labels   = pseudolabel_all(sentence)
            eval_records.append({
                "id":               rec["id"],
                "sentence":         sentence,
                "source":           rec.get("source", ""),
                "doc_type":         rec.get("doc_type", ""),
                "date":             rec.get("date", ""),
                "context_question": rec.get("context_question"),
                # Pseudo-labels (to be corrected during manual annotation)
                "top": labels["top"],
                "sen": labels["sen"],
                "ten": labels["ten"],
                "hor": labels["hor"],
                "com": labels["com"],
                "ris": labels["ris"],
                "wid": labels["wid"],
            })

    print(f"  Collected {len(eval_records)} records")

    # ── Summary stats ──────────────────────────────────────────────────────────
    from collections import Counter
    doc_dist = Counter(r["doc_type"] for r in eval_records)
    print(f"\n── Records per doc type ────────────────────────────────────────────")
    for dt, n in sorted(doc_dist.items()):
        print(f"  {dt:<35} {n}")

    print("\n── Distribution of pseudo-labels ──────────────────────────────────")
    for field in ["sen", "ten", "hor", "com", "ris", "wid"]:  # con removed
        counts = Counter(r[field] for r in eval_records)
        print(f"  {field:4s}: {dict(sorted(counts.items()))}")

    # ── Save ───────────────────────────────────────────────────────────────────
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_records, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(eval_records)} records → {OUT_PATH}")
