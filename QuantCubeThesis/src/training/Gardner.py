#@title Better Gardner
"""
Gardner, Scotti & Vega (2022) — Corrected Replication
======================================================
Changes vs. original code:
  1. Added Table A10 uninformative sentence filter (was fully missing)
  2. Added Table A9 monetary policy sub-index (was fully missing)
  3. Fixed A8 loop: break replaced with per-category deduplication
  4. Fixed typo in inflation neg modifiers: ease pattern was incorrect
  5. Normalization now uses word count of *filtered* sentences only
"""

import re
import pandas as pd


def sent_tokenize(text: str) -> list[str]:
    """
    Lightweight sentence tokenizer — no external downloads required.
    Splits on '.', '!', '?' followed by whitespace and a capital letter,
    or end-of-string.  Handles common FOMC abbreviations (U.S., p.m., etc.)
    by temporarily masking their periods before splitting.
    """
    # Mask periods inside known abbreviations so they don't trigger splits
    abbrev_pattern = re.compile(
        r'\b(?:U\.S|p\.m|a\.m|Mr|Mrs|Dr|vs|etc|approx|Corp|Ltd|Inc|No)\.',
        re.IGNORECASE
    )
    masked = abbrev_pattern.sub(lambda m: m.group(0).replace('.', '<!DOT!>'), text)

    # Split on sentence-ending punctuation
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z])', masked)

    # Restore masked dots and strip whitespace
    sentences = [s.replace('<!DOT!>', '.').strip() for s in raw if s.strip()]
    return sentences


# ===========================================================================
# TABLE A10 — Uninformative Sentence Filter
#
# A sentence is uninformative if it contains ALL sub-patterns in any one rule.
# Each rule is a list of sub-patterns; ALL must match (conjunctive AND logic).
# Uninformative sentences are removed before ANY scoring takes place.
# The filtered word count is used as the normalization denominator.
# ===========================================================================
A10_RULES = [
    # Policy action consequence sentences
    [r'\bwill\b',    r'\bassess\b',   r'as needed'],
    [r'\bwill\b',    r'\bmonitor\b',  r'as needed'],
    [r'promote a stronger',           r'as announced'],
    [r'\breview\b',  r'\bsize\b',     r'\bcomposition\b'],
    [r'promote a stronger',           r'dual mandate'],
    [r'\bsizable\b', r'still increasing holdings'],
    [r'\brecognize\b',                r'below its 2 percent objective'],
    [r'\bexpect\b',  r'gradual adjustments', r'will\s+\w+\s+(?:strengthen|remain strong)'],
    # Dual mandate restatements
    [r'appropriate policy accommodation', r'dual mandate'],
    [r'dual mandate', r'purchasing additional', r'agency mortgage.backed securities'],
    # Policy toolbox references
    [r'long.term prospects', r'unusual forces',  r'demand abate'],
    [r'sustain\w*\s+\w*\s*expansion',            r'symmetric 2 percent objective'],
    [r'federal reserve', r'employ all available tools|using its balance sheet'],
    [r"today's\s+.*?\s+action",                  r'\bhelp\b'],   # FIX: .*? not \S+
]

# Pre-compile each sub-pattern within each rule
A10_COMPILED = [
    [re.compile(sub, re.IGNORECASE) for sub in rule]
    for rule in A10_RULES
]


def is_uninformative(sentence: str) -> bool:
    """
    Return True if the sentence matches ANY rule in Table A10.
    A rule fires only when ALL of its sub-patterns are found (conjunctive).
    """
    for rule_patterns in A10_COMPILED:
        if all(p.search(sentence) for p in rule_patterns):
            return True
    return False


# ===========================================================================
# TABLE A9 — Monetary Policy Sub-Index
#
# Each rule is a list of sub-patterns that must ALL be present (conjunctive).
# The rule fires and assigns its score to the monetary policy category.
# Applied to the filtered sentence set (after A10 removal).
# ===========================================================================
A9_RULES = [
    # Score -1: accommodative / easing signals
    ([r'policy accommodation',  r'maintained'],                                 -1),
    ([r'continue its purchases'],                                               -1),
    ([r'ready to expand',       r'purchase'],                                   -1),
    ([r'await more evidence',   r'pace of its purchases'],                      -1),
    ([r'\bwill act\b',          r'as needed'],                                  -1),
    # Score 0: neutral / patient signals
    ([r'\bbe patient\b'],                                                         0),
    # Score +1: tightening / normalisation signals
    ([r'\bbelieve\b',           r'policy accommodation', r'removed'],            1),
    ([r'firming',               r'\bneed\b'],                                    1),
    ([r'\bexpects\b',           r'increases in the target range'],               1),
    ([r'\bjudges\b',            r'increases in the target range'],               1),
    ([r'\bwarrant\w*\b',        r'gradual increases'],                           1),
    ([r'balance sheet normalization'],                                            1),
    ([r'end purchase',          r'improvement'],                                  1),
    ([r'\breduce\b',            r'purchase'],                                     1),
    ([r'complete|moderate',     r'purchase',            r'improvement'],          1),
    ([r'decides to',            r'remove policy accommodation'],                  1),
]

# Pre-compile each sub-pattern within each rule
A9_COMPILED = [
    ([re.compile(sub, re.IGNORECASE) for sub in rule_patterns], score)
    for rule_patterns, score in A9_RULES
]


def score_monetary_policy(sentence: str) -> int:
    """
    Score a single sentence for the monetary policy sub-index (Table A9).
    Returns the score of the first matching rule, or 0 if no rule fires.
    Only the first matching rule is applied per sentence (one score per
    sentence, consistent with the paper's phrase-based approach).
    """
    for rule_patterns, score in A9_COMPILED:
        if all(p.search(sentence) for p in rule_patterns):
            return score
    return 0


# ===========================================================================
# GARDNER TOPICS — Table A5
# ===========================================================================
gardner_topics = [
    # Inflation
    {"keyword": "inflation",  "type": "word",   "score":  1, "category": "inf"},
    {"keyword": "price",      "type": "word",   "score":  1, "category": "inf"},
    {"keyword": "cost",       "type": "word",   "score":  1, "category": "inf"},
    # Labor
    {"keyword": "employers",  "type": "word",   "score":  1, "category": "labor"},
    {"keyword": "employment", "type": "word",   "score":  1, "category": "labor"},
    {"keyword": "job gains",  "type": "phrase", "score":  1, "category": "labor"},
    {"keyword": "job losses", "type": "phrase", "score": -1, "category": "labor"},
    {"keyword": "labor",      "type": "word",   "score":  1, "category": "labor"},
    {"keyword": "hiring",     "type": "word",   "score":  1, "category": "labor"},
    {"keyword": "underutilization of labor resources",
                              "type": "phrase", "score": -1, "category": "labor"},
    {"keyword": "unemployment",
                              "type": "word",   "score": -1, "category": "labor"},
    {"keyword": "utilization of the pool of available workers",
                              "type": "phrase", "score":  1, "category": "labor"},
    # Output
    {"keyword": "business conditions",
                              "type": "phrase", "score":  1, "category": "out"},
    {"keyword": "business outlook",
                              "type": "phrase", "score":  1, "category": "out"},
    {"keyword": "confidence", "type": "word",   "score":  1, "category": "out"},
    {"keyword": "consumption","type": "word",   "score":  1, "category": "out"},
    {"keyword": "strengthening in final demand",
                              "type": "phrase", "score":  1, "category": "out"},
    {"keyword": "demand",     "type": "word",   "score":  1, "category": "out"},
    {"keyword": "econom",     "type": "stem",   "score":  1, "category": "out"},
    {"keyword": "expenditures","type": "word",  "score":  1, "category": "out"},
    {"keyword": "export",     "type": "word",   "score":  1, "category": "out"},
    {"keyword": "income",     "type": "word",   "score":  1, "category": "out"},
    {"keyword": "indicators", "type": "word",   "score":  1, "category": "out"},
    {"keyword": "investment spending",
                              "type": "phrase", "score":  1, "category": "out"},
    {"keyword": "investment", "type": "word",   "score":  1, "category": "out"},
    {"keyword": "output",     "type": "word",   "score":  1, "category": "out"},
    {"keyword": "production", "type": "word",   "score":  1, "category": "out"},
    {"keyword": "sales",      "type": "word",   "score":  1, "category": "out"},
    {"keyword": "sentiment",  "type": "word",   "score":  1, "category": "out"},
    {"keyword": "spending",   "type": "word",   "score":  1, "category": "out"},
    # Financial
    {"keyword": "bank lending","type": "phrase","score":  1, "category": "fin"},
    {"keyword": "credit",     "type": "word",   "score":  1, "category": "fin"},
    {"keyword": "financial",  "type": "word",   "score":  1, "category": "fin"},
]

gardner_map = {item["keyword"].lower(): item for item in gardner_topics}


def _topic_to_regex(item):
    k = re.escape(item["keyword"].lower())
    if item["type"] == "stem":   return k + r'\w*'
    elif item["type"] == "word": return r'\b' + k + r'\b'
    else:                        return k


sorted_topics = sorted(gardner_topics, key=lambda x: len(x["keyword"]), reverse=True)
topic_pattern = re.compile(
    '(?:' + '|'.join(_topic_to_regex(item) for item in sorted_topics) + ')',
    re.IGNORECASE
)


# ===========================================================================
# MODIFIER BUILDER
# ===========================================================================
def to_regex(stems=None, words=None, phrases=None, special=None):
    patterns = []
    for s in (stems   or []): patterns.append(re.escape(s) + r'\w*')
    for w in (words   or []): patterns.append(r'\b' + re.escape(w) + r'\b')
    for p in (phrases or []): patterns.append(re.escape(p))
    for r_ in (special or []): patterns.append(r_)
    return patterns


# ===========================================================================
# MODIFIER DICTIONARIES — Tables A6, A7, A8
# FIX: ease pattern in inf neg modifiers was incorrect in original code
# ===========================================================================
sentiment_maps = {

    # ── Labor (Table A6) ─────────────────────────────────────────────────────
    "labor": {
        "pos": to_regex(
            stems   = ["elevat", "expand", "improv", "increas", "rebound",
                       "strength"],
            words   = ["gains", "high", "rise", "rising", "rose", "risen",
                       "solid", "strong", "upward", "up"],
            phrases = ["pick up", "picking up", "picked up", "record expansion"],
        ),
        "neu": to_regex(
            words   = ["balance", "mix", "stable", "stabilizing", "steady",
                       "unchanged"],
            phrases = ["little change"],
        ),
        "neg": to_regex(
            stems   = ["declin", "deteriorat", "diminish", "disappoint",
                       "inhibit", "restrain", "underutiliz"],
            words   = ["losses", "low", "modest", "moderated", "slow",
                       "subdued", "weak"],
            phrases = ["reluctant to add", "set back"],
            special = [r'\bsoft\b(?!ware)'],
        ),
    },

    # ── Inflation (Table A6 continued) ───────────────────────────────────────
    "inf": {
        "pos": to_regex(
            stems   = ["elevat", "expand", "foster", "height", "improv",
                       "increas", "persist", "sustain", "strength", "high"],
            words   = ["pressure", "rise", "rising", "rose", "risen",
                       "solid", "strong", "upward", "up"],
            phrases = ["pick up", "picking up", "picked up",
                       "moderate ",           # "moderate (space)"
                       "risk remain", "upside risk"],
        ),
        "neu": to_regex(
            words   = ["balance", "contain", "stable", "stabilizing", "steady",
                       "unchanged", "volatility", "uncertain"],
            phrases = ["equal probability", "little change"],
        ),
        "neg": to_regex(
            stems   = ["damp", "declin", "diminish", "restrain"],
            words   = ["below", "down", "easing", "low", "modest", "moderated",
                       "muted", "reduction", "slow", "subdued", "weak"],
            phrases = ["set back"],
            special = [r'\bease\b',             # FIX: original code had a typo here
                       r'\bsoft\b(?!ware)'],
        ),
    },

    # ── Financial Conditions (Table A7) ──────────────────────────────────────
    "fin": {
        "pos": to_regex(
            words   = ["supportive"],
        ),
        "neu": to_regex(
            words   = ["unchanged"],
        ),
        "neg": to_regex(
            stems   = ["strain", "stress"],
            words   = ["tight", "volatile", "turmoil"],
        ),
    },

    # ── Output (Table A7) ────────────────────────────────────────────────────
    "out": {
        "pos": to_regex(
            stems   = ["advanc", "bolster", "expand", "improv", "increas",
                       "rebound", "strength"],
            words   = ["gains", "high", "moderate", "rise", "rising", "rose",
                       "risen", "solid", "strong", "upward"],
            phrases = ["growing at a moderate pace", "remains firm", "firm",
                       "firmer", "grow at a solid pace",
                       "pick up", "picking up", "picked up", "record expansion"],
        ),
        "neu": to_regex(
            words   = ["abating", "balance", "evolve", "mix", "same",
                       "stable", "stabilizing", "sustain", "tentative"],
            phrases = ["leveling out", "temporarily depressed"],
        ),
        "neg": to_regex(
            stems   = ["contract", "damp", "decelerat", "depress", "declin",
                       "deteriorat", "diminish", "disappoint", "erod",
                       "inhibit", "restrain", "sluggish", "uncertain", "weigh"],
            words   = ["below", "cooling", "cut", "down", "drag", "flat",
                       "gap", "hesitancy", "low", "modest", "pause",
                       "reduction", "shortfall", "slow", "slump", "subdued",
                       "weak"],
            phrases = ["dislocation", "disruption", "increasing less rapidly",
                       "might not be strong enough", "moderating", "moderation",
                       "moderated", "remain moderate", "more moderate",
                       "set back", "yet to exhibit sustainable growth",
                       "weigh on", "weighing on"],
            special = [r'\bsoft\b(?!ware)'],
        ),
    },
}


# ===========================================================================
# TABLE A8 — Long-phrase sentence-level overrides
# ===========================================================================
A8_RULES = [
    # Labor positive
    (r'downside risks? to the outlook.*labor market.*diminish',                 1, "labor"),
    (r'declined? but remains elevated',                                          1, "labor"),
    (r'declined? notably in recent months but remains elevated',                 1, "labor"),
    (r'declined? somewhat since the summer.*remains elevated',                   1, "labor"),
    (r'deterioration in (?:the )?labor market is abating',                       1, "labor"),
    (r'underutilization of labor resources continues to diminish',               1, "labor"),
    # Labor neutral
    (r'although job losses have slowed.*new hiring has lagged',                  0, "labor"),
    # Inflation negative
    (r'despite the rise in energy prices.*inflation.*expectations have eased',  -1, "inf"),
    (r'the risk of inflation becoming undesirably low',                         -1, "inf"),
    # Output negative
    (r'pace of economic recovery is likely to be modest',                       -1, "out"),
    (r'recovery is continuing.*insufficient to bring down unemployment',        -1, "out"),
    (r'recovery is continuing.*insufficient.*significant improvement in labor', -1, "out"),
    (r'recovery is continuing at a moderate pace.*more slowly',                 -1, "out"),
    (r'expanding.*remains constrained',                                         -1, "out"),
    (r'stabilizing but remains constrained',                                    -1, "out"),
    (r'picked up recently but remains constrained',                             -1, "out"),
    (r'increasing.*but remains constrained',                                    -1, "out"),
    (r'increasing gradually.*remains constrained',                              -1, "out"),
    (r'solid pace of spending growth has slowed',                               -1, "out"),
    (r'rising.*though less rapidly than earlier in the year',                   -1, "out"),
    (r'picked up late last year.*remains constrained',                          -1, "out"),
    (r'rising at a somewhat slower pace',                                       -1, "out"),
    (r'improvement.*from a depressed level',                                    -1, "out"),
    # Output neutral
    (r'fiscal policy is restraining economic growth.*extent of restraint',       0, "out"),
    (r'hurricane.related disruptions',                                           0, "out"),
    (r'warrant.*(?:keeping|exceptionally low).*federal funds rate',              0, "out"),
    (r'expand for a time at a pace below the productivity',                      0, "out"),
]

A8_COMPILED = [(re.compile(p, re.IGNORECASE), score, cat)
               for p, score, cat in A8_RULES]


# ===========================================================================
# COMPILE MODIFIER PATTERNS
# ===========================================================================
sentiment_compiled = {}
for category, dicts in sentiment_maps.items():
    sentiment_compiled[category] = {
        pol: re.compile('|'.join(f'(?:{p})' for p in patterns), re.IGNORECASE)
        for pol, patterns in dicts.items()
        if patterns
    }


# ===========================================================================
# HELPER — closest modifier within a sentence
# ===========================================================================
def get_closest_modifier_score(clean_str, topic_token_idx, category,
                               topic_start, topic_end):
    """
    Find the closest sentiment modifier to the topic keyword within a sentence.
    Modifiers overlapping the topic span are skipped.
    Ties between equidistant modifiers of opposite polarity are neutralised.
    """
    closest_dist = float('inf')
    best_score   = None

    polarity_map = {"pos": 1, "neu": 0, "neg": -1}

    for pol, score in polarity_map.items():
        pattern = sentiment_compiled[category].get(pol)
        if pattern is None:
            continue
        for m in pattern.finditer(clean_str):
            if m.start() >= topic_start and m.end() <= topic_end:
                continue
            match_word_idx = len(clean_str[:m.start()].split())
            dist = abs(topic_token_idx - match_word_idx)

            if dist < closest_dist:
                closest_dist = dist
                best_score   = score
            elif dist == closest_dist and score != best_score:
                best_score = 0  # equidistant tie — neutralise

    return best_score if best_score is not None else 0


def preprocess(sentence: str) -> str:
    s = sentence.lower()
    s = re.sub(r'-', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)
    return s


# ===========================================================================
# MAIN NLP FUNCTION
# ===========================================================================
def NLP(text: str) -> pd.Series:
    """
    Gardner et al. (2021) replication — Tables A5, A6, A7, A8, A9, A10.

    Pipeline
    --------
    Step 0 : Tokenise into sentences.
    Step 1 : Filter uninformative sentences (Table A10) — conjunctive AND rules.
             Word count of remaining sentences used as normalisation denominator.
    Step 2 : Score monetary policy sub-index (Table A9) — conjunctive AND rules,
             first matching rule per sentence.
    Step 3 : Apply A8 sentence-level phrase overrides to remaining sentences.
             FIX: per-category deduplication instead of global break — a sentence
             can contribute to at most one A8 score per category, but multiple
             categories can fire from the same sentence.
    Step 4 : For sentences not fully handled by A8, apply topic-keyword +
             closest-modifier logic (Tables A5 / A6 / A7).
    Step 5 : Normalise each sub-index by sqrt(filtered word count).

    Returns
    -------
    pd.Series with keys:
        gardner_inf, gardner_labor, gardner_out, gardner_fin,
        gardner_mp, gardner_total
    All values are normalised by sqrt(filtered word count).
    """
    sentences = sent_tokenize(text)

    # ── Step 1: Remove uninformative sentences ────────────────────────────────
    filtered_sentences = [s for s in sentences if not is_uninformative(s)]

    # Word count denominator uses only filtered sentences
    filtered_text = ' '.join(filtered_sentences)
    word_count    = len(filtered_text.split())
    normaliser    = (word_count ** 0.5) if word_count > 0 else 1.0

    # ── Step 2: Monetary policy sub-index (Table A9) ──────────────────────────
    # Applied to filtered sentences; one score per sentence (first match wins).
    mp_total = 0
    for sentence in filtered_sentences:
        mp_total += score_monetary_policy(sentence)

    # ── Steps 3 & 4: Economic topic scoring ───────────────────────────────────
    totals = {"inf": 0, "labor": 0, "out": 0, "fin": 0}

    for sentence in filtered_sentences:
        clean_str = preprocess(sentence)

        # ── Step 3: A8 long-phrase overrides ─────────────────────────────────
        # FIX: track which categories have already been scored by A8 this
        # sentence, rather than breaking after the first match globally.
        # A sentence can match at most one A8 rule *per category*.
        a8_scored_cats = set()
        for pattern, score, cat in A8_COMPILED:
            if cat not in a8_scored_cats and pattern.search(clean_str):
                totals[cat] += score
                a8_scored_cats.add(cat)

        # ── Step 4: Topic-keyword + proximity modifier ────────────────────────
        # Only applied to categories not already handled by A8 in this sentence.
        for match in topic_pattern.finditer(clean_str):
            found_word = match.group(0).lower()

            topic_info = gardner_map.get(found_word)
            if topic_info is None:
                topic_info = next(
                    (item for item in sorted_topics
                     if item["type"] == "stem"
                     and found_word.startswith(item["keyword"].lower())),
                    None
                )
            if topic_info is None:
                continue

            # Skip topic if its category was already scored via A8 this sentence
            if topic_info["category"] in a8_scored_cats:
                continue

            topic_token_idx = len(clean_str[:match.start()].split())
            mod_score = get_closest_modifier_score(
                clean_str, topic_token_idx, topic_info["category"],
                match.start(), match.end()
            )
            totals[topic_info["category"]] += mod_score * topic_info["score"]

    # ── Step 5: Normalise by sqrt(filtered word count) ────────────────────────
    inf_norm   = totals["inf"]   / normaliser
    labor_norm = totals["labor"] / normaliser
    out_norm   = totals["out"]   / normaliser
    fin_norm   = totals["fin"]   / normaliser
    mp_norm    = mp_total        / normaliser
    total_norm = inf_norm + labor_norm + out_norm + fin_norm + mp_norm

    return pd.Series({
        'gardner_inf':   inf_norm,
        'gardner_labor': labor_norm,
        'gardner_out':   out_norm,
        'gardner_fin':   fin_norm,
        'gardner_mp':    mp_norm,
        'gardner_total': total_norm,
    })
