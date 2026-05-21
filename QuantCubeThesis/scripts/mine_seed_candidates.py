"""
mine_seed_candidates.py
-----------------------
Dictionary-based mining of high-confidence candidates for rare/hard labels
from the master unlabelled sentence pool.

Targets (60 examples each → 2 JSON files of 30):
  ris: skewed_downside
  ris: skewed_upside
  wid: elevated
  wid: contested
  sen: strongly_hawkish   (sen = 2, conditions forcing tightening)
  sen: strongly_dovish    (sen = -2, conditions forcing easing)

Usage:
    python scripts/mine_seed_candidates.py
    python scripts/mine_seed_candidates.py --target skewed_downside --n 60
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT       = Path(__file__).parent.parent
POOL_PATH  = ROOT / 'data' / 'all_unlabelled_sentences' / 'master_unlabelled_pool.json'
OUTPUT_DIR = ROOT / 'data' / 'QuantCube_Seed_Labelled'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_PER_TARGET  = 60
BATCH_SIZE    = 30

# ── Keyword dictionaries ──────────────────────────────────────────────────────
# Each entry is a list of (pattern, weight) pairs.
# Sentences are scored by summing weights of all matching patterns.
# Higher weight = stronger signal for that label.

DICTIONARIES = {

    'skewed_downside': [
        # Explicit risk phrases — highest signal
        (r'\bdownside risk', 3),
        (r'\brisks?\b.{0,40}\bdownside\b', 3),
        (r'\btilted?\b.{0,25}\bdownside\b', 3),
        (r'\bweighted?\b.{0,25}\bdownside\b', 3),
        (r'\bskewed?\b.{0,25}\bdownside\b', 3),
        (r'\bto the downside\b', 3),
        (r'\bpredominantly to the downside\b', 4),
        (r'\bweighted to the downside\b', 4),
        # Specific downside concerns
        (r'\bdownside scenario\b', 2),
        (r'\btail risk\b', 2),
        (r'\badverse scenario\b', 2),
        (r'\badverse shock\b', 2),
        (r'\bserious downside\b', 3),
        (r'\brisk of recession\b', 3),
        (r'\brisk of a sharp\b', 2),
        (r'\bpossibility of a more severe\b', 2),
        (r'\bpossibility that.{0,50}worse\b', 2),
        (r'\bremained concerned.{0,40}downside\b', 3),
        (r'\bpotential for.{0,40}deteriorat\b', 2),
        (r'\brisks? remain(?:ed|s)?.{0,30}downside\b', 3),
        # Contextual downside language
        (r'\bmore pessimistic\b', 2),
        (r'\bpersist(?:ent|ence|s)?.{0,20}weakness\b', 1),
        (r'\bdownside surpris\b', 2),
        (r'\bfurther deterior\b', 2),
        (r'\bworse than expected\b', 2),
        (r'\bmore pronounced.{0,25}slowdown\b', 2),
        (r'\bdeepen(?:ing|ed)?.{0,20}recession\b', 2),
    ],

    'skewed_upside': [
        # Explicit risk phrases
        (r'\bupside risk', 3),
        (r'\brisks?\b.{0,40}\bupside\b', 3),
        (r'\btilted?\b.{0,25}\bupside\b', 3),
        (r'\bweighted?\b.{0,25}\bupside\b', 3),
        (r'\bskewed?\b.{0,25}\bupside\b', 3),
        (r'\bto the upside\b', 3),
        (r'\bpredominantly to the upside\b', 4),
        (r'\brisks? remain(?:ed|s)?.{0,30}upside\b', 3),
        # Inflation upside
        (r'\bupside risk.{0,30}inflation\b', 4),
        (r'\binflation.{0,30}upside risk\b', 4),
        (r'\binflationary pressure.{0,30}persist\b', 2),
        (r'\binflationary pressure.{0,30}ris\b', 2),
        (r'\brisk.{0,30}inflation.{0,30}remain.{0,30}elevat\b', 3),
        (r'\bcould rise.{0,40}inflation\b', 2),
        (r'\bfurther.{0,20}acceleration.{0,20}inflation\b', 2),
        (r'\boverheating\b', 3),
        # Stronger-than-expected outcomes
        (r'\bstronger than expected\b', 2),
        (r'\bupside surpris\b', 2),
        (r'\bbetter than anticipated\b', 1),
        (r'\bmore robust than\b', 1),
        (r'\bmonitor(?:ing)?.{0,30}inflation.{0,30}closely\b', 2),
        (r'\bvigilant.{0,30}inflation\b', 3),
        (r'\bhighly accommodative.{0,40}too long\b', 3),
        (r'\brisk of.{0,30}overshoot\b', 3),
    ],

    'wid_elevated': [
        # Primary triggers from annotation guide
        (r'\bhighly uncertain\b', 4),
        (r'\bunusually uncertain\b', 4),
        (r'\belevated uncertainty\b', 4),
        (r'\bconsiderable uncertainty\b', 4),
        (r'\bsignificant uncertainty\b', 3),
        (r'\bsubstantial uncertainty\b', 3),
        (r'\bremained highly uncertain\b', 4),
        (r'\buncertainty remain(?:s|ed)?.{0,20}(?:high|elevated|substantial|considerable)\b', 4),
        # Epistemic limitation language
        (r'\bdifficult to (?:assess|gauge|predict|know|determine|quantify)\b', 3),
        (r'\bhard to (?:assess|gauge|predict|know|determine|quantify)\b', 3),
        (r'\bcannot know\b', 3),
        (r'\bcannot be known\b', 3),
        (r'\bimperfectly understood\b', 4),
        (r'\bnot well understood\b', 3),
        (r'\boutside (?:our|their|historical) experience\b', 4),
        (r'\bunprecedented\b', 3),
        (r'\bhistorical relationships? (?:may|might|could).{0,30}(?:not hold|break down|fail)\b', 4),
        (r'\bwide range of (?:possible |potential )?outcomes\b', 4),
        (r'\bchallenging to (?:assess|forecast|predict|project)\b', 3),
        (r'\bexceptionally uncertain\b', 4),
        (r'\bparticularly uncertain\b', 3),
        (r'\bunusually difficult to (?:assess|predict|gauge)\b', 4),
        (r'\buncertainty (?:surrounding|about|regarding|over).{0,40}(?:unusual|elevated|high|considerable|significant)\b', 3),
        (r'\bextremely uncertain\b', 4),
        (r'\bunknowable\b', 3),
        (r'\bbeyond.{0,20}ability to (?:forecast|predict|assess)\b', 3),
    ],

    'wid_contested': [
        # ── Canonical "two forces cancel" examples from the annotation guide ──
        # CRE/lending: growth strong but standards tightening
        (r'\b(?:robust|strong|solid|grew|growing).{0,80}\btighten(?:ing|ed)\b', 3),
        (r'\btighten(?:ing|ed).{0,80}\b(?:robust|strong|solid|supportive)\b', 3),
        # Near-term vs long-term inflation diverge
        (r'\bnear.term.{0,80}(?:long.term|longer.run|longer-run)', 3),
        (r'\b(?:long.term|longer.run).{0,80}anchor.{0,60}(?:near.term|short.term)', 3),
        # Headline vs core diverge
        (r'\bheadline.{0,80}\bcore\b', 3),
        (r'\bcore.{0,80}\bheadline\b', 3),
        # ── Adversative conjunctions with genuinely opposing economic signals ──
        # "although X [positive], Y [negative]" or vice versa
        (r'\balthough\b.{0,100}\b(?:tighten|declin|fell|weak|slow|deteriorat|contract|lower)\b', 2),
        (r'\balthough\b.{0,100}\b(?:robust|strong|solid|grew|increas|ris|improv|high)\b', 2),
        (r'\beven as\b.{0,100}\b(?:tighten|declin|fell|weak|slow|deteriorat|contract)\b', 2),
        (r'\beven as\b.{0,100}\b(?:robust|strong|solid|grew|increas|ris|improv)\b', 2),
        (r'\bdespite\b.{0,60}\b(?:tighten|declin|fell|weak|slow|deteriorat|contract)\b', 2),
        (r'\bdespite\b.{0,60}\b(?:robust|strong|solid|grew|increas|ris|improv)\b', 2),
        # ── Specific contested pairs ──
        # Strong growth but weak investment or consumption diverge
        (r'\bconsumption.{0,80}\binvestment\b.{0,50}\b(?:fell|declined|weak|slow|soft)\b', 3),
        (r'\binvestment.{0,80}\bconsumption\b.{0,50}\b(?:fell|declined|weak|slow|soft)\b', 3),
        # Labor: openings/employment rising but wages/participation falling
        (r'\bjob openings?.{0,60}(?:wages?|participation|hours)\b', 2),
        (r'\bemployment.{0,60}\b(?:wages?|participation|hours|earnings)\b.{0,60}\b(?:fell|declin|subdued|weak|slow)\b', 3),
        # Financial: asset prices up but credit standards tighten
        (r'\b(?:equity|stock|asset).{0,60}(?:rose|gained|increased).{0,80}(?:credit|lending|borrow).{0,60}tighten\b', 3),
        (r'\bspreads?.{0,50}(?:narrow|compres|tight).{0,80}(?:standard|condition|lend).{0,50}tighten\b', 3),
        # "but" with clear opposing economic forces (not just direction + level)
        (r'\b(?:grew|strong|solid|robust|increased|higher).{0,50}\bbut\b.{0,50}(?:weak|slow|declin|fell|lower|contract|deteriorat)\b', 2),
        (r'\b(?:weak|slow|declin|fell|lower|contract).{0,50}\bbut\b.{0,50}(?:strong|solid|robust|increas|rose|grew|improv)\b', 2),
        # Inflation: some measures high, others low / energy vs core diverge
        (r'\benergy.{0,80}\bcore\b.{0,50}(?:moderat|stable|anchor|low|subside)\b', 3),
        (r'\bcore.{0,80}\benergy\b.{0,50}(?:rose|surge|spike|high|elevated)\b', 3),
        # Mixed signals: "while X, Y" with two distinct economic forces
        (r'\bwhile.{0,50}(?:inflation|price|cost).{0,80}(?:employment|growth|output|activity)\b', 2),
        (r'\bwhile.{0,50}(?:employment|growth|output|activity).{0,80}(?:inflation|price|cost)\b', 2),
        # Short vs long rate divergence (yield curve signal)
        (r'\bshort.term.{0,80}(?:long.term|longer.term|longer.run)\b.{0,50}(?:fell|declin|lower|stable)\b', 3),
        # Survey vs market divergence
        (r'\bsurvey.{0,80}market.{0,50}(?:differ|diverge|contrast|while)\b', 3),
        (r'\bmarket.{0,80}survey.{0,50}(?:differ|diverge|contrast|while)\b', 3),
    ],

    'strongly_hawkish': [
        # Extreme inflation language
        (r'\bsurged?\b', 3),
        (r'\bsoared?\b', 3),
        (r'\bspiked?\b', 2),
        (r'\baccelerated sharply\b', 3),
        (r'\bwell above.{0,30}(?:target|expectation|forecast)\b', 3),
        (r'\bsignificantly above.{0,30}(?:target|expectation|forecast|objective)\b', 3),
        (r'\bfar above.{0,30}target\b', 3),
        (r'\bsubstantially above\b', 2),
        (r'\bmuch higher than\b', 2),
        (r'\bsharp(?:ly)?.{0,20}(?:increas|ris|accelerat).{0,20}(?:inflation|price|cost)\b', 3),
        (r'\bprice.{0,20}(?:surged?|soared?|spiked?|jumped? sharply)\b', 3),
        (r'\binflation.{0,20}(?:surged?|soared?|spiked?|jumped? sharply)\b', 3),
        (r'\bpersistently (?:high|elevated).{0,20}inflation\b', 3),
        (r'\binflation (?:remain(?:s|ed)?|persist(?:s|ed)?).{0,20}(?:well |far |significantly )?above\b', 3),
        (r'\bhighest.{0,20}(?:inflation|price|cpi|pce).{0,30}(?:decade|year|since)\b', 3),
        (r'\b(?:four|40|forty).year high\b', 3),
        (r'\bmulti.(?:decade|year) high.{0,20}inflation\b', 3),
        # Very tight labor market
        (r'\bunemployment.{0,20}(?:historic|record).{0,20}low\b', 3),
        (r'\blabor market.{0,20}(?:extremely|exceptionally|historically) tight\b', 3),
        (r'\bextremely tight.{0,20}labor\b', 3),
        (r'\bvacancies.{0,20}far exceed\b', 2),
        (r'\bjob openings.{0,30}(?:record|historically high|well above)\b', 2),
        # Strong explicit tightening calls
        (r'\baggressive(?:ly)?.{0,20}(?:tighten|raise|hike|restric)\b', 3),
        (r'\bexpeditious(?:ly)?.{0,20}(?:tighten|raise|move|normaliz)\b', 3),
        (r'\braising rates? (?:rapidly|quickly|aggressively|substantially)\b', 3),
        (r'\bfront.load\b', 3),
        (r'\blarge(?:r)?.{0,20}increase.{0,20}(?:rate|policy)\b', 2),
        (r'\b(?:50|75|100).basis.point\b', 2),
        (r'\brestrict(?:ive)?.{0,20}(?:territory|stance|enough)\b', 2),
        (r'\bwell into restrictive\b', 3),
        (r'\boverheat\b', 3),
    ],

    'conditional_commitment': [
        # Explicit condition phrases paired with policy action
        (r'\bif\b.{0,60}\b(?:inflation|employment|economy|conditions?|growth|labor|outlook)\b', 2),
        (r'\bshould\b.{0,60}\b(?:inflation|employment|economy|conditions?|growth|labor|outlook)\b', 2),
        (r'\bas long as\b', 3),
        (r'\buntil\b.{0,60}\b(?:inflation|employment|economy|conditions?|goals?|objectives?|mandate|target|2 percent)', 3),
        (r'\bdepending on\b', 3),
        (r'\bprovided that\b', 3),
        (r'\bcontingent on\b', 3),
        (r'\bin the event that\b', 3),
        (r'\bwere (?:the economy|inflation|conditions?|growth|labor|employment) to\b', 3),
        (r'\bif (?:the economy|inflation|conditions?|growth|labor|employment)\b', 2),
        (r'\bconditional(?:ly)?\b', 3),
        (r'\bsubject to\b.{0,40}\b(?:inflation|employment|conditions?|outlook|data)\b', 2),
        (r'\bif (?:such|this|that|these) (?:condition|progress|trend|development)', 2),
        (r'\bdepend(?:s|ing|ent)? on\b.{0,40}\b(?:data|incoming|economic|inflation)', 2),
        (r'\b(?:will|would) (?:continue|maintain|keep|raise|cut|reduce).{0,60}\b(?:if|until|as long|should|provided|contingent)', 2),
        (r'\bprovided\b.{0,40}\b(?:economy|inflation|conditions?|data|progress)', 2),
        (r'\bif (?:warranted|appropriate|necessary)\b', 3),
        (r'\b(?:appropriate|warranted|necessary).{0,30}\b(?:if|should|depending|given)', 2),
    ],

    'risk_language': [
        # Explicit risk mentions — high priority sentences for ris labels
        (r'\b(?:upside|downside) risk', 3),
        (r'\brisk(?:s)? (?:to|of|that|remain|include|are|were|have)', 3),
        (r'\brisks? (?:are|remain|were|have been).{0,30}\b(?:balanced|skewed|tilted|weighted|elevated)', 3),
        (r'\brisks? to (?:the |our |its )?(?:outlook|forecast|projection|growth|inflation|employment|economy)', 3),
        (r'\bupside risks?\b', 3),
        (r'\bdownside risks?\b', 3),
        (r'\btail risk\b', 3),
        (r'\brisk of (?:recession|deflation|overheating|inflation|stagflation)', 3),
        (r'\brisks? (?:to|on) (?:both sides|the upside|the downside)', 3),
        (r'\bbroadly balanced\b', 2),
        (r'\broughly balanced\b', 2),
        (r'\brisks? (?:appear|seem|judged|viewed|assessed|seen) (?:to be |as )?\b(?:balanced|skewed|tilted)', 3),
        (r'\bposes? (?:a |significant |considerable )?risk', 2),
        (r'\belevated (?:uncertainty|risk)', 2),
        (r'\brisk (?:management|appetite|premium)', 2),
    ],

    'strongly_dovish': [
        # Extreme economic weakness
        (r'\bcollapsed?\b', 3),
        (r'\bplummeted?\b', 3),
        (r'\bprecipitous(?:ly)?\b', 3),
        (r'\bsevere(?:ly)?.{0,20}(?:recession|contraction|weakness|deteriorat|impact)\b', 3),
        (r'\bsharp(?:ly)?.{0,20}(?:contraction|deterioration|decline|drop|fell|fallen)\b', 3),
        (r'\bworst.{0,20}(?:recession|downturn|contraction|since)\b', 3),
        (r'\bdeep(?:en(?:ing)?|er)?.{0,20}recession\b', 3),
        (r'\beconomic crisis\b', 3),
        (r'\bfinancial crisis\b', 2),
        (r'\bpandemic.{0,20}(?:impact|shock|effect|recession)\b', 2),
        (r'\bgreat recession\b', 3),
        (r'\bunprecedented.{0,30}(?:decline|contraction|weakness|impact|shock)\b', 3),
        (r'\bmassive.{0,20}(?:job loss|unemployment|decline)\b', 3),
        (r'\bjob loss(?:es)?.{0,20}(?:millions?|massive|severe|widespread|dramatic)\b', 3),
        (r'\bunemployment.{0,30}(?:surge|soar|spike|peak|highest)\b', 3),
        # Deflation / very low inflation
        (r'\bdeflationary\b', 3),
        (r'\bdeflation risk\b', 3),
        (r'\binflation.{0,30}(?:well |significantly |substantially )?below.{0,20}(?:target|objective|2)\b', 3),
        (r'\bpersistently below.{0,20}(?:target|2 percent|objective)\b', 3),
        (r'\binflation.{0,20}too low\b', 3),
        (r'\bprice stability.{0,30}(?:concern|risk|threat).{0,20}(?:downside|lower|deflation)\b', 2),
        # Explicit aggressive easing
        (r'\baggressive(?:ly)?.{0,20}(?:eas|accommodat|stimulat|lower|cut)\b', 3),
        (r'\bemergency.{0,20}(?:cut|eas|action|rate|meeting)\b', 4),
        (r'\binterpret(?:meeting|session)\b', 2),
        (r'\bunscheduled.{0,20}(?:meeting|action|cut)\b', 3),
        (r'\bmaximum.{0,20}(?:accommodation|stimulus|easing)\b', 3),
        (r'\bhighly accommodative.{0,20}(?:for|stance|policy)\b', 2),
        (r'\bnear.zero.{0,20}(?:rate|bound|policy)\b', 2),
        (r'\beffective lower bound\b', 2),
        (r'\bzero lower bound\b', 2),
        (r'\bquantitative easing\b', 2),
        (r'\bforward guidance.{0,20}(?:commit|pledge|promise|maintain)\b', 2),
        (r'\blarge.scale asset purchase\b', 2),
        (r'\bcut rates? (?:sharply|aggressively|substantially|rapidly|significantly)\b', 3),
        (r'\b(?:50|75|100).basis.point.{0,20}cut\b', 3),
    ],
}


def score_sentence(sentence: str, patterns: list) -> int:
    """Sum weights for all matching patterns (case-insensitive)."""
    text  = sentence.lower()
    score = 0
    for pattern, weight in patterns:
        if re.search(pattern, text):
            score += weight
    return score


def filter_already_labelled(records: list) -> list:
    """Keep only sentences with all label fields empty."""
    return [
        r for r in records
        if not any(r.get(f) for f in ('sen', 'ris', 'wid', 'top'))
    ]


def filter_boilerplate(records: list) -> list:
    """Drop obvious boilerplate: very short sentences or admin lines."""
    return [
        r for r in records
        if len(r.get('sentence', '').split()) >= 8
    ]


def mine(records: list, target: str, n: int = N_PER_TARGET) -> list:
    """Score all sentences and return top-n with score > 0, deduped by text."""
    patterns = DICTIONARIES[target]
    seen     = set()
    scored   = []
    for rec in records:
        text = rec['sentence']
        key  = text.strip().lower()
        if key in seen:
            continue
        sc = score_sentence(text, patterns)
        if sc > 0:
            scored.append((sc, rec))
            seen.add(key)

    scored.sort(key=lambda x: -x[0])

    # Return diverse selection: avoid too many from the same source document
    source_counts = defaultdict(int)
    selected = []
    MAX_PER_SOURCE = 5

    # First pass: pick best from each source
    for sc, rec in scored:
        src = rec.get('source', '')
        if source_counts[src] < MAX_PER_SOURCE:
            selected.append(rec)
            source_counts[src] += 1
        if len(selected) >= n:
            break

    # Second pass: fill up if needed
    if len(selected) < n:
        existing_ids = {r['id'] for r in selected}
        for sc, rec in scored:
            if rec['id'] not in existing_ids:
                selected.append(rec)
                if len(selected) >= n:
                    break

    return selected[:n]


def save_batches(records: list, target: str, batch_size: int = BATCH_SIZE):
    """Save records into JSON files of batch_size each."""
    tag = target.replace('_', '')
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        # Reset label fields so annotator fills them in fresh
        clean_batch = []
        for rec in batch:
            r = {
                'id':               rec['id'],
                'sentence':         rec['sentence'],
                'source':           rec['source'],
                'doc_type':         rec.get('doc_type', ''),
                'date':             rec.get('date', ''),
                'context_question': rec.get('context_question'),
                'top': '', 'sen': '', 'ten': '', 'hor': '',
                'com': '', 'ris': '', 'wid': '',
                '_mine_target':     target,
                '_mine_score':      score_sentence(rec['sentence'],
                                                   DICTIONARIES[target]),
            }
            clean_batch.append(r)

        batch_num = i // batch_size + 1
        fname = OUTPUT_DIR / f'seed_{tag}_batch{batch_num}.json'
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump(clean_batch, f, indent=2, ensure_ascii=False)
        print(f'  Saved {len(clean_batch):>3} sentences -> {fname.name}')


def main(target_filter=None, n=N_PER_TARGET):
    print(f'Loading pool: {POOL_PATH}')
    with open(POOL_PATH, encoding='utf-8', errors='replace') as f:
        all_records = json.load(f)
    print(f'  Total records: {len(all_records):,}')

    records = filter_already_labelled(all_records)
    records = filter_boilerplate(records)
    print(f'  After filtering: {len(records):,} candidates\n')

    targets = [target_filter] if target_filter else list(DICTIONARIES.keys())

    for target in targets:
        print(f'Mining: {target}')
        hits = mine(records, target, n=n)
        print(f'  Found {len(hits)} candidates (requested {n})')

        if len(hits) < n:
            print(f'  WARNING: only {len(hits)} candidates found — '
                  f'consider loosening dictionary for {target}')

        save_batches(hits, target)

        # Print a few examples for sanity check
        print(f'  Top 3 examples:')
        for rec in hits[:3]:
            sc = score_sentence(rec['sentence'], DICTIONARIES[target])
            print(f'    [score={sc}] {rec["sentence"][:120]}')
        print()

    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mine seed candidates for rare labels')
    parser.add_argument('--target', type=str, default=None,
                        choices=list(DICTIONARIES.keys()),
                        help='Mine only this target (default: all)')
    parser.add_argument('--n', type=int, default=N_PER_TARGET,
                        help=f'Sentences per target (default: {N_PER_TARGET})')
    args = parser.parse_args()
    main(target_filter=args.target, n=args.n)
