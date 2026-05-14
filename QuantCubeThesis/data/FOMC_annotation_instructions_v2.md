# FOMC Sentence Annotation Instructions v2

## Overview

You will be given a JSON file containing sentences extracted from FOMC communications (meeting minutes, speeches, statements, press conferences). Each sentence has an `id`, `sentence`, `source`, `doc_type`, and `date` field. All other fields are empty and must be filled in according to the rules below.

Return the complete JSON file with all fields populated. Never leave a field empty.

---

## Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `top` | array of strings | Topic(s) present in the sentence — includes conditions when topic is monetary_policy |
| `sen` | integer or "na" | Hawkish/dovish incentive score (-2 to 2, or "na") |
| `ten` | string | Temporal orientation: `"descriptive"` or `"interpretive"` |
| `hor` | boolean | Long-term horizon present: `true` or `false` |
| `com` | string | Commitment level (monetary policy only, else "none") |
| `ris` | string | Risk balance |
| `wid` | string | Distribution width |

---

## FIELD 1: `top` — Topic (multi-select array)

Select all topics explicitly present in the sentence. A sentence can have multiple topics (e.g., `["inflation", "unemployment"]`). When the sentence is about monetary policy AND references specific economic conditions as justification, include both `"monetary_policy"` and the referenced topic(s) (e.g., `["monetary_policy", "inflation", "unemployment"]`). The separate `con` field has been removed — all topic information lives here.

| Label | When to use |
|-------|-------------|
| `inflation` | PCE, CPI, price levels, inflation expectations, price stability as a data observation |
| `unemployment` | Labor market, payrolls, jobless claims, participation rate, wage growth, job openings |
| `economic_activity` | GDP, output, consumption, investment, housing, trade, exports, retail sales — specific sector measures |
| `macro` | Aggregate economic outlook where no single variable dominates. Use when the sentence references "economic conditions", "economic outlook", "economic developments", or similar aggregate phrases without specifying a single dominant variable. Do not conflate with `economic_activity` — `macro` is for unspecified aggregate framing. |
| `financial_conditions` | Credit markets, spreads, asset prices, bank lending, financial stability, money markets, funding conditions |
| `monetary_policy` | Rate decisions, asset purchases, balance sheet, forward guidance, policy stance — only when the sentence is explicitly about what the Fed is doing or will do with its policy tools. When chosen, choose ONLY this topic. |
| `boilerplate` | Procedural, administrative, legal directives, section headers, vote announcements, clearly formulaic language with no substantive policy signal |
| `no_topic` | Generic statements that convey no specific economic content but may still carry risk or uncertainty signal |

**Key distinctions:**
- Use `macro` only when truly no single variable dominates. If a multi-variable sentence has a clear dominant conclusion (e.g., "risks to real activity"), label that variable instead.
- `boilerplate` examples: vote announcements, legal/regulatory language, section headers, memorial tributes, formulaic monitoring pledges ("The Committee will continue to monitor..."), and generic dual-mandate reaffirmations ("The Federal Reserve is committed to both sides of its mandate") that carry no specific directional signal. Test: does removing this sentence lose any policy-relevant information? If no, it is boilerplate.
- `no_topic` is for generic sentences that may still carry risk or uncertainty (`ris` and `wid` still apply). `boilerplate` is for purely procedural/administrative sentences (default `ris = "na"`, `wid = "none"`).
- **Data dependence language:** "In determining the extent of additional policy firming, the Committee will take into account incoming data" is `boilerplate` if standalone and generic. If the sentence names a specific policy action or specific variables being monitored, label as `["monetary_policy", "macro"]` with `com = "conditional"`.

---

## FIELD 2: `ten` — Temporal Orientation (binary)

Captures whether the sentence's primary content is **descriptive** (reporting what is or was) or **interpretive** (signalling what will be, what should be, or what risks lie ahead). The distinction maps onto the informational role of the sentence: descriptive sentences convey data; interpretive sentences carry forward guidance.

| Label | When to use |
|-------|-------------|
| `"descriptive"` | Backward-looking (past data, historical events) OR present-state descriptions with no forward implication. Includes boilerplate and procedural sentences. |
| `"interpretive"` | Forward-signalling regardless of grammatical tense. Includes projections, outlooks, conditional/hypothetical structures ("if/should/were to"), and present-tense sentences where current observations clearly imply a future trajectory. Any sentence containing "outlook", "prospects", "trajectory", or equivalent projection language. |

**Key rules:**
- All conditional/hypothetical structures → `"interpretive"` (capture the conditionality in `ris`).
- Staff projections for the current or recently elapsed year → `"descriptive"`. Use `"interpretive"` only for projections extending meaningfully beyond the meeting date.
- `macro` sentences default to `"interpretive"` unless purely backward data description.
- Boilerplate and procedural sentences → `"descriptive"`.

---

## FIELD 3: `sen` — Sentiment as Hawkish/Dovish Incentive

**This field replaces the old `sen` (positive/negative) AND the old `dir` (hawkish/dovish) fields into a single unified scale.**

The scale captures the *policy incentive* implied by the sentence — regardless of whether the sentence is about economic data or monetary policy itself.

- **Positive scores = hawkish incentive** — the information pushes toward tightening
- **Negative scores = dovish incentive** — the information pushes toward easing
- **Zero = neutral** — no net directional policy incentive

Applies to ALL topics including `monetary_policy`. Use `"na"` only for `boilerplate`.

### Directional rules by topic (apply consistently):

| Topic | Hawkish (+) | Dovish (-) |
|-------|-------------|------------|
| `inflation` | Rising/high inflation | Falling/low inflation |
| `unemployment` | Falling unemployment / tight labor market | Rising unemployment / slack labor market |
| `economic_activity` | Stronger growth | Weaker growth |
| `financial_conditions` | Easier/more accommodative conditions | Tighter/more stressed conditions |
| `macro` | Stronger aggregate outlook | Weaker aggregate outlook |
| `monetary_policy` | Tightening action or signal | Easing action or signal |

**Critical rule for `financial_conditions`:** The polarity here is counterintuitive and must be applied carefully. Tighter financial conditions are a *hawkish impulse already transmitted by markets* — the Fed may therefore need to do less tightening itself, reducing the hawkish policy incentive. Easier financial conditions mean markets have done less of the work, so the Fed faces more pressure to act. The rule is:
- Easier/more accommodative conditions (low spreads, loose lending standards, rising asset prices) → **hawkish incentive (+)** — Fed needs to do more
- Tighter/more stressed conditions (rising spreads, tightening lending standards, falling asset prices) → **dovish incentive (-)** — markets have tightened for the Fed

### Score scale:

| Score | Meaning |
|-------|---------|
| `2` | Strongly hawkish incentive — linguistically intensified hawkish signal. Applies to crisis/emergency situations BUT ALSO to any sentence using explicit intensity markers relative to the Fed's normally measured tone. |
| `1` | Mildly hawkish incentive — directional hawkish signal without strong intensification |
| `0` | Neutral — balanced/mixed signals, purely factual with no evaluative stance, or hold/pause with no directional framing |
| `-1` | Mildly dovish incentive — directional dovish signal without strong intensification |
| `-2` | Strongly dovish incentive — linguistically intensified dovish signal. Same logic as `+2`: applies to crises but also to any clearly intensified language. |

**The ±2 threshold is about linguistic intensity, not event scale.** The key question is: does the language depart from the Fed's characteristically cautious, measured register? Intensifiers include:
- Adverbs: *significantly, substantially, markedly, sharply, dramatically, well above/below, far above/below*
- Adjectives/nouns in strong form: *surged, collapsed, plummeted, soared, severe* — *solid* and *robust* stay at ±1 as they are too common in Fed prose to signal genuine intensification
- Degree phrases: *highest in decades, record high, well above target, near historic lows*

Compare these pairs — left is `±1`, right is `±2`:
- "Growth was solid" → `+1` | "Growth surged well above expectations" → `+2`
- "Inflation rose" → `+1` | "Inflation surged well above target" → `+2`
- "Conditions deteriorated" → `-1` | "Conditions deteriorated sharply" → `-2`
- "The economy slowed" → `-1` | "The economy contracted severely" → `-2`

### Monetary policy sentences (`top` includes `"monetary_policy"`):

Apply `sen` to the policy action or signal itself:
- `+2`: Intensified tightening — includes crisis-scale actions (emergency hikes, "whatever it takes") AND standard actions described with strong intensity language (e.g., a 75bp hike described as a "significant increase" or "forceful response")
- `+1`: Standard rate hike, routine QT continuation, hawkish signal without intensity markers
- `0`: Hold/pause with no explicit directional framing; deliberative/neutral discussion
- `-1`: Standard rate cut, QE expansion, dovish signal without intensity markers
- `-2`: Intensified easing — includes crisis-scale easing (emergency cuts to zero, pandemic-era QE) AND standard actions described with strong intensity language

**Holds:** Default to `sen = 0` unless the sentence explicitly frames the hold as serving an accommodative or restrictive purpose. "Maintaining rates to support the expansion" → `-1`. "Maintaining rates at restrictive levels to keep downward pressure on inflation" → `+1`.

**Very hawkish/dovish threshold:** Use `+2`/`-2` when the language is explicitly intensified relative to the Fed's measured baseline — whether or not the event itself is an emergency. The test is the sentence's language, not the historical significance of the policy action.

**Minority participant views:** When a sentence attributes a position to "one member", "a few participants", or "some participants" (minority), cap at `+1`/`-1`. Reserve `+2`/`-2` for consensus/majority/Committee-level actions only.

**Important nuances:**
- Intensity modifiers matter: "prices rose somewhat" → `+1`; "prices surged well above target" → `+2`. Absence of strong intensifiers caps at `+1`/`-1`.
- `sen = 0` can mean true neutral OR contested signals that cancel. The `wid` field distinguishes these.
- For negations, label the *economic state being described*, not the grammatical surface. "Not seeing signs of deanchoring" = `-1` (dovish incentive — inflation well-anchored, less pressure to tighten).

---

## FIELD 4: `com` — Commitment Level

**Applies to `monetary_policy` sentences only. All other topics default to `"none"`** — economic assessments are Delphic (informational) by definition and do not constitute commitments.

| Label | When to use |
|-------|-------------|
| `unconditional` | Explicit commitment to a policy action with no stated condition |
| `conditional` | Policy action or path explicitly contingent on economic conditions. Includes all forms of conditionality: vague ("when conditions warrant") and specific ("until unemployment falls below 6.5%") — both are `conditional`. |
| `none` | Deliberative discussion, participants' views, no commitment made. Default for all non-monetary-policy topics. |

---

## FIELD 5: `hor` — Long-term Horizon (boolean)

`true` if the sentence contains explicit **long-term** horizon language. `false` otherwise (including near-term references and sentences with no horizon signal).

Only meaningful when `ten = "interpretive"`; default `false` for descriptive sentences.

| Value | When to use |
|-------|-------------|
| `true` | "for some time", "considerable time", "longer run", "over the medium term", "over time", "until normalization is well under way", projections extending multiple years out, "structural", "secular", "longer-run goal/objective" |
| `false` | No horizon signal, near-term language ("coming months", "this year", "at this meeting"), or `ten = "descriptive"` |

---

## FIELD 6: `ris` — Risk Balance

Captures directional tail asymmetry. Applies to all topics including `no_topic`.

**Critical distinction from `wid`:** `ris` captures *directional* uncertainty — the distribution is skewed one way. `wid` captures *epistemic* uncertainty — we don't know enough to assess direction. A sentence can have both.

| Label | When to use |
|-------|-------------|
| `skewed_downside` | Sentence explicitly frames risks as weighted toward worse-than-expected outcomes. Key phrases: "downside risk", "risks weighted to the downside", "risks tilted to the downside". Also applies when a sentence raises a specific downside concern even without exact phrase. |
| `skewed_upside` | Sentence explicitly frames risks toward better-than-expected or inflation-overshoot outcomes. For inflation: `skewed_upside` means more inflation is the tail risk. The sign convention is handled in regression, not in the label. |
| `symmetric` | Sentence explicitly asserts risks are balanced: "roughly balanced", "broadly balanced", "risks on both sides", "neither headwind nor tailwind" |
| `na` | No explicit risk framing — purely descriptive |

**Hypothetical/conditional structures:** When `ten = "forward"` and the sentence uses "if/should/were to" framing, populate `ris` to capture the direction of the conditional scenario. "If inflation were to persist, further tightening would be appropriate" → `ris = "skewed_upside"` (inflation tail is the active concern).

---

## FIELD 8: `wid` — Distribution Width

Captures whether the sentence signals the distribution of outcomes is wider than normal or genuinely contested.

**Critical distinction from `ris`:** `wid` is about *epistemic* uncertainty (how much do we know) and *genuine disagreement* between forces. `ris` is about *directional* skew (which tail is fatter).

| Label | When to use |
|-------|-------------|
| `elevated` | Explicit epistemic uncertainty or acknowledged limitations. Triggers: "highly uncertain", "unusually uncertain", "elevated uncertainty", "considerable uncertainty", "significant uncertainty", "difficult to assess", "hard to gauge", "cannot know", "unprecedented", "imperfectly understood", "wide range of possible outcomes", "uncertainty remained high", explicit acknowledgements of data or model limitations, high dispersion in participant views |
| `contested` | A single sentence (or a merged pair — see pipeline note below) contains **two genuinely opposing economic forces on the same topic in the same tense** that net to zero sentiment. Requirements: (1) both forces must be simultaneously operative — not sequential; (2) both must concern the same or closely related economic variable; (3) both must be in the same temporal frame. Linguistic markers: "but", "however", "although", "while", "despite", "even as". |
| `none` | Baseline. Modal verbs alone (may, might, could), standard hedges (appears, seems, suggests), bare "uncertain" without intensity modifier, data-dependence language, single-direction signals with benchmark references |

**Critical `contested` distinctions:**
- "Fell but remained high" → NOT contested. Single directional signal + level benchmark. 
- "Robust loan growth but tightening standards" → contested. Two genuine opposing forces in credit conditions.
- "Growth was solid, but uncertainty remains" → NOT contested. Second clause is a hedge, not an opposing economic force.
- Sequential tenses (past + forward): "Used to be X, now Y" → NOT contested. This is narrative of change, not genuine disagreement.
- Mixed-topic collision (inflation positive + unemployment negative): IS contested. A `sen = 0` from dual-mandate tension is informationally different from true neutrality — `wid = "contested"` tells the model the economy is paralysed by conflicting forces, not simply stable. Apply `sen = 0`, `wid = "contested"` for all genuine collisions whether same-topic or cross-topic.

**Pipeline note — merged sentence pairs:** When two consecutive sentences share the same `top` and have opposing `sen` scores (+1 followed by -1 or vice versa), the post-annotation parser will merge them into a single record with `sen = 0` and `wid = "contested"`. Annotators label each sentence independently; merging is handled downstream.

---

## Special Rules Summary

### When `top = ["boilerplate"]`:
- `ten = "descriptive"`, `sen = "na"`, `com = "none"`, `hor = false`, `ris = "na"`, `wid = "none"`

### When `top` includes `"monetary_policy"`:
- Fill `sen`, `com`, `hor`; include any referenced economic topics in `top` directly
- `ris` and `wid` still apply
- `com` = `"none"` for deliberative discussion; `"conditional"` or `"unconditional"` for actual commitments

### When `top = ["no_topic"]`:
- `sen = 0` (or non-zero if evaluative content present)
- `ris` and `wid` still apply
- `com = "none"`, `hor = false`

### Non-monetary-policy topics:
- `com = "none"` (always — economic assessments are not commitments)
- `hor = true` only if long-term horizon language is present and `ten = "interpretive"`

---

## Borderline Cases

### BC1: `wid = contested` — genuine opposing forces, same tense

`contested` requires two simultaneously operative forces in the same temporal frame. Forces may be on the **same topic** (e.g., loan growth up but lending standards tightening) OR **across different topics** (e.g., inflation hawkish + unemployment dovish — dual-mandate collision). What matters is that both forces are simultaneously operative, not sequential. A narrative of change (past → present) is NOT contested.

- ✅ "Loan growth robust but lending standards tightening" → contested (same topic, opposing forces)
- ✅ "Short-term inflation expectations rose but long-term remained anchored" → contested (same variable, opposing signals)
- ✅ "Inflation rising but unemployment also increasing" → contested (cross-topic dual-mandate collision)
- ❌ "Inflation used to be transitory; now it appears persistent" → NOT contested (sequential narrative)
- ❌ "Growth was solid, but uncertainty remains" → NOT contested (second clause is a hedge, not an opposing force)

### BC2: `wid = elevated` vs `ris` — epistemic vs directional

- `ris` = we know the skew direction. Use when a specific tail risk is identified.
- `wid = elevated` = we don't know enough to assign direction. Use for explicit uncertainty acknowledgements, model limitations, data gaps.
- Both can co-exist: "Upside risks to inflation remain, though the exact magnitude is highly uncertain" → `ris = "skewed_upside"`, `wid = "elevated"`.

### BC3: Holds — neutral unless explicitly framed

Holds default to `sen = 0`. Override only when the sentence explicitly states the hold's purpose:
- "Maintaining rates at restrictive levels to sustain downward pressure on inflation" → `sen = +1`
- "Maintaining rates to continue lending support to the expansion" → `sen = -1`
- "The Committee decided to maintain the target range at X%" (no framing) → `sen = 0`

The framing must appear in the sentence itself — do not infer from surrounding context.

### BC4: `+2`/`-2` threshold — emergency/crisis scale only

Reserve `+2`/`-2` for:
- Emergency or crisis-scale actions (cuts to zero, massive QE, pandemic-era measures)
- Language explicitly signaling urgency or extremity: "acting forcefully", "significantly reducing", "whatever it takes", "in unprecedented fashion"
- Routine hikes, routine cuts, and continuation of existing programs → `+1`/`-1`

**Minority participant views:** Cap at `+1`/`-1` regardless of their stated position. `+2`/`-2` reserved for Committee-level consensus actions.

### BC5: Mixed signals within a sentence — neutral + contested

When a single sentence contains one hawkish signal and one dovish signal on the **same topic**:
→ `sen = 0`, `wid = "contested"`

When signals conflict **across different topics** (e.g., inflation hawkish + unemployment dovish):
→ `sen = 0`, `wid = "contested"`, `top = ["inflation", "unemployment"]`

In both cases, do not attempt to net the scores — default to zero and let `wid` carry the tension signal.

### BC6: Data dependence language

Generic standalone data dependence: `boilerplate`
- "The Committee will continue to monitor incoming data and act as appropriate."

Specific data dependence naming a policy action and variables: `monetary_policy`, `com = "conditional"`, `con = ["macro"]`, `sen = 0`
- "In determining the extent of additional firming, the Committee will take into account cumulative tightening and incoming data."

Test: does the sentence name a *specific policy action* and *specific monitoring variables*? If yes, substantive label. If no, boilerplate.

### BC7: Negations — label the economic state, not the grammar

Label what the economic reality is, not the grammatical surface:
- "We are not seeing signs of deanchoring" → inflation well-anchored → `sen = -1` (dovish incentive)
- "Labor market conditions have not deteriorated" → labor market stable → `sen = 0`
- "Inflation has not returned to target" → above target → `sen = +1` (hawkish incentive)

### BC8: Hypothetical/conditional structures → `forward` + `ris`

All "if/should/were to/in the event that" structures:
- `ten = "forward"` (not hypothetical — that label is removed)
- Populate `ris` to capture the direction of the conditional scenario
- "If inflation were to persist, further tightening would be appropriate" → `ten = "forward"`, `ris = "skewed_upside"`, `sen = +1`
- "If labor market conditions were to weaken materially, easing would be appropriate" → `ten = "forward"`, `ris = "skewed_downside"`, `sen = -1`

---

## Worked Examples

### Example 1: Direction-of-change vs. level (NOT contested)

> "The private-sector job openings rate moved down markedly during February and March but remained high."

```json
{
  "top": ["unemployment"],
  "ten": "backward",
  "sen": -1,
  "com": "none",
  "hor": "none",
  "con": "na",
  "ris": "na",
  "wid": "none"
}
```
**Reasoning:** "Moved down markedly" is the directional signal (labor market weakening = dovish incentive = negative). "Remained high" is a level benchmark reference, not an opposing force. Single negative signal. NOT contested.

---

### Example 2: Genuine contested (opposing forces, same topic, same tense)

> "Although CRE loan growth on banks' balance sheets remained robust in the first quarter, the April SLOOS indicated that loan standards across all CRE loan categories tightened further."

```json
{
  "top": ["financial_conditions"],
  "ten": "backward",
  "sen": 0,
  "com": "none",
  "hor": "none",
  "con": "na",
  "ris": "na",
  "wid": "contested"
}
```
**Reasoning:** Loan growth robust (dovish incentive — easier conditions) AND lending standards tightening (hawkish incentive — tighter conditions). Two genuine opposing forces on financial conditions, same tense, netting to zero.

---

### Example 3: Falling inflation = dovish incentive (negative score)

> "These decreases reflected notable step-downs in both energy and core inflation amid slowing aggregate demand and declines in oil prices."

```json
{
  "top": ["inflation"],
  "ten": "backward",
  "sen": -1,
  "com": "none",
  "hor": "none",
  "con": "na",
  "ris": "na",
  "wid": "none"
}
```
**Reasoning:** Inflation falling = dovish incentive = negative score. Consistent regardless of whether inflation is above or below target.

---

### Example 4: Hold with explicit restrictive framing

> "In light of how far we have come in tightening policy, the Committee decided at today's meeting to maintain the target range for the federal funds rate at 5¼ to 5½ percent and to continue the process of significantly reducing our securities holdings."

```json
{
  "top": ["monetary_policy"],
  "ten": "present",
  "sen": 1,
  "com": "unconditional",
  "hor": "near_term",
  "con": ["none"],
  "ris": "na",
  "wid": "none"
}
```
**Reasoning:** The hold is at a peak restrictive rate AND QT is explicitly continuing ("significantly reducing"). The ongoing QT is a hawkish action (`sen = +1`). Not `+2` because this is continuation of an existing program, not a new emergency escalation. The phrase "how far we have come" is noted as a possible peak-hawkishness signal but is unreliable for systematic annotation — label the explicit action only.

---

### Example 5: Hypothetical structure → forward + ris

> "If incoming information indicates faster progress toward the Committee's employment and inflation objectives than the Committee now expects, then increases in the target range for the federal funds rate are likely to occur sooner than currently anticipated."

```json
{
  "top": ["monetary_policy"],
  "ten": "forward",
  "sen": 1,
  "com": "conditional",
  "hor": "near_term",
  "con": ["inflation", "unemployment"],
  "ris": "skewed_upside",
  "wid": "none"
}
```
**Reasoning:** "If" structure → `ten = "forward"` (not hypothetical). The conditional scenario is hawkish (sooner hikes) conditioned on upside progress → `ris = "skewed_upside"`. `sen = +1` because the sentence signals a hawkish reaction function.

---

### Example 6: Dual-mandate collision → neutral + contested

> "Participants judged that the war and related events were creating additional upward pressure on inflation and were weighing on global economic activity."

```json
{
  "top": ["inflation", "economic_activity"],
  "ten": "present",
  "sen": 0,
  "com": "none",
  "hor": "none",
  "con": "na",
  "ris": "na",
  "wid": "contested"
}
```
**Reasoning:** Rising inflation = hawkish incentive (+). Weakening economic activity = dovish incentive (-). Cross-topic collision netting to zero. `wid = "contested"` signals the tension.

---

### Example 7: Elevated uncertainty trigger

> "Participants acknowledged that the causes of movements in short- and longer-run inflation expectations, including the role of monetary policy, were imperfectly understood."

```json
{
  "top": ["inflation"],
  "ten": "present",
  "sen": 0,
  "com": "none",
  "hor": "none",
  "con": "na",
  "ris": "na",
  "wid": "elevated"
}
```
**Reasoning:** "Imperfectly understood" is an explicit epistemic limitation — trigger for `wid = "elevated"`. No directional assessment made, so `sen = 0`.

---

### Example 8: Upside risk — ris and wid can co-exist

> "But a few participants noted that the risk remained that inflationary pressures could rise as the expansion continued, especially if monetary policy remained highly accommodative for too long."

```json
{
  "top": ["inflation"],
  "ten": "forward",
  "sen": 1,
  "com": "none",
  "hor": "long_term",
  "con": "na",
  "ris": "skewed_upside",
  "wid": "none"
}
```
**Reasoning:** Risk of rising inflation → `ris = "skewed_upside"` (more inflation is the tail). Rising inflation = hawkish incentive = `sen = +1`. "If" structure → `ten = "forward"`. Minority ("a few participants") caps at `+1`.

---

### Example 9: Data dependence — boilerplate vs substantive

> "In determining the pace of future increases in the target range, the Committee will take into account the cumulative tightening of monetary policy, the lags with which monetary policy affects economic activity and inflation, and economic and financial developments."

```json
{
  "top": ["monetary_policy"],
  "ten": "forward",
  "sen": 0,
  "com": "conditional",
  "hor": "none",
  "con": ["macro"],
  "ris": "na",
  "wid": "none"
}
```
**Reasoning:** This names a specific policy action (pace of future rate increases) and specific variables being monitored (cumulative tightening, lags, economic and financial developments). Substantive, not boilerplate. `sen = 0` because data dependence is inherently neutral until the data arrives. `com = "conditional"` because action is contingent on incoming data.

---

### Example 10: Negation — label the economic state

> "That's not what we're seeing." *(in context of question about inflation deanchoring)*

```json
{
  "top": ["inflation"],
  "ten": "present",
  "sen": -1,
  "com": "none",
  "hor": "none",
  "con": "na",
  "ris": "na",
  "wid": "none"
}
```
**Reasoning:** Denying deanchoring = inflation well-anchored = dovish incentive = `sen = -1`. Label the economic state described, not the grammatical negation.

---

### Example 11: Minority view — cap at +1/-1

> "One member preferred a reduction in the target range of 50 basis points in the federal funds rate at this meeting."

```json
{
  "top": ["monetary_policy"],
  "ten": "present",
  "sen": -1,
  "com": "none",
  "hor": "near_term",
  "con": ["none"],
  "ris": "na",
  "wid": "none"
}
```
**Reasoning:** Single dissenting member calling for 50bp cut. Dovish signal, but minority view → capped at `-1`. `com = "none"` because this is a participant preference, not a Committee commitment.

---

### Example 12: `hor` applied to non-monetary-policy topic

> "With transitory factors expected to abate, the median inflation projection rises from 1.2 percent this year to 1.9 percent next year and 2 percent in 2018."

```json
{
  "top": ["inflation"],
  "ten": "forward",
  "sen": 1,
  "com": "none",
  "hor": "long_term",
  "con": "na",
  "ris": "na",
  "wid": "none"
}
```
**Reasoning:** Forward projection extending 2 years out → `hor = "long_term"`. Inflation rising toward target = hawkish incentive (less accommodation needed) = `sen = +1`.

---

## Quick Reference Card

```
top:  [inflation | unemployment | economic_activity | macro | financial_conditions
        | monetary_policy | boilerplate | no_topic]
      For monetary_policy sentences, add any referenced economic topics directly here.

ten:  "descriptive" | "interpretive"
      descriptive  = backward-looking OR present-state (no forward implication)
      interpretive = forward-signalling, projections, conditionals, outlooks

sen:  -2 | -1 | 0 | 1 | 2 | "na"
      na only for boilerplate
      HAWKISH/DOVISH INCENTIVE SCALE:
        inflation:            rising = +, falling = -
        unemployment:         tighter/lower = +, slack/higher = -
        economic_activity:    stronger = +, weaker = -
        financial_conditions: easier/accommodative = +, tighter/stressed = -
        monetary_policy:      tightening action/signal = +, easing action/signal = -

com:  unconditional | conditional | none
      non-monetary-policy → always "none"

hor:  true | false
      true  = long-term horizon language present ("longer run", "over time",
              "considerable time", "medium term", multi-year projections)
      false = no horizon signal, near-term language, or ten = "descriptive"

ris:  skewed_downside | skewed_upside | symmetric | na

wid:  elevated | contested | none

HOLDS default to sen = 0 unless explicitly framed as restrictive (+1) or accommodative (-1)
+2/-2 = LINGUISTICALLY INTENSIFIED signal — not exclusively crisis/emergency.
  Test: does the language depart from the Fed's measured baseline?
  Intensifiers: significantly, substantially, sharply, markedly, surged, collapsed,
                well above/below, record high, severely, dramatically
  NOT intensifiers: solid, robust (too common in Fed prose to signal intensity)
  Crisis events ARE +2/-2 but so is any sentence with explicit intensity markers.
  +1/-1 = directional without intensification ("solid", "gradual", "modest", standard hike/cut)
MINORITY views (one/few participants) capped at +1/-1
CONTESTED: sen = 0 from two genuinely opposing forces — applies to same-topic AND cross-topic collisions
  True neutral (nothing happening) → sen = 0, wid = "none"
  Paralysis/tension (forces cancel) → sen = 0, wid = "contested"
ELEVATED requires: explicit epistemic uncertainty trigger words
NEGATIONS: label the economic state described, not the grammatical surface
DATA DEPENDENCE: boilerplate if generic; top=["monetary_policy","macro"] + conditional if specific
```
