# FOMC Sentence Annotation Instructions v2

## Overview

You will be given a JSON file containing sentences extracted from FOMC communications (meeting minutes, speeches, statements, press conferences). Each sentence has an `id`, `sentence`, `source`, `doc_type`, and `date` field. All other fields are empty and must be filled in according to the rules below.

Return the complete JSON file with all fields populated. Never leave a field empty.

---

## Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `top` | array of strings | Topic(s) present in the sentence |
| `ten` | string | Tense / temporal orientation |
| `sen` | integer or "na" | Hawkish/dovish incentive score (-2 to 2, or "na") |
| `com` | string | Commitment level (monetary policy only, else "none") |
| `hor` | string | Horizon — applies to ALL topics when `ten = "forward"`, else "none" |
| `con` | array of strings or "na" | Conditions referenced (monetary policy only) |
| `ris` | string | Risk balance |
| `wid` | string | Distribution width |

---

## FIELD 1: `top` — Topic (multi-select array)

Select all topics explicitly present in the sentence. A sentence can have multiple topics (e.g., `["inflation", "unemployment"]`). When `top = ["monetary_policy"]`, do NOT also add other topics — the condition referenced field (`con`) handles that.

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
- **Data dependence language:** "In determining the extent of additional policy firming, the Committee will take into account incoming data" is `boilerplate` if standalone and generic. If the sentence names a specific policy action or specific variables being monitored, label as `monetary_policy` with `com = "conditional"` and `con = ["macro"]`.

---

## FIELD 2: `ten` — Tense

Captures the temporal orientation of the sentence's primary informational content — not just grammatical tense.

| Label | When to use |
|-------|-------------|
| `backward` | Describes observed past data or events.|
| `present` | Purely descriptive of current state with no implication about future direction |
| `forward` | Forward-signaling regardless of grammatical tense. Includes present-tense sentences where current observations imply a future trajectory. Includes all explicit conditional/hypothetical structures ("if/should/were to") — label these as `forward` and capture the conditionality in `ris`. Any sentence containing "outlook", "prospects", "trajectory", or equivalent projection language defaults to forward. All `macro` sentences default to forward unless purely backward data description. |
| `none` | Procedural or `boilerplate`/`no_topic` sentences |

**Critical tense rules:**
- `hypothetical` is removed. All conditional/hypothetical structures ("if inflation were to persist", "should conditions warrant") → `ten = "forward"` + populate `ris` to capture the conditionality.
- When a sentence reports staff projections for the current or recently elapsed year (e.g., a December 2022 meeting projecting 2022 figures), label `backward`. Use `forward` only for projections extending meaningfully beyond the meeting date.

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
| `2` | Strongly hawkish incentive — crisis-scale tightening actions, emergency measures or clear intensifiers added describing the situation such as "signficantly" |
| `1` | Mildly hawkish incentive — modest tightening, gradual strengthening |
| `0` | Neutral — balanced/mixed signals, purely factual with no evaluative stance, or hold/pause with no directional framing |
| `-1` | Mildly dovish incentive — modest easing, gradual weakening |
| `-2` | Strongly dovish incentive — emergency easing, crisis conditions, severe deterioration or clear intensifiers added describing the situation such as "signficantly" |

### Monetary policy sentences (`top = ["monetary_policy"]`):

Apply `sen` to the policy action or signal itself:
- `+2`: Emergency/crisis-scale tightening (e.g., aggressive hikes, emergency QT, "whatever it takes" against inflation)
- `+1`: Standard rate hike, routine QT continuation, hawkish signal
- `0`: Hold/pause with no explicit directional framing; deliberative/neutral discussion
- `-1`: Standard rate cut, QE expansion, dovish signal
- `-2`: Emergency/crisis-scale easing (e.g., cuts to zero, massive QE announcements, pandemic-era measures)

**Holds:** Default to `sen = 0` unless the sentence explicitly frames the hold as serving an accommodative or restrictive purpose. "Maintaining rates to support the expansion" → `-1`. "Maintaining rates at restrictive levels to keep downward pressure on inflation" → `+1`.

**Very hawkish/dovish threshold:** Reserve `+2`/`-2` for actions or signals of emergency/crisis scale or language explicitly signaling urgency or extremity ("acting forcefully", "significantly", "whatever it takes"). Routine hikes/cuts and continuation of existing programs stay at `+1`/`-1`.

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

## FIELD 5: `hor` — Horizon

**Now applies to ALL topics**, but only when `ten = "forward"`. Defaults to `"none"` in all other cases.

| Label | When to use |
|-------|-------------|
| `short_term` | Explicit near-term horizon: "at this meeting", "in coming months", "over the near term", imminent actions already announced |
| `long_term` | Extended horizon: "for some time", "considerable time", "longer run", "over the medium term", "until normalization is well under way", projections extending multiple years out |
| `none` | `ten != "forward"`, or no horizon signal present |

---

## FIELD 6: `con` — Condition Referenced

Applies **only** when `top = ["monetary_policy"]`. Lists the economic topics explicitly referenced as justification or conditions for the policy action. Multi-select array.

| Value | When to use |
|-------|-------------|
| `["inflation"]` | Policy conditioned on inflation outcomes |
| `["unemployment"]` | Policy conditioned on labor market outcomes |
| `["economic_activity"]` | Policy conditioned on specific growth/output measures |
| `["macro"]` | Policy conditioned on aggregate economic conditions without naming a specific variable |
| `["financial_conditions"]` | Policy conditioned on financial market/credit conditions |
| `["none"]` | No specific condition named |
| `"na"` | `top != ["monetary_policy"]` |

---

## FIELD 7: `ris` — Risk Balance

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
- `ten = "none"`, `sen = "na"`, `com = "none"`, `hor = "none"`, `con = "na"`, `ris = "na"`, `wid = "none"`

### When `top = ["monetary_policy"]`:
- Fill `sen`, `com`, `hor`, `con`
- `ris` and `wid` still apply
- `com` = `"none"` for deliberative discussion; `"conditional"` or `"unconditional"` for actual commitments

### When `top = ["no_topic"]`:
- `sen = 0` (or non-zero if evaluative content present)
- `ris` and `wid` still apply
- `com = "none"`, `hor = "none"`, `con = "na"`

### Non-monetary-policy topics:
- `com = "none"` (always — economic assessments are not commitments)
- `con = "na"` (always)
- `hor` applies if `ten = "forward"`

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
  "hor": "short_term",
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
  "hor": "short_term",
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
  "hor": "short_term",
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

ten:  backward | present | forward | none
      (hypothetical removed — use forward + ris)

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

hor:  short_term | long_term | none
      applies to ALL topics but only when ten = "forward"

con:  ["inflation"] | ["unemployment"] | ["economic_activity"] | ["macro"]
        | ["financial_conditions"] | ["none"] | "na"
      "na" for all non-monetary-policy topics

ris:  skewed_downside | skewed_upside | symmetric | na

wid:  elevated | contested | none

HOLDS default to sen = 0 unless explicitly framed as restrictive (+1) or accommodative (-1)
+2/-2 reserved for emergency/crisis-scale actions or explicit urgency language
MINORITY views (one/few participants) capped at +1/-1
CONTESTED: sen = 0 from two genuinely opposing forces — applies to same-topic AND cross-topic collisions
  True neutral (nothing happening) → sen = 0, wid = "none"
  Paralysis/tension (forces cancel) → sen = 0, wid = "contested"
ELEVATED requires: explicit epistemic uncertainty trigger words
NEGATIONS: label the economic state described, not the grammatical surface
DATA DEPENDENCE: boilerplate if generic; monetary_policy + conditional if names specific action + variables
```
