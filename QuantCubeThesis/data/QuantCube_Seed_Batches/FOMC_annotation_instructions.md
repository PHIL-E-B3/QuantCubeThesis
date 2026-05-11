# FOMC Sentence Annotation Instructions

## Overview

You will be given a JSON file containing sentences extracted from FOMC communications (meeting minutes, speeches, statements, press conferences). Each sentence has an `id`, `sentence`, `source`, `doc_type`, and `date` field. All other fields are empty and must be filled in according to the rules below.

Return the complete JSON file with all fields populated. Never leave a field empty.

---

## Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `top` | array of strings | Topic(s) present in the sentence |
| `ten` | string | Tense / temporal orientation |
| `sen` | integer or "na" | Sentiment score (-2 to 2, or "na") |
| `dir` | string | Direction (monetary policy only, else "na") |
| `com` | string | Commitment level (monetary policy only, else "na") |
| `hor` | string | Horizon (monetary policy only, else "na") |
| `con` | array of strings or "na" | Conditions referenced in monetary policy sentence |
| `dom` | string | Dominant topic (only when multiple topics or multiple conditions, and one is explicitly primary; else "na") |
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
| `macro` | Aggregate economic outlook, overall trajectory, multi-variable assessments where no single variable dominates |
| `financial_conditions` | Credit markets, spreads, asset prices, bank lending, financial stability, money markets, funding conditions |
| `monetary_policy` | Rate decisions, asset purchases, balance sheet, forward guidance, policy stance — only when the sentence is explicitly about what the Fed is doing or will do with its policy tools. When chosen, choose ONLY this topic. |
| `boilerplate` | Procedural, administrative, legal directives, section headers, vote announcements, clearly formulaic language |
| `no_topic` | Generic statements that convey no specific economic content but may still carry risk or uncertainty signal |

**Key distinctions:**
- Use `macro` only when truly no single variable dominates. If a multi-variable sentence has a clear dominant conclusion (e.g., "risks to real activity"), label that variable instead.
- `boilerplate` examples: "The Committee voted to approve...", "No decisions were made at the meeting", legal/regulatory language, section headers like "Staff Review of the Financial Situation", memorial tributes.
- `no_topic` is for generic sentences that still might carry risk or uncertainty (so `ris` and `wid` still apply). `boilerplate` is for purely procedural/administrative sentences (default `ris = "na"`, `wid = "none"`).

---

## FIELD 2: `ten` — Tense

Captures the temporal orientation of the sentence's primary informational content — not just grammatical tense.

| Label | When to use |
|-------|-------------|
| `backward` | Describes observed past data or events. **If a sentence pairs backward data with a near-term projection that directly follows from it, label by the data, not the projection clause.** |
| `present` | Purely descriptive of current state with no implication about future direction |
| `forward` | Forward-signaling regardless of grammatical tense. Includes present-tense sentences where current observations imply a future trajectory. Any sentence containing "outlook", "prospects", "trajectory", or equivalent projection language defaults to forward. All `macro` sentences default to forward unless purely backward data description. |
| `hypothetical` | Explicit conditional trigger required: "if / should / were to / in the event that" |
| `none` | Procedural or `boilerplate`/`no_topic` sentences |

**Critical tense rule:** When a sentence reports staff projections for the current or recently elapsed year (e.g., a December 2022 meeting projecting 2022 figures), label `backward`. Use `forward` only for projections extending meaningfully beyond the meeting date.

---

## FIELD 3: `sen` — Sentiment

Applies to all topics **except** `monetary_policy` and `boilerplate`, where `sen = "na"`. For `no_topic`, use integer sentiment (0 is fine for truly neutral; non-zero if the sentence conveys evaluative content).

**Critical directional rule — apply consistently across ALL topics:**
- **`economic_activity`**: more/stronger activity = positive; less/weaker = negative
- **`unemployment`**: lower unemployment / stronger labor market = positive; higher unemployment / weaker labor market = negative
- **`inflation`**: lower/falling inflation = positive (allows accommodation); higher/rising inflation = negative (forces tightening). This is consistent regardless of whether inflation is above or below target. The logic is: sentiment tracks the signal value for predicting policy stance.
- **`financial_conditions`**: easier/more accommodative conditions = positive; tighter/more stressed = negative

| Score | Meaning |
|-------|---------|
| `2` | Strongly positive — clear strength, significant improvement |
| `1` | Mildly positive — modest improvement, gradual strengthening |
| `0` | Neutral — balanced/mixed signals that cancel out, or purely factual with no evaluative stance |
| `-1` | Mildly negative — modest deterioration, moderate weakness |
| `-2` | Strongly negative — sharp deterioration, significant stress, alarming conditions |

**Important nuances:**
- A sentence with `sen = 0` can be zero for two different reasons: (a) true neutral — nothing notable happening; or (b) contested — two opposing forces cancel. The `wid` field captures this distinction.
- Intensity modifiers matter: "prices rose somewhat" → `-1`; "prices surged well above target" → `-2`. The absence of strong intensifiers generally caps at `-1`/`1`.
- For `financial_conditions`: low VIX / spreads near historical lows = positive; high leverage / tightening standards / market stress = negative.

---

## FIELDS 4–6: `dir`, `com`, `hor` — Monetary Policy Only

Set to `"na"` for all sentences where `top != ["monetary_policy"]`.

### `dir` — Direction

| Label | When to use |
|-------|-------------|
| `hawkish` | Signals tightening, rate increase, tapering, or balance sheet reduction |
| `dovish` | Signals easing, rate cut, accommodation, or maintenance of low rates as active support. **A hawkish action framed as "supporting the expansion" or "lending support" = dovish.** "Patient", "gradual", "for some time", "considerable time" paired with a hawkish action = dovish. Discussing slowing or stopping balance sheet runoff = dovish. |
| `neutral` | Holding/reaffirming current stance with no directional lean; pure maintenance; deliberative discussion of tools with no commitment |

### `com` — Commitment Level

| Label | When to use |
|-------|-------------|
| `unconditional` | Explicit commitment to a policy action with no stated condition |
| `conditional` | Policy action or path explicitly contingent on economic conditions |
| `none` | Deliberative discussion, participants' views, no commitment made |

### `hor` — Horizon (only when `com != "none"`)

| Label | When to use |
|-------|-------------|
| `near_term` | Explicit short-term horizon or imminent action ("at this meeting", "in coming months") |
| `long_term` | Extended horizon ("for some time", "considerable time", "longer run", "until normalization is well under way") |
| `na` | `com = "none"` |

---

## FIELD 7: `con` — Condition Referenced

Applies **only** when `top = ["monetary_policy"]`. Lists the economic topics explicitly referenced as justification or conditions for the policy action. Multi-select array.

| Value | When to use |
|-------|-------------|
| `["inflation"]` | Policy conditioned on inflation outcomes |
| `["unemployment"]` | Policy conditioned on labor market outcomes |
| `["economic_activity"]` | Policy conditioned on broader growth/output outcomes |
| `["financial_conditions"]` | Policy conditioned on financial market/credit conditions |
| `["none"]` | No specific condition named |
| `"na"` | `top != ["monetary_policy"]` |

---

## FIELD 8: `dom` — Dominant Topic

**Only populate when:**
- `top` has more than one topic AND one is explicitly stated as primary/dominant, OR
- `top = ["monetary_policy"]` AND `con` has multiple entries AND one is explicitly stated as the primary driver

Otherwise `dom = "na"`. Do not infer dominance — it must be explicit in the sentence.

---

## FIELD 9: `ris` — Risk Balance

Captures directional tail asymmetry. Applies to all topics including `no_topic`. Does NOT require `top = ["monetary_policy"]`.

| Label | When to use |
|-------|-------------|
| `skewed_downside` | Sentence explicitly frames risks as weighted toward worse-than-expected outcomes. Key phrases: "downside risk", "risks weighted to the downside", "risks tilted to the downside". Also applies when a sentence raises a specific downside possibility/concern even without the exact phrase. |
| `skewed_upside` | Sentence explicitly frames risks toward better-than-expected or inflation-overshoot outcomes. Key phrases: "upside risk", "risks to inflation are to the upside". **Note: for inflation, upside risk = more inflation = bad for policy = `skewed_upside` is still labeled as such. The sign convention is handled in regression, not in the label.** |
| `symmetric` | Sentence explicitly asserts risks are balanced. Key phrases: "roughly balanced", "broadly balanced", "risks on both sides", "neither headwind nor tailwind", "more or less negative than assumed" |
| `na` | No explicit risk framing — purely descriptive of data or conditions |

---

## FIELD 10: `wid` — Distribution Width

Captures whether the sentence signals the distribution of outcomes is wider than normal.

| Label | When to use |
|-------|-------------|
| `elevated` | Explicit intensity-modified uncertainty or epistemic limitation. Triggers: "highly uncertain", "unusually uncertain", "elevated uncertainty", "considerable uncertainty", "significant uncertainty", "difficult to assess", "hard to gauge", "cannot know", "unprecedented", "outside our experience", "historical relationships may not hold", "imperfectly understood", "wide range of possible outcomes", "uncertainty remained high" |
| `contested` | A single sentence contains **two genuinely opposing economic forces** on the same or related variables that net to zero sentiment. The linguistic markers are adversative conjunctions: "but", "however", "although", "while", "despite", "even as". **CRITICAL distinctions:** (1) Direction-of-change vs. level ("fell but remained high") is NOT contested — it is a single directional signal with a benchmark reference. (2) "Growth was solid, but uncertainty remains" is NOT contested — the second clause is a hedge, not an opposing economic force. (3) Short-term expectations rising while long-term anchored IS contested — two genuine opposing signals on the same variable. (4) CRE loan growth robust but lending standards tightening IS contested — opposing forces in credit conditions. |
| `none` | Baseline — no elevation signal. Modal verbs alone (may, might, could), standard hedges (appears, seems, suggests), bare "uncertain" without intensity modifier, data-dependence language |

---

## Special Rules Summary

### When `top = ["boilerplate"]`:
- `ten = "none"`, `sen = 0`, `dir = "na"`, `com = "na"`, `hor = "na"`, `con = "na"`, `dom = "na"`, `ris = "na"`, `wid = "none"`

### When `top = ["monetary_policy"]`:
- `sen = "na"`
- Fill `dir`, `com`, `hor`, `con`
- `ris` and `wid` still apply

### When `top = ["no_topic"]`:
- `sen = 0` (or non-zero if evaluative content is present)
- `ris` and `wid` still apply
- `dir = "na"`, `com = "na"`, `hor = "na"`, `con = "na"`

### `dom` default:
- `dom = "na"` in the vast majority of cases. Only set when explicit primacy is stated in the sentence text itself.

---

## Worked Examples — Borderline Cases

### Example 1: Direction-of-change vs. level (NOT contested)

> "The private-sector job openings rate moved down markedly during February and March but remained high."

```json
{
  "top": ["unemployment"],
  "ten": "backward",
  "sen": -1,
  "dom": "na",
  "ris": "na",
  "wid": "none"
}
```
**Reasoning:** "Moved down markedly" is the directional signal (negative). "Remained high" is a level benchmark reference, not an opposing force. This is a single negative signal. NOT contested. Sentiment = -1 because openings are falling (labor market weakening).

---

### Example 2: Genuine contested (opposing forces on same variable)

> "Although CRE loan growth on banks' balance sheets remained robust in the first quarter, the April SLOOS indicated that loan standards across all CRE loan categories tightened further."

```json
{
  "top": ["financial_conditions"],
  "ten": "backward",
  "sen": 0,
  "dom": "na",
  "ris": "na",
  "wid": "contested"
}
```
**Reasoning:** Loan growth robust (positive signal) AND lending standards tightening (negative signal) — two genuine opposing forces in credit conditions netting to zero.

---

### Example 3: Inflation sentiment — falling inflation is POSITIVE

> "These decreases reflected notable step-downs in both energy and core inflation amid slowing aggregate demand and declines in oil prices."

```json
{
  "top": ["inflation"],
  "ten": "backward",
  "sen": 1,
  "dom": "na",
  "ris": "na",
  "wid": "none"
}
```
**Reasoning:** Inflation falling = positive signal (allows accommodation). This applies regardless of whether inflation is above or below target. The direction always maps the same way: falling = positive, rising = negative.

---

### Example 4: Hawkish action framed as supportive = dovish

> "In these circumstances, she believed it appropriate to gradually increase the target range for the federal funds rate from its current low level, which would allow monetary policy to continue to lend support to the economic expansion."

```json
{
  "top": ["monetary_policy"],
  "ten": "forward",
  "sen": "na",
  "dir": "dovish",
  "com": "conditional",
  "hor": "na",
  "con": ["economic_activity"],
  "dom": "na",
  "ris": "na",
  "wid": "none"
}
```
**Reasoning:** The action is a rate hike (normally hawkish), but it is explicitly framed as "lending support to the economic expansion." Per the rules, a hawkish action framed as active support = dovish. Commitment is conditional on "these circumstances."

---

### Example 5: Tense driven by data content, not projection clause

> "Indicators of near-term labor market conditions were consistent with projections of moderate increases in employment in the coming months: job openings generally moved up, but hiring rates and hiring plans were subdued, and initial claims trended up."

```json
{
  "top": ["unemployment"],
  "ten": "backward",
  "sen": 0,
  "dom": "na",
  "ris": "na",
  "wid": "contested"
}
```
**Reasoning:** The contested signals (openings up vs. hiring subdued vs. claims up) are all backward-looking observed data. The forward projection clause ("moderate increases in coming months") directly follows from those observations. Tense is labeled by the dominant informational content = backward. Contested because genuine opposing signals on labor market.

---

### Example 6: Risk framing overrides neutral sentiment

> "A few participants raised the possibility that recent declines in inflation might suggest that the economic recovery was not as strong as some thought."

```json
{
  "top": ["inflation", "economic_activity"],
  "ten": "present",
  "sen": 0,
  "dom": "na",
  "ris": "skewed_downside",
  "wid": "none"
}
```
**Reasoning:** Declining inflation is positive for inflation but implies weaker recovery — opposing signals across two topics netting to zero. However, the framing raises a downside concern about economic activity, so `ris = skewed_downside` even though overall sentiment = 0.

---

### Example 7: Intensity determines -1 vs -2

> "Some participants highlighted the fact that price increases had become more widespread."

```json
{
  "top": ["inflation"],
  "ten": "backward",
  "sen": -1,
  "dom": "na",
  "ris": "na",
  "wid": "none"
}
```
**Reasoning:** Price increases broadening is negative for inflation. But "more widespread" lacks a strong intensifier ("significantly", "sharply", "alarmingly"). The comparative "more" suggests gradual broadening. Sen = -1, not -2.

---

### Example 8: Elevated uncertainty trigger

> "Participants acknowledged that the causes of movements in short- and longer-run inflation expectations, including the role of monetary policy, were imperfectly understood."

```json
{
  "top": ["inflation"],
  "ten": "present",
  "sen": 0,
  "dom": "na",
  "ris": "na",
  "wid": "elevated"
}
```
**Reasoning:** "Imperfectly understood" is an explicit epistemic limitation acknowledgment — a trigger for `wid = elevated`. Sentiment = 0 because no directional assessment is made.

---

### Example 9: Upside risk to inflation = skewed_upside (even though bad for policy)

> "But a few participants noted that the risk remained that inflationary pressures could rise as the expansion continued, especially if monetary policy remained highly accommodative for too long."

```json
{
  "top": ["inflation"],
  "ten": "forward",
  "sen": -1,
  "dom": "na",
  "ris": "skewed_upside",
  "wid": "none"
}
```
**Reasoning:** The risk of rising inflation = `skewed_upside` because risks are skewed toward more inflation (even though more inflation is bad). The sign convention is consistent: `skewed_upside` for inflation means the distribution is skewed toward higher inflation. The regression handles the fact that this is a negative outcome. Sen = -1 because the threat of rising inflation is a negative signal.

---

### Example 10: `dom` populated only with explicit primacy

> "Participants concurred that the economic outlook warranted moving to a restrictive stance of policy, and they recognized the possibility that an even more restrictive stance could be appropriate if elevated inflation pressures were to persist."

```json
{
  "top": ["monetary_policy"],
  "ten": "forward",
  "sen": "na",
  "dir": "hawkish",
  "com": "conditional",
  "hor": "na",
  "con": ["inflation", "economic_activity"],
  "dom": "inflation",
  "ris": "na",
  "wid": "none"
}
```
**Reasoning:** Both inflation and economic_activity are referenced as conditions. But the escalation clause ("even more restrictive if inflation pressures persist") explicitly names inflation as the primary driver of the tail scenario. `dom = "inflation"` is justified because the sentence explicitly prioritizes inflation.

---

### Example 11: Staff projection for current/near-elapsed year = backward

> "On a four-quarter change basis, total PCE price inflation was expected to be 5.5 percent in 2022, while core inflation was expected to be 4.7 percent, both lower than in the November projection."
*(From December 2022 meeting)*

```json
{
  "top": ["inflation"],
  "ten": "backward",
  "sen": -1,
  "dom": "na",
  "ris": "na",
  "wid": "none"
}
```
**Reasoning:** Although "expected" is forward-looking language, the projection is for 2022 — the current/near-elapsed year at the time of the December 2022 meeting. Tense = backward. Sen = -1 because 5.5% inflation, while lower than November's projection, is still far above target — rising/high inflation is always negative sentiment regardless of direction of revision.

---

## Quick Reference Card

```
top:  [inflation | unemployment | economic_activity | macro | financial_conditions | monetary_policy | boilerplate | no_topic]
ten:  backward | present | forward | hypothetical | none
sen:  -2 | -1 | 0 | 1 | 2 | "na"  (na only for monetary_policy and boilerplate)
dir:  hawkish | dovish | neutral | "na"  (only monetary_policy)
com:  unconditional | conditional | none | "na"  (only monetary_policy)
hor:  near_term | long_term | "na"  (only monetary_policy, and only when com != none)
con:  ["inflation"] | ["unemployment"] | ["economic_activity"] | ["financial_conditions"] | ["none"] | "na"
dom:  [topic] | "na"  (na in most cases; only when explicit primacy stated)
ris:  skewed_downside | skewed_upside | symmetric | na
wid:  elevated | contested | none

SENTIMENT DIRECTION (consistent across all topics):
  inflation:         falling = positive, rising = negative
  unemployment:      lower/tighter labor market = positive, higher/slack = negative
  economic_activity: stronger = positive, weaker = negative
  financial_conditions: easier/accommodative = positive, tighter/stressed = negative

CONTESTED vs. LEVEL-BENCHMARK:
  "fell but remained high"           → NOT contested (single direction + benchmark)
  "robust growth but tightening standards" → contested (two opposing forces)
  "rose somewhat but declined in March and remained downbeat" → NOT contested (direction won)

DOM rule: na in the vast majority of cases. Only set when the sentence explicitly states one topic/condition is primary.
```
