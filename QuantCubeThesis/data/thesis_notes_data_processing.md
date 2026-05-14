# Thesis Notes: Sentence-Level Data Processing Pipeline

**Purpose of this document:** Structured notes to be used as input for drafting the data processing section of the thesis. Written as thinking-aloud notes, not as polished prose. Claude should use this to write a coherent academic section following the logic laid out here.

---

## 1. Motivation: The Minimum Unit of Information Transmission

The core problem: Fed communications are long documents. A single FOMC meeting minutes document can be 10,000–15,000 words. A speech can run 5,000 words. Feeding entire documents into a classifier collapses all the heterogeneous content — a sentence about inflation, a procedural vote announcement, a forward guidance signal — into a single undifferentiated label. That destroys the granularity that makes the labelling useful.

The ideal would be a "minimum unit of information transmission" — the smallest chunk of text that conveys a complete, coherent idea about the economy or policy. This is not a sentence by definition, because:

- **Sometimes sentences are too short:** "Growth was solid." — this is grammatically complete but needs context. "Solid compared to what? Relative to what period? The previous sentence explained." Without context, this single sentence is almost uninterpretable.
- **Sometimes sentences are too long:** A single FOMC minutes sentence can span 80–100 words and contain two or three distinct economic assessments joined by subordinating clauses. Technically it could be split, but doing so reliably with hardcoded rules is impossible — the split points require understanding of semantic structure.

**Resolution:** The sentence is adopted as the base unit, with a targeted merging step applied afterward to re-attach sentences that are grammatically dependent on their predecessor. This is an imperfect but tractable operationalisation of the minimum information unit concept.

---

## 2. Source Documents and Text Extraction

Five document types are processed:

1. **Minutes** — formal narrative prose, meeting-frequency, structured into paragraphs by topic
2. **Statements** — short policy announcements released at decision time, highly formulaic
3. **Speeches** — longer-form communications, often structurally varied
4. **Press conference prepared remarks** — formal opening statement by the Chair
5. **Press conference Q&A** — conversational, informal, journalist-Chair exchanges

Documents are stored as JSON files in `data/raw/structured_json_*/` after prior scraping and parsing. Each JSON contains a `text` or `content` field holding the cleaned document body plus metadata (date, source, doc_type).

**Text pre-processing before tokenisation (`clean_text` function):**
- Unicode normalisation: curly quotes → straight apostrophes, em/en dashes → ASCII equivalents
- Removal of website boilerplate: navigation bars, "An official website of the United States Government", resource links
- `[SECTION]` tag splitting: some documents use an internal section tag; sections are reassembled with paragraph breaks
- "For release on delivery" header removal (statements and speeches)
- **Bottom cut:** text is truncated at the first occurrence of "Notation Vote" or "Following the FOMC policy vote" — everything after this is procedural record-keeping with no policy content
- **Top cut:** for minutes, text is trimmed from the beginning up to the first substantive keyword (review, developments, participants, outlook, guidance) to remove preamble headers

Press conferences receive additional pre-processing: the flat transcript string is split by ALL-CAPS speaker tags (e.g., `CHAIRMAN BERNANKE.`) which separate speaker turns. Chair turns are classified as prepared remarks or Q&A answers; reporter turns are extracted as `context_question` metadata.

---

## 3. Sentence Tokenisation

NLTK's `sent_tokenize` function (Punkt algorithm) is used for initial sentence splitting. Punkt is trained to distinguish sentence-ending periods from abbreviation periods by learning common abbreviation patterns from a corpus. This handles most FOMC-specific cases (U.S., p.m., Fed., etc.) adequately without requiring custom tokeniser training.

**Outcome:** A paragraph is split into a list of candidate sentences. Each sentence is then passed through the boilerplate filter and merging logic described below.

---

## 4. Boilerplate Detection and Removal

Before merging or labelling, sentences that carry no policy-relevant information are filtered out. A sentence is boilerplate if it:

- Matches a hardcoded regex from an extensive pattern list covering:
  - **Voting records:** "Voting for this action:", "The vote was", "voted as an alternate member", "by unanimous vote"
  - **Administrative/legal directives:** "Class I FOMC – Restricted Controlled", "Authorized for public release", "The Federal Open Market Committee authorizes and directs the Desk to"
  - **Attendance/adjournment:** "The meeting adjourned", "attended through", "present:", "absent:"
  - **Procedural restatements of the dual mandate:** Generic "committed to maximum employment and price stability" sentences with no specific directional content
  - **Journalist identification in press conferences:** "[Name], [outlet]", "Thank you for taking our questions", "Good afternoon, Chair Powell"
  - **Section headers and navigation fragments:** Short all-caps strings, standalone dots, web navigation text
  - **Citation and footnote patterns:** academic reference formats, "See [Author] (year)"
  - **Data-dependence boilerplate:** Generic standalone sentences like "The Committee will continue to monitor incoming data and act as appropriate" — no specific policy action named, no specific variables referenced
- Has a capital-letter density > 40% with length < 100 characters (likely a header)
- Has a numeric token ratio > 40% (likely a table entry or statistical footnote)

**Examples of boilerplate sentences that are removed:**

> "Voting for this action: Jerome H. Powell, Chair; John C. Williams, Vice Chair; Thomas I. Barkin; Michael S. Barr..."

> "The Federal Reserve is committed to using its full range of tools to support the U.S. economy in this challenging time."

> "The Committee will continue to monitor the implications of incoming information for the economic outlook."

> "For release on delivery 2:00 p.m. EDT Tuesday, March 21, 2023"

> "Class I FOMC – Restricted Controlled (FR)"

> "Thank you. I'd be happy to take your questions."

> "Steve Liesman, CNBC. Mr. Chairman, thank you."

**The fundamental limitation of hardcoded boilerplate detection:** These patterns were constructed by manually reading many FOMC documents and identifying recurring phrases. They are necessarily incomplete. If the rules are made more aggressive (e.g., filtering all short sentences, all sentences without certain economic keywords), they start removing genuinely informative content — a short sentence like "Growth was solid" is both short and potentially substantive. The current implementation accepts that some boilerplate will remain in the corpus in exchange for not destroying valid content. The consequence for annotation is that annotators will occasionally encounter sentences that should be classified as `boilerplate` under the labelling scheme, even though the pipeline's filter didn't catch them.

---

## 5. Sentence Merging: Reconstructing Units of Thought

After tokenisation and boilerplate removal, the remaining sentences are evaluated for merging with their immediate predecessor. This step implements the "minimum information unit" goal: two sentences are merged into one annotation unit if the second sentence is grammatically dependent on (continues the thought of) the first.

The merging procedure processes sentences sequentially within each paragraph. A "current chunk" accumulates sentences; when a new sentence does not satisfy any dependency condition, the current chunk is flushed as a complete annotation unit and the new sentence begins a new chunk.

### 5.1 Dependency conditions (all document types)

A sentence triggers a merge with the preceding chunk if:

**A. Starter triggers** — the sentence begins with one of:
- Formal consequential connectives: *"Therefore", "Thus", "Consequently", "As a result", "For example"*
- Bare demonstrative: *"That "* — e.g., "That decision was widely expected", "That said, ..."
- Epistemic + demonstrative pronoun: *"I think they/those/these/that", "We think they/those/these/that"*

**B. Demonstrative noun pattern** — the sentence contains a demonstrative pronoun (*this, that, these, those, such*) immediately followed by a noun from an expanded reference list. The noun list was deliberately designed to include nouns that are used as genuine cross-sentence references in FOMC text, while excluding temporal words (*year, month, meeting*) and comparative words (*level, rate, amount*) that appear in non-referential phrases like "this year" or "that level of activity".

The noun list covers two broad groups:
- *Original policy terms:* outcome, goal, objective, development, progress, condition, measure, action, policy, purchase, assessment, view, stance, trend, event, effect, forecast, projection, risk, imbalance, strain
- *Extended FOMC reference terms:* factor, aspect, area, issue, concern, pressure, indicator, approach, dynamic, tension, challenge, uncertainty, decision, situation, circumstance, environment, signal, pattern, finding, feature, consideration, shift, change, move, step, path, framework, backdrop, context, regime, episode, period, phase, cycle

This catches mid-sentence cross-sentence references that don't start the sentence with a connective. Example: "The Committee raised rates. **These actions** were intended to reduce inflation pressures." — "these actions" refers to the previous sentence. Equally: "Growth was solid. **These conditions** justified tightening." — "these conditions" refers to the preceding assessment.

**Conjunction guard on demonstrative nouns:** If the demonstrative noun pattern match appears after a subordinating conjunction (although, while, because, since, if, unless, before, after, despite, ...) within the candidate sentence, the match is rejected. This is because the demonstrative in that position refers back within its own clause, not to the preceding sentence. Example of a correctly rejected merge: "...the labour market was in balance. The economy was facing headwinds from tighter credit conditions, **although the extent of these effects** remained uncertain." — "these effects" follows "although" and refers to the headwinds described in the same sentence; merging with the preceding sentence would be incorrect.

### 5.2 Press conference additional triggers

For press conference documents (both prepared and Q&A), the following informal connectives are additionally recognised as merge triggers, reflecting the conversational nature of spoken language:
*"So ", "But ", "And ", "Then ", "And then", "And so"*

### 5.3 Constraints

- **Maximum chunk length:** 1,000 characters. If adding a dependent sentence would exceed this, the chunk is flushed and the dependent sentence begins a new chunk (potentially losing the dependency context — an acceptable trade-off).
- **Three-sentence cap (non-PC documents):** For minutes, speeches, and statements, a chunk is capped at three sentences. Even if the fourth sentence would satisfy a dependency condition, it begins a new chunk. This prevents runaway chains in formal prose while still allowing legitimate two-step dependency chains (e.g., a sentence starting with "If that outcome..." followed by "Consequently..." — two dependent sentences that together form a coherent unit). Press conferences are exempt from this cap, as conversational speech naturally chains multiple short clauses. The character limit (1,000 chars) acts as a hard backstop independent of sentence count.

### 5.4 Design rationale

The merge logic is intentionally conservative. The goal is not to reconstruct complete paragraphs but to recover the minimal amount of context needed to make the annotation unit interpretable. A sentence starting with "Therefore" with no preceding context is nearly unlabelable. A sentence starting with "Inflation surged" is fully self-contained. The triggers are designed to catch the former without unnecessarily grouping the latter.

---

## 6. Document-Level Notes

- **Minutes:** Written in long paragraph blocks. The two-sentence cap is important here because FOMC minutes frequently contain chains of sentences each starting with "Participants noted...", "Several participants...", "A few participants..." — these superficially look connected but are actually distinct observations that should be labelled independently.
- **Statements:** Very short documents. Sentences are usually self-contained. Merging is rare.
- **Speeches:** Narrative prose with genuine cross-sentence dependencies. The demonstrative noun pattern fires frequently and productively here.
- **Press conferences (prepared):** Similar to speeches; formal language.
- **Press conferences (Q&A):** Most merging happens here. The Chair's responses frequently use "So", "And", "But" to continue multi-clause answers. These are merged under the PC-specific triggers.

---

## 7. Summary of Pipeline Steps

1. Fetch raw document text from JSON → `clean_text()` → removes headers, footers, encoding artifacts
2. For press conferences: parse transcript into speaker turns; extract prepared remarks and Q&A pairs
3. `sent_tokenize()` (NLTK Punkt) → list of candidate sentences
4. `is_boilerplate()` filter → remove procedural/administrative sentences
5. `build_annotatable_records()` → sequential merging based on dependency conditions → output list of annotation-ready chunks with `merge_count` metadata
6. Deduplication of identical sentences (copy-pasted FOMC boilerplate that survived step 4)
7. Formatting with empty label fields → unlabelled pool for annotation

**Output fields per record:** `id`, `sentence`, `source`, `doc_type`, `date`, `context_question` (Q&A only), plus empty label fields: `top`, `sen`, `ten`, `hor`, `com`, `con`, `ris`, `wid`

---

## 8. Known Limitations and Honest Assessment

- **Boilerplate filter recall:** Estimated 15–20% of remaining sentences are still procedural/uninformative. Annotators encounter and label these as `boilerplate`, but they consume annotation budget.
- **Merge false positives:** Sentences can be incorrectly merged if the dependency trigger fires on intra-sentence language rather than cross-sentence reference. The conjunction guard mitigates this but cannot eliminate it. Post-annotation audits have found and corrected a small number of such cases.
- **Merge false negatives:** Some genuinely dependent sentences do not start with any recognised trigger and are not merged. Labelling them as standalone may result in unlabelable or misleadingly-labelled units. No systematic correction is applied; these are accepted as annotation noise.
- **Context loss at chunk boundaries:** The 1,000-character cap and the 2-sentence cap occasionally truncate a genuinely dependent chain. The annotation unit may then contain a reference ("these measures") whose antecedent is in the previous chunk. Again, accepted as annotation noise.
- **Press conference Q&A:** The conversational nature means sentence boundaries are often more arbitrary than in formal prose. Short utterances ("That's right." "I agree.") survive boilerplate filtering and appear as annotation units. Some of these have limited labellable content.
