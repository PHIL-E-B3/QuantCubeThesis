# Handoff Summary: LLM Annotation Pipeline Decisions
## For: Philip (teammate)
## Context: FOMC sentence annotation using Llama 3.1 8B

---

## 1. Architecture Decision: Generative (Final)

After extended discussion, the decision is to use a **generative approach** — the model
produces the full 7-field JSON object in a single forward pass — rather than a classifier
head architecture.

### Why not classifier heads
A sequential classifier head architecture was considered and is technically sound:
- 7 independent linear layers on top of a shared LLM backbone
- Run in dependency order: `top` → `ten` → `sen` → `hor` → `com` → `ris` → `wid`
- Hard post-processing rules enforce conditional constraints:
  - If `top = boilerplate` → force `sen = "na"`, `com = "none"`, `ris = "na"`, `wid = "none"`
  - If `top` excludes `monetary_policy` → force `com = "none"`
  - If `ten = descriptive` → force `hor = false`
- Multi-label `top` is handled via sigmoid (not softmax) — each topic is an independent
  binary decision
- Active learning is simpler (softmax entropy per head)

### Why generative wins on performance
The decisive argument: at inference time, the generative model reads and follows explicit
label definitions, disambiguation rules, and examples from the prompt. The classifier head
is blind to those definitions at prediction time — it relies entirely on learned patterns
from training examples. For a nuanced schema (where `wid: contested` requires understanding
the difference between a genuine opposing force and a hedge, where the level-overrides-
direction rule is counterintuitive), the ability to follow explicit instructions at
inference time is a real and meaningful advantage.

With ~100 examples per label (achievable with active learning), the generative model gets:
- Fine-tuned learned patterns from training
- Prompt-level guidance for edge cases the training data does not cover
Both simultaneously. The classifier head gets only the former.

### Active learning with generative model
Active learning IS compatible with generative models. Strategy:
- **Round 1 (from initial ~700):** Embedding-based diversity sampling — use LLM hidden
  states as sentence embeddings, select sentences furthest from already-annotated examples.
  Rationale: model not yet calibrated enough for reliable uncertainty estimates.
- **Round 2 (after first fine-tuning):** Token-probability uncertainty sampling on `sen`
  and `top` fields specifically — extract log probability of each label value at the
  generation step for that field. High entropy = high uncertainty = good annotation candidate.

Reference: Settles (2009) — foundational active learning text.

---

## 2. Prompt Engineering Experiment: 9 Prompts

### Design
A 2×4 grid plus one baseline:

| # | Verbosity | Shots | CoT | Description |
|---|-----------|-------|-----|-------------|
| P0 | Minimal | 0 | No | True baseline — field names and valid values only |
| P1 | Medium | 3 | No | |
| P2 | Medium | 3 | Yes | |
| P3 | Medium | 5 | No | |
| P4 | Medium | 5 | Yes | |
| P5 | High | 3 | No | |
| P6 | High | 3 | Yes | |
| P7 | High | 5 | No | |
| P8 | High | 5 | Yes | |

### Verbosity levels (what goes in each)

**P0 — Minimal baseline:**
- Field names and valid values only
- No definitions, no examples, no CoT
- Establishes the performance floor

**Medium verbosity includes:**
- Full field definitions and valid values
- Financial conditions polarity reversal (CRITICAL — counterintuitive, model will
  systematically fail without this)
- Level-overrides-direction rule (non-obvious, model defaults to direction otherwise)
- 50/75 bps = ±2 regardless of language
- Boilerplate cascade (all fields default)
- `com` applies to `monetary_policy` only
- Hold defaults to `sen = 0` unless explicitly framed otherwise
- Bare "uncertain" = `wid = elevated` (no modifier needed)

**High verbosity adds on top of medium:**
- Full cause-and-effect rule (three-pattern typology: desc+interp, desc+desc, interp+interp)
- Specific ±2 intensifier word list (surged, collapsed, plummeted, soared, severe)
- Minority participant views capped at ±1
- Negation rule (label economic state, not grammatical surface)
- Cumulative signals rule (3+ stacked same-direction signals = ±2)
- Full `contested` vs `elevated` disambiguation
- Cross-topic collision (dual mandate) = `contested`

**What was deliberately excluded even from high verbosity:**
- All economic topics default to `com = "none"` (inferable from definitions)
- `macro` defaults to `"interpretive"` (removed — not reliable enough as a rule)

### Few-shot example selection principles
Based on Min et al. (2022): label **coverage** dominates shot **count**.

3-shot examples should cover:
1. Boilerplate (shows cascade of na/none/false defaults)
2. Monetary policy with conditional commitment (Delphic guidance)
3. Contested sentence (wid = "contested", sen = 0)

5-shot adds:
4. Monetary policy unconditional (Odyssean guidance)
5. Long-horizon sentence with hor = true

All examples drawn from the already-annotated evaluation batch (confirmed labels).
Do NOT use evaluation batch sentences as training data — keep strictly separate.

### Field ordering in JSON output
Use dependency order consistently across all prompts:
`top` → `ten` → `sen` → `hor` → `com` → `ris` → `wid`

This is baked into the JSON structure of the few-shot examples. The model learns the
ordering implicitly from the examples — no need to state it explicitly in prose.

### CoT instruction (for P2, P4, P6, P8)
Ask the model to reason through each field in the dependency order before producing the
final JSON. One sentence of reasoning per field, then the JSON. No other text.

### Key references
- Brown et al. (2020) — few-shot prompting foundation
- Wei et al. (2022) — chain-of-thought improves multi-step conditional reasoning
- Min et al. (2022) — label coverage dominates shot count
- Ma et al. (2025, EMNLP) — field ordering matters for multi-label LLM output

---

## 3. Sequence Length and Token Budget

### Constraints
- Sentence length: ~200 tokens average
- JSON output (7 fields, fully populated): ~120-200 tokens
- CoT reasoning adds: ~100-150 tokens

### Decision: max_sequence_length = 768 for training, 1024 for inference

- 512 is too tight: medium verbosity prompt (~300-400 tokens) + sentence (~200 tokens)
  + output (~150 tokens) = ~650-750 tokens, already at ceiling before CoT
- 1024 is comfortable but roughly doubles training time vs 512
- 768 is the compromise: fits medium verbosity + 5-shot + sentence + output comfortably;
  use 1024 for the final training run with winning hyperparameters

### Training time estimates (RTX 4070 Ti Super, ~700-1300 examples, 3 epochs)
- 512 tokens: ~15-25 minutes per run
- 768 tokens: ~20-35 minutes per run
- 1024 tokens: ~25-45 minutes per run
- Optuna over 50 trials at 768: ~17-29 hours total

### Inference time estimates (full 120k sentence corpus)
- Standard HuggingFace (sequential): ~7 days — DO NOT USE
- vLLM with 4-bit quantization on RTX 4070 Ti Super: ~8-12 hours
  (output is ~150-200 tokens, not 5 tokens as some sources assume)

### Critical inference setup requirement
After QLoRA fine-tuning, merge the LoRA adapter into the base model weights before
loading into vLLM. vLLM does support native LoRA adapter loading as an alternative.
Do NOT use standard HuggingFace pipeline for the full corpus — it processes sequentially
and will take days.

---

## 4. Data Sufficiency by Label (Current State)

| Label | Count | Status |
|-------|-------|--------|
| wid: contested | 11 | CRITICAL — at absolute floor |
| ris: skewed_upside | 12 | CRITICAL — at absolute floor |
| wid: elevated | 14 | CRITICAL — at absolute floor |
| ris: skewed_downside | 29 | Marginal |
| hor: True | 33 | Marginal |
| sen: +2 | 37 | Marginal |
| com: conditional | 42 | Marginal |
| com: unconditional | 57 | Lower bound of reliable learning |
| sen: -2 | 55 | Lower bound of reliable learning |
| ten: interpretive | 161 | Adequate |
| ten: descriptive | 392 | Adequate |
| com: none | 454 | Adequate |

**Minimum threshold for reliable generalisation: ~50-100 examples per class**
Source: BERT learning curve analysis (PMC); few-shot fine-tuning literature.
Below ~10-30 examples, classifier heads learn essentially nothing class-specific even
with a pre-trained backbone.

**Plan:** Guillermo will spend tomorrow labelling to reach ~50-60 per rare label.
Active learning rounds target rare classes specifically (not random sampling).
At ~100 per class across both rounds, generative model performance becomes fully reliable.

**If rare classes do not reach 50 examples:** Drop `wid` and `ris` from the forecasting
exercise entirely rather than including a noisy indicator. Do not include unreliable
fields in the FAVAR augmentation.

---

## 5. Workflow for Tomorrow

1. Label sentences prioritising: `wid: contested`, `ris: skewed_upside`, `wid: elevated`,
   `ris: skewed_downside`, `hor: True`, `sen: +2`
2. Keep the prompt engineering experiment (9 prompts) on the base model (non-fine-tuned)
   using the evaluation batch as benchmark
3. Use the corrected annotation instructions v3 (FOMC_annotation_instructions_v3_corrected.md)
   — do not use the original v3 file, it has schema errors now fixed
4. The prompt to use during annotation labelling and during fine-tuning must be identical —
   finalise the winning prompt BEFORE generating training data
5. Use `max_new_tokens = 256` for output and `max_sequence_length = 768` for training

---

## 6. Key Files
- `FOMC_annotation_instructions_v3_corrected.md` — use this, not the original v3
- `newly_stripped_sentences_eval_annotated.json` — evaluation batch with confirmed labels
- All .tex and .bib files in deliverables/ folder

## 7. Open Decisions (still to resolve)
- Level vs change as dependent variable in FAVAR (tentative: predict change)
- Exact number of PCA factors retained (95% variance threshold — run and report k)
- Final prompt selection (pending 9-prompt experiment results)
