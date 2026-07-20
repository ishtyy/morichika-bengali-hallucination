# BICHAR Phase-2 heavy training and pipeline sequence

Version: 1.0  
Frozen: 2026-07-19 BDT  
Budget authority: USD 80 total program ceiling with more than USD 10 left
outside the program  
Training status: **not started; external-review gate is open**

This is the user-readable execution specification for training, evaluation,
offline inference and cost control. It separates demonstrated facts from
planned experiments. A number in a plan is not a result. No training job may
start merely because it appears in this document.

Machine-readable companion:
`configs/phase2_training_budget_v1.yaml`. Current spend ledger:
`artifacts/status/modal_workspace_usage_20260718.json`. Persistent project
memory: `artifacts/status/GOAL_MEMORY.md`.

Primary-paper design and ablation register:
`docs/PHASE2_TRAINING_PAPER_GROUNDED_DESIGN_V1_20260719.md`. Papers motivate
gates and ablations; only local held-set/runtime/cost measurements can promote
a configuration.

## 1. Intended outcome

Build one production/research-grade Bengali hallucination system for roughly
5,000 Phase-2 rows that:

1. treats contextual and closed-book questions as different semantic tasks;
2. retrieves from every rights-admitted source only for the closed-book lane,
   without treating every source as equally reliable;
3. uses deterministic Bengali policies, a compact trained verifier and a
   selective heavy evidence judge;
4. reports exact macro-F1, lane/class/family F1, calibration, evidence recall,
   runtime, peak RAM/VRAM and dollars;
5. reproduces after interruption and runs fully offline on Kaggle T4x2 below
   nine hours, with a 6.5-hour target;
6. selects the final system only from measured quality-runtime-resource Pareto
   evidence.

“Heavy training” here means a serious, clean program: full-parameter training
of compact components plus one 4B adapter and, only if earned, a replication or
second heavy finalist. It does not mean pushing all gathered bytes through a
model without rights, deduplication, correct labels or held evaluation.

## 2. Non-negotiable semantic split

### 2.1 Contextual lane

Input is exactly `(question, supplied context, response)`.
The target is one of:

- `supported` - the supplied context supports the exact requested property;
- `refuted` - the supplied context contradicts the response;
- `not_enough_information` - the context does not establish the response.

Competition mapping is `supported -> 1`; `refuted/NEI -> 0` under the current
contract. External world knowledge cannot rescue a response unsupported by the
supplied passage. A passage mentioning a related person, event, date or place is
not enough. The required entity-relation-property tuple must match the question.

**There is no lexical, dense, web, corpus, passage or counter-evidence
retrieval in the contextual lane.** The contextual model receives the complete
supplied context and compares the response with it directly. Predicting
supporting or contradicting character offsets is an auxiliary learned output;
it is not a retrieval stage and it cannot add text that was not supplied in the
row. If a context exceeds a candidate model's measured limit, deterministic
overlapping full-coverage windows and an auditable aggregation rule are used;
no relevance search is allowed to discard parts of the supplied context.

Question grounding is checked separately from response correctness. The model
first identifies the question's entity, factual premises, requested relation or
property, scope and required operands, then verifies which of those are
established by the supplied context. Question wording is not evidence for its
own premise. Only after answerability is established does the model test whether
the response answers that exact question. A passage that supports the response
as a related fact but does not support the question-response relation is NEI or
refuted under the frozen policy, not supported.

Permitted derivations use only context-grounded operands and a versioned rule:
date/age/duration arithmetic, bounded calendar/timezone conversion when the
calendar is explicit, narrow kinship composition such as father’s father, and
auditable lexical equivalence. Every operand, operation and uncertainty is
recorded. Negation, conflicting anchors, missing operands and multiple plausible
values force abstention/manual review.

### 2.2 Closed-book lane

Input is `(question, response, retrieved evidence, counter-evidence)` with
targets `correct`, `incorrect`, and `insufficient_or_conflicting_evidence`.
Closed-book retrieval may use Bengali sources and cross-lingual primary sources
for globally stable facts. A retrieval miss is not automatically a
hallucination. Calibration is separate from the contextual lane.

Evidence priority is:

1. page-provenanced books, user OCR and curated datasets;
2. primary law, government, gazette, textbook and official sources;
3. authoritative dictionaries and edited reference works;
4. corroborated major newspapers;
5. Banglapedia;
6. Wikipedia as corroboration, never as an automatic override;
7. unverified web or generated material.

Conflicts retain dates, editions, source authority and both propositions.

## 3. Demonstrated project state before training

- The label-blind typed-anchor shadow audit covered all 2,516 Phase-1 rows:
  1,361 contextual rows analyzed and 1,155 closed-book rows skipped by the
  contextual analyzer. Its flags are diagnostics, not automatic label changes.
- A 335-row manual semantic-trap queue and a 22-lock user adjudication overlay
  exist. The current label convention remains `0 = hallucination/incorrect`,
  `1 = correct/non-hallucination`.
- Source registry v6 contains 103 normalized records and preserves 41 broader
  resources. Registration is not runtime or training authorization.
- The locked sparse/lexical panel has 14,785 records and 2,048 queries. Exact
  term lookup reached R@1/R@10 1.0 on that narrow operation-aware task; RRF
  reached R@1 0.995117 and R@10 1.0. This does not prove hallucination F1.
- The proposed Nepali MiniLM retriever was measured and rejected: R@1 0.047363,
  R@10 0.136719, MRR@10 0.070340 and nDCG@10 0.085821, with no explicit model
  license. It is quality-rejected and rights-quarantined.
- A restart-safe paired Challenge350 Kaggle package exists and passed local
  tests. The remote paired run has not yet supplied the final baseline ledger.
- No new Modal fine-tuning job and no competition submission were launched by
  this training plan.

The selected Phase-1 empirical anchor is now the hash-bound combined-22 vector
at public macro-F1 **0.979** (Kaggle ref `54820308`, SHA-256
`13881a5b4052f9660584d20bc1c12df54373a59308da7ce6d6e628297c163808`).
The `.970` lexical34 + closed-source10 vector is its measured predecessor and a
source/retrieval-policy ablation, not the current backbone. Hidden gold is not
available locally, so both values remain public-leaderboard measurements rather
than locally recomputable Phase-1 F1. The 41-milestone provenance ledger is
`artifacts/status/PHASE1_LEADERBOARD_POLICY_TRAJECTORY_20260719.json`.

## 4. Data inventory and exact current amounts

| Data family | Measured amount | Planned role | Current disposition |
|---|---:|---|---|
| Granite Bengali question-passage pool | 12,477 deduplicated question-passage pairs, 3,905 groups, 3 sources | first closed-book retriever arm | closest to training-ready; exact model/data contract still rechecked before launch |
| TyDi-AS2 Bengali | 4,079,197 sentence judgments | answer-sentence selection and hard negatives | downloaded and hash-pinned; fail-closed until rights/lineage/overlap/split gates finish |
| TyDi-AS2 native Bengali | 1,765,083 = 2,534 positive + 1,762,549 negative | high-priority native retrieval contrasts | keep separate from translationese |
| TyDi-AS2 English-to-Bengali | 2,314,114 = 4,462 positive + 2,309,652 negative | lower-weight retrieval contrasts | translated stratum; may not dominate |
| IndicSQuAD Bengali | 142,192 = 92,749 answerable + 49,443 unanswerable | support/NEI/span grounding | downloaded; use stricter license interpretation until repository/card conflict closes |
| Earlier mega collection | 2,643,972 gross; 2,266,072 exact-unique; 377,900 duplicate; about 387.8M sampled-token estimate | mining, language/domain candidates | earlier audit admitted zero training rows under fail-closed rights policy |
| BanglaRQA-derived contextual pool | 24,456 prepared examples | native support/refute/absence | noncommercial/share-alike research status |
| BanglaCQA converted pool | 23,919 examples | contextual/counterfactual | shared lineage with BanglaRQA; never split as independent source |
| Evidence-trust selector pool | 24,688 examples | relation/source trust research | merged rights are not production-ready |
| Bangla-MMLU balanced | 170,769 examples | closed-book research/control | license unspecified; quarantine |
| BCS balanced pool | 7,390 examples | closed-book MCQ/hard negatives | underlying PDF provenance conditional |
| Bagdhara pool | 59,390 examples | idiom/figurative lexical training | share-alike/lineage review pending |
| Indic-RAG-Suite Bengali candidate | 3,320,042 public rows | filtered synthetic weak positives | public file appears pre-filter; Wikipedia and synthetic provenance risk |
| MSMARCO-XI Bengali candidate | 876,579 rows | selected/nonselected retrieval contrasts | translated; MS MARCO noncommercial/upstream terms |
| BanglaQA candidate | 178,012 QA over 15,997 native news articles | native support/unanswerable | upstream news rights and synthetic-question risk |
| NCTB-QA candidate | 87,805 QA: 50,270 answerable + 37,535 unanswerable | Bangladesh textbook grounding | noncommercial; generated questions/rationales; group by textbook |
| New BCS book scans | 1,388 pages: 1,090 + 298 | private retrieval/MCQ extraction | scan-only, permission flags restrictive, training/redistribution rights unverified |

The TyDi-AS2 positive rate is extremely small. Training must not ingest all
4.07M rows as if every negative were a verified contradiction. Use all 6,996
positives only after admission, then mine approximately 4-8 hard negatives per
positive with question/title/document grouping and false-negative review.

The earlier 2.64M-row collection is not “2.64M long-context examples.” It mixes
tasks, prompts, word lists, benchmark copies and documents. Rights, semantic
role and unique-token counts are evaluated per family.

## 5. What “include everything” means

Every gathered source receives a visible ledger row with exactly one current
disposition:

- `training_and_retrieval`;
- `retrieval_only`;
- `evaluation_only`;
- `quarantine`.

Nothing is silently dropped. However, quarantined content cannot enter model
weights or the Kaggle bundle. Rights-admitted knowledge may be indexed and
mined for the closed-book lane even when only a selected subset is repeated
through training. A contextual test row never consults that index.

MCQs become auditable records containing book ID, edition, page, question
number, options, marked key, explanation, literal text hashes and OCR quality.
Correct options become positive candidates; distractors become hard-negative
candidates. A marked answer is not truth when the page is uncertain or a
higher-authority source conflicts. Such records stay manual/quarantined.

## 6. Training records to build

### 6.1 Retriever records

`query, positive passage, hard-negative passage(s), source, document group,
question group, rights record, literal hashes`.

Initial exact pool: 12,477 positives, one matched negative per positive, two
epochs. TyDi expansion uses all admitted positives plus 4-8 mined negatives.

### 6.2 Contextual verifier pilot

Freeze exactly 90,000 records for the first full pilot:

- 30,000 supported;
- 30,000 refuted;
- 30,000 not-enough-information.

Each record contains question, supplied context, response, gold evidence spans
or explicit absence, requested property, entity/relation/value annotations,
derivation operands where applicable, source/document/question groups,
native/translation flag, length bucket and transformation family.

Required hard families include:

- same passage but different question;
- same entity but wrong relation/property;
- creator versus usage and birthplace versus role;
- year, date, age, duration, calendar and timezone;
- multiple names, places, events, numbers and units;
- negation, quantifier and scope;
- prior-event ordering when later events are otherwise indistinguishable;
- bounded kinship and arithmetic;
- antonym, source-language/register pairing, Guruchandali and grammar rules;
- Unicode equivalence, combining marks, Bengali digits, word breaks, OCR noise
  and code-mixing;
- cultural-default conflicts and context-versus-world-knowledge swaps.

Label balance is frozen. Source mixture is not frozen until admission because
we will not invent rights-clear rows to meet a quota.

### 6.3 How a wrong answer teaches the model

A wrong-answer label by itself does **not** reliably make a model discover why
the answer is wrong. Label-only training mostly teaches a decision boundary and
can reward accidental shortcuts. The contextual curriculum therefore uses a
grounded multi-task record:

- verdict: `supported`, `refuted`, or `not_enough_information`;
- question answerability and literal offsets grounding its entity, premises,
  requested relation/property and required operands;
- the exact entity, relation/property and value requested by the question;
- supporting or contradicting context character offsets, or an explicit
  `no_supporting_span` marker for NEI;
- a compact error type such as wrong entity, wrong relation, wrong date/value,
  age-versus-year, creator-versus-usage, negation/scope, unsupported addition,
  Unicode-only difference, or invalid derivation;
- for bounded math/date/kinship cases, context-grounded operands, the permitted
  operation and result;
- a short source-grounded justification assembled from those fields, never a
  hidden chain-of-thought target.

For the same `(context, question)`, training includes the correct response and
minimal counterfactual wrong responses produced by changing one controlled
fact: name, date, digit, unit, relation, polarity, scope or event. It also
includes the same passage with a different question, plausible but absent
answers for NEI, and Unicode/word-break variants that must not change meaning.
This forces comparison with the requested property instead of topic matching.

At inference the model performs the learned comparison and emits a structured
verdict, cited offsets and error type. Confidence is derived separately from
the constrained three-label likelihood and calibration-only temperature; it
is not generated text. The model may learn additional
latent reasoning patterns, but the system never assumes that reliable reasons
will emerge by themselves. Synthetic rationales are admitted only when every
claim maps to literal context offsets or a verified deterministic derivation;
otherwise the example remains label-only, lower-weight, or quarantined.

### 6.4 Context-grounded derivation and Bengali-policy curriculum

Contextual support is not limited to verbatim containment. If the supplied
context gives a definition, formula, rule or the operands of a permitted
deduction, a response produced by correctly applying it is `supported` even
when the final answer string does not occur in the passage. Training uses the
following auditable operator families:

1. **Definition or theory application.** Extract the definition/formula and its
   conditions from the context, bind the question's quantities, calculate, and
   check value and unit. For example, when a Bengali passage defines velocity
   and supplies the needed quantities, a correct calculation is supported.
2. **Date, year, age and duration.** Bind the correct person/event and temporal
   relation before arithmetic. Under the benchmark's simple stated-year
   convention, born in 1919 and married in 1940 gives `1940 - 1919 = 21`.
   Exact-date/completed-age questions use calendar dates when supplied; the
   operator ID records which convention was applied.
3. **Bounded relational composition.** Resolve explicit coreference and narrow
   relations such as father's father, while preserving direction and identity.
4. **Bengali grammar-rule application.** Learn sandhi, samas, antonym,
   bagdhara, prefix/suffix, spelling, register, tatsama/native pairing,
   Guruchandali and rule exceptions from high-quality rule-plus-example records.
   If a context states the relevant theory/rule and the response applies it
   correctly, the response is supported.
5. **Entity, title, place and event separation.** Build a local entity-event
   graph from the passage. A name, patronymic, birthplace, residence,
   nationality, office/title and jurisdiction are distinct relations. A passage
   connecting Abdullah bin al-Husain with Mecca, for example, cannot license
   “Amir of Mecca” unless that office relation is actually stated or validly
   derived. Related-topic overlap is never enough.

Before every operator, a question-grounding gate verifies that the passage
contains or unambiguously establishes the question's referents, premises,
requested relation/property and required operands. It then verifies that the
candidate response fills the requested slot rather than answering another
question about the same passage. A false or unsupported presupposition in the
question cannot be silently accepted as a fact.

Each supervised derivation stores `policy_id`, applicability, literal operand
or rule offsets, normalized operands, operator, intermediate typed value, final
value/unit, exceptions checked and verdict. This is a compact public proof
record, not hidden chain-of-thought. Minimal pairs change one operand, relation,
title, date, grammatical decomposition or rule condition while keeping the
rest fixed.

Policies are taught in four waves: explicit rule plus worked example; the same
rule with new entities/numbers/words; mixed long passages where the relevant
rule is not announced; and adversarial exceptions/ambiguities. Source,
document, template and transformation families remain group-held. The model is
evaluated both with and without a visible policy name so success cannot depend
on memorizing policy-ID tokens.

The versioned catalog is `configs/contextual_reasoning_policy_v1.yaml`. Only
human-adjudicated or source-grounded policies enter the gold tier. Hypotheses
remain experimental and cannot override gold labels. Every user correction is
converted into a regression lock plus nearby counterexamples before a
checkpoint is promotable.

The executable record validator is
`pipeline/contextual_training_record.py`. It forbids retrieval fields
recursively, binds question/evidence/rule/operand spans to literal context
offsets and hashes, recomputes supported numeric derivations, requires
authoritative human-grounded mode for grammar/kinship/entity-rule records, maps
the three semantic verdicts to competition labels, and content-hashes every
normalized record. It validates adjudicated supervision; it never invents a
gold verdict.

### 6.5 Closed-book verifier pilot

Target 100,000 admitted records across:

- correct answer with page/URL evidence;
- incorrect answer using MCQ distractors and typed entity/date/relation swaps;
- insufficient, conflicting or time-ambiguous evidence;
- C0 globally stable, C1 Bangladesh-specific/culturally situated, and C2
  recent/contested/time-sensitive facts;
- working coverage registry: closed QA, translation, summarization and
  math/reasoning. The organizer confirms four Phase-2 task types and names the
  closed-book-QA/translation endpoints, but the exact middle names are not
  asserted until the organizer taxonomy is recovered.

The final exact source proportions are frozen after the rights and overlap
ledger. No single generated or translated source may dominate. Current-affairs
records carry publication/event/valid-from/valid-to/access dates.

## 7. Long-context plan

### 7.1 Contextual rows: direct full-context analysis

The supplied context is the evidence universe. The contextual training and
inference path:

1. preserves the complete literal text, UTF-8 bytes and character offsets;
2. creates a separate normalized view only for auxiliary comparison features;
3. feeds the complete supplied context, question and response to the model;
4. predicts question answerability/grounding offsets, verdict, response evidence
   offsets, requested property, error type and derivation fields; confidence is
   computed separately from constrained verdict likelihood;
5. uses deterministic overlapping full-coverage windows plus aggregation only
   when the complete row exceeds the measured model limit.

There is no contextual index, lexical/dense retrieval, reranking, corpus lookup
or counter-evidence search. The model learns context analysis through attention,
explicit span/error supervision and controlled contrastive examples.

Training/evaluation buckets are 512, 1,024, 2,048, 4,096 and 8,192 tokens.
Gold evidence appears at the beginning, middle and end. Distractors use the same
entity with another relation, many dates/numbers, duplicated events and true
no-answer contexts. Report macro-F1, answerability, evidence recall and
beginning/middle/end needle recall by length bucket.

First-window truncation, relevance-based passage dropping and entity/keyword
overlap as support are forbidden. Compact models are tested on their complete
supported lengths. The 4B contextual arm carries the 4K/8K burden after its
tokenizer/throughput canary.

### 7.2 Closed-book rows: evidence-bound retrieval

Only this lane segments rights-admitted books and sources, runs lexical/dense
retrieval, applies source-priority and question-conditioned reranking, searches
for counter-evidence, and constructs compact cited evidence packets. Missing
retrieval is not automatically an incorrect answer. Conflicts or low margins
may be sent to the heavy closed-book judge.

## 8. Model roles and market-screen discipline

### 8.1 Retriever

First queued **closed-book-only** arm:
`ibm-granite/granite-embedding-311m-multilingual-r2` at pinned revision
`44399559930365213510b1ee2eb15ded83374f0e`, Apache-2.0, 311,664,384
parameters. Objective is question-to-supporting-passage retrieval. Similarity is
never a terminal hallucination label.

### 8.2 Compact verifier

Target size is 100M-300M bidirectional. IndicBERT-v3-270M is a leading
hypothesis only after gated-code, tokenizer, license and offline-package audits.
mDeBERTa is a control because the earlier 4,000-row adaptation achieved 0.788
synthetic validation macro-F1 but only 0.660 on 130 competition-context rows,
showing synthetic transfer optimism. BanglaBERT and other Bengali/multilingual
encoders remain in the rights-aware screen.

### 8.3 Heavy contextual reasoner and closed-book evidence judge

The user-authorized contextual-training base is **Gemma 4 E4B only**. Qwen and
other candidates remain benchmark comparators or closed-book component
hypotheses, but no non-Gemma model may receive contextual gradient training.
The exact staged snapshot is
`google/gemma-4-E4B-it-qat-q4_0-unquantized@ddb04d7360c7d7c353532e0e797038439dd1738a`;
its staged tree SHA-256 is
`13e1a72d6a32388c07dc4c404c58a193875afeaddef3ded1091c5a4feeda7658`.
Gemma E4B is selected from measured evidence, not its name: Challenge350
macro-F1 `0.824151024207` and confirmatory-1,000 macro-F1
`0.813838163981408`, including contextual `0.903306474021`, closed-book
`0.665963084976`, 302.305 s on Kaggle T4x2 and 2,895 MiB loaded VRAM per GPU.
Those measurements justify the contextual training hypothesis while exposing
the separate closed-book/retrieval deficit.

The 4B adapter has two explicitly separated record schemas: direct
`(question, full supplied context, response)` contextual analysis and
`(question, response, retrieved evidence, counter-evidence)` closed-book
verification. Retrieved fields are structurally impossible in the contextual
schema. Heavy training does not by itself force heavy inference on every row;
all-row versus selective contextual routing is chosen only from locked
quality-runtime Pareto measurements.

Heavy/light is a routing policy independent of model family.

## 9. Hyperparameter experiments

These are preregistered hypotheses. The canary freezes the winner before a
longer run.

### Granite retriever

- full-parameter BF16/FP16 after numeric canary;
- max query 128 tokens, passage 384 tokens;
- symmetric in-batch InfoNCE with one mined hard negative;
- temperature 0.05;
- AdamW; learning-rate candidates `1e-5`, `2e-5`;
- two epochs; gradient checkpointing;
- effective batch chosen from measured VRAM, never silently changed;
- 100-step canary before one complete source-held fold.

### Compact verifier

- three-way cross-entropy baseline before focal/class weighting;
- auxiliary evidence/property/entity/date/derivation heads tested separately;
- learning-rate candidates `1e-5`, `2e-5`;
- two or three epochs only if group-held loss continues improving;
- 512-token baseline, then 1,024/2,048 evidence-packet variant if supported;
- exact class-balanced 90K pilot;
- calibration trained out-of-fold, never on a locked gate.

### 4B adapter

- one pinned base, 4-bit NF4, BF16 compute when hardware permits;
- LoRA rank 16 versus 32 micro-canaries on identical data/order;
- alpha approximately `2 * rank`, dropout 0.05 as initial hypotheses;
- attention and MLP target-module variants compared in the micro-canary;
- paged AdamW, gradient clipping 1.0, about 3% warm-up;
- one parent example per causal sequence; concatenative packing is
  non-promotable unless verified block-diagonal/document attention isolation
  is implemented later;
- deterministic optimizer-step planning keeps every multi-window parent's
  complete window group within one eight-sequence step. Window target means
  receive `1/window_count`, then the step uses a fixed denominator of eight so
  every selected parent has the same global coefficient across a partial wave
  and forced resume;
- exact CPU materialization found a maximum of nine windows per parent at 2K,
  four at 4K and two at 8K. With eight sequence slots per optimizer step, 2K
  canaries are measurement-only and cannot win the quality selection in this
  release; 4K/8K remain eligible. Changing the episode size requires a new
  measured contract;
- first quality wave is 2M--4M exact native tokens. One sealed-data pass is
  about 8.87M tokens and is earned only by held quality, throughput and cost;
  20M is a later historical ceiling hypothesis, not the initial commitment;
- the throughput canary must hash-bind its producer, exact token workload and
  full evaluation panels, reconcile billed cost from runtime and live H100
  rate, and measure repeated full-gate runtime plus safety margin inside each
  call-boundary reserve;
- split canary evidence into an identical fixed-2K topology comparison and
  token-budget-matched per-limit 2K/4K/8K capacity probes derived from actual
  parent-complete plan prefixes. One short common workload cannot validate 4K
  or 8K behavior. Authenticated OOM/ineligible candidates may complete the
  screen but cannot win;
- quality-call admission uses exact milestone slots with bound initial/final
  steps, sequence ordinals, next-example hashes, nonces and authenticated prior
  terminal receipts. An underfilled or failed slot requires a new recovery
  schedule; it cannot silently consume another of the four calls;
- the approximate 238.4 native-token/s figure is only the zero-overhead floor.
  Final feasibility simulates model loading, encoding, checkpoint publication,
  all micro/full gates (including the first-call double gate), commits and
  finalization inside the actual call walls and rate-derived cost ceiling;
- contextual structured target: class, literal context offsets, requested
  property, error type and verified derivation fields;
- closed-book structured target: class, evidence IDs/relations, conflict state
  and evidence authority. Confidence is computed from constrained label
  likelihood and calibration-only temperature, never generated as gold text;
- no hidden chain-of-thought target or raw model reasoning archive.

The rank/module/learning-rate winner is frozen before labels from a locked gate
are opened. The 50M-token expansion is optional and never automatic.

## 10. Clean-pipeline preflight

Every training arm must pass all checks below.

### Data and split

- pinned revision, size and SHA-256 for every file;
- source and upstream license text preserved;
- exact NFKC/whitespace/case hashes;
- MinHash/SimHash and embedding near-duplicate clusters;
- source/document/passage/question/template/semantic-family group IDs;
- no Phase-1 2,516, Challenge350, locked 1,000-A/1,000-B or inferred
  competition label in training;
- native and translationese reported separately;
- response/evidence labels recomputed from immutable source records;
- source-ID/template shortcuts audited.

### Tokenizer and loader

- token fertility per Unicode code point and word;
- p50/p95/p99 lengths and truncation by source/register;
- combining marks, ZWJ/ZWNJ, Bengali digits, named entities, OCR noise,
  code-mixing, math units and normalization variants;
- deterministic shuffle/order hash;
- 256-record overfit test that drives training loss down and reproduces labels;
- shuffled-label control that must not show real held improvement;
- batch schema and attention masks visually/unit tested.

### Numeric/runtime

- 100 optimizer steps with finite loss/logits/gradients;
- peak VRAM/RAM, examples/s, tokens/s and checkpoint overhead measured;
- interrupt at a checkpoint and resume with identical next-batch hash;
- replay predictions before/after resume;
- cost projection includes image startup, CPU/memory and checkpoint overhead;
- hard timeout and budget reservation written before launch.

## 11. Leakage-safe evaluation design

Outer evaluation holds complete source/document/lineage and semantic clusters.
BanglaCQA and BanglaRQA-derived rows share a lineage. SQuAD- and TyDi-derived
copies share lineage across repositories. Official train/test filenames are not
proof of source independence.

Within admitted training sources, group-hash partitions create train,
out-of-fold calibration and internal evaluation. A separate leave-source-family
gate tests transfer. Competition predictions are frozen before joining any
approved evaluation labels.

Promotion order:

1. intrinsic source-held retrieval or verifier gate;
2. challenge-diverse 350 rows;
3. locked 1,000-A;
4. separately constructed locked 1,000-B;
5. full 2,516 Phase-1 replay;
6. approximately 5,000-row cold Kaggle T4x2 rehearsal.

Metrics: exact macro-F1; class F1; contextual/closed-book lane F1; C0/C1/C2;
task and hard-family slices; grouped bootstrap confidence intervals; ECE/Brier
and reliability; evidence Recall@1/3/5/10, MRR/nDCG; parse/fallback rate;
runtime percentiles; peak RAM/VRAM; and cost. Contextual reporting additionally
includes exact operator/derivation result, unit correctness, policy-family F1,
literal-offset validity, same-passage/different-question consistency, and
entity-role confusion matrices. Question-grounding answerability, premise
validity and requested-slot accuracy are reported separately from response
verification.

## 12. Success and stop rules

Retriever promotion requires source-held Recall@5 improvement, no material
source-floor loss, hard-negative error not worse, and positive downstream
quality. The frozen earlier proposal uses at least +0.03 R@5 on two of three
holdouts and no holdout loss greater than 0.01 as a strong gate.

Verifier canary promotion requires at least +0.005 macro-F1 point gain with no
lane/class loss above 0.005. Final acceptance requires a grouped-bootstrap lower
bound above zero, no lane/class-floor regression, non-worse calibration and
quality-runtime-resource Pareto membership.

Policy promotion also requires every frozen user-adjudication regression lock,
100% schema/offset validity, at least 99.5% exactness on deterministic reference
math/date/unit cases, and no policy-family floor regression. Grammar and complex
entity-relation slices must improve on source-held examples; generated-template
accuracy alone cannot promote a model.

Abort or decline expansion when any occurs:

- NaN/Inf, unstable gradients, corrupted checkpoint or next-batch mismatch;
- rights/overlap/split hash changes;
- held gain is absent after 20-25% of planned steps;
- source-ID/template shortcut control rises;
- general Bengali control drops more than one point or general control more
  than two points without a preregistered compelling trade;
- runtime projection misses the job timeout or final Kaggle ceiling;
- worst-case spend would cross the freshly verified aggregate authorization or
  reduce either active workspace below its per-workspace safety reserve.

## 13. Checkpoint and restart contract

Write atomically every ten minutes or 500-2,000 optimizer steps. Retain last two
and best three. Each checkpoint binds:

- model and adapter/head weights;
- optimizer, scheduler and gradient scaler;
- Python, NumPy, Torch, CUDA and sampler RNG states;
- global step, epoch, streaming shard cursor and completed shard hashes;
- tokenizer/model repository revision and file hashes;
- training configuration and environment-lock hashes;
- data manifest, split and near-duplicate-cluster hashes;
- validation metrics, runtime and cumulative spend snapshot.

A resume is rejected unless every bound hash matches and a dry replay produces
the identical next-batch hash. A cached validation replay must never overwrite
the immutable fresh-run measurement; it writes a separate verification record.

## 14. Training sequence, runtime and dollar ceilings

Current official Modal base GPU rates used for planning are approximately:
L4 USD 0.7992/h, A10 USD 1.1016/h, A100-40GB USD 2.0988/h, A100-80GB
USD 2.4984/h and H100 USD 3.9492/h. Region and non-preemptible multipliers can
raise these; the exact launch projection uses the selected configuration.

The older `$80 from >$90` plan is superseded by the user's later report that
workspaces `ab.hasan revenge` and `abdullah.00001.hasan` each hold only a little
over `$29`. Balances are not transferable by assumption. Until both are read
fresh immediately before launch, paid authorization is zero. If each has
exactly `$29`, preserving `$5` in each gives an aggregate planning ceiling below
`$48`; every launch reserves its own completion, checkpoint and failure margin
inside one workspace. Long jobs run sequentially.

| Stage | Work | Planning wall time | Provisional envelope | Go condition |
|---|---|---:|---:|---|
| 0 | sanitized WebGPT review, rights/split freeze, data tests | 1-4 h local | USD 0 | reviewer findings dispositioned and all preflight hashes frozen |
| 1 | Gemma E4B fresh one-step plus detached forced-resume infrastructure proof | two bounded invocations | <= USD 1.30 reserved | exact staged hashes, next-batch identity, no EOF EINVAL and durable checkpoint proof |
| 2 | Gemma rank/LR/module micro-canaries on the same policy-balanced order | canary-derived | <= USD 3 total | finite/reproducible, no shuffled-label gain, exact throughput projection |
| 3 | first Gemma contextual hard-data wave | canary-derived | <= USD 14 | source/group-held contextual gain and no policy/class floor loss |
| 4 | earned Gemma continuation/replication | canary-derived | <= USD 12 | learning curve, calibration and restart evidence justify it |
| 5 | closed-book Granite/verifier evidence-bound canaries | canary-derived | <= USD 8 | source-held retrieval/downstream direction positive |
| 6 | finalist reproduction and contingency | only when earned | <= USD 9.70 | every active job and safety reserve remain fully funded |

These provisional envelopes total at most `$48`. They are not authorization to
spend: the exact sum is recomputed from fresh dashboard balances and rates, and
unearned stages stay unspent. Only short sub-USD-1 canaries may overlap; all
longer paid jobs are sequential so a shared defect cannot burn multiple
workspaces.

## 15. Remaining-window execution schedule

The schedule is conditional on the external review response.

### Before training

- freeze this document, training YAML, source/rights/split manifests and v2
  review bundle;
- obtain external review and disposition every material finding;
- finish OCR/data work and Challenge350 baseline on local/Kaggle resources;
- bind the exact sealed 22,003-row contextual manifest, separate 27-cell
  diagnostic held extension, ten physical roles and exact-token exposure
  ledger without copying competition data.

### First training wave

- run the Gemma E4B fresh/resume infrastructure proof first;
- run only Gemma contextual micro-canaries, each with a prewritten hard stop;
- closed-book Granite/verifier canaries may use a separate workspace only after
  the contextual canary defect class is cleared and completion reserves remain;
- open results only after rankings/predictions are frozen.

### Main compact wave

- complete one Granite source-held fold;
- complete the 90K verifier pilot;
- evaluate Challenge350 and 1,000-A;
- continue to 20M tokens only when both quality and cost curves support it.

### Heavy wave

- run rank-16/rank-32 micro-canaries on the same selected base/data order;
- train one Gemma E4B adapter on the admitted, policy-balanced hard curriculum;
- evaluate 350, 1,000-A and 1,000-B before replication;
- continue or replicate only when the first wave's measured quality/cost curve
  and the live per-workspace completion reserve support it.

### Finalist wave

- calibrate contextual and closed-book lanes separately;
- replay all 2,516 rows and inspect deltas/manual conflicts;
- run the cold offline approximately 5,000-row Kaggle T4x2 rehearsal;
- freeze hashes, licenses, model/corpus caches and deterministic fallbacks.

## 16. Final inference pipeline sequence

1. Verify model, tokenizer, corpus, index, policy and calibration hashes.
2. Preserve literal UTF-8/byte diagnostics; create a separate NFKC comparison
   view without replacing the literal text.
3. Determine contextual versus closed-book lane without using a label.
4. Parse question entity, requested relation/property, dates/numbers/units,
   negation and scope.
5. Apply deterministic exact grammar/Unicode/math/kinship rules only within
   their validated domains.
6. **Contextual branch:** feed the complete supplied context directly to the
   contextual verifier/reasoner; do not query any index or external corpus.
   Check its cited offsets and derivation against the supplied context.
7. **Closed-book branch only:** retrieve books/datasets/primary sources before
   Wikipedia; preserve page/URL, date and authority, rerank evidence and search
   for counter-evidence.
8. Calibrate each branch separately by lane/family.
9. Apply the locked all-row or selective-heavy policy chosen from measured
   quality/runtime curves; missing closed-book evidence and low-margin/conflict
   cases receive explicit abstention/fallback treatment.
10. If heavy output fails parse, times out or conflicts without resolution, use
    a deterministic conservative fallback and record the reason.
11. Merge by immutable row ID, validate schema/count/order and write predictions.

The 5,000-row target is 6.5 hours, leaving 2.5 hours below Kaggle’s nine-hour
limit for startup, checkpointing, merge, validation and export.

## 17. OCR checkpoint relevant to closed-book data

The first 14-page BCS OCR policy projected 18.53 hours for 1,388 pages and was
not scaled. Adaptive v2 measured 83.5058 seconds for 14 fresh pages and projected
about 2.30 hours, using three fast-only and 11 fallback pages. It fixed a visible
table-layout error on Assurance page 100.

However, a cached replay subsequently overwrote the aggregate fresh-run
manifest at its mutable path. Page-atomic checkpoints remain, but the original
fresh manifest bytes are not currently at that path. The full OCR launch is
stopped until an immutable fresh-run manifest and separate replay-verification
artifact pass regression tests. OCR remains private, nonterminal and
training-ineligible until rights are resolved.

This defect does not enter training; it is documented as a checkpoint lesson.

## 18. External review gate

Before training, create a sanitized deterministic ZIP containing the necessary
code/configs/tests, dataset metadata and papers, present/prior scores, runtime,
VRAM/cost, architecture, intended work, risks, future ideas and the user’s
requirements. Exclude raw datasets/PDF/OCR, competition prompts/responses/labels
or submissions, secrets, account identities, browser state, weights and hidden
chain-of-thought.

The review prompt contains exactly:
`Return the bundle along with thinking and judgements`

It requests visible reasoning summaries, judgments, evidence, counterexamples,
explicit accept/reject/modify decisions, executable acceptance tests and
downloadable sanitized artifacts. Returned findings must be implemented or
declined with evidence; affected tests/gates rerun; response and disposition
hashes frozen.

The requested Chromium `doomers009` profile is not exposed by the current
browser-control capability. Standalone automation cannot be substituted because
it would not prove use of the user’s signed-in profile. Therefore training stays
blocked until the bundle is uploaded through that profile or manually and the
review is returned.

## 19. Demonstrated findings versus hypotheses

| Statement | Status |
|---|---|
| Phase-1 2,516 shadow coverage, 335 queue and 22 manual locks exist | demonstrated |
| TyDi-AS2 and IndicSQuAD local row/schema counts above | demonstrated acquisition metadata |
| Nepali MiniLM performs poorly on the locked narrow Bengali panel | demonstrated; not general hallucination F1 |
| Sparse exact/character retrieval wins that narrow lexical panel | demonstrated only for that panel |
| Adaptive OCR can be about 8x faster than v1 on the 14-page sample | demonstrated historical measurement with mutable-manifest limitation disclosed |
| Granite fine-tuning improves general retrieval/downstream F1 | hypothesis until trained |
| A 90K three-way verifier improves locked macro-F1 | hypothesis until trained |
| IndicBERT is the best compact verifier | hypothesis; not selected |
| Gemma, Qwen or another 4B base is the best heavy judge | hypothesis; market gate decides |
| 20M tokens are enough and 50M are better | staged hypotheses decided by curves |
| Final pipeline finishes 5,000 rows in 6.5 hours | target until cold Kaggle rehearsal |

## 20. User monitoring checklist

Before approving training, verify:

- [ ] external review response and disposition hash exist;
- [ ] exact training dataset counts and class/source/length tables are frozen;
- [ ] every source has rights and disposition;
- [ ] competition/evaluation overlap report is zero or explicitly quarantined;
- [ ] group split hashes and held source families are listed;
- [ ] 256-example overfit and shuffled-label controls pass;
- [ ] tokenizer/truncation report passes;
- [ ] 100-step loss/VRAM/throughput/resume report passes;
- [ ] worst-case job cost plus remaining stages fits below USD 80;
- [ ] more than USD 10 stays outside the program;
- [ ] prediction freeze occurs before label join;
- [ ] no lane/class floor is sacrificed for aggregate F1;
- [ ] final Kaggle run is offline, restart-safe and below nine hours.

## 21. Core public references

- Phase-1/2 lineage: [BenHalluEval](https://arxiv.org/abs/2605.31483)
- Compact factuality checking: [MiniCheck](https://aclanthology.org/2024.emnlp-main.499/)
- RAG hallucination evidence: [RAGTruth](https://aclanthology.org/2024.acl-long.585/)
- Bengali reading QA: [BanglaRQA](https://aclanthology.org/2022.findings-emnlp.186/)
- Bengali contextual/counterfactual QA: [BanglaCQA](https://arxiv.org/abs/2602.01451)
- Bangladesh textbook QA: [NCTB-QA](https://arxiv.org/abs/2603.05462)
- TyDi answer-sentence selection: [TyDi-AS2 dataset](https://huggingface.co/datasets/AmazonScience/tydi-as2), [paper](https://aclanthology.org/2023.findings-acl.796/)
- Large Indic RAG corpus: [Indic-RAG-Suite dataset](https://huggingface.co/datasets/ai4bharat/Indic-Rag-Suite), [paper](https://arxiv.org/abs/2506.01615)
- Modal rates and controls: [pricing](https://modal.com/pricing), [billing report](https://modal.com/docs/cli/latest/billing), [budgets](https://modal.com/docs/guide/budgets)

## 22. Final authority

The latest user instruction, machine-readable budget contract, source/split
hashes and review disposition are jointly authoritative. If they conflict, stop
the affected job and resolve the conflict explicitly. Never silently reinterpret
a budget, label semantic, source right or evaluation gate.

## 23. 17:19 BDT data and policy checkpoint

The exact-source contextual auxiliary is now sealed at
`artifacts/training/contextual_gold_aux_20m_v1_20260719`. Its 71,194 supported
rows contain 30,661,605 exact pinned-Gemma serialized tokens across splits; the
train split contains 65,369 rows / 28,142,534 tokens and its exact 20M pilot
contains 20,000,150 tokens. The independent replay rehashed all 44 bindings
(2,081,659,859 bytes), found zero drift and passed 4/4 focused tests. Manifest
ID: `586f2237a57156fd40d70b05f61ac0abb6c9026a30ab23e1a62574d10d0fee6e`;
manifest SHA-256:
`7577ab08cc3997fbb12c64698328136810d7cbff707b05bf7fb5e19076cd5e84`.

The separate 82,551-row balanced candidate has 27,517 complete three-verdict
sibling groups and 30,866,947 clean train tokens. Only its supported members
are exact-source gold. Its deterministic refuted/NEI siblings are controlled
silver; this tier separation is part of every score and report.

The human/WebGPT generation and review specification is
`prompts/WEBGPT_POLICY_SYNTHETIC_LONG_CONTEXT_GOLD_V1_20260719.md`, SHA-256
`F6B146FC3D1F3FFBA825ECC957C0DF14671B90671837A4DEAFC7C4F8E521AA57`.
It carries the complete bounded analysis procedure, measured competition
lessons, Bengali lexical/grammar distinctions, contextual versus closed-book
routing, C0/C1/C2 axes, evidence-authority policy, counterexample/negation/
quantifier/temporal/math/Unicode/OCR adversaries and leakage controls. Its 26
protected engineered cells are a synthetic curriculum, not a claim that the
organizers defined a canonical or exhaustive 26-family taxonomy. Additional
well-established phenomena are generated as shadow candidates and require a
crisp rule, counterexamples and locked-pilot evidence before admission.

The completion parser and canary/quality executor suites now pass 74/74. A
completion must be the exact suffix after the pinned prompt, end with the one
native terminator, contain no thought/channel/tool/control tokens and decode to
one verdict-first JSON object with the full structured target and valid context
spans. Free-generation parse rate is diagnostic; constrained three-verdict
prefix likelihood plus locked temperature calibration remains label authority.

The repeated-batch generator now targets at least 11,232 fresh core records
before shadow expansion and carries explicit domain floors for school/high-
school mathematics, physics/chemistry formulas, Bangladesh and international
geography/events, neutral religion/religious history, BCS-style reference
cards and publication-time current affairs. Real C1/C2 news requires an exact
snapshot, source/as-of anchor and training-rights proof; otherwise the context
is fully self-contained/fictional or remains quarantined.

Topic and policy are crossed deliberately: every accepted family spans
multiple unrelated domains, every domain carries multiple policies, and the
family x domain x verdict ledger detects shortcut correlations. Science/math
is not formula-only; 40-60% is numeric/formula/diagram/table derivation and the
remainder is conceptual theory, definition, law, classification, condition,
exception, explanation or worked-example transfer. Theory support requires
exact applicability, scope, entity/system, direction and exception binding.

NEI is trained as a terminal verdict with a supervised insufficiency subtype,
not as one homogeneous style. The taxonomy covers missing requested slots,
premises, referents, relation edges, operands/operators/constants/units,
rule-applicability conditions, temporal scope, unresolved source conflicts,
lexical sense/register/operation ambiguity, multiple values, granularity,
modal status, unrecoverable OCR/Unicode, translation ambiguity and unsupported
extra clauses. Hard same-topic NEI and one-evidence completion pairs prevent
shortcut learning; recall/calibration are reported by subtype.

Paid training is still blocked on the marker-clean v3 manifest, rebuilt
immutable canary bundle and base-initialized locked quality pilot. This is a
quality gate, not a reduction of the requested heavy-training ambition.
