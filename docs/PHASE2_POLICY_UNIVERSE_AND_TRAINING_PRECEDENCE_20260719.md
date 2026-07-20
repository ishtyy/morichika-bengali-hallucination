# BICHAR Phase-2 policy universe and training precedence

Status: frozen audit contract; not yet active in optimizer jobs  
Date: 2026-07-19 BDT  
Machine-readable authority: `configs/contextual_reasoning_policy_v2.yaml`

## Outcome

The Phase-2 model must not learn a single shortcut such as “context present,
use literal containment” or “Bengali grammar question, use outside lookup.” It
must first identify exactly what the question asks and then use the one support
mechanism authorized for that operation and evidence state.

The current public empirical anchor is the selected Phase-1 `.979` vector. The
older `.952`, `.961`, `.965`, and `.970` vectors are controlled ablations in its
history, not competing authorities. Public leaderboard movement is aggregate,
rounded evidence: it establishes that an exact vector behaved better or worse
on the public split, but does not reveal every row label or allow the displayed
gain to be assigned uniquely to individual changes.

## Why the earlier contract was incomplete

The v1 contextual contract correctly protected ordinary passages from generic
external retrieval and outside-world rescue. It was incomplete in two ways:

1. Phase-1 isolated probes empirically established narrow fixed lexical shells
   for idiom, antonym, and prefix questions. These shells use the context term
   as an exact lookup key even when the generic passage does not state the
   answer.
2. The corrected samas/sandhi audit established context-scoped deterministic
   grammar. The context must identify the same operation and term/operands, but
   the correctly derived form or class need not appear verbatim.

This does not authorize generic retrieval for contextual rows. A recognized
lexical shell may use an exact, hash-bound policy table or lexical knowledge
learned from admitted sources. It may not search an arbitrary corpus and choose
a semantically similar answer. Samas/sandhi may execute an admitted rule only
when the operation and operands match.

## Four distinct support mechanisms

| Mechanism | When it applies | Permitted support | Fails when |
|---|---|---|---|
| Ordinary context entailment or bounded derivation | Normal supplied passage, including factual, theory, legal, numeric and relational questions | Exact passage claim; supplied definition/formula/rule; bounded arithmetic, calendar, age, unit, ordinal or kinship derivation over passage operands | Wrong/missing entity, relation, property, event, time, scope, polarity, unit, answer type or operand; outside truth cannot rescue it |
| Fixed lexical shell exact pair | Exact organizer shell for antonym, idiom meaning or prefix origin | Exact operation/key record from an admitted dictionary/grammar/textbook or independently corroborated curated pair; trained lexical memory or a hash-bound exact table | Ambiguous shell/sense, approximate semantic opposite, wrong register/etymology, partial key, conflict or source-only guess |
| Context-scoped deterministic grammar | Exact samas or sandhi operation with the same term/ব্যাসবাক্য or operands established by context | Versioned grammar rule and deterministic execution | Wrong grammar operation, different term, missing operand or unrelated context; isolated truth does not rescue scope mismatch |
| Closed book | No supplied context | Rights-admitted offline retrieval, evidence ranking, counter-evidence and separately calibrated factual judgment | Retrieval miss alone is not false; printed key or one weak source alone is nonterminal |

## Exact antonym policy

“Opposite in meaning” is not enough. The model or exact table must verify:

- the requested key lexeme;
- the intended sense of a polysemous word;
- the conventional dictionary/grammar pair;
- the grammatical and register class;
- tatsama, native Bengali, Persian or other etymological compatibility when the
  question/rule makes it relevant;
- the question polarity and exact requested operation.

Only NFC-equivalent spelling or an explicitly attested orthographic variant of
the same lexeme is allowed as bounded equivalence. Embedding neighbors,
paraphrases, a generally plausible opposite, cross-register Guruchandali
pairing, and an answer belonging to another sense are nonterminal. Conflicts
abstain for human or stronger-authority review.

The measured evidence is narrow and must stay narrow:

- 10 exact antonym changes raised `.961` to `.964`;
- adding exact prefix changes with the antonyms produced `.965`;
- the later 30-row antonym expansion raised `.965` only to `.966`.

These results support the exact-pair family. They do not prove all 52 Phase-1
antonym rows, and they do not support turning every plausible opposite into
label 1.

## Complete Context-427 operation inventory

| Operation | Rows | Support mechanism |
|---|---:|---|
| Antonym lookup | 52 | Fixed lexical shell exact pair |
| Birth-date slot | 156 | Ordinary exact entity/date relation |
| Entity-text relation | 11 | Ordinary exact entity/work relation |
| Event date or status | 22 | Ordinary exact event/relation/time |
| Idiom meaning | 59 | Fixed lexical shell exact meaning |
| Legal definition or threshold | 1 | Ordinary supplied legal rule |
| Legal effective date | 1 | Ordinary exact date role |
| Legal maximum fine | 1 | Ordinary comparator/amount |
| Legal maximum imprisonment | 3 | Ordinary comparator/unit |
| Legal minimum meeting frequency | 1 | Ordinary minimum/frequency |
| Location slot | 1 | Ordinary entity/event location |
| Numeric or ordinal slot | 13 | Ordinary relation/scope/unit/order |
| Prefix origin/class | 15 | Fixed lexical shell exact subterm/class |
| Samas taxonomy | 49 | Context-scoped deterministic grammar |
| Sandhi formation | 42 | Context-scoped deterministic grammar |

Totals are 210 ordinary explicit-relation rows, 126 lexical-shell rows and 91
deterministic grammar rows, exactly 427.

## Broader policy families the training set must cover

The 15-operation Phase-1 inventory is a measured seed, not the complete Phase-2
universe. Source/group-independent training and evaluation must also cover:

- question answerability, premise validity and same-passage/different-question;
- direct support, contradiction, partial containment and wrong answer type;
- supplied definitions, formulas, theories and grammar rules;
- event/date/as-of roles, exact age, duration, calendar and timezone;
- arithmetic, ratios, percentages, units, totals/components, ranges, extrema and
  ordinal relations;
- kinship and bounded relational composition;
- creator/founder versus user/operator; birthplace versus residence,
  nationality, office, event place and jurisdiction;
- legal section, definition, publication/enactment/effective date, minimum,
  maximum, fine, imprisonment and frequency;
- negation, quantifier, modality, comparator and clause scope;
- antonym, idiom, prefix, affix, register, etymology, Guruchandali, spelling,
  natva and satva;
- samas/sandhi operation and operand collisions;
- Unicode NFC/NFD, joiners, conjunct/hasanta, Bengali/ASCII digits,
  punctuation, OCR digit loss and word/line breaks;
- ambiguity, contradictory anchors, invalid premises, invalid arithmetic and
  genuine missing information.

These bullets are policy/operator families. They must not be confused with
the benchmark's independent cultural-distance, task, domain, source and label
axes. A repository-wide family recovery ledger is being frozen separately so
aliases such as `date_age_duration`, `event_date_or_status_slot` and
`temporal_scope` can be related without pretending that they are identical.

## Independent benchmark axes: cultural distance and task type

Every closed-book record receives an independently reviewed cultural-distance
band:

| Band | Meaning | Required policy |
|---|---|---|
| C0 | Globally stable and expected to be language-invariant: universal science, world geography and mathematics | Stable primary/edited sources; cross-lingual evidence is allowed only after entity, number, date, unit and negation-preservation checks |
| C1 | Culturally situated or Bangladesh-specific; the local answer may differ from a globally dominant default | Prefer Bangladesh primary/local and register-appropriate authority; a Western/global default cannot overwrite the scoped Bangladeshi answer |
| C2 | Contested, recent or time-sensitive | Bind event, publication, effective, valid-from, valid-to, access and explicit as-of dates; keep conflicts; abstain when the temporal scope cannot be reconciled |

Keywords may propose a band for review, but cannot terminally assign it. C1 and
C2 can share secondary flags—e.g. a recent Bangladesh civic fact—but evaluation
uses one adjudicated primary band and preserves the secondary cultural and
temporal flags.

The organizer statement confirms four Phase-2 task types and names the range
from closed-book QA to translation, but no recovered local organizer artifact
enumerates the two middle names. The project therefore uses closed-book QA,
translation, summarization and math/reasoning as a **working coverage registry**,
not a claim that these are the organizer's exact four names. The BenHalluEval
paper's QA/reasoning/code-mixed-QA/summarization families and our 36-row
translation-control gate remain separate concepts. Task type does not override
whether a supplied context exists. The working coverage contracts are:

- closed-book QA requires evidence-bound factual or lexical verification;
- translation preserves meaning, entities, numbers, dates, negation, register,
  omissions and additions;
- summarization requires every material summary claim to be supported and
  forbids critical distortion or unsupported additions;
- math/reasoning verifies premise, interpretation, operator, operands, value
  and unit, including alternative interpretations and invalid premises.

Metrics and calibration therefore slice by lane, C0/C1/C2, organizer task type
when released (plus the working coverage registry until then), domain, source
family, policy/operator family, three-way verdict and language stratum. Counts
from these axes are never added into a single misleading “family” total.

## Math, logic, fact checking and counter-evidence

The recovered executable policy is more specific than generic semantic
similarity:

1. Parse the requested answer type, entity, relation/property, polarity,
   quantifier, comparator, numeric operands, units and temporal scope.
2. Reject a retrieved or supplied candidate that answers the same topic but a
   different relation, event, date role or question.
3. For math, enumerate bounded plausible interpretations, check the premise,
   reject zero denominators or missing operands, calculate with typed units and
   preserve ambiguity instead of selecting the most convenient result.
4. Negation (`না/নয়/নেই`), exceptions, all/some, minimum/maximum,
   before/after and causal versus correlational scope are hard structural
   features. A semantic match may not erase them.
5. A counterexample must contradict the exact scoped claim. A retrieval miss,
   different-question source, noisy printed key or low-authority disagreement
   is not counter-evidence by itself.
6. Support and counter-evidence retain separate source locators, authority,
   question alignment and as-of scope. Unresolved high-authority conflict is
   NEI, not a forced binary answer.

## Heavy retrieval architecture and the 0.970 anchor

The repository already contains the heavy evidence-bound architecture: exact
and character-sparse retrieval, structural query gates, high-authority source
priority, semantic residual retrieval, reranking, support/counter-evidence
packets, an evidence judge, separate closed-book calibration and deterministic
abstaining fallbacks. Heavy/light is a routing choice independent of model
family.

The Phase-1 evidence for this layer must be stated exactly. Closed-source10
scored `.968` from the `.965` base; its union with lexical34 scored `.970`
(Kaggle ref `54797211`, SHA
`8eee20aba3f80572ef864aecee92cfafb433cd3a136fea9211aec505f5a9d223`).
That is measured evidence for the source/lexical policy stack. It does not by
itself prove that the complete Phase-2 heavy retriever, reranker and judge are
the final Pareto winner. The selected Phase-1 vector is the later `.979`
combined22 result, while Phase-2 still requires a second locked 1,000 variant,
separate calibration ledgers and the full 5,000-row Kaggle-offline rehearsal.

## Decision precedence

For each record, the system must execute this sequence:

1. Preserve literal bytes, code points and offsets; build comparison views
   separately.
2. Select supplied-context versus closed-book lane without looking at a label.
3. Parse the question's entity, premises, relation/property, event/time/as-of
   role, scope, polarity, comparator, unit and answer type.
4. Recognize an exact host operation; an uncertain shell stays unrecognized.
5. Verify context applicability, coreference, exact term and required operands.
6. Select exactly one of the four support mechanisms or abstain.
7. Evaluate direct support/contradiction and supplied theory.
8. Execute only permitted bounded derivations.
9. Use exact lexical-pair knowledge only for the recognized lexical shell.
10. Use deterministic grammar only for the matching operation and operands.
11. Use Unicode normalization only to compare meaning-preserving variants.
12. Preserve conflicts, multiple plausible values and counter-evidence.
13. Emit supported, refuted or NEI with a compact visible proof record.

Direct context contradiction beats outside truth. Exact question/relation match
beats topic similarity. A high-authority source answering the wrong question
loses to evidence aligned to the exact question. Invalid premise or invalid
math fails closed. Ambiguity is never forced to supported.

## Training implications

- Phase-1 IDs, text and inferred labels remain evaluation/regression-only. They
  cannot enter optimizer data.
- Build analogous examples from independent, rights-admitted books, BCS
  sources, authoritative grammar/dictionaries, contextual QA corpora and owned
  formal generators.
- Preserve three-way supported/refuted/NEI targets. Historical binary zero may
  not be imported as refuted without new adjudication.
- Every training row records its support mechanism, operation, requested slot,
  premise validity, context applicability, evidence/contradiction offsets,
  derivation operands, exact lexical source when relevant, ambiguity and
  counter-evidence.
- Minimal-pair groups must include same passage/different question, one-field
  entity/relation/date changes, age versus year, creator versus usage,
  total/component, minimum/maximum, exact antonym versus near opposite,
  register-correct versus Guruchandali, samas versus sandhi, and Unicode
  equivalent versus meaning-changing word break.
- Generated examples remain quarantined until independent semantic, rights,
  overlap and group-split admission. Volume never substitutes for correctness.

## Activation gate

The v2 contract is frozen but deliberately not wired into a paid quality run.
Activation requires the full Phase-1 score trajectory to be hash-bound, all 15
operations and user-regression cases to pass, refuted versus NEI to be clean,
and every source/training record to pass rights, provenance, semantic truth,
competition-overlap and group-split checks. Until then, a GPU training run could
be operationally valid but semantically wrong, so quality training remains
blocked.

Focused exact-pair/cache/router/policy tests after the cultural/task-axis update:
31/31 passed; the two policy-contract files separately pass 14/14. This is a
policy/schema proof, not a model-quality result.
