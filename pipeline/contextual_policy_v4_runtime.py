"""Context-only policy packet for the MORICHIKA generalized hybrid.

The module is intentionally self-contained so its exact source can be embedded
in an offline Kaggle notebook.  It never retrieves, labels, or consults a
competition row identifier.  It converts the supplied context, question and
candidate response into an auditable instruction packet for the verifier.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Callable


VERSION = "morichika-context-policy-v4-full-coverage-map-aggregate"

# These are reasoning checks, not predicted classes.  Keeping the recovered
# inventory explicit prevents a later prompt edit from silently dropping an
# edge-case family.
CANONICAL_POLICY_FAMILIES = (
    "question_grounding_answerability_and_premise_validity",
    "exact_entity_relation_property_and_answer_type",
    "direct_support_contradiction_and_partial_containment",
    "supplied_definition_formula_theory_and_rule_application",
    "date_event_role_and_as_of_time",
    "year_age_duration_calendar_and_timezone",
    "bounded_arithmetic_units_ratio_percentage_and_order",
    "kinship_and_relational_composition",
    "creator_founder_user_operator_office_title_and_jurisdiction",
    "birthplace_residence_nationality_and_event_participation",
    "legal_definition_section_effective_date_minimum_maximum_and_frequency",
    "numeric_whole_component_range_extremum_ordinal_and_granularity",
    "negation_quantifier_comparator_modality_and_clause_scope",
    "antonym_idiom_prefix_register_etymology_and_guruchandali",
    "samas_sandhi_affix_spelling_natva_and_satva",
    "unicode_joiner_conjunct_digit_punctuation_ocr_and_word_break",
    "ambiguity_conflict_invalid_premise_and_no_world_rescue",
)

ENGINEERED_EVALUATION_CELLS = (
    "context_only_lane", "full_context_boundary", "answerable_supported",
    "answerable_refuted", "genuinely_missing_nei", "same_passage_different_question",
    "entity_swap", "relation_swap", "answer_type_swap", "creator_vs_user_operator",
    "birthplace_vs_residence_event_place", "partial_answer_extra_claim",
    "theory_application", "formula_operand_application", "age_vs_year",
    "relative_timeline", "event_phase_order", "kinship_composition",
    "unit_or_number_swap", "negation_scope", "quantifier_comparator_scope",
    "grammar_operation_swap", "lexical_exact_vs_semantic_near",
    "unicode_equivalent", "unicode_word_break_non_equivalent", "cross_window_conflict",
)

OPERATION_AXIS = (
    "antonym_lookup", "birth_date_slot", "entity_text_relation",
    "event_date_or_status_slot", "idiom_meaning_lookup",
    "legal_definition_or_threshold", "legal_effective_date", "legal_maximum_fine",
    "legal_maximum_imprisonment", "legal_minimum_meeting_frequency",
    "location_slot", "numeric_or_ordinal_slot", "prefix_origin_classification",
    "samas_taxonomy", "sandhi_formation",
)

_SIGNALS: tuple[tuple[str, str], ...] = (
    ("year_age_duration", r"ব[য়য়]স|জন্ম|বিবাহ|বিয়ে|বিয়ে|বছর|সাল|year|age"),
    ("calendar_date", r"তারিখ|কবে|দিন|মাস|জানুয়ারি|ফেব্রুয়ারি|মার্চ|ডিসেম্বর"),
    ("timeline_event_order", r"আগে|পরে|তারপর|পরবর্তীতে|প্রথমে|শেষে|ঘটে|ঘটেছিল"),
    ("relative_time_offset", r"(?:বছর|দিন|মাস)\s*(?:আগে|পরে)|after|before|later"),
    ("kinship_relation_composition", r"বাবা|পিতা|মা|মাতা|দাদা|পিতামহ|নানা|grandfather"),
    ("bounded_arithmetic", r"কত|যোগ|বিয়োগ|বিয়োগ|গুণ|ভাগ|শতাংশ|হিসাব|calculate"),
    ("unit_dimension", r"কিলোমিটার|মিটার|গ্রাম|লিটার|সেকেন্ড|মিনিট|ঘণ্টা|টাকা|শতাংশ"),
    ("negation_scope", r"(?:^|\s)(?:না|নয়|নয়|নেই|নি)(?:\s|$)|হয়নি|হয়নি|করেনি"),
    ("quantifier_scope", r"সব|সকল|কোনো|কোনও|কেবল|শুধু|প্রত্যেক|একটিও|কিছু"),
    ("comparator_scope", r"সর্বাধিক|সর্বনিম্ন|বেশি|কম|পূর্ববর্তী|পরবর্তী|minimum|maximum"),
    ("modality_clause_scope", r"পারে|উচিত|অবশ্যই|সম্ভব|শর্তে|যদি|হলে"),
    ("definition_theory_rule_application", r"সংজ্ঞা|তত্ত্ব|সূত্র|নিয়ম|নিয়ম|অর্থ"),
    ("antonym_exact_pair", r"বিপরীত\s*শব্দ|antonym"),
    ("idiom_exact_meaning", r"বাগধারা|প্রবাদ"),
    ("prefix_suffix_origin", r"উপসর্গ|প্রত্যয়|প্রত্যয়"),
    ("samas_exact_operation", r"সমাস"),
    ("sandhi_exact_operands", r"সন্ধি"),
    ("register_etymology_guruchandali", r"গুরুচণ্ডালী|তৎসম|তদ্ভব|ফারসি|সংস্কৃত"),
    ("natva_satva_spelling", r"ণত্ব|ষত্ব|বানান"),
    ("creator_user_operator_role", r"স্রষ্টা|প্রতিষ্ঠাতা|নির্মাতা|ব্যবহারকারী|পরিচালক"),
    ("birthplace_residence_event_place", r"জন্মস্থান|বাসস্থান|রাজধানী|কোথায়|কোথায়|স্থান"),
    ("legal_definition_section", r"আইন|ধারা|বিধি|সংজ্ঞা"),
    ("minimum_maximum_frequency", r"সর্বনিম্ন|সর্বোচ্চ|কমপক্ষে|বেশিরভাগ|বার"),
    ("ordinal_list_order", r"প্রথম|দ্বিতীয়|দ্বিতীয়|তৃতীয়|তৃতীয়|ক্রম|তম"),
)

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_JOINERS = {"\u200c", "\u200d", "\ufeff"}
_FIXED_LEXICAL_SHELLS = {
    "antonym_lookup": re.compile(r"(?:বিপরীত\s*শব্দ|বিপরীতার্থক\s*শব্দ).*(?:কী|কি|কোন)", re.IGNORECASE),
    "idiom_meaning_lookup": re.compile(r"(?:বাগধারা|প্রবাদ).*(?:অর্থ|মানে).*(?:কী|কি|কোন)", re.IGNORECASE),
    "prefix_origin_classification": re.compile(r"উপসর্গ.*(?:উৎস|শ্রেণি|জাতীয়|জাতীয়|কোন)", re.IGNORECASE),
}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def comparison_view(value: str) -> str:
    """Bounded comparison only; never rewrites literal evidence."""
    return " ".join(unicodedata.normalize("NFC", value).translate(_BN_DIGITS).casefold().split())


def unicode_receipt(value: str) -> dict[str, Any]:
    nfc = unicodedata.normalize("NFC", value)
    return {
        "literal_sha256": _sha(value),
        "utf8_bytes": len(value.encode("utf-8")),
        "codepoints": len(value),
        "nfc_identical": nfc == value,
        "nfc_sha256": _sha(nfc),
        "joiner_positions": [index for index, char in enumerate(value) if char in _JOINERS],
        "replacement_character_positions": [index for index, char in enumerate(value) if char == "\ufffd"],
        "comparison_view_sha256": _sha(comparison_view(value)),
    }


def contextual_exact_lexical_policy(
    question: str, response: str, records: list[dict[str, Any]] | None
) -> dict[str, Any]:
    """Expose only the frozen exact lexical-shell exception, nonterminally."""
    folded_question = comparison_view(question)
    folded_response = comparison_view(response)
    operation = next(
        (name for name, pattern in _FIXED_LEXICAL_SHELLS.items() if pattern.search(folded_question)),
        None,
    )
    base = {
        "lookup_mode": "not_applicable",
        "operation": operation,
        "key": None,
        "source_ids": [],
        "record_sha256": [],
        "exact_operation": operation is not None,
        "exact_key": False,
        "exact_sense_register": False,
        "conflict": False,
        "conflict_status": "none",
        "generic_retrieval_invoked": False,
        "terminal_label_authority": False,
        "model_nonterminal": True,
        "evidence": [],
    }
    if operation is None:
        base["lookup_mode"] = "forbidden_for_ordinary_context"
        return base

    matched: list[dict[str, Any]] = []
    conflicting: list[dict[str, Any]] = []
    for record in records or []:
        if record.get("operation") != operation:
            continue
        terms = [str(record.get("term_key") or "")] + [str(v) for v in record.get("display_terms", [])]
        exact_terms = []
        for term in terms:
            key = comparison_view(term)
            if key and re.search(rf"(?<![\u0980-\u09FFA-Za-z0-9]){re.escape(key)}(?![\u0980-\u09FFA-Za-z0-9])", folded_question):
                exact_terms.append(key)
        if not exact_terms:
            continue
        sense_values = [str(record.get(name) or "") for name in ("sense", "register", "etymological_class")]
        required_scope = [comparison_view(value) for value in sense_values if value]
        exact_scope = not required_scope or all(value in folded_question for value in required_scope)
        candidate = {**record, "_matched_key": sorted(exact_terms, key=lambda value: (-len(value), value))[0], "_exact_scope": exact_scope}
        if record.get("conflict_status") == "none" and exact_scope:
            matched.append(candidate)
        else:
            conflicting.append(candidate)

    if not matched and not conflicting:
        base["lookup_mode"] = "exact_shell_no_exact_key_nonterminal"
        return base
    all_candidates = matched + conflicting
    keys = {value["_matched_key"] for value in all_candidates}
    accepted_sets = {
        tuple(sorted(comparison_view(v) for v in value.get("accepted_answers", []) if str(v).strip()))
        for value in matched
    }
    conflict = bool(conflicting) or len(keys) != 1 or len(accepted_sets) > 1
    base.update({
        "lookup_mode": "exact_hash_bound_lexical_policy" if not conflict else "exact_lexical_conflict_nei_nonterminal",
        "key": sorted(keys)[0] if len(keys) == 1 else None,
        "source_ids": sorted({str(value.get("source_id") or "") for value in all_candidates if value.get("source_id")}),
        "record_sha256": sorted({_sha(_canonical({k: v for k, v in value.items() if not str(k).startswith("_")})) for value in all_candidates}),
        "exact_key": len(keys) == 1,
        "exact_sense_register": all(value.get("_exact_scope") is True for value in all_candidates),
        "conflict": conflict,
        "conflict_status": "conflict" if conflict else "none",
    })
    if not conflict:
        for value in matched[:3]:
            accepted = [comparison_view(v) for v in value.get("accepted_answers", [])]
            contrast = [comparison_view(v) for v in value.get("contrast_answers", [])]
            base["evidence"].append({
                "source_id": value.get("source_id"),
                "operation": operation,
                "term_key": value.get("term_key"),
                "accepted_answers": value.get("accepted_answers", []),
                "contrast_answers": value.get("contrast_answers", []),
                "response_relation": (
                    "support_candidate" if folded_response in accepted else
                    "counter_candidate" if folded_response in contrast else "neutral_candidate"
                ),
            })
    return base


def detected_policy_families(context: str, question: str, response: str) -> list[str]:
    joined = "\n".join((question, response, context))
    detected = {
        "lane_context_only", "full_context_evidence_boundary", "question_answerability",
        "question_premise_validity", "exact_answer_slot", "same_passage_different_question",
        "entity_identity", "relation_property_identity", "answer_type_identity",
        "direct_support", "direct_contradiction", "partial_containment_extra_claim",
        "ambiguity_conflict_nei", "unicode_nfc_joiner_conjunct_digit", "ocr_word_break_caution",
    }
    for family, pattern in _SIGNALS:
        if re.search(pattern, joined, re.IGNORECASE):
            detected.add(family)
    if re.search(r"[০-৯0-9]", joined):
        detected.update({"number_whole_component_range", "formula_operand_application"})
    order = [name for name, _ in _SIGNALS]
    order += [
        "lane_context_only", "full_context_evidence_boundary", "question_answerability",
        "question_premise_validity", "exact_answer_slot", "same_passage_different_question",
        "entity_identity", "relation_property_identity", "answer_type_identity",
        "direct_support", "direct_contradiction", "partial_containment_extra_claim",
        "ambiguity_conflict_nei", "unicode_nfc_joiner_conjunct_digit", "ocr_word_break_caution",
        "number_whole_component_range", "formula_operand_application",
    ]
    return [family for family in dict.fromkeys(order) if family in detected]


def full_coverage_windows(
    context: str, *, max_chars: int = 3000, overlap_chars: int = 320
) -> list[dict[str, Any]]:
    """Partition every context character into overlapping replayable windows."""
    if not context:
        raise ValueError("context must be non-empty")
    if max_chars < 256 or overlap_chars < 32 or overlap_chars >= max_chars:
        raise ValueError("invalid full-coverage window geometry")
    windows: list[dict[str, Any]] = []
    start = 0
    while start < len(context):
        hard_end = min(len(context), start + max_chars)
        end = hard_end
        if hard_end < len(context):
            boundaries = [context.rfind(mark, start + max_chars // 2, hard_end) for mark in ("\n", "।", ".", "?", "!")]
            boundary = max(boundaries)
            if boundary >= start:
                end = boundary + 1
        literal = context[start:end]
        windows.append({
            "window_index": len(windows),
            "context_char_start": start,
            "context_char_end": end,
            "literal_text": literal,
            "literal_span_sha256": _sha(literal),
        })
        if end == len(context):
            break
        start = end - overlap_chars
    covered = [False] * len(context)
    for index, window in enumerate(windows):
        start, end = window["context_char_start"], window["context_char_end"]
        literal = window["literal_text"]
        if context[start:end] != literal or _sha(literal) != window["literal_span_sha256"]:
            raise ValueError(f"full-coverage window {index} does not replay")
        for position in range(start, end):
            covered[position] = True
    if not all(covered):
        raise ValueError("full-coverage window ledger contains a character gap")
    return windows


def build_window_adjudication_user(
    window: dict[str, Any], question: str, response: str, total_windows: int
) -> str:
    return (
        "WINDOW-LOCAL CONTEXT PASS. Judge only evidence present in this literal supplied-context window. "
        "A local miss is not a contradiction: output not_enough_information unless this window directly "
        "supports or directly refutes the exact question slot. Do not use outside knowledge.\n\n"
        f"QUESTION (literal):\n{question}\n\nCANDIDATE RESPONSE (literal):\n{response}\n\n"
        f"WINDOW {window['window_index'] + 1}/{total_windows} "
        f"[chars={window['context_char_start']}:{window['context_char_end']}; "
        f"sha256={window['literal_span_sha256']}]:\n{window['literal_text']}"
    )


def _literal_question_excerpt(
    window: dict[str, Any], question: str, max_chars: int
) -> dict[str, Any]:
    """Select a literal, question-conditioned excerpt without rewriting it."""
    literal = str(window["literal_text"])
    if max_chars < 32:
        max_chars = 32
    q_tokens = {
        token.casefold() for token in re.findall(r"[\u0980-\u09FFA-Za-z0-9]+", question)
        if len(token) > 1
    }
    candidates = list(re.finditer(r"[^।.!?\n]+[।.!?]?", literal))
    if candidates:
        def score(match: re.Match[str]) -> tuple[int, int]:
            tokens = {
                token.casefold() for token in re.findall(r"[\u0980-\u09FFA-Za-z0-9]+", match.group())
            }
            return (len(tokens & q_tokens), -match.start())
        best = max(candidates, key=score)
        local_start = max(0, best.start() - max_chars // 4)
        local_end = min(len(literal), local_start + max_chars)
    else:
        local_start, local_end = 0, min(len(literal), max_chars)
    excerpt = literal[local_start:local_end]
    absolute_start = int(window["context_char_start"]) + local_start
    absolute_end = absolute_start + len(excerpt)
    return {
        "excerpt_char_start": absolute_start,
        "excerpt_char_end": absolute_end,
        "literal_excerpt": excerpt,
        "literal_excerpt_sha256": _sha(excerpt),
    }


def build_aggregation_user(
    question: str,
    response: str,
    window_results: list[dict[str, Any]],
    *,
    selected_notes: list[dict[str, Any]] | None = None,
    bounded_derivations: list[dict[str, Any]] | None = None,
    lexical_policy: dict[str, Any] | None = None,
) -> str:
    """Create the exact-question final adjudication over all window passes."""
    compact = [
        {
            "window_index": row["window_index"],
            "context_char_start": row["context_char_start"],
            "context_char_end": row["context_char_end"],
            "literal_span_sha256": row["literal_span_sha256"],
            "window_verdict": row["window_verdict"],
            "literal_excerpt": row.get("literal_excerpt", ""),
            "excerpt_char_start": row.get("excerpt_char_start"),
            "excerpt_char_end": row.get("excerpt_char_end"),
            "literal_excerpt_sha256": row.get("literal_excerpt_sha256"),
        }
        for row in window_results
    ]
    notes = list(selected_notes or [])[:32]
    derivations = list(bounded_derivations or [])[:8]
    return (
        "FINAL FULL-CONTEXT AGGREGATION. The window ledger collectively covered every supplied-context "
        "character. Rebind the exact question and candidate response. A not_enough_information window means "
        "only that its local span was silent; it is never counter-evidence. If any aligned window refutes the "
        "response, do not let an unrelated support override it. Unresolved support/refutation across different "
        "entities, slots, event phases, dates, operations, or ambiguous coreference is not_enough_information. "
        "Use no outside knowledge.\n\n"
        f"QUESTION (literal):\n{question}\n\nCANDIDATE RESPONSE (literal):\n{response}\n\n"
        "PER-WINDOW STRUCTURED RESULTS AND LITERAL EXCERPTS:\n" + _canonical(compact)
        + "\n\nFULL-CONTEXT ROUTER SOURCE-LINKED NOTES (advisory; offsets/hashes bind them to supplied context):\n"
        + _canonical(notes)
        + "\n\nBOUNDED DERIVATION CANDIDATES (advisory; verify operator, operands, slot and unit):\n"
        + _canonical(derivations)
        + "\n\nFROZEN EXACT LEXICAL-SHELL POLICY (nonterminal; ordinary retrieval forbidden):\n"
        + _canonical(lexical_policy or {"lookup_mode": "not_applicable", "generic_retrieval_invoked": False})
    )


def build_contextual_policy_packet(
    context: str,
    question: str,
    response: str,
    router: Callable[..., dict[str, Any]],
    lexical_records: list[dict[str, Any]] | None = None,
    *,
    max_windows: int = 8,
    full_context_char_limit: int = 6000,
) -> tuple[str, dict[str, Any]]:
    """Build a context-only prompt plus restart-safe diagnostics."""
    if not all(isinstance(value, str) for value in (context, question, response)):
        raise TypeError("context, question and response must be strings")
    if not context.strip() or not question.strip():
        raise ValueError("context and question must be non-empty")

    route = router(context, question, response, max_windows=max_windows)
    if route.get("external_retrieval_allowed") is not False:
        raise ValueError("context router must explicitly forbid external retrieval")
    if route.get("context_sha256") != _sha(context):
        raise ValueError("context router hash mismatch")

    window_calls: list[dict[str, Any]] = []
    if len(context) <= full_context_char_limit:
        evidence = context
        evidence_mode = "complete_literal_supplied_context"
        full_context_inference_coverage = True
    else:
        coverage = full_coverage_windows(context)
        excerpt_chars = max(48, min(320, 3200 // len(coverage)))
        window_calls = [
            {
                **{key: window[key] for key in (
                    "window_index", "context_char_start", "context_char_end", "literal_span_sha256"
                )},
                **_literal_question_excerpt(window, question, excerpt_chars),
                "user": build_window_adjudication_user(window, question, response, len(coverage)),
            }
            for window in coverage
        ]
        # The ordinary single-call payload is never used for a long context;
        # it is deliberately tiny so no truncated projection can be mistaken
        # for the evidence universe.
        evidence = "[FULL-COVERAGE WINDOW PASSES REQUIRED BEFORE AGGREGATION]"
        evidence_mode = "all_literal_windows_then_final_aggregation"
        full_context_inference_coverage = True

    signals = detected_policy_families(context, question, response)
    lexical_policy = contextual_exact_lexical_policy(question, response, lexical_records)
    aggregation_derivations = [
        value for value in list(route.get("bounded_derivation_candidates") or [])[:8]
        if isinstance(value, dict) and value.get("source_linked") is True
    ]
    operand_ids = {
        str(note_id)
        for value in aggregation_derivations
        for note_id in value.get("operand_note_ids", [])
    }
    routed_notes = list(route.get("selected_notes") or [])
    routed_notes.sort(key=lambda note: (str(note.get("note_id")) not in operand_ids, int(note.get("context_char_start", 0))))
    aggregation_notes = []
    note_budget = 2400
    used_note_chars = 0
    for note in routed_notes[:32]:
        start = int(note["context_char_start"])
        end = int(note["context_char_end"])
        literal = str(note["literal_text"])
        if context[start:end] != literal or _sha(literal) != note["literal_span_sha256"]:
            raise ValueError("router selected note does not replay against supplied context")
        serialized_chars = len(_canonical(note))
        if aggregation_notes and used_note_chars + serialized_chars > note_budget:
            continue
        aggregation_notes.append(note)
        used_note_chars += serialized_chars
    notes = {
        "selected_notes": aggregation_notes,
        "bounded_derivation_candidates": aggregation_derivations,
    }
    contract = {
        "version": VERSION,
        "evidence_universe": "only_the_literal_supplied_context",
        "external_retrieval_allowed": False,
        "outside_world_fact_rescue_allowed": False,
        "evidence_mode": evidence_mode,
        "full_context_inference_coverage": full_context_inference_coverage,
        "every_context_character_processed": True,
        "detected_policy_families": signals,
        "checks_in_order": [
            "validate question premise and whether the requested slot is answerable",
            "bind exact entity relation property answer type event phase time polarity comparator and unit",
            "seek direct support and direct contradiction for that exact slot",
            "apply only supplied definition theory rule formula and bounded operands",
            "separate refuted from missing ambiguous conflicting or locally silent window evidence",
            "compare Unicode only through bounded NFC digit joiner and attested word-break views; near-looking forms are not automatically equal",
        ],
        "verdict_rule": {
            "supported": "the complete response follows for the exact requested slot",
            "refuted": "the supplied context establishes an incompatible value or claim",
            "not_enough_information": "required evidence is absent omitted ambiguous conflicting or premise-invalid",
        },
        "fixed_lexical_exception": lexical_policy,
    }
    user = (
        "CONTEXT-ONLY VERIFICATION CONTRACT:\n" + _canonical(contract)
        + "\n\nQUESTION (literal):\n" + question
        + "\n\nSUPPLIED CONTEXT EVIDENCE (literal; this is the only evidence universe):\n" + evidence
        + "\n\nSOURCE-LINKED EXTRACTIVE NOTES (advisory, never new evidence):\n" + _canonical(notes)
        + "\n\nCANDIDATE RESPONSE (verify every material claim):\n" + response
    )
    diagnostic = {
        **route,
        "context_policy_version": VERSION,
        "policy_family_inventory_count": len(CANONICAL_POLICY_FAMILIES),
        "canonical_policy_family_count": len(CANONICAL_POLICY_FAMILIES),
        "engineered_evaluation_cell_count": len(ENGINEERED_EVALUATION_CELLS),
        "operation_axis_count": len(OPERATION_AXIS),
        "canonical_policy_families": list(CANONICAL_POLICY_FAMILIES),
        "engineered_evaluation_cells": list(ENGINEERED_EVALUATION_CELLS),
        "operation_axis": list(OPERATION_AXIS),
        "detected_policy_families": signals,
        "evidence_mode": evidence_mode,
        "full_context_inference_coverage": full_context_inference_coverage,
        "context_literal": unicode_receipt(context),
        "question_literal": unicode_receipt(question),
        "response_literal": unicode_receipt(response),
        "prompt_sha256": _sha(user),
        "requires_window_aggregation": bool(window_calls),
        "window_calls": window_calls,
        "window_count": len(window_calls) if window_calls else 1,
        "full_context_character_count": len(context),
        "aggregation_selected_notes": aggregation_notes,
        "aggregation_bounded_derivations": aggregation_derivations,
        "contextual_lexical_policy": lexical_policy,
    }
    diagnostic.pop("receipt_sha256", None)
    diagnostic["receipt_sha256"] = _sha(_canonical(diagnostic))
    return user, diagnostic


__all__ = [
    "VERSION", "CANONICAL_POLICY_FAMILIES", "ENGINEERED_EVALUATION_CELLS",
    "OPERATION_AXIS", "comparison_view", "unicode_receipt",
    "detected_policy_families", "full_coverage_windows",
    "build_window_adjudication_user", "build_aggregation_user",
    "build_contextual_policy_packet",
]
