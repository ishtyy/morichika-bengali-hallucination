"""Deterministic context-only evidence router for long contexts.

This module is deliberately extractive.  It does not browse, retrieve, infer a
gold label, or replace the supplied context with a summary.  It provides:

* full-coverage character windows with a literal residual overlap;
* byte-replayable, question-conditioned notes;
* visible, source-linked date arithmetic and bounded kinship derivations; and
* a fail-closed decision receipt tied to the exact requested slot.

The lightweight extractor and ranker are a deterministic primary front end for
long or structurally complex contextual rows, not a claim of complete Bengali
semantic parsing.  Missing extraction yields NEI or the ordinary full-window
model path; it never licenses outside knowledge.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from typing import Any, Iterable


VERSION = "bichar-contextual-evidence-router-v2"
NOTE_TYPES = {
    "entity", "date_or_time", "event", "relation", "quantity", "unit",
    "negation", "causal", "temporal", "rule_or_definition", "operand",
    "conflict_marker",
}
VERDICTS = {"supported", "refuted", "not_enough_information"}
OPERATORS = {"simple_year_difference", "relative_year_offset", "bounded_fathers_father"}
FORBIDDEN_KEY_PARTS = ("retriev", "web_search", "external_evidence", "outside_source", "corpus_lookup")

_BN_TO_ASCII = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_DATE_RE = re.compile(
    r"(?:[১২]?[০-৯]{1,3}|[12]?[0-9]{1,3})\s*(?:সাল(?:ে)?|খ্রিষ্টাব্দ|খ্রি\.?|বৈশাখ|জ্যৈষ্ঠ|আষাঢ়|শ্রাবণ|ভাদ্র|আশ্বিন|কার্তিক|অগ্রহায়ণ|পৌষ|মাঘ|ফাল্গুন|চৈত্র)"
)
_NUMBER_RE = re.compile(r"[+-]?[০-৯0-9]+(?:[.,][০-৯0-9]+)?")
_UNIT_RE = re.compile(r"(?:কিলোমিটার|মিটার|সেন্টিমিটার|কিলোগ্রাম|গ্রাম|লিটার|সেকেন্ড|মিনিট|ঘণ্টা|দিন|বছর|শতাংশ|টাকা|জন|টি|বার)")
_NEGATION_RE = re.compile(
    r"(?:দেওয়া হয়নি|হয়নি|করেনি|অসম্ভব|(?<![\u0980-\u09FF])(?:না|নয়|নেই|নি)(?![\u0980-\u09FF]))"
)
_CAUSAL_RE = re.compile(r"(?:কারণ|ফলে|তাই|সুতরাং|যেহেতু|এজন্য)")
_TEMPORAL_RE = re.compile(r"(?:আগে|পরে|পরবর্তীতে|প্রথমে|তারপর|শুরু|শেষ|জন্ম|বিবাহ|ঘোষণা|সমাপ্তি)")
_RELATIVE_YEAR_RE = re.compile(
    r"(?<![০-৯0-9,])(?P<offset>[০-৯0-9]{1,3})(?![০-৯0-9,])\s*(?:বছর|বর্ষ|years?)\s*"
    r"(?P<direction>পরে|পর|আগে|পূর্বে|later|after|before)", re.IGNORECASE
)
_GRAMMAR_RE = re.compile(r"(?:সমাস|সন্ধি|বাগধারা|বিপরীত শব্দ|উপসর্গ|প্রত্যয়|গুরুচণ্ডালী|ণত্ব|ষত্ব|ব্যাকরণ|অর্থ|নিয়ম)")
_EVENT_RE = re.compile(r"(?:ঘটে|ঘটেছিল|হয়|হয়েছিল|শুরু হয়|শেষ হয়|জন্মগ্রহণ|বিবাহ|ঘোষিত|অনুমোদিত)")
_RELATION_RE = re.compile(
    r"(?P<subject>[\u0980-\u09FF]+(?:\s+[\u0980-\u09FF]+){0,3}?)ের\s+"
    r"(?P<relation>বাবা|পিতা|মা|মাতা|দাদা|পিতামহ|স্রষ্টা|নির্মাতা|পরিচালক|জন্মস্থান|রাজধানী)\s+"
    r"(?P<object>[\u0980-\u09FF][\u0980-\u09FF\s]{0,40}?)(?=[।.!?\n]|$)"
)
_SENTENCE_RE = re.compile(r"[^।.!?\n]+[।.!?]?|\n")
_TOKEN_RE = re.compile(r"[\u0980-\u09FF]+|[A-Za-z]+|[০-৯0-9]+")
_QUESTION_STOPWORDS = {
    "কি", "কী", "কে", "কার", "কোন", "কোনটি", "কত", "কবে", "কোথায়",
    "কোথায়", "কেন", "কিভাবে", "কীভাবে", "হয়", "হয়", "ছিল", "আছে",
    "এটি", "এই", "একটি", "উত্তর", "বলুন", "লিখুন", "what", "which",
    "who", "when", "where", "why", "how", "is", "was", "the", "a",
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Window:
    window_index: int
    context_char_start: int
    context_char_end: int
    residual_from_previous_start: int | None
    residual_from_previous_end: int | None
    literal_text: str
    literal_span_sha256: str


@dataclass(frozen=True)
class Note:
    note_id: str
    type: str
    literal_text: str
    context_char_start: int
    context_char_end: int
    literal_span_sha256: str
    normalized_comparison_form: str
    window_indices: tuple[int, ...]
    question_slot_ids: tuple[str, ...]
    attributes: dict[str, Any]


def _fail_if_forbidden_keys(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            folded = str(key).lower().replace("-", "_")
            if folded == "external_retrieval_allowed" and child is False:
                pass
            elif any(token in folded for token in FORBIDDEN_KEY_PARTS):
                raise ValueError(f"contextual lane forbids external/retrieval field {path}.{key}")
            _fail_if_forbidden_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            _fail_if_forbidden_keys(child, f"{path}[{i}]")


def make_windows(context: str, max_chars: int = 640, overlap_chars: int = 96) -> list[Window]:
    """Create full-coverage windows with a literal, replayable residual tail."""
    if not context:
        raise ValueError("context must be non-empty")
    if max_chars < 64 or overlap_chars < 1 or overlap_chars >= max_chars:
        raise ValueError("require max_chars >= 64 and 0 < overlap_chars < max_chars")
    windows: list[Window] = []
    start = 0
    while start < len(context):
        hard_end = min(start + max_chars, len(context))
        end = hard_end
        if hard_end < len(context):
            candidates = [context.rfind(mark, start + max_chars // 2, hard_end) for mark in ("\n", "।", ".", "!", "?")]
            boundary = max(candidates)
            if boundary >= 0:
                end = boundary + 1
        if end <= start:
            end = hard_end
        residual_start = start if windows else None
        residual_end = min(windows[-1].context_char_end, end) if windows else None
        literal = context[start:end]
        windows.append(Window(
            window_index=len(windows), context_char_start=start, context_char_end=end,
            residual_from_previous_start=residual_start,
            residual_from_previous_end=residual_end,
            literal_text=literal, literal_span_sha256=_sha(literal),
        ))
        if end == len(context):
            break
        start = max(0, end - overlap_chars)
    validate_windows(context, windows, overlap_chars)
    return windows


def validate_windows(context: str, windows: Iterable[Window], required_overlap: int) -> None:
    windows = list(windows)
    if not windows or windows[0].context_char_start != 0 or windows[-1].context_char_end != len(context):
        raise ValueError("windows do not cover both context endpoints")
    covered_until = 0
    for i, w in enumerate(windows):
        if context[w.context_char_start:w.context_char_end] != w.literal_text or _sha(w.literal_text) != w.literal_span_sha256:
            raise ValueError(f"window {i} does not replay against supplied context")
        if w.context_char_start > covered_until:
            raise ValueError(f"coverage gap before window {i}")
        if i:
            overlap = windows[i - 1].context_char_end - w.context_char_start
            expected = min(required_overlap, windows[i - 1].context_char_end)
            if overlap < expected:
                raise ValueError(f"residual overlap too small at window {i}: {overlap} < {expected}")
        covered_until = max(covered_until, w.context_char_end)


def _window_indices(start: int, end: int, windows: list[Window]) -> tuple[int, ...]:
    return tuple(w.window_index for w in windows if start < w.context_char_end and end > w.context_char_start)


def make_note(
    context: str,
    windows: list[Window],
    note_type: str,
    start: int,
    end: int,
    slot_ids: Iterable[str],
    attributes: dict[str, Any] | None = None,
) -> Note:
    if note_type not in NOTE_TYPES or not (0 <= start < end <= len(context)):
        raise ValueError("invalid note type or span")
    literal = context[start:end]
    slots = tuple(sorted(set(slot_ids)))
    if not slots:
        raise ValueError("every note must constrain at least one question slot")
    attrs = dict(attributes or {})
    _fail_if_forbidden_keys(attrs)
    payload = {"type": note_type, "start": start, "end": end, "literal": literal, "slots": slots, "attributes": attrs}
    return Note(
        note_id="note:" + _sha(_canonical(payload))[:24], type=note_type,
        literal_text=literal, context_char_start=start, context_char_end=end,
        literal_span_sha256=_sha(literal), normalized_comparison_form=" ".join(literal.casefold().split()),
        window_indices=_window_indices(start, end, windows), question_slot_ids=slots, attributes=attrs,
    )


def extract_question_conditioned_notes(context: str, question: str, windows: list[Window]) -> list[Note]:
    """Extract a conservative literal fact index; no outside knowledge is used."""
    _fail_if_forbidden_keys({"context": context, "question": question})
    slot = "requested_answer"
    proposals: list[tuple[str, int, int, dict[str, Any]]] = []
    patterns = (
        ("date_or_time", _DATE_RE), ("quantity", _NUMBER_RE), ("unit", _UNIT_RE),
        ("negation", _NEGATION_RE), ("causal", _CAUSAL_RE), ("temporal", _TEMPORAL_RE),
    )
    for note_type, pattern in patterns:
        for match in pattern.finditer(context):
            proposals.append((note_type, match.start(), match.end(), {}))
    for match in _RELATIVE_YEAR_RE.finditer(context):
        proposals.append(("temporal", match.start(), match.end(), {
            "offset_years": int(match.group("offset").translate(_BN_TO_ASCII)),
            "direction": match.group("direction").casefold(),
        }))
    for match in _RELATION_RE.finditer(context):
        proposals.append(("relation", match.start(), match.end(), {
            "subject": match.group("subject").strip(), "relation": match.group("relation").strip(),
            "object": match.group("object").strip(),
        }))
        for group in ("subject", "object"):
            proposals.append(("entity", match.start(group), match.end(group), {"relation_role": group}))
    for sentence in _SENTENCE_RE.finditer(context):
        text = sentence.group()
        if _EVENT_RE.search(text):
            proposals.append(("event", sentence.start(), sentence.end(), {}))
        if _GRAMMAR_RE.search(text):
            proposals.append(("rule_or_definition", sentence.start(), sentence.end(), {}))
    notes: dict[tuple[str, int, int], Note] = {}
    for note_type, start, end, attrs in proposals:
        key = (note_type, start, end)
        notes[key] = make_note(context, windows, note_type, start, end, [slot], attrs)
    return sorted(notes.values(), key=lambda n: (n.context_char_start, n.context_char_end, n.type))


def _content_tokens(value: str) -> set[str]:
    return {
        token.casefold() for token in _TOKEN_RE.findall(value)
        if len(token) > 1 and token.casefold() not in _QUESTION_STOPWORDS
    }


def rank_context_windows(
    context: str,
    question: str,
    response: str,
    windows: list[Window] | None = None,
    *,
    max_windows: int = 6,
) -> dict[str, Any]:
    """Rank supplied-context windows without admitting outside evidence.

    Question overlap dominates response overlap so a fluent proposed answer
    cannot pull the router toward an unrelated same-passage fact.  Slot cues,
    numbers, dates, negation, rules and explicit relations add bounded bonuses.
    The selected set reserves neighboring windows and both response-aligned and
    response-conflicting spans when they exist.  Every selected byte remains
    replayable against the original context.
    """

    if max_windows < 1:
        raise ValueError("max_windows must be positive")
    windows = windows or make_windows(context)
    validate_windows(context, windows, required_overlap=1)
    q_tokens = _content_tokens(question)
    r_tokens = _content_tokens(response)
    question_numbers = set(_NUMBER_RE.findall(question))
    response_numbers = set(_NUMBER_RE.findall(response))
    query_needs_date = bool(_DATE_RE.search(question) or re.search(r"(?:কবে|তারিখ|সাল|বছর|বয়স|বয়স)", question))
    query_needs_relation = bool(re.search(r"(?:বাবা|পিতা|মা|মাতা|দাদা|পিতামহ|স্রষ্টা|নির্মাতা|পরিচালক|জন্মস্থান|রাজধানী|কার|কে)", question))
    query_needs_rule = bool(_GRAMMAR_RE.search(question))
    query_has_negation = bool(_NEGATION_RE.search(question))

    scored: list[dict[str, Any]] = []
    for window in windows:
        text = window.literal_text
        tokens = _content_tokens(text)
        q_overlap = sorted(q_tokens & tokens)
        r_overlap = sorted(r_tokens & tokens)
        number_tokens = set(_NUMBER_RE.findall(text))
        slot_bonus = 0
        slot_reasons: list[str] = []
        if query_needs_date and (_DATE_RE.search(text) or number_tokens):
            slot_bonus += 5
            slot_reasons.append("date_or_time")
        if query_needs_relation and _RELATION_RE.search(text):
            slot_bonus += 5
            slot_reasons.append("relation")
        if query_needs_rule and _GRAMMAR_RE.search(text):
            slot_bonus += 5
            slot_reasons.append("rule_or_definition")
        if query_has_negation and _NEGATION_RE.search(text):
            slot_bonus += 4
            slot_reasons.append("negation")
        if question_numbers & number_tokens:
            slot_bonus += 6
            slot_reasons.append("question_number")
        response_number_match = bool(response_numbers & number_tokens)
        # A window can legitimately contain both the proposed value and a
        # competing value.  Preserve both roles instead of collapsing it to
        # support merely because one response number matched.
        response_number_conflict = bool(response_numbers and (number_tokens - response_numbers))
        if response_number_match:
            slot_reasons.append("response_number_support_candidate")
        if response_number_conflict:
            slot_reasons.append("response_number_counter_candidate")
        score = 8 * len(q_overlap) + 2 * len(r_overlap) + slot_bonus
        scored.append({
            "window_index": window.window_index,
            "score": score,
            "question_overlap": q_overlap,
            "response_overlap": r_overlap,
            "slot_reasons": slot_reasons,
            "response_number_support_candidate": response_number_match,
            "response_number_counter_candidate": response_number_conflict,
        })

    ranked = sorted(scored, key=lambda row: (-int(row["score"]), int(row["window_index"])))
    chosen: set[int] = set()
    for row in ranked:
        if row["score"] <= 0 and chosen:
            break
        index = int(row["window_index"])
        chosen.add(index)
        # A single adjacent window preserves local antecedents and phase order.
        for neighbor in (index - 1, index + 1):
            if 0 <= neighbor < len(windows) and len(chosen) < max_windows:
                chosen.add(neighbor)
        if len(chosen) >= max_windows:
            break
    for role in ("response_number_support_candidate", "response_number_counter_candidate"):
        candidate = next((row for row in ranked if row[role]), None)
        if candidate is not None and len(chosen) < max_windows:
            chosen.add(int(candidate["window_index"]))
    if not chosen:
        chosen.add(0)
    selected = [windows[index] for index in sorted(chosen)[:max_windows]]
    selected_indices = {window.window_index for window in selected}
    notes = extract_question_conditioned_notes(context, question, windows)
    selected_notes = [
        note for note in notes if selected_indices.intersection(note.window_indices)
    ][:32]
    derivations: list[dict[str, Any]] = []
    dates = [note for note in notes if note.type == "date_or_time"]
    relatives = [
        note for note in notes
        if note.type == "temporal" and "offset_years" in note.attributes
    ]
    for relative in relatives:
        containing_sentence = next(
            (match for match in _SENTENCE_RE.finditer(context)
             if match.start() <= relative.context_char_start < match.end()),
            None,
        )
        # Only accept the unambiguous discourse form "N years later, EVENT"
        # near the beginning of a question-relevant target sentence.  Mid-
        # sentence durations such as "after 700 years the empire ended" do
        # not identify the omitted origin year and must remain nonterminal.
        if containing_sentence is None:
            continue
        relative_offset = relative.context_char_start - containing_sentence.start()
        sentence_tokens = _content_tokens(containing_sentence.group())
        if relative_offset > 6 or not (q_tokens & sentence_tokens):
            continue
        anchors = [date for date in dates if date.context_char_end <= relative.context_char_start]
        if anchors:
            derivations.append(relative_year_offset(anchors[-1], relative))
    explicit_age = re.search(r"[০-৯0-9]+\s*বছর\s*(?:বয়সে|বয়সে)", context)
    birth_marriage = re.search(r"(?:জন্ম|born)", context, re.IGNORECASE) and re.search(
        r"(?:বিবাহ|বিয়ে|বিয়ে|marri)", context, re.IGNORECASE
    )
    if (
        re.search(r"(?:বয়স|বয়স|কত বছর বয়সে|কত বছর বয়সে|age)", question, re.IGNORECASE)
        and birth_marriage and not explicit_age and len(dates) == 2
    ):
        derivations.append(simple_year_difference(context, windows, dates[0], dates[1]))
    receipt: dict[str, Any] = {
        "version": VERSION,
        "lane": "contextual_only",
        "routing_mode": "primary_for_long_or_complex_context",
        "external_retrieval_allowed": False,
        "context_sha256": _sha(context),
        "question_sha256": _sha(question),
        "response_sha256": _sha(response),
        "selected_windows": [asdict(window) for window in selected],
        "selected_notes": [asdict(note) for note in selected_notes],
        "bounded_derivation_candidates": derivations[:8],
        "ranking": ranked,
        "full_context_coverage_preserved_by_window_ledger": True,
    }
    receipt["receipt_sha256"] = _sha(_canonical(receipt))
    return receipt


def simple_year_difference(context: str, windows: list[Window], earlier: Note, later: Note) -> dict[str, Any]:
    if earlier.type != "date_or_time" or later.type != "date_or_time":
        raise ValueError("year difference requires two date notes")
    a = int(_NUMBER_RE.search(earlier.literal_text).group().translate(_BN_TO_ASCII))
    b = int(_NUMBER_RE.search(later.literal_text).group().translate(_BN_TO_ASCII))
    result = b - a
    return {
        "derivation_id": "derive:" + _sha(f"simple_year_difference|{earlier.note_id}|{later.note_id}|{result}")[:24],
        "operator_id": "simple_year_difference", "operand_note_ids": [earlier.note_id, later.note_id],
        "visible_expression": f"{b} - {a} = {result}", "result": {"type": "number", "value": result, "unit": "বছর"},
        "source_linked": True,
    }


def relative_year_offset(anchor: Note, relative: Note) -> dict[str, Any]:
    """Resolve an explicitly stated bounded year offset from a cited year."""
    if anchor.type != "date_or_time" or relative.type != "temporal":
        raise ValueError("relative year offset requires a date and temporal note")
    year_match = _NUMBER_RE.search(anchor.literal_text)
    offset = relative.attributes.get("offset_years")
    direction = str(relative.attributes.get("direction", "")).casefold()
    if year_match is None or not isinstance(offset, int) or offset < 0:
        raise ValueError("relative year offset operands are incomplete")
    year = int(year_match.group().translate(_BN_TO_ASCII))
    sign = -1 if direction in {"আগে", "পূর্বে", "before"} else 1
    result = year + sign * offset
    operator = "-" if sign < 0 else "+"
    return {
        "derivation_id": "derive:" + _sha(
            f"relative_year_offset|{anchor.note_id}|{relative.note_id}|{result}"
        )[:24],
        "operator_id": "relative_year_offset",
        "operand_note_ids": [anchor.note_id, relative.note_id],
        "visible_expression": f"{year} {operator} {offset} = {result}",
        "result": {"type": "year", "value": result, "unit": "সাল"},
        "source_linked": True,
    }


def bounded_fathers_father(first: Note, second: Note) -> dict[str, Any]:
    if first.type != "relation" or second.type != "relation":
        raise ValueError("kinship composition requires relation notes")
    a, b = first.attributes, second.attributes
    father_words = {"বাবা", "পিতা"}
    if a.get("relation") not in father_words or b.get("relation") not in father_words or a.get("object") != b.get("subject"):
        raise ValueError("relation edges do not form a bounded father's-father chain")
    result = b["object"]
    return {
        "derivation_id": "derive:" + _sha(f"bounded_fathers_father|{first.note_id}|{second.note_id}|{result}")[:24],
        "operator_id": "bounded_fathers_father", "operand_note_ids": [first.note_id, second.note_id],
        "visible_expression": f"{a['subject']}→পিতা→{a['object']}→পিতা→{result}",
        "result": {"type": "entity", "value": result, "unit": ""}, "source_linked": True,
    }


def decision_receipt(
    context: str,
    question: str,
    response: str,
    requested_relation_or_property: str,
    verdict: str,
    evidence_notes: Iterable[Note],
    derivation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if verdict not in VERDICTS or not requested_relation_or_property.strip():
        raise ValueError("invalid verdict or requested slot")
    notes = list(evidence_notes)
    if verdict != "not_enough_information" and not notes:
        raise ValueError("supported/refuted decisions require context evidence")
    if derivation:
        if derivation.get("operator_id") not in OPERATORS or not derivation.get("source_linked"):
            raise ValueError("unapproved or ungrounded derivation")
        available = {n.note_id for n in notes}
        if not set(derivation.get("operand_note_ids", [])).issubset(available):
            raise ValueError("derivation operands must be cited evidence notes")
    receipt = {
        "version": VERSION, "lane": "contextual_only", "external_retrieval_allowed": False,
        "context_sha256": _sha(context), "question": question, "question_sha256": _sha(question),
        "response": response, "response_sha256": _sha(response),
        "requested_relation_or_property": requested_relation_or_property,
        "verdict": verdict, "evidence_notes": [asdict(n) for n in notes], "derivation": derivation,
        "question_text_is_evidence": False,
    }
    _fail_if_forbidden_keys(receipt)
    receipt["receipt_sha256"] = _sha(_canonical(receipt))
    return receipt
