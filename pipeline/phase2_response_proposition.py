"""Conservative proposition checks for Bengali answer responses.

Retrieval and arithmetic helpers must not treat the mere presence of an
expected string or numeral as an asserted answer.  This module keeps the
original Unicode/operator surfaces, identifies bounded answer spans, and
rejects negated, corrected, alternative, or numerically competing claims.

The functions are deliberately admission-oriented: ``True``/``supported``
means that a narrow deterministic check succeeded.  Every ambiguous form
remains nonterminal for a verifier instead of being converted into a label.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable


VERSION = "phase2-response-proposition-v1-negation-correction-aware"
BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
NUMERIC_TOKEN_PATTERN = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
TOKEN_RE = re.compile(
    rf"{NUMERIC_TOKEN_PATTERN}|[a-z]+|[\u0980-\u09ff]+|[+*/%^×÷=]|[,;:.!?।]",
    re.I,
)
NUMBER_TOKEN_RE = re.compile(NUMERIC_TOKEN_PATTERN, re.I)

NEGATION_TOKENS = {
    "না", "নয়", "নয়", "নেই", "নাহ", "নাই", "not", "never", "incorrect",
    "ভুল",
}
NEGATIVE_VERB_SUFFIXES = ("েনি", "ানি", "য়নি", "য়নি", "েননি", "াননি")
CONTRAST_TOKENS = {
    "কিন্তু", "তবে", "বরং", "আসলে", "অথচ", "পরিবর্তে", "however", "but",
    "rather", "actually", "instead",
}
ALTERNATIVE_TOKENS = {
    "অথবা", "কিংবা", "নাকি", "বা", "or", "alternatively",
}
ANSWER_SCAFFOLD = {
    "উত্তর", "সঠিক", "ঠিক", "হলো", "হল", "হচ্ছে", "হয়", "হয়", "টি", "the",
    "answer", "is", "correct",
}
CLAUSE_BREAKS = {";", ".", "!", "?", "।"}


def normalize_proposition(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text).translate(BN_DIGITS).casefold()
    text = text.replace("\u200c", "").replace("\u200d", "")
    return re.sub(r"\s+", " ", text).strip()


def proposition_tokens(value: object) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(normalize_proposition(value)))


def _is_negation(token: str) -> bool:
    return token in NEGATION_TOKENS or (
        len(token) >= 4 and token.endswith(NEGATIVE_VERB_SUFFIXES)
    )


def _find_spans(tokens: tuple[str, ...], needle: tuple[str, ...]) -> list[tuple[int, int]]:
    if not needle or len(needle) > len(tokens):
        return []
    return [
        (index, index + len(needle))
        for index in range(len(tokens) - len(needle) + 1)
        if all(
            observed == expected
            or observed in {expected + "ই", expected + "ও"}
            for observed, expected in zip(
                tokens[index : index + len(needle)], needle, strict=True,
            )
        )
    ]


def _clause_bounds(tokens: tuple[str, ...], start: int, end: int) -> tuple[int, int]:
    left = start
    while left > 0 and tokens[left - 1] not in CLAUSE_BREAKS:
        left -= 1
    right = end
    while right < len(tokens) and tokens[right] not in CLAUSE_BREAKS:
        right += 1
    return left, right


def _span_is_denied(tokens: tuple[str, ...], start: int, end: int) -> bool:
    left, right = _clause_bounds(tokens, start, end)
    # Bengali copular negation commonly follows a short predicate tail:
    # ``ঢাকা রাজধানী নয়``.  A small bidirectional window also covers
    # ``না, ঢাকা নয়`` without making a distant negation global.
    local_left = max(left, start - 3)
    local_right = min(right, end + 5)
    return any(_is_negation(token) for token in tokens[local_left:local_right])


def _meaningful_tail(tokens: tuple[str, ...], position: int) -> tuple[str, ...]:
    return tuple(
        token for token in tokens[position:]
        if token not in ANSWER_SCAFFOLD
        and token not in CONTRAST_TOKENS
        and token not in CLAUSE_BREAKS
        and token not in {",", ":", "="}
    )


@dataclass(frozen=True)
class AnswerStance:
    status: str
    matched_answer: str
    match_count: int
    denied_match_count: int
    correction_or_alternative_present: bool
    version: str = VERSION


def classify_answer_stance(response: object, answers: Iterable[object]) -> AnswerStance:
    """Classify whether an accepted answer is cleanly asserted.

    The classifier does not decide a hallucination label.  It only admits a
    source/answer relation when at least one complete answer span is asserted
    and no span-level negation, correction, or alternative makes that relation
    ambiguous.
    """

    tokens = proposition_tokens(response)
    candidates: list[tuple[str, tuple[str, ...]]] = []
    seen: set[tuple[str, ...]] = set()
    for value in answers:
        text = normalize_proposition(value)
        candidate_tokens = tuple(
            token for token in proposition_tokens(text)
            if token not in ANSWER_SCAFFOLD and token not in CLAUSE_BREAKS
        )
        if text and candidate_tokens and candidate_tokens not in seen:
            seen.add(candidate_tokens)
            candidates.append((text, candidate_tokens))
    candidates.sort(key=lambda item: len(item[1]), reverse=True)
    matches: list[tuple[str, int, int, bool]] = []
    for text, candidate_tokens in candidates:
        for start, end in _find_spans(tokens, candidate_tokens):
            matches.append((text, start, end, _span_is_denied(tokens, start, end)))
    if not matches:
        return AnswerStance("unmatched", "", 0, 0, False)

    denied = sum(item[3] for item in matches)
    correction_positions = [
        index for index, token in enumerate(tokens)
        if token in CONTRAST_TOKENS or token in ALTERNATIVE_TOKENS
    ]
    correction_ambiguous = False
    for _, start, end, is_denied in matches:
        del start
        if is_denied:
            continue
        for marker in correction_positions:
            if marker >= end and _meaningful_tail(tokens, marker + 1):
                correction_ambiguous = True
                break
        if correction_ambiguous:
            break

    if denied == len(matches):
        status = "denied"
    elif denied or correction_ambiguous:
        status = "ambiguous"
    else:
        status = "supported"
    return AnswerStance(
        status=status,
        matched_answer=matches[0][0],
        match_count=len(matches),
        denied_match_count=denied,
        correction_or_alternative_present=correction_ambiguous,
    )


def numeric_response_supports(
    response: object, expected: float, *, tolerance: float = 0.02,
) -> bool:
    """Admit one unambiguous affirmative numeric proposition."""

    tokens = proposition_tokens(response)
    numeric: list[tuple[int, float]] = []
    for index, token in enumerate(tokens):
        if NUMBER_TOKEN_RE.fullmatch(token):
            try:
                value = float(token.replace(",", ""))
            except ValueError:
                continue
            if math.isfinite(value):
                numeric.append((index, value))
    if not numeric:
        return False

    bound = max(float(tolerance), abs(float(expected)) * 1e-4)
    expected_spans = [
        (index, index + 1) for index, value in numeric
        if abs(value - float(expected)) <= bound
    ]
    if not expected_spans:
        return False
    if any(_span_is_denied(tokens, start, end) for start, end in expected_spans):
        return False

    # A second, materially different numeric answer makes the response
    # nonterminal.  This catches alternatives and corrections while allowing
    # harmless repeated rendering of the same value.
    if any(abs(value - float(expected)) > bound for _, value in numeric):
        return False
    if any(
        token in CONTRAST_TOKENS or token in ALTERNATIVE_TOKENS
        for token in tokens
    ):
        return False
    return True
