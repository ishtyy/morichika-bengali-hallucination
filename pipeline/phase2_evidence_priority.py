"""Policy-compatible, label-free evidence ranking for the Phase 2 hybrid RAG.

The ranker never consumes competition labels.  It first enforces the lane and
the requested Bengali grammar/lexical operation, then reuses the independently
versioned source-authority policy.  This keeps a high-scoring synonym,
Wikipedia paragraph, or wrong grammar operation from displacing an attested
exact pair/rule merely because it is lexically similar to the query.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from pipeline.phase2_canonicalize import canonicalize
from pipeline.phase2_source_authority import annotate_candidate as annotate_authority


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "pipeline/resources/phase2_evidence_priority_v2.json"
POLICY_SHA256 = hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest()
POLICY = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
EXPECTED_VERSION = "phase2-evidence-priority-v2-exclusive-policy-first"
if POLICY.get("version") != EXPECTED_VERSION:
    raise ValueError("evidence priority policy version mismatch")
if (POLICY.get("safety") or {}).get("competition_test_labels_may_train_ranker") is not False:
    raise ValueError("competition labels must never train the evidence ranker")

EXCLUSIVE_OPERATIONS = frozenset(POLICY["exclusive_operations"])

_OPERATION_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("antonym_lookup", ("বিপরীত শব্দ", "বিপরীতার্থক শব্দ", "শুদ্ধ বিপরীত")),
    ("idiom_meaning_lookup", ("বাগধারা", "প্রবাদটির অর্থ", "প্রবাদ/বাগধারা")),
    ("prefix_origin_classification", ("উপসর্গ", "উপসর্গটি কোন শ্রেণির")),
    ("samas_taxonomy", ("সমাস", "সমাসের উদাহরণ")),
    ("sandhi_formation", ("সন্ধি", "সন্ধিতে")),
)

_DATE_MARKERS = re.compile(r"(?:তারিখ|সাল|বছর|কবে|মাস|দিন|খ্রিস্টাব্দ|সনে)")
_UNIT_MARKERS = re.compile(
    r"(?:কিলোমিটার|মিটার|সেন্টিমিটার|কেজি|কিলোগ্রাম|গ্রাম|সেকেন্ড|মিনিট|ঘণ্টা|"
    r"টাকা|শতাংশ|ডিগ্রি|নিউটন|জুল|ওয়াট|ওয়াট)"
)


def infer_operation(text: object) -> str:
    normalized = canonicalize(str(text or ""))
    for operation, markers in _OPERATION_MARKERS:
        if any(marker in normalized for marker in markers):
            return operation
    return ""


def _declared_operation(candidate: dict[str, Any]) -> str:
    for container in (
        candidate,
        candidate.get("model_facing_gate") or {},
        candidate.get("source_metadata") or {},
    ):
        for key in ("policy_operation", "operation", "query_operation"):
            value = str(container.get(key) or "")
            if value in EXCLUSIVE_OPERATIONS:
                return value
    return ""


def _query_operation(candidate: dict[str, Any]) -> str:
    declared = str(candidate.get("query_policy_operation") or "")
    if declared in EXCLUSIVE_OPERATIONS:
        return declared
    return infer_operation(candidate.get("query_text", ""))


def _source_operation(candidate: dict[str, Any]) -> str:
    declared = _declared_operation(candidate)
    if declared:
        return declared
    return infer_operation("\n".join((
        str(candidate.get("question") or ""),
        str(candidate.get("supporting_text") or ""),
    )))


def policy_compatibility(candidate: dict[str, Any]) -> tuple[int, str]:
    """Return a hard-before-authority compatibility tier and audit reason."""

    if candidate.get("query_context_available") is True:
        return 4, "contextual_external_retrieval_forbidden"
    query_operation = _query_operation(candidate)
    source_operation = _source_operation(candidate)
    if query_operation:
        if source_operation == query_operation:
            return 0, f"exclusive_operation_exact:{query_operation}"
        if source_operation:
            return 4, f"exclusive_operation_mismatch:{query_operation}!={source_operation}"
        # An untyped passage may still contain a useful attested pair/rule, but
        # it cannot outrank an operation-exact source or become terminal proof.
        return 2, f"exclusive_operation_untyped_source:{query_operation}"
    if source_operation:
        return 2, f"unsolicited_exclusive_operation:{source_operation}"
    return 1, "general_factual_policy"


def slot_compatibility(candidate: dict[str, Any]) -> tuple[int, str]:
    """Gate the factual slot before source reputation or retrieval score."""

    gate = candidate.get("model_facing_gate") or {}
    if candidate.get("model_facing_eligible") is not True or gate.get("eligible") is False:
        return 4, "upstream_quarantine"
    # A source-side item/page number (for example ``02`` in an OCR MCQ page)
    # is not a factual-slot conflict when the query itself requests no number.
    # Once the query contains a number/date, however, exact compatibility is a
    # hard prerequisite and a mismatch is quarantined.
    if (
        candidate.get("number_set_match") is False
        and (
            "query_numbers" not in candidate
            or bool(candidate.get("query_numbers"))
        )
    ):
        return 3, "number_mismatch"
    if candidate.get("negation_set_match") is False:
        return 3, "negation_mismatch"
    query_types = set(gate.get("query_target_types") or [])
    source_types = set(gate.get("source_target_types") or [])
    if query_types and source_types and query_types.isdisjoint(source_types):
        return 3, "answer_type_or_relation_mismatch"
    query = canonicalize(str(candidate.get("query_text") or ""))
    source = canonicalize("\n".join((
        str(candidate.get("question") or ""),
        str(candidate.get("supporting_text") or ""),
    )))
    if _DATE_MARKERS.search(query) and not _DATE_MARKERS.search(source):
        return 2, "date_slot_unconfirmed"
    query_units, source_units = set(_UNIT_MARKERS.findall(query)), set(_UNIT_MARKERS.findall(source))
    if query_units and source_units and query_units.isdisjoint(source_units):
        return 3, "unit_mismatch"
    return 0, "slot_compatible"


def annotate_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    value = annotate_authority(candidate)
    policy_tier, policy_status = policy_compatibility(value)
    slot_tier, slot_status = slot_compatibility(value)
    value["policy_compatibility_tier"] = policy_tier
    value["policy_compatibility_status"] = policy_status
    value["slot_compatibility_tier"] = slot_tier
    value["slot_compatibility_status"] = slot_status
    value["query_policy_operation"] = _query_operation(value)
    value["source_policy_operation"] = _source_operation(value)
    value["evidence_priority_policy_sha256"] = POLICY_SHA256
    if policy_tier >= 4 or slot_tier >= 3:
        value["model_facing_eligible"] = False
        gate = dict(value.get("model_facing_gate") or {})
        gate["eligible"] = False
        reasons = set(gate.get("reasons") or [])
        reasons.update((policy_status, slot_status))
        gate["reasons"] = sorted(reasons)
        value["model_facing_gate"] = gate
        value["source_verdict_candidate"] = None
        value["terminal_label_authority"] = False
    elif policy_tier == 2:
        # Useful for model inspection, never sufficient for exact lexical or
        # grammar proof when the source operation itself is untyped.
        value["source_verdict_candidate"] = None
        value["terminal_label_authority"] = False
    return value


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    value = annotate_candidate(candidate)
    relation = str(value.get("evidence_role") or value.get("response_answer_relation") or "none")
    # Support and counter are deliberately tied.  A response-aware ranker must
    # not favor agreement over contradiction; the bounded selector reserves
    # both roles independently.
    relation_tier = 0 if relation in {
        "support_candidate", "counter_candidate", "exact", "containment"
    } else 1
    return (
        int(value["policy_compatibility_tier"]),
        int(value["slot_compatibility_tier"]),
        relation_tier,
        int(value["authority_tier"]),
        0 if value.get("exact_normalized") is True else 1,
        int(value.get("rank", 10**9)),
        str(value.get("source_id", "")),
        int(value.get("source_record_index", -1)),
        str(value.get("source_record_sha256", "")),
    )


def annotate_and_rank(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated = [annotate_candidate(candidate) for candidate in candidates]
    return sorted(annotated, key=candidate_sort_key)
