"""Hash-bound, semantic-alignment-first source authority policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "pipeline/resources/phase2_source_authority_v1.json"
EXPECTED_VERSION = "phase2-source-authority-policy-v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


POLICY_SHA256 = sha256_file(POLICY_PATH)
POLICY = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
if POLICY.get("version") != EXPECTED_VERSION:
    raise ValueError("source authority policy version mismatch")
if (POLICY.get("safety") or {}).get("semantic_alignment_precedes_authority") is not True:
    raise ValueError("unsafe source authority decision order")
if (POLICY.get("safety") or {}).get("wikipedia_terminal_authority") is not False:
    raise ValueError("Wikipedia must remain nonterminal")

_SOURCE_AUTHORITY: dict[str, dict[str, Any]] = {}
for tier in POLICY.get("tiers") or []:
    authority_tier = int(tier["authority_tier"])
    authority_class = str(tier["authority_class"])
    for source_id in tier.get("source_ids") or []:
        source_id = str(source_id)
        if source_id in _SOURCE_AUTHORITY:
            raise ValueError(f"duplicate source authority assignment: {source_id}")
        _SOURCE_AUTHORITY[source_id] = {
            "authority_tier": authority_tier,
            "authority_class": authority_class,
            "authority_policy_sha256": POLICY_SHA256,
            "wikipedia_corroboration_only": authority_class
            == "community_encyclopedia_corroboration",
        }


def authority_for(source_id: object) -> dict[str, Any]:
    key = str(source_id)
    if key in _SOURCE_AUTHORITY:
        return dict(_SOURCE_AUTHORITY[key])
    default = POLICY["default"]
    return {
        "authority_tier": int(default["authority_tier"]),
        "authority_class": str(default["authority_class"]),
        "authority_policy_sha256": POLICY_SHA256,
        "wikipedia_corroboration_only": False,
    }


def semantic_alignment(candidate: dict[str, Any]) -> tuple[int, str]:
    """Return a sortable alignment tier and an auditable status.

    Source authority is intentionally evaluated only after claim alignment.
    A bad book/OCR hit must not outrank a semantically aligned Wikipedia hit.
    """

    gate = candidate.get("model_facing_gate") or {}
    if candidate.get("model_facing_eligible") is not True or gate.get("eligible") is False:
        return 3, "quarantined"
    if candidate.get("number_set_match") is False:
        return 2, "number_mismatch"
    if candidate.get("negation_set_match") is False:
        return 2, "negation_mismatch"
    query_types = set(gate.get("query_target_types") or [])
    source_types = set(gate.get("source_target_types") or [])
    if query_types and source_types and query_types.isdisjoint(source_types):
        return 2, "answer_type_or_relation_mismatch"
    if candidate.get("exact_normalized") is True:
        return 0, "exact_aligned"
    return 1, "fuzzy_or_passage_aligned"


def annotate_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    value = dict(candidate)
    authority = authority_for(value.get("source_id", ""))
    alignment_tier, alignment_status = semantic_alignment(value)
    value.update(authority)
    value["semantic_alignment_tier"] = alignment_tier
    value["semantic_alignment_status"] = alignment_status
    if authority["wikipedia_corroboration_only"]:
        if value.get("source_verdict_candidate") is not None:
            raise ValueError("Wikipedia candidate attempted terminal source authority")
        value["terminal_label_authority"] = False
    return value


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    value = annotate_candidate(candidate)
    return (
        int(value["semantic_alignment_tier"]),
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
