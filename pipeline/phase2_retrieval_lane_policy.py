"""Hash-bound separation of contextual and closed-book retrieval lanes.

The competition has two different label semantics.  A row that supplies a
context must be judged only from that context (including deterministic
derivations from rules, dates, quantities, units, and relationships stated in
it).  World knowledge or an offline corpus may never rescue such a row.  A
row without supplied context belongs to the separate closed-book lane, where
offline retrieval remains allowed.

Every contextual structured artifact embeds a :class:`RetrievalLaneContract`.
The policy and contract hashes make the boundary an executable runtime
invariant rather than a prompt convention.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable


VERSION = "phase2-retrieval-lane-policy-v1-context-isolation"
CONTEXTUAL_LANE = "context_only"
CLOSED_BOOK_LANE = "closed_book_retrieval"

POLICY = {
    "version": VERSION,
    "lane_selector": "context_available",
    "contextual": {
        "lane": CONTEXTUAL_LANE,
        "external_retrieval_allowed": False,
        "external_evidence_allowed": False,
        "world_knowledge_rescue_allowed": False,
        "internal_context_reasoning_allowed": True,
    },
    "closed_book": {
        "lane": CLOSED_BOOK_LANE,
        "external_retrieval_allowed": True,
        "external_evidence_allowed": True,
        "world_knowledge_rescue_allowed": True,
        "internal_context_reasoning_allowed": False,
    },
}


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


POLICY_SHA256 = _sha256_json(POLICY)


@dataclass(frozen=True)
class RetrievalLaneContract:
    context_available: bool
    lane: str
    external_retrieval_allowed: bool
    external_evidence_allowed: bool
    world_knowledge_rescue_allowed: bool
    internal_context_reasoning_allowed: bool
    policy_version: str = VERSION
    policy_sha256: str = POLICY_SHA256

    @property
    def contract_sha256(self) -> str:
        return _sha256_json(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "contract_sha256": self.contract_sha256}


def build_retrieval_lane_contract(
    context_available: bool,
) -> RetrievalLaneContract:
    if type(context_available) is not bool:
        raise ValueError("context_available must be an exact boolean")
    spec = POLICY["contextual" if context_available else "closed_book"]
    return RetrievalLaneContract(
        context_available=context_available,
        lane=str(spec["lane"]),
        external_retrieval_allowed=bool(spec["external_retrieval_allowed"]),
        external_evidence_allowed=bool(spec["external_evidence_allowed"]),
        world_knowledge_rescue_allowed=bool(
            spec["world_knowledge_rescue_allowed"]
        ),
        internal_context_reasoning_allowed=bool(
            spec["internal_context_reasoning_allowed"]
        ),
    )


def validate_retrieval_lane_contract(value: object) -> RetrievalLaneContract:
    if not isinstance(value, dict):
        raise ValueError("retrieval lane contract must be an object")
    raw = dict(value)
    observed_sha = str(raw.pop("contract_sha256", ""))
    expected = set(RetrievalLaneContract.__dataclass_fields__)
    if set(raw) != expected:
        raise ValueError("retrieval lane contract schema mismatch")
    for field in (
        "context_available", "external_retrieval_allowed",
        "external_evidence_allowed", "world_knowledge_rescue_allowed",
        "internal_context_reasoning_allowed",
    ):
        if type(raw.get(field)) is not bool:
            raise ValueError(f"retrieval lane contract {field} must be boolean")
    contract = RetrievalLaneContract(**raw)
    if observed_sha != contract.contract_sha256:
        raise ValueError("retrieval lane contract hash mismatch")
    expected_contract = build_retrieval_lane_contract(
        contract.context_available
    )
    if contract != expected_contract:
        raise ValueError("retrieval lane contract violates the pinned policy")
    return contract


def assert_contextual_no_external(
    lane_contract: RetrievalLaneContract,
    *,
    external_retrieval_admitted: object = False,
    query_plan: object = None,
    external_candidates: Iterable[object] = (),
    external_evidence: Iterable[object] = (),
) -> None:
    """Reject every outside-evidence surface for a supplied-context row.

    Closed-book inputs intentionally pass through unchanged; their independent
    retriever remains authorized by the closed-book lane contract.
    """

    expected = build_retrieval_lane_contract(lane_contract.context_available)
    if lane_contract != expected:
        raise ValueError("retrieval lane contract is not policy authoritative")
    if not lane_contract.context_available:
        return
    if external_retrieval_admitted is not False:
        raise ValueError("contextual external retrieval admission is forbidden")
    if query_plan is not None:
        raise ValueError("contextual query plans are forbidden")
    if any(True for _ in external_candidates):
        raise ValueError("contextual external retrieval candidates are forbidden")
    if any(True for _ in external_evidence):
        raise ValueError("contextual external evidence is forbidden")

