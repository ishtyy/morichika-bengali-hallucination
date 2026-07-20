"""Offline, hash-gated sparse retrieval for arbitrary Phase 2 inputs.

The stage consumes only label-free formatted JSONL.  It produces evidence and
conservative source-verdict *candidates*; fuzzy retrieval never becomes a
terminal label.  Existing Bangla-MMLU and NCTB-QA indexes are reused rather than
rebuilt inside the Kaggle kernel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.phase2_canonicalize import (  # noqa: E402
    VERSION as CANONICALIZER_VERSION,
    canonicalize,
)
from pipeline.phase2_mmap_retrieval import (  # noqa: E402
    NORMALIZER as MMAP_NORMALIZER,
    VERSION as MMAP_CACHE_VERSION,
    close_mmap_index,
    load_mmap_index,
)
from pipeline.phase2_composite_fts_retrieval import (  # noqa: E402
    VERSION as COMPOSITE_FTS_RUNTIME_VERSION,
    load as load_composite_fts,
    retrieve_authority_tier_pool_optimized,
)
from pipeline.phase2_response_proposition import classify_answer_stance  # noqa: E402
from pipeline.phase2_evidence_priority import (  # noqa: E402
    POLICY_PATH as SOURCE_AUTHORITY_POLICY_PATH,
    POLICY_SHA256 as SOURCE_AUTHORITY_POLICY_SHA256,
    annotate_and_rank as authority_rank_candidates,
)
from pipeline.phase2_retrieval_lane_policy import (  # noqa: E402
    POLICY_SHA256 as RETRIEVAL_LANE_POLICY_SHA256,
)


VERSION = "phase2-sparse-retrieval-v21-exclusive-policy-first-evidence-ranking"
STRICT_EXACT_CONFLICT_POLICY = "same_choice_set_answer_disagreement_quarantines"
COMPOSITE_QUERY_MODES = {"all_closed", "unresolved_only", "residual_only"}

# Bengali count answers are commonly written both joined and separated
# (``১৯টি`` / ``১৯ টি``).  The general canonicalizer deliberately preserves
# token boundaries, so normalize only the narrow numeral+counter boundary here.
# This must not become a global whitespace-strip: doing that would silently
# merge names, dates, units, or multi-token alternatives.
NUMERIC_COUNTER_SPACE_RE = re.compile(
    r"(?<=[0-9০-৯])\s+(?=(?:টি|টা|জন|খানা|খানি)(?:\s|$))"
)


def preferred_cache(directory: Path) -> Path:
    """Prefer a cache built for the current runtime/canonicalizer identity.

    ``_mmap_v2`` is checked first.  A historically named ``_mmap_v1`` shard is
    accepted only if its *manifest* declares the current cache and
    canonicalizer identity; the directory suffix is never treated as proof of
    compatibility.  Legacy joblib/NPZ indexes remain the fail-closed fallback.
    """

    # This runs while module defaults are initialized, before the streaming
    # ``sha256_file`` helper below is defined.  The canonicalizer module is
    # small, so a direct digest is both safe and deterministic here.
    normalizer_sha256 = hashlib.sha256(MMAP_NORMALIZER.read_bytes()).hexdigest()
    for suffix in ("_mmap_v2", "_mmap_v1"):
        candidate = directory.with_name(directory.name + suffix)
        manifest_path = candidate / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError, TypeError):
            continue
        fingerprint = manifest.get("fingerprint") or {}
        if (
            manifest.get("version") == MMAP_CACHE_VERSION
            and fingerprint.get("canonicalizer_version") == CANONICALIZER_VERSION
            and fingerprint.get("normalizer_sha256") == normalizer_sha256
        ):
            return candidate
    return directory


DEFAULT_INDEX_SPECS = [
    {
        "source_id": "bangla_mmlu",
        "directory": preferred_cache(ROOT / "artifacts" / "retrieval" / "bangla_mmlu_char_v1"),
        "rights_note": "dataset card license was not declared; local evidence use only until redistribution review",
        "terminal_policy": "exact question + duplicate keyed-answer consensus + response exact keyed option",
    },
    {
        "source_id": "nctb_qa_87805",
        "directory": preferred_cache(ROOT / "artifacts" / "retrieval" / "nctb_qa_87805_char_v1"),
        "rights_note": "CC BY-NC 4.0 snapshot; preserve attribution and noncommercial restriction",
        "terminal_policy": "exact question and positive answer alignment only; mismatch never implies false",
    },
    {
        "source_id": "nctb_education_aux",
        "directory": preferred_cache(ROOT / "artifacts" / "retrieval" / "nctb_education_aux_char_v1"),
        "rights_note": "mixed provenance: NCTBench CC BY 4.0 and SOMAJGYAAN MIT may be terminal under exact policy; BEnQA/SSC-BanglaTutor remain local corroboration only",
        "terminal_policy": "record-level exact consensus policy; fuzzy retrieval and rights-unclear rows are never terminal",
    },
    {
        "source_id": "downloads_bcs_10_50",
        "directory": preferred_cache(ROOT / "artifacts" / "retrieval" / "downloads_bcs_10_50_char_v1"),
        "rights_note": "user-supplied local BCS PDFs with PDF/page/raw-block provenance",
        "terminal_policy": "model-facing exact/fuzzy evidence only; OCR-derived answers never auto-terminal",
    },
    {
        "source_id": "nctb_schooltext",
        "directory": preferred_cache(ROOT / "artifacts" / "retrieval" / "nctb_schooltext_word_v1"),
        "rights_note": "CC BY 4.0 processed corpus; preserve attribution and embedded NCTB source-rights caveat",
        "terminal_policy": "model-facing passage retrieval only; no answer labels",
    },
    {
        "source_id": "joykoli_six_part",
        "directory": preferred_cache(ROOT / "artifacts" / "retrieval" / "joykoli_six_part_char_v3"),
        "rights_note": "user-provided private competition-use authorization; derived cache only; no public redistribution",
        "terminal_policy": "closed-book corroboration only; never promotes a verdict",
        "exact_conflict_policy": STRICT_EXACT_CONFLICT_POLICY,
    },
]
FORBIDDEN_FIELDS = {"gold", "gold_label", "label", "target", "is_hallucination"}
NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)")
NEGATIONS = ("না", "নয়", "নয়", "নেই", "ব্যতীত", "ছাড়া", "ছাড়া", "ভুল")
FUZZY_STOPWORDS = {
    "কি", "কী", "কে", "কার", "কোন", "কোনটি", "কত", "কবে", "কোথায়", "কোথায়",
    "নিচের", "নিম্নের", "সঠিক", "উত্তর", "হয়", "হয়", "ছিল", "আছে", "এটি",
    "এই", "একটি", "the", "a", "an", "is", "was", "what", "which", "who",
}

# These are interrogative/relation/type words, not a list of named entities.
# The first remaining ordered token is used as a conservative subject anchor
# for noisy, non-exact Joykoli evidence.  This also covers concepts such as
# ``নবায়নযোগ্য``: an anchor need not be a person's or place's name.
ANCHOR_GENERIC_TERMS = {
    "প্রথম", "শেষ", "প্রধান", "প্রাচীন", "বর্তমান", "বৃহত্তম", "ক্ষুদ্রতম",
    "সর্বপ্রথম", "সর্বশেষ", "নাম", "নামটি", "পরিচিত", "অন্তর্ভুক্ত", "অবস্থিত",
    "ছিল", "হয়", "হয়", "আছে", "রয়েছে", "রয়েছে", "করে", "করেন", "হিসেবে",
    "উদাহরণ", "সংকেত", "সম্পর্কিত", "বিষয়ে", "বিষয়ে", "বলা", "বলে",
    "রচিত", "রচনা", "রচয়িতা", "রচয়িতা", "লেখক", "কবি", "স্রষ্টা",
    "দেশ", "জেলা", "শহর", "গ্রাম", "রাজধানী", "স্থান", "জনপদ", "নদী",
    "উপনদী", "খাল", "হ্রদ", "সাগর", "মহাসাগর", "পর্বত", "গ্রন্থ", "বই",
    "কাব্য", "কবিতা", "উপন্যাস", "নাটক", "গান", "শব্দ", "শব্দদুটি",
    "বছর", "সাল", "তারিখ", "সময়", "সময়", "সংখ্যা", "ইঞ্জিন", "সার্চ",
    "কত", "কবে", "কোথায়", "কোথায়", "which", "what", "who", "where",
    "when", "name", "author", "wrote", "first", "last", "main", "ancient",
}
ANCHOR_INFLECTION_SUFFIXES = {
    "র", "এর", "ের", "কে", "তে", "ে", "য়", "য়", "টি", "টা", "টির",
    "দের", "গুলো", "গুলি", "দুটি",
}


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def id_set_sha256(values: set[str]) -> str:
    """Hash an unordered ID set without depending on input order."""

    payload = "".join(f"{value}\n" for value in sorted(values))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def walk_keys(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key).casefold()
            yield from walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_keys(child)


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def answer_equivalence_key(value: object) -> str:
    """Canonical answer key with bounded Bengali numeral-counter joining."""

    return NUMERIC_COUNTER_SPACE_RE.sub("", canonicalize(value))


def number_set(value: object) -> list[str]:
    return sorted(set(NUMBER_RE.findall(canonicalize(value))))


def negation_set(value: object) -> list[str]:
    normalized = canonicalize(value)
    tokens = set(normalized.split())
    return sorted(word for word in NEGATIONS if word in tokens)


def substantive_token_set(value: object) -> set[str]:
    return {
        token
        for token in canonicalize(value).split()
        if len(token) >= 3 and token not in FUZZY_STOPWORDS and not token.isdigit()
    }


def _generic_anchor_term(token: str) -> bool:
    if token in FUZZY_STOPWORDS or token in ANCHOR_GENERIC_TERMS:
        return True
    for stem in ANCHOR_GENERIC_TERMS:
        if token.startswith(stem) and token[len(stem):] in ANCHOR_INFLECTION_SUFFIXES:
            return True
    return False


def primary_query_subject_anchor(value: object) -> str | None:
    """Return the first distinctive ordered query token, when one exists.

    This deliberately abstains when a question contains relation/type words
    only.  It does not guess that every noun is a named entity.
    """

    for token in canonicalize(value).split():
        if len(token) < 3 or token.isdigit() or _generic_anchor_term(token):
            continue
        return token
    return None


def _anchor_token_match(anchor: str, token: str) -> str | None:
    if anchor == token:
        return "exact_token"
    shorter, longer = sorted((anchor, token), key=len)
    suffix = longer[len(shorter):] if longer.startswith(shorter) else ""
    if len(shorter) >= 4 and suffix in ANCHOR_INFLECTION_SUFFIXES:
        return "bounded_inflection"
    # OCR tolerance is intentionally restricted to long, similarly sized
    # tokens.  It recovers e.g. বাংলাভাষায় -> রাংলাভাযষায়, while বাংলাদেশের
    # or a partial person's/river's name cannot pass through a shared prefix.
    if (
        min(len(anchor), len(token)) >= 8
        and abs(len(anchor) - len(token)) <= 2
        and SequenceMatcher(None, anchor, token, autojunk=False).ratio() >= 0.84
    ):
        return "bounded_ocr_similarity"
    return None


def subject_anchor_match(query: object, evidence: object) -> dict[str, Any]:
    anchor = primary_query_subject_anchor(query)
    if anchor is None:
        return {
            "required": False, "anchor": None, "matched": None,
            "method": "no_distinctive_subject_anchor",
        }
    for token in canonicalize(evidence).split():
        method = _anchor_token_match(anchor, token)
        if method:
            return {
                "required": True, "anchor": anchor, "matched": token,
                "method": method,
            }
    return {
        "required": True, "anchor": anchor, "matched": None,
        "method": "missing",
    }


def question_target_types(value: object) -> set[str]:
    text = canonicalize(value)
    result: set[str] = set()
    if re.search(r"(?:নদী|নদ|উপনদী|খাল|হ্রদ|সাগর|মহাসাগর|river|lake|ocean)", text):
        result.add("river_or_waterbody")
    if re.search(r"(?:কোথায়|কোথায়|কোন দেশ|কোন জেলা|রাজধানী|কোন শহর|কোন গ্রাম|country|city|capital)", text):
        result.add("place_or_jurisdiction")
    if re.search(r"(?:কত সালে|কোন সালে|কবে|কত তারিখ|কোন তারিখ|when|year|date)", text):
        result.add("date_or_time")
    if re.search(r"(?:কতটি|কয়টি|কয়টি|কত জন|কত টাকা|কত দিনে|how many)", text):
        result.add("quantity")
    if re.search(r"(?:কোন রচনা|কোন কাব্য|কোন উপন্যাস|কোন নাটক|কোন গ্রন্থ|কোন গান|which (?:book|novel|poem|play|song))", text):
        result.add("creative_work")
    if re.search(r"(?:রচয়িতা|রচয়িতা|লেখক|কবি|স্রষ্টা|উপাচার্য|রাষ্ট্রপতি|প্রধানমন্ত্রী|who wrote|author)", text):
        result.add("person")
    return result


def fuzzy_candidate_gate(
    query: object,
    source_question: object,
    score: float,
    *,
    exact: bool = False,
) -> dict[str, Any]:
    """Keep weak/short/type-conflicting fuzzy hits out of model prompts."""
    if exact:
        return {
            "eligible": True,
            "reasons": [],
            "policy": "exact_full_normalized_question",
            "shared_substantive_tokens": sorted(
                substantive_token_set(query).intersection(substantive_token_set(source_question))
            ),
            "query_target_types": sorted(question_target_types(query)),
            "source_target_types": sorted(question_target_types(source_question)),
        }
    reasons: list[str] = []
    query_tokens = substantive_token_set(query)
    source_tokens = substantive_token_set(source_question)
    shared = query_tokens.intersection(source_tokens)
    if len(query_tokens) < 2:
        reasons.append("short_or_keyword_only_query")
    if float(score) < 0.22:
        reasons.append("similarity_below_0_22")
    query_numbers = number_set(query)
    source_numbers = number_set(source_question)
    if query_numbers != source_numbers:
        reasons.append("number_set_mismatch")
    query_negations = negation_set(query)
    source_negations = negation_set(source_question)
    if query_negations != source_negations:
        reasons.append("negation_set_mismatch")
    query_types = question_target_types(query)
    source_types = question_target_types(source_question)
    if query_types and source_types and query_types.isdisjoint(source_types):
        reasons.append("answer_type_intent_conflict")
    distinctive_single_overlap = (
        len(shared) == 1
        and len(next(iter(shared))) >= 8
        and float(score) >= 0.55
    )
    if len(shared) < 2 and not distinctive_single_overlap:
        reasons.append("insufficient_substantive_token_overlap")
    return {
        "eligible": not reasons,
        "reasons": sorted(set(reasons)),
        "policy": "fuzzy_similarity_overlap_number_negation_and_answer_type_gate",
        "shared_substantive_tokens": sorted(shared),
        "query_target_types": sorted(query_types),
        "source_target_types": sorted(source_types),
    }


def passage_candidate_gate(
    query: object, passage: object, score: float,
) -> dict[str, Any]:
    """Gate nonterminal page evidence without pretending it is a question.

    Dense OCR pages can mention many unrelated entities.  A passage therefore
    needs at least two substantive query-token matches, useful token coverage,
    and containment of every number/negation explicitly present in the query.
    Unlike question-to-question retrieval, extra numbers in a source passage
    are permitted because a source-grounded block may contain choices or an
    explanation.  The result is evidence only and can never create a verdict.
    """

    reasons: list[str] = []
    query_tokens = substantive_token_set(query)
    passage_tokens = substantive_token_set(passage)
    shared = query_tokens.intersection(passage_tokens)
    coverage = len(shared) / len(query_tokens) if query_tokens else 0.0
    if len(query_tokens) < 2:
        reasons.append("short_or_keyword_only_query")
    if float(score) < 0.20:
        reasons.append("passage_similarity_below_0_20")
    if len(shared) < 2:
        reasons.append("passage_requires_two_substantive_overlaps")
    if coverage < 0.35 and len(shared) < 3:
        reasons.append("insufficient_query_token_coverage")
    query_numbers = set(number_set(query))
    passage_numbers = set(number_set(passage))
    if not query_numbers.issubset(passage_numbers):
        reasons.append("query_number_missing_from_passage")
    query_negations = set(negation_set(query))
    passage_negations = set(negation_set(passage))
    if not query_negations.issubset(passage_negations):
        reasons.append("query_negation_missing_from_passage")
    return {
        "eligible": not reasons,
        "reasons": sorted(set(reasons)),
        "policy": "nonterminal_passage_overlap_coverage_number_negation_gate",
        "shared_substantive_tokens": sorted(shared),
        "query_token_coverage": round(coverage, 8),
        "query_target_types": sorted(question_target_types(query)),
        "source_target_types": [],
    }


def tighten_joykoli_structured_gate(
    gate: dict[str, Any], *, exact: bool, query: object | None = None,
    evidence: object | None = None, passage_evidence: bool = False,
) -> dict[str, Any]:
    """Apply Joykoli-only overlap and primary-subject protections."""

    if exact:
        return gate
    result = dict(gate)
    reasons = set(result.get("reasons") or [])
    policy_suffixes: list[str] = []
    if not passage_evidence and len(gate.get("shared_substantive_tokens") or []) < 2:
        reasons.add("joykoli_nonexact_requires_two_substantive_overlaps")
        policy_suffixes.append("joykoli_no_single_token_exception")
    if query is not None and evidence is not None:
        anchor = subject_anchor_match(query, evidence)
        result["query_primary_subject_anchor"] = anchor
        if anchor["required"] and not anchor["matched"]:
            reasons.add(
                "query_primary_anchor_missing_from_passage"
                if passage_evidence
                else "query_primary_anchor_missing_from_source_question"
            )
        policy_suffixes.append("joykoli_primary_subject_anchor")
    result["eligible"] = not reasons
    result["reasons"] = sorted(reasons)
    if policy_suffixes:
        result["policy"] = str(result.get("policy", "")) + "+" + "+".join(policy_suffixes)
    return result


def relation(response: object, answers: list[str]) -> str:
    if classify_answer_stance(response, answers).status != "supported":
        return "none"
    response_key = answer_equivalence_key(response)
    answer_keys = unique([answer_equivalence_key(value) for value in answers])
    if not response_key or not answer_keys:
        return "none"
    if response_key in answer_keys:
        return "exact"
    if any(
        min(len(response_key), len(answer)) >= 4
        and (response_key in answer or answer in response_key)
        for answer in answer_keys
    ):
        return "containment"
    return "none"


def load_input(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        forbidden = sorted(FORBIDDEN_FIELDS.intersection(walk_keys(row)))
        if forbidden:
            raise ValueError(f"input line {line_number} contains forbidden fields: {forbidden}")
        example_id = row.get("example_id", row.get("id"))
        if example_id is None or not str(example_id).strip():
            raise ValueError(f"input line {line_number} has no example_id/id")
        required = ("model_prompt_bn", "model_response_bn")
        if any(key not in row for key in required):
            raise ValueError(f"input line {line_number} is not Phase-2 formatted")
        rows.append({**row, "example_id": str(example_id)})
    if not rows:
        raise ValueError("retrieval input is empty")
    ids = [row["example_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("retrieval input has duplicate IDs")
    return rows


def load_terminal_context_ids(path: Path) -> set[str]:
    """Read only IDs already settled by the label-free context stage."""

    ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        example_id = str(row.get("example_id", "")).strip()
        if not example_id:
            raise ValueError(f"context line {line_number} has no example_id")
        if str(row.get("status", "")).startswith("terminal_deterministic_"):
            if row.get("verdict") not in {0, 1}:
                raise ValueError(f"context line {line_number} has invalid terminal verdict")
            ids.add(example_id)
    return ids


def load_context_external_lookup_ids(path: Path) -> set[str]:
    """Reject legacy contextual lookup authorization and return an empty set."""

    ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        example_id = str(row.get("example_id", "")).strip()
        if not example_id:
            raise ValueError(f"context line {line_number} has no example_id")
        route = row.get("context_policy_route") or {}
        analysis = row.get("rule_application_analysis") or {}
        if (
            analysis.get("external_lookup_allowed") is True
            or route.get("external_lookup_allowed") is True
        ):
            raise ValueError(
                "context grounding attempts to authorize external retrieval"
            )
    return ids


def load_index(spec: dict[str, Any]) -> dict[str, Any]:
    directory = Path(spec["directory"])
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") == MMAP_CACHE_VERSION:
        loaded = load_mmap_index(directory)
        rights = loaded["manifest"].get("rights_policy") or {}
        if rights.get("bundle_allowed") is not True or rights.get("quarantined") is not False:
            close_mmap_index(loaded)
            raise ValueError(f"{spec['source_id']} mmap cache is not retrieval-deployable")
        return {**spec, **loaded}
    required = ("vectorizer.joblib", "matrix.npz", "records.jsonl")
    for name in required:
        expected = manifest.get("files", {}).get(name)
        observed = sha256_file(directory / name)
        if expected != observed:
            raise ValueError(f"{spec['source_id']} index hash mismatch for {name}")
    records = [
        json.loads(line)
        for line in (directory / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [int(record["index"]) for record in records] != list(range(len(records))):
        raise ValueError(f"{spec['source_id']} record indexes are not dense and ordered")
    matrix = sparse.load_npz(directory / "matrix.npz").tocsr()
    if matrix.shape[0] != len(records):
        raise ValueError(f"{spec['source_id']} matrix/record row mismatch")
    exact_lookup: dict[str, list[int]] = {}
    for source_index, record in enumerate(records):
        key = canonicalize(record.get("normalized_question", record.get("question", "")))
        if key:
            exact_lookup.setdefault(key, []).append(source_index)
    return {
        **spec,
        "directory": directory,
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "vectorizer": joblib.load(directory / "vectorizer.joblib"),
        "matrix": matrix,
        "records": records,
        "exact_lookup": exact_lookup,
    }


def unpack_source(record: dict[str, Any], source_id: str) -> tuple[list[str], list[str], int]:
    if source_id == "bangla_mmlu":
        source_rows = list(record.get("records") or [])
        answers = unique([str(row.get("keyed_answer", "")).strip() for row in source_rows])
        choices = unique([
            str(choice).strip()
            for row in source_rows
            for choice in (row.get("choices") or [])
        ])
        return answers, choices, len(source_rows)
    answers = unique([str(value).strip() for value in (record.get("answers") or [])])
    choices = unique([str(value).strip() for value in (record.get("choices") or [])])
    return answers, choices, len(record.get("records") or [])


def source_verdict_candidate(
    source_id: str, *, exact: bool, response: str,
    answers: list[str], choices: list[str], response_relation: str,
    record: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not exact:
        return None
    answer_keys = unique([answer_equivalence_key(value) for value in answers])
    if source_id == "bangla_mmlu":
        if len(answer_keys) != 1:
            return None
        response_key = answer_equivalence_key(response)
        if response_key == answer_keys[0]:
            return {"verdict": 1, "rule": "exact_question_key_consensus_response_exact_key"}
        if response_key and response_key in {answer_equivalence_key(value) for value in choices}:
            return {"verdict": 0, "rule": "exact_question_key_consensus_response_exact_wrong_option"}
        return None
    if source_id == "nctb_qa_87805" and response_relation in {"exact", "containment"}:
        return {"verdict": 1, "rule": f"exact_question_positive_answer_{response_relation}"}
    if source_id == "nctb_education_aux" and record is not None:
        response_key = answer_equivalence_key(response)
        consensus = answer_equivalence_key(record.get("terminal_consensus_answer", ""))
        if consensus and response_key == consensus:
            return {"verdict": 1, "rule": "exact_question_rights_eligible_consensus_answer"}
        negative_keys = {
            answer_equivalence_key(value) for value in record.get("terminal_negative_choices", [])
        }
        if consensus and response_key in negative_keys:
            return {"verdict": 0, "rule": "exact_question_rights_eligible_consensus_wrong_option"}
    return None


def strict_exact_key_conflicts(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Find opt-in answer-key conflicts without granting a new source authority.

    A corroboration-only OCR/book source may challenge an existing exact-source
    verdict only when both records are exact normalized-question matches, both
    have one keyed answer, and their complete normalized MCQ choice sets are
    identical with at least two choices.  The challenge can only quarantine a
    terminal route; it can never create a verdict.
    """

    terminal = [
        candidate for candidate in candidates
        if candidate.get("exact_normalized") is True
        and candidate.get("source_verdict_candidate") is not None
    ]
    challengers = [
        candidate for candidate in candidates
        if candidate.get("exact_normalized") is True
        and candidate.get("exact_conflict_policy") == STRICT_EXACT_CONFLICT_POLICY
        and candidate.get("exact_conflict_eligible") is True
    ]
    conflicts: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, int]] = set()
    for authority in terminal:
        authority_answers = unique([
            answer_equivalence_key(value) for value in authority.get("answers", [])
        ])
        authority_choices = {
            answer_equivalence_key(value) for value in authority.get("choices", [])
            if answer_equivalence_key(value)
        }
        if len(authority_answers) != 1 or len(authority_choices) < 2:
            continue
        for challenger in challengers:
            if challenger.get("source_id") == authority.get("source_id"):
                continue
            challenger_answers = unique([
                answer_equivalence_key(value) for value in challenger.get("answers", [])
            ])
            challenger_choices = {
                answer_equivalence_key(value) for value in challenger.get("choices", [])
                if answer_equivalence_key(value)
            }
            if (
                len(challenger_answers) != 1
                or challenger_choices != authority_choices
                or challenger_answers[0] == authority_answers[0]
            ):
                continue
            key = (
                str(authority.get("source_id")),
                int(authority.get("source_record_index", -1)),
                str(challenger.get("source_id")),
                int(challenger.get("source_record_index", -1)),
            )
            if key in seen:
                continue
            seen.add(key)
            conflicts.append({
                "rule": STRICT_EXACT_CONFLICT_POLICY,
                "effect": "quarantine_only_no_verdict",
                "authority_source_id": key[0],
                "authority_source_record_index": key[1],
                "challenger_source_id": key[2],
                "challenger_source_record_index": key[3],
                "choice_set_sha256": hashlib.sha256(
                    "\n".join(sorted(authority_choices)).encode("utf-8")
                ).hexdigest(),
                "authority_answer_sha256": hashlib.sha256(
                    authority_answers[0].encode("utf-8")
                ).hexdigest(),
                "challenger_answer_sha256": hashlib.sha256(
                    challenger_answers[0].encode("utf-8")
                ).hexdigest(),
            })
    return conflicts


def retrieval_candidate(
    row: dict[str, Any],
    query_key: str,
    index: dict[str, Any],
    source_index: int,
    *,
    rank: int,
    score: float,
) -> dict[str, Any]:
    """Build one response-aware evidence record from a source record."""

    record = index["records"][source_index]
    source_question = str(record.get("question", ""))
    supporting_text = str(record.get("supporting_text", ""))
    passage_evidence = (
        str(record.get("record_kind", "")) == "page_ocr_repair_chunk"
        and not source_question.strip()
        and bool(supporting_text.strip())
    )
    source_key = canonicalize(record.get("normalized_question", source_question))
    answers, choices, record_count = unpack_source(record, index["source_id"])
    response = str(row["model_response_bn"])
    exact = bool(query_key and query_key == source_key)
    response_relation = relation(response, answers)
    verdict = source_verdict_candidate(
        index["source_id"],
        exact=exact,
        response=response,
        answers=answers,
        choices=choices,
        response_relation=response_relation,
        record=record,
    )
    query_numbers = row["_query_numbers"]
    query_negations = row["_query_negations"]
    gate_text = supporting_text if passage_evidence else source_question
    source_numbers = number_set(gate_text)
    source_negations = negation_set(gate_text)
    model_facing_gate = (
        passage_candidate_gate(row["model_prompt_bn"], gate_text, score)
        if passage_evidence and not exact
        else fuzzy_candidate_gate(
            row["model_prompt_bn"], source_question, score, exact=exact
        )
    )
    if index["source_id"] == "joykoli_six_part":
        model_facing_gate = tighten_joykoli_structured_gate(
            model_facing_gate, exact=exact, query=row["model_prompt_bn"],
            evidence=gate_text, passage_evidence=passage_evidence,
        )
    return {
        "source_id": index["source_id"],
        "source_record_index": int(source_index),
        "rank": int(rank),
        "score": round(float(score), 8),
        "exact_normalized": exact,
        "question": source_question,
        "supporting_text": supporting_text if supporting_text != source_question else "",
        "passage_evidence": passage_evidence,
        "source_metadata": record.get("metadata", {}),
        "answers": answers,
        "choices": choices,
        "exact_conflict_policy": str(index.get("exact_conflict_policy", "none")),
        "exact_conflict_eligible": bool(
            (record.get("metadata") or {}).get("exact_conflict_eligible", False)
        ),
        "source_record_count": record_count,
        "query_numbers": query_numbers,
        "source_numbers": source_numbers,
        "number_set_match": query_numbers == source_numbers,
        "query_negations": query_negations,
        "source_negations": source_negations,
        "negation_set_match": query_negations == source_negations,
        "model_facing_eligible": bool(model_facing_gate["eligible"]),
        "model_facing_gate": model_facing_gate,
        "response_answer_relation": response_relation,
        "source_verdict_candidate": verdict,
        "query_text": str(row["model_prompt_bn"]),
        "query_context_available": bool(row.get("context_available")),
        "query_policy_operation": str(
            (row.get("context_policy_route") or {}).get("operation") or ""
        ),
    }


def composite_retrieval_candidate(
    row: dict[str, Any], value: dict[str, Any], *, rank: int,
) -> dict[str, Any]:
    """Adapt one FTS hit to the shared nonterminal evidence schema."""

    if value.get("model_facing") is not True:
        raise ValueError("component/audit FTS hit cannot enter model-facing candidates")
    if value.get("semantic_role") != "closed_book_knowledge_evidence_candidate":
        raise ValueError("non-knowledge FTS role cannot enter factual evidence")
    if value.get("terminal_label_authority") is not False or value.get("verdict") != "NEI":
        raise ValueError("composite FTS evidence must remain nonterminal NEI")
    source_question = str(value.get("question") or "")
    supporting_text = str(value.get("evidence_excerpt") or "")
    answer = str(value.get("answer") or "")
    answers = [answer] if answer else []
    response = str(row["model_response_bn"])
    source_text = "\n".join(
        item for item in (source_question, supporting_text) if item.strip()
    )
    query_numbers = row.get("_query_numbers", number_set(row["model_prompt_bn"]))
    query_negations = row.get("_query_negations", negation_set(row["model_prompt_bn"]))
    source_numbers = number_set(source_text)
    source_negations = negation_set(source_text)
    raw_rank = float(value.get("retrieval_score", 0.0))
    display_score = max(0.0, -raw_rank)
    exact = bool(value.get("exact_question"))
    if source_question:
        model_facing_gate = fuzzy_candidate_gate(
            row["model_prompt_bn"], source_question, display_score, exact=exact
        )
    else:
        model_facing_gate = passage_candidate_gate(
            row["model_prompt_bn"], supporting_text, display_score
        )
    gate_reasons = set(model_facing_gate.get("reasons") or [])
    gate_reasons.update({
        "rights_authorized_closed_book_knowledge_role",
        "bounded_content_conjunction_or_exact_question",
        "nonterminal_support_or_counter_candidate",
    })
    model_facing_gate["reasons"] = sorted(gate_reasons)
    model_facing_gate["semantic_verification_required"] = True
    return {
        "source_id": str(value["source_id"]),
        "source_record_index": int(value["source_record_index"]),
        "source_locator": str(value["source_locator"]),
        "source_record_sha256": str(value["record_sha256"]),
        "semantic_role": str(value["semantic_role"]),
        "rank": int(value.get("within_authority_tier_rank", rank)),
        "score": round(display_score, 8),
        "score_kind": str(
            value.get("retrieval_score_kind")
            or "sqlite_fts5_bm25_negated_for_display_only"
        ),
        "exact_normalized": exact,
        "question": source_question,
        "supporting_text": supporting_text,
        "passage_evidence": bool(supporting_text),
        "source_metadata": value.get("metadata") or {},
        "answers": answers,
        "choices": list(value.get("choices") or []),
        "exact_conflict_policy": "none",
        "exact_conflict_eligible": False,
        "source_record_count": 1,
        "query_numbers": query_numbers,
        "source_numbers": source_numbers,
        "number_set_match": query_numbers == source_numbers,
        "query_negations": query_negations,
        "source_negations": source_negations,
        "negation_set_match": query_negations == source_negations,
        "model_facing_eligible": bool(model_facing_gate["eligible"]),
        "model_facing_gate": model_facing_gate,
        "response_answer_relation": relation(response, answers),
        "source_verdict_candidate": None,
        "query_text": str(row["model_prompt_bn"]),
        "query_context_available": bool(row.get("context_available")),
        "query_policy_operation": str(
            (row.get("context_policy_route") or {}).get("operation") or ""
        ),
    }


def _composite_evidence_role(candidate: dict[str, Any]) -> str:
    relation_value = str(candidate.get("response_answer_relation", "none"))
    if relation_value in {"exact", "containment"}:
        return "support_candidate"
    if candidate.get("exact_normalized") is True and candidate.get("answers"):
        return "counter_candidate"
    return "neutral_candidate"


def rank_and_bound_composite_candidates(
    candidates: list[dict[str, Any]], *, top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply semantic safety and authority before the final composite top-k.

    Quarantined candidates never displace eligible evidence.  When top-k has
    at least two slots, retain the highest-authority aligned exact counter
    candidate if ordinary authority truncation would remove every counter.
    The result remains deterministic under input permutation.
    """

    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be 1..20")
    ranked = authority_rank_candidates(candidates)
    for candidate in ranked:
        candidate["evidence_role"] = _composite_evidence_role(candidate)
    eligible = [
        candidate for candidate in ranked
        if candidate.get("model_facing_eligible") is True
    ]
    quarantined = [
        candidate for candidate in ranked
        if candidate.get("model_facing_eligible") is not True
    ]
    selected = eligible[:top_k]
    counter_available = next(
        (
            candidate for candidate in eligible
            if candidate.get("evidence_role") == "counter_candidate"
        ),
        None,
    )
    counter_forced = False
    if (
        top_k >= 2
        and counter_available is not None
        and counter_available not in selected
        and not any(
            candidate.get("evidence_role") == "counter_candidate"
            for candidate in selected
        )
    ):
        # Retain the strongest evidence in slot zero.  Replace the weakest
        # remaining non-counter slot, then restore canonical authority order.
        selected[-1] = counter_available
        selected = authority_rank_candidates(selected)
        counter_forced = True
    bounded_quarantine = quarantined[:top_k]
    audit = {
        "pool_candidates": len(candidates),
        "eligible_candidates": len(eligible),
        "quarantined_candidates": len(quarantined),
        "selected_candidates": len(selected),
        "selected_source_ids": [candidate["source_id"] for candidate in selected],
        "counter_available": counter_available is not None,
        "counter_selected": any(
            candidate.get("evidence_role") == "counter_candidate"
            for candidate in selected
        ),
        "counter_forced_into_top_k": counter_forced,
        "selection_policy": (
            "lane_then_exclusive_policy_then_factual_slots_then_response_"
            "symmetric_support_counter_then_authority_then_exactness_with_"
            "reserved_counter_slot"
        ),
    }
    # Quarantined evidence remains audit-only and bounded independently; it is
    # re-ranked after eligible evidence by the same shared authority function.
    return authority_rank_candidates(selected + bounded_quarantine), audit


def _inspectable_retrieval_evidence(candidate: dict[str, Any]) -> bool:
    """Require both a source locator and human-readable relation evidence."""

    has_locator = bool(str(candidate.get("source_id", "")).strip()) and (
        candidate.get("source_record_index") is not None
    )
    has_evidence = any(
        bool(value)
        for value in (
            candidate.get("question"), candidate.get("supporting_text"),
            candidate.get("answers"), candidate.get("choices"),
        )
    )
    return has_locator and has_evidence


def write_closed_book_eval_evidence_ledger(
    input_rows: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Persist every inspectable closed-book retrieval flag for eval labeling.

    This is deliberately evaluator-private.  It records the prompt, response,
    source locator, relation, eligible and quarantined evidence, exact source
    candidates, and counter-evidence cues.  It never assigns a label and is
    not a model-facing input or terminal-adapter artifact.
    """

    if len(input_rows) != len(retrieval_rows):
        raise ValueError("retrieval/eval-ledger row populations differ")
    source_by_id = {str(row["example_id"]): row for row in input_rows}
    if len(source_by_id) != len(input_rows):
        raise ValueError("eval-ledger input has duplicate IDs")

    queue_rows: list[dict[str, Any]] = []
    evidence_items: list[dict[str, Any]] = []
    contextual_excluded = 0
    closed_without_inspectable_evidence = 0
    disposition_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for retrieval in retrieval_rows:
        example_id = str(retrieval["example_id"])
        source = source_by_id.get(example_id)
        if source is None:
            raise ValueError(f"eval-ledger retrieval ID absent from input: {example_id}")
        if bool(source.get("context_available")):
            contextual_excluded += 1
            continue
        candidates = [
            ("model_facing_eligible", candidate)
            for candidate in retrieval.get("retrieval_candidates", [])
        ] + [
            ("audit_quarantined", candidate)
            for candidate in retrieval.get("retrieval_audit_quarantined_candidates", [])
        ]
        candidates = [
            (disposition, candidate) for disposition, candidate in candidates
            if _inspectable_retrieval_evidence(candidate)
        ]
        if not candidates:
            closed_without_inspectable_evidence += 1
            continue

        item_ids: list[str] = []
        possible_counter_ids: list[str] = []
        exact_source_candidate_ids: list[str] = []
        for disposition, candidate in candidates:
            item_core = {
                "example_id": example_id,
                "lane": "closed_book",
                "disposition": disposition,
                "source_id": str(candidate["source_id"]),
                "source_record_index": int(candidate["source_record_index"]),
                "rank": int(candidate.get("rank", 0)),
                "score": float(candidate.get("score", 0.0)),
                "exact_normalized": bool(candidate.get("exact_normalized")),
                "response_answer_relation": str(
                    candidate.get("response_answer_relation", "none")
                ),
                "model_facing_gate": candidate.get("model_facing_gate") or {},
                "question": candidate.get("question"),
                "supporting_text": candidate.get("supporting_text"),
                "answers": candidate.get("answers") or [],
                "choices": candidate.get("choices") or [],
                "source_metadata": candidate.get("source_metadata") or {},
                "source_verdict_candidate": candidate.get("source_verdict_candidate"),
                "evidence_use_policy": (
                    "private_eval_labeling_only_pending_adjudication;"
                    "not_an_assigned_label;not_model_facing"
                ),
            }
            item_id = sha256_json(item_core)
            evidence_items.append({**item_core, "evidence_item_id": item_id})
            item_ids.append(item_id)
            disposition_counts[disposition] += 1
            source_counts[item_core["source_id"]] += 1
            if (
                item_core["answers"]
                and item_core["response_answer_relation"] == "none"
            ):
                possible_counter_ids.append(item_id)
            if item_core["source_verdict_candidate"] is not None:
                exact_source_candidate_ids.append(item_id)

        merged = retrieval.get("merged_source_candidate") or {}
        queue_core = {
            "example_id": example_id,
            "source_index": int(source.get("source_index", retrieval.get("source_index", 0))),
            "lane": "closed_book",
            "eval_label_status": "pending_adjudication",
            "eval_label": None,
            "prompt_bn": str(source.get("prompt_bn", source.get("model_prompt_bn", ""))),
            "response_bn": str(source.get("response_bn", source.get("model_response_bn", ""))),
            "query_sha256": retrieval["query_sha256"],
            "response_sha256": retrieval["response_sha256"],
            "retrieval_flag": "inspectable_evidence_present",
            "merged_source_candidate": merged,
            "evidence_item_ids": item_ids,
            "possible_counter_evidence_item_ids": possible_counter_ids,
            "exact_source_candidate_item_ids": exact_source_candidate_ids,
            "model_facing_evidence_count": sum(
                disposition == "model_facing_eligible"
                for disposition, _ in candidates
            ),
            "quarantined_evidence_count": sum(
                disposition == "audit_quarantined"
                for disposition, _ in candidates
            ),
            "label_assignment_policy": (
                "human_or_exact_policy_adjudication_required;retrieval_flag_is_not_gold"
            ),
        }
        queue_rows.append({**queue_core, "eval_record_id": sha256_json(queue_core)})

    queue_path = output_dir / "closed_book_eval_label_queue.private.jsonl"
    evidence_path = output_dir / "closed_book_eval_evidence_items.private.jsonl"
    csv_path = output_dir / "closed_book_eval_label_queue.private.csv"
    with queue_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in queue_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with evidence_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in evidence_items:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    evidence_by_id = {row["evidence_item_id"]: row for row in evidence_items}
    csv_fields = (
        "example_id", "source_index", "eval_label_status", "eval_label",
        "prompt_bn", "response_bn", "merged_candidate_status",
        "merged_candidate_verdict", "model_facing_evidence_count",
        "quarantined_evidence_count", "source_ids", "possible_counter_count",
        "top_evidence_question", "top_evidence_answers", "top_supporting_text",
        "eval_record_id",
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in queue_rows:
            items = [evidence_by_id[item_id] for item_id in row["evidence_item_ids"]]
            top = items[0]
            merged = row["merged_source_candidate"]
            writer.writerow({
                "example_id": row["example_id"],
                "source_index": row["source_index"],
                "eval_label_status": row["eval_label_status"],
                "eval_label": "",
                "prompt_bn": row["prompt_bn"],
                "response_bn": row["response_bn"],
                "merged_candidate_status": merged.get("status"),
                "merged_candidate_verdict": merged.get("verdict"),
                "model_facing_evidence_count": row["model_facing_evidence_count"],
                "quarantined_evidence_count": row["quarantined_evidence_count"],
                "source_ids": "|".join(sorted({item["source_id"] for item in items})),
                "possible_counter_count": len(row["possible_counter_evidence_item_ids"]),
                "top_evidence_question": str(top.get("question") or "")[:1000],
                "top_evidence_answers": " | ".join(map(str, top.get("answers") or []))[:1000],
                "top_supporting_text": str(top.get("supporting_text") or "")[:1000],
                "eval_record_id": row["eval_record_id"],
            })

    return {
        "policy": (
            "every inspectable closed-book retrieval flag is retained for private "
            "eval labeling; no label is assigned automatically"
        ),
        "model_facing_input": False,
        "terminal_label_assignment": False,
        "records": len(queue_rows),
        "evidence_items": len(evidence_items),
        "contextual_rows_excluded": contextual_excluded,
        "closed_rows_without_inspectable_evidence": closed_without_inspectable_evidence,
        "disposition_counts": dict(sorted(disposition_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "queue": {
            "path": str(queue_path), "bytes": queue_path.stat().st_size,
            "sha256": sha256_file(queue_path),
        },
        "evidence": {
            "path": str(evidence_path), "bytes": evidence_path.stat().st_size,
            "sha256": sha256_file(evidence_path),
        },
        "monitor_csv": {
            "path": str(csv_path), "bytes": csv_path.stat().st_size,
            "sha256": sha256_file(csv_path),
        },
    }


def select_composite_query_rows(
    rows: list[dict[str, Any]],
    candidates_by_row: list[list[dict[str, Any]]],
    *,
    skip_example_ids: set[str],
    mode: str,
) -> list[int]:
    """Select closed-book rows for the large composite FTS cache.

    ``all_closed`` is the evidence-heavy research path: it preserves composite
    counter-evidence even when a smaller source already landed.
    ``unresolved_only`` is the selective evidence path: it preserves composite
    support/counter-evidence for rows without an exact terminal source
    candidate.  ``residual_only`` is the minimum-latency ablation: it queries
    the composite cache only when the mmap/book/dataset stage produced no
    eligible evidence.  No mode permits contextual lookup or changes terminal
    authority.
    """

    if mode not in COMPOSITE_QUERY_MODES:
        raise ValueError(f"invalid composite query mode: {mode}")
    if len(rows) != len(candidates_by_row):
        raise ValueError("composite row/candidate cardinality mismatch")
    selected: list[int] = []
    for row_index, (row, candidates) in enumerate(
        zip(rows, candidates_by_row, strict=True)
    ):
        if (
            str(row.get("example_id")) in skip_example_ids
            or bool(row.get("context_available"))
        ):
            continue
        if mode == "unresolved_only" and any(
            candidate.get("source_verdict_candidate") is not None
            for candidate in candidates
        ):
            continue
        if mode == "residual_only" and any(
            candidate.get("model_facing_eligible") is True
            for candidate in candidates
        ):
            continue
        selected.append(row_index)
    return selected


def build(
    input_path: Path,
    output_dir: Path,
    *,
    index_specs: list[dict[str, Any]] | None = None,
    top_k: int = 5,
    batch_size: int = 64,
    skip_example_ids: set[str] | None = None,
    context_external_lookup_ids: set[str] | None = None,
    composite_cache_dir: Path | None = None,
    composite_query_mode: str = "all_closed",
) -> dict[str, Any]:
    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be 1..20")
    if not 1 <= batch_size <= 512:
        raise ValueError("batch_size must be 1..512")
    if composite_query_mode not in COMPOSITE_QUERY_MODES:
        raise ValueError(
            f"composite_query_mode must be one of {sorted(COMPOSITE_QUERY_MODES)}"
        )
    started = time.perf_counter()
    rows = load_input(input_path)
    if any(
        "context_available" in row
        and type(row.get("context_available")) is not bool
        for row in rows
    ):
        raise ValueError("context_available must be an exact boolean")
    skip_ids = {str(value) for value in (skip_example_ids or set())}
    contextual_ids = {
        str(row["example_id"]) for row in rows if bool(row.get("context_available"))
    }
    requested_context_lookup_ids = {
        str(value) for value in (context_external_lookup_ids or set())
    }
    if requested_context_lookup_ids:
        raise ValueError("contextual external lookup IDs are forbidden")
    context_lookup_ids: set[str] = set()
    disabled_contextual_ids = set(contextual_ids)
    input_ids = {row["example_id"] for row in rows}
    unknown_skip_ids = skip_ids - input_ids
    if unknown_skip_ids:
        raise ValueError(f"preterminal skip IDs are absent from input: {sorted(unknown_skip_ids)[:5]}")
    # ``None`` means the historical/default mmap source set.  An explicit
    # empty list is materially different: it is the strict composite-only
    # runtime used by MORICHIKA v3, where no legacy source may be opened by
    # accident merely because the list is falsey.
    effective_index_specs = DEFAULT_INDEX_SPECS if index_specs is None else index_specs
    indexes = [load_index(spec) for spec in effective_index_specs]
    normalized_queries = [canonicalize(row["model_prompt_bn"]) for row in rows]
    for row in rows:
        row["_query_numbers"] = number_set(row["model_prompt_bn"])
        row["_query_negations"] = negation_set(row["model_prompt_bn"])
    candidates_by_row: list[list[dict[str, Any]]] = [[] for _ in rows]
    index_seconds: dict[str, float] = {}
    composite_manifest: dict[str, Any] | None = None

    # Exact lookups are cheaper and more authoritative than sparse search.  Run
    # them across every source first so cross-source conflicts remain visible.
    for index in indexes:
        for row_index, (row, query_key) in enumerate(zip(rows, normalized_queries, strict=True)):
            if row["example_id"] in skip_ids or row["example_id"] in disabled_contextual_ids:
                continue
            for rank, source_index in enumerate(index["exact_lookup"].get(query_key, []), start=1):
                candidates_by_row[row_index].append(retrieval_candidate(
                    row, query_key, index, source_index, rank=rank, score=1.0
                ))

    exact_terminal_rows = {
        row_index
        for row_index, candidates in enumerate(candidates_by_row)
        if any(candidate["source_verdict_candidate"] is not None for candidate in candidates)
    }
    fuzzy_row_indices = [
        index for index, row in enumerate(rows)
        if row["example_id"] not in skip_ids
        and row["example_id"] not in disabled_contextual_ids
        and index not in exact_terminal_rows
    ]
    query_to_rows: dict[str, list[int]] = {}
    for row_index in fuzzy_row_indices:
        query_to_rows.setdefault(normalized_queries[row_index], []).append(row_index)
    unique_queries = list(query_to_rows)

    for index in indexes:
        source_started = time.perf_counter()
        vectorizer = index["vectorizer"]
        matrix = index["matrix"]
        # Joykoli OCR passages are deliberately gated more tightly than other
        # sources.  Search a wider sparse pool, then retain at most ``top_k``
        # candidates, preferring eligible evidence.  This lets a page with an
        # OCR-damaged anchor displace high-scoring but wrong-entity questions
        # without growing the downstream prompt/evidence budget.
        pool_k = min(
            matrix.shape[0],
            top_k * 4 if index["source_id"] == "joykoli_six_part" else top_k,
        )
        for start in range(0, len(unique_queries), batch_size):
            batch_queries = unique_queries[start : start + batch_size]
            query_matrix = vectorizer.transform(batch_queries)
            # Keep the product sparse.  Materializing batch x corpus dense
            # arrays was the dominant memory cost for 5,000-row inference.
            similarities = (query_matrix @ matrix.T).tocsr()
            for offset, query_key in enumerate(batch_queries):
                left, right = similarities.indptr[offset : offset + 2]
                source_indices = similarities.indices[left:right]
                scores = similarities.data[left:right]
                count = min(pool_k, len(scores))
                if count == 0:
                    continue
                selected = np.argpartition(scores, -count)[-count:]
                selected = selected[np.argsort(scores[selected])[::-1]]
                top = [
                    (int(source_indices[position]), float(scores[position]))
                    for position in selected
                ]
                for row_index in query_to_rows[query_key]:
                    existing = {
                        (candidate["source_id"], candidate["source_record_index"])
                        for candidate in candidates_by_row[row_index]
                    }
                    ranked_new: list[dict[str, Any]] = []
                    rank = 0
                    for source_index, score in top:
                        if (index["source_id"], source_index) in existing:
                            continue
                        rank += 1
                        ranked_new.append(retrieval_candidate(
                            rows[row_index], query_key, index, source_index,
                            rank=rank, score=score,
                        ))
                    if index["source_id"] == "joykoli_six_part":
                        eligible = [
                            candidate for candidate in ranked_new
                            if candidate.get("model_facing_eligible") is True
                        ][:top_k]
                        if len(eligible) < top_k:
                            eligible.extend([
                                candidate for candidate in ranked_new
                                if candidate.get("model_facing_eligible") is not True
                            ][:top_k - len(eligible)])
                        ranked_new = sorted(eligible, key=lambda candidate: candidate["rank"])
                    candidates_by_row[row_index].extend(ranked_new)
        index_seconds[index["source_id"]] = time.perf_counter() - source_started

    # The composite cache is closed-book-only.  The default evidence-heavy
    # mode queries every closed row so counter-evidence is retained.  The
    # selective mode queries only rows without an exact terminal source
    # candidate.  A separately measured residual-only ablation may skip rows
    # that already have eligible mmap/book/dataset evidence.  Contextual rows
    # never enter any path.
    composite_query_row_indices: list[int] = []
    composite_selection_audits: dict[str, dict[str, Any]] = {}
    composite_fuzzy_query_count = 0
    composite_exact_only_query_count = 0
    if composite_cache_dir is not None:
        composite_started = time.perf_counter()
        composite_manifest, composite_connection = load_composite_fts(
            composite_cache_dir.resolve()
        )
        try:
            composite_query_row_indices = select_composite_query_rows(
                rows,
                candidates_by_row,
                skip_example_ids=skip_ids,
                mode=composite_query_mode,
            )
            raw_query_cache: dict[tuple[str, bool], list[dict[str, Any]]] = {}
            for row_index in composite_query_row_indices:
                row = rows[row_index]
                query = str(row["model_prompt_bn"])
                # Fuzzy composite candidates are never counter-evidence: the
                # counter contract requires an exact normalized question with
                # a keyed answer.  Rows already carrying an exact terminal
                # source candidate therefore need only the cheap composite
                # exact lookup; unresolved rows retain the complete bounded
                # fuzzy path.  This preserves counter discovery while avoiding
                # a redundant six-source FTS scan on already-resolved rows.
                fuzzy_search = not (
                    composite_query_mode == "all_closed"
                    and row_index in exact_terminal_rows
                )
                cache_key = (query, fuzzy_search)
                if cache_key not in raw_query_cache:
                    raw_query_cache[cache_key] = retrieve_authority_tier_pool_optimized(
                        composite_connection,
                        query,
                        per_authority_tier_k=top_k,
                        fuzzy_search=fuzzy_search,
                    )
                if fuzzy_search:
                    composite_fuzzy_query_count += 1
                else:
                    composite_exact_only_query_count += 1
                adapted_pool = [
                    composite_retrieval_candidate(row, value, rank=rank)
                    for rank, value in enumerate(raw_query_cache[cache_key], start=1)
                ]
                selected, selection_audit = rank_and_bound_composite_candidates(
                    adapted_pool, top_k=top_k
                )
                composite_selection_audits[str(row["example_id"])] = {
                    **selection_audit,
                    "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                }
                candidates_by_row[row_index].extend(selected)
        finally:
            composite_connection.close()
        index_seconds["phase2_composite_fts"] = time.perf_counter() - composite_started

    outputs: list[dict[str, Any]] = []
    terminal_counts: Counter[str] = Counter()
    raw_candidate_count = 0
    model_facing_candidate_count = 0
    quarantined_candidate_count = 0
    quarantine_reason_counts: Counter[str] = Counter()
    for row, candidates in zip(rows, candidates_by_row, strict=True):
        candidates = authority_rank_candidates(candidates)
        model_facing_candidates = [
            candidate for candidate in candidates
            if candidate.get("model_facing_eligible") is True
        ]
        quarantined_candidates = [
            candidate for candidate in candidates
            if candidate.get("model_facing_eligible") is not True
        ]
        raw_candidate_count += len(candidates)
        model_facing_candidate_count += len(model_facing_candidates)
        quarantined_candidate_count += len(quarantined_candidates)
        for candidate in quarantined_candidates:
            quarantine_reason_counts.update(
                (candidate.get("model_facing_gate") or {}).get("reasons", [])
            )
        exact_candidates = [
            {"source_id": candidate["source_id"], **candidate["source_verdict_candidate"]}
            for candidate in candidates
            if candidate["source_verdict_candidate"] is not None
        ]
        verdicts = {int(candidate["verdict"]) for candidate in exact_candidates}
        exact_key_conflicts = strict_exact_key_conflicts(candidates)
        if row["example_id"] in skip_ids:
            merged = {
                "verdict": None,
                "status": "skipped_terminal_context",
                "evidence": [],
            }
        elif row["example_id"] in disabled_contextual_ids:
            merged = {
                "verdict": None,
                "status": "contextual_external_retrieval_disabled",
                "evidence": [],
            }
        elif len(verdicts) > 1 or exact_key_conflicts:
            merged = {
                "verdict": None,
                "status": "source_conflict_quarantined",
                "evidence": exact_candidates,
                "strict_exact_key_conflicts": exact_key_conflicts,
            }
        elif len(verdicts) == 1:
            merged = {
                "verdict": verdicts.pop(),
                "status": "source_consensus_candidate",
                "evidence": exact_candidates,
            }
        else:
            merged = {"verdict": None, "status": "no_terminal_source_candidate", "evidence": []}
        terminal_counts[merged["status"]] += 1
        outputs.append({
            "example_id": row["example_id"],
            "source_index": int(row.get("source_index", len(outputs))),
            "formatting_status": (row.get("formatting_provenance") or {}).get("status", "unknown"),
            "query_field": "model_prompt_bn",
            "query_sha256": hashlib.sha256(str(row["model_prompt_bn"]).encode("utf-8")).hexdigest(),
            "response_sha256": hashlib.sha256(str(row["model_response_bn"]).encode("utf-8")).hexdigest(),
            "retrieval_candidates": model_facing_candidates,
            "retrieval_audit_quarantined_candidates": quarantined_candidates,
            "raw_retrieval_candidate_count": len(candidates),
            "model_facing_retrieval_candidate_count": len(model_facing_candidates),
            "merged_source_candidate": merged,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "retrieval.jsonl"
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in outputs:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    closed_book_eval_ledger = write_closed_book_eval_evidence_ledger(
        rows, outputs, output_dir
    )
    elapsed = time.perf_counter() - started
    implementation = {
        "sparse_retrieval_sha256": sha256_file(Path(__file__)),
        "canonicalizer_sha256": sha256_file(
            ROOT / "pipeline/phase2_canonicalize.py"
        ),
        "mmap_retrieval_runtime_sha256": sha256_file(
            ROOT / "pipeline/phase2_mmap_retrieval.py"
        ),
        "composite_fts_runtime_sha256": sha256_file(
            ROOT / "pipeline/phase2_composite_fts_retrieval.py"
        ),
        "response_proposition_sha256": sha256_file(
            ROOT / "pipeline/phase2_response_proposition.py"
        ),
        "source_authority_runtime_sha256": sha256_file(
            ROOT / "pipeline/phase2_source_authority.py"
        ),
        "source_authority_policy_sha256": SOURCE_AUTHORITY_POLICY_SHA256,
    }
    manifest = {
        "version": VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records": len(rows),
        "top_k_per_source": top_k,
        "batch_size": batch_size,
        "unique_normalized_queries": len(set(normalized_queries)),
        "fuzzy_unique_queries": len(unique_queries),
        "fuzzy_rows": len(fuzzy_row_indices),
        "exact_terminal_rows_skipped_from_fuzzy": len(exact_terminal_rows),
        "dense_similarity_materialized": False,
        "preterminal_skip": {
            "reason": "terminal_deterministic_context_grammar_or_math",
            "count": len(skip_ids),
            "example_ids_sha256": id_set_sha256(skip_ids),
        },
        "contextual_external_retrieval": {
            "enabled": False,
            "enabled_for_typed_rule_theory_scope_only": False,
            "ordinary_context_enabled": False,
            "terminal_policy": "all supplied-context rows are context-only; closed-book retrieval is separate",
            "world_knowledge_rescue_allowed": False,
            "external_evidence_allowed": False,
            "internal_context_reasoning_allowed": True,
            "retrieval_lane_policy_sha256": RETRIEVAL_LANE_POLICY_SHA256,
            "count": len(contextual_ids),
            "example_ids_sha256": id_set_sha256(contextual_ids),
            "typed_rule_theory_lookup_count": len(context_lookup_ids),
            "typed_rule_theory_lookup_ids_sha256": id_set_sha256(context_lookup_ids),
            "disabled_ordinary_context_count": len(disabled_contextual_ids),
            "disabled_ordinary_context_ids_sha256": id_set_sha256(disabled_contextual_ids),
        },
        "implementation_sha256": implementation["sparse_retrieval_sha256"],
        "implementation": implementation,
        "source_authority_policy": {
            "path": str(SOURCE_AUTHORITY_POLICY_PATH),
            "sha256": SOURCE_AUTHORITY_POLICY_SHA256,
            "decision_order": [
                "semantic_alignment", "authority_tier", "exactness",
                "within_source_rank", "deterministic_identity",
            ],
            "books_user_ocr_and_curated_data_before_wikipedia_when_equally_aligned": True,
            "wikipedia_corroboration_only": True,
        },
        "labels_read": False,
        "fuzzy_retrieval_terminal_labels": False,
        "closed_book_eval_evidence_ledger": closed_book_eval_ledger,
        "raw_retrieval_candidate_count": raw_candidate_count,
        "model_facing_retrieval_candidate_count": model_facing_candidate_count,
        "quarantined_retrieval_candidate_count": quarantined_candidate_count,
        "quarantine_reason_counts": dict(sorted(quarantine_reason_counts.items())),
        "model_facing_fuzzy_policy": (
            "score>=0.22 plus substantive overlap, exact number/negation sets, "
            "compatible answer-type intent; nonterminal page passages require score>=0.20, "
            "at least two substantive overlaps, query-token coverage, and query number/negation "
            "containment; Joykoli non-exact evidence additionally requires its primary "
            "query subject/entity anchor with bounded OCR tolerance and uses 4x sparse "
            "candidate overfetch before retaining top_k; short keyword-only queries abstain"
        ),
        "terminal_candidate_status_counts": dict(sorted(terminal_counts.items())),
        "runtime_seconds": elapsed,
        "rows_per_second": len(rows) / elapsed if elapsed else None,
        "per_index_seconds": index_seconds,
        "input": {"path": str(input_path), "sha256": sha256_file(input_path)},
        "indexes": [
            {
                "source_id": index["source_id"],
                "manifest_sha256": index["manifest_sha256"],
                "matrix_shape": list(index["matrix"].shape),
                "cache_format": index.get("cache_format", "legacy_npz_jsonl_joblib"),
                "cache_id": index.get("cache_id", ""),
                "rights_note": index["rights_note"],
                "terminal_policy": index["terminal_policy"],
            }
            for index in indexes
        ],
        "composite_fts": (
            {
                "enabled": True,
                "runtime_version": COMPOSITE_FTS_RUNTIME_VERSION,
                "runtime_sha256": sha256_file(
                    ROOT / "pipeline/phase2_composite_fts_retrieval.py"
                ),
                "cache_dir": str(composite_cache_dir.resolve()),
                "cache_id": composite_manifest["cache_id"],
                "manifest_sha256": sha256_file(
                    composite_cache_dir.resolve() / "manifest.json"
                ),
                "database_sha256": composite_manifest["database"]["sha256"],
                "closed_book_only": True,
                "terminal_label_authority": False,
                "query_mode": composite_query_mode,
                "closed_rows_considered": sum(
                    row["example_id"] not in skip_ids
                    and not bool(row.get("context_available"))
                    for row in rows
                ),
                "rows_queried": len(composite_query_row_indices),
                "candidate_pool_policy": (
                    "exact_all_closed_then_bounded_fuzzy_only_without_existing_"
                    "terminal_candidate_then_semantic_authority_final_top_k"
                ),
                "exact_only_query_count": composite_exact_only_query_count,
                "bounded_fuzzy_query_count": composite_fuzzy_query_count,
                "fuzzy_counter_candidate_possible": False,
                "exact_counter_lookup_preserved_for_all_queried_rows": True,
                "per_authority_tier_pool_k": top_k,
                "final_top_k": top_k,
                "unique_query_count": len(composite_selection_audits),
                "pool_candidates": sum(
                    int(row["pool_candidates"])
                    for row in composite_selection_audits.values()
                ),
                "eligible_candidates_before_final_top_k": sum(
                    int(row["eligible_candidates"])
                    for row in composite_selection_audits.values()
                ),
                "quarantined_candidates_before_final_top_k": sum(
                    int(row["quarantined_candidates"])
                    for row in composite_selection_audits.values()
                ),
                "selected_candidates_after_final_top_k": sum(
                    int(row["selected_candidates"])
                    for row in composite_selection_audits.values()
                ),
                "queries_with_counter_available": sum(
                    row["counter_available"] is True
                    for row in composite_selection_audits.values()
                ),
                "queries_with_counter_selected": sum(
                    row["counter_selected"] is True
                    for row in composite_selection_audits.values()
                ),
                "queries_with_counter_forced_into_top_k": sum(
                    row["counter_forced_into_top_k"] is True
                    for row in composite_selection_audits.values()
                ),
                "rows_skipped_by_existing_eligible_evidence": (
                    sum(
                        row["example_id"] not in skip_ids
                        and not bool(row.get("context_available"))
                        for row in rows
                    )
                    - len(composite_query_row_indices)
                ),
                "rows_skipped_by_precomposite_policy": (
                    sum(
                        row["example_id"] not in skip_ids
                        and not bool(row.get("context_available"))
                        for row in rows
                    )
                    - len(composite_query_row_indices)
                ),
            }
            if composite_manifest is not None and composite_cache_dir is not None
            else {"enabled": False}
        ),
        "output": {"path": str(output_path), "sha256": sha256_file(output_path), "bytes": output_path.stat().st_size},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for index in indexes:
        if index.get("cache_format") == MMAP_CACHE_VERSION:
            close_mmap_index(index)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context-grounding", type=Path)
    parser.add_argument("--composite-cache", type=Path)
    parser.add_argument(
        "--composite-query-mode",
        choices=sorted(COMPOSITE_QUERY_MODES),
        default="all_closed",
        help=(
            "all_closed preserves the heavy counter-evidence path; "
            "unresolved_only preserves it for rows without a terminal source; "
            "residual_only is a separately benchmarked latency ablation"
        ),
    )
    args = parser.parse_args()
    skip_ids = load_terminal_context_ids(args.context_grounding) if args.context_grounding else None
    context_lookup_ids = (
        load_context_external_lookup_ids(args.context_grounding)
        if args.context_grounding else None
    )
    print(json.dumps(build(
        args.input,
        args.output_dir,
        top_k=args.top_k,
        batch_size=args.batch_size,
        skip_example_ids=skip_ids,
        context_external_lookup_ids=context_lookup_ids,
        composite_cache_dir=args.composite_cache,
        composite_query_mode=args.composite_query_mode,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
