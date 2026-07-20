"""Runtime query API for the nonterminal Phase 2 composite FTS cache."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.phase2_canonicalize import canonicalize
from pipeline.phase2_source_authority import authority_for


DEFAULT_CACHE = ROOT / "artifacts/retrieval/phase2_composite_fts_v2"
VERSION = "phase2-composite-fts-runtime-v5-bounded-source-scan-exact-mode"
SUPPORTED_CACHE_VERSIONS = {
    "phase2-composite-fts-cache-v2-nctbench-nonterminal",
    "phase2-strict-rights-safe-fts-v3-nonterminal",
}
QUERY_TOKEN_RE = re.compile(r"[\w\u0980-\u09ff]+", re.UNICODE)
QUERY_STOPWORDS = {
    "কি", "কী", "কে", "কার", "কোন", "কোনটি", "কত", "কবে", "কোথায়",
    "কোথায়", "কিভাবে", "কীভাবে", "কেন", "হয়", "হয়", "ছিল", "আছে",
    "একটি", "এই", "ও", "এবং", "the", "a", "an", "is", "was", "what",
    "which", "who", "where", "when", "how",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(cache_dir: Path) -> tuple[dict[str, Any], sqlite3.Connection]:
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") not in SUPPORTED_CACHE_VERSIONS:
        raise ValueError("composite cache version mismatch")
    policy = manifest.get("policy") or {}
    if (
        policy.get("closed_book_only") is not True
        or policy.get("contextual_external_retrieval_allowed") is not False
        or policy.get("retrieval_is_terminal") is not False
        or policy.get("retrieval_miss_means") != "NEI"
    ):
        raise ValueError("unsafe composite cache policy")
    database = cache_dir / manifest["database"]["path"]
    if (
        database.stat().st_size != int(manifest["database"]["bytes"])
        or sha256_file(database) != manifest["database"]["sha256"]
    ):
        raise ValueError("composite cache database hash/size mismatch")
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return manifest, connection


def query_tokens(text: str, *, max_terms: int = 8) -> list[str]:
    tokens = []
    for token in QUERY_TOKEN_RE.findall(canonicalize(text)):
        if len(token) < 2 or token in QUERY_STOPWORDS or token in tokens:
            continue
        tokens.append(token)
        if len(tokens) >= max_terms:
            break
    return tokens


def quote_fts(token: str) -> str:
    return f'"{token.replace(chr(34), chr(34) * 2)}"'


def fts_query(text: str, *, max_terms: int = 8) -> str:
    """Strict query: require every admitted content token (bounded to three)."""

    tokens = query_tokens(text, max_terms=max_terms)
    if not tokens:
        return ""
    if len(tokens) == 1:
        return quote_fts(tokens[0])
    strict_tokens = tokens if len(tokens) <= 3 else [tokens[0], tokens[-2], tokens[-1]]
    return " AND ".join(quote_fts(token) for token in strict_tokens)


def relaxed_fts_query(text: str, *, max_terms: int = 8) -> str:
    """Fallback requiring content-token pairs, never singleton OR terms."""

    tokens = query_tokens(text, max_terms=max_terms)
    if not tokens:
        return ""
    if len(tokens) == 1:
        return quote_fts(tokens[0])
    if len(tokens) <= 4:
        # Pair relaxation tolerates one OCR/spelling mismatch while preventing a
        # single generic word such as "first" from admitting a candidate.
        pairs = [
            f"({quote_fts(left)} AND {quote_fts(right)})"
            for left, right in itertools.combinations(tokens, 2)
        ]
        return " OR ".join(pairs)
    # Query construction must not grow quadratically with long prompts.
    return " AND ".join(quote_fts(token) for token in tokens[:3])


def exact_question_variants(text: str) -> list[str]:
    normalized = canonicalize(text)
    # Bengali interrogative কি/কী varies in otherwise exact source questions.
    # Keep this local to question-key lookup; it must not rewrite evidence text.
    variants = [normalized]
    for old, new in (("কী", "কি"), ("কি", "কী")):
        candidate = re.sub(rf"(?<!\w){old}(?!\w)", new, normalized)
        if candidate not in variants:
            variants.append(candidate)
    return variants


def excerpt(text: str, query: str, *, limit: int = 900) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    tokens = [token for token in QUERY_TOKEN_RE.findall(canonicalize(query)) if len(token) >= 3]
    # ``compact`` is source display text, not a comparison key.  Case folding
    # is sufficient to place the excerpt around an already-canonical query
    # term and avoids repeatedly normalizing an entire long passage.
    lowered = compact.casefold()
    positions = [lowered.find(token) for token in tokens if lowered.find(token) >= 0]
    center = min(positions) if positions else 0
    start = max(0, center - limit // 3)
    return compact[start : start + limit]


def _lexical_display_score(
    query_terms: list[str], row: sqlite3.Row,
) -> float:
    """Return a deterministic bounded score for the existing semantic gates.

    FTS5's weighted BM25 is useful for ordering, but asking SQLite to score and
    sort every match in the 1.7 GB composite is the dominant runtime cost.  The
    optimized path instead asks FTS only for a small deterministic rowid pool,
    then scores the exact text that the downstream gate will inspect.  This is
    deliberately not a verdict and cannot bypass overlap, number, negation or
    answer-type checks.
    """

    if not query_terms:
        return 0.0
    question = str(row["question"] or "")
    candidate_text = question or "\n".join(
        str(row[field] or "") for field in ("title", "supporting_text")
    )
    # FTS5 has already applied the strict/relaxed Unicode token match.  A
    # second full symbol-safe canonicalization of every scanned passage was
    # pure duplicate work (and the largest remaining Python cost).  Tokenize
    # the source text directly for this bounded display score; number and
    # negation normalization remain enforced by the downstream semantic gate.
    candidate_terms = set(QUERY_TOKEN_RE.findall(candidate_text.casefold()))
    shared = [term for term in query_terms if term in candidate_terms]
    coverage = len(shared) / len(query_terms)
    # Preserve the pre-existing gate's explicit long distinctive-token
    # exception.  The gate itself still decides eligibility.
    if len(shared) == 1 and len(shared[0]) >= 8:
        coverage = max(coverage, 0.55)
    return round(min(1.0, coverage), 8)


def _row_to_result(
    row: sqlite3.Row,
    query: str,
    *,
    exact_question: bool,
    retrieval_score: float,
) -> dict[str, Any]:
    supporting_text = str(row["supporting_text"])
    return {
        "source_record_index": int(row["id"]),
        "source_id": row["source_id"],
        "source_locator": row["source_locator"],
        "semantic_role": row["semantic_role"],
        "model_facing": bool(row["model_facing"]),
        "question": row["question"],
        "answer": row["answer"],
        "title": row["title"],
        "evidence_excerpt": excerpt(supporting_text, query),
        "choices": json.loads(row["choices_json"]),
        "metadata": json.loads(row["metadata_json"]),
        "record_sha256": row["record_sha256"],
        "exact_question": exact_question,
        "retrieval_score": retrieval_score,
        "terminal_label_authority": False,
        "verdict": "NEI",
    }


def retrieve(
    connection: sqlite3.Connection,
    query: str,
    *,
    top_k: int = 20,
    model_facing_only: bool = True,
    source_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    normalized_variants = exact_question_variants(query)
    exact_clauses = [
        "normalized_question IN (%s)" % ",".join("?" for _ in normalized_variants),
        "(?=0 OR model_facing=1)",
    ]
    exact_parameters: list[Any] = [*normalized_variants, int(model_facing_only)]
    if source_ids:
        exact_clauses.append("source_id IN (%s)" % ",".join("?" for _ in source_ids))
        exact_parameters.extend(source_ids)
    exact_parameters.append(top_k)
    exact_rows = connection.execute(
        "SELECT *, 1 AS exact_question, -1000.0 AS rank FROM evidence WHERE "
        + " AND ".join(exact_clauses)
        + " ORDER BY source_id, source_locator, id LIMIT ?",
        exact_parameters,
    ).fetchall()
    seen = {int(row["id"]) for row in exact_rows}
    rows = list(exact_rows)
    remaining = max(0, top_k - len(rows))
    # An exact question landing is sufficient evidence retrieval.  Broad
    # top-up after exact matches adds latency and unrelated collisions.
    if remaining and not exact_rows:
        expressions = [fts_query(query)]
        relaxed = relaxed_fts_query(query)
        if relaxed and relaxed not in expressions:
            expressions.append(relaxed)
        for expression in expressions:
            if not expression:
                continue
            clauses = ["evidence_fts MATCH ?", "(?=0 OR e.model_facing=1)"]
            parameters: list[Any] = [expression, int(model_facing_only)]
            if source_ids:
                clauses.append("e.source_id IN (%s)" % ",".join("?" for _ in source_ids))
                parameters.extend(source_ids)
            parameters.append(remaining)
            fuzzy = connection.execute(
                "SELECT e.*, 0 AS exact_question, bm25(evidence_fts, 3.0, 1.5, 2.0, 1.0) AS rank "
                "FROM evidence_fts JOIN evidence e ON e.id=evidence_fts.rowid WHERE "
                + " AND ".join(clauses)
                + " ORDER BY rank, e.source_id, e.source_locator, e.id LIMIT ?",
                parameters,
            ).fetchall()
            if fuzzy:
                rows.extend(fuzzy)
                break
    return [
        _row_to_result(
            row,
            query,
            exact_question=bool(row["exact_question"]),
            retrieval_score=float(row["rank"]),
        )
        for row in rows[:top_k]
    ]


def model_facing_source_ids(connection: sqlite3.Connection) -> list[str]:
    """Return the immutable source universe admitted by the built cache.

    The cache builder is the rights/admission boundary.  This lookup cannot
    enable a source that was not already stored as model-facing evidence.
    """

    return [
        str(row[0])
        for row in connection.execute(
            "SELECT DISTINCT source_id FROM evidence "
            "WHERE model_facing=1 ORDER BY source_id"
        ).fetchall()
    ]


_SOURCE_ROUTING_CACHE: dict[
    sqlite3.Connection, dict[str, dict[str, Any]]
] = {}


def _source_routing_profile(
    connection: sqlite3.Connection,
) -> dict[str, dict[str, Any]]:
    """Describe admitted source rowid intervals once per open connection."""

    cached = _SOURCE_ROUTING_CACHE.get(connection)
    if cached is not None:
        return cached
    profile: dict[str, dict[str, Any]] = {}
    rows = connection.execute(
        "SELECT source_id, MIN(id), MAX(id), COUNT(*) FROM evidence "
        "WHERE model_facing=1 GROUP BY source_id ORDER BY source_id"
    ).fetchall()
    for source_id, minimum, maximum, count in rows:
        minimum, maximum, count = int(minimum), int(maximum), int(count)
        profile[str(source_id)] = {
            "minimum_rowid": minimum,
            "maximum_rowid": maximum,
            "record_count": count,
            "contiguous_rowids": maximum - minimum + 1 == count,
        }
    _SOURCE_ROUTING_CACHE[connection] = profile
    return profile


def _bounded_source_scan(
    connection: sqlite3.Connection,
    expression: str,
    *,
    source_id: str,
    scan_limit: int,
    routing: dict[str, dict[str, Any]],
) -> list[int]:
    """Return a deterministic bounded FTS rowid pool for one admitted source."""

    source = routing[source_id]
    if source["contiguous_rowids"]:
        rows = connection.execute(
            "SELECT rowid FROM evidence_fts WHERE evidence_fts MATCH ? "
            "AND rowid BETWEEN ? AND ? ORDER BY rowid LIMIT ?",
            (
                expression,
                source["minimum_rowid"],
                source["maximum_rowid"],
                scan_limit,
            ),
        ).fetchall()
    else:
        # Fail-safe for a future cache whose source rows are interleaved.  The
        # current v2 cache is contiguous, but correctness must not depend on
        # undocumented insertion order.
        rows = connection.execute(
            "SELECT evidence_fts.rowid FROM evidence_fts "
            "JOIN evidence e ON e.id=evidence_fts.rowid "
            "WHERE evidence_fts MATCH ? AND e.model_facing=1 "
            "AND e.source_id=? ORDER BY evidence_fts.rowid LIMIT ?",
            (expression, source_id, scan_limit),
        ).fetchall()
    return [int(row[0]) for row in rows]


def retrieve_authority_tier_pool_optimized(
    connection: sqlite3.Connection,
    query: str,
    *,
    per_authority_tier_k: int = 20,
    source_scan_multiplier: int = 4,
    source_ids: list[str] | None = None,
    fuzzy_search: bool = True,
) -> list[dict[str, Any]]:
    """Fast authority-before-top-k retrieval with bounded source pools.

    Exact-question lookup is unchanged.  Fuzzy lookup scans a small,
    deterministic rowid pool independently for every admitted source, thereby
    preventing a large source from starving a book/curated source without the
    expensive global BM25 sort.  The existing caller still performs all
    semantic gates, authority ordering, final top-k and counter retention.
    """

    if not 1 <= per_authority_tier_k <= 100:
        raise ValueError("per_authority_tier_k must be 1..100")
    if not 1 <= source_scan_multiplier <= 16:
        raise ValueError("source_scan_multiplier must be 1..16")
    routing = _source_routing_profile(connection)
    admitted = set(routing)
    selected_sources = (
        sorted(admitted)
        if source_ids is None
        else sorted({str(source_id) for source_id in source_ids})
    )
    unknown = sorted(set(selected_sources) - admitted)
    if unknown:
        raise ValueError(
            f"bounded source pool requested unadmitted sources: {unknown}"
        )
    sources_by_tier: dict[int, list[str]] = {}
    for source_id in selected_sources:
        tier = int(authority_for(source_id)["authority_tier"])
        sources_by_tier.setdefault(tier, []).append(source_id)

    variants = exact_question_variants(query)
    exact_parameters: list[Any] = [*variants]
    clauses = [
        "normalized_question IN (%s)" % ",".join("?" for _ in variants),
        "model_facing=1",
    ]
    if selected_sources:
        clauses.append(
            "source_id IN (%s)" % ",".join("?" for _ in selected_sources)
        )
        exact_parameters.extend(selected_sources)
    exact_rows = connection.execute(
        "SELECT * FROM evidence WHERE " + " AND ".join(clauses)
        + " ORDER BY source_id, source_locator, id",
        exact_parameters,
    ).fetchall()
    exact_by_tier: dict[int, list[sqlite3.Row]] = {}
    for row in exact_rows:
        tier = int(authority_for(row["source_id"])["authority_tier"])
        exact_by_tier.setdefault(tier, []).append(row)

    scan_limit = max(16, per_authority_tier_k * source_scan_multiplier)
    strict = fts_query(query) if fuzzy_search else ""
    relaxed = relaxed_fts_query(query) if fuzzy_search else ""
    score_query_terms = query_tokens(query)
    fuzzy_rowids_by_tier: dict[int, list[int]] = {}
    for tier, tier_sources in sorted(sources_by_tier.items()):
        if exact_by_tier.get(tier) or not strict:
            continue
        rowids: list[int] = []
        for source_id in tier_sources:
            rowids.extend(_bounded_source_scan(
                connection,
                strict,
                source_id=source_id,
                scan_limit=scan_limit,
                routing=routing,
            ))
        if not rowids and relaxed and relaxed != strict:
            for source_id in tier_sources:
                rowids.extend(_bounded_source_scan(
                    connection,
                    relaxed,
                    source_id=source_id,
                    scan_limit=scan_limit,
                    routing=routing,
                ))
        fuzzy_rowids_by_tier[tier] = sorted(set(rowids))

    all_fuzzy_rowids = sorted({
        rowid for rowids in fuzzy_rowids_by_tier.values() for rowid in rowids
    })
    fuzzy_by_id: dict[int, sqlite3.Row] = {}
    # SQLite's default variable limit comfortably exceeds the current bounded
    # maximum (six admitted sources * 100 * 16), but keep batched reads robust.
    for offset in range(0, len(all_fuzzy_rowids), 500):
        batch = all_fuzzy_rowids[offset : offset + 500]
        if not batch:
            continue
        rows = connection.execute(
            "SELECT * FROM evidence WHERE id IN (%s)" % ",".join("?" for _ in batch),
            batch,
        ).fetchall()
        fuzzy_by_id.update({int(row["id"]): row for row in rows})

    pool: list[dict[str, Any]] = []
    for tier in sorted(sources_by_tier):
        if exact_by_tier.get(tier):
            tier_results = [
                _row_to_result(
                    row,
                    query,
                    exact_question=True,
                    retrieval_score=-1000.0,
                )
                for row in exact_by_tier[tier][:per_authority_tier_k]
            ]
        else:
            scored: list[tuple[tuple[Any, ...], sqlite3.Row, float]] = []
            for rowid in fuzzy_rowids_by_tier.get(tier, []):
                row = fuzzy_by_id.get(rowid)
                if row is None:
                    continue
                score = _lexical_display_score(score_query_terms, row)
                key = (
                    -score,
                    str(row["source_id"]),
                    str(row["source_locator"]),
                    int(row["id"]),
                    str(row["record_sha256"]),
                )
                scored.append((key, row, score))
            scored.sort(key=lambda value: value[0])
            tier_results = [
                _row_to_result(
                    row,
                    query,
                    exact_question=False,
                    retrieval_score=-score,
                )
                for _, row, score in scored[:per_authority_tier_k]
            ]
        for within_tier_rank, result in enumerate(tier_results, start=1):
            pool.append({
                **result,
                "authority_pool_tier": tier,
                "within_authority_tier_rank": within_tier_rank,
                "retrieval_score_kind": (
                    "exact_question_sentinel"
                    if result["exact_question"]
                    else "bounded_query_token_coverage"
                ),
            })
    return sorted(
        pool,
        key=lambda row: (
            int(row["authority_pool_tier"]),
            int(row["within_authority_tier_rank"]),
            str(row["source_id"]),
            int(row["source_record_index"]),
            str(row["record_sha256"]),
        ),
    )


def retrieve_authority_tier_pool(
    connection: sqlite3.Connection,
    query: str,
    *,
    per_authority_tier_k: int = 20,
    source_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve a bounded candidate pool per authority tier before top-k.

    A single global SQL ``LIMIT`` can allow a large source to consume the
    whole pool before books/user OCR/curated-data authority is evaluated.
    Querying every already-admitted authority tier independently retains up to
    ``per_authority_tier_k`` candidates from books, curated data, default
    authorized sources, and Wikipedia separately.  The caller must run the
    semantic gates and shared authority ranking before final top-k.
    """

    if not 1 <= per_authority_tier_k <= 100:
        raise ValueError("per_authority_tier_k must be 1..100")
    admitted = set(model_facing_source_ids(connection))
    selected_sources = (
        sorted(admitted)
        if source_ids is None
        else sorted({str(source_id) for source_id in source_ids})
    )
    unknown = sorted(set(selected_sources) - admitted)
    if unknown:
        raise ValueError(f"source-diverse pool requested unadmitted sources: {unknown}")
    sources_by_tier: dict[int, list[str]] = {}
    for source_id in selected_sources:
        tier = int(authority_for(source_id)["authority_tier"])
        sources_by_tier.setdefault(tier, []).append(source_id)
    pool: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for authority_tier, tier_sources in sorted(sources_by_tier.items()):
        rows = retrieve(
            connection,
            query,
            top_k=per_authority_tier_k,
            model_facing_only=True,
            source_ids=sorted(tier_sources),
        )
        for within_tier_rank, row in enumerate(rows, start=1):
            identity = (str(row["source_id"]), int(row["source_record_index"]))
            if identity in seen:
                continue
            seen.add(identity)
            pool.append({
                **row,
                "authority_pool_tier": authority_tier,
                "within_authority_tier_rank": within_tier_rank,
            })
    return sorted(
        pool,
        key=lambda row: (
            int(row["authority_pool_tier"]),
            int(row["within_authority_tier_rank"]),
            str(row["source_id"]),
            int(row["source_record_index"]),
            str(row["record_sha256"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--include-components", action="store_true")
    args = parser.parse_args()
    _, connection = load(args.cache.resolve())
    try:
        print(json.dumps(retrieve(
            connection,
            args.query,
            top_k=args.top_k,
            model_facing_only=not args.include_components,
        ), ensure_ascii=False, indent=2))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
