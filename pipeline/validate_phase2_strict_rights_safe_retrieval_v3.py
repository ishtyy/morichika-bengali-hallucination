"""Focused fail-closed validation for the strict retrieval-v3 local package."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.phase2_composite_fts_retrieval import load as load_fts, retrieve


PACKAGE = ROOT / "artifacts/kaggle/morichika_phase2_retrieval_strict_v3_20260720"
OUTPUT = ROOT / "artifacts/status/phase2_strict_rights_safe_retrieval_v3_20260720/verification.json"
EXPECTED_FTS = {
    "openai_mmmlu_bn": 14042,
    "nctb_passage_qa_mendeley_v1": 2802,
    "uddipok_v3": 3797,
    "bangla_wordnet": 29733,
}
EXPECTED_MMAP = {
    "bangla_mmlu": ("bangla_mmlu_char_v1_mmap_v2", 85018),
    "nctb_qa_87805": ("nctb_qa_87805_char_v1_mmap_v2", 47945),
    "nctb_education_aux": ("nctb_education_aux_char_v1_mmap_v2", 25890),
    "downloads_bcs_10_50": ("downloads_bcs_10_50_char_v1_mmap_v2", 5309),
    "nctb_schooltext": ("nctb_schooltext_word_v1_mmap_v2", 58872),
    "joykoli_six_part": ("joykoli_six_part_char_v3_mmap_v2", 14719),
}
FORBIDDEN_SOURCE_IDS = {
    "squad_bn", "benhallueval", "bangla_mmlu", "joykoli_six_part",
    "downloads_bcs_10_50", "nctb_education_aux", "nctb_qa_87805",
    "current_affairs", "bdlaws_targeted_phase1_audit", "bengali_wikipedia_20210320",
}
REQUIRED_RUNTIME_FILES = {
    "pipeline/phase2_sparse_retrieval.py",
    "pipeline/phase2_response_proposition.py",
    "pipeline/contextual_note_taker_fallback.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(
    package: Path,
    expected_dataset_id: str = "ishtyy/morichika-phase2-retrieval-strict-v3-20260720",
) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    manifest = json.loads((package / "bundle_manifest.json").read_text(encoding="utf-8"))
    manifest_core = {key: value for key, value in manifest.items() if key != "manifest_id"}
    gates["bundle_dataset_and_manifest_identity_valid"] = (
        manifest.get("dataset_id") == expected_dataset_id
        and manifest.get("manifest_id") == hashlib.sha256(
            json.dumps(manifest_core, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )
    declared = {item["path"]: item for item in manifest["files"]}
    mismatches = []
    for relative, expected in declared.items():
        path = package / relative
        if not path.is_file():
            mismatches.append({"path": relative, "error": "missing"})
            continue
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        if actual_size != int(expected["bytes"]) or actual_hash != expected["sha256"]:
            mismatches.append({"path": relative, "error": "size_or_hash_mismatch"})
    gates["all_declared_payload_hashes_match"] = not mismatches
    gates["payload_file_count_matches"] = len(declared) == int(manifest["payload_files"])
    gates["payload_byte_count_matches"] = sum(int(row["bytes"]) for row in declared.values()) == int(manifest["payload_bytes"])
    gates["generalized_router_runtime_files_present"] = REQUIRED_RUNTIME_FILES <= set(declared)

    cache = package / "artifacts/retrieval/phase2_strict_rights_safe_fts_v3_final"
    cache_manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    database = cache / cache_manifest["database"]["path"]
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        rows = dict(connection.execute("SELECT source_id, COUNT(*) FROM evidence GROUP BY source_id").fetchall())
        model_facing = dict(connection.execute("SELECT source_id, COUNT(*) FROM evidence WHERE model_facing=1 GROUP BY source_id").fetchall())
        duplicate_locators = connection.execute(
            "SELECT COUNT(*) FROM (SELECT source_id, source_locator, COUNT(*) c FROM evidence GROUP BY source_id, source_locator HAVING c>1)"
        ).fetchone()[0]
        duplicate_hashes = connection.execute(
            "SELECT COUNT(*) FROM (SELECT record_sha256, COUNT(*) c FROM evidence GROUP BY record_sha256 HAVING c>1)"
        ).fetchone()[0]
        sample = connection.execute(
            "SELECT question, normalized_question FROM evidence WHERE source_id='openai_mmmlu_bn' AND normalized_question<>'' ORDER BY id LIMIT 1"
        ).fetchone()
        exact_canary = 0 if sample is None else connection.execute(
            "SELECT COUNT(*) FROM evidence WHERE normalized_question=? AND source_id='openai_mmmlu_bn'",
            (sample[1],),
        ).fetchone()[0]
        blank_uddipok = connection.execute(
            "SELECT COUNT(*) FROM evidence WHERE source_id='uddipok_v3' "
            "AND (trim(question)='' OR trim(answer)='' OR trim(supporting_text)='')"
        ).fetchone()[0]
        observed_uddipok = {
            row[0]: row[1] for row in connection.execute(
                "SELECT source_locator, record_sha256 FROM evidence WHERE source_id='uddipok_v3'"
            )
        }
    finally:
        connection.close()
    gates["sqlite_integrity_ok"] = integrity == "ok"
    gates["fts_source_counts_exact"] = rows == EXPECTED_FTS
    gates["no_forbidden_fts_source_ids"] = not (set(rows) & FORBIDDEN_SOURCE_IDS)
    gates["model_facing_source_ids_exact"] = set(model_facing) == set(EXPECTED_FTS) - {"bangla_wordnet"}
    gates["wordnet_not_model_facing_truth"] = "bangla_wordnet" not in model_facing
    gates["uddipok_retained_rows_complete"] = blank_uddipok == 0
    gates["unique_source_locators"] = duplicate_locators == 0
    gates["unique_record_hashes"] = duplicate_hashes == 0
    gates["exact_question_canary"] = exact_canary >= 1
    runtime_manifest, runtime_connection = load_fts(cache)
    try:
        runtime_hits = [] if sample is None else retrieve(runtime_connection, sample[0], top_k=5)
    finally:
        runtime_connection.close()
    gates["runtime_loader_accepts_strict_v3"] = runtime_manifest.get("version") == "phase2-strict-rights-safe-fts-v3-nonterminal"
    gates["runtime_exact_retrieval_canary"] = (
        len(runtime_hits) >= 1
        and runtime_hits[0]["source_id"] == "openai_mmmlu_bn"
        and runtime_hits[0]["exact_question"] is True
    )

    counts = json.loads((package / "SOURCE_COUNTS.json").read_text(encoding="utf-8"))
    gates["stored_record_total_exact"] = int(counts["all_stored_records"]) == 302912
    gates["model_facing_record_total_exact"] = (
        int(counts["fts_model_facing_records"]) + int(counts["mmap_model_facing_records"])
    ) == 258394
    gates["lexical_record_total_exact"] = int(counts["lexical_exact_records"]) == 14785

    admission = package / "admission/uddipok_v3_strict_admission_v1"
    admission_manifest = json.loads((admission / "manifest.json").read_text(encoding="utf-8"))
    admission_hashes_match = all(
        (admission / name).is_file()
        and (admission / name).stat().st_size == int(expected["bytes"])
        and sha256_file(admission / name) == expected["sha256"]
        for name, expected in admission_manifest["files"].items()
    )
    admission_rows = [
        json.loads(line) for line in (admission / "row_ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    admitted_rows = [row for row in admission_rows if row["status"] == "admitted"]
    excluded_rows = [row for row in admission_rows if row["status"] == "excluded"]
    group_splits: dict[str, set[str]] = {}
    passage_splits: dict[str, set[str]] = {}
    question_splits: dict[str, set[str]] = {}
    for row in admitted_rows:
        group_splits.setdefault(row["group_id"], set()).add(row["split"])
        passage_splits.setdefault(row["passage_sha256"], set()).add(row["split"])
        question_splits.setdefault(row["question_sha256"], set()).add(row["split"])
    expected_uddipok = {row["source_locator"]: row["fts_record_sha256"] for row in admitted_rows}
    gates["uddipok_admission_payload_hashes_match"] = admission_hashes_match
    gates["uddipok_exactly_two_incomplete_rows_excluded"] = (
        len(excluded_rows) == 2
        and [row["upstream_row_index"] for row in excluded_rows] == [398, 496]
        and {row["reason"] for row in excluded_rows} == {"missing_question"}
    )
    gates["uddipok_all_other_rows_admitted"] = len(admitted_rows) == 3797 and len(admission_rows) == 3799
    gates["uddipok_cache_matches_admission_ledger"] = observed_uddipok == expected_uddipok
    gates["uddipok_excluded_locators_absent"] = not ({row["source_locator"] for row in excluded_rows} & set(observed_uddipok))
    gates["uddipok_connected_groups_split_disjoint"] = all(len(splits) == 1 for splits in group_splits.values())
    gates["uddipok_passages_split_disjoint"] = all(len(splits) == 1 for splits in passage_splits.values())
    gates["uddipok_exact_questions_split_disjoint"] = all(len(splits) == 1 for splits in question_splits.values())
    gates["uddipok_all_splits_nonempty"] = {row["split"] for row in admitted_rows} == {"fit", "calibration", "evaluation"}

    mmap_ok = True
    mmap_counts: dict[str, int] = {}
    for source_id, (directory_name, expected_records) in EXPECTED_MMAP.items():
        mmap_dir = package / "artifacts/retrieval" / directory_name
        mmap_manifest = json.loads((mmap_dir / "manifest.json").read_text(encoding="utf-8-sig"))
        rights_policy = mmap_manifest.get("rights_policy") or {}
        files_ok = all(
            (mmap_dir / name).is_file()
            and (mmap_dir / name).stat().st_size == int(expected["bytes"])
            and sha256_file(mmap_dir / name) == expected["sha256"]
            for name, expected in mmap_manifest["files"].items()
        )
        mmap_counts[source_id] = int(mmap_manifest["records"])
        mmap_ok = mmap_ok and (
            mmap_manifest.get("version") == "phase2-mmap-sparse-cache-v2-idempotent-canonicalizer"
            and mmap_manifest.get("source_id") == source_id
            and int(mmap_manifest["records"]) == expected_records
            and rights_policy.get("bundle_allowed") is True
            and rights_policy.get("quarantined") is False
            and files_ok
        )
    gates["all_default_v2_mmap_caches_hash_rights_valid"] = mmap_ok
    gates["all_default_v2_mmap_counts_exact"] = mmap_counts == {
        source_id: expected[1] for source_id, expected in EXPECTED_MMAP.items()
    }

    rights = json.loads((package / "RIGHTS_SUMMARY.json").read_text(encoding="utf-8"))
    included_ids = {row["source_id"] for row in rights["included"]}
    gates["rights_summary_strict"] = (
        rights.get("all_included_sources_have_explicit_rights_disposition") is True
        and rights.get("all_model_facing_sources_independently_licensed") is False
        and rights.get("competition_use_only") is True
        and rights.get("public_redistribution_authorized_by_this_manifest") is False
        and rights.get("competition_test_or_manual_label_artifacts_included") is False
        and rights.get("current_affairs_included") is False
        and rights.get("quarantined_or_unverified_sources_included") is False
    )
    gates["rights_source_ids_cover_payload"] = set(EXPECTED_FTS) | set(EXPECTED_MMAP) <= included_ids
    gates["wikipedia_excluded_source_wide"] = (
        "bengali_wikipedia_20210320" not in rows
        and any("Wikipedia 2021 source-wide" in value for value in rights["explicitly_excluded"])
    )

    priority = json.loads((package / "SOURCE_PRIORITY.json").read_text(encoding="utf-8"))
    priority_ids = {source_id for tier in priority["tiers"] for source_id in tier["source_ids"]}
    gates["priority_covers_every_admitted_source"] = included_ids <= priority_ids
    gates["priority_semantic_first"] = priority["safety"].get("semantic_alignment_precedes_authority") is True
    gates["contextual_retrieval_disabled"] = priority["safety"].get("contextual_external_retrieval_allowed") is False
    gates["retrieval_nonterminal"] = priority["safety"].get("retrieval_is_terminal") is False

    all_pass = all(value is True for value in gates.values())
    return {
        "version": "phase2-strict-rights-safe-retrieval-v3-verification-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "package": package.relative_to(ROOT).as_posix(),
        "package_id": manifest["package_id"],
        "gates": gates,
        "hash_mismatches": mismatches,
        "observed": {
            "fts_records_by_source": rows,
            "fts_model_facing_records_by_source": model_facing,
            "all_stored_records": counts["all_stored_records"],
            "model_facing_records": counts["fts_model_facing_records"] + counts["mmap_model_facing_records"],
            "lexical_exact_records": counts["lexical_exact_records"],
            "mmap_records_by_source": mmap_counts,
        },
        "all_gates_pass": all_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=PACKAGE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--expected-dataset-id",
        default="ishtyy/morichika-phase2-retrieval-strict-v3-20260720",
    )
    args = parser.parse_args()
    result = validate(args.package.resolve(), expected_dataset_id=args.expected_dataset_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
