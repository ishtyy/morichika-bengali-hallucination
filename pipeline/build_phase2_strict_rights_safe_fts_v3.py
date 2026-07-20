"""Build the strict independently-licensed Phase 2 retrieval-v3 FTS cache.

This is deliberately a narrow admission wrapper around the already validated
composite-FTS adapters.  It admits only local, pinned, non-test-targeted sources
whose adapters were exercised by the v2 composite build.  Recommended sources
without an existing validated adapter/cache remain explicit pending entries.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import build_phase2_composite_fts_cache as base


OUTPUT = ROOT / "artifacts/retrieval/phase2_strict_rights_safe_fts_v3_final"
OLD_CACHE = ROOT / "artifacts/retrieval/phase2_composite_fts_v2"
UDDIPOK_ADMISSION = ROOT / "artifacts/assurance/uddipok_v3_strict_admission_v1_20260720"
VERSION = "phase2-strict-rights-safe-fts-v3-nonterminal"


ADMITTED = {
    "openai_mmmlu_bn": {
        "license": "MIT",
        "attribution_manifest": "external_data/openai_mmmlu_bn/manifest.json",
        "authority_tier": 1,
        "authority_class": "curated_dataset",
        "runtime_scope": "nonterminal_closed_book_evidence",
    },
    "nctb_passage_qa_mendeley_v1": {
        "license": "CC BY 4.0",
        "attribution_manifest": "external_data/nctb_passage_qa_mendeley_v1/manifest.json",
        "authority_tier": 0,
        "authority_class": "books_and_user_ocr_priority",
        "runtime_scope": "nonterminal_closed_book_evidence",
    },
    "uddipok_v3": {
        "license": "CC BY 4.0",
        "attribution_manifest": "external_data/uddipok_v3/manifest.json",
        "authority_tier": 1,
        "authority_class": "curated_dataset",
        "runtime_scope": "nonterminal_closed_book_evidence",
    },
    "bangla_wordnet": {
        "license": "GPL-3.0",
        "attribution_manifest": "external_data/bangla_wordnet/manifest.json",
        "authority_tier": 2,
        "authority_class": "query_expansion_component",
        "runtime_scope": "query_expansion_only_not_model_facing_truth",
    },
}


PENDING = [
    {
        "source_id": "bengali_wikipedia_20210320",
        "license": "CC BY-SA / GFDL article-text provenance",
        "records": 167786,
        "reason": "the record cache is validated, but page/revision/permanent-URL attribution sidecars required for strict reuse are absent",
    },
    {
        "source_id": "banglaquad_v1",
        "license": "MIT repository; Wikipedia attribution/share-alike provenance retained",
        "records": 20085,
        "reason": "local SQuAD payload is validated, but no separately validated strict-v3 adapter/cache exists",
    },
    {
        "source_id": "proshno_binnash_v2",
        "license": "CC BY 4.0",
        "records": 4068,
        "reason": "local workbook is validated, but no privacy-stripping exact-answer adapter/cache has been validated",
    },
    {
        "source_id": "bengali_antonym_dictionary_commons_v1",
        "license": "CC BY-SA 4.0",
        "records": 0,
        "reason": "OCR candidates are nonterminal and still require ordered pair, sense, POS, register, and independent reconciliation gates",
    },
    {
        "source_id": "bnwiktionary_full_reviewed_extensions",
        "license": "CC BY-SA 4.0 / GFDL",
        "records": 0,
        "reason": "the full dump is pinned, but only the already-admitted 3720-record exact lexical subset has a validated runtime cache",
    },
]


def build(output: Path, inventory: Path, old_cache: Path) -> dict[str, object]:
    del inventory  # Admission is inherited only from the hash-verified v2 cache.
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    old_manifest_path = old_cache / "manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    if old_manifest.get("version") != "phase2-composite-fts-cache-v2-nctbench-nonterminal":
        raise ValueError("unexpected parent cache version")
    old_database = old_cache / old_manifest["database"]["path"]
    if (
        old_database.stat().st_size != int(old_manifest["database"]["bytes"])
        or base.sha256_file(old_database) != old_manifest["database"]["sha256"]
    ):
        raise ValueError("parent cache hash/size mismatch")

    old_sources = {source["source_id"]: source for source in old_manifest["sources"]}
    if not set(ADMITTED) <= set(old_sources):
        raise ValueError("parent cache does not contain every strict source")

    admission_manifest_path = UDDIPOK_ADMISSION / "manifest.json"
    admission_manifest = json.loads(admission_manifest_path.read_text(encoding="utf-8"))
    if admission_manifest.get("version") != "uddipok-v3-strict-admission-ledger-v1":
        raise ValueError("unexpected UDDIPOK admission-ledger version")
    for name, expected in admission_manifest["files"].items():
        path = UDDIPOK_ADMISSION / name
        if path.stat().st_size != int(expected["bytes"]) or base.sha256_file(path) != expected["sha256"]:
            raise ValueError(f"UDDIPOK admission-ledger hash mismatch: {name}")
    ledger = [json.loads(line) for line in (UDDIPOK_ADMISSION / "row_ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    admitted_uddipok = [row for row in ledger if row["status"] == "admitted"]
    excluded_uddipok = [row for row in ledger if row["status"] == "excluded"]
    if len(admitted_uddipok) != 3797 or [row["upstream_row_index"] for row in excluded_uddipok] != [398, 496]:
        raise ValueError("UDDIPOK admission counts/exclusions are not exact")
    excluded_locators = [row["source_locator"] for row in excluded_uddipok]
    expected_uddipok_hashes = {row["fts_record_sha256"] for row in admitted_uddipok}

    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary build exists: {temporary}")
    temporary.mkdir(parents=True)
    database = temporary / "evidence.sqlite3"
    connection = sqlite3.connect(database)
    source_ids = list(ADMITTED)
    placeholders = ",".join("?" for _ in source_ids)
    try:
        connection.executescript(base.SCHEMA)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("ATTACH DATABASE ? AS parent", (str(old_database),))
        connection.execute(
            "INSERT INTO evidence SELECT * FROM parent.evidence WHERE source_id IN (" + placeholders + ")",
            source_ids,
        )
        connection.executemany(
            "DELETE FROM evidence WHERE source_id='uddipok_v3' AND source_locator=?",
            [(locator,) for locator in excluded_locators],
        )
        connection.execute(
            "INSERT INTO evidence_fts(rowid, question, answer, title, supporting_text) "
            "SELECT id, question, answer, title, supporting_text FROM evidence ORDER BY id"
        )
        connection.execute("INSERT INTO evidence_fts(evidence_fts) VALUES('optimize')")
        connection.execute("PRAGMA optimize")
        connection.commit()
        counts = dict(connection.execute(
            "SELECT source_id, COUNT(*) FROM evidence GROUP BY source_id ORDER BY source_id"
        ).fetchall())
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"SQLite integrity failure: {integrity}")
        observed_uddipok_hashes = {
            row[0] for row in connection.execute(
                "SELECT record_sha256 FROM evidence WHERE source_id='uddipok_v3'"
            )
        }
        if observed_uddipok_hashes != expected_uddipok_hashes:
            raise ValueError("UDDIPOK retained-row set does not match admission ledger")
    except BaseException:
        connection.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        try:
            connection.close()
        except Exception:
            pass

    source_rows = []
    for source_id in source_ids:
        source = dict(old_sources[source_id])
        if source_id != "uddipok_v3" and int(source["records"]) != int(counts[source_id]):
            raise ValueError(f"record-count drift for {source_id}")
        if source_id == "uddipok_v3":
            source["parent_records"] = int(source["records"])
            source["records"] = int(counts[source_id])
            source["excluded_records"] = len(excluded_uddipok)
        source_rows.append(source)
    manifest: dict[str, object] = {
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "closed_book_only": True,
            "contextual_external_retrieval_allowed": False,
            "retrieval_is_terminal": False,
            "retrieval_miss_means": "NEI",
            "training_or_evaluation_sources_included": False,
            "component_resources_are_model_facing_truth": False,
            "synthetic_math_cot_or_pot_included": False,
        },
        "inputs": {
            "parent_cache_manifest": old_manifest_path.relative_to(ROOT).as_posix(),
            "parent_cache_manifest_sha256": base.sha256_file(old_manifest_path),
            "parent_cache_id": old_manifest["cache_id"],
            "parent_database_sha256": old_manifest["database"]["sha256"],
            "derivation": "row-exact admitted source subset, with the UDDIPOK exclusion set bound to its acceptance ledger, plus a fresh FTS5 index",
            "uddipok_admission_manifest": admission_manifest_path.relative_to(ROOT).as_posix(),
            "uddipok_admission_manifest_sha256": base.sha256_file(admission_manifest_path),
            "uddipok_admission_id": admission_manifest["admission_id"],
        },
        "sources": source_rows,
        "summary": {
            "sources": len(source_rows),
            "records": sum(counts.values()),
            "model_facing_sources": sum(bool(row["model_facing"]) for row in source_rows),
            "component_sources": sum(not bool(row["model_facing"]) for row in source_rows),
            "records_by_source": counts,
        },
        "database": {
            "path": "evidence.sqlite3",
            "bytes": database.stat().st_size,
            "sha256": base.sha256_file(database),
            "integrity_check": "ok",
        },
    }

    for source in manifest["sources"]:
        policy = ADMITTED[source["source_id"]]
        attribution = ROOT / policy["attribution_manifest"]
        source["rights"] = {
            "license": policy["license"],
            "independently_licensed": True,
            "non_test_targeted": True,
            "attribution_manifest": policy["attribution_manifest"],
            "attribution_manifest_sha256": base.sha256_file(attribution),
        }
        source["priority"] = {
            "authority_tier": policy["authority_tier"],
            "authority_class": policy["authority_class"],
            "semantic_alignment_precedes_authority": True,
        }
        source["runtime_scope"] = policy["runtime_scope"]
        if source["source_id"] == "uddipok_v3":
            source["split_acceptance"] = {
                "ledger_manifest": admission_manifest_path.relative_to(ROOT).as_posix(),
                "ledger_manifest_sha256": base.sha256_file(admission_manifest_path),
                "admission_id": admission_manifest["admission_id"],
                "retained_records": 3797,
                "excluded_source_locators": excluded_locators,
                "passage_identity_cross_split_overlap": 0,
                "question_identity_cross_split_overlap": 0,
                "connected_group_cross_split_overlap": 0,
            }

    manifest["strict_admission"] = {
        "independently_licensed_only": True,
        "non_test_targeted_only": True,
        "current_affairs_included": False,
        "competition_labels_or_manual_decisions_included": False,
        "quarantined_sources_included": False,
        "contextual_external_retrieval_allowed": False,
        "wikipedia_terminal_authority": False,
        "semantic_alignment_precedes_source_priority": True,
        "source_wide_exclusions": {
            "bengali_wikipedia_20210320": "missing page/revision/permanent-URL attribution sidecars"
        },
    }
    manifest["pending_recommended_sources"] = PENDING
    manifest["inputs"]["strict_builder_wrapper"] = Path(__file__).relative_to(ROOT).as_posix()
    manifest["inputs"]["strict_builder_wrapper_sha256"] = base.sha256_file(Path(__file__))
    manifest["cache_id"] = base.sha256_json({key: value for key, value in manifest.items() if key != "cache_id"})
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=base.INVENTORY)
    parser.add_argument("--parent-cache", type=Path, default=OLD_CACHE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = build(args.output.resolve(), args.inventory.resolve(), args.parent_cache.resolve())
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"cache_id={result['cache_id']}")


if __name__ == "__main__":
    main()
