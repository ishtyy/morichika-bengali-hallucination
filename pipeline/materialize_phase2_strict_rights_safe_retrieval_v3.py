"""Materialize the local-only strict rights-safe retrieval-v3 package.

No network or Kaggle operation is performed.  The private competition-use
package combines independently licensed strict-v3 sources with explicitly
rights-dispositioned private/noncommercial v2 caches and their runtime,
attribution, priority, and hash metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/kaggle/morichika_phase2_retrieval_strict_v3_20260720"
STRICT_FTS = ROOT / "artifacts/retrieval/phase2_strict_rights_safe_fts_v3_final"
LEXICAL = ROOT / "artifacts/retrieval/phase2_lexical_seed_v2"
UDDIPOK_ADMISSION = ROOT / "artifacts/assurance/uddipok_v3_strict_admission_v1_20260720"
SOURCE_REGISTRY = ROOT / "artifacts/phase2/source_registry_v6"
MMAP_CACHES = {
    "bangla_mmlu": ROOT / "artifacts/retrieval/bangla_mmlu_char_v1_mmap_v2",
    "nctb_qa_87805": ROOT / "artifacts/retrieval/nctb_qa_87805_char_v1_mmap_v2",
    "nctb_education_aux": ROOT / "artifacts/retrieval/nctb_education_aux_char_v1_mmap_v2",
    "downloads_bcs_10_50": ROOT / "artifacts/retrieval/downloads_bcs_10_50_char_v1_mmap_v2",
    "nctb_schooltext": ROOT / "artifacts/retrieval/nctb_schooltext_word_v1_mmap_v2",
    "joykoli_six_part": ROOT / "artifacts/retrieval/joykoli_six_part_char_v3_mmap_v2",
}


PIPELINE_FILES = [
    "pipeline/phase2_canonicalize.py",
    "pipeline/phase2_composite_fts_retrieval.py",
    "pipeline/phase2_mmap_retrieval.py",
    "pipeline/phase2_sparse_retrieval.py",
    "pipeline/phase2_response_proposition.py",
    "pipeline/contextual_note_taker_fallback.py",
    "pipeline/phase2_source_authority.py",
    "pipeline/phase2_evidence_priority.py",
    "pipeline/phase2_retrieval_lane_policy.py",
    "pipeline/resources/phase2_source_authority_v1.json",
    "pipeline/resources/phase2_evidence_priority_v2.json",
]


ATTRIBUTION_FILES = [
    "external_data/openai_mmmlu_bn/manifest.json",
    "external_data/nctb_passage_qa_mendeley_v1/manifest.json",
    "external_data/uddipok_v3/manifest.json",
    "external_data/nctb_qa_87805/manifest.json",
    "external_data/nctbench/manifest.json",
    "external_data/somajgyaan/manifest.json",
    "external_data/benqa/manifest.json",
    "external_data/ssc_banglatutor_v2/manifest.json",
    "external_data/bcs_downloads_10_50/manifest.json",
    "external_data/joykoli_six_part_v3/manifest.json",
    "external_data/nctb_schooltext_mendeley_v1/manifest.json",
    "external_data/bangla_wordnet/manifest.json",
    "artifacts/retrieval/phase2_lexical_seed_v2/ATTRIBUTION.json",
    "artifacts/retrieval/phase2_lexical_seed_v2/manifest.json",
    "artifacts/retrieval/nctb_schooltext_word_v1_mmap_v2/manifest.json",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def copy_tree(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(source, target)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def materialize(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    temp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)

    strict_manifest = json.loads((STRICT_FTS / "manifest.json").read_text(encoding="utf-8"))
    lexical_manifest = json.loads((LEXICAL / "manifest.json").read_text(encoding="utf-8"))
    mmap_manifests = {
        source_id: json.loads((directory / "manifest.json").read_text(encoding="utf-8-sig"))
        for source_id, directory in MMAP_CACHES.items()
    }
    if strict_manifest.get("version") != "phase2-strict-rights-safe-fts-v3-nonterminal":
        raise ValueError("strict FTS version mismatch")
    if lexical_manifest.get("version") != "phase2-lexical-seed-cache-v2-bagdhara-bnwiktionary":
        raise ValueError("lexical cache version mismatch")
    for source_id, mmap_manifest in mmap_manifests.items():
        rights = mmap_manifest.get("rights_policy") or {}
        if mmap_manifest.get("version") != "phase2-mmap-sparse-cache-v2-idempotent-canonicalizer":
            raise ValueError(f"{source_id} mmap version mismatch")
        if rights.get("bundle_allowed") is not True or rights.get("quarantined") is not False:
            raise ValueError(f"{source_id} is not private-bundle eligible")

    copy_tree(STRICT_FTS, temp / "artifacts/retrieval/phase2_strict_rights_safe_fts_v3_final")
    copy_tree(LEXICAL, temp / "artifacts/retrieval/phase2_lexical_seed_v2")
    for directory in MMAP_CACHES.values():
        copy_tree(directory, temp / "artifacts/retrieval" / directory.name)
    copy_tree(UDDIPOK_ADMISSION, temp / "admission/uddipok_v3_strict_admission_v1")
    copy_tree(SOURCE_REGISTRY, temp / "source_registry_v6")

    for relative in PIPELINE_FILES:
        source = ROOT / relative
        target = temp / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for relative in ATTRIBUTION_FILES:
        source = ROOT / relative
        target = temp / "attribution" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    records_by_source = dict(strict_manifest["summary"]["records_by_source"])
    records_by_source.update({source_id: int(manifest["records"]) for source_id, manifest in mmap_manifests.items()})
    records_by_source.update({
        f"lexical::{key}": int(value)
        for key, value in lexical_manifest["source_record_counts"].items()
    })
    source_counts = {
        "package_sources": 12,
        "fts_sources": int(strict_manifest["summary"]["sources"]),
        "fts_records": int(strict_manifest["summary"]["records"]),
        "fts_model_facing_records": sum(
            int(source["records"]) for source in strict_manifest["sources"] if source["model_facing"]
        ),
        "mmap_sources": len(mmap_manifests),
        "mmap_model_facing_records": sum(int(manifest["records"]) for manifest in mmap_manifests.values()),
        "lexical_exact_records": int(lexical_manifest["records"]),
        "all_stored_records": int(strict_manifest["summary"]["records"])
        + sum(int(manifest["records"]) for manifest in mmap_manifests.values())
        + int(lexical_manifest["records"]),
        "records_by_source": records_by_source,
    }
    write_json(temp / "SOURCE_COUNTS.json", source_counts)

    source_priority = {
        "version": "phase2-strict-rights-safe-source-priority-v3",
        "decision_order": [
            "closed_book_lane_only",
            "operation_and_semantic_alignment",
            "entity_relation_property_date_number_unit_negation_alignment",
            "support_counter_neutral_relation",
            "source_authority_tier",
            "exactness",
            "within_source_rank",
            "deterministic_identity",
        ],
        "tiers": [
            {"tier": 0, "class": "books_and_user_ocr_priority", "source_ids": ["downloads_bcs_10_50", "joykoli_six_part", "nctb_passage_qa_mendeley_v1", "nctb_schooltext"]},
            {"tier": 1, "class": "curated_dataset", "source_ids": ["bangla_mmlu", "nctb_education_aux", "nctb_qa_87805", "openai_mmmlu_bn", "uddipok_v3"]},
            {"tier": 2, "class": "query_expansion_component", "source_ids": ["bangla_wordnet"], "model_facing_truth": False},
            {"tier": "operation_scoped", "class": "exact_lexical_cache", "source_ids": ["bangla_bagdhara_cc_by_sa_4", "bnwiktionary_cc_by_sa_4_20260701"]},
        ],
        "safety": {
            "semantic_alignment_precedes_authority": True,
            "contextual_external_retrieval_allowed": False,
            "retrieval_is_terminal": False,
            "retrieval_miss_means": "NEI",
            "wikipedia_terminal_authority": False,
            "wordnet_is_truth": False,
            "lexical_fuzzy_lookup_terminal": False,
        },
    }
    write_json(temp / "SOURCE_PRIORITY.json", source_priority)

    rights_summary = {
        "version": "morichika-phase2-retrieval-private-strict-v3-rights-summary",
        "private_dataset": True,
        "competition_use_only": True,
        "public_redistribution_authorized_by_this_manifest": False,
        "all_model_facing_sources_independently_licensed": False,
        "all_included_sources_have_explicit_rights_disposition": True,
        "competition_test_or_manual_label_artifacts_included": False,
        "current_affairs_included": False,
        "quarantined_or_unverified_sources_included": False,
        "included": [
            {"source_id": "nctb_passage_qa_mendeley_v1", "license": "CC BY 4.0", "scope": "nonterminal closed-book evidence"},
            {"source_id": "nctb_schooltext", "license": "CC BY 4.0 processed corpus", "scope": "nonterminal book/textbook evidence"},
            {"source_id": "openai_mmmlu_bn", "license": "MIT", "scope": "nonterminal curated QA evidence"},
            {"source_id": "uddipok_v3", "license": "CC BY 4.0", "scope": "3,797 complete rows; nonterminal curated reading-comprehension evidence; fixed passage/question-connected split ledger"},
            {"source_id": "bangla_wordnet", "license": "GPL-3.0", "scope": "query expansion only; not truth"},
            {"source_id": "bangla_bagdhara_cc_by_sa_4", "license": "CC BY-SA 4.0", "scope": "exact operation-scoped lexical lookup"},
            {"source_id": "bnwiktionary_cc_by_sa_4_20260701", "license": "CC BY-SA 4.0/GFDL", "scope": "exact operation-scoped lexical lookup"},
            {"source_id": "bangla_mmlu", "license": "undeclared", "scope": "private competition evidence only; no public redistribution"},
            {"source_id": "nctb_qa_87805", "license": "CC BY-NC 4.0", "scope": "noncommercial nonterminal evidence"},
            {"source_id": "nctb_education_aux", "license": "mixed/record-specific", "scope": "private corroboration with record-level terminal policy"},
            {"source_id": "downloads_bcs_10_50", "license": "user-supplied private authorization", "scope": "private nonterminal OCR evidence"},
            {"source_id": "joykoli_six_part", "license": "user-provided private competition-use authorization", "scope": "corroboration only; never promotes verdict"},
        ],
        "pending_not_included": strict_manifest["pending_recommended_sources"],
        "explicitly_excluded": [
            "BenHalluEval and every Phase-1 row-targeted/manual-label artifact",
            "SQuAD-BN (unresolved translated-text/original-copyright rights gate)",
            "Bengali Wikipedia 2021 source-wide (missing page/revision/permanent-URL attribution sidecars)",
            "current-affairs corpora",
            "unfinished OCR, mutable web, targeted bdlaws, and hard-context trees",
        ],
    }
    write_json(temp / "RIGHTS_SUMMARY.json", rights_summary)
    write_json(temp / "dataset-metadata.json", {
        "title": "MORICHIKA Phase 2 Strict Rights-Safe Retrieval v3",
        "id": "ishtyy/morichika-phase2-retrieval-strict-v3-20260720",
        "licenses": [{"name": "other"}],
        "isPrivate": True,
    })

    payload = []
    for path in sorted(item for item in temp.rglob("*") if item.is_file()):
        relative = path.relative_to(temp).as_posix()
        payload.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "version": "morichika-phase2-retrieval-strict-v3-local-package",
        "dataset_id": "ishtyy/morichika-phase2-retrieval-strict-v3-20260720",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "uploaded": False,
        "package_id": sha256_json(payload),
        "payload_files": len(payload),
        "payload_bytes": sum(item["bytes"] for item in payload),
        "source_counts": source_counts,
        "strict_fts_cache_id": strict_manifest["cache_id"],
        "files": payload,
    }
    manifest["manifest_id"] = sha256_json(manifest)
    write_json(temp / "bundle_manifest.json", manifest)
    os.replace(temp, output)
    return manifest


def main() -> None:
    result = materialize(OUTPUT)
    print(json.dumps({key: result[key] for key in ("package_id", "payload_files", "payload_bytes", "source_counts")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
