from __future__ import annotations

import json
import zipfile
from pathlib import Path

from pipeline import build_morichika_phase2_generalized_hybrid_v3 as builder
from pipeline.phase2_sparse_retrieval import build as build_sparse_retrieval


DATASET_ID = "ishtyy/morichika-phase2-retrieval-strict-v3-fixture"
MANIFEST_ID = "a" * 64


def transformed() -> str:
    return builder.transformed_runner(DATASET_ID, MANIFEST_ID)


def test_v3_runner_is_phase1_free_and_legacy_terminal_shortcut_is_removed() -> None:
    code = transformed()
    lowered = code.casefold()
    for forbidden in ("phase1_", "route_audit", "gold_label", "gold_labels"):
        assert forbidden not in lowered
    assert "exact_source_consensus_candidate" not in code
    assert "terminal_source_candidate" not in code
    assert 'method = "closed_exact_key_terminal"' in code
    assert "TERMINAL_ELIGIBLE_EXACT_SOURCES" in code
    assert 'composite_query_mode="all_closed"' in code
    assert '"retrieval_miss_means": "NEI_not_refutation"' in code


def test_context_policy_families_and_isolation_are_explicit() -> None:
    code = transformed()
    for marker in (
        "question-context-response exact slot",
        "Unicode NFC/joiner/digit",
        "grammar theory/definition/rule",
        "negation/quantifier/comparator/modality/clause scope",
        "entity-relation-property",
        "event phase",
        "bounded math",
    ):
        assert marker in code
    assert "শুধু প্রদত্ত context ব্যবহার করবে" in code
    assert "lexical cache" in code
    assert 'if row["context_available"]:' in code
    assert "strict_v3_composite_cache_missing" in code


def test_closed_exact_candidate_still_calls_model(tmp_path: Path) -> None:
    namespace = {"__name__": "morichika_v3_test"}
    exec(compile(transformed(), "<morichika-v3>", "exec"), namespace)

    calls: list[dict] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {"content": '{"verdict":"refuted"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            }

    class Requests:
        @staticmethod
        def post(url: str, **kwargs):
            calls.append({"url": url, **kwargs})
            return Response()

    namespace["requests"] = Requests
    namespace["RUN_ROOT"] = tmp_path
    row = {
        "example_id": "closed-1",
        "source_index": 0,
        "model_prompt_bn": "বাংলাদেশের রাজধানী কী?",
        "model_response_bn": "চট্টগ্রাম",
        "model_context_bn": "",
        "context_available": False,
    }
    retrieval = {
        "closed-1": {
            "retrieval_candidates": [],
            "merged_source_candidate": {
                "verdict": 1,
                "status": "source_consensus_candidate",
            },
        }
    }
    output = namespace["run_queue"](
        [row], 8080, 0, retrieval, [], lambda *args, **kwargs: {},
        namespace["time"].perf_counter(),
    )
    assert len(calls) == 1
    assert output[0]["label"] == 0
    assert output[0]["method"] == "gbnf_verdict_only"


def _exact_candidate(
    *, source_id: str = "bangla_mmlu", verdict: int = 1,
    policy_tier: int = 1, evidence_role: str = "support_candidate",
) -> dict:
    return {
        "source_id": source_id,
        "source_record_index": 1,
        "exact_normalized": True,
        "model_facing_eligible": True,
        "semantic_alignment_tier": 0,
        "slot_compatibility_tier": 0,
        "policy_compatibility_tier": policy_tier,
        "number_set_match": True,
        "negation_set_match": True,
        "answers": ["ঢাকা"],
        "choices": ["ঢাকা", "চট্টগ্রাম"],
        "evidence_role": evidence_role,
        "source_verdict_candidate": {
            "verdict": verdict,
            "rule": "exact_question_key_consensus_response_exact_key",
        },
    }


def _closed_row() -> dict:
    return {
        "example_id": "closed-exact",
        "source_index": 0,
        "model_prompt_bn": "বাংলাদেশের রাজধানী কী?",
        "model_response_bn": "ঢাকা",
        "model_context_bn": "",
        "context_available": False,
    }


def test_exact_terminal_is_order_invariant_and_rejects_operation_mismatch() -> None:
    namespace = {"__name__": "morichika_v3_test"}
    exec(compile(transformed(), "<morichika-v3>", "exec"), namespace)
    decide = namespace["closed_exact_terminal"]
    row = _closed_row()
    first = _exact_candidate()
    second = _exact_candidate(source_id="nctb_education_aux")
    merged = {"verdict": 1, "status": "source_consensus_candidate"}
    forward = decide(row, {
        "retrieval_candidates": [first, second],
        "merged_source_candidate": merged,
    })
    reverse = decide(row, {
        "retrieval_candidates": [second, first],
        "merged_source_candidate": merged,
    })
    assert forward == reverse
    assert forward["verdict"] == 1
    assert forward["source_ids"] == ["bangla_mmlu", "nctb_education_aux"]

    mismatch = _exact_candidate(policy_tier=4)
    assert decide(row, {
        "retrieval_candidates": [mismatch],
        "merged_source_candidate": merged,
    }) is None


def test_exact_terminal_abstains_on_admitted_counterevidence() -> None:
    namespace = {"__name__": "morichika_v3_test"}
    exec(compile(transformed(), "<morichika-v3>", "exec"), namespace)
    support = _exact_candidate()
    counter = {
        **_exact_candidate(
            source_id="openai_mmmlu_bn", evidence_role="counter_candidate"
        ),
        "source_verdict_candidate": None,
        "answers": ["চট্টগ্রাম"],
    }
    result = namespace["closed_exact_terminal"](_closed_row(), {
        "retrieval_candidates": [counter, support],
        "merged_source_candidate": {
            "verdict": 1, "status": "source_consensus_candidate"
        },
    })
    assert result is None


def test_exact_terminal_rejects_fuzzy_containment_and_ineligible_sources() -> None:
    namespace = {"__name__": "morichika_v3_test"}
    exec(compile(transformed(), "<morichika-v3>", "exec"), namespace)
    decide = namespace["closed_exact_terminal"]
    row = _closed_row()
    merged = {"verdict": 1, "status": "source_consensus_candidate"}

    fuzzy = {**_exact_candidate(), "exact_normalized": False}
    assert decide(row, {
        "retrieval_candidates": [fuzzy], "merged_source_candidate": merged,
    }) is None

    containment = _exact_candidate()
    containment["source_verdict_candidate"] = {
        "verdict": 1,
        "rule": "exact_question_positive_answer_containment",
    }
    assert decide(row, {
        "retrieval_candidates": [containment], "merged_source_candidate": merged,
    }) is None

    for source_id in ("bengali_wikipedia", "bangla_wordnet", "uddipok_v3"):
        candidate = _exact_candidate(source_id=source_id)
        assert decide(row, {
            "retrieval_candidates": [candidate],
            "merged_source_candidate": merged,
        }) is None


def test_explicit_empty_sparse_sources_do_not_reopen_private_defaults(
    tmp_path: Path,
) -> None:
    source = tmp_path / "inputs.jsonl"
    source.write_text(
        json.dumps(
            {
                "example_id": "x",
                "source_index": 0,
                "model_prompt_bn": "প্রশ্ন",
                "model_response_bn": "উত্তর",
                "context_available": False,
                "formatting_provenance": {"status": "fixture"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = build_sparse_retrieval(
        source,
        tmp_path / "out",
        index_specs=[],
        composite_cache_dir=None,
    )
    assert manifest["indexes"] == []
    assert manifest["terminal_candidate_status_counts"] == {
        "no_terminal_source_candidate": 1
    }


def test_monolithic_payload_zip_is_manifest_bound_and_hash_checked(
    tmp_path: Path,
) -> None:
    namespace = {"__name__": "morichika_v3_test"}
    exec(compile(transformed(), "<morichika-v3>", "exec"), namespace)
    payload = b"strict-v3-fixture"
    spec = {
        "path": "pipeline/fixture.txt",
        "bytes": len(payload),
        "sha256": builder.hashlib.sha256(payload).hexdigest(),
    }
    core = {"dataset_id": DATASET_ID, "files": [spec]}
    manifest_id = builder.hashlib.sha256(
        builder.canonical(core).encode("utf-8")
    ).hexdigest()
    manifest = {**core, "manifest_id": manifest_id}
    input_root = tmp_path / "input"
    input_root.mkdir()
    with zipfile.ZipFile(input_root / "payload.zip", "w") as archive:
        archive.writestr("bundle_manifest.json", json.dumps(manifest))
        archive.writestr(spec["path"], payload)
    namespace["INPUT_ROOT"] = input_root
    namespace["RUN_ROOT"] = tmp_path / "working"
    namespace["RETRIEVAL_MANIFEST_ID"] = manifest_id
    manifest_path, observed = namespace["_materialize_bound_payload_zip"]()
    assert observed == manifest
    assert manifest_path.is_file()
    assert (manifest_path.parent / spec["path"]).read_bytes() == payload


def test_local_package_build_binds_retrieval_and_never_pushes(tmp_path: Path) -> None:
    retrieval = tmp_path / "retrieval"
    retrieval.mkdir()
    core = {
        "dataset_id": DATASET_ID,
        "files": [
            {"path": path, "bytes": 1, "sha256": "0" * 64}
            for path in sorted(builder.REQUIRED_RETRIEVAL_PAYLOAD_PATHS)
        ],
        "source_counts": {"package_sources": 13},
    }
    manifest = {
        **core,
        "manifest_id": builder.hashlib.sha256(
            builder.canonical(core).encode("utf-8")
        ).hexdigest(),
    }
    (retrieval / "bundle_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    output = tmp_path / "kernel"
    receipt = builder.build(output, retrieval)
    assert receipt["status"].endswith("not_submitted")
    assert receipt["push_performed"] is False
    assert receipt["architecture"]["context_terminal_allowlist"] == []
    assert receipt["architecture"]["closed_retrieval_terminal"] == (
        "exact_aligned_conflict_free_key_only"
    )
    assert receipt["architecture"]["strict_v3_fts_terminal"] is False
    assert receipt["architecture"]["phase1_ids_labels_route_audit_used"] is False
    metadata = json.loads((output / "kernel-metadata.json").read_text())
    assert DATASET_ID in metadata["dataset_sources"]
