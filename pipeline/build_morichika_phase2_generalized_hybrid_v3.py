"""Build the local-only MORICHIKA Phase 2 generalized hybrid v4 kernel.

The generated runner combines the private/authorized v2 mmap evidence sources
with the independently admitted strict-v3 composite FTS.  The strict FTS and
all fuzzy evidence are nonterminal; only the generalized, conflict-free exact
key lane from explicitly terminal-eligible sources may decide without a model.
Contextual rows are isolated from every external cache and use full-coverage
window adjudication plus an exact-question aggregation pass.

This builder never uploads, launches, or submits anything.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "kaggle/phase2_gemma4_e4b_base_heavy_hybrid_v1/run_hybrid.py"
RETRIEVAL_PACKAGE = (
    ROOT / "artifacts/kaggle/morichika_phase2_retrieval_strict_v3_20260720"
)
CONTEXT_TRUST_POLICY = ROOT / "artifacts/phase2/context_rule_trust_policy_v1.json"
CONTEXT_RUNTIME_V4 = ROOT / "pipeline/contextual_policy_v4_runtime.py"
OUT = (
    ROOT
    / "artifacts/kaggle/morichika_phase2_generalized_hybrid_v4_kernel_20260720"
)
NOTEBOOK_NAME = "morichika-phase2-generalized-hybrid-v4.ipynb"
KERNEL_ID = "ishtyy/morichika-phase2-generalized-hybrid-v4-20260720"
KERNEL_TITLE = "MORICHIKA Phase2 Generalized Hybrid v4 20260720"
MODEL_DATASETS = [
    "ishtyy/bichar-gemma4-e4b-qat-udq4-20260718",
    "ishtyy/bichar-llama-cpp-e4b-sm75-runtime-20260718",
]
REQUIRED_RETRIEVAL_PAYLOAD_PATHS = {
    "artifacts/retrieval/phase2_strict_rights_safe_fts_v3_final/manifest.json",
    "artifacts/retrieval/phase2_strict_rights_safe_fts_v3_final/evidence.sqlite3",
    "artifacts/retrieval/phase2_lexical_seed_v2/manifest.json",
    "artifacts/retrieval/bangla_mmlu_char_v1_mmap_v2/manifest.json",
    "artifacts/retrieval/nctb_qa_87805_char_v1_mmap_v2/manifest.json",
    "artifacts/retrieval/nctb_education_aux_char_v1_mmap_v2/manifest.json",
    "artifacts/retrieval/downloads_bcs_10_50_char_v1_mmap_v2/manifest.json",
    "artifacts/retrieval/nctb_schooltext_word_v1_mmap_v2/manifest.json",
    "artifacts/retrieval/joykoli_six_part_char_v3_mmap_v2/manifest.json",
    "pipeline/phase2_sparse_retrieval.py",
    "pipeline/phase2_composite_fts_retrieval.py",
    "pipeline/contextual_note_taker_fallback.py",
}


CONTEXT_SYSTEM_V4 = (
    "তুমি বাংলা context-only factuality verifier; শুধু প্রদত্ত context ব্যবহার করবে। "
    "CONTEXT-ONLY VERIFICATION CONTRACT-এর ক্রম মেনে "
    "Recovered checks: question-context-response exact slot; Unicode NFC/joiner/digit; "
    "grammar theory/definition/rule; negation/quantifier/comparator/modality/clause scope; "
    "entity-relation-property; event phase; bounded math। "
    "প্রথমে question premise/answerability, তারপর exact entity-relation-property-answer type, event phase, "
    "time/date/number/unit এবং negation-quantifier-comparator-modality scope বাঁধবে। একই passage-এর অন্য "
    "question, slot, role, place, event বা date-এর answer গ্রহণ করবে না। Direct statement ছাড়াও কেবল supplied "
    "context-এর definition/theory/rule/formula এবং স্পষ্ট operands দিয়ে bounded math, age/duration, relative "
    "timeline, ordinal ও kinship relation করা যাবে। Grammar-এ exact operation/term/operands এবং conventional "
    "pair/register মানবে; context-এ theory/rule থাকলে তার সঠিক application supported হতে পারে। Unicode NFC, "
    "Bengali/ASCII digit, joiner বা attested OCR word-break comparison literal evidence বদলায় না এবং near-looking "
    "ভিন্ন শব্দকে সমান করে না। Partial containment কোনো unsupported extra claim ঢাকে না। বাইরের জ্ঞান, retrieval, "
    "web, lexical cache বা closed-book corpus নিষিদ্ধ। Ambiguous/conflicting/missing evidence, invalid premise, "
    "কোনো local window-এ তথ্য না থাকা refutation নয়; সব window-এর পরে unresolved conflict হলে not_enough_information। "
    "শুধু exact recognized antonym/idiom/prefix shell-এ hash-bound lexical table nonterminal evidence হতে পারে; "
    "exact operation, key, sense/register এবং conflict_status=none আবশ্যক। Fuzzy/generic corpus/Wikipedia নিষিদ্ধ। "
    "Samas/sandhi/affix/natva/satva supplied rule ও exact operands ছাড়া lookup করবে না। "
    "শুধু {\"verdict\":\"supported\"}, {\"verdict\":\"refuted\"}, অথবা "
    "{\"verdict\":\"not_enough_information\"} দাও।"
)

CLOSED_SYSTEM_V3 = (
    "তুমি বাংলা closed-book factuality verifier। নিচের offline retrieval কেবল nonterminal evidence; "
    "কোন hit, keyed answer, consensus candidate বা retrieval score নিজে verdict নয়। Retrieval miss মানে "
    "NEI, refutation নয়। প্রথমে exact requested operation, question slot, entity, relation/property, "
    "answer type, event/date, number/unit, negation এবং time/cultural scope মিলাও। তারপর সমানভাবে aligned "
    "support ও counter-evidence বিচার করো। Books/user OCR/BCS এবং curated sources Wikipedia-র উপরে, কিন্তু "
    "misaligned উচ্চ-authority evidence aligned নিম্ন-authority evidence-কে হারাতে পারবে না; Wikipedia "
    "corroboration-only। Antonym/idiom/prefix/sandhi/samas-এ exact operation, term, pair, sense ও register ছাড়া "
    "semantic near-match গ্রহণ করবে না। Evidence না থাকলে বা conflict unresolved হলে "
    "not_enough_information। শুধু {\"verdict\":\"supported\"}, {\"verdict\":\"refuted\"}, অথবা "
    "{\"verdict\":\"not_enough_information\"} দাও।"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def load_retrieval_identity(package: Path) -> tuple[str, str, dict[str, Any]]:
    manifest_path = package / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "final combined strict-v3 retrieval package does not exist: "
            f"{manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict):
        raise ValueError("strict-v3 bundle manifest must be an object")
    dataset_id = str(manifest.get("dataset_id") or "")
    manifest_id = str(manifest.get("manifest_id") or "")
    core = {key: value for key, value in manifest.items() if key != "manifest_id"}
    expected = hashlib.sha256(canonical(core).encode("utf-8")).hexdigest()
    if manifest_id != expected or len(manifest_id) != 64:
        raise ValueError("strict-v3 bundle manifest identity mismatch")
    if not dataset_id or "retrieval" not in dataset_id or "v3" not in dataset_id:
        raise ValueError("strict-v3 retrieval dataset identity is invalid")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("strict-v3 bundle manifest files must be a list")
    declared_paths = {
        str(item.get("path") or "") for item in files if isinstance(item, dict)
    }
    missing = sorted(REQUIRED_RETRIEVAL_PAYLOAD_PATHS - declared_paths)
    if missing:
        raise ValueError(
            "combined private-v2 plus strict-v3 retrieval payload is incomplete: "
            + ", ".join(missing)
        )
    return dataset_id, manifest_id, manifest


def _replace_once(code: str, old: str, new: str) -> str:
    if code.count(old) != 1:
        raise RuntimeError(f"runner transformation anchor drifted: {old[:120]}")
    return code.replace(old, new)


def _replace_assignment_block(
    code: str, name: str, next_name: str, replacement: str
) -> str:
    start = code.index(f"{name} = (")
    end = code.index(f"\n{next_name} = (", start)
    return code[:start] + f'{name} = {replacement!r}\n' + code[end + 1 :]


def transformed_runner(dataset_id: str, manifest_id: str) -> str:
    code = SOURCE.read_text(encoding="utf-8")
    code = _replace_once(
        code,
        'VERSION = "bichar-phase2-gemma4-e4b-base-heavy-hybrid-v1"',
        'VERSION = "morichika-phase2-generalized-hybrid-v4"',
    )
    code = _replace_once(
        code,
        'RUN_ROOT = WORKING_ROOT / "bichar_phase2"',
        'RUN_ROOT = WORKING_ROOT / "morichika_phase2_generalized_v4"',
    )
    code = _replace_once(
        code,
        'RETRIEVAL_DATASET_ID = "ishtyy/morichika-phase2-retrieval-v2-20260720"',
        f"RETRIEVAL_DATASET_ID = {dataset_id!r}",
    )
    code = _replace_once(
        code,
        'RETRIEVAL_MANIFEST_ID = "38523490f9a1ecf539b0412259a3549a6a31daaa1e491d96477574017d252fb3"',
        f"RETRIEVAL_MANIFEST_ID = {manifest_id!r}",
    )
    code = _replace_once(code, "MAX_NEW_TOKENS = 64", "MAX_NEW_TOKENS = 16")
    code = _replace_once(code, "import time\n", "import time\nimport unicodedata\n")
    code = _replace_assignment_block(
        code, "CONTEXT_SYSTEM", "CLOSED_SYSTEM", CONTEXT_SYSTEM_V4
    )
    # CLOSED_SYSTEM is followed by build_messages rather than another
    # assignment, so replace its exact source block separately.
    closed_start = code.index("CLOSED_SYSTEM = (")
    closed_end = code.index("\n\n\ndef build_messages", closed_start)
    code = (
        code[:closed_start]
        + f"CLOSED_SYSTEM = {CLOSED_SYSTEM_V3!r}"
        + code[closed_end:]
    )

    payload_helper_anchor = "def materialize_retrieval_payload() -> tuple[Path, dict[str, Any]]:\n"
    payload_helper = r'''def _materialize_bound_payload_zip() -> tuple[Path, dict[str, Any]] | None:
    """Safely expand the one manifest-bound payload.zip Kaggle may expose."""
    matches = []
    for archive_path in INPUT_ROOT.rglob("payload.zip"):
        if not archive_path.is_file() or archive_path.is_symlink():
            continue
        try:
            with zipfile.ZipFile(archive_path) as archive:
                infos = archive.infolist()
                if len({info.filename for info in infos}) != len(infos):
                    raise HybridFailure("duplicate_monolithic_retrieval_member")
                manifest_infos = []
                for info in infos:
                    path = PurePosixPath(info.filename)
                    mode = (info.external_attr >> 16) & 0xFFFF
                    if (
                        path.is_absolute() or ".." in path.parts or "\\" in info.filename
                        or info.flag_bits & 1 or stat.S_ISLNK(mode)
                    ):
                        raise HybridFailure(f"unsafe_monolithic_retrieval_member:{info.filename}")
                    if not info.is_dir() and path.name == "bundle_manifest.json":
                        if info.file_size > 2 * 1024 * 1024:
                            raise HybridFailure("retrieval_bundle_manifest_oversized")
                        manifest_infos.append(info)
                for info in manifest_infos:
                    value = json.loads(archive.read(info).decode("utf-8-sig"))
                    if (
                        isinstance(value, dict)
                        and value.get("dataset_id") == RETRIEVAL_DATASET_ID
                        and value.get("manifest_id") == RETRIEVAL_MANIFEST_ID
                    ):
                        matches.append((archive_path, info, value))
        except (zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError):
            continue
    if not matches:
        return None
    if len(matches) != 1:
        raise HybridFailure(f"monolithic_retrieval_bundle_not_unique:{len(matches)}")
    archive_path, manifest_info, manifest = matches[0]
    core = {key: value for key, value in manifest.items() if key != "manifest_id"}
    if digest(core) != RETRIEVAL_MANIFEST_ID:
        raise HybridFailure("private_retrieval_manifest_identity_mismatch")
    specs = {str(spec["path"]): spec for spec in manifest.get("files", [])}
    prefix = PurePosixPath(manifest_info.filename).parent
    mapped = {}
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            path = PurePosixPath(info.filename)
            if prefix.parts:
                if tuple(path.parts[:len(prefix.parts)]) != tuple(prefix.parts):
                    raise HybridFailure(f"monolithic_retrieval_mixed_prefix:{info.filename}")
                relative = PurePosixPath(*path.parts[len(prefix.parts):]).as_posix()
            else:
                relative = path.as_posix()
            if relative == "bundle_manifest.json":
                continue
            if relative not in specs or relative in mapped:
                raise HybridFailure(f"undeclared_or_duplicate_monolithic_member:{relative}")
            mapped[relative] = info
    if set(mapped) != set(specs):
        raise HybridFailure("monolithic_retrieval_archive_incomplete")
    destination = RUN_ROOT / "payload_zip_input"
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(destination / "bundle_manifest.json", manifest)
    for relative, info in mapped.items():
        spec = specs[relative]
        if info.file_size != int(spec["bytes"]):
            raise HybridFailure(f"retrieval_archive_member_size_mismatch:{relative}")
        _copy_archive_member(
            archive_path, info, destination.joinpath(*PurePosixPath(relative).parts),
            int(spec["bytes"]), str(spec["sha256"]),
        )
    return destination / "bundle_manifest.json", manifest


'''
    code = _replace_once(code, payload_helper_anchor, payload_helper + payload_helper_anchor)
    code = _replace_once(
        code,
        '''    if not manifests:
        raise HybridFailure("private_retrieval_bundle_missing")
    manifest_path, manifest = sorted(manifests, key=lambda item: (len(item[0].parts), str(item[0])))[0]
''',
        '''    if not manifests:
        materialized = _materialize_bound_payload_zip()
        if materialized is None:
            raise HybridFailure("private_retrieval_bundle_missing")
        manifests.append(materialized)
    manifest_path, manifest = sorted(manifests, key=lambda item: (len(item[0].parts), str(item[0])))[0]
''',
    )

    helper_anchor = '''def compact_retrieval(item: dict[str, Any], lexical: list[dict[str, Any]], limit: int = 8) -> str:
'''
    helper = '''TERMINAL_ELIGIBLE_EXACT_SOURCES = frozenset({
    "bangla_mmlu", "nctb_qa_87805", "nctb_education_aux",
})
_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_NUMERIC_COUNTER_GAP = re.compile(r"(?<=[0-9])\\s+(?=(?:টি|টা|জন|খানা|খানি)(?:\\s|$))")


def exact_answer_key(value: object) -> str:
    text = unicodedata.normalize("NFC", str(value or "")).translate(_BN_DIGITS)
    text = " ".join(text.casefold().split())
    return _NUMERIC_COUNTER_GAP.sub("", text)


def closed_exact_terminal(row: dict[str, Any], item: dict[str, Any]) -> dict[str, Any] | None:
    """Admit only conflict-free, operation/slot-aligned exact keyed evidence.

    Strict-v3 FTS, Wikipedia, WordNet, generated answers, fuzzy matches and
    containment are structurally unable to pass this gate.
    """

    if row.get("context_available") is True:
        return None
    merged = item.get("merged_source_candidate") or {}
    verdict = merged.get("verdict")
    if (
        verdict not in (0, 1)
        or merged.get("status") != "source_consensus_candidate"
        or merged.get("strict_exact_key_conflicts")
    ):
        return None
    response_key = exact_answer_key(row.get("model_response_bn"))
    if not response_key:
        return None
    candidates = list(item.get("retrieval_candidates") or [])
    qualified = []
    for candidate in candidates:
        source_verdict = candidate.get("source_verdict_candidate") or {}
        candidate_verdict = source_verdict.get("verdict")
        answer_keys = {exact_answer_key(value) for value in candidate.get("answers") or []}
        choice_keys = {exact_answer_key(value) for value in candidate.get("choices") or []}
        rule = str(source_verdict.get("rule") or "")
        aligned = (
            candidate.get("source_id") in TERMINAL_ELIGIBLE_EXACT_SOURCES
            and candidate.get("exact_normalized") is True
            and candidate.get("model_facing_eligible") is True
            and candidate.get("semantic_alignment_tier") == 0
            and candidate.get("slot_compatibility_tier") == 0
            and candidate.get("policy_compatibility_tier") in (0, 1)
            and candidate.get("number_set_match") is not False
            and candidate.get("negation_set_match") is not False
            and "containment" not in rule
        )
        exact_response_relation = (
            candidate_verdict == 1 and response_key in answer_keys
        ) or (
            candidate_verdict == 0
            and response_key in choice_keys
            and response_key not in answer_keys
        )
        if aligned and exact_response_relation:
            qualified.append(candidate)
    if not qualified or {int((value["source_verdict_candidate"])["verdict"]) for value in qualified} != {int(verdict)}:
        return None

    # An admitted exact candidate playing the opposite response-aware role is
    # a conflict even if it has no legacy terminal adapter.  Candidate order
    # is irrelevant: this is a set-style veto over the complete admitted pool.
    opposite_role = "counter_candidate" if int(verdict) == 1 else "support_candidate"
    for candidate in candidates:
        observed_role = candidate.get("evidence_role")
        if not observed_role:
            relation = str(candidate.get("response_answer_relation") or "none")
            observed_role = (
                "support_candidate"
                if relation == "exact"
                else (
                    "counter_candidate"
                    if candidate.get("exact_normalized") is True
                    and bool(candidate.get("answers"))
                    else "neutral_candidate"
                )
            )
        if (
            candidate not in qualified
            and candidate.get("exact_normalized") is True
            and candidate.get("model_facing_eligible") is True
            and candidate.get("semantic_alignment_tier") == 0
            and candidate.get("slot_compatibility_tier") == 0
            and candidate.get("policy_compatibility_tier") in (0, 1)
            and observed_role == opposite_role
        ):
            return None
    return {
        "verdict": int(verdict),
        "status": "closed_exact_key_terminal",
        "source_ids": sorted({str(value["source_id"]) for value in qualified}),
        "rules": sorted({str(value["source_verdict_candidate"]["rule"]) for value in qualified}),
        "exact_normalized": True,
        "conflict_free": True,
    }


'''
    code = _replace_once(code, helper_anchor, helper + helper_anchor)

    old_context_branch = r'''    if row["context_available"]:
        context = row["model_context_bn"]
        route = router(context, question, response, max_windows=6)
        if len(context) <= 6000:
            evidence = context
            mode = "complete_context"
        else:
            evidence = "\n\n".join(
                f"[chars={value['context_char_start']}:{value['context_char_end']}; sha256={value['literal_span_sha256']}]\n{value['literal_text']}"
                for value in route["selected_windows"]
            )[:7000]
            mode = "primary_question_conditioned_context_windows"
        notes = canonical({
            "selected_notes": route.get("selected_notes", []),
            "bounded_derivation_candidates": route.get("bounded_derivation_candidates", []),
        })
        user = f"QUESTION:\n{question}\n\nSUPPLIED CONTEXT EVIDENCE ({mode}):\n{evidence}\n\nBOUNDED SOURCE-LINKED NOTES:\n{notes}\n\nRESPONSE TO VERIFY:\n{response}"
        return [{"role": "system", "content": CONTEXT_SYSTEM}, {"role": "user", "content": user}], route
'''
    new_context_branch = '''    if row["context_available"]:
        user, route = build_contextual_policy_packet(
            row["model_context_bn"], question, response, router,
            lexical_records=lexical_records, max_windows=8, full_context_char_limit=6000,
        )
        return [{"role": "system", "content": CONTEXT_SYSTEM}, {"role": "user", "content": user}], route
'''
    code = _replace_once(code, old_context_branch, new_context_branch)

    request_anchor = '''            body = requests.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
'''
    map_aggregate = '''            window_results = []
            if row["context_available"] and route_receipt.get("requires_window_aggregation"):
                window_calls = list(route_receipt.get("window_calls") or [])
                if not window_calls:
                    raise HybridFailure("context_window_plan_empty")
                for call in window_calls:
                    window_messages = [
                        {"role": "system", "content": CONTEXT_SYSTEM},
                        {"role": "user", "content": call["user"]},
                    ]
                    window_body = requests.post(
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        json={
                            "messages": window_messages, "temperature": 0.0, "top_p": 1.0,
                            "top_k": 40, "min_p": 0.05, "repeat_penalty": 1.0,
                            "seed": BASE_SEED + int(row["source_index"]) + int(call["window_index"]),
                            "max_tokens": MAX_NEW_TOKENS, "grammar": VERDICT_ONLY_GBNF,
                            "chat_template_kwargs": {"enable_thinking": False}, "stream": False,
                        }, timeout=900,
                    )
                    window_body.raise_for_status()
                    window_payload = window_body.json()
                    window_content = ((window_payload["choices"][0].get("message") or {}).get("content"))
                    if window_content not in VERDICT_ONLY_LITERALS:
                        window_content = '{"verdict":"not_enough_information"}'
                    window_results.append({
                        "window_index": call["window_index"],
                        "context_char_start": call["context_char_start"],
                        "context_char_end": call["context_char_end"],
                        "literal_span_sha256": call["literal_span_sha256"],
                        "literal_excerpt": call["literal_excerpt"],
                        "excerpt_char_start": call["excerpt_char_start"],
                        "excerpt_char_end": call["excerpt_char_end"],
                        "literal_excerpt_sha256": call["literal_excerpt_sha256"],
                        "window_verdict": json.loads(window_content)["verdict"],
                    })
                aggregate_user = build_aggregation_user(
                    row["model_prompt_bn"], row["model_response_bn"], window_results,
                    selected_notes=route_receipt.get("aggregation_selected_notes", []),
                    bounded_derivations=route_receipt.get("aggregation_bounded_derivations", []),
                    lexical_policy=route_receipt.get("contextual_lexical_policy"),
                )
                messages = [
                    {"role": "system", "content": CONTEXT_SYSTEM},
                    {"role": "user", "content": aggregate_user},
                ]
                route_receipt["window_result_ledger"] = window_results
                route_receipt["aggregation_prompt_sha256"] = hashlib.sha256(aggregate_user.encode("utf-8")).hexdigest()
                route_receipt["window_verdict_counts"] = {
                    verdict: sum(result["window_verdict"] == verdict for result in window_results)
                    for verdict in ("supported", "refuted", "not_enough_information")
                }
                route_receipt.pop("window_calls", None)
            body = requests.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
'''
    code = _replace_once(code, request_anchor, map_aggregate)

    parsed_anchor = '''            parsed = VERDICT_ONLY_LITERALS.get(content)
            label = int(parsed) if parsed in (0, 1) else 0
            method = "gbnf_verdict_only" if parsed in (0, 1) else "invalid_generation_fail_closed_nei"
'''
    conflict_safe_parse = '''            parsed = VERDICT_ONLY_LITERALS.get(content)
            window_verdicts = {result["window_verdict"] for result in window_results}
            cross_window_conflict = {"supported", "refuted"} <= window_verdicts
            if cross_window_conflict:
                parsed = 0
                route_receipt["cross_window_conflict_forced_nei"] = True
                method = "cross_window_conflict_fail_closed_nei"
            else:
                method = "gbnf_verdict_only" if parsed in (0, 1) else "invalid_generation_fail_closed_nei"
            label = int(parsed) if parsed in (0, 1) else 0
'''
    code = _replace_once(code, parsed_anchor, conflict_safe_parse)

    checkpoint_anchor = '''        outputs.append(output)
        append_checkpoint(checkpoint, outputs)
'''
    contextual_checkpoint = '''        if row["context_available"]:
            output["context_diagnostic"] = {
                "policy_version": route_receipt.get("context_policy_version"),
                "policy_family_inventory_count": route_receipt.get("policy_family_inventory_count"),
                "canonical_policy_family_count": route_receipt.get("canonical_policy_family_count"),
                "engineered_evaluation_cell_count": route_receipt.get("engineered_evaluation_cell_count"),
                "operation_axis_count": route_receipt.get("operation_axis_count"),
                "canonical_policy_families": route_receipt.get("canonical_policy_families"),
                "engineered_evaluation_cells": route_receipt.get("engineered_evaluation_cells"),
                "operation_axis": route_receipt.get("operation_axis"),
                "detected_policy_families": route_receipt.get("detected_policy_families", []),
                "evidence_mode": route_receipt.get("evidence_mode"),
                "full_context_inference_coverage": route_receipt.get("full_context_inference_coverage"),
                "context_sha256": route_receipt.get("context_sha256"),
                "question_sha256": route_receipt.get("question_sha256"),
                "response_sha256": route_receipt.get("response_sha256"),
                "prompt_sha256": route_receipt.get("prompt_sha256"),
                "policy_receipt_sha256": route_receipt.get("receipt_sha256"),
                "external_retrieval_allowed": route_receipt.get("external_retrieval_allowed"),
                "window_count": route_receipt.get("window_count"),
                "window_verdict_counts": route_receipt.get("window_verdict_counts"),
                "cross_window_conflict_forced_nei": route_receipt.get("cross_window_conflict_forced_nei", False),
                "lexical_policy": route_receipt.get("contextual_lexical_policy"),
            }
        outputs.append(output)
        append_checkpoint(checkpoint, outputs)
'''
    code = _replace_once(code, checkpoint_anchor, contextual_checkpoint)

    terminal_branch = '''        terminal = (item.get("merged_source_candidate") or {}).get("verdict")
        if not row["context_available"] and terminal in (0, 1):
            label = int(terminal)
            method = "exact_source_consensus_candidate"
            route_receipt = {"terminal_source_candidate": True}
            finish_reason = "deterministic"
            prompt_tokens = completion_tokens = 0
        elif time.perf_counter() - started > DEADLINE_SECONDS:
'''
    exact_terminal_branch = '''        terminal = closed_exact_terminal(row, item)
        if terminal is not None:
            label = int(terminal["verdict"])
            method = "closed_exact_key_terminal"
            route_receipt = terminal
            finish_reason = "deterministic"
            prompt_tokens = completion_tokens = 0
        elif time.perf_counter() - started > DEADLINE_SECONDS:
'''
    code = _replace_once(code, terminal_branch, exact_terminal_branch)

    old_build = '''        retrieval_receipt = build_retrieval(normalized_path, retrieval_dir, top_k=6, batch_size=128, composite_cache_dir=None)
        retrieved = retrieval_map(retrieval_dir / "retrieval.jsonl")
'''
    new_build = '''        strict_composite = stage / "artifacts/retrieval/phase2_strict_rights_safe_fts_v3_final"
        if not (strict_composite / "manifest.json").is_file():
            raise HybridFailure("strict_v3_composite_cache_missing")
        retrieval_receipt = build_retrieval(
            normalized_path, retrieval_dir, top_k=8, batch_size=128,
            composite_cache_dir=strict_composite,
            composite_query_mode="all_closed",
        )
        composite_receipt = retrieval_receipt.get("composite_fts") or {}
        if (
            composite_receipt.get("enabled") is not True
            or composite_receipt.get("terminal_label_authority") is not False
            or composite_receipt.get("closed_book_only") is not True
        ):
            raise HybridFailure("strict_v3_composite_runtime_contract_failed")
        retrieved = retrieval_map(retrieval_dir / "retrieval.jsonl")
'''
    code = _replace_once(code, old_build, new_build)

    code = _replace_once(
        code,
        '            "contextual_external_retrieval_allowed": False,\n            "current_affairs_included": False,',
        '            "contextual_external_retrieval_allowed": False,\n'
        '            "retrieval_terminal_label_authority": False,\n'
        '            "retrieval_miss_means": "NEI_not_refutation",\n'
        '            "closed_sources": "combined_private_v2_plus_strict_v3",\n'
        '            "context_terminal_policy": "no_production_terminal_without_source_disjoint_held_admission",\n'
        '            "context_policy_contract": "morichika-context-policy-v4-full-coverage-map-aggregate",\n'
        '            "canonical_context_policy_families": 17,\n'
        '            "engineered_context_evaluation_cells": 26,\n'
        '            "context_operation_axis": 15,\n'
        '            "current_affairs_included": False,',
    )

    forbidden = ("route_audit", "phase1_", "gold_label", "gold_labels")
    lowered = code.casefold()
    leaked = [token for token in forbidden if token in lowered]
    if leaked:
        raise RuntimeError(f"v3 production runner contains forbidden Phase 1 material: {leaked}")
    required = (
        "strict_v3_composite_cache_missing",
        'composite_query_mode="all_closed"',
        '"retrieval_miss_means": "NEI_not_refutation"',
        "closed_exact_key_terminal",
        "build_contextual_policy_packet(",
        '"canonical_context_policy_families": 17',
        '"engineered_context_evaluation_cells": 26',
        '"context_operation_axis": 15',
        'output["context_diagnostic"]',
        "CONTEXT_SYSTEM =",
        "CLOSED_SYSTEM =",
    )
    missing = [token for token in required if token not in code]
    if missing:
        raise RuntimeError(f"v3 runner contract missing: {missing}")
    compile(code, str(SOURCE), "exec")
    return code


def build(output: Path = OUT, retrieval_package: Path = RETRIEVAL_PACKAGE) -> dict[str, Any]:
    dataset_id, manifest_id, bundle = load_retrieval_identity(retrieval_package)
    context_trust = json.loads(CONTEXT_TRUST_POLICY.read_text(encoding="utf-8-sig"))
    if (
        context_trust.get("version")
        != "phase2-context-rule-trust-policy-v1-source-disjoint-intersection"
        or context_trust.get("admitted_pairs") != []
    ):
        raise ValueError(
            "v3 context terminal policy changed; explicit held-policy review required"
        )
    stage = output.with_name(output.name + ".staging")
    prior = output.with_name(output.name + ".previous")
    for path in (stage, prior):
        if path.exists():
            if path.parent != output.parent or not path.name.startswith(output.name):
                raise RuntimeError(f"unsafe cleanup path: {path}")
            shutil.rmtree(path)
    stage.mkdir(parents=True)

    code = transformed_runner(dataset_id, manifest_id)
    context_runtime_source = CONTEXT_RUNTIME_V4.read_text(encoding="utf-8")
    compile(context_runtime_source, str(CONTEXT_RUNTIME_V4), "exec")
    shutil.copy2(CONTEXT_RUNTIME_V4, stage / CONTEXT_RUNTIME_V4.name)
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# MORICHIKA Phase 2 — generalized hybrid v4\n",
                    "Context-only policy routing plus combined private-v2 and strict-v3 "
                    "nonterminal closed-book evidence. Writes submission.csv; never submits.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in context_runtime_source.splitlines()],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in code.splitlines()],
            },
        ],
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path = stage / NOTEBOOK_NAME
    atomic_json(notebook_path, notebook)
    metadata = {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": NOTEBOOK_NAME,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [*MODEL_DATASETS, dataset_id],
        "kernel_sources": [],
        "competition_sources": ["bengali-hallucination"],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    metadata_path = stage / "kernel-metadata.json"
    atomic_json(metadata_path, metadata)

    receipt: dict[str, Any] = {
        "version": "morichika-phase2-generalized-hybrid-v4-build-receipt",
        "status": "built_private_offline_not_pushed_not_launched_not_submitted",
        "kernel_id": KERNEL_ID,
        "retrieval_dataset_id": dataset_id,
        "retrieval_manifest_id": manifest_id,
        "retrieval_package_source_count": (bundle.get("source_counts") or {}).get(
            "package_sources"
        ),
        "source_runner": SOURCE.relative_to(ROOT).as_posix(),
        "source_runner_sha256": sha256_file(SOURCE),
        "embedded_runner_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "notebook": NOTEBOOK_NAME,
        "notebook_bytes": notebook_path.stat().st_size,
        "notebook_sha256": sha256_file(notebook_path),
        "metadata_bytes": metadata_path.stat().st_size,
        "metadata_sha256": sha256_file(metadata_path),
        "architecture": {
            "contextual_external_retrieval": False,
            "contextual_default": "note_taker_plus_model",
            "context_policy_contract": "morichika-context-policy-v4-full-coverage-map-aggregate",
            "canonical_context_policy_families": 17,
            "engineered_context_evaluation_cells": 26,
            "context_operation_axis": 15,
            "long_context_adjudication": "all_windows_then_final_aggregation",
            "context_terminal_allowlist": [],
            "closed_retrieval": "combined_private_v2_plus_strict_v3",
            "closed_retrieval_terminal": "exact_aligned_conflict_free_key_only",
            "strict_v3_fts_terminal": False,
            "closed_model_residual": True,
            "retrieval_miss": "NEI_not_refutation",
            "phase1_ids_labels_route_audit_used": False,
        },
        "context_terminal_admission": {
            "policy": CONTEXT_TRUST_POLICY.relative_to(ROOT).as_posix(),
            "policy_sha256": sha256_file(CONTEXT_TRUST_POLICY),
            "canonical_sha256": context_trust["canonical_sha256"],
            "source_disjoint_admitted_pairs": 0,
            "effect": "all_context_policy_families_use_note_taker_plus_model",
        },
        "private": True,
        "offline": True,
        "t4x2_required": True,
        "push_performed": False,
        "launch_performed": False,
        "submission_performed": False,
        "context_runtime": CONTEXT_RUNTIME_V4.relative_to(ROOT).as_posix(),
        "context_runtime_sha256": sha256_file(CONTEXT_RUNTIME_V4),
    }
    receipt["receipt_id"] = hashlib.sha256(
        canonical(receipt).encode("utf-8")
    ).hexdigest()
    receipt_path = stage / "MORICHIKA_V4_BUILD_RECEIPT.json"
    atomic_json(receipt_path, receipt)
    atomic_json(
        stage / "MORICHIKA_V4_READY.json",
        {
            "status": "READY_LOCAL_PACKAGE_NOT_PUSHED",
            "receipt_id": receipt["receipt_id"],
            "receipt_sha256": sha256_file(receipt_path),
            "notebook_sha256": receipt["notebook_sha256"],
            "metadata_sha256": receipt["metadata_sha256"],
        },
    )

    if output.exists():
        output.replace(prior)
    stage.replace(output)
    if prior.exists():
        shutil.rmtree(prior)
    return receipt


def main() -> None:
    receipt = build()
    print(
        json.dumps(
            {
                "output": str(OUT),
                "kernel_id": KERNEL_ID,
                "retrieval_dataset_id": receipt["retrieval_dataset_id"],
                "retrieval_manifest_id": receipt["retrieval_manifest_id"],
                "notebook_sha256": receipt["notebook_sha256"],
                "receipt_id": receipt["receipt_id"],
                "push_performed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
