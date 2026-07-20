"""Build the private offline MORICHIKA generalized-v4 LoRA Kaggle variant.

This is deliberately a thin, audited derivative of the final generalized-v4
runner.  It changes only the model binding (the pinned step-10 LoRA is applied
with llama.cpp ``--lora``), output identity, and model receipt.  Retrieval,
context isolation, the corrected 17-family/26-cell/15-operation policy runtime,
full-context map/aggregate adjudication, and fail-closed exact
terminal rules remain byte-for-byte inherited from the base builder.

The builder performs no upload, launch, or competition submission.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import build_morichika_phase2_generalized_hybrid_v3 as base


OUT = ROOT / "artifacts/kaggle/morichika_phase2_generalized_lora_hybrid_v4_kernel_20260720"
NOTEBOOK_NAME = "morichika-phase2-generalized-lora-hybrid-v4.ipynb"
KERNEL_ID = "ishtyy/morichika-p2-generalized-lora-v4-20260720"
KERNEL_TITLE = "MORICHIKA P2 Generalized LoRA v4 20260720"

ADAPTER_DATASET_ID = "ishtyy/bichar-gemma4-e4b-step10-lora-gguf-20260720"
ADAPTER_FILE = "gemma4-e4b-lr2e6-verdict45-step10-f16.gguf"
ADAPTER_BYTES = 69_799_104
ADAPTER_SHA256 = "ebef35b50d943ccb94a0749993b20aa408331d0615a2639525be263cdfe8a8e2"
ADAPTER_MANIFEST_ID = "67b234323047dbf932f43133ccde5d1de720aea8f15b418a32fa90101ec9ee0b"
ADAPTER_CONVERSION_ID = "2ee373a1aff488777a3442a2ec0d085955b14726db4439987f0ad95954313526"
ADAPTER_RIGHTS_ID = "875e4b7eeae22f3266507fb202574f3adcf7612070f439fac44db2d36e00d897"


def _replace_once(code: str, old: str, new: str) -> str:
    if code.count(old) != 1:
        raise RuntimeError(f"LoRA transformation anchor drifted: {old[:120]}")
    return code.replace(old, new)


def transformed_lora_runner(dataset_id: str, manifest_id: str) -> str:
    """Return final generalized-v4 with the single pinned adapter applied."""
    code = base.transformed_runner(dataset_id, manifest_id)
    code = _replace_once(
        code,
        'VERSION = "morichika-phase2-generalized-hybrid-v4"',
        'VERSION = "morichika-phase2-generalized-lora-hybrid-v4"',
    )
    code = _replace_once(
        code,
        'RUN_ROOT = WORKING_ROOT / "morichika_phase2_generalized_v4"',
        'RUN_ROOT = WORKING_ROOT / "morichika_phase2_generalized_lora_v4"',
    )
    model_anchor = (
        'MODEL_SHA256 = "df0fd4ee07072c607c29a0a1cb4f98918426cca12f45a2776bdd6ee6d09a4de3"\n'
    )
    adapter_constants = model_anchor + (
        f'ADAPTER_DATASET_ID = {ADAPTER_DATASET_ID!r}\n'
        f'ADAPTER_FILE = {ADAPTER_FILE!r}\n'
        f'ADAPTER_BYTES = {ADAPTER_BYTES}\n'
        f'ADAPTER_SHA256 = {ADAPTER_SHA256!r}\n'
        f'ADAPTER_MANIFEST_ID = {ADAPTER_MANIFEST_ID!r}\n'
        f'ADAPTER_CONVERSION_ID = {ADAPTER_CONVERSION_ID!r}\n'
        f'ADAPTER_RIGHTS_ID = {ADAPTER_RIGHTS_ID!r}\n'
    )
    code = _replace_once(code, model_anchor, adapter_constants)

    server_anchor = 'def server_command(binary: Path, model: Path, port: int) -> list[str]:\n'
    adapter_binding = r'''def resolve_adapter_binding() -> tuple[Path, dict[str, Any]]:
    """Resolve one rights- and manifest-bound private adapter dataset root."""
    matches = []
    for manifest_path in INPUT_ROOT.rglob("artifact_manifest.json"):
        if not manifest_path.is_file() or manifest_path.is_symlink():
            continue
        try:
            manifest = read_json(manifest_path)
        except Exception:
            continue
        if manifest.get("dataset_id") == ADAPTER_DATASET_ID:
            matches.append((manifest_path.parent.resolve(), manifest))
    unique = {str(root): (root, manifest) for root, manifest in matches}
    if len(unique) != 1:
        raise HybridFailure(f"adapter_dataset_root_not_unique:{len(unique)}")
    root, manifest = next(iter(unique.values()))
    core = {key: value for key, value in manifest.items() if key != "manifest_id"}
    if (
        manifest.get("manifest_id") != ADAPTER_MANIFEST_ID
        or manifest.get("manifest_id") != digest(core)
        or manifest.get("private") is not True
        or manifest.get("private_competition_labels_present") is not False
        or manifest.get("lora_file") != ADAPTER_FILE
        or manifest.get("lora_bytes") != ADAPTER_BYTES
        or manifest.get("lora_sha256") != ADAPTER_SHA256
    ):
        raise HybridFailure("adapter_manifest_contract_mismatch")
    for name, expected in manifest.get("files", {}).items():
        path = root / name
        if (
            not path.is_file() or path.is_symlink()
            or path.stat().st_size != int(expected.get("bytes", -1))
            or sha256_file(path) != expected.get("sha256")
        ):
            raise HybridFailure(f"adapter_file_binding_mismatch:{name}")
    conversion = read_json(root / "conversion_receipt.json")
    conversion_core = {
        key: value for key, value in conversion.items()
        if key != "conversion_receipt_id"
    }
    if (
        conversion.get("conversion_receipt_id") != ADAPTER_CONVERSION_ID
        or conversion.get("conversion_receipt_id") != digest(conversion_core)
        or conversion.get("lora_gguf", {}).get("sha256") != ADAPTER_SHA256
        or conversion.get("llama_cpp_commit") != LLAMA_COMMIT
        or conversion.get("base_weights_read_by_converter") is not False
    ):
        raise HybridFailure("adapter_conversion_contract_mismatch")
    rights = read_json(root / "rights_record.json")
    rights_core = {key: value for key, value in rights.items() if key != "rights_record_id"}
    if (
        rights.get("rights_record_id") != ADAPTER_RIGHTS_ID
        or rights.get("rights_record_id") != digest(rights_core)
        or rights.get("visibility") != "private"
        or rights.get("private_competition_labels_used") is not False
        or rights.get("raw_model_reasoning_included") is not False
    ):
        raise HybridFailure("adapter_rights_contract_mismatch")
    adapter = root / ADAPTER_FILE
    if (
        adapter.stat().st_size != ADAPTER_BYTES
        or sha256_file(adapter) != ADAPTER_SHA256
    ):
        raise HybridFailure("adapter_gguf_binding_mismatch")
    return adapter, manifest


def server_command(binary: Path, model: Path, adapter: Path, port: int) -> list[str]:
'''
    code = _replace_once(code, server_anchor, adapter_binding)
    code = _replace_once(
        code,
        '        str(binary), "-m", str(model), "-ngl", "99", "-c", "8192",',
        '        str(binary), "-m", str(model), "--lora", str(adapter), "-ngl", "99", "-c", "8192",',
    )
    code = _replace_once(
        code,
        '        model = resolve_bound_file(MODEL_FILE, MODEL_BYTES, MODEL_SHA256)\n'
        '        runtime_payload = resolve_bound_file(RUNTIME_FILE, RUNTIME_BYTES, RUNTIME_SHA256)',
        '        model = resolve_bound_file(MODEL_FILE, MODEL_BYTES, MODEL_SHA256)\n'
        '        adapter, adapter_manifest = resolve_adapter_binding()\n'
        '        runtime_payload = resolve_bound_file(RUNTIME_FILE, RUNTIME_BYTES, RUNTIME_SHA256)',
    )
    code = _replace_once(
        code,
        'server_command(binary, model, port)',
        'server_command(binary, model, adapter, port)',
    )
    code = _replace_once(
        code,
        '"model": {"file": MODEL_FILE, "bytes": MODEL_BYTES, "sha256": MODEL_SHA256, "adapter_used": False},',
        '"model": {\n'
        '                "file": MODEL_FILE, "bytes": MODEL_BYTES, "sha256": MODEL_SHA256,\n'
        '                "adapter_used": True, "adapter_dataset_id": ADAPTER_DATASET_ID,\n'
        '                "adapter_file": ADAPTER_FILE, "adapter_bytes": ADAPTER_BYTES,\n'
        '                "adapter_sha256": ADAPTER_SHA256,\n'
        '                "adapter_manifest_id": adapter_manifest["manifest_id"],\n'
        '                "adapter_server_argument": "--lora",\n'
        '            },',
    )

    forbidden = ("route_audit", "phase1_", "gold_label", "gold_labels")
    leaked = [token for token in forbidden if token in code.casefold()]
    if leaked:
        raise RuntimeError(f"LoRA generalized runner contains forbidden material: {leaked}")
    required = (
        '"canonical_policy_family_count"',
        'composite_query_mode="all_closed"',
        "closed_exact_key_terminal",
        "resolve_adapter_binding",
        '"--lora"',
        ADAPTER_SHA256,
    )
    missing = [token for token in required if token not in code]
    if missing:
        raise RuntimeError(f"LoRA generalized runner contract missing: {missing}")
    if code.count('"--lora"') != 2:
        raise RuntimeError("expected one server argument and one receipt --lora marker")
    compile(code, "<morichika-generalized-lora-v4>", "exec")
    return code


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def build(
    output: Path = OUT,
    retrieval_package: Path = base.RETRIEVAL_PACKAGE,
) -> dict[str, Any]:
    dataset_id, manifest_id, bundle = base.load_retrieval_identity(retrieval_package)
    stage = output.with_name(output.name + ".staging")
    prior = output.with_name(output.name + ".previous")
    for path in (stage, prior):
        if path.exists():
            if path.parent != output.parent or not path.name.startswith(output.name):
                raise RuntimeError(f"unsafe cleanup path: {path}")
            shutil.rmtree(path)
    stage.mkdir(parents=True)

    code = transformed_lora_runner(dataset_id, manifest_id)
    context_source = base.CONTEXT_RUNTIME_V4.read_text(encoding="utf-8")
    compile(context_source, str(base.CONTEXT_RUNTIME_V4), "exec")
    shutil.copy2(base.CONTEXT_RUNTIME_V4, stage / base.CONTEXT_RUNTIME_V4.name)
    notebook = {
        "cells": [
            {
                "cell_type": "markdown", "metadata": {},
                "source": [
                    "# MORICHIKA Phase 2 — generalized LoRA hybrid v4\n",
                    "Same final full-coverage corpus/policy router as base v4, with one hash-bound "
                    "private step-10 LoRA. Writes submission.csv; never submits.\n",
                ],
            },
            {
                "cell_type": "code", "execution_count": None,
                "metadata": {}, "outputs": [],
                "source": [line + "\n" for line in context_source.splitlines()],
            },
            {
                "cell_type": "code", "execution_count": None,
                "metadata": {}, "outputs": [],
                "source": [line + "\n" for line in code.splitlines()],
            },
        ],
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path = stage / NOTEBOOK_NAME
    _atomic_json(notebook_path, notebook)
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
        "dataset_sources": [*base.MODEL_DATASETS, ADAPTER_DATASET_ID, dataset_id],
        "kernel_sources": [],
        "competition_sources": ["bengali-hallucination"],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    metadata_path = stage / "kernel-metadata.json"
    _atomic_json(metadata_path, metadata)
    receipt: dict[str, Any] = {
        "version": "morichika-phase2-generalized-lora-hybrid-v4-build-receipt",
        "status": "built_private_offline_not_pushed_not_launched_not_submitted",
        "kernel_id": KERNEL_ID,
        "retrieval_dataset_id": dataset_id,
        "retrieval_manifest_id": manifest_id,
        "retrieval_package_source_count": (bundle.get("source_counts") or {}).get("package_sources"),
        "base_generalized_builder": base.__file__,
        "base_generalized_builder_sha256": base.sha256_file(Path(base.__file__)),
        "context_runtime": base.CONTEXT_RUNTIME_V4.relative_to(ROOT).as_posix(),
        "context_runtime_sha256": base.sha256_file(base.CONTEXT_RUNTIME_V4),
        "adapter": {
            "dataset_id": ADAPTER_DATASET_ID,
            "file": ADAPTER_FILE,
            "bytes": ADAPTER_BYTES,
            "sha256": ADAPTER_SHA256,
            "manifest_id": ADAPTER_MANIFEST_ID,
            "conversion_receipt_id": ADAPTER_CONVERSION_ID,
            "rights_record_id": ADAPTER_RIGHTS_ID,
            "server_argument": "--lora",
        },
        "architecture": {
            "contextual_external_retrieval": False,
            "context_policy_contract": "morichika-context-policy-v4-full-coverage-map-aggregate",
            "canonical_context_policy_families": 17,
            "engineered_context_evaluation_cells": 26,
            "context_operation_axis": 15,
            "long_context_adjudication": "all_windows_then_final_aggregation",
            "closed_retrieval": "combined_private_v2_plus_strict_v3",
            "closed_retrieval_terminal": "exact_aligned_conflict_free_key_only",
            "closed_model_residual": True,
            "retrieval_miss": "NEI_not_refutation",
            "phase1_ids_labels_route_audit_used": False,
        },
        "embedded_runner_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "notebook": NOTEBOOK_NAME,
        "notebook_bytes": notebook_path.stat().st_size,
        "notebook_sha256": base.sha256_file(notebook_path),
        "metadata_bytes": metadata_path.stat().st_size,
        "metadata_sha256": base.sha256_file(metadata_path),
        "private": True,
        "offline": True,
        "t4x2_required": True,
        "push_performed": False,
        "launch_performed": False,
        "submission_performed": False,
    }
    receipt["receipt_id"] = hashlib.sha256(base.canonical(receipt).encode("utf-8")).hexdigest()
    receipt_path = stage / "MORICHIKA_LORA_V4_BUILD_RECEIPT.json"
    _atomic_json(receipt_path, receipt)
    _atomic_json(stage / "MORICHIKA_LORA_V4_READY.json", {
        "status": "READY_LOCAL_PACKAGE_NOT_PUSHED",
        "receipt_id": receipt["receipt_id"],
        "receipt_sha256": base.sha256_file(receipt_path),
        "notebook_sha256": receipt["notebook_sha256"],
        "metadata_sha256": receipt["metadata_sha256"],
    })
    if output.exists():
        output.replace(prior)
    stage.replace(output)
    if prior.exists():
        shutil.rmtree(prior)
    return receipt


def main() -> None:
    print(json.dumps(build(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
