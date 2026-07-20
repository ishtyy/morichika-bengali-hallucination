"""Safe runtime reader for content-addressed Phase 2 sparse caches."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from pipeline.phase2_canonicalize import VERSION as CANONICALIZER_VERSION


VERSION = "phase2-mmap-sparse-cache-v2-idempotent-canonicalizer"
NORMALIZER = Path(__file__).resolve().with_name("phase2_canonicalize.py")
RIGHTS_BOOLEAN_FIELDS = (
    "bundle_allowed", "public_redistribution", "attribution_required",
    "noncommercial_only", "share_alike", "local_only",
    "targeted_evidence_only", "quarantined",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_cache_rights_policy(value: object) -> dict[str, Any]:
    """Validate the rights ledger embedded in every deployable mmap cache."""

    if not isinstance(value, dict):
        raise ValueError("mmap cache rights policy must be an object")
    for field in RIGHTS_BOOLEAN_FIELDS:
        if not isinstance(value.get(field), bool):
            raise ValueError(f"mmap cache rights policy requires Boolean {field}")
    if not str(value.get("note", "")).strip():
        raise ValueError("mmap cache rights policy requires a nonempty note")
    if value["local_only"] and value["public_redistribution"]:
        raise ValueError("local-only cache cannot permit public redistribution")
    return dict(value)


def _dtype(value: str) -> type[np.generic]:
    mapping: dict[str, type[np.generic]] = {
        "float32": np.float32,
        "float64": np.float64,
    }
    if value not in mapping:
        raise ValueError(f"unsupported vectorizer dtype: {value}")
    return mapping[value]


def load_safe_vectorizer(directory: Path, manifest: dict[str, Any]) -> TfidfVectorizer:
    config = json.loads((directory / "vectorizer.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((directory / "vocabulary.json").read_text(encoding="utf-8"))
    if not isinstance(vocabulary, list) or len(vocabulary) != int(config["vocabulary_size"]):
        raise ValueError("safe vectorizer vocabulary size mismatch")
    if len(vocabulary) != len(set(vocabulary)):
        raise ValueError("safe vectorizer vocabulary contains duplicates")
    vectorizer = TfidfVectorizer(
        analyzer=config["analyzer"],
        binary=bool(config["binary"]),
        decode_error=config["decode_error"],
        dtype=_dtype(config["dtype"]),
        encoding=config["encoding"],
        input=config["input"],
        lowercase=bool(config["lowercase"]),
        ngram_range=tuple(config["ngram_range"]),
        norm=config["norm"],
        smooth_idf=bool(config["smooth_idf"]),
        strip_accents=config["strip_accents"],
        sublinear_tf=bool(config["sublinear_tf"]),
        token_pattern=config["token_pattern"],
        use_idf=bool(config["use_idf"]),
        vocabulary={token: index for index, token in enumerate(vocabulary)},
    )
    if config["use_idf"]:
        idf = np.load(directory / "idf.npy", mmap_mode="r", allow_pickle=False)
        if idf.ndim != 1 or idf.shape[0] != len(vocabulary):
            raise ValueError("safe vectorizer IDF shape mismatch")
        vectorizer.idf_ = np.asarray(idf)
    if len(vocabulary) != int(manifest["matrix_shape"][1]):
        raise ValueError("safe vectorizer/matrix feature mismatch")
    return vectorizer


class CompressedSqliteRecords:
    def __init__(self, path: Path, records: int) -> None:
        self.path = path.resolve()
        self.records = int(records)
        uri = f"file:{self.path.as_posix()}?mode=ro&immutable=1"
        self.connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self.connection.execute("PRAGMA query_only=ON")

    def __len__(self) -> int:
        return self.records

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= int(index) < self.records:
            raise IndexError(index)
        row = self.connection.execute(
            "SELECT payload FROM records WHERE idx=?", (int(index),)
        ).fetchone()
        if row is None:
            raise IndexError(index)
        return json.loads(zlib.decompress(bytes(row[0])).decode("utf-8"))


class SqliteExactLookup:
    def __init__(self, records: CompressedSqliteRecords) -> None:
        self.records = records

    def get(self, key: str, default: Any = None) -> list[int] | Any:
        if not key:
            return default
        rows = self.records.connection.execute(
            "SELECT idx FROM exact_lookup WHERE question_key=? ORDER BY idx", (key,)
        ).fetchall()
        return [int(row[0]) for row in rows] if rows else default


def load_mmap_index(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("version") != VERSION:
        raise ValueError("unsupported Phase 2 mmap cache version")
    fingerprint = manifest.get("fingerprint") or {}
    if fingerprint.get("canonicalizer_version") != CANONICALIZER_VERSION:
        raise ValueError("Phase 2 mmap cache canonicalizer version mismatch")
    if fingerprint.get("normalizer_sha256") != sha256_file(NORMALIZER):
        raise ValueError("Phase 2 mmap cache canonicalizer hash mismatch")
    validate_cache_rights_policy(manifest.get("rights_policy"))
    declared = manifest.get("files") or {}
    expected_names = {
        "vectorizer.json", "vocabulary.json", "idf.npy", "data.npy",
        "indices.npy", "indptr.npy", "records.sqlite3",
    }
    if set(declared) != expected_names:
        raise ValueError("mmap cache file declaration mismatch")
    for name, metadata in declared.items():
        path = directory / name
        if (
            not path.is_file()
            or sha256_file(path) != str(metadata.get("sha256", ""))
            or path.stat().st_size != int(metadata.get("bytes", -1))
        ):
            raise ValueError(f"mmap cache hash/size mismatch: {name}")

    shape = tuple(int(value) for value in manifest["matrix_shape"])
    data = np.load(directory / "data.npy", mmap_mode="r", allow_pickle=False)
    indices = np.load(directory / "indices.npy", mmap_mode="r", allow_pickle=False)
    indptr = np.load(directory / "indptr.npy", mmap_mode="r", allow_pickle=False)
    if data.ndim != 1 or indices.shape != data.shape or indptr.shape != (shape[0] + 1,):
        raise ValueError("mmap CSR array shape mismatch")
    if int(indptr[-1]) != data.shape[0] or int(manifest["matrix_nnz"]) != data.shape[0]:
        raise ValueError("mmap CSR nnz mismatch")
    matrix = sparse.csr_matrix((data, indices, indptr), shape=shape, copy=False)
    records = CompressedSqliteRecords(directory / "records.sqlite3", shape[0])
    return {
        "directory": directory,
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "vectorizer": load_safe_vectorizer(directory, manifest),
        "matrix": matrix,
        "records": records,
        "exact_lookup": SqliteExactLookup(records),
        "cache_id": manifest["cache_id"],
        "cache_format": VERSION,
    }


def close_mmap_index(index: dict[str, Any]) -> None:
    """Release SQLite and NumPy mmap handles, including on Windows."""

    records = index.get("records")
    connection = getattr(records, "connection", None)
    if connection is not None:
        connection.close()
    arrays = []
    matrix = index.get("matrix")
    if matrix is not None:
        arrays.extend((matrix.data, matrix.indices, matrix.indptr))
    vectorizer = index.get("vectorizer")
    idf_values = getattr(getattr(vectorizer, "_tfidf", None), "idf_", None)
    if idf_values is not None:
        arrays.append(idf_values)
    idf_diag = getattr(getattr(vectorizer, "_tfidf", None), "_idf_diag", None)
    if idf_diag is not None:
        arrays.extend((idf_diag.data, idf_diag.indices, idf_diag.indptr))
    seen: set[int] = set()
    for array in arrays:
        current = array
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, np.memmap):
                mmap = getattr(current, "_mmap", None)
                if mmap is not None:
                    mmap.close()
                break
            current = getattr(current, "base", None)
