"""Build and load immutable, verified policy-retrieval artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, SupportsIndex, SupportsInt

import numpy as np

from retrieval.dense import DenseFaissIndex
from retrieval.lexical import BM25LexicalIndex
from retrieval.reranker import CrossEncoderReranker
from storage.release import ReleaseError, discard_staging, make_staging_directory, sha256_file


class RetrievalArtifactError(ReleaseError):
    """Raised when a retrieval artifact is incomplete or no longer immutable."""


def _sha(path: Path) -> str:
    return sha256_file(path)


def _int_or_zero(value: object) -> int:
    candidate = value or 0
    if isinstance(candidate, bool):
        raise RetrievalArtifactError("boolean is not a valid manifest integer")
    if isinstance(candidate, (str, bytes, bytearray, int, float)):
        return int(candidate)
    if isinstance(candidate, SupportsInt):
        return int(candidate)
    if isinstance(candidate, SupportsIndex):
        return int(candidate)
    raise RetrievalArtifactError(f"manifest integer is invalid: {candidate!r}")


def artifact_directory(root: str | Path, dataset_version: str) -> Path:
    """Return the immutable directory for one dataset version.

    Dataset versions are intentionally never overwritten.  The aggregate
    release layer adds a content-addressed directory above this, while this
    function preserves the versioned artifact layout consumed by the runtime.
    """

    return Path(root) / dataset_version


def _write_documents(path: Path, documents: list[dict[str, object]]) -> None:
    values = []
    identifiers: set[str] = set()
    for item in documents:
        identifier = str(item.get("chunk_id") or "").strip()
        if not identifier:
            raise RetrievalArtifactError("retrieval document has no chunk_id")
        if identifier in identifiers:
            raise RetrievalArtifactError(f"retrieval document chunk_id is duplicated: {identifier}")
        identifiers.add(identifier)
        values.append(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    if not values:
        raise RetrievalArtifactError("retrieval artifact cannot be built from an empty corpus")
    path.write_text("".join(values), encoding="utf-8")


def _load_documents(path: Path) -> tuple[dict[str, object], ...]:
    if not path.is_file():
        raise RetrievalArtifactError(f"retrieval documents are missing: {path}")
    values: list[dict[str, object]] = []
    identifiers: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RetrievalArtifactError(f"cannot read retrieval documents: {path}") from exc
    for index, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RetrievalArtifactError(f"retrieval document line {index} is not JSON") from exc
        if not isinstance(value, dict):
            raise RetrievalArtifactError(f"retrieval document line {index} is not an object")
        identifier = str(value.get("chunk_id") or "").strip()
        if not identifier:
            raise RetrievalArtifactError(f"retrieval document line {index} has no chunk_id")
        if identifier in identifiers:
            raise RetrievalArtifactError(f"retrieval document chunk_id is duplicated: {identifier}")
        identifiers.add(identifier)
        values.append(value)
    if not values:
        raise RetrievalArtifactError("retrieval documents are empty")
    return tuple(values)


def _manifest_file_hashes(mode: str) -> dict[str, str]:
    return {
        "documents.jsonl": "documents_sha256",
        **(
            {
                "doc_ids.json": "doc_ids_sha256",
                "vectors.npy": "vectors_sha256",
                "faiss.index": "index_sha256",
            }
            if mode == "hybrid"
            else {}
        ),
    }


def validate_retrieval_artifact(
    directory: str | Path, manifest: dict[str, object] | None = None
) -> dict[str, object]:
    """Validate every file that makes an artifact runnable before loading it.

    This does not instantiate transformers or FAISS, so it is safe for startup
    diagnostics and lightweight CI.  Hybrid runtime construction performs the
    additional model/index checks after this structural verification succeeds.
    """

    root = Path(directory)
    manifest_path = root / "retrieval_manifest.json"
    if manifest is None:
        if not manifest_path.is_file():
            raise RetrievalArtifactError(f"retrieval manifest is missing: {manifest_path}")
        try:
            parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RetrievalArtifactError(f"retrieval manifest is unreadable: {manifest_path}") from exc
        if not isinstance(parsed, dict):
            raise RetrievalArtifactError("retrieval manifest must be a JSON object")
        manifest = parsed
    mode = str(manifest.get("retrieval_mode") or "")
    if mode not in {"lexical", "hybrid"}:
        raise RetrievalArtifactError("retrieval manifest has an unsupported mode")
    dataset_version = str(manifest.get("dataset_version") or "")
    if not dataset_version or root.name != dataset_version:
        raise RetrievalArtifactError("retrieval manifest dataset version does not match artifact directory")
    documents = _load_documents(root / "documents.jsonl")
    if _int_or_zero(manifest.get("chunk_count")) != len(documents):
        raise RetrievalArtifactError("retrieval manifest chunk count does not match documents")
    for filename, key in _manifest_file_hashes(mode).items():
        path = root / filename
        expected = str(manifest.get(key) or "")
        if not path.is_file() or not expected:
            raise RetrievalArtifactError(f"retrieval artifact is missing required file/hash: {filename}")
        if _sha(path) != expected:
            raise RetrievalArtifactError(f"retrieval artifact hash mismatch: {filename}")
    if mode == "lexical":
        if str(manifest.get("embedding_model") or "") != "none":
            raise RetrievalArtifactError("lexical artifact must not claim an embedding model")
        if _int_or_zero(manifest.get("embedding_dimension")) != 0:
            raise RetrievalArtifactError("lexical artifact must have zero embedding dimension")
        return manifest

    identifiers_path = root / "doc_ids.json"
    try:
        identifiers = json.loads(identifiers_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalArtifactError("dense document ids are unreadable") from exc
    if not isinstance(identifiers, list) or any(
        not isinstance(value, str) or not value for value in identifiers
    ):
        raise RetrievalArtifactError("dense document ids must be a non-empty string list")
    document_ids = [str(item["chunk_id"]) for item in documents]
    if identifiers != document_ids:
        raise RetrievalArtifactError("dense document ids do not preserve corpus order")
    dimension = _int_or_zero(manifest.get("embedding_dimension"))
    if dimension <= 0 or not str(manifest.get("embedding_model") or ""):
        raise RetrievalArtifactError("hybrid artifact has no embedding model/dimension")
    if not str(manifest.get("reranker_model") or "") or str(manifest.get("reranker_model")) == "none":
        raise RetrievalArtifactError("hybrid artifact has no reranker model")
    for field in ("embedding_model_sha256", "reranker_model_sha256"):
        digest = manifest.get(field)
        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RetrievalArtifactError(f"hybrid artifact has an invalid {field}")
    try:
        vectors = np.load(root / "vectors.npy", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise RetrievalArtifactError("dense vectors are unreadable") from exc
    if vectors.shape != (len(document_ids), dimension) or not np.isfinite(vectors).all():
        raise RetrievalArtifactError("dense vectors do not match the hybrid manifest")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
        raise RetrievalArtifactError("dense vectors are not normalized")
    return manifest


def _return_artifact(directory: Path, manifest: dict[str, object]) -> dict[str, object]:
    return {**manifest, "directory": str(directory), "manifest": str(directory / "retrieval_manifest.json")}


def build_retrieval_index(
    documents: list[dict[str, object]],
    *,
    dataset_version: str,
    source_hash: str,
    output_root: str | Path = "artifacts/retrieval",
    mode: Literal["lexical", "hybrid"] = "hybrid",
    embedding_model: str = "BAAI/bge-base-zh-v1.5",
    reranker_model: str = "BAAI/bge-reranker-base",
    embedding_model_sha256: str | None = None,
    reranker_model_sha256: str | None = None,
) -> dict[str, object]:
    """Build an immutable artifact via same-filesystem staging and validation."""

    if not dataset_version or Path(dataset_version).name != dataset_version:
        raise RetrievalArtifactError("dataset_version must be a single non-empty path segment")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    target = artifact_directory(root, dataset_version)
    staging = make_staging_directory(root)
    directory = staging / dataset_version
    directory.mkdir()
    try:
        corpus_path = directory / "documents.jsonl"
        _write_documents(corpus_path, documents)
        # Constructing the lexical index validates tokenization/corpus integrity
        # even though BM25 is reconstructed in memory at runtime.
        BM25LexicalIndex(documents)
        manifest: dict[str, object] = {
            "dataset_version": dataset_version,
            "source_hash": source_hash,
            "retrieval_mode": mode,
            "embedding_model": "none",
            "embedding_model_sha256": None,
            "embedding_dimension": 0,
            "reranker_model": "none",
            "reranker_model_sha256": None,
            "chunk_count": len(documents),
            "documents_sha256": _sha(corpus_path),
            "index_sha256": _sha(corpus_path),
        }
        if mode == "hybrid":
            dense, _ = DenseFaissIndex.build(documents, model_name=embedding_model)
            dense.save(directory)
            # Construct the model during build so a manifest never advertises a
            # reranker that the environment cannot instantiate.
            CrossEncoderReranker(reranker_model)
            manifest.update(
                embedding_model=embedding_model,
                embedding_model_sha256=embedding_model_sha256,
                embedding_dimension=dense.dimension,
                reranker_model=reranker_model,
                reranker_model_sha256=reranker_model_sha256,
                index_sha256=_sha(directory / "faiss.index"),
                doc_ids_sha256=_sha(directory / "doc_ids.json"),
                vectors_sha256=_sha(directory / "vectors.npy"),
            )
        manifest_path = directory / "retrieval_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        validate_retrieval_artifact(directory, manifest)
        if target.exists():
            existing = validate_retrieval_artifact(target)
            if existing != manifest:
                raise RetrievalArtifactError(
                    f"refusing to overwrite immutable retrieval artifact: {target}"
                )
            return _return_artifact(target, existing)
        # Rename the contained directory, then delete the now-empty staging
        # root in ``finally``. Consumers never observe a partial target.
        os.replace(directory, target)
        validate_retrieval_artifact(target)
        return _return_artifact(target, manifest)
    finally:
        if staging.exists():
            discard_staging(staging)


def load_manifest(root: str | Path, dataset_version: str) -> tuple[Path, dict[str, object]]:
    directory = artifact_directory(root, dataset_version)
    manifest = validate_retrieval_artifact(directory)
    return directory, manifest


__all__ = [
    "RetrievalArtifactError",
    "artifact_directory",
    "build_retrieval_index",
    "load_manifest",
    "validate_retrieval_artifact",
]
