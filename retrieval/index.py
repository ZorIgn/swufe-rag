"""Strict chunk loading and persistent FAISS index management."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from contracts import (
    CONTRACT_VERSION,
    ContractError,
    KnowledgeBaseNotReadyError,
    KnowledgeChunk,
    validate_chunk,
)
from retrieval.embed import BGEEncoder, Encoder, normalize_rows


MANIFEST_FILE = "manifest.json"
VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.json"
CHUNK_IDS_FILE = "chunk_ids.json"
FAISS_FILE = "index.faiss"


def _import_faiss():
    try:
        import faiss
    except ImportError:
        return None
    return faiss


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_chunks(path: str | Path) -> list[KnowledgeChunk]:
    source = Path(path)
    if not source.is_file():
        raise KnowledgeBaseNotReadyError(f"knowledge chunks do not exist: {source}")

    chunks: list[KnowledgeChunk] = []
    seen_ids: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(
                    f"invalid JSON: {exc.msg}", line_number=line_number
                ) from exc
            chunk = validate_chunk(raw, line_number=line_number)
            if chunk["chunk_id"] in seen_ids:
                raise ContractError(
                    "duplicate chunk_id",
                    line_number=line_number,
                    chunk_id=chunk["chunk_id"],
                    field="chunk_id",
                )
            seen_ids.add(chunk["chunk_id"])
            chunks.append(chunk)

    if not chunks:
        raise KnowledgeBaseNotReadyError(f"knowledge chunks are empty: {source}")
    return chunks


@dataclass(frozen=True)
class IndexBundle:
    chunks: list[KnowledgeChunk]
    embeddings: np.ndarray
    model_name: str
    source_hash: str
    backend: str
    faiss_index: Any | None = None


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def build_index(
    chunks_path: str | Path,
    artifacts_dir: str | Path,
    encoder: Encoder,
    *,
    allow_test_backend: bool = False,
) -> dict[str, Any]:
    """Build artifacts, writing the manifest last as the commit marker.

    ``allow_test_backend`` deliberately selects the NumPy test backend.  Tests
    must not silently become production FAISS builds merely because faiss-cpu
    happens to be installed in the current environment.
    """

    source = Path(chunks_path)
    artifacts = Path(artifacts_dir)
    chunks = load_chunks(source)
    embeddings = normalize_rows(encoder.encode_documents([item["text"] for item in chunks]))
    if embeddings.shape != (len(chunks), encoder.dimension):
        raise ValueError(
            f"encoder returned {embeddings.shape}; expected ({len(chunks)}, {encoder.dimension})"
        )

    faiss = None if allow_test_backend else _import_faiss()
    if faiss is None and not allow_test_backend:
        raise KnowledgeBaseNotReadyError(
            "faiss-cpu is required to build production artifacts; install requirements.txt"
        )

    artifacts.mkdir(parents=True, exist_ok=True)
    manifest_path = artifacts / MANIFEST_FILE
    if manifest_path.exists():
        manifest_path.unlink()

    _atomic_npy(artifacts / VECTORS_FILE, embeddings)
    _atomic_json(artifacts / CHUNKS_FILE, chunks)
    _atomic_json(artifacts / CHUNK_IDS_FILE, [item["chunk_id"] for item in chunks])

    backend = "faiss-index-flat-ip" if faiss is not None else "numpy-test-only"
    if faiss is not None:
        index = faiss.IndexFlatIP(encoder.dimension)
        index.add(embeddings)
        temporary_index = artifacts / (FAISS_FILE + ".tmp")
        faiss.write_index(index, str(temporary_index))
        os.replace(temporary_index, artifacts / FAISS_FILE)
    else:
        stale_faiss = artifacts / FAISS_FILE
        if stale_faiss.exists():
            stale_faiss.unlink()

    manifest = {
        "contract_version": CONTRACT_VERSION,
        "model_name": encoder.model_name,
        "dimension": encoder.dimension,
        "chunk_count": len(chunks),
        "chunks_sha256": file_sha256(source),
        "backend": backend,
        "files": {
            "vectors": VECTORS_FILE,
            "chunks": CHUNKS_FILE,
            "chunk_ids": CHUNK_IDS_FILE,
            "faiss": FAISS_FILE if faiss is not None else None,
        },
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def load_index(
    chunks_path: str | Path,
    artifacts_dir: str | Path,
    encoder: Encoder,
    *,
    allow_test_backend: bool = False,
) -> IndexBundle:
    source = Path(chunks_path)
    artifacts = Path(artifacts_dir)
    manifest_path = artifacts / MANIFEST_FILE
    if not manifest_path.is_file():
        raise KnowledgeBaseNotReadyError(
            f"retrieval manifest does not exist: {manifest_path}; rebuild the index"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KnowledgeBaseNotReadyError("retrieval manifest is unreadable") from exc

    actual_hash = file_sha256(source) if source.is_file() else None
    if actual_hash != manifest.get("chunks_sha256"):
        raise KnowledgeBaseNotReadyError(
            "chunks.jsonl does not match the retrieval artifacts; rebuild the index"
        )
    if manifest.get("contract_version") != CONTRACT_VERSION:
        raise KnowledgeBaseNotReadyError("artifact contract version is incompatible")
    if manifest.get("model_name") != encoder.model_name:
        raise KnowledgeBaseNotReadyError("artifact embedding model does not match configuration")
    if manifest.get("dimension") != encoder.dimension:
        raise KnowledgeBaseNotReadyError("artifact embedding dimension does not match encoder")

    chunks = load_chunks(source)
    vectors_path = artifacts / VECTORS_FILE
    ids_path = artifacts / CHUNK_IDS_FILE
    try:
        embeddings = np.load(vectors_path, allow_pickle=False)
        saved_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise KnowledgeBaseNotReadyError("retrieval artifacts are incomplete") from exc

    expected_shape = (len(chunks), encoder.dimension)
    if embeddings.shape != expected_shape:
        raise KnowledgeBaseNotReadyError(
            f"vector shape mismatch: found {embeddings.shape}, expected {expected_shape}"
        )
    if saved_ids != [item["chunk_id"] for item in chunks]:
        raise KnowledgeBaseNotReadyError("chunk_id mapping does not match chunks.jsonl")
    if manifest.get("chunk_count") != len(chunks):
        raise KnowledgeBaseNotReadyError("manifest chunk count does not match chunks.jsonl")

    backend = str(manifest.get("backend"))
    faiss_index = None
    if backend == "faiss-index-flat-ip":
        faiss = _import_faiss()
        if faiss is None:
            raise KnowledgeBaseNotReadyError("faiss-cpu is required to load this index")
        try:
            faiss_index = faiss.read_index(str(artifacts / FAISS_FILE))
        except Exception as exc:
            raise KnowledgeBaseNotReadyError("FAISS index is unreadable") from exc
        if faiss_index.ntotal != len(chunks) or faiss_index.d != encoder.dimension:
            raise KnowledgeBaseNotReadyError("FAISS index metadata is inconsistent")
    elif backend == "numpy-test-only":
        if not allow_test_backend:
            raise KnowledgeBaseNotReadyError(
                "test-only NumPy artifacts cannot be loaded in production"
            )
    else:
        raise KnowledgeBaseNotReadyError(f"unsupported retrieval backend: {backend}")

    return IndexBundle(
        chunks=chunks,
        embeddings=normalize_rows(embeddings),
        model_name=encoder.model_name,
        source_hash=str(actual_hash),
        backend=backend,
        faiss_index=faiss_index,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build swufe-rag retrieval artifacts")
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--model", default="BAAI/bge-large-zh-v1.5")
    args = parser.parse_args()
    manifest = build_index(args.chunks, args.artifacts, BGEEncoder(args.model))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

