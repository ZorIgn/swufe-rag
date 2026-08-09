"""Build and load versioned policy retrieval artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from retrieval.dense import DenseFaissIndex
from retrieval.lexical import BM25LexicalIndex
from retrieval.reranker import CrossEncoderReranker


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_directory(root: str | Path, dataset_version: str) -> Path:
    return Path(root) / dataset_version


def build_retrieval_index(
    documents: list[dict[str, object]],
    *,
    dataset_version: str,
    source_hash: str,
    output_root: str | Path = "artifacts/retrieval",
    mode: Literal["lexical", "hybrid"] = "hybrid",
    embedding_model: str = "BAAI/bge-base-zh-v1.5",
    reranker_model: str = "BAAI/bge-reranker-base",
) -> dict[str, object]:
    """Write immutable retrieval artifacts and a verifiable manifest."""

    directory = artifact_directory(output_root, dataset_version)
    directory.mkdir(parents=True, exist_ok=True)
    corpus_path = directory / "documents.jsonl"
    corpus_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in documents),
        encoding="utf-8",
    )
    # Constructing the lexical index validates tokenization/corpus integrity even
    # though BM25 is reconstructed in memory at runtime.
    BM25LexicalIndex(documents)
    manifest: dict[str, object] = {
        "dataset_version": dataset_version,
        "source_hash": source_hash,
        "retrieval_mode": mode,
        "embedding_model": "none",
        "embedding_dimension": 0,
        "reranker_model": "none",
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
            embedding_dimension=dense.dimension,
            reranker_model=reranker_model,
            index_sha256=_sha(directory / "faiss.index"),
            doc_ids_sha256=_sha(directory / "doc_ids.json"),
            vectors_sha256=_sha(directory / "vectors.npy"),
        )
    manifest_path = directory / "retrieval_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {**manifest, "directory": str(directory), "manifest": str(manifest_path)}


def load_manifest(root: str | Path, dataset_version: str) -> tuple[Path, dict[str, object]]:
    directory = artifact_directory(root, dataset_version)
    path = directory / "retrieval_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"retrieval manifest is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("retrieval manifest must be a JSON object")
    return directory, value


__all__ = ["artifact_directory", "build_retrieval_index", "load_manifest"]
