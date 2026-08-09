"""Migrate the checked local legacy FAISS bundle into the versioned layout.

This is intentionally a one-way, integrity-checked migration.  It never treats
the presence of the old 249 MB files as proof that they belong to the current
database: source hash, ordered chunk ids, row count, and dimension must all
match before an artifact is published atomically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from academic.database import AcademicRepository

try:
    import faiss
except ImportError as exc:  # pragma: no cover - depends on retrieval extra
    raise RuntimeError(
        "legacy migration requires `uv sync --extra retrieval` (FAISS is missing)"
    ) from exc


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=Path("data/academic.sqlite3"))
    parser.add_argument("--legacy-root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/retrieval")
    )
    parser.add_argument(
        "--dataset-manifest-dir", type=Path, default=Path("artifacts/manifests")
    )
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    args = parser.parse_args()

    legacy_manifest = json.loads(
        (args.legacy_root / "manifest.json").read_text(encoding="utf-8")
    )
    repository = AcademicRepository(args.database)
    try:
        metadata = repository.metadata()
        documents = repository.retrieval_documents()
    finally:
        repository.close()
    dataset_version = metadata.get("dataset_version", "")
    if not dataset_version:
        raise RuntimeError("database dataset_version is missing")
    chunks_hash = metadata.get("chunks_sha256", "")
    if legacy_manifest.get("chunks_sha256") != chunks_hash:
        raise RuntimeError("legacy bundle source hash does not match the database")
    source_hash = metadata.get("evidence_state_sha256", chunks_hash)

    legacy_ids = tuple(
        json.loads((args.legacy_root / "chunk_ids.json").read_text(encoding="utf-8"))
    )
    document_ids = tuple(str(item["chunk_id"]) for item in documents)
    dimension = int(legacy_manifest.get("dimension") or 0)
    if dimension <= 0 or int(legacy_manifest.get("chunk_count") or 0) != len(document_ids):
        raise RuntimeError("legacy dense shape metadata is invalid")
    if len(set(legacy_ids)) != len(legacy_ids) or len(set(document_ids)) != len(document_ids):
        raise RuntimeError("dense row identifiers must be unique")
    if set(legacy_ids) != set(document_ids):
        raise RuntimeError("legacy dense rows do not contain the current retrieval documents")

    legacy_vectors = np.load(
        args.legacy_root / "vectors.npy", mmap_mode="r", allow_pickle=False
    )
    expected_shape = (len(legacy_ids), dimension)
    if legacy_vectors.shape != expected_shape or legacy_vectors.dtype != np.float32:
        raise RuntimeError("legacy vector matrix shape or dtype is invalid")
    legacy_index = faiss.read_index(str(args.legacy_root / "index.faiss"))
    if legacy_index.ntotal != len(legacy_ids) or legacy_index.d != dimension:
        raise RuntimeError("legacy FAISS shape does not match its manifest")
    for row in {0, len(legacy_ids) // 2, len(legacy_ids) - 1}:
        if not np.allclose(
            legacy_index.reconstruct(row), legacy_vectors[row], rtol=1e-5, atol=1e-6
        ):
            raise RuntimeError("legacy FAISS rows do not match vectors.npy")

    dataset_manifest_path = args.dataset_manifest_dir / f"{dataset_version}.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if str(dataset_manifest.get("dataset_version") or "") != dataset_version:
        raise RuntimeError("dataset manifest version does not match the database")
    source_hashes = dataset_manifest.get("source_hashes")
    if not isinstance(source_hashes, dict) or source_hashes.get("chunks") != chunks_hash:
        raise RuntimeError("dataset manifest chunks hash does not match the database")

    output_root = args.output_root.resolve()
    target = (output_root / dataset_version).resolve()
    if output_root not in target.parents:
        raise RuntimeError("retrieval target must stay within the configured output root")
    temporary = target.with_name(target.name + ".tmp")
    if output_root not in temporary.parents:
        raise RuntimeError("temporary retrieval target escaped the output root")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    row_order_rewritten = legacy_ids != document_ids
    if not row_order_rewritten:
        shutil.copy2(args.legacy_root / "index.faiss", temporary / "faiss.index")
        shutil.copy2(args.legacy_root / "vectors.npy", temporary / "vectors.npy")
    else:
        legacy_position = {identifier: index for index, identifier in enumerate(legacy_ids)}
        order = np.fromiter(
            (legacy_position[identifier] for identifier in document_ids),
            dtype=np.int64,
            count=len(document_ids),
        )
        migrated_vectors = np.lib.format.open_memmap(
            temporary / "vectors.npy",
            mode="w+",
            dtype=np.float32,
            shape=expected_shape,
        )
        migrated_index = faiss.IndexFlatIP(dimension)
        batch_size = 4096
        for start in range(0, len(order), batch_size):
            stop = min(len(order), start + batch_size)
            batch = np.asarray(legacy_vectors[order[start:stop]], dtype=np.float32)
            migrated_vectors[start:stop] = batch
            migrated_index.add(batch)
        migrated_vectors.flush()
        del migrated_vectors
        faiss.write_index(migrated_index, str(temporary / "faiss.index"))
    (temporary / "doc_ids.json").write_text(
        json.dumps(document_ids, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    documents_path = temporary / "documents.jsonl"
    documents_path.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in documents
        ),
        encoding="utf-8",
    )
    manifest = {
        "dataset_version": dataset_version,
        "source_hash": source_hash,
        "retrieval_mode": "hybrid",
        "embedding_model": str(legacy_manifest.get("model_name") or ""),
        "embedding_dimension": dimension,
        "reranker_model": args.reranker_model,
        "chunk_count": len(document_ids),
        "documents_sha256": _sha(documents_path),
        "index_sha256": _sha(temporary / "faiss.index"),
        "doc_ids_sha256": _sha(temporary / "doc_ids.json"),
        "vectors_sha256": _sha(temporary / "vectors.npy"),
        "migrated_from": str(args.legacy_root / "manifest.json"),
        "row_order_rewritten": row_order_rewritten,
    }
    (temporary / "retrieval_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if target.exists():
        shutil.rmtree(target)
    temporary.replace(target)

    dataset_manifest.update(
        retrieval_mode="hybrid",
        embedding_model=manifest["embedding_model"],
        embedding_dimension=dimension,
        index_sha256=manifest["index_sha256"],
        retrieval_manifest=str(target / "retrieval_manifest.json"),
        retrieval_migrated_at=datetime.now(timezone.utc).isoformat(),
    )
    manifest_temporary = dataset_manifest_path.with_suffix(".json.tmp")
    manifest_temporary.write_text(
        json.dumps(dataset_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_temporary.replace(dataset_manifest_path)
    print(json.dumps({**manifest, "directory": str(target)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
