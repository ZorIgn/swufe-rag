"""Optional sentence-transformer / FAISS dense retrieval implementation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, cast

import numpy as np

from retrieval.scoring import RetrievedCandidate


class DenseUnavailableError(RuntimeError):
    """Raised when hybrid mode lacks its required dense dependencies/artifacts."""


class _FaissIndex(Protocol):
    d: int
    ntotal: int

    def add(self, vectors: np.ndarray) -> None: ...

    def search(self, vectors: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]: ...


class _FaissModule(Protocol):
    def IndexFlatIP(self, dimension: int) -> _FaissIndex: ...

    def read_index(self, path: str) -> _FaissIndex: ...

    def write_index(self, index: _FaissIndex, path: str) -> None: ...


class _SentenceEncoder(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int | None = None,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
    ) -> object: ...


class _SentenceTransformerFactory(Protocol):
    def __call__(
        self, model_name: str, *, local_files_only: bool = False
    ) -> _SentenceEncoder: ...


class DenseFaissIndex:
    """A normalized-vector FAISS index with stable chunk-id alignment."""

    def __init__(
        self,
        *,
        model_name: str,
        chunk_ids: tuple[str, ...],
        index: _FaissIndex,
        encoder: _SentenceEncoder,
        dimension: int,
        vectors: np.ndarray,
    ) -> None:
        self.model_name = model_name
        self.chunk_ids = chunk_ids
        self._index = index
        self._encoder = encoder
        self.dimension = dimension
        self._vectors = vectors
        self._positions = {identifier: index for index, identifier in enumerate(chunk_ids)}

    @staticmethod
    def _dependencies() -> tuple[_FaissModule, _SentenceTransformerFactory]:
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional deployment extra
            raise DenseUnavailableError(
                "hybrid retrieval requires `uv sync --extra retrieval`"
            ) from exc
        # These optional packages do not provide complete stubs.  Constrain
        # them to the small, runtime-verified surface consumed by this index.
        return cast(_FaissModule, faiss), cast(_SentenceTransformerFactory, SentenceTransformer)

    @classmethod
    def build(
        cls,
        documents: Iterable[dict[str, object]],
        *,
        model_name: str,
        batch_size: int = 32,
    ) -> tuple[DenseFaissIndex, np.ndarray]:
        faiss, sentence_transformer = cls._dependencies()
        values = tuple(dict(item) for item in documents)
        try:
            encoder = sentence_transformer(model_name, local_files_only=True)
        except OSError as exc:
            raise DenseUnavailableError(
                f"local embedding model is unavailable: {model_name}"
            ) from exc
        embeddings = np.asarray(
            encoder.encode(
                [str(item.get("text", "")) for item in values],
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype="float32",
        )
        if embeddings.ndim != 2 or not len(values):
            raise DenseUnavailableError("dense embedding build returned an empty matrix")
        index = faiss.IndexFlatIP(int(embeddings.shape[1]))
        index.add(embeddings)
        return (
            cls(
                model_name=model_name,
                chunk_ids=tuple(str(item["chunk_id"]) for item in values),
                index=index,
                encoder=encoder,
                dimension=int(embeddings.shape[1]),
                vectors=embeddings,
            ),
            embeddings,
        )

    @classmethod
    def load(cls, directory: Path, *, model_name: str, dimension: int) -> DenseFaissIndex:
        faiss, sentence_transformer = cls._dependencies()
        import json

        chunk_ids = tuple(json.loads((directory / "doc_ids.json").read_text(encoding="utf-8")))
        if not chunk_ids or len(set(chunk_ids)) != len(chunk_ids):
            raise DenseUnavailableError("dense document ids are empty or duplicated")
        index = faiss.read_index(str(directory / "faiss.index"))
        vectors = np.load(directory / "vectors.npy", allow_pickle=False)
        if int(index.d) != dimension:
            raise DenseUnavailableError("FAISS dimension does not match retrieval manifest")
        if int(index.ntotal) != len(chunk_ids):
            raise DenseUnavailableError("FAISS row count does not match dense document ids")
        if vectors.shape != (len(chunk_ids), dimension):
            raise DenseUnavailableError("dense vectors do not match retrieval manifest")
        if not np.isfinite(vectors).all():
            raise DenseUnavailableError("dense vectors contain non-finite values")
        try:
            encoder = sentence_transformer(model_name, local_files_only=True)
        except OSError as exc:
            raise DenseUnavailableError(
                f"local embedding model is unavailable: {model_name}"
            ) from exc
        return cls(
            model_name=model_name,
            chunk_ids=chunk_ids,
            index=index,
            encoder=encoder,
            dimension=dimension,
            vectors=np.asarray(vectors, dtype="float32"),
        )

    def save(self, directory: Path) -> None:
        faiss, _ = self._dependencies()
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / "faiss.index"))
        import json

        (directory / "doc_ids.json").write_text(
            json.dumps(self.chunk_ids, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        np.save(directory / "vectors.npy", self._vectors)

    def vector_for(self, chunk_id: str) -> np.ndarray | None:
        position = self._positions.get(chunk_id)
        return None if position is None else self._vectors[position]

    def rank(
        self,
        query: str,
        documents: dict[str, dict[str, object]],
        candidate_ids: set[str],
        *,
        limit: int,
    ) -> tuple[RetrievedCandidate, ...]:
        vector = np.asarray(
            self._encoder.encode([query], normalize_embeddings=True, show_progress_bar=False),
            dtype="float32",
        )
        scores, positions = self._index.search(
            vector, min(len(self.chunk_ids), max(limit * 8, limit))
        )
        ranked: list[RetrievedCandidate] = []
        for score, position in zip(scores[0], positions[0], strict=True):
            if position < 0:
                continue
            identifier = self.chunk_ids[int(position)]
            if identifier not in candidate_ids:
                continue
            document = documents[identifier]
            ranked.append(
                RetrievedCandidate(
                    chunk_id=identifier,
                    text=str(document.get("text", "")),
                    metadata=document,
                    dense_score=float(score),
                    dense_rank=len(ranked) + 1,
                )
            )
            if len(ranked) >= limit:
                break
        return tuple(ranked)


__all__ = ["DenseFaissIndex", "DenseUnavailableError"]
