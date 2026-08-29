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
    def __call__(self, model_name: str, *, local_files_only: bool = False) -> _SentenceEncoder: ...


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
        if dimension <= 0:
            raise DenseUnavailableError("dense embedding dimension must be positive")
        if not chunk_ids or len(set(chunk_ids)) != len(chunk_ids):
            raise DenseUnavailableError("dense document ids are empty or duplicated")
        self.model_name = model_name
        self.chunk_ids = chunk_ids
        self._index = index
        self._encoder = encoder
        self.dimension = dimension
        self._vectors = self._validated_vectors(
            vectors, chunk_count=len(chunk_ids), dimension=dimension
        )
        self._positions = {identifier: index for index, identifier in enumerate(chunk_ids)}

    @staticmethod
    def _validated_vectors(vectors: np.ndarray, *, chunk_count: int, dimension: int) -> np.ndarray:
        """Validate the normalized row vectors used as the scoped rank oracle."""

        try:
            value = np.asarray(vectors, dtype="float32")
        except (TypeError, ValueError, OverflowError) as exc:
            raise DenseUnavailableError("dense vectors are not a numeric matrix") from exc
        if value.shape != (chunk_count, dimension):
            raise DenseUnavailableError("dense vectors do not match dense document ids")
        if not np.isfinite(value).all():
            raise DenseUnavailableError("dense vectors contain non-finite values")
        norms = np.linalg.norm(value, axis=1)
        if not np.isfinite(norms).all() or not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
            raise DenseUnavailableError("dense vectors must be finite, non-zero, normalized rows")
        return value

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

    def _query_vector(self, query: str) -> np.ndarray:
        """Encode exactly one finite, normalized query vector for inner products."""

        try:
            encoded = self._encoder.encode(
                [query], normalize_embeddings=True, show_progress_bar=False
            )
            vector = np.asarray(encoded, dtype="float32")
        except (TypeError, ValueError, OverflowError) as exc:
            raise DenseUnavailableError("dense query embedding is not numeric") from exc
        if vector.shape != (1, self.dimension):
            raise DenseUnavailableError(
                "dense query embedding shape does not match the configured dimension"
            )
        if not np.isfinite(vector).all():
            raise DenseUnavailableError("dense query embedding contains non-finite values")
        query_vector = vector[0]
        norm = float(np.linalg.norm(query_vector))
        if not np.isfinite(norm) or not np.isclose(norm, 1.0, rtol=1e-4, atol=1e-5):
            raise DenseUnavailableError("dense query embedding must be finite and normalized")
        return query_vector

    def rank(
        self,
        query: str,
        documents: dict[str, dict[str, object]],
        candidate_ids: set[str],
        *,
        limit: int,
    ) -> tuple[RetrievedCandidate, ...]:
        if limit <= 0 or not candidate_ids:
            return ()

        # Scope is a correctness boundary, not an over-fetch heuristic.  Map
        # only in-scope IDs to their immutable vectors.npy rows before scoring.
        # Missing vector/document IDs are ignored deterministically so a stale
        # caller cannot turn a retrieval request into a KeyError.
        scoped = sorted(
            (identifier, self._positions[identifier])
            for identifier in candidate_ids
            if identifier in self._positions and identifier in documents
        )
        if not scoped:
            return ()

        vector = self._query_vector(query)
        identifiers = tuple(identifier for identifier, _position in scoped)
        positions = np.asarray([position for _identifier, position in scoped], dtype=np.intp)
        scores = self._vectors[positions] @ vector
        if not np.isfinite(scores).all():
            raise DenseUnavailableError("scoped dense scores contain non-finite values")

        # Python sorting makes equal cosine scores deterministic across FAISS,
        # NumPy, and platform versions.  We deliberately do not call
        # ``self._index.search`` here: global top-N followed by filtering can
        # erase the best document in a narrow scope.
        ranked = sorted(
            zip(identifiers, scores, strict=True), key=lambda item: (-float(item[1]), item[0])
        )[:limit]
        return tuple(
            RetrievedCandidate(
                chunk_id=identifier,
                text=str(documents[identifier].get("text", "")),
                metadata=documents[identifier],
                dense_score=float(score),
                dense_rank=rank,
            )
            for rank, (identifier, score) in enumerate(ranked, start=1)
        )


__all__ = ["DenseFaissIndex", "DenseUnavailableError"]
