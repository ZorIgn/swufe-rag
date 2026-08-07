"""Optional sentence-transformer / FAISS dense retrieval implementation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from retrieval.scoring import RetrievedCandidate


class DenseUnavailableError(RuntimeError):
    """Raised when hybrid mode lacks its required dense dependencies/artifacts."""


class DenseFaissIndex:
    """A normalized-vector FAISS index with stable chunk-id alignment."""

    def __init__(
        self,
        *,
        model_name: str,
        chunk_ids: tuple[str, ...],
        index: object,
        encoder: object,
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
    def _dependencies() -> tuple[object, object]:
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional deployment extra
            raise DenseUnavailableError(
                "hybrid retrieval requires `uv sync --extra retrieval`"
            ) from exc
        return faiss, SentenceTransformer

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
        encoder = sentence_transformer(model_name)
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
        index = faiss.read_index(str(directory / "faiss.index"))
        vectors = np.load(directory / "vectors.npy")
        if int(index.d) != dimension:
            raise DenseUnavailableError("FAISS dimension does not match retrieval manifest")
        if vectors.shape != (len(chunk_ids), dimension):
            raise DenseUnavailableError("dense vectors do not match retrieval manifest")
        encoder = sentence_transformer(model_name)
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
