"""Cross-encoder reranking used only in configured hybrid mode."""

from __future__ import annotations

from collections.abc import Iterable

from retrieval.dense import DenseUnavailableError
from retrieval.scoring import RetrievedCandidate


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - optional deployment extra
            raise DenseUnavailableError(
                "hybrid reranking requires `uv sync --extra retrieval`"
            ) from exc
        self.model_name = model_name
        try:
            self._model = CrossEncoder(model_name, local_files_only=True)
        except OSError as exc:
            raise DenseUnavailableError(
                f"local reranker model is unavailable: {model_name}"
            ) from exc

    def rerank(
        self, query: str, candidates: Iterable[RetrievedCandidate]
    ) -> tuple[RetrievedCandidate, ...]:
        values = tuple(candidates)
        if not values:
            return ()
        scores = self._model.predict([(query, item.text) for item in values])
        ranked = [
            item.model_copy(update={"reranker_score": float(score)})
            for item, score in zip(values, scores, strict=True)
        ]
        return tuple(
            sorted(
                ranked,
                key=lambda item: (-(item.reranker_score or 0.0), item.chunk_id),
            )
        )


__all__ = ["CrossEncoderReranker"]
