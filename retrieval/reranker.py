"""Optional second-stage rerankers inspired by mature hybrid RAG systems."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from retrieval.query import QueryAnalysis, chunk_search_text, exact_signal_score, lexical_tokens


DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"


class Reranker(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> np.ndarray: ...


def normalize_scores(values: Sequence[float] | np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float32).reshape(-1)
    if not len(scores):
        return scores
    if np.all((scores >= 0) & (scores <= 1)):
        return scores
    clipped = np.clip(scores, -30, 30)
    return 1.0 / (1.0 + np.exp(-clipped))


class BGEReranker:
    """Lazy CrossEncoder adapter; outputs normalized relevance scores."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        *,
        device: str | None = None,
        max_length: int = 1024,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for BGE reranking"
                ) from exc
            kwargs = {"device": self.device} if self.device else {}
            self._model = CrossEncoder(
                self.model_name, max_length=self.max_length, **kwargs
            )
        return self._model

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        if not documents:
            return np.empty(0, dtype=np.float32)
        pairs = [[query, document] for document in documents]
        raw = self._load().predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return normalize_scores(raw)


class HeuristicReranker:
    """Deterministic evidence-aware reranker for tests and CPU fallback."""

    def __init__(self, analysis: QueryAnalysis, chunks: Sequence[dict]) -> None:
        self.analysis = analysis
        self.chunks = list(chunks)

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        results: list[float] = []
        for index, document in enumerate(documents):
            tokens = lexical_tokens(document)
            coverage = (
                len(self.analysis.tokens & tokens) / len(self.analysis.tokens)
                if self.analysis.tokens
                else 0.0
            )
            exact = exact_signal_score(self.analysis, self.chunks[index])
            results.append(min(1.0, coverage * 0.65 + exact * 0.35))
        return np.asarray(results, dtype=np.float32)


def rerank_documents(reranker: Reranker, query: str, chunks: Sequence[dict]) -> np.ndarray:
    documents = [chunk_search_text(chunk) for chunk in chunks]
    scores = normalize_scores(reranker.score(query, documents))
    if scores.shape != (len(chunks),):
        raise ValueError(
            f"reranker returned {scores.shape}; expected ({len(chunks)},)"
        )
    if not np.isfinite(scores).all():
        raise ValueError("reranker returned a non-finite score")
    return scores
