"""Rank-based score contracts used by the policy retrieval pipeline."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RetrievedCandidate(BaseModel):
    """A candidate with independently meaningful ranking signals.

    Raw BM25 and dense values are retained for diagnostics only. Fusion is
    performed from rank lists, never by adding incompatible raw score scales.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    text: str
    metadata: dict[str, object]
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0
    reranker_score: float | None = None
    exact_entity_score: float = 0.0
    scope_score: float = 0.0
    authority_score: float = 0.0
    final_score: float = 0.0
    lexical_rank: int | None = Field(default=None, ge=1)
    dense_rank: int | None = Field(default=None, ge=1)
    fused_rank: int | None = Field(default=None, ge=1)


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    values: dict[str, float] = {}
    for ranking in rankings:
        for position, identifier in enumerate(ranking, start=1):
            values[identifier] = values.get(identifier, 0.0) + 1 / (k + position)
    return values


def normalized_rank_score(position: int | None, total: int) -> float:
    if position is None or total <= 0:
        return 0.0
    return max(0.0, 1.0 - (position - 1) / max(1, total))


def presentation_score(
    *,
    fused_rank: int | None,
    total: int,
    reranker_score: float | None,
    authority_score: float,
    scope_score: float,
) -> float:
    """A display score after rank fusion and optional reranking.

    It deliberately combines normalized post-ranking values rather than raw
    BM25 / embedding similarities.
    """

    rank_score = normalized_rank_score(fused_rank, total)
    rerank = reranker_score if reranker_score is not None else rank_score
    return 0.60 * rerank + 0.25 * rank_score + 0.10 * authority_score + 0.05 * scope_score


__all__ = ["RetrievedCandidate", "presentation_score", "reciprocal_rank_fusion"]
