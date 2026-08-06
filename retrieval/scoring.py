"""Transparent retrieval score components and generic rank fusion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedCandidate:
    chunk_id: str
    text: str
    metadata: dict[str, object]
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    reranker_score: float = 0.0
    exact_entity_score: float = 0.0
    scope_score: float = 0.0
    final_score: float = 0.0


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    values: dict[str, float] = {}
    for ranking in rankings:
        for position, identifier in enumerate(ranking, start=1):
            values[identifier] = values.get(identifier, 0.0) + 1 / (k + position)
    return values


def final_score(candidate: RetrievedCandidate) -> float:
    return (
        0.20 * candidate.dense_score + 0.20 * candidate.bm25_score + 0.25 * candidate.rrf_score
        + 0.20 * candidate.reranker_score + 0.10 * candidate.exact_entity_score + 0.05 * candidate.scope_score
    )
