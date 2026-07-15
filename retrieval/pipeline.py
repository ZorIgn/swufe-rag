"""Advanced retrieval orchestration preserving the frozen output contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from contracts import RetrievedChunk
from retrieval.embed import BGEEncoder, Encoder
from retrieval.query import analyze_query, chunk_search_text, exact_signal_score, lexical_tokens
from retrieval.reranker import BGEReranker, HeuristicReranker, Reranker, rerank_documents
from retrieval.retriever import HybridRetriever


@dataclass(frozen=True)
class RetrievalTuning:
    candidate_k: int = 20
    dense_weight: float = 0.35
    lexical_weight: float = 0.25
    rerank_weight: float = 0.35
    rank_prior_weight: float = 0.05
    mmr_lambda: float = 0.88

    def __post_init__(self) -> None:
        weights = (
            self.dense_weight,
            self.lexical_weight,
            self.rerank_weight,
            self.rank_prior_weight,
        )
        if self.candidate_k < 5:
            raise ValueError("candidate_k must be at least 5")
        if any(weight < 0 for weight in weights) or not np.isclose(sum(weights), 1.0):
            raise ValueError("retrieval weights must be non-negative and sum to 1")
        if not 0 <= self.mmr_lambda <= 1:
            raise ValueError("mmr_lambda must be between 0 and 1")


class AdvancedRetriever:
    """Query expansion, exact-signal weighting, reranking, dedupe and MMR."""

    def __init__(
        self,
        core: HybridRetriever,
        *,
        reranker: Reranker | None = None,
        tuning: RetrievalTuning | None = None,
    ) -> None:
        self.core = core
        self.reranker = reranker
        self.tuning = tuning or RetrievalTuning(candidate_k=core.candidate_k)
        self._embedding_by_id = {
            chunk["chunk_id"]: core.bundle.embeddings[index]
            for index, chunk in enumerate(core.bundle.chunks)
        }

    @classmethod
    def from_artifacts(
        cls,
        chunks_path: str | Path = "data/chunks.jsonl",
        artifacts_dir: str | Path = "artifacts",
        encoder: Encoder | None = None,
        *,
        use_reranker: bool = True,
        rerank_model: str = "BAAI/bge-reranker-base",
        tuning: RetrievalTuning | None = None,
    ) -> "AdvancedRetriever":
        actual_encoder = encoder or BGEEncoder()
        actual_tuning = tuning or RetrievalTuning()
        core = HybridRetriever.from_artifacts(
            chunks_path,
            artifacts_dir,
            actual_encoder,
            candidate_k=actual_tuning.candidate_k,
        )
        reranker = BGEReranker(rerank_model) if use_reranker else None
        return cls(core, reranker=reranker, tuning=actual_tuning)

    @staticmethod
    def _deduplicate(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        seen_ids: set[str] = set()
        seen_text: set[str] = set()
        unique: list[RetrievedChunk] = []
        for chunk in chunks:
            fingerprint = "".join(
                character.lower()
                for character in str(chunk["text"])
                if character.isalnum()
            )
            if chunk["chunk_id"] in seen_ids or fingerprint in seen_text:
                continue
            seen_ids.add(chunk["chunk_id"])
            seen_text.add(fingerprint)
            unique.append(chunk)
        return unique

    def _lexical_score(self, analysis, chunk: RetrievedChunk) -> float:
        document_tokens = lexical_tokens(chunk_search_text(chunk))
        coverage = (
            len(analysis.tokens & document_tokens) / len(analysis.tokens)
            if analysis.tokens
            else 0.0
        )
        return min(1.0, coverage * 0.65 + exact_signal_score(analysis, chunk) * 0.35)

    def _select_mmr(
        self,
        chunks: list[RetrievedChunk],
        relevance: np.ndarray,
        top_k: int,
    ) -> list[int]:
        if not chunks:
            return []
        remaining = set(range(len(chunks)))
        selected: list[int] = []
        while remaining and len(selected) < top_k:
            best_index = None
            best_key = None
            for index in sorted(remaining):
                diversity = 0.0
                vector = self._embedding_by_id.get(chunks[index]["chunk_id"])
                if selected and vector is not None:
                    similarities = []
                    for chosen in selected:
                        chosen_vector = self._embedding_by_id.get(chunks[chosen]["chunk_id"])
                        if chosen_vector is not None:
                            similarities.append(float(vector @ chosen_vector))
                    diversity = max(similarities, default=0.0)
                mmr = self.tuning.mmr_lambda * float(relevance[index]) - (
                    1 - self.tuning.mmr_lambda
                ) * max(diversity, 0.0)
                key = (mmr, float(relevance[index]), -index)
                if best_key is None or key > best_key:
                    best_key = key
                    best_index = index
            assert best_index is not None
            selected.append(best_index)
            remaining.remove(best_index)
        return selected

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        college: str | None = None,
        cohort: str | None = None,
    ) -> list[RetrievedChunk]:
        return self.retrieve_scoped(
            query,
            top_k=top_k,
            college=college,
            cohort=cohort,
        )

    def retrieve_scoped(
        self,
        query: str,
        top_k: int = 5,
        college: str | None = None,
        cohort: str | None = None,
        *,
        policy_year: int | None = None,
        topic: str | None = None,
    ) -> list[RetrievedChunk]:
        analysis = analyze_query(query)
        window = min(50, max(self.tuning.candidate_k, top_k * 4))
        candidates = self._deduplicate(
            self.core.retrieve_scoped(
                analysis.expanded,
                window,
                college,
                cohort,
                policy_year=policy_year,
                topic=topic,
            )
        )
        if not candidates:
            return []

        dense = np.asarray(
            [max(0.0, min(1.0, (chunk["score"] + 1.0) / 2.0)) for chunk in candidates],
            dtype=np.float32,
        )
        lexical = np.asarray(
            [self._lexical_score(analysis, chunk) for chunk in candidates],
            dtype=np.float32,
        )
        prior = np.asarray(
            [1.0 / (rank + 1) for rank in range(len(candidates))], dtype=np.float32
        )
        active_reranker = self.reranker or HeuristicReranker(analysis, candidates)
        reranked = rerank_documents(active_reranker, analysis.normalized, candidates)
        relevance = (
            self.tuning.dense_weight * dense
            + self.tuning.lexical_weight * lexical
            + self.tuning.rerank_weight * reranked
            + self.tuning.rank_prior_weight * prior
        )
        stable_order = sorted(
            range(len(candidates)),
            key=lambda index: (
                -float(relevance[index]),
                -float(reranked[index]),
                candidates[index]["chunk_id"],
            ),
        )
        ordered_chunks = [candidates[index] for index in stable_order]
        ordered_scores = relevance[stable_order]
        selected = self._select_mmr(ordered_chunks, ordered_scores, min(top_k, len(ordered_chunks)))
        return [ordered_chunks[index] for index in selected]


_default_retriever: AdvancedRetriever | None = None
_default_lock = RLock()


def configure_default(retriever: AdvancedRetriever | None) -> None:
    global _default_retriever
    with _default_lock:
        _default_retriever = retriever


def _get_default() -> AdvancedRetriever:
    global _default_retriever
    with _default_lock:
        if _default_retriever is None:
            _default_retriever = AdvancedRetriever.from_artifacts()
        return _default_retriever


def retrieve(
    query: str,
    top_k: int = 5,
    college: str | None = None,
    cohort: str | None = None,
) -> list[RetrievedChunk]:
    return _get_default().retrieve(query, top_k, college, cohort)
