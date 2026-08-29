"""Scoped lexical and hybrid policy retrievers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import date, datetime
from time import perf_counter
from typing import Protocol, SupportsFloat, SupportsIndex

import numpy as np

from retrieval.dense import DenseUnavailableError
from retrieval.lexical import BM25LexicalIndex, tokenize
from retrieval.models import PolicyRetrievalRequest, PolicyRetrievalResult
from retrieval.scope import policy_scope_matches
from retrieval.scoring import RetrievedCandidate, presentation_score, reciprocal_rank_fusion


def _string_values(value: object) -> tuple[str, ...]:
    """Narrow a JSON-like iterable while retaining legacy iterable semantics."""

    values = value or ()
    if not isinstance(values, Iterable):
        raise TypeError(f"expected an iterable policy metadata value, got {values!r}")
    return tuple(str(item) for item in values)


def _float_or_zero(value: object) -> float:
    """Mirror ``float(value or 0)`` after narrowing an untyped metadata field."""

    candidate = value or 0
    if isinstance(candidate, (str, bytes, bytearray, int, float)):
        return float(candidate)
    if isinstance(candidate, SupportsFloat):
        return float(candidate)
    if isinstance(candidate, SupportsIndex):
        return float(candidate)
    raise TypeError(f"policy numeric value is not convertible to float: {candidate!r}")


def _authority_score(metadata: dict[str, object]) -> float:
    return min(1.0, _float_or_zero(metadata.get("authority_level")) / 3.0)


def _as_date(value: object) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) == 4 and text.isdigit():
        return date(int(text), 12, 31)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _requested_date(value: str | None) -> date:
    if value is None:
        return datetime.now().date()
    parsed = _as_date(value)
    if parsed is None:
        raise ValueError("policy as_of must be an ISO date or four-digit year")
    return parsed


def _scope_match(document: dict[str, object], request: PolicyRetrievalRequest) -> bool:
    """Apply trust, time, supersession, cohort, college and program scope first."""

    if str(document.get("superseded") or "").lower() == "true":
        return False
    target = _requested_date(request.as_of)
    start = _as_date(document.get("effective_from"))
    end = _as_date(document.get("effective_to"))
    if start is not None and start > target:
        return False
    if end is not None and end < target:
        return False
    if request.as_of is None and str(document.get("status") or "") != "现行":
        return False
    if not policy_scope_matches(
        document,
        cohort=request.cohort,
        program_ids=request.program_ids,
        college_ids=request.college_ids,
    ):
        return False
    if not request.topics:
        return True
    requested_topics = {value.lower() for value in request.topics}
    topics = {value.lower() for value in _string_values(document.get("topics", ()))}
    if topics:
        return bool(requested_topics.intersection(topics))
    haystack = " ".join(
        str(document.get(key) or "") for key in ("title", "article", "text")
    ).lower()
    return any(topic in haystack for topic in requested_topics)


def _exact_entity_score(query: str, document: dict[str, object]) -> float:
    haystack = " ".join(
        [
            str(document.get("title") or ""),
            str(document.get("article") or ""),
            str(document.get("text") or ""),
        ]
    ).lower()
    terms = set(tokenize(query))
    return sum(1.0 for term in terms if term in haystack)


class DenseIndex(Protocol):
    chunk_ids: tuple[str, ...]
    dimension: int

    def vector_for(self, chunk_id: str) -> np.ndarray | None: ...

    def rank(
        self,
        query: str,
        documents: dict[str, dict[str, object]],
        candidate_ids: set[str],
        *,
        limit: int,
    ) -> tuple[RetrievedCandidate, ...]: ...


class CandidateReranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: Iterable[RetrievedCandidate],
    ) -> tuple[RetrievedCandidate, ...]: ...


class HybridPolicyRetriever:
    """Production policy retrieval with real BM25, optional dense RRF, rerank and MMR."""

    def __init__(
        self,
        documents: Iterable[dict[str, object]],
        *,
        mode: str = "lexical",
        dense_index: DenseIndex | None = None,
        reranker: CandidateReranker | None = None,
        dataset_version: str = "unknown",
        index_version: str | None = None,
        expected_chunk_ids: tuple[str, ...] | None = None,
        expected_dimension: int | None = None,
        reranker_min_score: float = 0.50,
        metric_sink: Callable[..., None] | None = None,
    ) -> None:
        if mode not in {"lexical", "hybrid"}:
            raise ValueError("retrieval mode must be lexical or hybrid")
        self.mode = mode
        self.dataset_version = dataset_version
        self.index_version = index_version
        self._documents = {str(value["chunk_id"]): dict(value) for value in documents}
        self._lexical = BM25LexicalIndex(self._documents.values())
        self._dense = dense_index
        self._reranker = reranker
        self._expected_chunk_ids = expected_chunk_ids
        self._expected_dimension = expected_dimension
        self._reranker_min_score = reranker_min_score
        self._metric_sink = metric_sink

    def _emit(self, name: str, value: float = 1.0, **attributes: object) -> None:
        if self._metric_sink is not None:
            self._metric_sink(name, value, retrieval_mode=self.mode, **attributes)

    def _finish(
        self, result: PolicyRetrievalResult, *, started_at: float, scoped_count: int
    ) -> PolicyRetrievalResult:
        self._emit("retrieval_latency_ms", (perf_counter() - started_at) * 1000)
        self._emit("retrieval_candidate_count", float(result.candidate_count))
        self._emit(
            "retrieval_scope_filtered_count",
            float(max(0, len(self._documents) - scoped_count)),
        )
        return result

    def readiness(self) -> tuple[bool, tuple[str, ...]]:
        reasons: list[str] = []
        if not self._documents:
            reasons.append("retrieval_documents_missing")
        if self.mode == "hybrid":
            if self._dense is None:
                reasons.append("dense_index_missing")
            if self._reranker is None:
                reasons.append("reranker_missing")
            if self.index_version != self.dataset_version:
                reasons.append("retrieval_dataset_version_mismatch")
            if self._dense is not None:
                document_ids = tuple(self._documents)
                if self._dense.chunk_ids != document_ids:
                    reasons.append("retrieval_index_chunk_ids_mismatch")
                if (
                    self._expected_chunk_ids is not None
                    and self._expected_chunk_ids != document_ids
                ):
                    reasons.append("retrieval_manifest_chunk_ids_mismatch")
                if (
                    self._expected_dimension is not None
                    and self._dense.dimension != self._expected_dimension
                ):
                    reasons.append("retrieval_embedding_dimension_mismatch")
        return not reasons, tuple(reasons)

    def _scoped(
        self, request: PolicyRetrievalRequest
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        scoped = [
            document for document in self._documents.values() if _scope_match(document, request)
        ]
        superseded_source_ids = {
            str(document["supersedes_source_id"])
            for document in scoped
            if document.get("supersedes_source_id")
        }
        scoped = [
            document
            for document in scoped
            if str(document.get("source_id") or "") not in superseded_source_ids
        ]
        verified = [
            document for document in scoped if str(document.get("review_status")) == "verified"
        ]
        review = [
            document for document in scoped if str(document.get("review_status")) != "verified"
        ]
        return verified, review

    def _rank_review(
        self, request: PolicyRetrievalRequest, documents: list[dict[str, object]]
    ) -> tuple[RetrievedCandidate, ...]:
        return self._lexical.rank(
            request.query, (str(item["chunk_id"]) for item in documents), limit=request.top_k
        )

    def _mmr(
        self, candidates: tuple[RetrievedCandidate, ...], *, limit: int
    ) -> tuple[RetrievedCandidate, ...]:
        dense = self._dense
        if dense is None:
            return candidates[:limit]
        remaining = list(candidates)
        selected: list[RetrievedCandidate] = []
        while remaining and len(selected) < limit:

            def score(candidate: RetrievedCandidate) -> tuple[float, str]:
                vector = dense.vector_for(candidate.chunk_id)
                diversity = 0.0
                if vector is not None and selected:
                    diversity = max(
                        float(np.dot(vector, selected_vector))
                        for selected_vector in (
                            dense.vector_for(item.chunk_id) for item in selected
                        )
                        if selected_vector is not None
                    )
                relevance = candidate.reranker_score
                if relevance is None:
                    relevance = candidate.rrf_score
                return (0.80 * relevance - 0.20 * diversity, candidate.chunk_id)

            best = max(remaining, key=score)
            remaining.remove(best)
            selected.append(best)
        return tuple(selected)

    def retrieve(self, request: PolicyRetrievalRequest) -> PolicyRetrievalResult:
        started_at = perf_counter()
        result = self._retrieve(request)
        return self._finish(result, started_at=started_at, scoped_count=result.scope_filtered_count)

    def _retrieve(self, request: PolicyRetrievalRequest) -> PolicyRetrievalResult:
        verified, review = self._scoped(request)
        review_candidates = self._rank_review(request, review)
        if not verified:
            warnings: tuple[str, ...] = ("policy_verified_evidence_unavailable",)
            if review_candidates:
                warnings += ("policy_review_candidates_available",)
            return PolicyRetrievalResult(
                review_candidates=review_candidates,
                scope_filtered_count=len(verified) + len(review),
                candidate_count=0,
                retrieval_mode=self.mode,
                warnings=warnings,
            )
        ids = tuple(str(item["chunk_id"]) for item in verified)
        lexical = self._lexical.rank(
            request.query, ids, limit=max(request.top_k * 8, request.top_k)
        )
        if self.mode == "lexical":
            rrf = reciprocal_rank_fusion([[item.chunk_id for item in lexical]])
            values = tuple(
                item.model_copy(
                    update={
                        "rrf_score": rrf[item.chunk_id],
                        "fused_rank": item.lexical_rank,
                        "exact_entity_score": _exact_entity_score(request.query, item.metadata),
                        "scope_score": 1.0,
                        "authority_score": _authority_score(item.metadata),
                    }
                )
                for item in lexical[: request.top_k]
            )
            ranked = tuple(
                item.model_copy(
                    update={
                        "final_score": presentation_score(
                            fused_rank=item.fused_rank,
                            total=max(1, len(values)),
                            reranker_score=None,
                            authority_score=item.authority_score,
                            scope_score=item.scope_score,
                        )
                    }
                )
                for item in values
            )
            return PolicyRetrievalResult(
                candidates=ranked,
                review_candidates=review_candidates,
                scope_filtered_count=len(verified) + len(review),
                candidate_count=len(ranked),
                retrieval_mode=self.mode,
                warnings=tuple(
                    value
                    for value, include in (
                        ("lexical_mode", True),
                        ("policy_review_candidates_available", bool(review_candidates)),
                        ("policy_no_relevant_candidates", not ranked),
                    )
                    if include
                ),
            )
        if self._dense is None or self._reranker is None:
            raise DenseUnavailableError("hybrid policy retrieval is not ready")
        dense_scope_ids = set(ids)
        dense = self._dense.rank(
            request.query,
            self._documents,
            dense_scope_ids,
            limit=max(request.top_k * 8, request.top_k),
        )
        leaked_dense_ids = tuple(
            item.chunk_id for item in dense if item.chunk_id not in dense_scope_ids
        )
        if leaked_dense_ids:
            raise DenseUnavailableError(
                "dense retriever returned out-of-scope candidates: "
                + ", ".join(sorted(set(leaked_dense_ids)))
            )
        lexical_by_id = {item.chunk_id: item for item in lexical}
        dense_by_id = {item.chunk_id: item for item in dense}
        rrf = reciprocal_rank_fusion(
            [[item.chunk_id for item in lexical], [item.chunk_id for item in dense]]
        )
        fused_ids = sorted(rrf, key=lambda item: (-rrf[item], item))[
            : max(request.top_k * 6, request.top_k)
        ]
        fused: list[RetrievedCandidate] = []
        for rank, identifier in enumerate(fused_ids, start=1):
            lexical_item = lexical_by_id.get(identifier)
            dense_item = dense_by_id.get(identifier)
            source = lexical_item or dense_item
            assert source is not None
            fused.append(
                source.model_copy(
                    update={
                        "bm25_score": lexical_item.bm25_score if lexical_item else None,
                        "dense_score": dense_item.dense_score if dense_item else None,
                        "lexical_rank": lexical_item.lexical_rank if lexical_item else None,
                        "dense_rank": dense_item.dense_rank if dense_item else None,
                        "rrf_score": rrf[identifier],
                        "fused_rank": rank,
                        "exact_entity_score": _exact_entity_score(request.query, source.metadata),
                        "scope_score": 1.0,
                        "authority_score": _authority_score(source.metadata),
                    }
                )
            )
        reranked = self._reranker.rerank(request.query, fused)
        relevant = tuple(
            item
            for item in reranked
            if item.reranker_score is not None and item.reranker_score >= self._reranker_min_score
        )
        diversified = self._mmr(relevant, limit=request.top_k)
        values = tuple(
            item.model_copy(
                update={
                    "final_score": presentation_score(
                        fused_rank=item.fused_rank,
                        total=max(1, len(fused)),
                        reranker_score=item.reranker_score,
                        authority_score=item.authority_score,
                        scope_score=item.scope_score,
                    )
                }
            )
            for item in diversified
        )
        return PolicyRetrievalResult(
            candidates=values,
            review_candidates=review_candidates,
            scope_filtered_count=len(verified) + len(review),
            candidate_count=len(values),
            retrieval_mode=self.mode,
            warnings=tuple(
                value
                for value, include in (
                    ("policy_review_candidates_available", bool(review_candidates)),
                    ("policy_no_relevant_candidates", not values),
                )
                if include
            ),
        )


__all__ = ["CandidateReranker", "DenseIndex", "HybridPolicyRetriever"]
