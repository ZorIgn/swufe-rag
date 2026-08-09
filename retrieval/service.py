"""Bounded lexical retrieval utility used for offline diagnostics.

Production policy retrieval is composed through :class:`HybridPolicyRetriever`.
This module intentionally remains a real BM25 implementation for callers that
need an explicitly lexical baseline, rather than a character-count surrogate.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import SupportsFloat, SupportsIndex

from retrieval.lexical import BM25LexicalIndex
from retrieval.scoring import RetrievedCandidate, presentation_score, reciprocal_rank_fusion


def _float_or_zero(value: object) -> float:
    """Mirror ``float(value or 0)`` after narrowing an untyped metadata field."""

    candidate = value or 0
    if isinstance(candidate, (str, bytes, bytearray, int, float)):
        return float(candidate)
    if isinstance(candidate, SupportsFloat):
        return float(candidate)
    if isinstance(candidate, SupportsIndex):
        return float(candidate)
    raise TypeError(f"metadata numeric value is not convertible to float: {candidate!r}")


def _authority_score(metadata: Mapping[str, object]) -> float:
    return min(1.0, _float_or_zero(metadata.get("authority_level")) / 3.0)


@dataclass
class BoundedScopeCache:
    """Small LRU cache for metadata-filtered candidate identifiers."""

    max_entries: int = 128
    dataset_version: str = "unknown"
    _values: OrderedDict[tuple[tuple[str, str], ...], tuple[str, ...]] = field(
        default_factory=OrderedDict
    )
    hits: int = 0
    misses: int = 0

    def get(self, key: tuple[tuple[str, str], ...], dataset_version: str) -> tuple[str, ...] | None:
        if dataset_version != self.dataset_version:
            self._values.clear()
            self.dataset_version = dataset_version
        value = self._values.get(key)
        if value is None:
            self.misses += 1
            return None
        self._values.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key: tuple[tuple[str, str], ...], value: tuple[str, ...]) -> None:
        self._values[key] = value
        self._values.move_to_end(key)
        while len(self._values) > self.max_entries:
            self._values.popitem(last=False)

    def metrics(self) -> dict[str, int]:
        return {
            "max_entries": self.max_entries,
            "entries": len(self._values),
            "hits": self.hits,
            "misses": self.misses,
        }


class ScopedRetriever:
    """Actual BM25 baseline with metadata filtering performed before ranking."""

    def __init__(
        self,
        documents: Iterable[dict[str, object]],
        *,
        dataset_version: str,
        scope_cache: BoundedScopeCache | None = None,
    ) -> None:
        self._documents = {str(item["chunk_id"]): dict(item) for item in documents}
        self._dataset_version = dataset_version
        self._cache = scope_cache or BoundedScopeCache(dataset_version=dataset_version)
        self._lexical = BM25LexicalIndex(self._documents.values())

    def retrieve(
        self,
        query: str,
        *,
        scope: dict[str, str] | None = None,
        limit: int = 8,
    ) -> tuple[RetrievedCandidate, ...]:
        requested_scope = tuple(sorted((scope or {}).items()))
        candidate_ids = self._cache.get(requested_scope, self._dataset_version)
        if candidate_ids is None:
            candidate_ids = tuple(
                identifier
                for identifier, document in self._documents.items()
                if all(str(document.get(field, "")) == value for field, value in requested_scope)
            )
            self._cache.put(requested_scope, candidate_ids)
        lexical = self._lexical.rank(query, candidate_ids, limit=max(1, limit))
        rrf = reciprocal_rank_fusion([[item.chunk_id for item in lexical]])
        return tuple(
            item.model_copy(
                update={
                    "rrf_score": rrf[item.chunk_id],
                    "fused_rank": item.lexical_rank,
                    "scope_score": 1.0 if requested_scope else 0.5,
                    "authority_score": _authority_score(item.metadata),
                    "final_score": presentation_score(
                        fused_rank=item.lexical_rank,
                        total=max(1, len(lexical)),
                        reranker_score=None,
                        authority_score=_authority_score(item.metadata),
                        scope_score=1.0 if requested_scope else 0.5,
                    ),
                }
            )
            for item in lexical
        )

    @property
    def cache_metrics(self) -> dict[str, int]:
        return self._cache.metrics()


__all__ = ["BoundedScopeCache", "ScopedRetriever"]
