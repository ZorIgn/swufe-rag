"""Bounded, metadata-filtered lexical baseline for canonical policy retrieval.

Dense encoders and rerankers can be supplied by deployment code, but the public
score model remains explicit and the scope cache cannot grow without bound.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field

from retrieval.scoring import RetrievedCandidate, final_score, reciprocal_rank_fusion


def _terms(query: str) -> tuple[str, ...]:
    tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}", query)
    return tuple(dict.fromkeys(token.lower() for token in tokens if len(token) > 1))


@dataclass
class BoundedScopeCache:
    max_entries: int = 128
    dataset_version: str = "unknown"
    _values: OrderedDict[tuple[str, tuple[tuple[str, str], ...]], tuple[str, ...]] = field(default_factory=OrderedDict)
    hits: int = 0
    misses: int = 0

    def get(self, key: tuple[str, tuple[tuple[str, str], ...]], dataset_version: str) -> tuple[str, ...] | None:
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

    def put(self, key: tuple[str, tuple[tuple[str, str], ...]], value: tuple[str, ...]) -> None:
        self._values[key] = value
        self._values.move_to_end(key)
        while len(self._values) > self.max_entries:
            self._values.popitem(last=False)

    def metrics(self) -> dict[str, int]:
        return {"max_entries": self.max_entries, "entries": len(self._values), "hits": self.hits, "misses": self.misses, "memory_estimate_bytes": sum(sum(len(item) for item in value) for value in self._values.values())}


class ScopedRetriever:
    def __init__(self, documents: Iterable[dict[str, object]], *, dataset_version: str, scope_cache: BoundedScopeCache | None = None) -> None:
        self._documents = {str(value["chunk_id"]): dict(value) for value in documents}
        self._dataset_version = dataset_version
        self._cache = scope_cache or BoundedScopeCache(dataset_version=dataset_version)

    def retrieve(self, query: str, *, scope: dict[str, str] | None = None, limit: int = 8) -> tuple[RetrievedCandidate, ...]:
        requested_scope = tuple(sorted((scope or {}).items()))
        key = (query, requested_scope)
        candidate_ids = self._cache.get(key, self._dataset_version)
        if candidate_ids is None:
            candidate_ids = tuple(identifier for identifier, document in self._documents.items() if all(str(document.get(field, "")) == value for field, value in requested_scope))
            self._cache.put(key, candidate_ids)
        terms = _terms(query)
        candidates: list[RetrievedCandidate] = []
        for identifier in candidate_ids:
            document = self._documents[identifier]
            text = str(document.get("text", ""))
            lexical = sum(text.lower().count(term) * len(term) for term in terms)
            entity = sum(float(term in text.lower()) for term in terms)
            base = RetrievedCandidate(chunk_id=identifier, text=text, metadata=document, bm25_score=float(lexical), exact_entity_score=entity, scope_score=1.0 if requested_scope else 0.5)
            candidates.append(base)
        lexical_order = [item.chunk_id for item in sorted(candidates, key=lambda item: item.bm25_score, reverse=True)]
        rrf = reciprocal_rank_fusion([lexical_order])
        ranked = [item.__class__(**{**item.__dict__, "rrf_score": rrf.get(item.chunk_id, 0.0)}) for item in candidates]
        ranked = [item.__class__(**{**item.__dict__, "final_score": final_score(item)}) for item in ranked]
        return tuple(sorted(ranked, key=lambda item: item.final_score, reverse=True)[:limit])

    @property
    def cache_metrics(self) -> dict[str, int]:
        return self._cache.metrics()
