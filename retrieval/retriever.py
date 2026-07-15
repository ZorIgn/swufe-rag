"""Filtered dense + BM25 retrieval with deterministic RRF fusion."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import re
from threading import RLock
from typing import Any, Sequence

import numpy as np

from contracts import CHUNK_FIELDS, RetrievedChunk
from retrieval.embed import BGEEncoder, Encoder, normalize_rows
from retrieval.index import IndexBundle, load_index
from storage.metadata_db import MetadataDB


def tokenize(text: str) -> list[str]:
    normalized = text.lower()
    ascii_tokens = re.findall(r"[a-z]+\d*|\d+(?:\.\d+)?%?", normalized)
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", normalized)
    try:
        import jieba
    except ImportError:
        chinese_tokens: list[str] = []
        for run in chinese_runs:
            chinese_tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
            chinese_tokens.extend(run)
    else:
        chinese_tokens = [
            token.strip()
            for run in chinese_runs
            for token in jieba.lcut(run, cut_all=False)
            if token.strip()
        ]
    return ascii_tokens + chinese_tokens


class SimpleBM25:
    """Dependency-free BM25 fallback used when rank_bm25 is unavailable."""

    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = [list(document) for document in corpus]
        self.k1 = k1
        self.b = b
        self.lengths = [len(document) for document in self.corpus]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        self.term_frequencies = [Counter(document) for document in self.corpus]
        document_frequency: Counter[str] = Counter()
        for document in self.corpus:
            document_frequency.update(set(document))
        count = len(self.corpus)
        self.idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def get_scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        scores = np.zeros(len(self.corpus), dtype=np.float32)
        for index, frequencies in enumerate(self.term_frequencies):
            length_ratio = self.lengths[index] / max(self.average_length, 1.0)
            for term in query_tokens:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                scores[index] += self.idf.get(term, 0.0) * (
                    frequency * (self.k1 + 1) / denominator
                )
        return scores


def make_bm25(corpus: Sequence[Sequence[str]]):
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return SimpleBM25(corpus)
    return BM25Okapi(corpus)


@dataclass
class ScopeView:
    global_indices: np.ndarray
    embeddings: np.ndarray
    bm25: Any


class HybridRetriever:
    def __init__(
        self,
        bundle: IndexBundle,
        encoder: Encoder,
        *,
        candidate_k: int = 20,
        rrf_k: int = 60,
        use_bm25: bool = True,
        metadata_db: MetadataDB | None = None,
    ) -> None:
        if candidate_k < 1 or rrf_k < 1:
            raise ValueError("candidate_k and rrf_k must be positive")
        if bundle.model_name != encoder.model_name:
            raise ValueError("bundle and encoder model names do not match")
        self.bundle = bundle
        self.encoder = encoder
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.use_bm25 = use_bm25
        self.metadata_db = metadata_db or MetadataDB.from_chunks(
            bundle.chunks, trusted_by_default=True
        )
        self._scope_cache: dict[
            tuple[
                str | None,
                str | None,
                int | None,
                str | None,
                tuple[int, ...],
            ],
            ScopeView,
        ] = {}
        self._lock = RLock()

    @classmethod
    def from_artifacts(
        cls,
        chunks_path: str | Path = "data/chunks.jsonl",
        artifacts_dir: str | Path = "artifacts",
        encoder: Encoder | None = None,
        *,
        candidate_k: int = 20,
        rrf_k: int = 60,
        use_bm25: bool = True,
        sources_path: str | Path = "data/sources.csv",
        metadata_path: str | Path = "data/metadata.sqlite3",
    ) -> "HybridRetriever":
        actual_encoder = encoder or BGEEncoder()
        bundle = load_index(chunks_path, artifacts_dir, actual_encoder)
        metadata_db = MetadataDB.from_files(
            sources_path=sources_path,
            chunks_path=chunks_path,
            database=metadata_path,
        )
        return cls(
            bundle,
            actual_encoder,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            use_bm25=use_bm25,
            metadata_db=metadata_db,
        )

    def _eligible(
        self,
        college: str | None,
        cohort: str | None,
        policy_year: int | None,
        topic: str | None,
    ) -> np.ndarray:
        eligible = self.metadata_db.candidate_rows(
            college=college,
            cohort=cohort,
            policy_year=policy_year,
            topic=topic,
        )
        if any(index >= len(self.bundle.chunks) for index in eligible):
            raise ValueError("metadata embedding_row is outside the loaded index")
        return np.asarray(eligible, dtype=np.int64)

    def _scope(
        self,
        college: str | None,
        cohort: str | None,
        policy_year: int | None,
        topic: str | None,
    ) -> ScopeView:
        # Execute the SQL allow-list on every request.  The row tuple is part of
        # the cache key, so disabling or distrusting a source cannot reuse a
        # stale embedding/BM25 view.
        indices = self._eligible(college, cohort, policy_year, topic)
        key = (college, cohort, policy_year, topic, tuple(indices.tolist()))
        with self._lock:
            cached = self._scope_cache.get(key)
            if cached is not None:
                return cached
            embeddings = (
                self.bundle.embeddings[indices]
                if len(indices)
                else np.empty((0, self.encoder.dimension), dtype=np.float32)
            )
            corpus = [tokenize(self.bundle.chunks[int(index)]["text"]) for index in indices]
            view = ScopeView(
                indices,
                embeddings,
                make_bm25(corpus) if corpus else SimpleBM25([]),
            )
            self._scope_cache[key] = view
            return view

    @staticmethod
    def _validate_arguments(
        query: str, top_k: int, college: str | None, cohort: str | None
    ) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be blank")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 50:
            raise ValueError("top_k must be an integer between 1 and 50")
        for name, value in (("college", college), ("cohort", cohort)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be None or a non-empty string")
        return query.strip()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        college: str | None = None,
        cohort: str | None = None,
    ) -> list[RetrievedChunk]:
        """Frozen retrieval contract; scope values are SQL-bound before ranking."""

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
        clean_query = self._validate_arguments(query, top_k, college, cohort)
        view = self._scope(
            college.strip() if college else None,
            cohort.strip() if cohort else None,
            policy_year,
            topic.strip() if topic else None,
        )
        if len(view.global_indices) == 0:
            return []

        query_vector = normalize_rows(self.encoder.encode_query(clean_query))[0]
        dense_scores = view.embeddings @ query_vector
        dense_order = sorted(
            range(len(dense_scores)),
            key=lambda local: (
                -float(dense_scores[local]),
                self.bundle.chunks[int(view.global_indices[local])]["chunk_id"],
            ),
        )[: min(self.candidate_k, len(dense_scores))]

        bm25_order: list[int] = []
        if self.use_bm25:
            bm25_scores = np.asarray(view.bm25.get_scores(tokenize(clean_query)))
            bm25_order = [
                local
                for local in sorted(
                    range(len(bm25_scores)),
                    key=lambda item: (
                        -float(bm25_scores[item]),
                        self.bundle.chunks[int(view.global_indices[item])]["chunk_id"],
                    ),
                )
                if float(bm25_scores[local]) > 0
            ][: min(self.candidate_k, len(bm25_scores))]

        rrf_scores: dict[int, float] = {}
        for rank, local in enumerate(dense_order, start=1):
            rrf_scores[local] = rrf_scores.get(local, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, local in enumerate(bm25_order, start=1):
            rrf_scores[local] = rrf_scores.get(local, 0.0) + 1.0 / (self.rrf_k + rank)

        final_order = sorted(
            rrf_scores,
            key=lambda local: (
                -rrf_scores[local],
                -float(dense_scores[local]),
                self.bundle.chunks[int(view.global_indices[local])]["chunk_id"],
            ),
        )[: min(top_k, len(rrf_scores))]

        results: list[RetrievedChunk] = []
        for local in final_order:
            chunk = self.bundle.chunks[int(view.global_indices[local])]
            result = {key: chunk[key] for key in CHUNK_FIELDS}
            result["score"] = float(dense_scores[local])
            results.append(result)  # type: ignore[arg-type]
        return results


_default_retriever: HybridRetriever | None = None
_default_lock = RLock()


def configure_default(retriever: HybridRetriever | None) -> None:
    """Inject a retriever for tests or reset to lazy production loading."""

    global _default_retriever
    with _default_lock:
        _default_retriever = retriever


def _get_default() -> HybridRetriever:
    global _default_retriever
    with _default_lock:
        if _default_retriever is None:
            _default_retriever = HybridRetriever.from_artifacts()
        return _default_retriever


def retrieve(
    query: str,
    top_k: int = 5,
    college: str | None = None,
    cohort: str | None = None,
) -> list[RetrievedChunk]:
    """Frozen contract-2 public entry point."""

    return _get_default().retrieve(query, top_k, college, cohort)

