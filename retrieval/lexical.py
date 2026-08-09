"""Real BM25 lexical retrieval over metadata-filtered policy sections."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - dependency-light offline fixtures

    class BM25Okapi:  # type: ignore[no-redef]
        """Small standards-compliant BM25Okapi fallback for offline builds."""

        def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
            self.corpus = corpus
            self.k1 = k1
            self.b = b
            self.doc_len = [len(item) for item in corpus]
            self.avgdl = sum(self.doc_len) / max(1, len(self.doc_len))
            self.term_freqs = [Counter(item) for item in corpus]
            document_frequency: Counter[str] = Counter()
            for item in self.term_freqs:
                document_frequency.update(item.keys())
            self.idf = {
                term: math.log(1.0 + (len(corpus) - frequency + 0.5) / (frequency + 0.5))
                for term, frequency in document_frequency.items()
            }

        def get_scores(self, query: list[str]) -> list[float]:
            query_terms = set(query)
            scores: list[float] = []
            for frequencies, length in zip(self.term_freqs, self.doc_len, strict=True):
                score = 0.0
                for term in query_terms:
                    frequency = frequencies.get(term, 0)
                    if not frequency:
                        continue
                    denominator = frequency + self.k1 * (
                        1 - self.b + self.b * length / max(1.0, self.avgdl)
                    )
                    score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / denominator
                scores.append(score)
            return scores


from retrieval.scoring import RetrievedCandidate

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u3400-\u9fff]")
_MEANINGFUL_RE = re.compile(r"[A-Za-z0-9]+|[\u3400-\u9fff]+")
_GENERIC_QUERY_TERMS = frozenset(
    {
        "什么",
        "哪些",
        "怎么",
        "如何",
        "是否",
        "可以",
        "规定",
        "办法",
        "条件",
        "要求",
        "说明",
        "相关",
    }
)


def tokenize(text: str) -> list[str]:
    """Use CJK characters plus alphanumeric spans for the compact BM25 index."""

    return [value.lower() for value in _TOKEN_RE.findall(text) if value.strip()]


def _meaningful_terms(text: str) -> set[str]:
    cleaned = text
    for generic in sorted(_GENERIC_QUERY_TERMS, key=len, reverse=True):
        cleaned = cleaned.replace(generic, " ")
    values: set[str] = set()
    for value in _MEANINGFUL_RE.findall(cleaned):
        if all("\u3400" <= character <= "\u9fff" for character in value):
            if len(value) >= 2:
                values.update(value[index : index + 2] for index in range(len(value) - 1))
        else:
            values.add(value.lower())
    return values - _GENERIC_QUERY_TERMS


class BM25LexicalIndex:
    """An immutable BM25 corpus; scope is applied before scoring."""

    def __init__(self, documents: Iterable[dict[str, object]]) -> None:
        self._documents = tuple(dict(item) for item in documents)
        self._ids = tuple(str(item["chunk_id"]) for item in self._documents)
        self._positions = {identifier: index for index, identifier in enumerate(self._ids)}
        self._bm25 = (
            BM25Okapi([tokenize(str(item.get("text", ""))) for item in self._documents])
            if self._documents
            else None
        )

    @property
    def documents(self) -> tuple[dict[str, object], ...]:
        return self._documents

    def rank(
        self,
        query: str,
        candidate_ids: Iterable[str],
        *,
        limit: int,
        min_score: float | None = None,
    ) -> tuple[RetrievedCandidate, ...]:
        ids = tuple(candidate_ids)
        if not ids or self._bm25 is None:
            return ()
        query_terms = _meaningful_terms(query)
        if not query_terms:
            return ()
        scores = self._bm25.get_scores(tokenize(query))
        values: list[tuple[str, float]] = []
        for identifier in ids:
            position = self._positions.get(identifier)
            if position is not None:
                score = float(scores[position])
                # BM25 libraries return a score for every document.  A zero
                # score means the query shares no lexical evidence with that
                # document and must not be promoted merely because it has a
                # good scope/authority prior later in the pipeline.
                document_terms = _meaningful_terms(
                    str(self._documents[position].get("text", ""))
                )
                overlap = len(query_terms.intersection(document_terms))
                score_is_relevant = score != 0.0 if min_score is None else score > min_score
                if score_is_relevant and overlap >= 1:
                    values.append((identifier, score))
        values.sort(key=lambda item: (-item[1], item[0]))
        candidates: list[RetrievedCandidate] = []
        for rank, (identifier, score) in enumerate(values[:limit], start=1):
            document = self._documents[self._positions[identifier]]
            candidates.append(
                RetrievedCandidate(
                    chunk_id=identifier,
                    text=str(document.get("text", "")),
                    metadata=document,
                    bm25_score=score,
                    lexical_rank=rank,
                )
            )
        return tuple(candidates)


__all__ = ["BM25LexicalIndex", "tokenize"]
