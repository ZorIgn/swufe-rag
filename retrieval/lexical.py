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


def tokenize(text: str) -> list[str]:
    """Use CJK characters plus alphanumeric spans as stable BM25 terms."""

    return [value.lower() for value in _TOKEN_RE.findall(text) if value.strip()]


class BM25LexicalIndex:
    """An immutable BM25 corpus; scope is applied before scoring."""

    def __init__(self, documents: Iterable[dict[str, object]]) -> None:
        self._documents = tuple(dict(item) for item in documents)
        self._ids = tuple(str(item["chunk_id"]) for item in self._documents)
        self._positions = {identifier: index for index, identifier in enumerate(self._ids)}
        self._bm25 = BM25Okapi([tokenize(str(item.get("text", ""))) for item in self._documents])

    @property
    def documents(self) -> tuple[dict[str, object], ...]:
        return self._documents

    def rank(
        self, query: str, candidate_ids: Iterable[str], *, limit: int
    ) -> tuple[RetrievedCandidate, ...]:
        ids = tuple(candidate_ids)
        if not ids:
            return ()
        scores = self._bm25.get_scores(tokenize(query))
        values: list[tuple[str, float]] = []
        for identifier in ids:
            position = self._positions.get(identifier)
            if position is not None:
                values.append((identifier, float(scores[position])))
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
