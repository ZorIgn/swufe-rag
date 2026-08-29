"""Regression coverage for exact dense ranking inside a retrieval scope."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from retrieval.dense import DenseFaissIndex, DenseUnavailableError


@dataclass
class _GlobalTopFaiss:
    """A FAISS-shaped fake whose global top rows exclude a narrow-scope row."""

    d: int
    ntotal: int
    search_calls: int = 0

    def add(self, vectors: np.ndarray) -> None:  # pragma: no cover - protocol surface only
        del vectors

    def search(self, vectors: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
        self.search_calls += 1
        del vectors
        positions = np.arange(min(limit, self.ntotal), dtype=np.int64)
        scores = np.linspace(1.0, 0.1, num=len(positions), dtype=np.float32)
        return scores[None, :], positions[None, :]


@dataclass
class _StaticEncoder:
    """Offline encoder with explicit query output and call recording."""

    result: object
    calls: list[tuple[tuple[str, ...], bool, bool]] = field(default_factory=list)

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int | None = None,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
    ) -> object:
        assert batch_size is None
        self.calls.append((tuple(sentences), normalize_embeddings, show_progress_bar))
        return self.result


def _documents(chunk_ids: tuple[str, ...]) -> dict[str, dict[str, object]]:
    return {
        identifier: {"chunk_id": identifier, "text": f"evidence for {identifier}"}
        for identifier in chunk_ids
    }


def _index(
    *,
    chunk_ids: tuple[str, ...],
    vectors: np.ndarray,
    query_vector: object,
) -> tuple[DenseFaissIndex, _GlobalTopFaiss, _StaticEncoder]:
    faiss = _GlobalTopFaiss(d=int(vectors.shape[1]), ntotal=len(chunk_ids))
    encoder = _StaticEncoder(query_vector)
    return (
        DenseFaissIndex(
            model_name="offline-test-encoder",
            chunk_ids=chunk_ids,
            index=faiss,
            encoder=encoder,
            dimension=int(vectors.shape[1]),
            vectors=vectors,
        ),
        faiss,
        encoder,
    )


def _unit_rows(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_scoped_dense_returns_a_narrow_scope_row_evicted_from_global_faiss_top_n() -> None:
    # The 20 global rows all beat ``scope-target`` for query [1, 0].  The old
    # global FAISS top-(limit * 8) implementation never saw the target.
    global_ids = tuple(f"global-{index:02d}" for index in range(20))
    chunk_ids = (*global_ids, "scope-target")
    vectors = np.asarray([[1.0, 0.0] for _identifier in global_ids] + [[0.6, 0.8]], dtype="float32")
    dense, faiss, encoder = _index(
        chunk_ids=chunk_ids,
        vectors=vectors,
        query_vector=np.asarray([[1.0, 0.0]], dtype="float32"),
    )

    results = dense.rank("narrow scope query", _documents(chunk_ids), {"scope-target"}, limit=1)

    assert [item.chunk_id for item in results] == ["scope-target"]
    assert results[0].dense_score == pytest.approx(0.6)
    assert results[0].dense_rank == 1
    assert faiss.search_calls == 0
    assert encoder.calls == [(("narrow scope query",), True, False)]


def test_scoped_dense_matches_the_exact_scope_oracle_and_never_leaks() -> None:
    generator = np.random.default_rng(20260819)
    chunk_ids = tuple(f"chunk-{index:02d}" for index in range(37))
    vectors = _unit_rows(generator.normal(size=(len(chunk_ids), 7)).astype("float32"))
    query = _unit_rows(generator.normal(size=(1, 7)).astype("float32"))
    dense, faiss, _encoder = _index(
        chunk_ids=chunk_ids,
        vectors=vectors,
        query_vector=query,
    )
    documents = _documents(chunk_ids)
    documents.pop("chunk-31")

    for scope, limit in (
        ({"chunk-31", "chunk-19", "chunk-05", "unknown"}, 3),
        (set(chunk_ids[3:8]), 4),
    ):
        expected = sorted(
            (
                (identifier, float(vectors[chunk_ids.index(identifier)] @ query[0]))
                for identifier in scope
                if identifier in documents and identifier in chunk_ids
            ),
            key=lambda item: (-item[1], item[0]),
        )[:limit]

        results = dense.rank("oracle query", documents, scope, limit=limit)

        assert [item.chunk_id for item in results] == [
            identifier for identifier, _score in expected
        ]
        assert [item.dense_score for item in results] == pytest.approx(
            [score for _identifier, score in expected]
        )
        assert {item.chunk_id for item in results}.issubset(scope)
        assert [item.dense_rank for item in results] == list(range(1, len(results) + 1))

    assert faiss.search_calls == 0


def test_scoped_dense_ties_are_chunk_id_deterministic_and_missing_scopes_are_stable() -> None:
    chunk_ids = ("zeta", "alpha", "row-without-document")
    vectors = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype="float32")
    dense, faiss, encoder = _index(
        chunk_ids=chunk_ids,
        vectors=vectors,
        query_vector=np.asarray([[1.0, 0.0]], dtype="float32"),
    )
    documents = _documents(("zeta", "alpha"))

    assert dense.rank("unused", documents, set(), limit=2) == ()
    assert dense.rank("also unused", documents, {"missing", "row-without-document"}, limit=2) == ()
    results = dense.rank("tie query", documents, {"zeta", "alpha", "missing"}, limit=2)

    assert [item.chunk_id for item in results] == ["alpha", "zeta"]
    assert [item.dense_rank for item in results] == [1, 2]
    assert faiss.search_calls == 0
    assert encoder.calls == [(("tie query",), True, False)]


@pytest.mark.parametrize(
    ("query_vector", "message"),
    [
        (np.asarray([1.0, 0.0], dtype="float32"), "shape"),
        (np.asarray([[1.0, 0.0, 0.0]], dtype="float32"), "shape"),
        (np.asarray([[np.nan, 0.0]], dtype="float32"), "non-finite"),
        (np.asarray([[0.0, 0.0]], dtype="float32"), "normalized"),
        (np.asarray([[2.0, 0.0]], dtype="float32"), "normalized"),
    ],
)
def test_scoped_dense_rejects_invalid_query_embedding_contract(
    query_vector: np.ndarray, message: str
) -> None:
    chunk_ids = ("only",)
    dense, faiss, _encoder = _index(
        chunk_ids=chunk_ids,
        vectors=np.asarray([[1.0, 0.0]], dtype="float32"),
        query_vector=query_vector,
    )

    with pytest.raises(DenseUnavailableError, match=message):
        dense.rank("invalid query", _documents(chunk_ids), {"only"}, limit=1)

    assert faiss.search_calls == 0


def test_dense_rejects_non_normalized_vector_rows_before_scoped_ranking() -> None:
    with pytest.raises(DenseUnavailableError, match="normalized rows"):
        _index(
            chunk_ids=("only",),
            vectors=np.asarray([[2.0, 0.0]], dtype="float32"),
            query_vector=np.asarray([[1.0, 0.0]], dtype="float32"),
        )
