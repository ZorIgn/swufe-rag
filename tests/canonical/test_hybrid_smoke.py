"""Offline contract smoke for the production hybrid policy retrieval path."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from agent.factory import _build_retriever
from retrieval.dense import DenseFaissIndex, DenseUnavailableError
from retrieval.hybrid import HybridPolicyRetriever
from retrieval.models import PolicyRetrievalRequest
from retrieval.reranker import CrossEncoderReranker
from retrieval.scoring import RetrievedCandidate


@dataclass
class _DeterministicDenseIndex:
    """Small in-memory dense double; no model weights or network access."""

    chunk_ids: tuple[str, ...]
    dimension: int = 2
    calls: list[tuple[str, frozenset[str], int]] = field(default_factory=list)

    _vectors = {
        "policy-a": np.asarray([1.0, 0.0], dtype="float32"),
        "policy-b": np.asarray([0.0, 1.0], dtype="float32"),
        "policy-c": np.asarray([0.5, 0.5], dtype="float32"),
    }

    def vector_for(self, chunk_id: str) -> np.ndarray | None:
        return self._vectors.get(chunk_id)

    def rank(
        self,
        query: str,
        documents: dict[str, dict[str, object]],
        candidate_ids: set[str],
        *,
        limit: int,
    ) -> tuple[RetrievedCandidate, ...]:
        self.calls.append((query, frozenset(candidate_ids), limit))
        order = ("policy-b", "policy-a", "policy-c")
        values = [
            RetrievedCandidate(
                chunk_id=chunk_id,
                text=str(documents[chunk_id]["text"]),
                metadata=documents[chunk_id],
                dense_score=1.0 - position * 0.1,
                dense_rank=position,
            )
            for position, chunk_id in enumerate(order, start=1)
            if chunk_id in candidate_ids
        ]
        return tuple(values[:limit])


@dataclass
class _DeterministicReranker:
    """Records the fused candidates and returns a non-lexical order."""

    calls: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)

    def rerank(
        self, query: str, candidates: tuple[RetrievedCandidate, ...]
    ) -> tuple[RetrievedCandidate, ...]:
        self.calls.append((query, tuple(candidate.chunk_id for candidate in candidates)))
        scores = {"policy-b": 0.95, "policy-a": 0.70, "policy-c": 0.20}
        return tuple(
            sorted(
                (
                    candidate.model_copy(update={"reranker_score": scores[candidate.chunk_id]})
                    for candidate in candidates
                ),
                key=lambda candidate: (-(candidate.reranker_score or 0.0), candidate.chunk_id),
            )
        )


@dataclass
class _LeakingDenseIndex:
    """Deliberately violates the scoped dense contract for boundary coverage."""

    chunk_ids: tuple[str, ...]
    documents: dict[str, dict[str, object]]
    dimension: int = 2

    def vector_for(self, chunk_id: str) -> np.ndarray | None:
        return _DeterministicDenseIndex._vectors.get(chunk_id)

    def rank(
        self,
        query: str,
        documents: dict[str, dict[str, object]],
        candidate_ids: set[str],
        *,
        limit: int,
    ) -> tuple[RetrievedCandidate, ...]:
        del query, documents, candidate_ids, limit
        document = self.documents["policy-b"]
        return (
            RetrievedCandidate(
                chunk_id="policy-b",
                text=str(document["text"]),
                metadata=document,
                dense_score=0.99,
                dense_rank=1,
            ),
        )


def _document(chunk_id: str, text: str) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "review_status": "verified",
        "status": "现行",
        "cohort": "2024",
        "authority_level": 3,
        "effective_from": "2024",
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hybrid_pipeline_executes_dense_rrf_rerank_and_mmr_offline() -> None:
    documents = (
        _document("policy-a", "转专业申请管理办法。"),
        _document("policy-b", "学生转专业资格规定。"),
        _document("policy-c", "学籍异动的补充说明。"),
    )
    dense = _DeterministicDenseIndex(tuple(document["chunk_id"] for document in documents))
    reranker = _DeterministicReranker()
    retriever = HybridPolicyRetriever(
        documents,
        mode="hybrid",
        dense_index=cast(DenseFaissIndex, dense),
        reranker=cast(CrossEncoderReranker, reranker),
        dataset_version="hybrid-smoke-v1",
        index_version="hybrid-smoke-v1",
        expected_chunk_ids=dense.chunk_ids,
        expected_dimension=dense.dimension,
    )

    ready, reasons = retriever.readiness()
    result = retriever.retrieve(
        PolicyRetrievalRequest(
            query="转专业资格有什么规定？", cohort=2024, as_of="2026-01-01", top_k=2
        )
    )

    assert ready, reasons
    assert dense.calls == [("转专业资格有什么规定？", frozenset(dense.chunk_ids), 16)]
    assert reranker.calls and set(reranker.calls[0][1]) == set(dense.chunk_ids)
    assert result.retrieval_mode == "hybrid"
    assert tuple(candidate.chunk_id for candidate in result.candidates) == ("policy-b", "policy-a")
    assert all(candidate.bm25_score is not None for candidate in result.candidates)
    assert all(candidate.dense_score is not None for candidate in result.candidates)
    assert all(candidate.rrf_score > 0.0 for candidate in result.candidates)
    assert all(candidate.reranker_score is not None for candidate in result.candidates)
    assert all(candidate.final_score > 0.0 for candidate in result.candidates)


def test_hybrid_fails_closed_if_a_dense_implementation_leaks_out_of_scope() -> None:
    allowed = _document("policy-a", "转专业申请管理办法。")
    allowed["college_id"] = "allowed-college"
    outside = _document("policy-b", "学生转专业资格规定。")
    outside["college_id"] = "outside-college"
    documents = (allowed, outside)
    dense = _LeakingDenseIndex(
        tuple(str(document["chunk_id"]) for document in documents),
        {str(document["chunk_id"]): document for document in documents},
    )
    retriever = HybridPolicyRetriever(
        documents,
        mode="hybrid",
        dense_index=cast(DenseFaissIndex, dense),
        reranker=cast(CrossEncoderReranker, _DeterministicReranker()),
        dataset_version="hybrid-scope-boundary-v1",
        index_version="hybrid-scope-boundary-v1",
        expected_chunk_ids=dense.chunk_ids,
        expected_dimension=dense.dimension,
    )

    with pytest.raises(DenseUnavailableError, match="out-of-scope"):
        retriever.retrieve(
            PolicyRetrievalRequest(
                query="转专业资格有什么规定？",
                cohort=2024,
                college_ids=("allowed-college",),
                as_of="2026-01-01",
            )
        )


def test_factory_verifies_complete_hybrid_artifact_contract(
    canonical_runtime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = canonical_runtime.repository
    documents = repository.retrieval_documents()
    chunk_ids = tuple(str(document["chunk_id"]) for document in documents)
    dataset_version = repository.metadata()["dataset_version"]
    artifact_root = tmp_path / "artifacts" / "retrieval"
    directory = artifact_root / dataset_version
    directory.mkdir(parents=True)
    documents_path = directory / "documents.jsonl"
    documents_path.write_text(
        "".join(
            json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n"
            for document in documents
        ),
        encoding="utf-8",
    )
    doc_ids_path = directory / "doc_ids.json"
    doc_ids_path.write_text(json.dumps(chunk_ids) + "\n", encoding="utf-8")
    vectors_path = directory / "vectors.npy"
    np.save(
        vectors_path,
        np.tile(np.asarray([[1.0, 0.0]], dtype="float32"), (len(chunk_ids), 1)),
    )
    index_path = directory / "faiss.index"
    index_path.write_bytes(b"offline-faiss-contract")
    manifest = {
        "dataset_version": dataset_version,
        "source_hash": repository.metadata()["evidence_state_sha256"],
        "retrieval_mode": "hybrid",
        "embedding_model": "local-test-encoder",
        "embedding_dimension": 2,
        "reranker_model": "local-test-reranker",
        "chunk_count": len(documents),
        "documents_sha256": _sha(documents_path),
        "doc_ids_sha256": _sha(doc_ids_path),
        "vectors_sha256": _sha(vectors_path),
        "index_sha256": _sha(index_path),
    }
    (directory / "retrieval_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    dataset_manifest_dir = artifact_root.parent / "manifests"
    dataset_manifest_dir.mkdir()
    (dataset_manifest_dir / f"{dataset_version}.json").write_text(
        json.dumps({"dataset_version": dataset_version, "retrieval_mode": "hybrid"}),
        encoding="utf-8",
    )

    dense = _DeterministicDenseIndex(chunk_ids)
    reranker = _DeterministicReranker()
    monkeypatch.setenv("SWUFE_RETRIEVAL_MODE", "hybrid")
    monkeypatch.setenv("SWUFE_RETRIEVAL_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setattr(
        DenseFaissIndex,
        "load",
        classmethod(lambda cls, directory, *, model_name, dimension: cast(DenseFaissIndex, dense)),
    )
    monkeypatch.setattr(
        "agent.factory.CrossEncoderReranker",
        lambda model_name: cast(CrossEncoderReranker, reranker),
    )

    retriever, mode, reasons = _build_retriever(repository)
    assert mode == "hybrid"
    assert reasons == ()
    assert retriever.readiness() == (True, ())

    vectors_path.write_bytes(b"tampered")
    _retriever, _mode, reasons = _build_retriever(repository)
    assert "retrieval_vectors.npy_hash_mismatch" in reasons
