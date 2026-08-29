from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

import agent.factory as factory_module
import eval.candidate_release as candidate_module
from agent.factory import build_runtime
from agent.orchestrator import AgentRuntime
from eval.candidate_release import (
    CandidateEvaluationError,
    load_candidate_evaluation_context,
)
from eval.holdout import load_holdout_manifest
from storage.json_contract import canonical_json
from storage.release import (
    build_release_manifest,
    make_staging_directory,
    publish_release,
    sha256_directory,
    sha256_file,
)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(dict(value)))


def _write_restricted_holdout(
    root: Path,
    *,
    dataset_version: str,
    documents: bytes,
) -> tuple[Path, dict[str, object]]:
    root.mkdir(parents=True, exist_ok=True)
    agent_cases = root / "agent_cases.json"
    retrieval_documents = root / "retrieval_documents.jsonl"
    retrieval_queries = root / "retrieval_queries.json"
    labels = root / "labels.json"
    agent_cases.write_bytes(b'[{"id":"agent-1"}]')
    retrieval_documents.write_bytes(documents)
    retrieval_queries.write_bytes(b'[{"id":"query-1"}]')
    labels.write_bytes(b'{"owner":"restricted-eval"}')

    inputs = {
        "agent_cases": {
            "path": agent_cases.name,
            "sha256": sha256_file(agent_cases),
            "count": 1,
        },
        "retrieval_documents": {
            "path": retrieval_documents.name,
            "sha256": sha256_file(retrieval_documents),
            "count": 2,
        },
        "retrieval_queries": {
            "path": retrieval_queries.name,
            "sha256": sha256_file(retrieval_queries),
            "count": 1,
        },
    }
    additional_files = {labels.name: sha256_file(labels)}
    bundle_sha256 = hashlib.sha256(
        canonical_json(
            {
                "inputs": inputs,
                "additional_files": additional_files,
            }
        )
    ).hexdigest()
    manifest_path = root / "manifest.json"
    _write_json(
        manifest_path,
        {
            "holdout_contract_version": "2",
            "kind": "restricted_holdout",
            "holdout_id": "restricted-candidate-eval-v1",
            "dataset_version": dataset_version,
            "access": "restricted",
            "bundle_sha256": bundle_sha256,
            "inputs": inputs,
            "additional_files": additional_files,
        },
    )
    manifest_path.with_suffix(".json.sha256").write_text(
        f"{sha256_file(manifest_path)}  manifest.json\n",
        encoding="ascii",
    )
    holdout = load_holdout_manifest(manifest_path)
    return manifest_path, holdout.release_lock()


@dataclass(frozen=True)
class _CandidateFixture:
    candidate_manifest: Path
    holdout_manifest: Path
    embedding_model_file: Path
    release_id: str


def _build_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    database_path: Path,
    metadata: Mapping[str, object],
    holdout_documents: bytes,
    candidate_documents: bytes | None = None,
    evaluator_dirty: bool = False,
    retrieval_is_fixture: bool = False,
) -> _CandidateFixture:
    dataset_version = metadata["dataset_version"]
    evidence_state_sha256 = metadata["evidence_state_sha256"]
    assert isinstance(dataset_version, str)
    assert isinstance(evidence_state_sha256, str)
    candidate_documents = candidate_documents or holdout_documents
    holdout_path, holdout_lock = _write_restricted_holdout(
        tmp_path / "restricted-holdout",
        dataset_version=dataset_version,
        documents=holdout_documents,
    )

    embedding_model = tmp_path / "models" / "embedding"
    reranker_model = tmp_path / "models" / "reranker"
    embedding_model.mkdir(parents=True)
    reranker_model.mkdir(parents=True)
    embedding_model_file = embedding_model / "model.bin"
    embedding_model_file.write_bytes(b"embedding snapshot")
    (reranker_model / "model.bin").write_bytes(b"reranker snapshot")
    embedding_model_sha256 = sha256_directory(embedding_model)
    reranker_model_sha256 = sha256_directory(reranker_model)

    releases_root = tmp_path / "releases"
    staging = make_staging_directory(releases_root)
    staged_database = staging / "academic.sqlite3"
    shutil.copy2(database_path, staged_database)
    retrieval_directory = staging / "retrieval" / dataset_version
    retrieval_directory.mkdir(parents=True)
    documents_path = retrieval_directory / "documents.jsonl"
    doc_ids_path = retrieval_directory / "doc_ids.json"
    vectors_path = retrieval_directory / "vectors.npy"
    index_path = retrieval_directory / "faiss.index"
    documents_path.write_bytes(candidate_documents)
    doc_ids_path.write_bytes(b'["doc-1","doc-2"]')
    vectors_path.write_bytes(b"candidate vectors")
    index_path.write_bytes(b"candidate index")
    retrieval_manifest: dict[str, object] = {
        "dataset_version": dataset_version,
        "retrieval_mode": "hybrid",
        "source_hash": evidence_state_sha256,
        "test_fixture": retrieval_is_fixture,
        "embedding_model": str(embedding_model),
        "embedding_model_sha256": embedding_model_sha256,
        "reranker_model": str(reranker_model),
        "reranker_model_sha256": reranker_model_sha256,
        "documents_sha256": sha256_file(documents_path),
        "doc_ids_sha256": sha256_file(doc_ids_path),
        "vectors_sha256": sha256_file(vectors_path),
        "index_sha256": sha256_file(index_path),
    }
    retrieval_manifest_path = retrieval_directory / "retrieval_manifest.json"
    _write_json(retrieval_manifest_path, retrieval_manifest)
    dataset_manifest_path = staging / "dataset_manifest.json"
    _write_json(
        dataset_manifest_path,
        {
            "dataset_version": dataset_version,
            "retrieval_mode": "hybrid",
            "holdout": holdout_lock,
        },
    )

    commit = "f" * 40
    clean_git = {
        "available": True,
        "commit": commit,
        "dirty": False,
        "diff_sha256": None,
    }
    identity = {
        "dataset_version": dataset_version,
        "schema_version": str(metadata.get("schema_version") or "1"),
        "database_sha256": sha256_file(staged_database),
        "evidence_state_sha256": evidence_state_sha256,
        "retrieval_manifest_sha256": sha256_file(retrieval_manifest_path),
        "retrieval_mode": "hybrid",
        "embedding_model": str(embedding_model),
        "embedding_model_sha256": embedding_model_sha256,
        "reranker_model": str(reranker_model),
        "reranker_model_sha256": reranker_model_sha256,
        "holdout": holdout_lock,
        "release_tier": "candidate",
        "git_commit": commit,
        "git_provenance": clean_git,
    }
    manifest = build_release_manifest(
        identity=identity,
        payload={
            "database": {
                "path": staged_database.name,
                "sha256": sha256_file(staged_database),
            },
            "dataset_manifest": dataset_manifest_path.name,
            "retrieval": {
                "dataset_version": dataset_version,
                "root": "retrieval",
                "manifest": (
                    f"retrieval/{dataset_version}/retrieval_manifest.json"
                ),
            },
        },
        staging_directory=staging,
    )
    published = publish_release(
        staging,
        releases_root=releases_root,
        manifest=manifest,
        activate=False,
    )

    monkeypatch.setattr(
        candidate_module,
        "validate_retrieval_artifact",
        lambda _directory: dict(retrieval_manifest),
    )
    evaluator_git = {
        **clean_git,
        "dirty": evaluator_dirty,
        "diff_sha256": "0" * 64 if evaluator_dirty else None,
    }
    monkeypatch.setattr(
        candidate_module,
        "git_provenance",
        lambda: dict(evaluator_git),
    )
    return _CandidateFixture(
        candidate_manifest=published.manifest_path,
        holdout_manifest=holdout_path,
        embedding_model_file=embedding_model_file,
        release_id=published.release_id,
    )


def _documents() -> bytes:
    return (
        b'{"chunk_id":"doc-1","text":"policy one"}\n'
        b'{"chunk_id":"doc-2","text":"policy two"}\n'
    )


def test_candidate_context_binds_release_holdout_models_and_evaluator(
    canonical_runtime: AgentRuntime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = canonical_runtime.repository
    fixture = _build_candidate(
        tmp_path,
        monkeypatch,
        database_path=repository.path,
        metadata=repository.metadata(),
        holdout_documents=_documents(),
    )

    context = load_candidate_evaluation_context(
        fixture.candidate_manifest,
        fixture.holdout_manifest,
    )

    assert context.runtime_bundle.release_id == fixture.release_id
    assert context.release_subject["release_id"] == fixture.release_id
    assert context.holdout.kind == "restricted_holdout"
    assert context.holdout.access == "restricted"
    assert context.runtime_provenance()["documents_sha256"] == sha256_file(
        context.documents_path
    )


def test_candidate_context_rejects_a_different_frozen_retrieval_corpus(
    canonical_runtime: AgentRuntime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = canonical_runtime.repository
    fixture = _build_candidate(
        tmp_path,
        monkeypatch,
        database_path=repository.path,
        metadata=repository.metadata(),
        holdout_documents=_documents(),
        candidate_documents=(
            b'{"chunk_id":"doc-1","text":"different corpus"}\n'
            b'{"chunk_id":"doc-2","text":"different corpus"}\n'
        ),
    )

    with pytest.raises(
        CandidateEvaluationError,
        match="documents differ from frozen evaluation corpus",
    ):
        load_candidate_evaluation_context(
            fixture.candidate_manifest,
            fixture.holdout_manifest,
        )


def test_candidate_context_rejects_model_snapshot_drift(
    canonical_runtime: AgentRuntime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = canonical_runtime.repository
    fixture = _build_candidate(
        tmp_path,
        monkeypatch,
        database_path=repository.path,
        metadata=repository.metadata(),
        holdout_documents=_documents(),
    )
    fixture.embedding_model_file.write_bytes(b"mutated embedding snapshot")

    with pytest.raises(
        CandidateEvaluationError,
        match="model snapshot digest differs",
    ):
        load_candidate_evaluation_context(
            fixture.candidate_manifest,
            fixture.holdout_manifest,
        )


def test_candidate_context_allows_digest_verified_model_path_remapping(
    canonical_runtime: AgentRuntime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = canonical_runtime.repository
    fixture = _build_candidate(
        tmp_path,
        monkeypatch,
        database_path=repository.path,
        metadata=repository.metadata(),
        holdout_documents=_documents(),
    )
    original_models = fixture.embedding_model_file.parents[1]
    relocated_models = tmp_path / "runtime-models"
    shutil.move(str(original_models), relocated_models)
    monkeypatch.setenv("SWUFE_EMBEDDING_MODEL", str(relocated_models / "embedding"))
    monkeypatch.setenv("SWUFE_RERANKER_MODEL", str(relocated_models / "reranker"))

    context = load_candidate_evaluation_context(
        fixture.candidate_manifest,
        fixture.holdout_manifest,
    )

    assert context.release_subject["embedding_model_sha256"] == sha256_directory(
        relocated_models / "embedding"
    )
    assert context.release_subject["reranker_model_sha256"] == sha256_directory(
        relocated_models / "reranker"
    )


def test_release_runtime_rejects_model_snapshot_drift_after_evaluation(
    canonical_runtime: AgentRuntime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = canonical_runtime.repository
    fixture = _build_candidate(
        tmp_path,
        monkeypatch,
        database_path=repository.path,
        metadata=repository.metadata(),
        holdout_documents=_documents(),
    )
    context = load_candidate_evaluation_context(
        fixture.candidate_manifest,
        fixture.holdout_manifest,
    )
    retrieval_manifest = json.loads(
        context.retrieval_manifest_path.read_text(encoding="utf-8")
    )
    retrieval_manifest.update(
        {
            "chunk_count": len(repository.retrieval_documents()),
            "embedding_dimension": 2,
        }
    )
    monkeypatch.setattr(
        factory_module,
        "load_manifest",
        lambda _root, _version: (
            context.retrieval_directory,
            retrieval_manifest,
        ),
    )
    fixture.embedding_model_file.write_bytes(b"mutated after evaluation")

    runtime = build_runtime(release_bundle=context.runtime_bundle)
    try:
        ready, reasons = runtime.readiness()
        assert ready is False
        assert "retrieval_embedding_model_hash_mismatch" in reasons
    finally:
        runtime.repository.close()


def test_candidate_context_rejects_dirty_or_fixture_evaluation(
    canonical_runtime: AgentRuntime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = canonical_runtime.repository
    dirty = _build_candidate(
        tmp_path / "dirty",
        monkeypatch,
        database_path=repository.path,
        metadata=repository.metadata(),
        holdout_documents=_documents(),
        evaluator_dirty=True,
    )
    with pytest.raises(
        CandidateEvaluationError,
        match="evaluator must be clean",
    ):
        load_candidate_evaluation_context(
            dirty.candidate_manifest,
            dirty.holdout_manifest,
        )

    fixture = _build_candidate(
        tmp_path / "fixture",
        monkeypatch,
        database_path=repository.path,
        metadata=repository.metadata(),
        holdout_documents=_documents(),
        retrieval_is_fixture=True,
    )
    with pytest.raises(
        CandidateEvaluationError,
        match="test fixture retrieval artifacts cannot be promoted",
    ):
        load_candidate_evaluation_context(
            fixture.candidate_manifest,
            fixture.holdout_manifest,
        )
