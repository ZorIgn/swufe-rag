"""Explicit production composition root; no import-time patching."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import SupportsIndex, SupportsInt

from academic.database import AcademicRepository
from academic.tools import AcademicTools
from agent.coverage_gate import CoverageGate
from agent.orchestrator import AgentRuntime, RuntimeDependencies
from agent.otel import OpenTelemetryTracer
from agent.policies import RuntimePolicy
from agent.repair import RepairPlanner
from agent.session import InMemoryTTLSessionStore
from agent.tools import PlanExecutor, standard_registry
from generation.synthesizer import DeterministicSynthesizer, StructuredModel
from generation.validator import ClaimValidator
from query.context import RequestContext
from query.normalization import normalize
from query.planner import build_plan
from query.schemas import NormalizedQuery, UnderstandingDraft
from query.understanding import QuestionUnderstanding
from retrieval.dense import DenseFaissIndex, DenseUnavailableError
from retrieval.hybrid import HybridPolicyRetriever
from retrieval.index import load_manifest
from retrieval.reranker import CrossEncoderReranker


def _int_or_zero(value: object) -> int:
    """Mirror ``int(value or 0)`` after narrowing artifact manifest values."""

    candidate = value or 0
    if isinstance(candidate, (str, bytes, bytearray, int, float)):
        return int(candidate)
    if isinstance(candidate, SupportsInt):
        return int(candidate)
    if isinstance(candidate, SupportsIndex):
        return int(candidate)
    raise TypeError(f"retrieval manifest integer is not convertible: {candidate!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_retriever(
    repository: AcademicRepository,
    *,
    metric_sink: Callable[..., None] | None = None,
) -> tuple[HybridPolicyRetriever, str, tuple[str, ...]]:
    mode = os.getenv("SWUFE_RETRIEVAL_MODE", "hybrid").strip().lower() or "hybrid"
    if mode not in {"lexical", "hybrid"}:
        raise ValueError("SWUFE_RETRIEVAL_MODE must be lexical or hybrid")
    dataset_version = repository.metadata().get("dataset_version", "unknown")
    documents = repository.retrieval_documents()
    if mode == "lexical":
        retriever = HybridPolicyRetriever(
            documents,
            mode="lexical",
            dataset_version=dataset_version,
            index_version=dataset_version,
            metric_sink=metric_sink,
        )
        return retriever, mode, ()

    artifact_root = Path(os.getenv("SWUFE_RETRIEVAL_ARTIFACT_ROOT", "artifacts/retrieval"))
    dense = None
    reranker = None
    manifest_reasons: list[str] = []
    manifest: dict[str, object] = {}
    try:
        directory, manifest = load_manifest(artifact_root, dataset_version)
        dataset_manifest_path = artifact_root.parent / "manifests" / f"{dataset_version}.json"
        if not dataset_manifest_path.is_file():
            manifest_reasons.append("dataset_manifest_missing")
        else:
            try:
                dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
                if str(dataset_manifest.get("dataset_version") or "") != dataset_version:
                    manifest_reasons.append("dataset_manifest_version_mismatch")
                if str(dataset_manifest.get("retrieval_mode") or "") != "hybrid":
                    manifest_reasons.append("dataset_manifest_retrieval_mode_mismatch")
            except (OSError, json.JSONDecodeError):
                manifest_reasons.append("dataset_manifest_invalid")
        manifest_version = str(manifest.get("dataset_version") or "")
        expected_source_hash = repository.metadata().get(
            "evidence_state_sha256", repository.metadata().get("chunks_sha256", "")
        )
        if expected_source_hash and str(manifest.get("source_hash") or "") != expected_source_hash:
            manifest_reasons.append("retrieval_source_hash_mismatch")
        if manifest_version != dataset_version:
            manifest_reasons.append("retrieval_dataset_version_mismatch")
        artifact_hashes = {
            "documents.jsonl": "documents_sha256",
            "doc_ids.json": "doc_ids_sha256",
            "vectors.npy": "vectors_sha256",
            "faiss.index": "index_sha256",
        }
        for filename, manifest_key in artifact_hashes.items():
            path = directory / filename
            expected_hash = str(manifest.get(manifest_key) or "")
            if not path.is_file() or not expected_hash:
                manifest_reasons.append(f"retrieval_{filename}_missing")
            elif _sha256(path) != expected_hash:
                manifest_reasons.append(f"retrieval_{filename}_hash_mismatch")
        if str(manifest.get("retrieval_mode") or "") != "hybrid":
            manifest_reasons.append("retrieval_manifest_mode_mismatch")

        doc_ids_path = directory / "doc_ids.json"
        expected_ids = (
            tuple(json.loads(doc_ids_path.read_text(encoding="utf-8")))
            if doc_ids_path.is_file()
            else ()
        )
        if not expected_ids:
            manifest_reasons.append("retrieval_doc_ids_missing")
        if _int_or_zero(manifest.get("chunk_count")) != len(documents):
            manifest_reasons.append("retrieval_chunk_count_mismatch")
        model_name = str(manifest.get("embedding_model") or "")
        dimension = _int_or_zero(manifest.get("embedding_dimension"))
        reranker_name = str(manifest.get("reranker_model") or "")
        if not model_name or dimension <= 0 or not reranker_name:
            manifest_reasons.append("retrieval_manifest_model_missing")
        else:
            dense = DenseFaissIndex.load(directory, model_name=model_name, dimension=dimension)
            reranker = CrossEncoderReranker(reranker_name)
    except (
        FileNotFoundError,
        ValueError,
        OSError,
        DenseUnavailableError,
        json.JSONDecodeError,
    ) as exc:
        manifest_reasons.append(f"retrieval_artifact_unavailable:{type(exc).__name__}")
        expected_ids = ()

    retriever = HybridPolicyRetriever(
        documents,
        mode="hybrid",
        dense_index=dense,
        reranker=reranker,
        dataset_version=dataset_version,
        index_version=str(manifest.get("dataset_version") or "") or None,
        expected_chunk_ids=expected_ids,
        expected_dimension=(_int_or_zero(manifest.get("embedding_dimension")) or None),
        reranker_min_score=float(os.getenv("SWUFE_RERANKER_MIN_SCORE", "0.5")),
        metric_sink=metric_sink,
    )
    return retriever, mode, tuple(manifest_reasons)


def build_runtime(
    database_path: str | Path = "data/academic.sqlite3",
    *,
    model: StructuredModel | None = None,
    policy: RuntimePolicy | None = None,
) -> AgentRuntime:
    runtime_policy = policy or RuntimePolicy()
    repository = AcademicRepository(database_path)
    tracer = OpenTelemetryTracer()
    policy_retriever, retrieval_mode, artifact_reasons = _build_retriever(
        repository, metric_sink=tracer.increment
    )
    academic = AcademicTools(repository=repository, policy_retriever=policy_retriever)
    registry = standard_registry(academic, runtime_policy)

    def normalizer(
        draft: UnderstandingDraft,
        question: str,
        *,
        context: RequestContext,
        inherited_program_id: str | None = None,
        inherited_cohort: int | None = None,
    ) -> NormalizedQuery:
        return normalize(
            draft,
            question,
            repository,
            context=context,
            inherited_program_id=inherited_program_id,
            inherited_cohort=inherited_cohort,
        )

    def readiness() -> tuple[bool, tuple[str, ...]]:
        reasons: list[str] = list(artifact_reasons)
        if not repository.path.is_file():
            reasons.append("database_missing")
        metadata = repository.metadata()
        if not metadata.get("dataset_version"):
            reasons.append("dataset_manifest_missing")
        evidence_ready, evidence_reasons = repository.evidence_readiness()
        reasons.extend(evidence_reasons)
        retriever_ready, retriever_reasons = policy_retriever.readiness()
        reasons.extend(retriever_reasons)
        return not reasons and evidence_ready and retriever_ready, tuple(dict.fromkeys(reasons))

    return AgentRuntime(
        RuntimeDependencies(
            understanding=QuestionUnderstanding(model),
            normalizer=normalizer,
            planner=build_plan,
            executor=PlanExecutor(registry, runtime_policy),
            synthesizer=DeterministicSynthesizer(),
            validator=ClaimValidator(),
            coverage_gate=CoverageGate(),
            repair_planner=RepairPlanner(),
            sessions=InMemoryTTLSessionStore(
                dataset_version=repository.metadata().get("dataset_version", "unknown")
            ),
            tracer=tracer,
            repository=repository,
            retrieval_mode=retrieval_mode,
            readiness=readiness,
            max_validation_retries=1,
        )
    )


__all__ = ["build_runtime"]
