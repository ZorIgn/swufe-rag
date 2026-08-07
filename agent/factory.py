"""Explicit production composition root; no import-time patching."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

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


def _build_retriever(
    repository: AcademicRepository,
    *,
    metric_sink: Callable[..., None] | None = None,
) -> tuple[HybridPolicyRetriever, str, tuple[str, ...]]:
    mode = os.getenv("SWUFE_RETRIEVAL_MODE", "lexical").strip().lower() or "lexical"
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
            except (OSError, json.JSONDecodeError):
                manifest_reasons.append("dataset_manifest_invalid")
        manifest_version = str(manifest.get("dataset_version") or "")
        expected_source_hash = repository.metadata().get("chunks_sha256", "")
        if expected_source_hash and str(manifest.get("source_hash") or "") != expected_source_hash:
            manifest_reasons.append("retrieval_source_hash_mismatch")
        if manifest_version != dataset_version:
            manifest_reasons.append("retrieval_dataset_version_mismatch")
        doc_ids_path = directory / "doc_ids.json"
        expected_ids = (
            tuple(json.loads(doc_ids_path.read_text(encoding="utf-8")))
            if doc_ids_path.is_file()
            else ()
        )
        if not expected_ids:
            manifest_reasons.append("retrieval_doc_ids_missing")
        index_file = directory / (
            "faiss.index" if (directory / "faiss.index").is_file() else "documents.jsonl"
        )
        expected_index_hash = str(manifest.get("index_sha256") or "")
        if not index_file.is_file() or not expected_index_hash:
            manifest_reasons.append("retrieval_index_missing")
        else:
            digest = hashlib.sha256(index_file.read_bytes()).hexdigest()
            if digest != expected_index_hash:
                manifest_reasons.append("retrieval_index_hash_mismatch")
        if int(manifest.get("chunk_count") or 0) != len(documents):
            manifest_reasons.append("retrieval_chunk_count_mismatch")
        model_name = str(manifest.get("embedding_model") or "")
        dimension = int(manifest.get("embedding_dimension") or 0)
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
        expected_dimension=(int(manifest.get("embedding_dimension") or 0) or None),
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
        retriever_ready, retriever_reasons = policy_retriever.readiness()
        reasons.extend(retriever_reasons)
        return not reasons and retriever_ready, tuple(dict.fromkeys(reasons))

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
