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
from agent.interfaces import SessionStore
from agent.orchestrator import AgentRuntime, RuntimeDependencies
from agent.otel import OpenTelemetryTracer
from agent.policies import RuntimePolicy
from agent.repair import RepairPlanner
from agent.session import InMemoryTTLSessionStore, RedisSessionStore
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
from retrieval.index import RetrievalArtifactError, load_manifest
from retrieval.reranker import CrossEncoderReranker
from storage.attestation import TrustedAttestationKey, attestation_key_id
from storage.release import (
    ReleaseError,
    ReleaseRuntimeBundle,
    load_active_release,
    sha256_directory,
)


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


def _strict_env_int(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _strict_env_float(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _build_session_store(dataset_version: str) -> SessionStore:
    """Compose the explicitly selected session backend without fallback."""

    backend = os.getenv("SWUFE_SESSION_BACKEND", "memory").strip().lower() or "memory"
    ttl_seconds = _strict_env_int(
        "SWUFE_SESSION_TTL_SECONDS", 1800, minimum=60, maximum=7 * 24 * 60 * 60
    )
    max_messages = _strict_env_int(
        "SWUFE_SESSION_MAX_MESSAGES", 12, minimum=1, maximum=100
    )
    max_payload_bytes = _strict_env_int(
        "SWUFE_SESSION_MAX_BYTES", 16 * 1024, minimum=1024, maximum=1024 * 1024
    )
    if backend == "memory":
        return InMemoryTTLSessionStore(
            ttl_seconds=float(ttl_seconds),
            max_messages=max_messages,
            max_sessions=_strict_env_int(
                "SWUFE_SESSION_MAX_COUNT", 10_000, minimum=1, maximum=1_000_000
            ),
            max_payload_bytes=max_payload_bytes,
            dataset_version=dataset_version,
        )
    if backend != "redis":
        raise ValueError("SWUFE_SESSION_BACKEND must be memory or redis")
    redis_url = os.getenv("SWUFE_REDIS_URL", "").strip()
    if not redis_url:
        raise ValueError("SWUFE_REDIS_URL is required when SWUFE_SESSION_BACKEND=redis")
    return RedisSessionStore(
        redis_url,
        ttl_seconds=ttl_seconds,
        dataset_version=dataset_version,
        key_namespace=(
            os.getenv("SWUFE_REDIS_SESSION_NAMESPACE", "swufe-rag:sessions").strip()
            or "swufe-rag:sessions"
        ),
        max_messages=max_messages,
        max_payload_bytes=max_payload_bytes,
        socket_timeout_seconds=_strict_env_float(
            "SWUFE_REDIS_TIMEOUT_SECONDS", 1.0, minimum=0.1, maximum=30.0
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _release_relative_path(release_directory: Path, value: object, *, field: str) -> Path:
    """Resolve a manifest path only when it stays inside an immutable release."""

    relative = Path(str(value or "").strip())
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ReleaseError(f"active release has an unsafe {field} path")
    candidate = release_directory
    for part in relative.parts:
        candidate /= part
        if candidate.is_symlink():
            raise ReleaseError(
                f"active release {field} path cannot contain a symlink"
            )
    try:
        candidate.resolve().relative_to(release_directory.resolve())
    except ValueError as exc:
        raise ReleaseError(f"active release {field} escapes the release directory") from exc
    return candidate


def _resolve_release_paths(
    database_path: str | Path,
) -> tuple[Path, Path | None, Path | None, str | None]:
    """Prefer a verified active release for the default production database.

    Passing an explicit database path remains useful for tests and local
    inspection.  In contrast, a configured ``SWUFE_RELEASE_ROOT`` is an
    operator assertion and must fail closed if its pointer or hashes are bad.
    """

    requested_database = Path(database_path)
    deployment_mode = os.getenv("SWUFE_DEPLOYMENT_MODE", "local").strip().lower()
    if deployment_mode not in {"local", "production"}:
        raise ReleaseError("SWUFE_DEPLOYMENT_MODE must be local or production")
    configured_root = os.getenv("SWUFE_RELEASE_ROOT", "").strip()
    implicit_root = Path("artifacts/releases")
    if configured_root:
        release_root: Path | None = Path(configured_root)
    elif requested_database == Path("data/academic.sqlite3") and (implicit_root / "active.json").is_file():
        release_root = implicit_root
    else:
        release_root = None
    if release_root is None:
        return requested_database, None, None, None

    allow_unattested = os.getenv(
        "SWUFE_ALLOW_UNATTESTED_ACTIVE", ""
    ).strip().lower() in {"1", "true", "yes"}
    if deployment_mode == "production" and allow_unattested:
        raise ReleaseError(
            "production deployment cannot allow an unattested active release"
        )
    public_key = os.getenv("SWUFE_RELEASE_ATTESTATION_PUBLIC_KEY", "").strip()
    issuer = os.getenv("SWUFE_RELEASE_ATTESTATION_ISSUER", "").strip()
    trusted_keys = None
    if public_key or issuer:
        if not public_key or not issuer:
            raise ReleaseError(
                "both SWUFE_RELEASE_ATTESTATION_PUBLIC_KEY and issuer are required"
            )
        key_id = attestation_key_id(public_key)
        trusted_keys = {
            key_id: TrustedAttestationKey(
                issuer=issuer,
                public_key=public_key,
            )
        }
    release_directory, release_manifest = load_active_release(
        release_root,
        require_attestation=not allow_unattested,
        trusted_attestation_keys=trusted_keys,
    )
    payload = release_manifest.get("payload")
    if not isinstance(payload, dict):
        raise ReleaseError("active release lacks a payload")
    database = payload.get("database")
    retrieval = payload.get("retrieval")
    if not isinstance(database, dict) or not isinstance(retrieval, dict):
        raise ReleaseError("active release lacks database/retrieval metadata")
    resolved_database = _release_relative_path(
        release_directory, database.get("path"), field="database"
    )
    if not resolved_database.is_file():
        raise ReleaseError("active release database is missing")
    if _sha256(resolved_database) != str(database.get("sha256") or ""):
        raise ReleaseError("active release database hash does not match manifest")
    artifact_root = _release_relative_path(release_directory, retrieval.get("root"), field="retrieval")
    dataset_manifest = _release_relative_path(
        release_directory, payload.get("dataset_manifest"), field="dataset_manifest"
    )
    if not artifact_root.is_dir() or not dataset_manifest.is_file():
        raise ReleaseError("active release retrieval artifact or dataset manifest is missing")
    return resolved_database, artifact_root, dataset_manifest, str(release_manifest["release_id"])


def _build_retriever(
    repository: AcademicRepository,
    *,
    metric_sink: Callable[..., None] | None = None,
    artifact_root_override: Path | None = None,
    dataset_manifest_override: Path | None = None,
    forced_mode: str | None = None,
    reranker_min_score_override: float | None = None,
) -> tuple[HybridPolicyRetriever, str, tuple[str, ...]]:
    mode = forced_mode or (os.getenv("SWUFE_RETRIEVAL_MODE", "hybrid").strip().lower() or "hybrid")
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

    artifact_root = artifact_root_override or Path(
        os.getenv("SWUFE_RETRIEVAL_ARTIFACT_ROOT", "artifacts/retrieval")
    )
    dense = None
    reranker = None
    manifest_reasons: list[str] = []
    manifest: dict[str, object] = {}
    try:
        directory, manifest = load_manifest(artifact_root, dataset_version)
        dataset_manifest_path = dataset_manifest_override or (
            artifact_root.parent / "manifests" / f"{dataset_version}.json"
        )
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
            release_bound = artifact_root_override is not None
            embedding_snapshot = os.getenv("SWUFE_EMBEDDING_MODEL", "").strip() or model_name
            reranker_snapshot = os.getenv("SWUFE_RERANKER_MODEL", "").strip() or reranker_name
            model_snapshots = (
                (
                    "embedding",
                    embedding_snapshot,
                    manifest.get("embedding_model_sha256"),
                ),
                (
                    "reranker",
                    reranker_snapshot,
                    manifest.get("reranker_model_sha256"),
                ),
            )
            model_snapshot_failed = False
            for label, snapshot_name, expected_digest in model_snapshots:
                if expected_digest is None:
                    if release_bound:
                        manifest_reasons.append(
                            f"retrieval_{label}_model_digest_missing"
                        )
                        model_snapshot_failed = True
                    continue
                snapshot = Path(snapshot_name)
                if not snapshot.is_dir() or snapshot.is_symlink():
                    manifest_reasons.append(
                        f"retrieval_{label}_model_snapshot_unavailable"
                    )
                    model_snapshot_failed = True
                    continue
                try:
                    observed_digest = sha256_directory(snapshot)
                except (OSError, ReleaseError):
                    manifest_reasons.append(
                        f"retrieval_{label}_model_snapshot_invalid"
                    )
                    model_snapshot_failed = True
                    continue
                if observed_digest != expected_digest:
                    manifest_reasons.append(
                        f"retrieval_{label}_model_hash_mismatch"
                    )
                    model_snapshot_failed = True
            if not model_snapshot_failed:
                dense = DenseFaissIndex.load(
                    directory,
                    model_name=embedding_snapshot,
                    dimension=dimension,
                )
                reranker = CrossEncoderReranker(reranker_snapshot)
    except (
        FileNotFoundError,
        ValueError,
        OSError,
        DenseUnavailableError,
        RetrievalArtifactError,
        json.JSONDecodeError,
    ) as exc:
        manifest_reasons.append(f"retrieval_artifact_unavailable:{type(exc).__name__}")
        message = str(exc)
        prefix = "retrieval artifact hash mismatch: "
        if message.startswith(prefix):
            manifest_reasons.append("retrieval_" + message.removeprefix(prefix) + "_hash_mismatch")
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
        reranker_min_score=(
            reranker_min_score_override
            if reranker_min_score_override is not None
            else float(os.getenv("SWUFE_RERANKER_MIN_SCORE", "0.5"))
        ),
        metric_sink=metric_sink,
    )
    return retriever, mode, tuple(manifest_reasons)


def build_runtime(
    database_path: str | Path = "data/academic.sqlite3",
    *,
    model: StructuredModel | None = None,
    policy: RuntimePolicy | None = None,
    release_bundle: ReleaseRuntimeBundle | None = None,
    session_store: SessionStore | None = None,
) -> AgentRuntime:
    runtime_policy = policy or RuntimePolicy()
    if release_bundle is None:
        (
            resolved_database,
            release_artifact_root,
            release_dataset_manifest,
            _release_id,
        ) = _resolve_release_paths(database_path)
        forced_mode = None
        reranker_min_score = None
    else:
        resolved_database = release_bundle.database_path
        release_artifact_root = release_bundle.retrieval_root
        release_dataset_manifest = release_bundle.dataset_manifest_path
        forced_mode = release_bundle.retrieval_mode
        reranker_min_score = 0.5
    repository = AcademicRepository(resolved_database)
    tracer = OpenTelemetryTracer()
    policy_retriever, retrieval_mode, artifact_reasons = _build_retriever(
        repository,
        metric_sink=tracer.increment,
        artifact_root_override=release_artifact_root,
        dataset_manifest_override=release_dataset_manifest,
        forced_mode=forced_mode,
        reranker_min_score_override=reranker_min_score,
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
            sessions=(
                session_store
                or _build_session_store(
                    repository.metadata().get("dataset_version", "unknown")
                )
            ),
            tracer=tracer,
            repository=repository,
            retrieval_mode=retrieval_mode,
            readiness=readiness,
            max_validation_retries=1,
        )
    )


__all__ = ["build_runtime"]
