"""Single fail-closed loader for promotion-eligible candidate evaluation."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from academic.database import AcademicRepository
from eval.holdout import (
    HoldoutContractError,
    HoldoutManifest,
    load_holdout_manifest,
    validate_restricted_release_lock,
)
from retrieval.index import RetrievalArtifactError, validate_retrieval_artifact
from storage.attestation import AttestationError, release_evaluation_subject
from storage.json_contract import StrictJSONError, load_strict_json_file
from storage.provenance import git_provenance
from storage.release import (
    RELEASE_MANIFEST_NAME,
    ReleaseError,
    ReleaseRuntimeBundle,
    sha256_directory,
    sha256_file,
    validate_release_directory,
)


class CandidateEvaluationError(RuntimeError):
    """Raised when evaluated bytes cannot be proven to belong to one candidate."""


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CandidateEvaluationError(f"{label} must be an object")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise CandidateEvaluationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateEvaluationError(f"{label} must be a non-empty string")
    return value.strip()


def _release_path(root: Path, value: object, *, label: str) -> Path:
    raw = _text(value, label)
    if "\\" in raw:
        raise CandidateEvaluationError(f"{label} must use POSIX separators")
    relative = PurePosixPath(raw)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise CandidateEvaluationError(f"{label} must be a safe relative path")
    candidate = root
    for part in relative.parts:
        candidate /= part
        if candidate.is_symlink():
            raise CandidateEvaluationError(
                f"{label} cannot contain a symlink component"
            )
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise CandidateEvaluationError(f"{label} escapes the release directory") from exc
    return candidate


def _model_snapshot(
    model: object,
    digest: object,
    *,
    label: str,
    environment_name: str,
) -> tuple[str, str]:
    model_name = _text(model, f"{label} model")
    expected = _digest(digest, f"{label} model digest")
    runtime_name = os.getenv(environment_name, "").strip() or model_name
    path = Path(runtime_name)
    if not path.is_dir() or path.is_symlink():
        raise CandidateEvaluationError(
            f"{label} model must be an available, symlink-free local snapshot"
        )
    observed = sha256_directory(path)
    if observed != expected:
        raise CandidateEvaluationError(f"{label} model snapshot digest differs from candidate")
    return model_name, expected


@dataclass(frozen=True)
class CandidateEvaluationContext:
    release_directory: Path
    release_manifest_path: Path
    release_manifest: dict[str, object]
    release_manifest_sha256: str
    release_subject: dict[str, object]
    database_path: Path
    dataset_manifest_path: Path
    retrieval_root: Path
    retrieval_directory: Path
    retrieval_manifest_path: Path
    retrieval_manifest: dict[str, object]
    documents_path: Path
    holdout: HoldoutManifest
    evaluator_git: dict[str, object]
    runtime_bundle: ReleaseRuntimeBundle

    def runtime_provenance(self) -> dict[str, object]:
        return {
            "release_id": self.runtime_bundle.release_id,
            "database_sha256": sha256_file(self.database_path),
            "dataset_version": self.holdout.dataset_version,
            "retrieval_mode": self.runtime_bundle.retrieval_mode,
            "dataset_manifest_sha256": sha256_file(self.dataset_manifest_path),
            "retrieval_manifest_sha256": sha256_file(self.retrieval_manifest_path),
            "documents_sha256": sha256_file(self.documents_path),
            "doc_ids_sha256": self.retrieval_manifest.get("doc_ids_sha256"),
            "vectors_sha256": self.retrieval_manifest.get("vectors_sha256"),
            "index_sha256": self.retrieval_manifest.get("index_sha256"),
            "embedding_model_sha256": self.retrieval_manifest.get(
                "embedding_model_sha256"
            ),
            "reranker_model_sha256": self.retrieval_manifest.get(
                "reranker_model_sha256"
            ),
        }


def load_candidate_evaluation_context(
    candidate_manifest_path: str | Path,
    holdout_manifest_path: str | Path,
) -> CandidateEvaluationContext:
    """Validate candidate, runtime files, restricted holdout, models, and evaluator."""

    original_manifest_path = Path(candidate_manifest_path)
    if (
        original_manifest_path.name != RELEASE_MANIFEST_NAME
        or not original_manifest_path.is_file()
        or original_manifest_path.is_symlink()
    ):
        raise CandidateEvaluationError(
            f"candidate manifest must be a regular {RELEASE_MANIFEST_NAME} file"
        )
    manifest_path = original_manifest_path.resolve()
    release_directory = manifest_path.parent
    if release_directory.is_symlink() or release_directory.parent.is_symlink():
        raise CandidateEvaluationError("candidate release path cannot contain a symlink root")
    try:
        manifest = validate_release_directory(release_directory)
    except ReleaseError as exc:
        raise CandidateEvaluationError(str(exc)) from exc
    release_id = _text(manifest.get("release_id"), "candidate release_id")
    if release_directory.name != release_id:
        raise CandidateEvaluationError("candidate directory name differs from release_id")
    manifest_sha256 = sha256_file(manifest_path)

    identity = _mapping(manifest.get("identity"), "candidate identity")
    payload = _mapping(manifest.get("payload"), "candidate payload")
    dataset_version = _text(identity.get("dataset_version"), "candidate dataset_version")
    if identity.get("release_tier") != "candidate":
        raise CandidateEvaluationError("promotion evaluation requires release_tier=candidate")
    if identity.get("retrieval_mode") != "hybrid":
        raise CandidateEvaluationError("promotion evaluation requires a hybrid candidate")

    database_payload = _mapping(payload.get("database"), "candidate database payload")
    database_path = _release_path(
        release_directory, database_payload.get("path"), label="candidate database path"
    )
    if not database_path.is_file():
        raise CandidateEvaluationError("candidate database is missing")
    database_sha256 = sha256_file(database_path)
    if database_sha256 != _digest(
        database_payload.get("sha256"), "candidate payload database digest"
    ) or database_sha256 != _digest(
        identity.get("database_sha256"), "candidate identity database digest"
    ):
        raise CandidateEvaluationError("candidate database digests are inconsistent")

    retrieval_payload = _mapping(payload.get("retrieval"), "candidate retrieval payload")
    if retrieval_payload.get("dataset_version") != dataset_version:
        raise CandidateEvaluationError("candidate retrieval payload dataset_version differs")
    retrieval_root = _release_path(
        release_directory, retrieval_payload.get("root"), label="candidate retrieval root"
    )
    retrieval_directory = retrieval_root / dataset_version
    expected_manifest_path = _release_path(
        release_directory,
        retrieval_payload.get("manifest"),
        label="candidate retrieval manifest path",
    )
    retrieval_manifest_path = retrieval_directory / "retrieval_manifest.json"
    if expected_manifest_path != retrieval_manifest_path:
        raise CandidateEvaluationError("candidate retrieval manifest path is inconsistent")
    try:
        retrieval_manifest = validate_retrieval_artifact(retrieval_directory)
    except RetrievalArtifactError as exc:
        raise CandidateEvaluationError(str(exc)) from exc
    if sha256_file(retrieval_manifest_path) != _digest(
        identity.get("retrieval_manifest_sha256"),
        "candidate retrieval manifest digest",
    ):
        raise CandidateEvaluationError("candidate retrieval manifest digest differs")
    if (
        retrieval_manifest.get("dataset_version") != dataset_version
        or retrieval_manifest.get("retrieval_mode") != "hybrid"
        or retrieval_manifest.get("source_hash")
        != identity.get("evidence_state_sha256")
    ):
        raise CandidateEvaluationError("candidate retrieval identity is inconsistent")
    if retrieval_manifest.get("test_fixture") is True:
        raise CandidateEvaluationError("test fixture retrieval artifacts cannot be promoted")
    embedding_model, embedding_sha = _model_snapshot(
        identity.get("embedding_model"),
        identity.get("embedding_model_sha256"),
        label="embedding",
        environment_name="SWUFE_EMBEDDING_MODEL",
    )
    reranker_model, reranker_sha = _model_snapshot(
        identity.get("reranker_model"),
        identity.get("reranker_model_sha256"),
        label="reranker",
        environment_name="SWUFE_RERANKER_MODEL",
    )
    if (
        retrieval_manifest.get("embedding_model") != embedding_model
        or retrieval_manifest.get("embedding_model_sha256") != embedding_sha
        or retrieval_manifest.get("reranker_model") != reranker_model
        or retrieval_manifest.get("reranker_model_sha256") != reranker_sha
    ):
        raise CandidateEvaluationError("candidate retrieval model snapshots are inconsistent")

    dataset_manifest_path = _release_path(
        release_directory,
        payload.get("dataset_manifest"),
        label="candidate dataset manifest path",
    )
    try:
        dataset_manifest_value = load_strict_json_file(
            dataset_manifest_path, label="candidate dataset manifest"
        )
    except StrictJSONError as exc:
        raise CandidateEvaluationError(str(exc)) from exc
    dataset_manifest = _mapping(dataset_manifest_value, "candidate dataset manifest")
    if (
        dataset_manifest.get("dataset_version") != dataset_version
        or dataset_manifest.get("retrieval_mode") != "hybrid"
        or dataset_manifest.get("holdout") != identity.get("holdout")
    ):
        raise CandidateEvaluationError("candidate dataset manifest is inconsistent")

    repository = AcademicRepository(database_path)
    try:
        metadata = repository.metadata()
    finally:
        repository.close()
    if (
        metadata.get("dataset_version") != dataset_version
        or metadata.get("evidence_state_sha256")
        != identity.get("evidence_state_sha256")
    ):
        raise CandidateEvaluationError("candidate database metadata differs from identity")

    try:
        holdout = load_holdout_manifest(holdout_manifest_path, verify_files=True)
        release_lock = validate_restricted_release_lock(
            identity.get("holdout"), dataset_version=dataset_version
        )
        if holdout.release_lock() != release_lock:
            raise CandidateEvaluationError("restricted holdout differs from candidate lock")
    except HoldoutContractError as exc:
        raise CandidateEvaluationError(str(exc)) from exc
    documents_path = retrieval_directory / "documents.jsonl"
    if sha256_file(documents_path) != holdout.inputs["retrieval_documents"].sha256:
        raise CandidateEvaluationError(
            "candidate retrieval documents differ from frozen evaluation corpus"
        )

    evaluator_git = git_provenance()
    candidate_git = _mapping(identity.get("git_provenance"), "candidate git provenance")
    if (
        evaluator_git.get("available") is not True
        or evaluator_git.get("dirty") is not False
        or evaluator_git.get("commit") != candidate_git.get("commit")
    ):
        raise CandidateEvaluationError(
            "promotion evaluator must be clean and at the candidate Git commit"
        )
    try:
        subject = release_evaluation_subject(
            manifest,
            manifest_sha256=manifest_sha256,
        )
    except AttestationError as exc:
        raise CandidateEvaluationError(str(exc)) from exc
    bundle = ReleaseRuntimeBundle(
        release_id=release_id,
        database_path=database_path,
        retrieval_root=retrieval_root,
        dataset_manifest_path=dataset_manifest_path,
        retrieval_mode="hybrid",
    )
    return CandidateEvaluationContext(
        release_directory=release_directory,
        release_manifest_path=manifest_path,
        release_manifest=manifest,
        release_manifest_sha256=manifest_sha256,
        release_subject=subject,
        database_path=database_path,
        dataset_manifest_path=dataset_manifest_path,
        retrieval_root=retrieval_root,
        retrieval_directory=retrieval_directory,
        retrieval_manifest_path=retrieval_manifest_path,
        retrieval_manifest=retrieval_manifest,
        documents_path=documents_path,
        holdout=holdout,
        evaluator_git=evaluator_git,
        runtime_bundle=bundle,
    )


__all__ = [
    "CandidateEvaluationContext",
    "CandidateEvaluationError",
    "load_candidate_evaluation_context",
]
