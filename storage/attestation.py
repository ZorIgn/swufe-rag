"""Ed25519 evaluation attestations for promoting one exact immutable release."""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from eval.holdout import HoldoutContractError, validate_restricted_release_lock
from eval.promotion_policy import (
    PROMOTION_POLICY_SHA256,
    PROMOTION_POLICY_VERSION,
    PromotionPolicyError,
    validate_agent_report,
    validate_retrieval_report,
)
from storage.json_contract import canonical_json

ATTESTATION_CONTRACT_VERSION = "2"
ATTESTATION_ALGORITHM = "ed25519"
_RELEASE_ID_RE = re.compile(r"^sha256-[0-9a-f]{64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AttestationError(RuntimeError):
    """Raised when evaluation evidence is unbound, incomplete, or untrusted."""


@dataclass(frozen=True)
class TrustedAttestationKey:
    """One allowlisted verifier identity; runtime never receives signing material."""

    issuer: str
    public_key: str | bytes
    revoked: bool = False


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AttestationError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise AttestationError(f"{label} keys must be strings")
    return value


def _exact_keys(
    value: Mapping[str, object], expected: frozenset[str], label: str
) -> None:
    observed = frozenset(value)
    if observed != expected:
        raise AttestationError(
            f"{label} fields mismatch; missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)}"
        )


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise AttestationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AttestationError(f"{label} must be a non-empty string")
    return value.strip()


def _relative(value: object, label: str) -> str:
    raw = _text(value, label)
    if "\\" in raw:
        raise AttestationError(f"{label} must use POSIX separators")
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise AttestationError(f"{label} must be a safe relative path")
    return path.as_posix()


def _file_digest(files: Mapping[str, object], relative: str, label: str) -> str:
    return _sha256(files.get(relative), label)


def release_evaluation_subject(
    manifest: Mapping[str, object],
    *,
    manifest_sha256: str,
) -> dict[str, object]:
    """Normalize the exact clean, model-locked, restricted candidate subject."""

    release_id = _text(manifest.get("release_id"), "candidate release_id")
    if _RELEASE_ID_RE.fullmatch(release_id) is None:
        raise AttestationError("candidate release id is invalid")
    if manifest.get("release_contract_version") != "1":
        raise AttestationError("candidate release contract is unsupported")
    identity = _mapping(manifest.get("identity"), "release identity")
    payload = _mapping(manifest.get("payload"), "release payload")
    files = _mapping(manifest.get("files"), "release file hashes")
    dataset_version = _text(identity.get("dataset_version"), "candidate dataset_version")
    if identity.get("release_tier") != "candidate":
        raise AttestationError("only an immutable candidate can be evaluated and promoted")
    if identity.get("retrieval_mode") != "hybrid":
        raise AttestationError("production promotion requires a hybrid candidate")

    git = _mapping(identity.get("git_provenance"), "git provenance")
    _exact_keys(
        git,
        frozenset({"available", "commit", "dirty", "diff_sha256"}),
        "git provenance",
    )
    commit = _text(git.get("commit"), "candidate Git commit")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise AttestationError("candidate has no valid Git commit")
    if (
        git.get("available") is not True
        or git.get("dirty") is not False
        or git.get("diff_sha256") is not None
        or identity.get("git_commit") != commit
    ):
        raise AttestationError("a candidate must come from the exact clean Git commit")

    try:
        holdout = validate_restricted_release_lock(
            identity.get("holdout"), dataset_version=dataset_version
        )
    except HoldoutContractError as exc:
        raise AttestationError(str(exc)) from exc

    database = _mapping(payload.get("database"), "release database payload")
    database_path = _relative(database.get("path"), "release database path")
    database_digest = _sha256(identity.get("database_sha256"), "database digest")
    if (
        _sha256(database.get("sha256"), "payload database digest")
        != database_digest
        or _file_digest(files, database_path, "release database file digest")
        != database_digest
    ):
        raise AttestationError("release database digests are inconsistent")

    dataset_manifest_path = _relative(
        payload.get("dataset_manifest"), "dataset manifest path"
    )
    dataset_manifest_digest = _file_digest(
        files, dataset_manifest_path, "dataset manifest digest"
    )
    retrieval = _mapping(payload.get("retrieval"), "release retrieval payload")
    if retrieval.get("dataset_version") != dataset_version:
        raise AttestationError("release retrieval dataset version is inconsistent")
    retrieval_root = _relative(retrieval.get("root"), "retrieval root")
    retrieval_manifest_path = _relative(
        retrieval.get("manifest"), "retrieval manifest path"
    )
    expected_retrieval_manifest = (
        PurePosixPath(retrieval_root) / dataset_version / "retrieval_manifest.json"
    ).as_posix()
    if retrieval_manifest_path != expected_retrieval_manifest:
        raise AttestationError("release retrieval manifest path is inconsistent")
    retrieval_manifest_digest = _sha256(
        identity.get("retrieval_manifest_sha256"), "retrieval manifest digest"
    )
    if _file_digest(
        files, retrieval_manifest_path, "release retrieval manifest file digest"
    ) != retrieval_manifest_digest:
        raise AttestationError("release retrieval manifest digests are inconsistent")
    retrieval_directory = PurePosixPath(retrieval_root) / dataset_version
    retrieval_files = {
        name: _file_digest(
            files,
            (retrieval_directory / filename).as_posix(),
            f"retrieval {name} digest",
        )
        for name, filename in {
            "documents_sha256": "documents.jsonl",
            "doc_ids_sha256": "doc_ids.json",
            "vectors_sha256": "vectors.npy",
            "index_sha256": "faiss.index",
        }.items()
    }

    embedding_model = _text(identity.get("embedding_model"), "embedding model")
    reranker_model = _text(identity.get("reranker_model"), "reranker model")
    embedding_digest = _sha256(
        identity.get("embedding_model_sha256"), "embedding model digest"
    )
    reranker_digest = _sha256(
        identity.get("reranker_model_sha256"), "reranker model digest"
    )
    return {
        "release_contract_version": "1",
        "release_id": release_id,
        "release_manifest_sha256": _sha256(
            manifest_sha256, "release manifest digest"
        ),
        "dataset_version": dataset_version,
        "schema_version": _text(identity.get("schema_version"), "schema version"),
        "database_sha256": database_digest,
        "dataset_manifest_sha256": dataset_manifest_digest,
        "evidence_state_sha256": _sha256(
            identity.get("evidence_state_sha256"), "evidence state digest"
        ),
        "retrieval_mode": "hybrid",
        "retrieval_manifest_sha256": retrieval_manifest_digest,
        "retrieval_files": retrieval_files,
        "embedding_model": embedding_model,
        "embedding_model_sha256": embedding_digest,
        "reranker_model": reranker_model,
        "reranker_model_sha256": reranker_digest,
        "holdout": holdout,
        "git_provenance": {
            "available": True,
            "commit": commit,
            "dirty": False,
            "diff_sha256": None,
        },
    }


def _expected_runtime(subject: Mapping[str, object]) -> dict[str, object]:
    files = _mapping(subject.get("retrieval_files"), "subject retrieval files")
    return {
        "release_id": subject["release_id"],
        "database_sha256": subject["database_sha256"],
        "dataset_version": subject["dataset_version"],
        "retrieval_mode": subject["retrieval_mode"],
        "dataset_manifest_sha256": subject["dataset_manifest_sha256"],
        "retrieval_manifest_sha256": subject["retrieval_manifest_sha256"],
        "documents_sha256": files["documents_sha256"],
        "doc_ids_sha256": files["doc_ids_sha256"],
        "vectors_sha256": files["vectors_sha256"],
        "index_sha256": files["index_sha256"],
        "embedding_model_sha256": subject["embedding_model_sha256"],
        "reranker_model_sha256": subject["reranker_model_sha256"],
    }


def _holdout_counts(subject: Mapping[str, object]) -> dict[str, int]:
    try:
        lock = validate_restricted_release_lock(subject.get("holdout"))
    except HoldoutContractError as exc:
        raise AttestationError(str(exc)) from exc
    inputs = _mapping(lock["inputs"], "subject holdout inputs")
    values: dict[str, int] = {}
    for role in ("agent_cases", "retrieval_documents", "retrieval_queries"):
        item = _mapping(inputs[role], f"subject holdout {role}")
        count = item.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise AttestationError(f"subject holdout {role} count is invalid")
        values[role] = count
    return values


def _validate_report_binding(
    kind: str,
    report: Mapping[str, object],
    subject: Mapping[str, object],
) -> None:
    counts = _holdout_counts(subject)
    try:
        if kind == "agent":
            validate_agent_report(
                report, expected_question_count=counts["agent_cases"]
            )
        else:
            validate_retrieval_report(
                report,
                expected_query_count=counts["retrieval_queries"],
                expected_document_count=counts["retrieval_documents"],
            )
    except PromotionPolicyError as exc:
        raise AttestationError(f"{kind} report violates promotion policy: {exc}") from exc
    provenance = _mapping(report.get("provenance"), f"{kind} provenance")
    if provenance.get("release_subject") != dict(subject):
        raise AttestationError(f"{kind} report targets a different release subject")
    if provenance.get("holdout") != subject.get("holdout"):
        raise AttestationError(f"{kind} report targets a different restricted holdout")
    if provenance.get("runtime") != _expected_runtime(subject):
        raise AttestationError(f"{kind} runtime files differ from the candidate")
    if provenance.get("evaluator_git") != subject.get("git_provenance"):
        raise AttestationError(f"{kind} evaluator Git state differs from the candidate")


def _report_attestation_view(
    kind: str,
    report: Mapping[str, object],
) -> dict[str, object]:
    """Keep signed policy evidence without copying protected questions/outcomes."""

    common = (
        "report_contract_version",
        "promotion_policy_version",
        "promotion_policy_sha256",
        "promotion_eligible",
        "evaluation_config",
    )
    keys: tuple[str, ...]
    if kind == "agent":
        keys = common + (
            "question_count",
            "intent_accuracy",
            "plan_exact_match",
            "tool_precision",
            "tool_recall",
            "answer_containment",
            "safe_rejection",
            "scope_pollution_rate",
            "refusal_count",
            "clarification_count",
            "gates",
            "passed",
            "provenance",
        )
    else:
        keys = common + (
            "query_count",
            "document_count",
            "variants",
            "thresholds",
            "hybrid_missing",
            "results",
            "status",
            "passed",
            "provenance",
        )
    missing = [key for key in keys if key not in report]
    if missing:
        raise AttestationError(f"{kind} report is missing signed fields: {missing}")
    return {
        "report_sha256": hashlib.sha256(canonical_json(dict(report))).hexdigest(),
        **{key: report[key] for key in keys},
    }


def _signature_payload(attestation: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in attestation.items() if key != "signature"}


def _decode_key(value: str | bytes, *, label: str) -> bytes:
    if isinstance(value, bytes) and len(value) == 32:
        return value
    encoded = value.encode("ascii") if isinstance(value, str) else value
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error, UnicodeEncodeError) as exc:
        raise AttestationError(f"{label} must be base64-encoded raw key bytes") from exc
    if len(decoded) != 32:
        raise AttestationError(f"{label} must contain exactly 32 raw bytes")
    return decoded


def attestation_key_id(public_key: str | bytes) -> str:
    raw = _decode_key(public_key, label="attestation public key")
    return "ed25519:" + hashlib.sha256(raw).hexdigest()


def _private_key(value: str | bytes) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(
        _decode_key(value, label="attestation private key")
    )


def _public_key(value: str | bytes) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(
        _decode_key(value, label="attestation public key")
    )


def _issued_at(value: object) -> str:
    text = _text(value, "attestation issued_at")
    if not text.endswith("Z"):
        raise AttestationError("attestation issued_at must be UTC RFC3339 ending in Z")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise AttestationError("attestation issued_at is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise AttestationError("attestation issued_at must be UTC")
    return text


def create_evaluation_attestation(
    *,
    subject: Mapping[str, object],
    agent_report: Mapping[str, object],
    retrieval_report: Mapping[str, object],
    issuer: str,
    private_key: str | bytes,
) -> dict[str, object]:
    """Recompute both reports and sign one exact candidate/holdout statement."""

    normalized_issuer = _text(issuer, "attestation issuer")
    _validate_report_binding("agent", agent_report, subject)
    _validate_report_binding("retrieval", retrieval_report, subject)
    agent_view = _report_attestation_view("agent", agent_report)
    retrieval_view = _report_attestation_view("retrieval", retrieval_report)
    signing_key = _private_key(private_key)
    public_raw = signing_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    key_id = attestation_key_id(public_raw)
    attestation: dict[str, object] = {
        "attestation_contract_version": ATTESTATION_CONTRACT_VERSION,
        "issued_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "issuer": normalized_issuer,
        "key_id": key_id,
        "subject": dict(subject),
        "promotion_policy": {
            "version": PROMOTION_POLICY_VERSION,
            "sha256": PROMOTION_POLICY_SHA256,
        },
        "reports": {
            "agent": agent_view,
            "retrieval": retrieval_view,
        },
        "all_gates_passed": True,
    }
    signature = signing_key.sign(canonical_json(_signature_payload(attestation)))
    attestation["signature"] = {
        "algorithm": ATTESTATION_ALGORITHM,
        "value": base64.b64encode(signature).decode("ascii"),
    }
    return attestation


def verify_evaluation_attestation(
    attestation: Mapping[str, object],
    *,
    expected_subject: Mapping[str, object],
    trusted_keys: Mapping[str, TrustedAttestationKey],
) -> None:
    """Verify allowlisted issuer/key, signature, subject, and every policy gate."""

    _exact_keys(
        attestation,
        frozenset(
            {
                "attestation_contract_version",
                "issued_at",
                "issuer",
                "key_id",
                "subject",
                "promotion_policy",
                "reports",
                "all_gates_passed",
                "signature",
            }
        ),
        "evaluation attestation",
    )
    if attestation.get("attestation_contract_version") != ATTESTATION_CONTRACT_VERSION:
        raise AttestationError("evaluation attestation contract is unsupported")
    if attestation.get("all_gates_passed") is not True:
        raise AttestationError("evaluation attestation is not passing")
    _issued_at(attestation.get("issued_at"))
    key_id = _text(attestation.get("key_id"), "attestation key_id")
    trusted = trusted_keys.get(key_id)
    if trusted is None or trusted.revoked:
        raise AttestationError("attestation key_id is unknown or revoked")
    if key_id != attestation_key_id(trusted.public_key):
        raise AttestationError("trusted key registry key_id is inconsistent")
    if attestation.get("issuer") != trusted.issuer:
        raise AttestationError("attestation issuer is not trusted for this key")
    signature = _mapping(attestation.get("signature"), "attestation signature")
    _exact_keys(signature, frozenset({"algorithm", "value"}), "attestation signature")
    if signature.get("algorithm") != ATTESTATION_ALGORITHM:
        raise AttestationError("evaluation attestation signature algorithm is unsupported")
    encoded_signature = signature.get("value")
    if not isinstance(encoded_signature, str):
        raise AttestationError("evaluation attestation signature is invalid")
    try:
        signature_bytes = base64.b64decode(encoded_signature.encode("ascii"), validate=True)
    except (ValueError, binascii.Error, UnicodeEncodeError) as exc:
        raise AttestationError("evaluation attestation signature is invalid") from exc
    if len(signature_bytes) != 64:
        raise AttestationError("evaluation attestation signature is invalid")
    try:
        _public_key(trusted.public_key).verify(
            signature_bytes,
            canonical_json(_signature_payload(attestation)),
        )
    except InvalidSignature as exc:
        raise AttestationError("evaluation attestation signature is invalid") from exc

    subject = _mapping(attestation.get("subject"), "attestation subject")
    if dict(subject) != dict(expected_subject):
        raise AttestationError("evaluation attestation targets a different release")
    policy = _mapping(attestation.get("promotion_policy"), "promotion policy")
    _exact_keys(policy, frozenset({"version", "sha256"}), "promotion policy")
    if (
        policy.get("version") != PROMOTION_POLICY_VERSION
        or policy.get("sha256") != PROMOTION_POLICY_SHA256
    ):
        raise AttestationError("evaluation attestation uses an untrusted promotion policy")
    reports = _mapping(attestation.get("reports"), "attestation reports")
    _exact_keys(reports, frozenset({"agent", "retrieval"}), "attestation reports")
    _validate_report_binding(
        "agent", _mapping(reports.get("agent"), "agent report"), subject
    )
    _validate_report_binding(
        "retrieval",
        _mapping(reports.get("retrieval"), "retrieval report"),
        subject,
    )


__all__ = [
    "ATTESTATION_ALGORITHM",
    "ATTESTATION_CONTRACT_VERSION",
    "AttestationError",
    "TrustedAttestationKey",
    "attestation_key_id",
    "create_evaluation_attestation",
    "release_evaluation_subject",
    "verify_evaluation_attestation",
]
