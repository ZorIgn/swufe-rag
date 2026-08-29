from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from eval.promotion_policy import (
    AGENT_EVALUATION_CONFIG,
    AGENT_GATE_RULES,
    PROMOTION_POLICY_SHA256,
    PROMOTION_POLICY_VERSION,
    REPORT_CONTRACT_VERSION,
    RETRIEVAL_EVALUATION_CONFIG,
    RETRIEVAL_GATE_RULES,
    GateRule,
)
from storage.attestation import (
    ATTESTATION_CONTRACT_VERSION,
    AttestationError,
    TrustedAttestationKey,
    attestation_key_id,
    create_evaluation_attestation,
    release_evaluation_subject,
    verify_evaluation_attestation,
)
from storage.json_contract import canonical_json
from storage.release import (
    ATTESTATIONS_DIRECTORY_NAME,
    ReleaseError,
    activate_release,
    build_release_manifest,
    load_active_release,
    make_staging_directory,
    publish_attestation,
    publish_release,
    sha256_file,
    validate_release_directory,
)

ISSUER = "ci-release-job"
PRIVATE_KEY = bytes(range(1, 33))
PUBLIC_KEY = (
    Ed25519PrivateKey.from_private_bytes(PRIVATE_KEY)
    .public_key()
    .public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
)
KEY_ID = attestation_key_id(PUBLIC_KEY)
TRUSTED_KEYS = {
    KEY_ID: TrustedAttestationKey(issuer=ISSUER, public_key=PUBLIC_KEY)
}


def _holdout_lock() -> dict[str, object]:
    inputs = {
        "agent_cases": {
            "path": "agent_cases.json",
            "sha256": "9" * 64,
            "count": 20,
        },
        "retrieval_documents": {
            "path": "retrieval_documents.jsonl",
            "sha256": "a" * 64,
            "count": 100,
        },
        "retrieval_queries": {
            "path": "retrieval_queries.json",
            "sha256": "b" * 64,
            "count": 20,
        },
    }
    additional_files: dict[str, str] = {}
    bundle_sha256 = hashlib.sha256(
        canonical_json(
            {
                "inputs": inputs,
                "additional_files": additional_files,
            }
        )
    ).hexdigest()
    return {
        "status": "frozen",
        "holdout_contract_version": "2",
        "kind": "restricted_holdout",
        "holdout_id": "restricted-eval-v1",
        "dataset_version": "dataset-v1",
        "access": "restricted",
        "bundle_sha256": bundle_sha256,
        "manifest_sha256": "8" * 64,
        "inputs": inputs,
        "additional_files": additional_files,
    }


def _write_candidate_files(staging: Path) -> dict[str, str]:
    paths = {
        "database": staging / "data" / "academic.sqlite3",
        "dataset_manifest": staging / "data" / "dataset_manifest.json",
        "retrieval_manifest": (
            staging / "retrieval" / "dataset-v1" / "retrieval_manifest.json"
        ),
        "documents": staging / "retrieval" / "dataset-v1" / "documents.jsonl",
        "doc_ids": staging / "retrieval" / "dataset-v1" / "doc_ids.json",
        "vectors": staging / "retrieval" / "dataset-v1" / "vectors.npy",
        "index": staging / "retrieval" / "dataset-v1" / "faiss.index",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    paths["database"].write_bytes(b"sqlite candidate")
    paths["dataset_manifest"].write_text(
        '{"dataset_version":"dataset-v1"}', encoding="utf-8"
    )
    paths["retrieval_manifest"].write_text(
        '{"dataset_version":"dataset-v1"}', encoding="utf-8"
    )
    paths["documents"].write_text('{"chunk_id":"doc-1"}\n', encoding="utf-8")
    paths["doc_ids"].write_text('["doc-1"]', encoding="utf-8")
    paths["vectors"].write_bytes(b"deterministic vectors")
    paths["index"].write_bytes(b"deterministic index")
    return {name: sha256_file(path) for name, path in paths.items()}


def _identity(hashes: Mapping[str, str], *, dirty: bool = False) -> dict[str, object]:
    commit = "f" * 40
    return {
        "dataset_version": "dataset-v1",
        "schema_version": "1",
        "database_sha256": hashes["database"],
        "evidence_state_sha256": "c" * 64,
        "retrieval_manifest_sha256": hashes["retrieval_manifest"],
        "retrieval_mode": "hybrid",
        "embedding_model": "models/embedding",
        "embedding_model_sha256": "d" * 64,
        "reranker_model": "models/reranker",
        "reranker_model_sha256": "e" * 64,
        "holdout": _holdout_lock(),
        "release_tier": "candidate",
        "git_commit": commit,
        "git_provenance": {
            "available": True,
            "commit": commit,
            "dirty": dirty,
            "diff_sha256": "0" * 64 if dirty else None,
        },
    }


def _payload(hashes: Mapping[str, str]) -> dict[str, object]:
    return {
        "database": {
            "path": "data/academic.sqlite3",
            "sha256": hashes["database"],
        },
        "dataset_manifest": "data/dataset_manifest.json",
        "retrieval": {
            "dataset_version": "dataset-v1",
            "root": "retrieval",
            "manifest": "retrieval/dataset-v1/retrieval_manifest.json",
        },
    }


def _metric(report: Mapping[str, object], path: tuple[str, ...]) -> float:
    value: object = report
    for part in path:
        assert isinstance(value, Mapping)
        value = value[part]
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    return float(value)


def _gates(
    rules: Mapping[str, GateRule],
    metrics: Mapping[str, object],
    *,
    sample_count: int,
    hard_negative_count: int | None = None,
) -> dict[str, object]:
    values: dict[str, object] = {}
    for name, rule in rules.items():
        actual = _metric(metrics, rule.metric_path)
        count = (
            hard_negative_count
            if name == "hard_negative_rate_at_cutoff"
            else sample_count
        )
        assert count is not None
        values[name] = {
            "name": name,
            "operator": rule.operator,
            "threshold": rule.threshold,
            "actual": actual,
            "sample_count": count,
            "status": "measured",
            "passed": True,
        }
    return values


def _runtime(subject: Mapping[str, object]) -> dict[str, object]:
    files = subject["retrieval_files"]
    assert isinstance(files, Mapping)
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


def _provenance(subject: Mapping[str, object]) -> dict[str, object]:
    return {
        "release_subject": dict(subject),
        "holdout": subject["holdout"],
        "runtime": _runtime(subject),
        "evaluator_git": subject["git_provenance"],
    }


def _reports(subject: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    agent: dict[str, object] = {
        "report_contract_version": REPORT_CONTRACT_VERSION,
        "promotion_policy_version": PROMOTION_POLICY_VERSION,
        "promotion_policy_sha256": PROMOTION_POLICY_SHA256,
        "promotion_eligible": True,
        "evaluation_config": AGENT_EVALUATION_CONFIG,
        "question_count": 20,
        "intent_accuracy": 1.0,
        "plan_exact_match": 1.0,
        "tool_precision": 1.0,
        "tool_recall": 1.0,
        "answer_containment": 1.0,
        "safe_rejection": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
        "scope_pollution_rate": 0.0,
        "refusal_count": 1,
        "clarification_count": 1,
        "passed": True,
        "provenance": _provenance(subject),
        "outcomes": [{"question": "protected holdout question"}],
    }
    agent["gates"] = _gates(AGENT_GATE_RULES, agent, sample_count=20)

    results: dict[str, object] = {}
    for name in ("lexical", "hybrid"):
        variant: dict[str, object] = {
            "status": "measured",
            "recall_at_1": 1.0,
            "recall_at_5": 1.0,
            "recall_at_10": 1.0,
            "mrr": 1.0,
            "ndcg_at_10": 1.0,
            "hard_negative_rate_at_cutoff": 0.0,
            "scope_violation_rate": 0.0,
            "hard_negative_labeled_queries": 5,
            "passed": True,
        }
        variant["gates"] = _gates(
            RETRIEVAL_GATE_RULES,
            variant,
            sample_count=20,
            hard_negative_count=5,
        )
        results[name] = variant
    retrieval = {
        "report_contract_version": REPORT_CONTRACT_VERSION,
        "promotion_policy_version": PROMOTION_POLICY_VERSION,
        "promotion_policy_sha256": PROMOTION_POLICY_SHA256,
        "promotion_eligible": True,
        "evaluation_config": RETRIEVAL_EVALUATION_CONFIG,
        "query_count": 20,
        "document_count": 100,
        "variants": ["lexical", "hybrid"],
        "thresholds": {
            name: {"operator": rule.operator, "threshold": rule.threshold}
            for name, rule in RETRIEVAL_GATE_RULES.items()
        },
        "hybrid_missing": "fail",
        "results": results,
        "status": "passed",
        "passed": True,
        "provenance": _provenance(subject),
    }
    return agent, retrieval


def _published_candidate(root: Path) -> tuple[str, Path, dict[str, object]]:
    staging = make_staging_directory(root)
    hashes = _write_candidate_files(staging)
    manifest = build_release_manifest(
        identity=_identity(hashes),
        payload=_payload(hashes),
        staging_directory=staging,
    )
    published = publish_release(
        staging,
        releases_root=root,
        manifest=manifest,
        activate=False,
    )
    validated = validate_release_directory(published.directory)
    return published.release_id, published.directory, validated


def test_signed_attestation_activates_and_loads_the_exact_candidate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "releases"
    release_id, directory, manifest = _published_candidate(root)
    manifest_digest = sha256_file(directory / "release_manifest.json")
    subject = release_evaluation_subject(
        manifest,
        manifest_sha256=manifest_digest,
    )
    agent, retrieval = _reports(subject)
    attestation = create_evaluation_attestation(
        subject=subject,
        agent_report=agent,
        retrieval_report=retrieval,
        issuer=ISSUER,
        private_key=PRIVATE_KEY,
    )

    reports = attestation["reports"]
    assert isinstance(reports, Mapping)
    signed_agent = reports["agent"]
    assert isinstance(signed_agent, Mapping)
    assert "outcomes" not in signed_agent
    relative, digest, attestation_path = publish_attestation(
        root, release_id, attestation
    )
    assert relative == (
        Path(ATTESTATIONS_DIRECTORY_NAME) / release_id / f"{digest}.json"
    )
    assert publish_attestation(root, release_id, attestation) == (
        relative,
        digest,
        attestation_path,
    )
    activate_release(
        root,
        release_id,
        expected_manifest_sha256=manifest_digest,
        promotion={
            "attestation_contract_version": ATTESTATION_CONTRACT_VERSION,
            "path": relative.as_posix(),
            "sha256": digest,
        },
    )

    loaded_directory, loaded_manifest = load_active_release(
        root,
        trusted_attestation_keys=TRUSTED_KEYS,
    )

    assert loaded_directory == directory
    assert loaded_manifest["release_id"] == release_id


def test_attestation_rejects_failed_gate_tampering_and_untrusted_keys(
    tmp_path: Path,
) -> None:
    root = tmp_path / "releases"
    _release_id, directory, manifest = _published_candidate(root)
    subject = release_evaluation_subject(
        manifest,
        manifest_sha256=sha256_file(directory / "release_manifest.json"),
    )
    agent, retrieval = _reports(subject)
    failed_agent = {**agent, "passed": False}
    with pytest.raises(AttestationError, match="did not pass"):
        create_evaluation_attestation(
            subject=subject,
            agent_report=failed_agent,
            retrieval_report=retrieval,
            issuer=ISSUER,
            private_key=PRIVATE_KEY,
        )

    attestation = create_evaluation_attestation(
        subject=subject,
        agent_report=agent,
        retrieval_report=retrieval,
        issuer=ISSUER,
        private_key=PRIVATE_KEY,
    )
    tampered = json.loads(json.dumps(attestation))
    tampered["issued_at"] = "2026-01-01T00:00:00Z"
    with pytest.raises(AttestationError, match="signature is invalid"):
        verify_evaluation_attestation(
            tampered,
            expected_subject=subject,
            trusted_keys=TRUSTED_KEYS,
        )

    with pytest.raises(AttestationError, match="unknown or revoked"):
        verify_evaluation_attestation(
            attestation,
            expected_subject=subject,
            trusted_keys={
                KEY_ID: TrustedAttestationKey(
                    issuer=ISSUER,
                    public_key=PUBLIC_KEY,
                    revoked=True,
                )
            },
        )


def test_dirty_candidate_cannot_become_an_attestation_subject(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    _release_id, directory, manifest = _published_candidate(root)
    identity = manifest["identity"]
    assert isinstance(identity, Mapping)
    dirty_identity = dict(identity)
    git = dirty_identity["git_provenance"]
    assert isinstance(git, Mapping)
    dirty_identity["git_provenance"] = {
        **git,
        "dirty": True,
        "diff_sha256": "0" * 64,
    }
    dirty_manifest = {**manifest, "identity": dirty_identity}
    with pytest.raises(AttestationError, match="exact clean Git commit"):
        release_evaluation_subject(
            dirty_manifest,
            manifest_sha256=sha256_file(directory / "release_manifest.json"),
        )


def test_unattested_active_pointer_is_rejected_by_default(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    release_id, _directory, _manifest = _published_candidate(root)
    activate_release(root, release_id)

    with pytest.raises(ReleaseError, match="no evaluation promotion attestation"):
        load_active_release(root)

    directory, manifest = load_active_release(root, require_attestation=False)
    assert directory.name == release_id
    assert manifest["release_id"] == release_id
