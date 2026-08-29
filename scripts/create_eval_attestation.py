"""Create a redacted Ed25519 promotion attestation for one candidate release."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path

from storage.attestation import (
    AttestationError,
    create_evaluation_attestation,
    release_evaluation_subject,
)
from storage.json_contract import StrictJSONError, load_strict_json_snapshot
from storage.release import (
    RELEASE_MANIFEST_NAME,
    ReleaseError,
    atomic_write_json,
    sha256_file,
    validate_release_directory,
)


def _load_object(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise AttestationError(f"{label} is missing or unsafe: {path}")
    try:
        value, _digest, _raw = load_strict_json_snapshot(path, label=label)
    except StrictJSONError as exc:
        raise AttestationError(str(exc)) from exc
    if not isinstance(value, Mapping):
        raise AttestationError(f"{label} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise AttestationError(f"{label} keys must be strings")
    return dict(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-release-manifest", type=Path, required=True)
    parser.add_argument("--agent-report", type=Path, required=True)
    parser.add_argument("--retrieval-report", type=Path, required=True)
    parser.add_argument("--issuer", required=True)
    parser.add_argument(
        "--private-key-env",
        default="SWUFE_RELEASE_ATTESTATION_PRIVATE_KEY",
        help="environment variable containing a base64 raw Ed25519 private key",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    private_key = os.getenv(args.private_key_env, "")
    if not private_key:
        raise SystemExit(
            f"attestation signing private key is missing from {args.private_key_env}"
        )
    manifest_path = args.candidate_release_manifest
    if (
        manifest_path.name != RELEASE_MANIFEST_NAME
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
    ):
        raise SystemExit(f"candidate manifest must be a regular {RELEASE_MANIFEST_NAME}")
    manifest_path = manifest_path.resolve()
    try:
        manifest = validate_release_directory(manifest_path.parent)
        subject = release_evaluation_subject(
            manifest,
            manifest_sha256=sha256_file(manifest_path),
        )
        attestation = create_evaluation_attestation(
            subject=subject,
            agent_report=_load_object(args.agent_report, "agent report"),
            retrieval_report=_load_object(args.retrieval_report, "retrieval report"),
            issuer=args.issuer,
            private_key=private_key,
        )
        if args.output.is_symlink():
            raise AttestationError("attestation output cannot be a symlink")
        atomic_write_json(args.output, attestation)
    except (AttestationError, ReleaseError, OSError, ValueError) as exc:
        raise SystemExit(f"cannot create evaluation attestation: {exc}") from exc

    reports = attestation["reports"]
    assert isinstance(reports, dict)
    agent = reports["agent"]
    retrieval = reports["retrieval"]
    assert isinstance(agent, dict) and isinstance(retrieval, dict)
    print(
        json.dumps(
            {
                "attestation": str(args.output),
                "release_id": subject["release_id"],
                "issuer": attestation["issuer"],
                "key_id": attestation["key_id"],
                "agent_report_sha256": agent["report_sha256"],
                "retrieval_report_sha256": retrieval["report_sha256"],
                "all_gates_passed": True,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
