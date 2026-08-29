"""Promote an evaluated immutable candidate via an Ed25519 attestation."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path

from storage.attestation import (
    ATTESTATION_CONTRACT_VERSION,
    AttestationError,
    TrustedAttestationKey,
    attestation_key_id,
    release_evaluation_subject,
    verify_evaluation_attestation,
)
from storage.json_contract import StrictJSONError, load_strict_json_snapshot
from storage.release import (
    RELEASE_MANIFEST_NAME,
    ReleaseError,
    activate_release,
    publish_attestation,
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
    parser.add_argument("--release-root", type=Path, default=Path("artifacts/releases"))
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--trusted-issuer", required=True)
    parser.add_argument(
        "--public-key-env",
        default="SWUFE_RELEASE_ATTESTATION_PUBLIC_KEY",
        help="environment variable containing a base64 raw Ed25519 public key",
    )
    args = parser.parse_args()

    public_key = os.getenv(args.public_key_env, "")
    if not public_key:
        raise SystemExit(
            f"attestation verification public key is missing from {args.public_key_env}"
        )
    release_directory = args.release_root / args.release_id
    try:
        manifest = validate_release_directory(release_directory)
        if manifest.get("release_id") != args.release_id:
            raise ReleaseError("release directory and manifest id differ")
        manifest_path = release_directory / RELEASE_MANIFEST_NAME
        manifest_sha256 = sha256_file(manifest_path)
        subject = release_evaluation_subject(
            manifest,
            manifest_sha256=manifest_sha256,
        )
        attestation = _load_object(args.attestation, "evaluation attestation")
        key_id = attestation_key_id(public_key)
        trusted_keys = {
            key_id: TrustedAttestationKey(
                issuer=args.trusted_issuer,
                public_key=public_key,
            )
        }
        verify_evaluation_attestation(
            attestation,
            expected_subject=subject,
            trusted_keys=trusted_keys,
        )
        relative, digest, attestation_target = publish_attestation(
            args.release_root,
            args.release_id,
            attestation,
        )
        pointer = activate_release(
            args.release_root,
            args.release_id,
            promotion={
                "attestation_contract_version": ATTESTATION_CONTRACT_VERSION,
                "path": relative.as_posix(),
                "sha256": digest,
            },
            expected_manifest_sha256=manifest_sha256,
        )
    except (AttestationError, ReleaseError, OSError, ValueError) as exc:
        raise SystemExit(f"release promotion failed: {exc}") from exc

    print(
        json.dumps(
            {
                "release_id": args.release_id,
                "active_pointer": str(pointer),
                "attestation": str(attestation_target),
                "attestation_sha256": digest,
                "status": "production",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
