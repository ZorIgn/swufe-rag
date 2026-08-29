"""Content-addressed, verified, atomically activated data releases.

The application deliberately keeps a release as one unit: the SQLite database,
the retrieval artifact and the manifests that describe both.  Building those
files directly into their live paths makes a partially-written index look like
a usable dataset after an interruption.  This module centralises the small
release protocol used by :mod:`scripts.build_all`:

``staging -> validate every file -> immutable release directory -> active pointer``.

It is intentionally filesystem-only.  A future object store or lakehouse can
implement the same manifest/pointer contract without changing the runtime's
trust boundary.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from storage.json_contract import StrictJSONError, canonical_json, load_strict_json_snapshot

if TYPE_CHECKING:
    from storage.attestation import TrustedAttestationKey

RELEASE_CONTRACT_VERSION = "1"
ACTIVE_POINTER_NAME = "active.json"
RELEASE_MANIFEST_NAME = "release_manifest.json"
ATTESTATIONS_DIRECTORY_NAME = "attestations"
_RELEASE_ID_RE = re.compile(r"^sha256-[0-9a-f]{64}$")


class ReleaseError(RuntimeError):
    """Raised when a release cannot be validated or atomically published."""


@dataclass(frozen=True)
class PublishedRelease:
    """The immutable directory and pointer produced by a successful publish."""

    release_id: str
    directory: Path
    manifest_path: Path
    active_pointer: Path | None


@dataclass(frozen=True)
class ReleaseRuntimeBundle:
    """A validated database/index unit that cannot be split by ambient config."""

    release_id: str
    database_path: Path
    retrieval_root: Path
    dataset_manifest_path: Path
    retrieval_mode: str


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest without loading a data artifact at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_directory(path: str | Path) -> str:
    """Hash a complete, symlink-free model snapshot by relative file name."""

    values = file_hashes(path, exclude=frozenset())
    return sha256_bytes(canonical_json(values))


def make_staging_directory(releases_root: str | Path) -> Path:
    """Create a same-filesystem temporary release directory.

    Keeping staging under ``releases_root`` is important: ``os.replace`` is
    atomic only on one filesystem.  The directory name is deliberately hidden
    so runtime discovery, which only accepts a validated active pointer, never
    observes it.
    """

    root = Path(releases_root)
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=".staging-", dir=root))


def discard_staging(directory: str | Path) -> None:
    """Remove a directory only when it is an explicitly-created staging path."""

    path = Path(directory)
    if not path.name.startswith(".staging-"):
        raise ReleaseError(f"refusing to discard a non-staging directory: {path}")
    if path.exists():
        shutil.rmtree(path)


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    """Durably replace one JSON file without exposing a truncated pointer."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(canonical_json(dict(value)))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        # ``os.replace`` removes it in the successful case.  A failed write
        # should not leave a future loader considering this a release file.
        if temporary.exists():
            temporary.unlink()


def _safe_relative_file(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    if relative.is_absolute() or ".." in relative.parts:
        raise ReleaseError(f"release file escapes its root: {path}")
    return relative.as_posix()


def file_hashes(
    root: str | Path, *, exclude: frozenset[str] = frozenset({RELEASE_MANIFEST_NAME})
) -> dict[str, str]:
    """Hash every regular release file in deterministic relative-path order.

    Symlinks are not allowed in a release.  They make a manifest validate a
    different object after publication, particularly on developer machines.
    """

    directory = Path(root)
    if not directory.is_dir():
        raise ReleaseError(f"release directory is missing: {directory}")
    values: dict[str, str] = {}
    for candidate in sorted(directory.rglob("*"), key=lambda item: item.as_posix()):
        if candidate.is_symlink():
            raise ReleaseError(f"release cannot contain a symlink: {candidate}")
        if not candidate.is_file():
            continue
        relative = _safe_relative_file(candidate, directory)
        if relative in exclude:
            continue
        values[relative] = sha256_file(candidate)
    if not values:
        raise ReleaseError("release contains no files")
    return values


def content_release_id(identity: Mapping[str, object]) -> str:
    """Create a human-readable full SHA-256 release identity from immutable inputs."""

    return "sha256-" + sha256_bytes(canonical_json(dict(identity)))


def build_release_manifest(
    *,
    identity: Mapping[str, object],
    payload: Mapping[str, object],
    staging_directory: str | Path,
) -> dict[str, object]:
    """Build a self-contained manifest after all staged files exist.

    ``identity`` must exclude wall-clock fields.  The generated release id is
    therefore reproducible from the exact database/index/holdout content,
    while ``payload`` can record non-identity provenance such as build time.
    """

    release_id = content_release_id(identity)
    manifest: dict[str, object] = {
        "release_contract_version": RELEASE_CONTRACT_VERSION,
        "release_id": release_id,
        "identity": dict(identity),
        "payload": dict(payload),
        "files": file_hashes(staging_directory),
    }
    return manifest


def validate_release_directory(directory: str | Path) -> dict[str, object]:
    """Fail closed unless the immutable manifest exactly matches its tree."""

    root = Path(directory)
    if root.is_symlink():
        raise ReleaseError("release directory cannot be a symlink")
    manifest_path = root / RELEASE_MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ReleaseError(f"release manifest is missing: {manifest_path}")
    try:
        parsed, _digest, _raw = load_strict_json_snapshot(manifest_path, label="release manifest")
    except StrictJSONError as exc:
        raise ReleaseError(f"release manifest is unreadable: {manifest_path}") from exc
    if not isinstance(parsed, dict):
        raise ReleaseError("release manifest must be a JSON object")
    if str(parsed.get("release_contract_version") or "") != RELEASE_CONTRACT_VERSION:
        raise ReleaseError("release contract version is unsupported")
    release_id = str(parsed.get("release_id") or "")
    identity = parsed.get("identity")
    files = parsed.get("files")
    if not release_id or not isinstance(identity, dict) or not isinstance(files, dict):
        raise ReleaseError("release manifest lacks required identity or file hashes")
    if content_release_id(identity) != release_id:
        raise ReleaseError("release id does not match its immutable identity")
    expected = {str(path): str(digest) for path, digest in files.items()}
    actual = file_hashes(root)
    if expected != actual:
        raise ReleaseError("release file hashes do not match the manifest")
    return parsed


def publish_release(
    staging_directory: str | Path,
    *,
    releases_root: str | Path,
    manifest: Mapping[str, object],
    activate: bool,
) -> PublishedRelease:
    """Publish one validated, immutable release and optionally switch the pointer.

    The previous active pointer is left untouched until the directory has been
    atomically renamed and its manifest has been re-read.  A build failure thus
    preserves the last runnable release instead of making a half-built one
    appear current.
    """

    root = Path(releases_root)
    staging = Path(staging_directory)
    if staging.parent.resolve() != root.resolve() or not staging.name.startswith(".staging-"):
        raise ReleaseError("staging directory must be an explicit child of release root")
    release_id = str(manifest.get("release_id") or "")
    if _RELEASE_ID_RE.fullmatch(release_id) is None:
        raise ReleaseError("release manifest has no content-addressed release id")
    atomic_write_json(staging / RELEASE_MANIFEST_NAME, dict(manifest))
    validated = validate_release_directory(staging)
    if validated != dict(manifest):
        raise ReleaseError("staged release changed while its manifest was being written")

    target = root / release_id
    if target.exists():
        # Reusing an already-published byte-identical release is safe.  Anything
        # else is a collision or an operator attempt to overwrite an immutable
        # release and must stop before the active pointer is touched.
        existing = validate_release_directory(target)
        if existing != dict(manifest):
            raise ReleaseError(f"immutable release id collision: {release_id}")
        discard_staging(staging)
    else:
        try:
            os.replace(staging, target)
        except OSError as exc:
            raise ReleaseError(f"cannot atomically publish release {release_id}") from exc
        validate_release_directory(target)

    pointer: Path | None = None
    if activate:
        pointer = activate_release(root, release_id)
    return PublishedRelease(
        release_id=release_id,
        directory=target,
        manifest_path=target / RELEASE_MANIFEST_NAME,
        active_pointer=pointer,
    )


def activate_release(
    releases_root: str | Path,
    release_id: str,
    *,
    promotion: Mapping[str, object] | None = None,
    expected_manifest_sha256: str | None = None,
) -> Path:
    """Atomically select a revalidated immutable release."""

    root = Path(releases_root)
    if root.is_symlink():
        raise ReleaseError("release root cannot be a symlink")
    if _RELEASE_ID_RE.fullmatch(release_id) is None:
        raise ReleaseError("cannot activate an invalid release id")
    directory = root / release_id
    validate_release_directory(directory)
    manifest_sha256 = sha256_file(directory / RELEASE_MANIFEST_NAME)
    if (
        expected_manifest_sha256 is not None
        and manifest_sha256 != expected_manifest_sha256
    ):
        raise ReleaseError("candidate changed after promotion validation")
    pointer_value: dict[str, object] = {
        "release_contract_version": RELEASE_CONTRACT_VERSION,
        "release_id": release_id,
        "manifest_sha256": manifest_sha256,
    }
    if promotion is not None:
        if frozenset(promotion) != frozenset(
            {"attestation_contract_version", "path", "sha256"}
        ):
            raise ReleaseError("promotion pointer metadata fields are invalid")
        if promotion.get("attestation_contract_version") != "2":
            raise ReleaseError("promotion pointer contract is unsupported")
        pointer_value["promotion"] = dict(promotion)
    pointer = root / ACTIVE_POINTER_NAME
    if pointer.is_symlink():
        raise ReleaseError("active release pointer cannot be a symlink")
    atomic_write_json(pointer, pointer_value)
    return pointer


def publish_attestation(
    releases_root: str | Path,
    release_id: str,
    attestation: Mapping[str, object],
) -> tuple[Path, str, Path]:
    """Publish canonical attestation bytes by digest without overwrite races."""

    root = Path(releases_root)
    if root.is_symlink() or _RELEASE_ID_RE.fullmatch(release_id) is None:
        raise ReleaseError("attestation target release is invalid")
    raw = canonical_json(dict(attestation))
    digest = sha256_bytes(raw)
    relative = (
        Path(ATTESTATIONS_DIRECTORY_NAME) / release_id / f"{digest}.json"
    )
    target = root / relative
    attestations_root = root / ATTESTATIONS_DIRECTORY_NAME
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if attestations_root.is_symlink() or parent.is_symlink():
        raise ReleaseError("attestation directory cannot be a symlink")
    temporary = attestations_root / f".{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            try:
                existing = target.read_bytes()
            except OSError as exc:
                raise ReleaseError("published attestation is unreadable") from exc
            if existing != raw:
                raise ReleaseError("content-addressed attestation collision") from None
    finally:
        if temporary.exists():
            temporary.unlink()
    if sha256_file(target) != digest:
        raise ReleaseError("published attestation digest mismatch")
    return relative, digest, target


def _load_promotion_attestation(
    root: Path,
    release_id: str,
    promotion: object,
) -> tuple[dict[str, object], str]:
    if not isinstance(promotion, Mapping):
        raise ReleaseError("active release has no evaluation promotion attestation")
    if frozenset(promotion) != frozenset(
        {"attestation_contract_version", "path", "sha256"}
    ):
        raise ReleaseError("active release promotion fields are invalid")
    if promotion.get("attestation_contract_version") != "2":
        raise ReleaseError("active release promotion contract is unsupported")
    expected_digest = str(promotion.get("sha256") or "")
    if re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None:
        raise ReleaseError("active release promotion digest is invalid")
    relative = Path(str(promotion.get("path") or ""))
    expected_relative = (
        Path(ATTESTATIONS_DIRECTORY_NAME)
        / release_id
        / f"{expected_digest}.json"
    )
    if relative != expected_relative or relative.is_absolute() or ".." in relative.parts:
        raise ReleaseError("active release promotion attestation path is invalid")
    path = root / relative
    if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
        raise ReleaseError("active release promotion attestation is missing")
    try:
        parsed, digest, _raw = load_strict_json_snapshot(
            path, label="promotion attestation"
        )
    except StrictJSONError as exc:
        raise ReleaseError("active release promotion attestation is unreadable") from exc
    if digest != expected_digest:
        raise ReleaseError("active release promotion attestation hash does not match")
    if not isinstance(parsed, dict):
        raise ReleaseError("active release promotion attestation must be an object")
    return parsed, digest


def load_active_release(
    releases_root: str | Path,
    *,
    require_attestation: bool = True,
    trusted_attestation_keys: Mapping[str, TrustedAttestationKey] | None = None,
) -> tuple[Path, dict[str, object]]:
    """Resolve and verify the active release; never guess or split its artifacts."""

    root = Path(releases_root)
    if root.is_symlink():
        raise ReleaseError("release root cannot be a symlink")
    pointer = root / ACTIVE_POINTER_NAME
    if pointer.is_symlink() or not pointer.is_file():
        raise ReleaseError(f"active release pointer is missing: {pointer}")
    try:
        value, _pointer_digest, _raw = load_strict_json_snapshot(
            pointer, label="active pointer"
        )
    except StrictJSONError as exc:
        raise ReleaseError(f"active release pointer is unreadable: {pointer}") from exc
    if not isinstance(value, dict):
        raise ReleaseError("active release pointer must be an object")
    expected_keys = {
        "release_contract_version",
        "release_id",
        "manifest_sha256",
    }
    if "promotion" in value:
        expected_keys.add("promotion")
    if frozenset(value) != frozenset(expected_keys):
        raise ReleaseError("active release pointer fields are invalid")
    if value.get("release_contract_version") != RELEASE_CONTRACT_VERSION:
        raise ReleaseError("active release pointer has an unsupported contract")
    release_id = str(value.get("release_id") or "")
    if _RELEASE_ID_RE.fullmatch(release_id) is None:
        raise ReleaseError("active release pointer has an invalid release id")
    directory = root / release_id
    manifest = validate_release_directory(directory)
    expected_manifest_sha = str(value.get("manifest_sha256") or "")
    actual_manifest_sha = sha256_file(directory / RELEASE_MANIFEST_NAME)
    if expected_manifest_sha != actual_manifest_sha:
        raise ReleaseError("active release pointer does not match its manifest")

    promotion = value.get("promotion")
    if require_attestation or promotion is not None:
        attestation, _digest = _load_promotion_attestation(
            root, release_id, promotion
        )
        if trusted_attestation_keys is None:
            raise ReleaseError("trusted release attestation keys are not configured")
        from storage.attestation import (
            AttestationError,
            release_evaluation_subject,
            verify_evaluation_attestation,
        )

        try:
            subject = release_evaluation_subject(
                manifest,
                manifest_sha256=actual_manifest_sha,
            )
            verify_evaluation_attestation(
                attestation,
                expected_subject=subject,
                trusted_keys=trusted_attestation_keys,
            )
        except AttestationError as exc:
            raise ReleaseError(
                f"active release evaluation attestation is invalid: {exc}"
            ) from exc
    return directory, manifest


__all__ = [
    "ACTIVE_POINTER_NAME",
    "ATTESTATIONS_DIRECTORY_NAME",
    "RELEASE_CONTRACT_VERSION",
    "RELEASE_MANIFEST_NAME",
    "PublishedRelease",
    "ReleaseRuntimeBundle",
    "ReleaseError",
    "activate_release",
    "atomic_write_json",
    "build_release_manifest",
    "canonical_json",
    "content_release_id",
    "discard_staging",
    "file_hashes",
    "load_active_release",
    "make_staging_directory",
    "publish_attestation",
    "publish_release",
    "sha256_bytes",
    "sha256_directory",
    "sha256_file",
    "validate_release_directory",
]
