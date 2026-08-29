"""One strict, discriminated contract for public fixtures and restricted holdouts."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from storage.json_contract import (
    StrictJSONError,
    canonical_json,
    load_strict_json_file,
)

HOLDOUT_CONTRACT_VERSION = "2"
REQUIRED_INPUT_ROLES = (
    "agent_cases",
    "retrieval_documents",
    "retrieval_queries",
)
HoldoutKind = Literal["test_fixture", "restricted_holdout"]


class HoldoutContractError(ValueError):
    """Raised when a holdout descriptor or referenced input is unsafe."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOP_LEVEL_KEYS = frozenset(
    {
        "holdout_contract_version",
        "kind",
        "holdout_id",
        "dataset_version",
        "access",
        "bundle_sha256",
        "inputs",
        "additional_files",
    }
)
_RELEASE_LOCK_KEYS = frozenset({"status", *_TOP_LEVEL_KEYS, "manifest_sha256"})
_INPUT_KEYS = frozenset({"path", "sha256", "count"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise HoldoutContractError(f"holdout file is unreadable: {path}") from exc
    return digest.hexdigest()


def _exact_keys(value: dict[str, object], expected: frozenset[str], label: str) -> None:
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        raise HoldoutContractError(
            f"{label} fields do not match contract; missing={missing}, unknown={unknown}"
        )


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HoldoutContractError(f"{label} must be a non-empty string")
    return value.strip()


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise HoldoutContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _relative_path(value: object, label: str) -> str:
    raw = _text(value, label)
    if "\\" in raw:
        raise HoldoutContractError(f"{label} must use POSIX separators")
    path = PurePosixPath(raw)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise HoldoutContractError(f"{label} must be a safe relative path")
    return path.as_posix()


@dataclass(frozen=True)
class HoldoutInput:
    role: str
    path: str
    sha256: str
    count: int

    def as_json(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "count": self.count}


@dataclass(frozen=True)
class HoldoutManifest:
    path: Path
    kind: HoldoutKind
    holdout_id: str
    dataset_version: str
    access: Literal["public", "restricted"]
    inputs: dict[str, HoldoutInput]
    additional_files: dict[str, str]
    bundle_sha256: str
    manifest_sha256: str

    @property
    def root(self) -> Path:
        return self.path.parent

    @property
    def status(self) -> str:
        """Compatibility label used by diagnostic evaluation paths."""

        return self.kind

    @property
    def files(self) -> dict[str, str]:
        values = {item.path: item.sha256 for item in self.inputs.values()}
        values.update(self.additional_files)
        return values

    def descriptor(self) -> dict[str, object]:
        return {
            "holdout_contract_version": HOLDOUT_CONTRACT_VERSION,
            "kind": self.kind,
            "holdout_id": self.holdout_id,
            "dataset_version": self.dataset_version,
            "access": self.access,
            "bundle_sha256": self.bundle_sha256,
            "manifest_sha256": self.manifest_sha256,
            "inputs": {
                role: item.as_json() for role, item in sorted(self.inputs.items())
            },
            "additional_files": dict(sorted(self.additional_files.items())),
        }

    def release_lock(self) -> dict[str, object]:
        if self.kind != "restricted_holdout" or self.access != "restricted":
            raise HoldoutContractError("only a restricted holdout can be bound for promotion")
        return {"status": "frozen", **self.descriptor()}

    def _relative(self, path: str | Path) -> str:
        original = Path(path)
        candidate = original.resolve()
        try:
            relative = candidate.relative_to(self.root.resolve())
        except ValueError as exc:
            raise HoldoutContractError(
                f"file is outside the holdout root: {candidate}"
            ) from exc
        normalized = relative.as_posix()
        if normalized not in self.files:
            raise HoldoutContractError(
                f"file is not listed in holdout manifest: {normalized}"
            )
        return normalized

    def verify_file(self, path: str | Path) -> Path:
        original = Path(os.path.abspath(path))
        root = Path(os.path.abspath(self.root))
        try:
            original.relative_to(root)
        except ValueError as exc:
            raise HoldoutContractError(
                f"file is outside the holdout root: {original}"
            ) from exc
        current = original
        while current != root:
            if current.is_symlink():
                raise HoldoutContractError(f"holdout path cannot contain a symlink: {original}")
            if current.parent == current:
                break
            current = current.parent
        if root.is_symlink():
            raise HoldoutContractError(
                f"holdout path cannot contain a symlink: {original}"
            )
        candidate = original.resolve()
        relative = self._relative(candidate)
        observed = _sha256(candidate)
        expected = self.files[relative]
        if observed != expected:
            raise HoldoutContractError(
                f"hash mismatch for {relative}: expected {expected}, observed {observed}"
            )
        return candidate

    def verify_role(self, role: str, path: str | Path) -> Path:
        if role not in self.inputs:
            raise HoldoutContractError(f"unknown holdout input role: {role}")
        candidate = self.verify_file(path)
        relative = self._relative(candidate)
        if relative != self.inputs[role].path:
            raise HoldoutContractError(
                f"holdout path is not assigned to role {role}: {relative}"
            )
        return candidate

    def verify_role_count(self, role: str, observed: int) -> None:
        expected = self.inputs[role].count
        if observed != expected:
            raise HoldoutContractError(
                f"holdout role {role} count mismatch: expected {expected}, observed {observed}"
            )

    def provenance(self) -> dict[str, object]:
        return self.descriptor()


def _input(role: str, value: object) -> HoldoutInput:
    if not isinstance(value, dict):
        raise HoldoutContractError(f"holdout input {role} must be an object")
    normalized = {str(key): item for key, item in value.items()}
    _exact_keys(normalized, _INPUT_KEYS, f"holdout input {role}")
    count = normalized["count"]
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise HoldoutContractError(f"holdout input {role} count must be a positive integer")
    return HoldoutInput(
        role=role,
        path=_relative_path(normalized["path"], f"holdout input {role} path"),
        sha256=_digest(normalized["sha256"], f"holdout input {role} digest"),
        count=count,
    )


def _bundle_digest(
    inputs: dict[str, HoldoutInput], additional_files: dict[str, str]
) -> str:
    payload = {
        "inputs": {role: item.as_json() for role, item in sorted(inputs.items())},
        "additional_files": dict(sorted(additional_files.items())),
    }
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def load_holdout_manifest(
    path: str | Path,
    *,
    verify_files: bool = True,
) -> HoldoutManifest:
    """Load either contract branch through one exact-key discriminated parser."""

    original_manifest_path = Path(path)
    if (
        original_manifest_path.is_symlink()
        or original_manifest_path.parent.is_symlink()
        or not original_manifest_path.is_file()
    ):
        raise HoldoutContractError(f"holdout manifest is missing or unsafe: {original_manifest_path}")
    manifest_path = original_manifest_path.resolve()
    if not manifest_path.is_file():
        raise HoldoutContractError(f"holdout manifest is missing or unsafe: {manifest_path}")
    sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.is_symlink():
        raise HoldoutContractError(f"holdout manifest hash sidecar is missing: {sidecar}")
    try:
        sidecar_value = sidecar.read_text(encoding="ascii").strip().split()[0]
    except (OSError, IndexError) as exc:
        raise HoldoutContractError(f"holdout manifest hash sidecar is invalid: {sidecar}") from exc
    _digest(sidecar_value, "holdout manifest sidecar")
    observed_manifest_hash = _sha256(manifest_path)
    if observed_manifest_hash != sidecar_value:
        raise HoldoutContractError(
            "holdout manifest hash mismatch: "
            f"expected {sidecar_value}, observed {observed_manifest_hash}"
        )
    try:
        parsed = load_strict_json_file(manifest_path, label="holdout manifest")
    except StrictJSONError as exc:
        raise HoldoutContractError(str(exc)) from exc
    if not isinstance(parsed, dict):
        raise HoldoutContractError("holdout manifest must be a JSON object")
    raw = {str(key): item for key, item in parsed.items()}
    _exact_keys(raw, _TOP_LEVEL_KEYS, "holdout manifest")
    if raw["holdout_contract_version"] != HOLDOUT_CONTRACT_VERSION:
        raise HoldoutContractError("unsupported holdout contract version")
    kind = raw["kind"]
    if kind not in {"test_fixture", "restricted_holdout"}:
        raise HoldoutContractError("holdout kind must be test_fixture or restricted_holdout")
    access = raw["access"]
    expected_access = "public" if kind == "test_fixture" else "restricted"
    if access != expected_access:
        raise HoldoutContractError(f"holdout kind {kind} requires access={expected_access}")
    raw_inputs = raw["inputs"]
    if not isinstance(raw_inputs, dict) or frozenset(raw_inputs) != frozenset(REQUIRED_INPUT_ROLES):
        raise HoldoutContractError(
            "holdout inputs must contain exactly: " + ", ".join(REQUIRED_INPUT_ROLES)
        )
    inputs = {role: _input(role, raw_inputs[role]) for role in REQUIRED_INPUT_ROLES}
    paths = [item.path for item in inputs.values()]
    if len(set(paths)) != len(paths):
        raise HoldoutContractError("holdout input roles must use distinct paths")
    raw_additional = raw["additional_files"]
    if not isinstance(raw_additional, dict):
        raise HoldoutContractError("holdout additional_files must be an object")
    additional_files: dict[str, str] = {}
    for raw_path, raw_hash in raw_additional.items():
        normalized_path = _relative_path(raw_path, "holdout additional file path")
        if normalized_path in paths:
            raise HoldoutContractError("holdout additional file duplicates an input role path")
        additional_files[normalized_path] = _digest(
            raw_hash, f"holdout additional file {normalized_path} digest"
        )
    computed_bundle = _bundle_digest(inputs, additional_files)
    declared_bundle = _digest(raw["bundle_sha256"], "holdout bundle digest")
    if computed_bundle != declared_bundle:
        raise HoldoutContractError(
            f"holdout bundle digest mismatch: expected {declared_bundle}, observed {computed_bundle}"
        )
    manifest = HoldoutManifest(
        path=manifest_path,
        kind=cast(HoldoutKind, kind),
        holdout_id=_text(raw["holdout_id"], "holdout_id"),
        dataset_version=_text(raw["dataset_version"], "holdout dataset_version"),
        access=cast(Literal["public", "restricted"], access),
        inputs=inputs,
        additional_files=additional_files,
        bundle_sha256=computed_bundle,
        manifest_sha256=observed_manifest_hash,
    )
    if verify_files:
        for relative in manifest.files:
            manifest.verify_file(manifest.root / relative)
    return manifest


def load_restricted_holdout_descriptor(
    path: str | Path,
    *,
    dataset_version: str,
) -> HoldoutManifest:
    manifest = load_holdout_manifest(path, verify_files=False)
    if manifest.kind != "restricted_holdout" or manifest.access != "restricted":
        raise HoldoutContractError("production candidates require a restricted_holdout manifest")
    if manifest.dataset_version != dataset_version:
        raise HoldoutContractError(
            "holdout dataset_version must exactly match the candidate: "
            f"expected {dataset_version!r}, got {manifest.dataset_version!r}"
        )
    return manifest


def validate_restricted_release_lock(
    value: object,
    *,
    dataset_version: str | None = None,
) -> dict[str, object]:
    """Validate the exact metadata snapshot embedded in a candidate identity."""

    if not isinstance(value, dict):
        raise HoldoutContractError("release holdout lock must be an object")
    raw = {str(key): item for key, item in value.items()}
    _exact_keys(raw, _RELEASE_LOCK_KEYS, "release holdout lock")
    if raw["status"] != "frozen":
        raise HoldoutContractError("release holdout lock must have status=frozen")
    if raw["holdout_contract_version"] != HOLDOUT_CONTRACT_VERSION:
        raise HoldoutContractError("release holdout contract version is unsupported")
    if raw["kind"] != "restricted_holdout" or raw["access"] != "restricted":
        raise HoldoutContractError("release holdout lock must be restricted")
    locked_dataset_version = _text(
        raw["dataset_version"], "release holdout dataset_version"
    )
    if dataset_version is not None and locked_dataset_version != dataset_version:
        raise HoldoutContractError(
            "release holdout dataset_version differs from candidate"
        )
    inputs_value = raw["inputs"]
    if (
        not isinstance(inputs_value, dict)
        or frozenset(str(key) for key in inputs_value)
        != frozenset(REQUIRED_INPUT_ROLES)
    ):
        raise HoldoutContractError(
            "release holdout inputs must contain exactly: "
            + ", ".join(REQUIRED_INPUT_ROLES)
        )
    inputs = {
        role: _input(role, inputs_value[role])
        for role in REQUIRED_INPUT_ROLES
    }
    if len({item.path for item in inputs.values()}) != len(inputs):
        raise HoldoutContractError(
            "release holdout input roles must use distinct paths"
        )
    raw_additional = raw["additional_files"]
    if not isinstance(raw_additional, dict):
        raise HoldoutContractError("release holdout additional_files must be an object")
    input_paths = {item.path for item in inputs.values()}
    additional_files: dict[str, str] = {}
    for raw_path, raw_hash in raw_additional.items():
        normalized_path = _relative_path(
            raw_path, "release holdout additional file path"
        )
        if normalized_path in input_paths:
            raise HoldoutContractError(
                "release holdout additional file duplicates an input role path"
            )
        additional_files[normalized_path] = _digest(
            raw_hash,
            f"release holdout additional file {normalized_path} digest",
        )
    declared_bundle = _digest(
        raw["bundle_sha256"], "release holdout bundle digest"
    )
    observed_bundle = _bundle_digest(inputs, additional_files)
    if declared_bundle != observed_bundle:
        raise HoldoutContractError(
            "release holdout bundle digest does not match its locked inputs"
        )
    return {
        "status": "frozen",
        "holdout_contract_version": HOLDOUT_CONTRACT_VERSION,
        "kind": "restricted_holdout",
        "holdout_id": _text(raw["holdout_id"], "release holdout_id"),
        "dataset_version": locked_dataset_version,
        "access": "restricted",
        "bundle_sha256": declared_bundle,
        "manifest_sha256": _digest(
            raw["manifest_sha256"], "release holdout manifest digest"
        ),
        "inputs": {
            role: item.as_json() for role, item in sorted(inputs.items())
        },
        "additional_files": dict(sorted(additional_files.items())),
    }


__all__ = [
    "HOLDOUT_CONTRACT_VERSION",
    "REQUIRED_INPUT_ROLES",
    "HoldoutContractError",
    "HoldoutInput",
    "HoldoutManifest",
    "load_holdout_manifest",
    "load_restricted_holdout_descriptor",
    "validate_restricted_release_lock",
]
