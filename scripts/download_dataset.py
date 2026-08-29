"""Stage and verify an externally distributed canonical dataset package.

Data installation is deliberately a release operation, not ``urlretrieve`` +
``unpack_archive``.  An archive must be pinned by SHA-256, use an explicitly
allowed HTTPS host, remain inside its staging root, and contain the review
ledgers plus the dataset manifest that binds every input hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse
from urllib.request import Request, urlopen

REQUIRED = (
    "sources.csv",
    "chunks.jsonl",
    "curriculum_catalog.json",
    "source_review.csv",
    "evidence_review.csv",
    "dataset_manifest.json",
)
_INPUT_HASH_KEYS = {
    "catalog": "curriculum_catalog.json",
    "sources": "sources.csv",
    "chunks": "chunks.jsonl",
    "source_review": "source_review.csv",
    "evidence_review": "evidence_review.csv",
}
_CHUNK_BYTES = 1024 * 1024


class DatasetInstallError(RuntimeError):
    """A package cannot be safely installed as a canonical input dataset."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _safe_file(root: Path, relative: str) -> Path:
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise DatasetInstallError(f"dataset member escapes staging root: {relative!r}") from exc
    if candidate.is_symlink() or not candidate.is_file():
        raise DatasetInstallError(f"dataset member is missing, not a file, or a symlink: {relative}")
    return candidate


def _validate_manifest(root: Path) -> dict[str, object]:
    manifest_path = _safe_file(root, "dataset_manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetInstallError("dataset_manifest.json is unreadable") from exc
    if not isinstance(manifest, dict):
        raise DatasetInstallError("dataset_manifest.json must be a JSON object")
    if not isinstance(manifest.get("dataset_version"), str) or not str(
        manifest["dataset_version"]
    ).strip():
        raise DatasetInstallError("dataset_manifest.json needs a non-empty dataset_version")
    source_hashes = manifest.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise DatasetInstallError("dataset_manifest.json needs source_hashes")
    for key, filename in _INPUT_HASH_KEYS.items():
        expected = str(source_hashes.get(key) or "").strip().lower()
        if not _is_sha256(expected):
            raise DatasetInstallError(f"dataset manifest has no valid SHA-256 for {key}")
        observed = _sha256(_safe_file(root, filename))
        if observed != expected:
            raise DatasetInstallError(
                f"dataset input hash mismatch for {filename}: expected {expected}, observed {observed}"
            )
    return manifest


def validate_dataset_directory(directory: str | Path) -> dict[str, object]:
    """Verify required files, symlink safety, ledgers and manifest input hashes."""

    root = Path(directory)
    if not root.is_dir():
        raise DatasetInstallError(f"dataset directory is missing: {root}")
    for name in REQUIRED:
        _safe_file(root, name)
    return _validate_manifest(root)


def _copy_required(source: Path, staging: Path) -> None:
    validate_dataset_directory(source)
    for name in REQUIRED:
        shutil.copy2(_safe_file(source, name), staging / name)
    raw_source = source / "raw"
    if raw_source.exists():
        if raw_source.is_symlink() or not raw_source.is_dir():
            raise DatasetInstallError("optional raw source directory must be a real directory")
        for candidate in raw_source.rglob("*"):
            if candidate.is_symlink():
                raise DatasetInstallError(f"optional raw source directory contains a symlink: {candidate}")
        shutil.copytree(raw_source, staging / "raw")


def _validated_url(url: str, allowed_hosts: set[str]) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not host:
        raise DatasetInstallError("dataset URL must use HTTPS and contain a hostname")
    if not allowed_hosts:
        raise DatasetInstallError("remote dataset download requires at least one --allowed-host")
    if host not in allowed_hosts:
        raise DatasetInstallError(
            f"dataset URL host {host!r} is not in the explicit allowlist: {sorted(allowed_hosts)!r}"
        )
    return url


def _download(
    url: str,
    destination: Path,
    *,
    expected_sha256: str,
    max_bytes: int,
) -> None:
    digest = hashlib.sha256()
    total = 0
    request = Request(url, headers={"User-Agent": "swufe-rag-dataset-installer/1"})
    try:
        with urlopen(request, timeout=30) as response, destination.open("wb") as output:  # nosec B310
            raw_length = response.headers.get("Content-Length")
            if raw_length is not None and int(raw_length) > max_bytes:
                raise DatasetInstallError("dataset response Content-Length exceeds --max-bytes")
            for block in iter(lambda: response.read(_CHUNK_BYTES), b""):
                total += len(block)
                if total > max_bytes:
                    raise DatasetInstallError("dataset response exceeds --max-bytes")
                digest.update(block)
                output.write(block)
    except DatasetInstallError:
        raise
    except OSError as exc:
        raise DatasetInstallError("dataset download failed") from exc
    observed = digest.hexdigest()
    if observed != expected_sha256:
        raise DatasetInstallError(
            f"download SHA-256 mismatch: expected {expected_sha256}, observed {observed}"
        )


def _safe_extract_zip(archive: Path, destination: Path, *, max_bytes: int) -> None:
    """Extract a ZIP only after validating every member path and aggregate size."""

    try:
        package = zipfile.ZipFile(archive)
    except (OSError, zipfile.BadZipFile) as exc:
        raise DatasetInstallError("dataset package must be a readable ZIP archive") from exc
    with package:
        members = package.infolist()
        total = 0
        validated: list[tuple[zipfile.ZipInfo, Path]] = []
        for member in members:
            raw = member.filename.replace("\\", "/")
            path = PurePosixPath(raw)
            if not raw or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
                raise DatasetInstallError(f"unsafe archive member path: {member.filename!r}")
            # Unix symlink bits in ZIP metadata are not safe to recreate.
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise DatasetInstallError(f"archive contains a symlink: {member.filename!r}")
            if member.is_dir():
                continue
            total += member.file_size
            if total > max_bytes:
                raise DatasetInstallError("expanded dataset archive exceeds --max-bytes")
            target = destination.joinpath(*path.parts)
            try:
                target.resolve().relative_to(destination.resolve())
            except ValueError as exc:
                raise DatasetInstallError(f"archive member escapes staging root: {raw!r}") from exc
            validated.append((member, target))
        for member, target in validated:
            target.parent.mkdir(parents=True, exist_ok=True)
            with package.open(member, "r") as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=_CHUNK_BYTES)


def _archive_dataset_root(staging: Path) -> Path:
    """Accept direct files or exactly one top-level folder, never a guessed tree."""

    if all((staging / name).is_file() for name in REQUIRED):
        return staging
    children = [item for item in staging.iterdir() if not item.name.startswith(".")]
    if len(children) == 1 and children[0].is_dir() and all(
        (children[0] / name).is_file() for name in REQUIRED
    ):
        return children[0]
    raise DatasetInstallError("archive must contain one dataset root with every required release file")


def _install_atomically(staging_data: Path, target: Path) -> None:
    """Install a verified package only into a previously absent target directory."""

    target_parent = target.parent
    if target.exists():
        raise DatasetInstallError(
            f"target data directory already exists: {target}; choose a new --data-dir instead of overwrite"
        )
    target_parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging_data, target)


def _positive_size(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--max-bytes must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("--max-bytes must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/released"))
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--source-dir", type=Path)
    source.add_argument("--url", default=os.getenv("SWUFE_DATASET_URL"))
    parser.add_argument("--expected-sha256", default=os.getenv("SWUFE_DATASET_SHA256"))
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[host.strip() for host in os.getenv("SWUFE_DATASET_ALLOWED_HOSTS", "").split(",") if host.strip()],
        help="repeatable exact HTTPS hostname allowlist for --url",
    )
    parser.add_argument("--max-bytes", type=_positive_size, default=2 * 1024 * 1024 * 1024)
    args = parser.parse_args()
    if args.source_dir is None and not args.url:
        raise SystemExit("dataset unavailable; configure --source-dir or --url with an explicit SHA-256")
    if args.data_dir.exists():
        raise SystemExit(
            f"dataset target already exists: {args.data_dir}; use a new --data-dir to preserve the prior release"
        )
    staging_parent = args.data_dir.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".dataset-staging-", dir=staging_parent))
    try:
        staged_data = staging / "dataset"
        staged_data.mkdir()
        if args.source_dir is not None:
            _copy_required(args.source_dir, staged_data)
        else:
            expected = str(args.expected_sha256 or "").strip().lower()
            if not _is_sha256(expected):
                raise DatasetInstallError("remote dataset download requires --expected-sha256")
            url = _validated_url(str(args.url), {str(host).lower() for host in args.allowed_host})
            archive = staging / "dataset.zip"
            _download(url, archive, expected_sha256=expected, max_bytes=args.max_bytes)
            extracted = staging / "extracted"
            extracted.mkdir()
            _safe_extract_zip(archive, extracted, max_bytes=args.max_bytes)
            root = _archive_dataset_root(extracted)
            _copy_required(root, staged_data)
        manifest = validate_dataset_directory(staged_data)
        _install_atomically(staged_data, args.data_dir)
    except DatasetInstallError as exc:
        raise SystemExit(f"dataset installation failed: {exc}") from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(
        json.dumps(
            {
                "dataset": str(args.data_dir.resolve()),
                "dataset_version": manifest["dataset_version"],
                "manifest_sha256": _sha256(args.data_dir / "dataset_manifest.json"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
