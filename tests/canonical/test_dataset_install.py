from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

from scripts import download_dataset


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _released_inputs(directory: Path) -> Path:
    directory.mkdir()
    values = {
        "sources.csv": "source registry\n",
        "chunks.jsonl": '{"chunk_id":"p-1"}\n',
        "curriculum_catalog.json": '{"catalog_version":"fixture"}\n',
        "source_review.csv": "source review\n",
        "evidence_review.csv": "evidence review\n",
    }
    for name, value in values.items():
        (directory / name).write_text(value, encoding="utf-8")
    source_hashes = {
        "catalog": _sha(directory / "curriculum_catalog.json"),
        "sources": _sha(directory / "sources.csv"),
        "chunks": _sha(directory / "chunks.jsonl"),
        "source_review": _sha(directory / "source_review.csv"),
        "evidence_review": _sha(directory / "evidence_review.csv"),
    }
    (directory / "dataset_manifest.json").write_text(
        json.dumps({"dataset_version": "fixture-v1", "source_hashes": source_hashes}),
        encoding="utf-8",
    )
    return directory


def test_validate_dataset_directory_requires_hashed_review_ledgers(tmp_path: Path) -> None:
    source = _released_inputs(tmp_path / "source")
    assert download_dataset.validate_dataset_directory(source)["dataset_version"] == "fixture-v1"
    (source / "evidence_review.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(download_dataset.DatasetInstallError, match="input hash mismatch"):
        download_dataset.validate_dataset_directory(source)


def test_local_dataset_install_stages_then_publishes_complete_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _released_inputs(tmp_path / "source")
    raw = source / "raw"
    raw.mkdir()
    (raw / "official.pdf").write_bytes(b"immutable-source-bytes")
    target = tmp_path / "installed"
    monkeypatch.setattr(
        sys,
        "argv",
        ["download_dataset", "--source-dir", str(source), "--data-dir", str(target)],
    )

    download_dataset.main()

    assert download_dataset.validate_dataset_directory(target)["dataset_version"] == "fixture-v1"
    assert (target / "raw" / "official.pdf").read_bytes() == b"immutable-source-bytes"
    assert not any(path.name.startswith(".dataset-staging-") for path in tmp_path.iterdir())


def test_archive_extraction_rejects_path_traversal_and_symlinks(tmp_path: Path) -> None:
    traversal = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal, "w") as archive:
        archive.writestr("../outside.txt", "no")
    with pytest.raises(download_dataset.DatasetInstallError, match="unsafe archive member"):
        download_dataset._safe_extract_zip(traversal, tmp_path / "extract", max_bytes=1024)

    symlink = tmp_path / "symlink.zip"
    with zipfile.ZipFile(symlink, "w") as archive:
        info = zipfile.ZipInfo("link")
        info.external_attr = 0o120777 << 16
        archive.writestr(info, "target")
    with pytest.raises(download_dataset.DatasetInstallError, match="symlink"):
        download_dataset._safe_extract_zip(symlink, tmp_path / "extract-link", max_bytes=1024)


def test_remote_dataset_requires_https_pinning_and_explicit_allowlist() -> None:
    with pytest.raises(download_dataset.DatasetInstallError, match="HTTPS"):
        download_dataset._validated_url("http://example.test/dataset.zip", {"example.test"})
    with pytest.raises(download_dataset.DatasetInstallError, match="allowed-host"):
        download_dataset._validated_url("https://example.test/dataset.zip", set())
    with pytest.raises(download_dataset.DatasetInstallError, match="not in the explicit allowlist"):
        download_dataset._validated_url("https://example.test/dataset.zip", {"other.test"})
