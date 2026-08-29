from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from agent.factory import build_runtime
from retrieval.index import RetrievalArtifactError, build_retrieval_index, load_manifest
from scripts import build_all
from storage.release import (
    ReleaseError,
    build_release_manifest,
    load_active_release,
    make_staging_directory,
    publish_release,
    sha256_file,
    validate_release_directory,
)

FIXTURE_DATA = Path(__file__).parent / "data"


def _manifest(staging: Path, *, label: str) -> dict[str, object]:
    (staging / "payload.txt").write_text(label, encoding="utf-8")
    return build_release_manifest(
        identity={"label": label, "contract": "test"},
        payload={"label": label},
        staging_directory=staging,
    )


def test_release_publish_is_immutable_and_preserves_previous_active_pointer(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    first_staging = make_staging_directory(root)
    first = publish_release(
        first_staging,
        releases_root=root,
        manifest=_manifest(first_staging, label="first"),
        activate=True,
    )
    pointer_before = (root / "active.json").read_bytes()
    directory, manifest = load_active_release(root, require_attestation=False)
    assert directory == first.directory
    assert manifest["release_id"] == first.release_id

    broken_staging = make_staging_directory(root)
    broken_manifest = _manifest(broken_staging, label="broken")
    (broken_staging / "payload.txt").write_text("tampered-after-manifest", encoding="utf-8")
    with pytest.raises(ReleaseError, match="hashes"):
        publish_release(
            broken_staging,
            releases_root=root,
            manifest=broken_manifest,
            activate=True,
        )
    assert (root / "active.json").read_bytes() == pointer_before
    assert validate_release_directory(first.directory)["release_id"] == first.release_id


def test_retrieval_artifact_is_atomic_immutable_and_hash_verified(tmp_path: Path) -> None:
    root = tmp_path / "retrieval"
    documents = [
        {"chunk_id": "p-1", "text": "转专业政策", "review_status": "verified"},
        {"chunk_id": "p-2", "text": "英语免修政策", "review_status": "verified"},
    ]
    built = build_retrieval_index(
        documents,
        dataset_version="fixture-v1",
        source_hash="a" * 64,
        output_root=root,
        mode="lexical",
    )
    directory, loaded = load_manifest(root, "fixture-v1")
    assert loaded["documents_sha256"] == built["documents_sha256"]
    assert not any(path.name.startswith(".staging-") for path in root.iterdir())

    # Same bytes may reuse an immutable target, but a divergent corpus may not
    # silently overwrite a released dataset version.
    reused = build_retrieval_index(
        documents,
        dataset_version="fixture-v1",
        source_hash="a" * 64,
        output_root=root,
        mode="lexical",
    )
    assert reused["directory"] == str(directory)
    with pytest.raises(RetrievalArtifactError, match="overwrite immutable"):
        build_retrieval_index(
            documents + [{"chunk_id": "p-3", "text": "考试政策", "review_status": "verified"}],
            dataset_version="fixture-v1",
            source_hash="a" * 64,
            output_root=root,
            mode="lexical",
        )
    (directory / "documents.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RetrievalArtifactError, match="not JSON|hash mismatch"):
        load_manifest(root, "fixture-v1")


def test_build_all_publishes_candidate_release_and_explicit_compatibility_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    database = tmp_path / "compatibility.sqlite3"
    releases = tmp_path / "releases"
    manifest_dir = tmp_path / "manifests"
    retrieval_root = tmp_path / "retrieval"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_all",
            "--database",
            str(database),
            "--catalog",
            str(FIXTURE_DATA / "catalog.json"),
            "--sources",
            str(FIXTURE_DATA / "sources.csv"),
            "--chunks",
            str(FIXTURE_DATA / "chunks.jsonl"),
            "--aliases",
            str(FIXTURE_DATA / "aliases.json"),
            "--source-review",
            str(FIXTURE_DATA / "source_review.csv"),
            "--evidence-review",
            str(FIXTURE_DATA / "evidence_review.csv"),
            "--manifest-dir",
            str(manifest_dir),
            "--retrieval-root",
            str(retrieval_root),
            "--release-root",
            str(releases),
            "--retrieval-mode",
            "lexical",
        ],
    )

    build_all.main()

    result = json.loads(capsys.readouterr().out)
    directory = Path(str(result["release_directory"]))
    assert result["release_tier"] == "candidate"
    assert result["active_pointer"] is None
    assert database.is_file()
    assert (retrieval_root / "canonical-ci-fixture-1" / "retrieval_manifest.json").is_file()
    assert (manifest_dir / "canonical-ci-fixture-1.json").is_file()
    assert not (releases / "active.json").exists()
    manifest = validate_release_directory(directory)
    assert manifest["payload"]["database"]["sha256"] == sha256_file(directory / "academic.sqlite3")
    assert manifest["payload"]["holdout"]["status"] == "not_supplied"


def test_runtime_uses_verified_active_release_instead_of_legacy_default(
    canonical_runtime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "releases"
    staging = make_staging_directory(root)
    source_database = canonical_runtime.repository.path
    staged_database = staging / "academic.sqlite3"
    shutil.copy2(source_database, staged_database)
    documents = list(canonical_runtime.repository.retrieval_documents())
    dataset_version = canonical_runtime.repository.metadata()["dataset_version"]
    build_retrieval_index(
        documents,
        dataset_version=dataset_version,
        source_hash=canonical_runtime.repository.metadata()["evidence_state_sha256"],
        output_root=staging / "retrieval",
        mode="lexical",
    )
    (staging / "dataset_manifest.json").write_text(
        json.dumps({"dataset_version": dataset_version, "retrieval_mode": "lexical"}),
        encoding="utf-8",
    )
    release = publish_release(
        staging,
        releases_root=root,
        manifest=build_release_manifest(
            identity={"dataset_version": dataset_version, "database": sha256_file(staged_database)},
            payload={
                "database": {"path": "academic.sqlite3", "sha256": sha256_file(staged_database)},
                "retrieval": {"root": "retrieval", "dataset_version": dataset_version},
                "dataset_manifest": "dataset_manifest.json",
            },
            staging_directory=staging,
        ),
        activate=True,
    )
    monkeypatch.setenv("SWUFE_RELEASE_ROOT", str(root))
    monkeypatch.setenv("SWUFE_RETRIEVAL_MODE", "lexical")
    monkeypatch.setenv("SWUFE_ALLOW_UNATTESTED_ACTIVE", "1")
    runtime = build_runtime()
    try:
        assert runtime.repository.path == release.directory / "academic.sqlite3"
        assert runtime.readiness() == (True, ())
    finally:
        runtime.repository.close()
