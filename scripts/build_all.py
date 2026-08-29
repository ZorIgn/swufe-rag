"""Build a verified database/index bundle and publish it as one release unit."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from pathlib import Path

from academic.database import AcademicRepository, build_database
from eval.holdout import HoldoutContractError, load_restricted_holdout_descriptor
from retrieval.index import build_retrieval_index, validate_retrieval_artifact
from scripts.verify_dataset import DatasetVerificationError, verify_database
from storage.provenance import git_provenance as measure_git_provenance
from storage.release import (
    ReleaseError,
    atomic_write_json,
    build_release_manifest,
    discard_staging,
    make_staging_directory,
    publish_release,
    sha256_directory,
    sha256_file,
)


class BuildError(RuntimeError):
    """Raised before a staged dataset is allowed to become a release."""




def _source_count(path: Path) -> int:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _load_holdout_lock(
    path: Path | None, *, dataset_version: str, require_holdout: bool
) -> dict[str, object]:
    """Bind a strict restricted descriptor without reading protected labels."""

    if path is None:
        if require_holdout:
            raise BuildError("production candidate requires --holdout-manifest")
        return {"status": "not_supplied"}
    try:
        manifest = load_restricted_holdout_descriptor(
            path,
            dataset_version=dataset_version,
        )
        return manifest.release_lock()
    except HoldoutContractError as exc:
        raise BuildError(f"holdout manifest is not promotion-eligible: {exc}") from exc


def _copy_file_atomically(source: Path, target: Path) -> None:
    """Materialize an explicitly requested compatibility copy after validation."""

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(raw_temporary)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _copy_tree_atomically(source: Path, target: Path) -> None:
    """Materialize a non-authoritative legacy layout without in-place writes."""

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    staged = temporary / target.name
    try:
        shutil.copytree(source, staged)
        if target.exists():
            # A caller that explicitly asks for the legacy layout must not have
            # a different release silently substituted underneath it.
            expected = validate_retrieval_artifact(source)
            existing = validate_retrieval_artifact(target)
            if expected != existing:
                raise BuildError(f"refusing to overwrite compatibility artifact: {target}")
            return
        os.replace(staged, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _write_compatibility_manifest(path: Path, manifest: dict[str, object]) -> None:
    """Write a convenience pointer; the active release remains authoritative."""

    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BuildError(f"compatibility manifest is unreadable: {path}") from exc
        if existing != manifest:
            raise BuildError(f"refusing to overwrite compatibility manifest: {path}")
        return
    atomic_write_json(path, manifest)


def _compatibility_outputs(
    *,
    database: Path | None,
    retrieval_root: Path | None,
    manifest_dir: Path | None,
    staged_database: Path,
    staged_retrieval: Path,
    dataset_version: str,
    dataset_manifest: dict[str, object],
) -> None:
    """Support explicit legacy paths without making them the release contract."""

    if database is not None:
        _copy_file_atomically(staged_database, database)
    if retrieval_root is not None:
        _copy_tree_atomically(staged_retrieval, retrieval_root / dataset_version)
    if manifest_dir is not None:
        _write_compatibility_manifest(manifest_dir / f"{dataset_version}.json", dataset_manifest)


def _build_dataset_manifest(
    *,
    report: dict[str, object],
    source_count: int,
    page_count: int,
    retrieval: dict[str, object],
    retrieval_mode: str,
    holdout: dict[str, object],
) -> dict[str, object]:
    return {
        "dataset_version": str(report["dataset_version"]),
        "schema_version": report["schema_version"],
        "parser_version": report["parser_version"],
        "source_count": source_count,
        "source_hashes": {
            "catalog": report["catalog_sha256"],
            "sources": report["sources_sha256"],
            "chunks": report["chunks_sha256"],
            "source_review": report["source_review_sha256"],
            "evidence_review": report["evidence_review_sha256"],
        },
        "evidence_state_sha256": report["evidence_state_sha256"],
        "page_count": page_count,
        "chunk_count": report["chunk_count"],
        "program_count": report["program_count"],
        "course_count": report["offering_count"],
        "requirement_count": report["requirement_count"],
        "quarantined_requirement_count": report["quarantined_requirement_count"],
        "reconciliation_contract": report["reconciliation_contract"],
        "reconciliation_counts": report["reconciliation_counts"],
        "source_authenticity_counts": report["source_authenticity_counts"],
        "extraction_quality_counts": report["extraction_quality_counts"],
        "field_verification_counts": report["field_verification_counts"],
        "retrieval_mode": retrieval_mode,
        "embedding_model": retrieval["embedding_model"],
        "embedding_dimension": retrieval["embedding_dimension"],
        "reranker_model": retrieval["reranker_model"],
        "index_sha256": retrieval["index_sha256"],
        "retrieval_manifest": "retrieval/" + str(report["dataset_version"]) + "/retrieval_manifest.json",
        "holdout": holdout,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    # The authoritative output is a content-addressed release.  Explicit paths
    # below are compatibility materialisations for older local commands/tests.
    parser.add_argument("--database", type=Path, help="optional compatibility database output")
    parser.add_argument("--catalog", type=Path, default=Path("data/curriculum_catalog.json"))
    parser.add_argument("--sources", type=Path, default=Path("data/sources.csv"))
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.jsonl"))
    parser.add_argument("--aliases", type=Path, default=Path("config/entity_aliases.json"))
    parser.add_argument("--source-root", type=Path, help="explicit raw-source root used for byte hashes")
    parser.add_argument(
        "--source-review",
        type=Path,
        default=Path("data/source_review.csv"),
        help="independent source-authenticity reviewer ledger",
    )
    parser.add_argument(
        "--evidence-review",
        type=Path,
        default=Path("data/evidence_review.csv"),
        help="independent chunk/field evidence reviewer ledger",
    )
    parser.add_argument("--manifest-dir", type=Path, help="optional compatibility manifest directory")
    parser.add_argument("--retrieval-root", type=Path, help="optional compatibility retrieval root")
    parser.add_argument("--release-root", type=Path, default=Path("artifacts/releases"))
    parser.add_argument(
        "--release-tier",
        choices=("candidate", "production"),
        default="candidate",
        help="builds are candidates; production is an explicit signed promotion",
    )
    parser.add_argument(
        "--holdout-manifest",
        type=Path,
        help="frozen holdout lock bound into candidate evaluation provenance",
    )
    parser.add_argument(
        "--allow-review-required-requirements",
        action="store_true",
        help="candidate-only exception for manually reviewed requirements; never permits unverified rows",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=("lexical", "hybrid"),
        default=os.getenv("SWUFE_RETRIEVAL_MODE", "hybrid"),
    )
    parser.add_argument(
        "--embedding-model", default=os.getenv("SWUFE_EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")
    )
    parser.add_argument(
        "--reranker-model", default=os.getenv("SWUFE_RERANKER_MODEL", "BAAI/bge-reranker-base")
    )
    args = parser.parse_args()
    if args.release_tier == "production":
        raise SystemExit(
            "direct production builds are disabled: build an immutable candidate, "
            "run both holdout evaluations, create a signed attestation, then use "
            "python -m scripts.promote_release"
        )

    git_provenance = measure_git_provenance()
    embedding_model_path = Path(args.embedding_model)
    reranker_model_path = Path(args.reranker_model)
    embedding_model_sha256 = (
        sha256_directory(embedding_model_path)
        if args.retrieval_mode == "hybrid" and embedding_model_path.is_dir()
        else None
    )
    reranker_model_sha256 = (
        sha256_directory(reranker_model_path)
        if args.retrieval_mode == "hybrid" and reranker_model_path.is_dir()
        else None
    )

    staging = make_staging_directory(args.release_root)
    try:
        staged_database = staging / "academic.sqlite3"
        report = build_database(
            staged_database,
            catalog_path=args.catalog,
            sources_path=args.sources,
            chunks_path=args.chunks,
            aliases_path=args.aliases,
            source_review_path=args.source_review,
            evidence_review_path=args.evidence_review,
            source_root=args.source_root,
        )
        try:
            verification = verify_database(staged_database)
        except DatasetVerificationError as exc:
            raise BuildError("staged database cannot be verified") from exc
        allowed = {"review_required_requirement"} if args.allow_review_required_requirements else set()
        failures = {name: value for name, value in verification.items() if value and name not in allowed}
        if failures:
            raise BuildError(f"staged database verification failed: {failures}")

        dataset_version = str(report["dataset_version"])
        repository = AcademicRepository(staged_database)
        try:
            page_row = repository._one(  # noqa: SLF001 - build-time integrity aggregate
                "SELECT count(DISTINCT source_id || ':' || physical_page) "
                "FROM source_sections WHERE physical_page IS NOT NULL"
            )
            if page_row is None:
                raise BuildError("staged database page aggregate is unavailable")
            page_count = int(page_row[0])
            retrieval = build_retrieval_index(
                list(repository.retrieval_documents()),
                dataset_version=dataset_version,
                source_hash=str(report["evidence_state_sha256"]),
                output_root=staging / "retrieval",
                mode=args.retrieval_mode,
                embedding_model=args.embedding_model,
                reranker_model=args.reranker_model,
                embedding_model_sha256=embedding_model_sha256,
                reranker_model_sha256=reranker_model_sha256,
            )
        finally:
            repository.close()
        staged_retrieval = staging / "retrieval" / dataset_version
        retrieval_manifest = validate_retrieval_artifact(staged_retrieval)
        holdout = _load_holdout_lock(
            args.holdout_manifest,
            dataset_version=dataset_version,
            require_holdout=False,
        )
        dataset_manifest = _build_dataset_manifest(
            report=report,
            source_count=_source_count(args.sources),
            page_count=page_count,
            retrieval=retrieval,
            retrieval_mode=args.retrieval_mode,
            holdout=holdout,
        )
        atomic_write_json(staging / "dataset_manifest.json", dataset_manifest)
        database_sha256 = sha256_file(staged_database)
        identity = {
            "dataset_version": dataset_version,
            "schema_version": str(report["schema_version"]),
            "database_sha256": database_sha256,
            "source_hashes": dataset_manifest["source_hashes"],
            "evidence_state_sha256": report["evidence_state_sha256"],
            "retrieval_manifest_sha256": sha256_file(staged_retrieval / "retrieval_manifest.json"),
            "retrieval_mode": args.retrieval_mode,
            "embedding_model": retrieval_manifest["embedding_model"],
            "embedding_model_sha256": retrieval_manifest.get("embedding_model_sha256"),
            "reranker_model": retrieval_manifest["reranker_model"],
            "reranker_model_sha256": retrieval_manifest.get("reranker_model_sha256"),
            "holdout": holdout,
            "release_tier": "candidate",
            "git_commit": git_provenance["commit"],
            "git_provenance": git_provenance,
        }
        payload = {
            "dataset_manifest": "dataset_manifest.json",
            "database": {"path": "academic.sqlite3", "sha256": database_sha256},
            "retrieval": {
                "root": "retrieval",
                "dataset_version": dataset_version,
                "manifest": "retrieval/" + dataset_version + "/retrieval_manifest.json",
            },
            "holdout": holdout,
            "verification": verification,
            "provenance": {
                "git": git_provenance,
            },
        }
        release_manifest = build_release_manifest(
            identity=identity, payload=payload, staging_directory=staging
        )

        # Explicit compatibility paths are copied from the fully validated
        # staging tree.  They are never the source of runtime readiness.
        _compatibility_outputs(
            database=args.database,
            retrieval_root=args.retrieval_root,
            manifest_dir=args.manifest_dir,
            staged_database=staged_database,
            staged_retrieval=staged_retrieval,
            dataset_version=dataset_version,
            dataset_manifest=dataset_manifest,
        )
        published = publish_release(
            staging,
            releases_root=args.release_root,
            manifest=release_manifest,
            activate=False,
        )
    except (BuildError, ReleaseError, OSError, ValueError) as exc:
        if staging.exists():
            discard_staging(staging)
        raise SystemExit(f"release build failed: {exc}") from exc
    print(
        json.dumps(
            {
                "release_id": published.release_id,
                "release_directory": str(published.directory),
                "active_pointer": str(published.active_pointer) if published.active_pointer else None,
                "dataset_manifest": str(published.directory / "dataset_manifest.json"),
                "database": str(published.directory / "academic.sqlite3"),
                "retrieval": str(published.directory / "retrieval" / dataset_version),
                "release_tier": "candidate",
                "holdout": holdout,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
