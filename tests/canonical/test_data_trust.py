from __future__ import annotations

import csv
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from academic.database import AcademicRepository, build_database
from agent.factory import build_runtime
from ingest.contracts import validate_chunk
from ingest.models import DocumentElement, ParsedDocument
from ingest.pipeline import ingest_sources
from ingest.sources import SOURCE_FIELDS
from scripts import build_all, verify_dataset

FIXTURE_DATA = Path(__file__).with_name("data")
CURRICULUM_SHA = "c" * 64
POLICY_SHA = "d" * 64


def _field_verification(
    *, source_sha256: str, chunk_id: str, page: int, row: int, cell: str, span: str
) -> dict[str, object]:
    return {
        "verification_status": "verified",
        "lineage": {
            "source_sha256": source_sha256,
            "chunk_id": chunk_id,
            "page": page,
            "row": row,
            "cell": cell,
            "span": span,
        },
    }


def _source_row() -> dict[str, str]:
    return {
        "file": "fixture-source.pdf",
        "doc_title": "可信状态测试培养方案",
        "level": "院级",
        "college": "测试学院",
        "cohort": "2024",
        "year": "2024",
        "status": "现行",
        "page_url": "https://example.test/fixture-source.pdf#page=1",
        "file_url": "https://example.test/fixture-source.pdf",
        "collected_at": "2026-01-01",
        "doc_type": "curriculum",
        "topics": "",
        "source_sha256": CURRICULUM_SHA,
        "authenticity_status": "verified",
    }


def _build_dataset(
    tmp_path: Path,
    *,
    chunk_review_status: str = "unverified",
    ledger_decision: str | None = None,
    evidence_decision: str | None = None,
    add_unreferenced_chunk: bool = False,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    sources = tmp_path / "sources.csv"
    with sources.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_FIELDS)
        writer.writeheader()
        writer.writerow(_source_row())
        if add_unreferenced_chunk:
            writer.writerow(
                {
                    "file": "policy-source.pdf",
                    "doc_title": "可信状态测试政策",
                    "level": "校级",
                    "college": "全校",
                    "cohort": "不限",
                    "year": "2024",
                    "status": "现行",
                    "page_url": "https://example.test/policy-source.pdf#page=1",
                    "file_url": "https://example.test/policy-source.pdf",
                    "collected_at": "2026-01-01",
                    "doc_type": "policy",
                    "topics": "transfer",
                    "source_sha256": POLICY_SHA,
                    "authenticity_status": "verified",
                }
            )

    chunks = [
        {
            "chunk_id": "curriculum-evidence",
            "text": "测试专业的专业选修课最低要求为 3 学分，测试课程在第 1 学期开设。",
            "doc_title": "可信状态测试培养方案",
            "article": "培养要求 / 原文件第1页",
            "cohort": "2024",
            "is_table": False,
            "review_status": chunk_review_status,
            "doc_type": "curriculum",
            "topics": [],
            "source_sha256": CURRICULUM_SHA,
            "extraction_quality": "verified",
        }
    ]
    if add_unreferenced_chunk:
        chunks.append(
            {
                "chunk_id": "policy-only-evidence",
                "text": "这是未关联结构化课程或要求的政策正文，适用于转专业申请。",
                "doc_title": "可信状态测试政策",
                "article": "政策正文 / 原文件第2页",
                "cohort": "不限",
                "is_table": False,
                "review_status": chunk_review_status,
                "doc_type": "policy",
                "topics": ["transfer"],
                "source_sha256": POLICY_SHA,
                "extraction_quality": "verified",
            }
        )
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in chunks),
        encoding="utf-8",
    )

    catalog = {
        "catalog_version": "trust-fixture-1",
        "plans": [
            {
                "college": "测试学院",
                "cohort": "2024",
                "major": "测试专业",
                "source_title": "可信状态测试培养方案",
                "modules": [
                    {
                        "name": "专业选修课",
                        "required_credits": 3,
                        "listed_credits": 3,
                        "rule_text": "最低 3 学分",
                        "evidence": {
                            "chunk_id": "curriculum-evidence",
                            "article": "原文件第1页",
                        },
                        "field_verification": {
                            "module": _field_verification(
                                source_sha256=CURRICULUM_SHA,
                                chunk_id="curriculum-evidence",
                                page=1,
                                row=1,
                                cell="A1",
                                span="专业选修课",
                            ),
                            "required_credits": _field_verification(
                                source_sha256=CURRICULUM_SHA,
                                chunk_id="curriculum-evidence",
                                page=1,
                                row=1,
                                cell="B1",
                                span="最低要求为 3 学分",
                            ),
                        },
                    }
                ],
            }
        ],
        "courses": [
            {
                "code": "TST101",
                "name": "测试课程",
                "credits": 3,
                "nature": "选修",
                "semester": "1",
                "department": "测试学院",
                "college": "测试学院",
                "cohort": "2024",
                "major": "测试专业",
                "module": "专业选修课",
                "source_title": "可信状态测试培养方案",
                "page": 1,
                "source_row": 1,
                "evidence": {"chunk_id": "curriculum-evidence"},
                "field_verification": {
                    "module": _field_verification(
                        source_sha256=CURRICULUM_SHA,
                        chunk_id="curriculum-evidence",
                        page=1,
                        row=2,
                        cell="A2",
                        span="专业选修课",
                    ),
                    "credits": _field_verification(
                        source_sha256=CURRICULUM_SHA,
                        chunk_id="curriculum-evidence",
                        page=1,
                        row=2,
                        cell="B2",
                        span="3 学分",
                    ),
                    "semester": _field_verification(
                        source_sha256=CURRICULUM_SHA,
                        chunk_id="curriculum-evidence",
                        page=1,
                        row=2,
                        cell="C2",
                        span="第 1 学期开设",
                    ),
                    "nature": _field_verification(
                        source_sha256=CURRICULUM_SHA,
                        chunk_id="curriculum-evidence",
                        page=1,
                        row=2,
                        cell="D2",
                        span="选修",
                    ),
                },
            }
        ],
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
    aliases_path = tmp_path / "aliases.json"
    aliases_path.write_text(
        json.dumps({"program_aliases": {}, "module_aliases": {}, "course_aliases": {}}),
        encoding="utf-8",
    )

    source_review_path: Path | None = None
    if ledger_decision is not None:
        source_review_path = tmp_path / "source_review.csv"
        with source_review_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "original_title",
                    "corrected_title",
                    "decision",
                    "reviewer",
                    "method",
                    "reviewed_at",
                ),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "original_title": "fixture-source.pdf",
                    "corrected_title": "可信状态测试培养方案",
                    "decision": ledger_decision,
                    "reviewer": "test-reviewer",
                    "method": "manual",
                    "reviewed_at": "2026-01-01T00:00:00+00:00",
                }
            )
            if add_unreferenced_chunk and evidence_decision is None:
                writer.writerow(
                    {
                        "original_title": "policy-source.pdf",
                        "corrected_title": "可信状态测试政策",
                        "decision": ledger_decision,
                        "reviewer": "test-reviewer",
                        "method": "manual",
                        "reviewed_at": "2026-01-01T00:00:00+00:00",
                    }
                )

    evidence_review_path: Path | None = None
    if evidence_decision is not None:
        evidence_review_path = tmp_path / "evidence_review.csv"
        with evidence_review_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "chunk_id",
                    "decision",
                    "scope",
                    "reviewer",
                    "method",
                    "reviewed_at",
                ),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "chunk_id": "curriculum-evidence",
                    "decision": evidence_decision,
                    "scope": "测试专业 / 2024",
                    "reviewer": "test-reviewer",
                    "method": "page-level-manual",
                    "reviewed_at": "2026-01-01T00:00:00+00:00",
                }
            )

    database = tmp_path / "academic.sqlite3"
    build_database(
        database,
        catalog_path=catalog_path,
        sources_path=sources,
        chunks_path=chunks_path,
        aliases_path=aliases_path,
        source_review_path=source_review_path,
        evidence_review_path=evidence_review_path,
    )
    return database


def _trust_statuses(database: Path) -> dict[str, str]:
    with sqlite3.connect(database) as connection:
        return {
            "section": str(
                connection.execute(
                    "SELECT review_status FROM source_sections WHERE chunk_id='curriculum-evidence'"
                ).fetchone()[0]
            ),
            "course": str(connection.execute("SELECT review_status FROM program_courses").fetchone()[0]),
            "requirement": str(
                connection.execute("SELECT review_status FROM requirements").fetchone()[0]
            ),
        }


def test_build_all_cli_consumes_explicit_review_ledgers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "academic.sqlite3"
    manifest_dir = tmp_path / "manifests"
    retrieval_root = tmp_path / "retrieval"
    release_root = tmp_path / "releases"
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
            str(release_root),
            "--retrieval-mode",
            "lexical",
        ],
    )
    build_all.main()
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT count(*) FROM source_sections WHERE review_status='verified'"
        ).fetchone()[0] == 3
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        assert metadata["source_review_sha256"]
        assert metadata["evidence_review_sha256"]
        assert metadata["evidence_state_sha256"]
    manifest = json.loads((manifest_dir / "canonical-ci-fixture-1.json").read_text(encoding="utf-8"))
    assert manifest["source_hashes"]["source_review"]
    assert manifest["source_hashes"]["evidence_review"]


def test_formal_ingest_emits_explicit_review_required_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "policy.pdf").write_bytes(b"fixture")
    sources = tmp_path / "sources.csv"
    row = _source_row() | {
        "file": "policy.pdf",
        "doc_title": "正式导入测试文件",
        "level": "校级",
        "college": "全校",
        "cohort": "不限",
        "page_url": "https://jwc.swufe.edu.cn/policy.pdf",
        "file_url": "https://jwc.swufe.edu.cn/policy.pdf",
    }
    with sources.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_FIELDS)
        writer.writeheader()
        writer.writerow(row)

    def fake_parse(path: Path, *, ocr_provider: object | None = None) -> ParsedDocument:
        del ocr_provider
        return ParsedDocument(
            path=path,
            elements=[DocumentElement(kind="paragraph", text="正式导入内容。", page=1)],
            page_count=1,
        )

    monkeypatch.setattr("ingest.pipeline.parse_document", fake_parse)
    output = tmp_path / "chunks.jsonl"
    report = ingest_sources(sources, raw_dir, output)
    chunk = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert report["chunk_count"] == 1
    assert chunk["review_status"] == "review_required"
    assert validate_chunk(chunk)["review_status"] == "review_required"


def test_untrusted_chunk_cannot_self_certify_and_statuses_follow_evidence(tmp_path: Path) -> None:
    database = _build_dataset(tmp_path, chunk_review_status="verified")
    assert _trust_statuses(database) == {
        "section": "review_required",
        "course": "review_required",
        "requirement": "review_required",
    }


@pytest.mark.parametrize(
    "decision", ["include", "include_ocr", "include_converted", "include_split"]
)
def test_explicitly_allowed_review_ledger_decisions_promote_consistent_status(
    tmp_path: Path, decision: str
) -> None:
    database = _build_dataset(
        tmp_path, chunk_review_status="unverified", ledger_decision=decision
    )
    assert _trust_statuses(database) == {
        "section": "verified",
        "course": "verified",
        "requirement": "verified",
    }


@pytest.mark.parametrize("decision", ["verified", "include"])
def test_evidence_review_ledger_promotes_only_the_named_chunk(
    tmp_path: Path, decision: str
) -> None:
    database = _build_dataset(
        tmp_path,
        chunk_review_status="unverified",
        ledger_decision="include",
        evidence_decision=decision,
        add_unreferenced_chunk=True,
    )
    assert _trust_statuses(database) == {
        "section": "verified",
        "course": "verified",
        "requirement": "verified",
    }
    with sqlite3.connect(database) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        assert metadata["evidence_review_sha256"]
        assert metadata["evidence_state_sha256"]
        assert connection.execute(
            "SELECT review_status FROM source_sections WHERE chunk_id='policy-only-evidence'"
        ).fetchone()[0] == "unverified"


def test_evidence_review_ledger_uses_an_exact_allowlist(tmp_path: Path) -> None:
    database = _build_dataset(
        tmp_path, chunk_review_status="unverified", evidence_decision="include_ocr"
    )
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT review_status FROM source_sections LIMIT 1"
        ).fetchone()[0] == "unverified"
        assert connection.execute(
            "SELECT review_status FROM program_courses LIMIT 1"
        ).fetchone()[0] == "unverified"
        # An untrusted requirement is quarantined from the production
        # projection rather than being exposed as a partially usable row.
        assert connection.execute("SELECT count(*) FROM requirements").fetchone()[0] == 0


def test_runtime_readiness_requires_verified_core_business_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SWUFE_RETRIEVAL_MODE", "lexical")
    unreviewed = build_runtime(_build_dataset(tmp_path / "unreviewed"))
    try:
        ready, reasons = unreviewed.readiness()
        assert not ready
        assert "verified_evidence_missing" in reasons
        assert "verified_policy_evidence_missing" in reasons
        assert "core_business_unanswerable" in reasons
    finally:
        unreviewed.repository.close()

    structured_only = build_runtime(
        _build_dataset(tmp_path / "structured-only", ledger_decision="include")
    )
    try:
        ready, reasons = structured_only.readiness()
        assert not ready
        assert "verified_policy_evidence_missing" in reasons
    finally:
        structured_only.repository.close()

    reviewed = build_runtime(
        _build_dataset(
            tmp_path / "reviewed", ledger_decision="include", add_unreferenced_chunk=True
        )
    )
    try:
        assert reviewed.readiness() == (True, ())
    finally:
        reviewed.repository.close()



def test_readiness_requires_every_active_program_to_be_answerable(
    tmp_path: Path,
) -> None:
    database = _build_dataset(
        tmp_path,
        ledger_decision="include",
        add_unreferenced_chunk=True,
    )
    with sqlite3.connect(database) as connection:
        source_id = connection.execute(
            "SELECT source_id FROM sources WHERE status='现行' ORDER BY source_id LIMIT 1"
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO programs VALUES (?, ?, ?, ?, ?)",
            ("program-incomplete", "未完成专业", "测试学院", 2024, source_id),
        )

    repository = AcademicRepository(database)
    try:
        ready, reasons = repository.evidence_readiness()
    finally:
        repository.close()

    assert not ready
    assert "program_readiness_incomplete" in reasons
    assert "program_unanswerable:program-incomplete" in reasons
    assert "core_business_unanswerable" not in reasons

def test_dataset_verifier_allows_only_explicit_review_required_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = _build_dataset(
        tmp_path,
        chunk_review_status="review_required",
        add_unreferenced_chunk=True,
    )
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            verify_dataset.CHECKS["orphan_requirement_evidence"]
        ).fetchone()[0] == 0

    monkeypatch.setattr(sys, "argv", ["verify_dataset", "--database", str(database)])
    with pytest.raises(SystemExit, match="review_required_requirement"):
        verify_dataset.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_dataset",
            "--database",
            str(database),
            "--allow-review-required-requirements",
        ],
    )
    verify_dataset.main()
