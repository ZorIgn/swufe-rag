from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from academic.database import SCHEMA, AcademicRepository, build_database
from academic.tools import AcademicTools
from evidence.models import (
    ClaimSpan,
    ClaimValidation,
    DerivedFact,
    Evidence,
    EvidencePacket,
    Fact,
    FinalAnswer,
    Provenance,
)
from generation.validator import ClaimValidator
from query.schemas import RetrievePolicyArgs, RetrievePolicyOperation


def _insert_source(
    connection: sqlite3.Connection,
    source_id: str,
    *,
    title: str,
    status: str,
    effective_from: str,
    effective_to: str | None = None,
    supersedes: str | None = None,
) -> None:
    connection.execute(
        "INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            source_id,
            title,
            "校级",
            "test-college",
            "2024",
            2,
            effective_from,
            effective_from,
            effective_to,
            supersedes,
            status,
            "https://example.test/policy",
            "https://example.test/policy.pdf",
            "source-hash",
            "2026-01-01",
        ),
    )


def _insert_section(
    connection: sqlite3.Connection,
    chunk_id: str,
    source_id: str,
    *,
    article: str,
    text: str,
    page: int,
) -> None:
    connection.execute(
        "INSERT INTO source_sections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            chunk_id,
            source_id,
            article,
            text,
            page,
            0,
            "test-parser",
            "2026-01-01T00:00:00+00:00",
            1.0,
            "verified",
        ),
    )


def _policy_repository(tmp_path: Path) -> AcademicRepository:
    path = tmp_path / "policy.sqlite3"
    connection = sqlite3.connect(path)
    try:
        connection.executescript(SCHEMA)
        _insert_source(
            connection,
            "old",
            title="旧版上限规定",
            status="历史",
            effective_from="2024-01-01",
            effective_to="2024-12-31",
        )
        _insert_section(
            connection,
            "old-section",
            "old",
            article="修读上限 / 原文件第1页",
            text="学生修读上限为 20 学分。",
            page=1,
        )
        _insert_source(
            connection,
            "current",
            title="现行上限规定",
            status="现行",
            effective_from="2025-01-01",
            supersedes="old",
        )
        _insert_section(
            connection,
            "current-section",
            "current",
            article="修读上限 / 原文件第1页",
            text="学生修读上限为 30 学分。",
            page=1,
        )
        _insert_source(
            connection,
            "conflict-a",
            title="同级规定甲",
            status="现行",
            effective_from="2025-01-01",
        )
        _insert_section(
            connection,
            "conflict-a-section",
            "conflict-a",
            article="专业选修要求 / 原文件第1页",
            text="专业选修最低要求为 3 学分。",
            page=1,
        )
        _insert_source(
            connection,
            "conflict-b",
            title="同级规定乙",
            status="现行",
            effective_from="2025-01-01",
        )
        _insert_section(
            connection,
            "conflict-b-section",
            "conflict-b",
            article="专业选修要求 / 原文件第2页",
            text="专业选修最低要求为 4 学分。",
            page=2,
        )
        connection.commit()
    finally:
        connection.close()
    return AcademicRepository(path)


def _evidence() -> Evidence:
    return Evidence(
        evidence_id="e1",
        source_id="source",
        chunk_id="chunk",
        title="测试来源",
        quote="课程为 3 学分。",
        provenance=Provenance(
            record_id="record",
            source_id="source",
            chunk_id="chunk",
            physical_page=1,
            parser_version="test",
            extracted_at=datetime.now(timezone.utc),
            confidence=1.0,
            review_status="verified",
        ),
    )


def test_build_persists_source_version_metadata(tmp_path: Path) -> None:
    sources = tmp_path / "sources.csv"
    with sources.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "file",
                "doc_title",
                "level",
                "college",
                "cohort",
                "year",
                "status",
                "page_url",
                "file_url",
                "authority_level",
                "published_at",
                "effective_from",
                "effective_to",
                "supersedes_source_id",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "file": "missing.pdf",
                "doc_title": "版本来源",
                "level": "院级",
                "college": "测试学院",
                "cohort": "2024",
                "year": "2024",
                "status": "现行",
                "page_url": "https://example.test/page",
                "file_url": "https://example.test/file",
                "authority_level": "7",
                "published_at": "2024-01-02",
                "effective_from": "2024-02-01",
                "effective_to": "2025-01-31",
                "supersedes_source_id": "old-source-id",
            }
        )
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps({"catalog_version": "test", "plans": [], "courses": []}),
        encoding="utf-8",
    )
    database = tmp_path / "built.sqlite3"
    build_database(
        database,
        catalog_path=catalog,
        sources_path=sources,
        chunks_path=tmp_path / "missing.jsonl",
        aliases_path=tmp_path / "missing-aliases.json",
    )

    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT authority_level, published_at, effective_from, effective_to, supersedes_source_id "
            "FROM sources"
        ).fetchone()
    assert row == (7, "2024-01-02", "2024-02-01", "2025-01-31", "old-source-id")


def test_policy_search_prefers_current_version_and_supports_historical_as_of(
    tmp_path: Path,
) -> None:
    repository = _policy_repository(tmp_path)
    try:
        assert [row["source_id"] for row in repository.policy_candidates("修读上限", 2024)] == [
            "current"
        ]
        assert [
            row["source_id"]
            for row in repository.policy_candidates(
                "修读上限",
                2024,
                as_of="2024-06-01",
            )
        ] == ["old"]
    finally:
        repository.close()


def test_equal_authority_policy_conflict_is_exposed_and_blocks_answer(
    tmp_path: Path,
) -> None:
    repository = _policy_repository(tmp_path)
    try:
        rows = repository.policy_candidates("专业选修最低要求", 2024)
        conflicts = repository.policy_conflicts(rows)
        assert conflicts
        assert "同级规定甲" in conflicts[0]
        assert "同级规定乙" in conflicts[0]

        packet = AcademicTools(repository).retrieve_policy(
            RetrievePolicyOperation(
                operation_id="policy-op",
                args=RetrievePolicyArgs(question="专业选修最低要求", cohort=2024),
            )
        )
        assert packet.coverage.policy is not None
        assert packet.coverage.policy.conflict_free is False
        assert packet.coverage.policy.version_resolved is False
        assert packet.conflicts

        fact = packet.facts[0]
        answer = FinalAnswer(
            answer_md="专业选修最低要求为 3 学分。",
            claims=(
                ClaimSpan(
                    text="专业选修最低要求为 3 学分。",
                    fact_ids=(fact.fact_id,),
                    evidence_ids=fact.evidence_ids,
                    validation=ClaimValidation(claim_id="claim", passed=False),
                ),
            ),
            citations=(),
        )
        result = ClaimValidator().validate(answer, packet)
        assert result.refused
        assert "不会自动选择" in result.answer_md
        assert {value.source_id for value in result.citations} >= {
            "conflict-a",
            "conflict-b",
        }
    finally:
        repository.close()


def test_school_fact_requires_evidence_binding() -> None:
    packet = EvidencePacket(
        packet_id="packet",
        facts=(
            Fact(
                fact_id="course-credit",
                type="course",
                subject="course",
                predicate="credits",
                value=3,
                source_record_ids=("record",),
            ),
        ),
    )
    answer = FinalAnswer(
        answer_md="需要 3 学分。",
        claims=(
            ClaimSpan(
                text="需要 3 学分。",
                fact_ids=("course-credit",),
                evidence_ids=(),
                validation=ClaimValidation(claim_id="claim", passed=False),
            ),
        ),
        citations=(),
    )

    result = ClaimValidator().validate(answer, packet)

    assert result.refused
    assert "school_fact_without_evidence" in result.claims[0].validation.reasons
    assert "school_fact_missing_evidence_binding" in result.claims[0].validation.reasons


def test_derived_fact_inherits_evidence_from_recursive_inputs() -> None:
    evidence = _evidence()
    input_fact = Fact(
        fact_id="completed-credit",
        type="course",
        subject="course",
        predicate="credits",
        value=3,
        source_record_ids=("record",),
        evidence_ids=(evidence.evidence_id,),
    )
    derived = DerivedFact(
        fact_id="remaining-credit",
        type="progress",
        subject="module",
        predicate="remaining_credits",
        value=3,
        source_record_ids=("requirement",),
        operator="difference",
        input_fact_ids=(input_fact.fact_id,),
    )
    packet = EvidencePacket(
        packet_id="packet",
        facts=(input_fact, derived),
        evidence=(evidence,),
    )
    answer = FinalAnswer(
        answer_md="尚差 3 学分。",
        claims=(
            ClaimSpan(
                text="尚差 3 学分。",
                fact_ids=(derived.fact_id,),
                evidence_ids=(evidence.evidence_id,),
                validation=ClaimValidation(claim_id="claim", passed=False),
            ),
        ),
        citations=(),
    )

    result = ClaimValidator().validate(answer, packet)

    assert not result.refused
    assert result.citations == (evidence,)
