from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import pytest

from academic.database import SCHEMA, AcademicRepository, DataIntegrityError, build_database

SOURCE_FIELDS = (
    "file",
    "doc_title",
    "level",
    "college",
    "cohort",
    "year",
    "status",
    "page_url",
    "file_url",
    "collected_at",
)


def _source(title: str, cohort: str, college: str, file_name: str) -> dict[str, str]:
    return {
        "file": file_name,
        "doc_title": title,
        "level": "校级" if college == "全校" else "院级",
        "college": college,
        "cohort": cohort,
        "year": cohort if cohort.isdigit() else "2024",
        "status": "现行",
        "page_url": f"https://example.test/{file_name}",
        "file_url": f"https://example.test/{file_name}",
        "collected_at": "2026-01-01",
    }


def _build(
    tmp_path: Path,
    *,
    sources: list[dict[str, str]],
    catalog: dict[str, object],
    chunks: list[dict[str, object]],
) -> Path:
    sources_path = tmp_path / "sources.csv"
    with sources_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_FIELDS)
        writer.writeheader()
        writer.writerows(sources)
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chunks),
        encoding="utf-8",
    )
    aliases_path = tmp_path / "aliases.json"
    aliases_path.write_text(
        json.dumps({"program_aliases": {}, "module_aliases": {}, "course_aliases": {}}),
        encoding="utf-8",
    )
    database = tmp_path / "academic.sqlite3"
    build_database(
        database,
        catalog_path=catalog_path,
        sources_path=sources_path,
        chunks_path=chunks_path,
        aliases_path=aliases_path,
    )
    return database


def _repository(tmp_path: Path) -> AcademicRepository:
    path = tmp_path / "scope.sqlite3"
    connection = sqlite3.connect(path)
    try:
        connection.executescript(SCHEMA)
        connection.executemany(
            "INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                (
                    "source-a", "培养方案甲", "院级", "学院甲", "2024", 1, "2024-01-01",
                    "2024-01-01", None, None, "现行", "https://example.test/a",
                    "https://example.test/a.pdf", "a-hash", "2026-01-01",
                ),
                (
                    "source-b", "培养方案乙", "院级", "学院乙", "2024", 1, "2024-01-01",
                    "2024-01-01", None, None, "现行", "https://example.test/b",
                    "https://example.test/b.pdf", "b-hash", "2026-01-01",
                ),
                (
                    "source-old", "培养方案甲", "院级", "学院甲", "2023", 1, "2023-01-01",
                    "2023-01-01", None, None, "现行", "https://example.test/old",
                    "https://example.test/old.pdf", "old-hash", "2026-01-01",
                ),
                (
                    "source-policy", "学校政策", "校级", "全校", "不限", 2, "2024-01-01",
                    "2024-01-01", None, None, "现行", "https://example.test/policy",
                    "https://example.test/policy.pdf", "policy-hash", "2026-01-01",
                ),
            ),
        )
        connection.executemany(
            "INSERT INTO source_taxonomy VALUES (?, ?, ?)",
            (
                ("source-a", "curriculum", "[]"),
                ("source-b", "curriculum", "[]"),
                ("source-old", "curriculum", "[]"),
                ("source-policy", "policy", '["transfer"]'),
            ),
        )
        connection.executemany(
            "INSERT INTO source_authenticity VALUES (?, ?, ?, ?, ?)",
            (
                ("source-a", "a-hash", "a-hash", "verified", "include"),
                ("source-b", "b-hash", "b-hash", "verified", "include"),
                ("source-old", "old-hash", "old-hash", "verified", "include"),
                ("source-policy", "policy-hash", "policy-hash", "verified", "include"),
            ),
        )
        connection.executemany(
            "INSERT INTO programs VALUES (?, ?, ?, ?, ?)",
            (
                ("program-a", "专业甲", "学院甲", 2024, "source-a"),
                ("program-b", "专业乙", "学院乙", 2024, "source-b"),
                ("program-old", "专业甲", "学院甲", 2023, "source-old"),
            ),
        )
        connection.executemany(
            "INSERT INTO modules VALUES (?, ?, ?)",
            (
                ("module-a", "program-a", "共同模块"),
                ("module-b", "program-b", "共同模块"),
                ("module-old", "program-old", "共同模块"),
            ),
        )
        connection.executemany(
            "INSERT INTO module_aliases VALUES (?, ?, ?)",
            (
                ("共同模块", "共同模块", "module-a"),
                ("共同模块", "共同模块", "module-b"),
                ("共同模块", "共同模块", "module-old"),
            ),
        )
        connection.executemany(
            "INSERT INTO courses VALUES (?, ?, ?)",
            (
                ("course-a", "ALP101", "同名课程"),
                ("course-b", "BET101", "同名课程"),
                ("course-old", "OLD101", "跨届课程"),
            ),
        )
        connection.executemany(
            "INSERT INTO course_aliases VALUES (?, ?, ?)",
            (
                ("ALP101", "alp101", "course-a"),
                ("同名课程", "同名课程", "course-a"),
                ("BET101", "bet101", "course-b"),
                ("同名课程", "同名课程", "course-b"),
                ("OLD101", "old101", "course-old"),
                ("跨届课程", "跨届课程", "course-old"),
            ),
        )
        offerings = (
            ("record-a", "program-a", "module-a", "course-a", "必修", "1", 3.0, None, None, None, None, "学院甲", "source-a", 1, 1, "section-a", "parser", 1.0, "verified"),
            ("record-b", "program-b", "module-b", "course-b", "必修", "1", 4.0, None, None, None, None, "学院乙", "source-b", 1, 1, "section-b", "parser", 1.0, "verified"),
            ("record-old", "program-old", "module-old", "course-old", "必修", "1", 2.0, None, None, None, None, "学院甲", "source-old", 1, 1, "section-old", "parser", 1.0, "verified"),
        )
        connection.executemany(
            "INSERT INTO program_courses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            offerings,
        )
        connection.executemany(
            "INSERT INTO source_sections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ("section-a", "source-a", "第一条", "学院甲专业甲规定。", 1, 0, "parser", "2026-01-01T00:00:00+00:00", 1.0, "verified"),
                ("section-b", "source-b", "第一条", "学院乙专业乙规定。", 1, 0, "parser", "2026-01-01T00:00:00+00:00", 1.0, "unverified"),
                ("section-old", "source-old", "第一条", "旧届规定。", 1, 0, "parser", "2026-01-01T00:00:00+00:00", 1.0, "verified"),
                ("section-policy", "source-policy", "第二条", "学校范围内的政策。", 2, 0, "parser", "2026-01-01T00:00:00+00:00", 1.0, "review_required"),
            ),
        )
        connection.executemany(
            "INSERT INTO section_extraction_quality VALUES (?, ?, ?)",
            (
                ("section-a", "verified", "[]"),
                ("section-b", "verified", "[]"),
                ("section-old", "verified", "[]"),
                ("section-policy", "verified", "[]"),
            ),
        )
        connection.execute("INSERT INTO metadata VALUES ('schema_version', '2')")
        connection.commit()
    finally:
        connection.close()
    return AcademicRepository(path)


def test_program_detection_does_not_treat_course_wording_as_a_major(tmp_path: Path) -> None:
    database = _build(
        tmp_path,
        sources=[_source("英语培养方案", "2024", "外国语学院", "english.pdf")],
        catalog={
            "catalog_version": "program-detection",
            "plans": [
                {
                    "college": "外国语学院",
                    "cohort": "2024",
                    "major": "英语专业",
                    "source_title": "英语培养方案",
                    "modules": [],
                }
            ],
            "courses": [],
        },
        chunks=[],
    )
    repository = AcademicRepository(database)
    try:
        assert repository.programs_in_text("大学英语达到什么条件可以免修？", 2024) == ()
        assert tuple(
            item.canonical_name
            for item in repository.programs_in_text("2024级英语专业培养方案", 2024)
        ) == ("英语专业",)
    finally:
        repository.close()


def test_source_lookup_requires_exact_scope_or_unlimited(tmp_path: Path) -> None:
    catalog = {
        "catalog_version": "scope-test",
        "plans": [
            {"college": "学院甲", "cohort": "2024", "major": "专业甲", "source_title": "同标题", "modules": []}
        ],
        "courses": [],
    }
    with pytest.raises(DataIntegrityError, match="universal scope"):
        _build(
            tmp_path,
            sources=[_source("同标题", "2023", "学院甲", "old.pdf")],
            catalog=catalog,
            chunks=[],
        )


def test_chunk_source_lookup_does_not_fallback_across_cohorts(tmp_path: Path) -> None:
    with pytest.raises(DataIntegrityError, match="universal scope"):
        _build(
            tmp_path,
            sources=[_source("同标题", "2023", "学院甲", "old.pdf")],
            catalog={"catalog_version": "chunk-scope", "plans": [], "courses": []},
            chunks=[
                {
                    "chunk_id": "wrong-cohort",
                    "text": "2024 内容",
                    "doc_title": "同标题",
                    "article": "第一条",
                    "cohort": "2024",
                    "is_table": False,
                }
            ],
        )

def test_chunk_scope_and_review_status_are_validated(tmp_path: Path) -> None:
    catalog = {"catalog_version": "chunk-test", "plans": [], "courses": []}
    with pytest.raises(DataIntegrityError, match="invalid review_status"):
        _build(
            tmp_path,
            sources=[_source("学校政策", "不限", "全校", "policy.pdf")],
            catalog=catalog,
            chunks=[
                {
                    "chunk_id": "invalid-status",
                    "text": "政策内容",
                    "doc_title": "学校政策",
                    "article": "第一条",
                    "cohort": "不限",
                    "is_table": False,
                    "review_status": "pending",
                }
            ],
        )


def test_course_and_module_resolution_respects_program_and_cohort(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    try:
        assert repository.resolve_course("同名课程", 2024) is None
        assert len(repository.resolve_course_candidates("同名课程", 2024)) == 2
        alpha = repository.resolve_course("ALP101", 2024, "program-a")
        assert alpha is not None
        assert repository.resolve_course("ALP101", 2024, "program-b") is None
        assert repository.resolve_course("OLD101", 2024, "program-a") is None
        assert repository.courses_in_text("请修读 ALP101 同名课程", 2024, "program-a") == (alpha,)
        assert repository.resolve_module("共同模块") is None
        module = repository.resolve_module("共同模块", "program-a")
        assert module is not None
        assert repository.modules_in_text("共同模块要求", "program-a") == (module,)
    finally:
        repository.close()
def test_repository_rejects_specific_policy_when_request_scope_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    try:
        base = dict(repository.retrieval_documents()[0])
        specific = {
            **base,
            "cohort": "2024",
            "college_id": "学院甲",
            "program_ids": ("program-a",),
        }
        monkeypatch.setattr(repository, "retrieval_documents", lambda: (specific,))

        assert repository.scoped_policy_documents() == ()
        assert repository.scoped_policy_documents(cohort=2024) == ()
        exact = repository.scoped_policy_documents(
            cohort=2024,
            program_ids=("program-a",),
        )
        assert {item["chunk_id"] for item in exact} == {"section-policy"}
    finally:
        repository.close()

def test_college_scope_and_retrieval_provider_are_data_driven(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    try:
        assert repository.college_ids_for_programs(("program-a", "program-b")) == ("学院甲", "学院乙")
        assert repository.program_ids_for_colleges(("学院甲",), 2024) == ("program-a",)
        documents = {str(item["chunk_id"]): item for item in repository.retrieval_documents()}
        required = {
            "chunk_id", "text", "source_id", "title", "article", "physical_page", "page_url",
            "file_url", "review_status", "college_id", "cohort", "authority_level",
            "effective_from", "effective_to", "status", "supersedes_source_id",
        }
        assert set(documents) == {"section-policy"}
        assert required.issubset(documents["section-policy"])
        assert documents["section-policy"]["doc_type"] == "policy"
        assert documents["section-policy"]["topics"] == ("transfer",)
        assert "program_ids" not in documents["section-policy"]
        assert {
            item["chunk_id"] for item in repository.scoped_policy_documents()
        } == {"section-policy"}
        scoped = repository.scoped_policy_documents(cohort=2024, program_ids=("program-a",))
        assert {item["chunk_id"] for item in scoped} == {"section-policy"}
    finally:
        repository.close()
