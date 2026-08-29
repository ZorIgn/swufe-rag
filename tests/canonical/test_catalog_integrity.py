from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import pytest

from academic.database import DataIntegrityError, build_database
from ingest.sources import SOURCE_FIELDS

SOURCE_SHA = "e" * 64


def _lineage(field: str, value: str) -> dict[str, object]:
    return {
        "verification_status": "verified",
        "lineage": {
            "source_sha256": SOURCE_SHA,
            "chunk_id": "curriculum-row",
            "page": 1,
            "row": 2,
            "cell": field,
            "span": value,
        },
    }


def _catalog(*, with_lineage: bool = True) -> dict[str, object]:
    module: dict[str, object] = {
        "name": "专业选修课",
        "required_credits": 3,
        "listed_credits": 3,
        "rule_text": "最低要求为 3 学分",
        "evidence": {"chunk_id": "curriculum-row", "article": "原文件第1页"},
    }
    course: dict[str, object] = {
        "code": "TST101",
        "name": "测试算法",
        "credits": 3,
        "nature": "选修",
        "semester": "1",
        "department": "测试学院",
        "college": "测试学院",
        "cohort": "2024",
        "major": "测试专业",
        "module": "专业选修课",
        "source_title": "测试培养方案",
        "page": 1,
        "source_row": 2,
        "evidence": {"chunk_id": "curriculum-row"},
    }
    if with_lineage:
        module["field_verification"] = {
            "module": _lineage("module", "专业选修课"),
            "required_credits": _lineage("required_credits", "3 学分"),
        }
        course["field_verification"] = {
            "module": _lineage("module", "专业选修课"),
            "credits": _lineage("credits", "3 学分"),
            "semester": _lineage("semester", "第 1 学期开设"),
            "nature": _lineage("nature", "选修"),
        }
    return {
        "catalog_version": "integrity-fixture-1",
        "plans": [
            {
                "college": "测试学院",
                "cohort": "2024",
                "major": "测试专业",
                "source_title": "测试培养方案",
                "modules": [module],
            }
        ],
        "courses": [course],
    }


def _write_inputs(tmp_path: Path, catalog: dict[str, object]) -> tuple[Path, Path, Path, Path]:
    sources = tmp_path / "sources.csv"
    with sources.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "file": "curriculum.pdf",
                "doc_title": "测试培养方案",
                "level": "院级",
                "college": "测试学院",
                "cohort": "2024",
                "year": "2024",
                "status": "现行",
                "page_url": "https://example.test/curriculum.pdf#page=1",
                "file_url": "https://example.test/curriculum.pdf",
                "collected_at": "2026-01-01",
                "doc_type": "curriculum",
                "topics": "",
                "source_sha256": SOURCE_SHA,
                "authenticity_status": "verified",
            }
        )
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        json.dumps(
            {
                "chunk_id": "curriculum-row",
                "text": "专业选修课最低要求为 3 学分。专业选修课包含测试算法（TST101），选修，第 1 学期开设，3 学分。",
                "doc_title": "测试培养方案",
                "article": "原文件第1页",
                "cohort": "2024",
                "is_table": True,
                "review_status": "verified",
                "doc_type": "curriculum",
                "topics": [],
                "source_sha256": SOURCE_SHA,
                "extraction_quality": "verified",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
    source_review = tmp_path / "source_review.csv"
    source_review.write_text(
        "original_title,corrected_title,decision\n"
        "curriculum.pdf,测试培养方案,include\n",
        encoding="utf-8",
    )
    aliases = tmp_path / "aliases.json"
    aliases.write_text(
        json.dumps({"program_aliases": {}, "module_aliases": {}, "course_aliases": {}}),
        encoding="utf-8",
    )
    return sources, chunks, catalog_path, aliases


def _build(tmp_path: Path, catalog: dict[str, object]) -> tuple[Path, dict[str, object]]:
    sources, chunks, catalog_path, aliases = _write_inputs(tmp_path, catalog)
    database = tmp_path / "academic.sqlite3"
    report = build_database(
        database,
        catalog_path=catalog_path,
        sources_path=sources,
        chunks_path=chunks,
        aliases_path=aliases,
        source_review_path=tmp_path / "source_review.csv",
    )
    return database, report


def test_conflicting_course_credits_fail_before_persistence(tmp_path: Path) -> None:
    catalog = _catalog()
    courses = catalog["courses"]
    assert isinstance(courses, list)
    conflicting = dict(courses[0])
    conflicting["credits"] = 9
    courses.append(conflicting)

    with pytest.raises(DataIntegrityError, match="course_offerings canonical-key conflict") as error:
        _build(tmp_path, catalog)

    message = str(error.value)
    assert "input_rows=[1, 2]" in message
    assert '"credits"' in message
    assert "9" in message
    assert not (tmp_path / "academic.sqlite3").exists()


def test_explicit_source_root_requires_registered_bytes_and_hash(tmp_path: Path) -> None:
    sources, chunks, catalog_path, aliases = _write_inputs(tmp_path, _catalog())

    with pytest.raises(DataIntegrityError, match="explicit source_root"):
        build_database(
            tmp_path / "academic.sqlite3",
            catalog_path=catalog_path,
            sources_path=sources,
            chunks_path=chunks,
            aliases_path=aliases,
            source_review_path=tmp_path / "source_review.csv",
            source_root=tmp_path / "missing-raw",
        )


def test_exact_duplicate_is_deduplicated_with_conserved_counts(tmp_path: Path) -> None:
    catalog = _catalog()
    courses = catalog["courses"]
    assert isinstance(courses, list)
    courses.append(json.loads(json.dumps(courses[0])))

    database, report = _build(tmp_path, catalog)

    counts = report["reconciliation_counts"]
    assert isinstance(counts, dict)
    assert counts["course_offerings"] == {
        "input": 2,
        "accepted": 1,
        "exact_duplicates": 1,
        "quarantined": 0,
    }
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM program_courses").fetchone()[0] == 1
        metadata = json.loads(
            connection.execute(
                "SELECT value FROM metadata WHERE key='reconciliation_counts'"
            ).fetchone()[0]
        )
    assert metadata["course_offerings"] == counts["course_offerings"]


def test_source_review_cannot_verify_numeric_fields_without_lineage(tmp_path: Path) -> None:
    database, _report = _build(tmp_path, _catalog(with_lineage=False))

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT review_status FROM source_sections").fetchone()[0] == "verified"
        assert connection.execute("SELECT review_status FROM program_courses").fetchone()[0] == "review_required"
        assert connection.execute("SELECT review_status FROM requirements").fetchone()[0] == "review_required"
        rows = connection.execute(
            "SELECT field_name, verification_status FROM field_verifications ORDER BY entity_type, field_name"
        ).fetchall()
    assert rows
    assert {status for _field, status in rows} == {"review_required"}


def test_reviewed_field_lineage_is_required_for_verified_rows(tmp_path: Path) -> None:
    database, _report = _build(tmp_path, _catalog(with_lineage=True))

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT review_status FROM program_courses").fetchone()[0] == "verified"
        assert connection.execute("SELECT review_status FROM requirements").fetchone()[0] == "verified"
        rows = connection.execute(
            "SELECT field_name, source_sha256, physical_page, table_row, cell_ref, text_span, verification_status "
            "FROM field_verifications ORDER BY entity_type, field_name"
        ).fetchall()
    assert len(rows) == 6
    assert all(row[-1] == "verified" for row in rows)
    assert all(row[1] == SOURCE_SHA and row[2] == 1 and row[5] for row in rows)


@pytest.mark.parametrize(
    ("field", "misleading_span"),
    [
        ("credits", "30 学分"),
        ("semester", "第 10 学期开设"),
    ],
)
def test_numeric_field_lineage_requires_the_exact_value_not_a_prefix(
    tmp_path: Path, field: str, misleading_span: str
) -> None:
    """A field review cannot turn 30 credits / term 10 into 3 / term 1."""

    catalog = _catalog()
    courses = catalog["courses"]
    assert isinstance(courses, list)
    field_verification = courses[0]["field_verification"]
    assert isinstance(field_verification, dict)
    lineage = field_verification[field]
    assert isinstance(lineage, dict)
    provenance = lineage["lineage"]
    assert isinstance(provenance, dict)
    provenance["span"] = misleading_span

    database, _report = _build(tmp_path, catalog)

    with sqlite3.connect(database) as connection:
        course_status = connection.execute(
            "SELECT review_status FROM program_courses"
        ).fetchone()[0]
        field_status = connection.execute(
            "SELECT verification_status FROM field_verifications "
            "WHERE entity_type='program_course' AND field_name=?",
            (field,),
        ).fetchone()[0]
    assert course_status == "review_required"
    assert field_status == "review_required"
