"""Regression coverage for the explicit reviewed-draft catalog compiler."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from copy import deepcopy
from pathlib import Path

import pytest

from academic.database import build_database
from ingest.catalog import apply_review_ledger, extract_catalog_draft
from ingest.catalog_materialize import CatalogMaterializationError, materialize_catalog
from ingest.models import DocumentElement, ParsedDocument, SourceRecord
from ingest.sources import SOURCE_FIELDS
from scripts.materialize_catalog import main as materialize_cli

SOURCE_BYTES = b"catalog materialization source fixture"
SOURCE_SHA = hashlib.sha256(SOURCE_BYTES).hexdigest()
SOURCE_TITLE = "公开培养方案物化测试"
CHUNK_ID = "curriculum-row"


def _source() -> SourceRecord:
    return SourceRecord(
        file="public-curriculum.pdf",
        doc_title=SOURCE_TITLE,
        level="院级",
        college="测试学院",
        cohort="2024",
        year=2024,
        status="现行",
        page_url="https://it.swufe.edu.cn/public-curriculum.pdf#page=7",
        file_url="https://it.swufe.edu.cn/public-curriculum.pdf",
        collected_at="2026-08-19",
        doc_type="curriculum",
        source_sha256=SOURCE_SHA,
        authenticity_status="verified",
    )


def _reviewed_draft() -> dict[str, object]:
    document = ParsedDocument(
        path=Path("public-curriculum.pdf"),
        elements=[
            DocumentElement(
                "table",
                "\n".join(
                    [
                        "| 课程代码 | 课程名称 | 学分 | 开课学期 | 课程性质 | 课程模块 |",
                        "| --- | --- | --- | --- | --- | --- |",
                        "| CS101 | 程序设计基础 | 3 | 第1学期 | 必修 | 专业基础模块 |",
                    ]
                ),
                page=7,
            )
        ],
        page_count=7,
    )
    draft = extract_catalog_draft(document, source=_source())
    courses = draft["courses"]
    assert isinstance(courses, list) and len(courses) == 1
    record_id = courses[0]["record_id"]
    assert isinstance(record_id, str)
    return apply_review_ledger(
        draft,
        [
            {
                "record_id": record_id,
                "decision": "approve",
                "reviewer": "catalog-reviewer",
                "reviewed_at": "2026-08-19T12:00:00+00:00",
                "field_updates": {},
            }
        ],
    )


def _record_id(reviewed_draft: dict[str, object]) -> str:
    courses = reviewed_draft["courses"]
    assert isinstance(courses, list) and len(courses) == 1
    record_id = courses[0]["record_id"]
    assert isinstance(record_id, str)
    return record_id


def _lineage(*, cell: str, span: str) -> dict[str, object]:
    return {
        "verification_status": "verified",
        "lineage": {
            "source_sha256": SOURCE_SHA,
            "chunk_id": CHUNK_ID,
            "page": 7,
            "row": 3,
            "cell": cell,
            "span": span,
        },
    }


def _plan_scaffold(record_id: str) -> dict[str, object]:
    return {
        "catalog_version": "materialized-catalog-fixture-1",
        "plans": [
            {
                "college": "测试学院",
                "cohort": "2024",
                "major": "测试专业",
                "source_title": SOURCE_TITLE,
                "modules": [
                    {
                        "name": "专业基础模块",
                        # This independently reviewed requirement deliberately
                        # differs from the course credit value below: the
                        # materializer must copy it, never infer it from rows.
                        "required_credits": 12,
                        "listed_credits": 12,
                        "rule_text": "专业基础模块最低要求为12学分",
                        "evidence": {"chunk_id": CHUNK_ID},
                        "field_verification": {
                            "module": _lineage(cell="6", span="专业基础模块"),
                            "required_credits": _lineage(cell="R1", span="最低要求为12学分"),
                        },
                    }
                ],
            }
        ],
        "course_assignments": [
            {
                "record_id": record_id,
                "college": "测试学院",
                "cohort": "2024",
                "major": "测试专业",
                "module": "专业基础模块",
                "source_title": SOURCE_TITLE,
                "department": "测试学院",
            }
        ],
    }


def _evidence_mapping(record_id: str) -> dict[str, object]:
    return {
        "mappings": [
            {
                "record_id": record_id,
                "source_title": SOURCE_TITLE,
                "source_sha256": SOURCE_SHA,
                "evidence": {"chunk_id": CHUNK_ID},
                "page": 7,
                "source_row": 3,
                "fields": {
                    "module": _lineage(cell="6", span="专业基础模块"),
                    "credits": _lineage(cell="3", span="3学分"),
                    "semester": _lineage(cell="4", span="第1学期"),
                    "nature": _lineage(cell="5", span="必修"),
                },
            }
        ]
    }


def _write_database_inputs(
    tmp_path: Path, catalog: dict[str, object]
) -> tuple[Path, Path, Path, Path, Path, Path]:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "public-curriculum.pdf").write_bytes(SOURCE_BYTES)
    sources = tmp_path / "sources.csv"
    with sources.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SOURCE_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "file": "public-curriculum.pdf",
                "doc_title": SOURCE_TITLE,
                "level": "院级",
                "college": "测试学院",
                "cohort": "2024",
                "year": "2024",
                "status": "现行",
                "page_url": "https://it.swufe.edu.cn/public-curriculum.pdf#page=7",
                "file_url": "https://it.swufe.edu.cn/public-curriculum.pdf",
                "collected_at": "2026-08-19",
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
                "chunk_id": CHUNK_ID,
                "text": "专业基础模块最低要求为12学分。程序设计基础（CS101）必修，第1学期开设，3学分。",
                "doc_title": SOURCE_TITLE,
                "article": "原文件第7页",
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
    catalog_path = tmp_path / "curriculum_catalog.json"
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
    source_review = tmp_path / "source_review.csv"
    source_review.write_text(
        "original_title,corrected_title,decision\n"
        f"public-curriculum.pdf,{SOURCE_TITLE},include\n",
        encoding="utf-8",
    )
    aliases = tmp_path / "aliases.json"
    aliases.write_text(
        json.dumps({"program_aliases": {}, "module_aliases": {}, "course_aliases": {}}),
        encoding="utf-8",
    )
    return raw_dir, sources, chunks, catalog_path, source_review, aliases


def test_materialization_is_explicit_traceable_and_database_compatible(tmp_path: Path) -> None:
    reviewed = _reviewed_draft()
    record_id = _record_id(reviewed)
    scaffold = _plan_scaffold(record_id)
    mapping = _evidence_mapping(record_id)

    materialized = materialize_catalog(
        reviewed,
        scaffold,
        mapping,
        input_file_hashes={
            "reviewed_draft_file_sha256": "a" * 64,
            "plan_scaffold_file_sha256": "b" * 64,
            "evidence_mapping_file_sha256": "c" * 64,
        },
    )

    assert materialized["catalog_version"] == scaffold["catalog_version"]
    courses = materialized["courses"]
    assert isinstance(courses, list) and len(courses) == 1
    course = courses[0]
    assert {field: course[field] for field in ("college", "cohort", "major", "source_title")} == {
        "college": "测试学院",
        "cohort": "2024",
        "major": "测试专业",
        "source_title": SOURCE_TITLE,
    }
    assert (course["page"], course["source_row"], course["evidence"]) == (
        7,
        3,
        {"chunk_id": CHUNK_ID},
    )
    assert course["credits"] == 3
    field_verification = course["field_verification"]
    assert set(field_verification) == {"module", "credits", "semester", "nature"}
    assert all(item["verification_status"] == "verified" for item in field_verification.values())
    assert all(
        item["lineage"]["chunk_id"] == CHUNK_ID
        and item["lineage"]["source_sha256"] == SOURCE_SHA
        and item["lineage"]["page"] == 7
        and item["lineage"]["row"] == 3
        for item in field_verification.values()
    )
    metadata = materialized["materialization"]
    assert metadata["counts"] == {
        "input_courses": 1,
        "materialized_courses": 1,
        "quarantined_courses": 0,
        "upstream_quarantine_records": 0,
        "assignment_records": 1,
        "evidence_mapping_records": 1,
        "assignment_records_for_upstream_quarantine": 0,
        "evidence_mapping_records_for_upstream_quarantine": 0,
    }
    assert all(len(value) == 64 for value in metadata["input_hashes"].values())
    assert metadata["input_file_hashes"]["evidence_mapping_file_sha256"] == "c" * 64
    assert metadata["records"][0]["lineage"]["credits"]["draft_lineages"][0]["raw_value"] == "3"
    plans = materialized["plans"]
    assert plans[0]["modules"][0]["required_credits"] == 12
    assert metadata["module_requirements_boundary"]["owner"] == "plan_scaffold"

    raw_dir, sources, chunks, catalog_path, source_review, aliases = _write_database_inputs(
        tmp_path, materialized
    )
    database = tmp_path / "academic.sqlite3"
    report = build_database(
        database,
        catalog_path=catalog_path,
        sources_path=sources,
        chunks_path=chunks,
        aliases_path=aliases,
        source_review_path=source_review,
        evidence_review_path=None,
        source_root=raw_dir,
    )
    assert report["offering_count"] == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT review_status FROM program_courses").fetchone()[0] == "verified"
        statuses = connection.execute(
            "SELECT field_name, verification_status FROM field_verifications "
            "WHERE entity_type='program_course' ORDER BY field_name"
        ).fetchall()
    assert statuses == [
        ("credits", "verified"),
        ("module", "verified"),
        ("nature", "verified"),
        ("semester", "verified"),
    ]


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("unapproved", "course_not_explicitly_approved"),
        ("missing_evidence", "missing_evidence_mapping"),
        ("missing_assignment", "missing_plan_mapping"),
        ("unknown_plan_scope", "missing_plan_mapping"),
        ("missing_strict_field", "evidence_mapping_missing_field"),
        ("unapproved_field", "course_field_not_explicitly_approved"),
    ],
)
def test_materialization_quarantines_unverified_or_unmapped_records_with_reasons(
    mutation: str, expected_reason: str
) -> None:
    reviewed = _reviewed_draft()
    record_id = _record_id(reviewed)
    scaffold = _plan_scaffold(record_id)
    mapping = _evidence_mapping(record_id)
    courses = reviewed["courses"]
    assert isinstance(courses, list)
    course = courses[0]
    assert isinstance(course, dict)

    if mutation == "unapproved":
        course["review_status"] = "review_required"
    elif mutation == "missing_evidence":
        mapping["mappings"] = []
    elif mutation == "missing_assignment":
        scaffold["course_assignments"] = []
    elif mutation == "unknown_plan_scope":
        assignments = scaffold["course_assignments"]
        assert isinstance(assignments, list)
        assignments[0]["major"] = "未建档专业"
    elif mutation == "missing_strict_field":
        mappings = mapping["mappings"]
        assert isinstance(mappings, list)
        fields = mappings[0]["fields"]
        assert isinstance(fields, dict)
        fields.pop("nature")
    elif mutation == "unapproved_field":
        field_verification = course["field_verification"]
        assert isinstance(field_verification, dict)
        field_verification["credits"] = "review_required"
    else:  # pragma: no cover - keeps future parametrization exhaustive.
        raise AssertionError(mutation)

    materialized = materialize_catalog(reviewed, scaffold, mapping)

    assert materialized["courses"] == []
    metadata = materialized["materialization"]
    assert metadata["counts"]["input_courses"] == 1
    assert metadata["counts"]["materialized_courses"] == 0
    assert metadata["counts"]["quarantined_courses"] == 1
    assert metadata["quarantine"][0]["record_id"] == record_id
    assert metadata["quarantine"][0]["reason"] == expected_reason


def test_fail_on_quarantine_raises_before_any_canonical_output_is_returned() -> None:
    reviewed = _reviewed_draft()
    record_id = _record_id(reviewed)

    with pytest.raises(CatalogMaterializationError, match="missing_evidence_mapping"):
        materialize_catalog(
            reviewed,
            _plan_scaffold(record_id),
            {"mappings": []},
            fail_on_quarantine=True,
        )


@pytest.mark.parametrize(
    ("field", "value", "expected_reason"),
    [
        ("college", "错误学院", "source_scope_mismatch"),
        ("cohort", "2023", "source_scope_mismatch"),
        ("doc_type", "policy", "source_document_type_not_catalog"),
        ("source_authenticity_status", "review_required", "source_not_authenticity_verified"),
    ],
)
def test_materialization_binds_source_scope_type_and_authenticity_to_assignment(
    field: str, value: str, expected_reason: str
) -> None:
    reviewed = _reviewed_draft()
    record_id = _record_id(reviewed)
    courses = reviewed["courses"]
    assert isinstance(courses, list) and isinstance(courses[0], dict)
    source = courses[0]["source"]
    assert isinstance(source, dict)
    source[field] = value

    materialized = materialize_catalog(
        reviewed, _plan_scaffold(record_id), _evidence_mapping(record_id)
    )

    assert materialized["courses"] == []
    assert materialized["materialization"]["quarantine"][0]["reason"] == expected_reason


def test_each_active_course_requires_a_matching_terminal_review_ledger_event() -> None:
    reviewed = _reviewed_draft()
    record_id = _record_id(reviewed)
    reviewed["review_ledger"] = []

    materialized = materialize_catalog(
        reviewed, _plan_scaffold(record_id), _evidence_mapping(record_id)
    )

    assert materialized["courses"] == []
    assert (
        materialized["materialization"]["quarantine"][0]["reason"]
        == "course_missing_review_ledger_entry"
    )


def test_unknown_review_ledger_or_wrong_schema_fails_closed() -> None:
    reviewed = _reviewed_draft()
    record_id = _record_id(reviewed)
    bad_schema = deepcopy(reviewed)
    bad_schema["schema_version"] = "catalog-draft/v0"
    with pytest.raises(CatalogMaterializationError, match="schema_version"):
        materialize_catalog(bad_schema, _plan_scaffold(record_id), _evidence_mapping(record_id))

    unknown_ledger = deepcopy(reviewed)
    entries = unknown_ledger["review_ledger"]
    assert isinstance(entries, list)
    entries[0]["record_id"] = "course-not-in-draft"
    with pytest.raises(CatalogMaterializationError, match="unknown record_id"):
        materialize_catalog(unknown_ledger, _plan_scaffold(record_id), _evidence_mapping(record_id))


@pytest.mark.parametrize("kind", ["assignment", "mapping"])
def test_orphan_explicit_records_are_rejected(kind: str) -> None:
    reviewed = _reviewed_draft()
    record_id = _record_id(reviewed)
    scaffold = _plan_scaffold(record_id)
    mapping = _evidence_mapping(record_id)
    if kind == "assignment":
        assignments = scaffold["course_assignments"]
        assert isinstance(assignments, list)
        orphan = deepcopy(assignments[0])
        orphan["record_id"] = "course-orphan"
        assignments.append(orphan)
    else:
        mappings = mapping["mappings"]
        assert isinstance(mappings, list)
        orphan = deepcopy(mappings[0])
        orphan["record_id"] = "course-orphan"
        mappings.append(orphan)

    with pytest.raises(CatalogMaterializationError, match="orphan record_id"):
        materialize_catalog(reviewed, scaffold, mapping)


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("invalid", "invalid_draft_field_lineage"),
        ("ambiguous", "ambiguous_draft_field_lineage"),
        ("wrong_reviewer", "reviewer_lineage_value_mismatch"),
    ],
)
def test_materialization_rejects_malformed_ambiguous_or_conflicting_draft_lineage(
    mutation: str, expected_reason: str
) -> None:
    reviewed = _reviewed_draft()
    record_id = _record_id(reviewed)
    courses = reviewed["courses"]
    assert isinstance(courses, list) and isinstance(courses[0], dict)
    lineage = courses[0]["field_lineage"]
    assert isinstance(lineage, dict)
    credits = lineage["credits"]
    assert isinstance(credits, list)
    if mutation == "invalid":
        credits.append("not-a-lineage-object")
    elif mutation == "ambiguous":
        conflicting = deepcopy(credits[0])
        conflicting["cell"] = 99
        credits.append(conflicting)
    else:
        credits.append(
            {
                "origin": "reviewer",
                "reviewer": "catalog-reviewer",
                "reviewed_at": "2026-08-19T12:00:00+00:00",
                "prior_value": 3.0,
                "review_value": 4.0,
            }
        )

    materialized = materialize_catalog(
        reviewed, _plan_scaffold(record_id), _evidence_mapping(record_id)
    )

    assert materialized["courses"] == []
    assert materialized["materialization"]["quarantine"][0]["reason"] == expected_reason


@pytest.mark.parametrize(
    ("field", "bad_span"),
    [("credits", "30学分"), ("semester", "第10学期")],
)
def test_numeric_field_evidence_rejects_prefix_matches(field: str, bad_span: str) -> None:
    reviewed = _reviewed_draft()
    record_id = _record_id(reviewed)
    mapping = _evidence_mapping(record_id)
    mappings = mapping["mappings"]
    assert isinstance(mappings, list)
    fields = mappings[0]["fields"]
    assert isinstance(fields, dict)
    field_mapping = fields[field]
    assert isinstance(field_mapping, dict)
    lineage = field_mapping["lineage"]
    assert isinstance(lineage, dict)
    lineage["span"] = bad_span

    materialized = materialize_catalog(reviewed, _plan_scaffold(record_id), mapping)

    assert materialized["courses"] == []
    assert (
        materialized["materialization"]["quarantine"][0]["reason"]
        == "evidence_span_does_not_cover_field_value"
    )


@pytest.mark.parametrize(
    ("field", "wrong_semantic_span"),
    [("credits", "第3学期"), ("semester", "1学分")],
)
def test_numeric_field_evidence_rejects_values_from_a_different_column_semantic(
    field: str, wrong_semantic_span: str
) -> None:
    reviewed = _reviewed_draft()
    record_id = _record_id(reviewed)
    mapping = _evidence_mapping(record_id)
    mappings = mapping["mappings"]
    assert isinstance(mappings, list)
    fields = mappings[0]["fields"]
    assert isinstance(fields, dict)
    field_mapping = fields[field]
    assert isinstance(field_mapping, dict)
    lineage = field_mapping["lineage"]
    assert isinstance(lineage, dict)
    lineage["span"] = wrong_semantic_span

    materialized = materialize_catalog(reviewed, _plan_scaffold(record_id), mapping)

    assert materialized["courses"] == []
    assert (
        materialized["materialization"]["quarantine"][0]["reason"]
        == "evidence_span_does_not_cover_field_value"
    )


@pytest.mark.parametrize(
    ("field", "non_exact_span"),
    [("module", "专业基础模块扩展"), ("nature", "必修课程")],
)
def test_text_field_evidence_rejects_non_exact_substrings(field: str, non_exact_span: str) -> None:
    reviewed = _reviewed_draft()
    record_id = _record_id(reviewed)
    mapping = _evidence_mapping(record_id)
    mappings = mapping["mappings"]
    assert isinstance(mappings, list)
    fields = mappings[0]["fields"]
    assert isinstance(fields, dict)
    field_mapping = fields[field]
    assert isinstance(field_mapping, dict)
    lineage = field_mapping["lineage"]
    assert isinstance(lineage, dict)
    lineage["span"] = non_exact_span

    materialized = materialize_catalog(reviewed, _plan_scaffold(record_id), mapping)

    assert materialized["courses"] == []
    assert (
        materialized["materialization"]["quarantine"][0]["reason"]
        == "evidence_span_does_not_cover_field_value"
    )


def test_department_is_optional_and_never_inferred_from_course_text() -> None:
    reviewed = _reviewed_draft()
    record_id = _record_id(reviewed)
    scaffold = _plan_scaffold(record_id)
    assignments = scaffold["course_assignments"]
    assert isinstance(assignments, list)
    assignments[0].pop("department")

    materialized = materialize_catalog(reviewed, scaffold, _evidence_mapping(record_id))

    courses = materialized["courses"]
    assert isinstance(courses, list) and len(courses) == 1
    assert courses[0]["department"] is None


def test_upstream_review_quarantine_is_preserved_and_not_misclassified_as_orphan() -> None:
    source = _source()
    draft = extract_catalog_draft(
        ParsedDocument(
            path=Path("public-curriculum.pdf"),
            elements=[
                DocumentElement(
                    "table",
                    "\n".join(
                        [
                            "| 课程代码 | 课程名称 | 学分 | 开课学期 | 课程性质 | 课程模块 |",
                            "| --- | --- | --- | --- | --- | --- |",
                            "| CS101 | 程序设计基础 | 3 | 第1学期 | 必修 | 专业基础模块 |",
                        ]
                    ),
                    page=7,
                )
            ],
            page_count=7,
        ),
        source=source,
    )
    raw_courses = draft["courses"]
    assert isinstance(raw_courses, list)
    record_id = raw_courses[0]["record_id"]
    assert isinstance(record_id, str)
    reviewed = apply_review_ledger(
        draft,
        [
            {
                "record_id": record_id,
                "decision": "quarantine",
                "reviewer": "catalog-reviewer",
                "reviewed_at": "2026-08-19T12:00:00+00:00",
                "field_updates": {},
            }
        ],
    )

    materialized = materialize_catalog(
        reviewed, _plan_scaffold(record_id), _evidence_mapping(record_id)
    )

    assert materialized["courses"] == []
    metadata = materialized["materialization"]
    assert metadata["upstream_quarantine"][0]["record_id"] == record_id
    assert metadata["counts"]["upstream_quarantine_records"] == 1
    assert metadata["counts"]["assignment_records_for_upstream_quarantine"] == 1
    assert metadata["counts"]["evidence_mapping_records_for_upstream_quarantine"] == 1


def test_cli_records_raw_input_hashes_and_writes_a_separate_quarantine_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reviewed = _reviewed_draft()
    record_id = _record_id(reviewed)
    reviewed_path = tmp_path / "reviewed.json"
    scaffold_path = tmp_path / "plan.json"
    mapping_path = tmp_path / "mapping.json"
    output_path = tmp_path / "curriculum_catalog.json"
    quarantine_path = tmp_path / "quarantine.json"
    for path, payload in (
        (reviewed_path, reviewed),
        (scaffold_path, _plan_scaffold(record_id)),
        (mapping_path, _evidence_mapping(record_id)),
    ):
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    assert (
        materialize_cli(
            [
                "--reviewed-draft",
                str(reviewed_path),
                "--plan-scaffold",
                str(scaffold_path),
                "--evidence-mapping",
                str(mapping_path),
                "--output",
                str(output_path),
                "--quarantine-report",
                str(quarantine_path),
            ]
        )
        == 0
    )
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["output"] == str(output_path)
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["materialization"]["input_file_hashes"] == {
        "reviewed_draft_file_sha256": hashlib.sha256(reviewed_path.read_bytes()).hexdigest(),
        "plan_scaffold_file_sha256": hashlib.sha256(scaffold_path.read_bytes()).hexdigest(),
        "evidence_mapping_file_sha256": hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
    }
    report = json.loads(quarantine_path.read_text(encoding="utf-8"))
    assert report["quarantine"] == []
    assert report["upstream_quarantine"] == []


def test_cli_fail_on_quarantine_never_exposes_a_partial_catalog(tmp_path: Path) -> None:
    reviewed = _reviewed_draft()
    record_id = _record_id(reviewed)
    reviewed_path = tmp_path / "reviewed.json"
    scaffold_path = tmp_path / "plan.json"
    mapping_path = tmp_path / "mapping.json"
    output_path = tmp_path / "curriculum_catalog.json"
    for path, payload in (
        (reviewed_path, reviewed),
        (scaffold_path, _plan_scaffold(record_id)),
        (mapping_path, {"mappings": []}),
    ):
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        materialize_cli(
            [
                "--reviewed-draft",
                str(reviewed_path),
                "--plan-scaffold",
                str(scaffold_path),
                "--evidence-mapping",
                str(mapping_path),
                "--output",
                str(output_path),
                "--fail-on-quarantine",
            ]
        )

    assert error.value.code == 2
    assert not output_path.exists()


def test_orphan_mapping_is_rejected_instead_of_being_used_as_a_textual_fallback() -> None:
    reviewed = _reviewed_draft()
    record_id = _record_id(reviewed)
    mapping = _evidence_mapping(record_id)
    mappings = mapping["mappings"]
    assert isinstance(mappings, list)
    unrelated = deepcopy(mappings[0])
    unrelated["record_id"] = "course_unrelated_but_textually_similar"
    mapping["mappings"] = [unrelated]

    with pytest.raises(CatalogMaterializationError, match="orphan record_id"):
        materialize_catalog(reviewed, _plan_scaffold(record_id), mapping)
