"""Public, review-gated PDF/table-to-catalog draft pipeline contracts."""

from __future__ import annotations

import csv
import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

from ingest.catalog import (
    CatalogExtractionError,
    CatalogReviewError,
    apply_review_ledger,
    extract_catalog_draft,
)
from ingest.models import DocumentElement, ParsedDocument, SourceRecord
from ingest.parse import extraction_quality_ledger, parse_document
from ingest.pipeline import ingest_catalog_draft as run_catalog_draft
from ingest.sources import SOURCE_FIELDS


def _source() -> SourceRecord:
    return SourceRecord(
        file="public-curriculum.pdf",
        doc_title="公开培养方案测试",
        level="院级",
        college="测试学院",
        cohort="2024",
        year=2024,
        status="现行",
        page_url="https://example.test/public-curriculum.pdf#page=1",
        file_url="https://example.test/public-curriculum.pdf",
        collected_at="2026-08-19",
        doc_type="curriculum",
        source_sha256="a" * 64,
        authenticity_status="review_required",
    )


def _table(rows: str) -> str:
    return "\n".join(
        [
            "| 课程代码 | 课程名称 | 学分 | 开课学期 | 课程性质 | 课程模块 |",
            "| --- | --- | --- | --- | --- | --- |",
            rows,
        ]
    )


def _document(rows: str) -> ParsedDocument:
    return ParsedDocument(
        path=Path("public-curriculum.pdf"),
        elements=[DocumentElement("table", _table(rows), page=7)],
        page_count=7,
    )


def test_table_markdown_emits_schema_draft_and_field_level_lineage() -> None:
    source = _source()
    document = _document("| CS101 | 程序设计基础 | 3 | 第1学期 | 必修 | 专业基础模块 |")

    draft = extract_catalog_draft(document, source=source)

    assert draft["review_status"] == "review_required"
    assert draft["counts"] == {
        "table_count": 1,
        "course_draft_count": 1,
        "quarantine_count": 0,
        "verified_course_count": 0,
    }
    course = draft["courses"][0]
    assert {
        field: course[field]
        for field in ("code", "name", "credits", "semester", "nature", "module")
    } == {
        "code": "CS101",
        "name": "程序设计基础",
        "credits": 3.0,
        "semester": "1",
        "nature": "必修",
        "module": "专业基础模块",
    }
    code_lineage = course["field_lineage"]["code"][0]
    assert code_lineage["source_sha256"] == "a" * 64
    assert code_lineage["page"] == 7
    assert (code_lineage["table"], code_lineage["row"], code_lineage["cell"]) == (1, 3, 1)
    start = code_lineage["char_span"]["start"]
    end = code_lineage["char_span"]["end"]
    assert document.elements[0].text[start:end] == "CS101"
    page_quality = next(item for item in draft["quality_ledger"] if item["page"] == 7)
    assert page_quality["table_status"] == "ok"
    assert course["review_status"] == "review_required"


def test_credit_or_semester_conflicts_are_quarantined_not_silently_selected() -> None:
    draft = extract_catalog_draft(
        _document(
            "\n".join(
                [
                    "| CS101 | 程序设计基础 | 3 | 1 | 必修 | 专业基础模块 |",
                    "| CS101 | 程序设计基础 | 4 | 2 | 必修 | 专业基础模块 |",
                ]
            )
        ),
        source=_source(),
    )

    assert draft["courses"] == []
    conflicts = [
        item for item in draft["quarantine"] if item["reason"] == "conflicting_duplicate_course"
    ]
    assert len(conflicts) == 2
    assert {"credits", "semester"}.issubset(set(conflicts[0]["conflict_fields"]))


def test_ambiguous_or_unknown_required_columns_are_quarantined() -> None:
    document = ParsedDocument(
        path=Path("public-curriculum.pdf"),
        elements=[
            DocumentElement(
                "table",
                "\n".join(
                    [
                        "| 课程代码 | 课程编号 | 课程名称 | 学分 | 开课学期 | 课程性质 | 课程模块 |",
                        "| --- | --- | --- | --- | --- | --- | --- |",
                        "| CS101 | CS101 | 程序设计基础 | 3 | 1 | 必修 | 专业基础模块 |",
                    ]
                ),
                page=3,
            )
        ],
        page_count=3,
    )

    draft = extract_catalog_draft(document, source=_source())

    assert draft["courses"] == []
    issue = draft["quarantine"][0]
    assert issue["reason"] == "unmapped_or_ambiguous_columns"
    assert {"reason": "ambiguous_column", "field": "code", "header_cells": [1, 2]} in issue[
        "details"
    ]["issues"]


def test_non_catalog_document_type_is_quarantined_before_table_extraction() -> None:
    source = replace(_source(), doc_type="policy")

    draft = extract_catalog_draft(
        _document("| CS101 | 程序设计基础 | 3 | 1 | 必修 | 专业基础模块 |"), source=source
    )

    assert draft["courses"] == []
    assert draft["quarantine"][0]["reason"] == "source_document_type_not_catalog"


def test_explicit_reviewer_edit_records_a_field_diff_and_preserves_source_lineage() -> None:
    draft = extract_catalog_draft(
        _document("| CS101 | 程序设计基础 | 3 | 1 | 必修 | 专业基础模块 |"), source=_source()
    )
    record_id = draft["courses"][0]["record_id"]

    reviewed = apply_review_ledger(
        draft,
        [
            {
                "record_id": record_id,
                "decision": "edit",
                "reviewer": "catalog-reviewer",
                "reviewed_at": "2026-08-19T12:00:00+00:00",
                "field_updates": {"credits": "4"},
            }
        ],
    )

    course = reviewed["courses"][0]
    assert course["review_status"] == "verified"
    assert course["credits"] == 4.0
    assert course["field_lineage"]["credits"][0]["raw_value"] == "3"
    assert course["field_lineage"]["credits"][1] == {
        "origin": "reviewer",
        "reviewer": "catalog-reviewer",
        "reviewed_at": "2026-08-19T12:00:00+00:00",
        "prior_value": 3.0,
        "review_value": 4.0,
    }
    assert reviewed["review_diff"][0]["changes"] == [
        {"field": "credits", "before": 3.0, "after": 4.0}
    ]


def test_reviewer_edit_cannot_reintroduce_a_conflicting_course_identity() -> None:
    draft = extract_catalog_draft(
        _document(
            "\n".join(
                [
                    "| CS101 | 程序设计基础 | 3 | 1 | 必修 | 专业基础模块 |",
                    "| CS102 | 数据结构 | 4 | 2 | 必修 | 专业基础模块 |",
                ]
            )
        ),
        source=_source(),
    )
    record_id = next(
        course["record_id"] for course in draft["courses"] if course["code"] == "CS102"
    )

    with pytest.raises(CatalogReviewError, match="creates conflicting course code CS101"):
        apply_review_ledger(
            draft,
            [
                {
                    "record_id": record_id,
                    "decision": "edit",
                    "reviewer": "catalog-reviewer",
                    "reviewed_at": "2026-08-19T12:00:00+00:00",
                    "field_updates": {"code": "CS101"},
                }
            ],
        )


class _FailingTablePage:
    images: list[object] = []

    def extract_text(self) -> str:
        return "正文" * 100

    def extract_tables(self) -> list[list[list[object]]]:
        raise RuntimeError("table backend failed")


class _FakePdf:
    pages = [_FailingTablePage()]

    def __enter__(self) -> _FakePdf:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        del exc_type, exc, traceback
        return False


def test_pdf_table_exception_enters_quality_ledger_and_blocks_catalog(
    monkeypatch, tmp_path: Path
) -> None:
    source_path = tmp_path / "public-curriculum.pdf"
    source_path.write_bytes(b"not-a-real-pdf; parser dependency is monkeypatched")
    monkeypatch.setitem(
        sys.modules, "pdfplumber", types.SimpleNamespace(open=lambda path: _FakePdf())
    )

    parsed = parse_document(source_path)
    ledger = extraction_quality_ledger(parsed)
    failed_page = next(item for item in ledger if item["page"] == 1)

    assert failed_page["table_status"] == "failed"
    assert failed_page["critical"] is True
    assert "table_extraction_failed" in failed_page["issues"]
    with pytest.raises(CatalogExtractionError, match="critical table extraction failure"):
        extract_catalog_draft(parsed, source=_source())


def test_quality_ledger_preserves_ocr_and_docx_image_warnings() -> None:
    ledger = extraction_quality_ledger(
        ParsedDocument(
            path=Path("mixed-input.docx"),
            elements=[DocumentElement("paragraph", "正文", page=1)],
            page_count=2,
            warnings=[
                "quality:ocr_used:page=2",
                "quality:docx_inline_image:detail=count_1",
            ],
        )
    )

    assert next(item for item in ledger if item["page"] == 2)["ocr_status"] == "used"
    global_entry = next(item for item in ledger if item["page"] is None)
    assert global_entry["ocr_status"] == "not_performed"
    assert "docx_inline_image" in global_entry["issues"]


def test_catalog_pipeline_fails_closed_before_writing_when_parser_reports_critical_table_failure(
    monkeypatch, tmp_path: Path
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "public-curriculum.pdf").write_bytes(b"fixture")
    sources = tmp_path / "sources.csv"
    row = {
        "file": "public-curriculum.pdf",
        "doc_title": "公开培养方案测试",
        "level": "院级",
        "college": "测试学院",
        "cohort": "2024",
        "year": "2024",
        "status": "现行",
        "page_url": "https://it.swufe.edu.cn/public-curriculum.pdf",
        "file_url": "https://it.swufe.edu.cn/public-curriculum.pdf",
        "collected_at": "2026-08-19",
        "doc_type": "curriculum",
        "topics": "",
        "source_sha256": "a" * 64,
        "authenticity_status": "review_required",
    }
    with sources.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_FIELDS)
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in SOURCE_FIELDS})

    def fake_parse(path: Path, *, ocr_provider: object | None = None) -> ParsedDocument:
        del ocr_provider
        return ParsedDocument(
            path=path,
            elements=[DocumentElement("paragraph", "保留的正文", page=1)],
            page_count=1,
            warnings=["quality:table_extraction_failed:page=1:critical=true:detail=RuntimeError"],
        )

    monkeypatch.setattr("ingest.pipeline.parse_document", fake_parse)
    output = tmp_path / "catalog-draft.json"

    with pytest.raises(CatalogExtractionError):
        run_catalog_draft(sources, raw_dir, output)

    assert not output.exists()


def test_catalog_pipeline_writes_only_review_required_drafts_and_quality_ledgers(
    monkeypatch, tmp_path: Path
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "public-curriculum.pdf").write_bytes(b"fixture")
    sources = tmp_path / "sources.csv"
    row = {
        "file": "public-curriculum.pdf",
        "doc_title": "公开培养方案测试",
        "level": "院级",
        "college": "测试学院",
        "cohort": "2024",
        "year": "2024",
        "status": "现行",
        "page_url": "https://it.swufe.edu.cn/public-curriculum.pdf",
        "file_url": "https://it.swufe.edu.cn/public-curriculum.pdf",
        "collected_at": "2026-08-19",
        "doc_type": "curriculum",
        "topics": "",
        "source_sha256": "a" * 64,
        "authenticity_status": "review_required",
    }
    with sources.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_FIELDS)
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in SOURCE_FIELDS})

    def fake_parse(path: Path, *, ocr_provider: object | None = None) -> ParsedDocument:
        del path, ocr_provider
        return _document("| CS101 | 程序设计基础 | 3 | 1 | 必修 | 专业基础模块 |")

    monkeypatch.setattr("ingest.pipeline.parse_document", fake_parse)
    output = tmp_path / "catalog-draft.json"
    quality = tmp_path / "quality-ledger.json"

    report = run_catalog_draft(sources, raw_dir, output, quality_ledger_path=quality)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert report["verified_course_count"] == 0
    assert payload["review_status"] == "review_required"
    assert payload["courses"][0]["review_status"] == "review_required"
    assert json.loads(quality.read_text(encoding="utf-8"))[0]["entries"]
