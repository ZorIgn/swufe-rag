from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from academic.database import build_database
from agent.factory import build_runtime


@pytest.fixture()
def canonical_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Canonical unit tests exercise the explicit no-model fallback. Production
    # composition defaults to artifact-backed hybrid retrieval.
    monkeypatch.setenv("SWUFE_RETRIEVAL_MODE", "lexical")
    sources = tmp_path / "sources.csv"
    with sources.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "doc_title", "level", "college", "cohort", "year", "status", "page_url", "file_url", "collected_at"])
        writer.writeheader()
        writer.writerow({"file": "test.pdf", "doc_title": "测试培养方案", "level": "院级", "college": "测试学院", "cohort": "2024", "year": "2024", "status": "现行", "page_url": "https://example.test/plan.pdf#page=1", "file_url": "https://example.test/plan.pdf", "collected_at": "2026-01-01"})
    chunks = tmp_path / "chunks.jsonl"
    entries = [
        {"chunk_id": "test-plan-1", "text": "测试专业X培养方案。测试专业X专业选修最低要求为 3 学分。", "doc_title": "测试培养方案", "article": "测试专业X / 原文件第1页", "cohort": "2024", "is_table": False, "page_url": "https://example.test/plan.pdf#page=1", "file_url": "https://example.test/plan.pdf", "review_status": "verified"},
        {"chunk_id": "test-plan-2", "text": "测试专业Y培养方案。测试专业Y专业选修最低要求为 4 学分。", "doc_title": "测试培养方案", "article": "测试专业Y / 原文件第2页", "cohort": "2024", "is_table": False, "page_url": "https://example.test/plan.pdf#page=2", "file_url": "https://example.test/plan.pdf", "review_status": "verified"},
        {"chunk_id": "test-policy-1", "text": "学生申请转专业应提交材料，具体条件以学校转专业管理办法为准；跨专业选修课程须遵守学校选课规定。", "doc_title": "测试培养方案", "article": "学校政策 / 原文件第3页", "cohort": "2024", "is_table": False, "page_url": "https://example.test/plan.pdf#page=3", "file_url": "https://example.test/plan.pdf", "review_status": "verified"},
    ]
    chunks.write_text("\n".join(json.dumps(value, ensure_ascii=False) for value in entries) + "\n", encoding="utf-8")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({
        "catalog_version": "fixture-1", "plans": [
            {"college": "测试学院", "cohort": "2024", "major": "测试专业X", "source_title": "测试培养方案", "modules": [{"name": "专业选修课", "required_credits": 3, "listed_credits": 3, "rule_text": "最低3学分", "evidence": {"chunk_id": "test-plan-1", "article": "原文件第1页"}}]},
            {"college": "测试学院", "cohort": "2024", "major": "测试专业Y", "source_title": "测试培养方案", "modules": [{"name": "专业选修课", "required_credits": 4, "listed_credits": 4, "rule_text": "最低4学分", "evidence": {"chunk_id": "test-plan-2", "article": "原文件第2页"}}]},
        ], "courses": [
            {"code": "TST101", "name": "测试算法", "credits": 3, "nature": "选修", "semester": "1", "department": "测试学院", "college": "测试学院", "cohort": "2024", "major": "测试专业X", "module": "专业选修课", "source_title": "测试培养方案", "page": 1, "source_row": 1, "evidence": {"chunk_id": "test-plan-1"}},
            {"code": "TST201", "name": "测试系统", "credits": 4, "nature": "选修", "semester": "1", "department": "测试学院", "college": "测试学院", "cohort": "2024", "major": "测试专业Y", "module": "专业选修课", "source_title": "测试培养方案", "page": 2, "source_row": 1, "evidence": {"chunk_id": "test-plan-2"}},
        ]
    }, ensure_ascii=False), encoding="utf-8")
    aliases = tmp_path / "aliases.json"
    aliases.write_text(json.dumps({"program_aliases": {"X专业": "测试专业X", "Y专业": "测试专业Y"}, "module_aliases": {"专选": "专业选修课"}, "course_aliases": {}}, ensure_ascii=False), encoding="utf-8")
    source_review = tmp_path / "source_review.csv"
    with source_review.open("w", encoding="utf-8", newline="") as handle:
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
                "original_title": "test.pdf",
                "corrected_title": "测试培养方案",
                "decision": "include",
                "reviewer": "canonical-fixture",
                "method": "manual",
                "reviewed_at": "2026-01-01T00:00:00+00:00",
            }
        )
    database = tmp_path / "academic.sqlite3"
    build_database(
        database,
        catalog_path=catalog,
        sources_path=sources,
        chunks_path=chunks,
        aliases_path=aliases,
        source_review_path=source_review,
    )
    return build_runtime(database)
