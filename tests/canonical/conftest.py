from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from academic.database import build_database
from agent.factory import build_runtime
from ingest.sources import SOURCE_FIELDS

CURRICULUM_SHA = "a" * 64
POLICY_SHA = "b" * 64


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


@pytest.fixture()
def canonical_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Canonical unit tests exercise the explicit no-model fallback. Production
    # composition defaults to artifact-backed hybrid retrieval.
    monkeypatch.setenv("SWUFE_RETRIEVAL_MODE", "lexical")
    sources = tmp_path / "sources.csv"
    with sources.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_FIELDS)
        writer.writeheader()
        writer.writerows(
            (
                {
                    "file": "test-curriculum.pdf",
                    "doc_title": "测试培养方案",
                    "level": "院级",
                    "college": "测试学院",
                    "cohort": "2024",
                    "year": "2024",
                    "status": "现行",
                    "page_url": "https://example.test/plan.pdf#page=1",
                    "file_url": "https://example.test/plan.pdf",
                    "collected_at": "2026-01-01",
                    "doc_type": "curriculum",
                    "topics": "",
                    "source_sha256": CURRICULUM_SHA,
                    "authenticity_status": "verified",
                },
                {
                    "file": "test-policy.pdf",
                    "doc_title": "测试学校政策",
                    "level": "校级",
                    "college": "全校",
                    "cohort": "不限",
                    "year": "2024",
                    "status": "现行",
                    "page_url": "https://example.test/policy.pdf#page=1",
                    "file_url": "https://example.test/policy.pdf",
                    "collected_at": "2026-01-01",
                    "doc_type": "policy",
                    "topics": "转专业",
                    "source_sha256": POLICY_SHA,
                    "authenticity_status": "verified",
                },
            )
        )
    chunks = tmp_path / "chunks.jsonl"
    entries = [
        {"chunk_id": "test-plan-1", "text": "测试专业X专业选修课最低要求为 3 学分。专业选修课包含测试算法（TST101），选修，第1学期开设，3学分。", "doc_title": "测试培养方案", "article": "测试专业X / 原文件第1页", "cohort": "2024", "is_table": False, "review_status": "verified", "doc_type": "curriculum", "topics": [], "source_sha256": CURRICULUM_SHA, "extraction_quality": "verified"},
        {"chunk_id": "test-plan-2", "text": "测试专业Y专业选修课最低要求为 4 学分。专业选修课包含测试系统（TST201），选修，第1学期开设，4学分。", "doc_title": "测试培养方案", "article": "测试专业Y / 原文件第2页", "cohort": "2024", "is_table": False, "review_status": "verified", "doc_type": "curriculum", "topics": [], "source_sha256": CURRICULUM_SHA, "extraction_quality": "verified"},
        {"chunk_id": "test-policy-1", "text": "学生申请转专业应提交材料，具体条件以学校转专业管理办法为准；跨专业选修课程须遵守学校选课规定。", "doc_title": "测试学校政策", "article": "学校政策 / 原文件第1页", "cohort": "不限", "is_table": False, "review_status": "verified", "doc_type": "policy", "topics": ["转专业"], "source_sha256": POLICY_SHA, "extraction_quality": "verified"},
    ]
    chunks.write_text("\n".join(json.dumps(value, ensure_ascii=False) for value in entries) + "\n", encoding="utf-8")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({
        "catalog_version": "fixture-1", "plans": [
            {"college": "测试学院", "cohort": "2024", "major": "测试专业X", "source_title": "测试培养方案", "modules": [{"name": "专业选修课", "required_credits": 3, "listed_credits": 3, "rule_text": "最低3学分", "evidence": {"chunk_id": "test-plan-1", "article": "原文件第1页"}, "field_verification": {"module": _field_verification(source_sha256=CURRICULUM_SHA, chunk_id="test-plan-1", page=1, row=1, cell="A1", span="专业选修课"), "required_credits": _field_verification(source_sha256=CURRICULUM_SHA, chunk_id="test-plan-1", page=1, row=1, cell="B1", span="最低要求为 3 学分")}}]},
            {"college": "测试学院", "cohort": "2024", "major": "测试专业Y", "source_title": "测试培养方案", "modules": [{"name": "专业选修课", "required_credits": 4, "listed_credits": 4, "rule_text": "最低4学分", "evidence": {"chunk_id": "test-plan-2", "article": "原文件第2页"}, "field_verification": {"module": _field_verification(source_sha256=CURRICULUM_SHA, chunk_id="test-plan-2", page=2, row=1, cell="A1", span="专业选修课"), "required_credits": _field_verification(source_sha256=CURRICULUM_SHA, chunk_id="test-plan-2", page=2, row=1, cell="B1", span="最低要求为 4 学分")}}]},
        ], "courses": [
            {"code": "TST101", "name": "测试算法", "credits": 3, "nature": "选修", "semester": "1", "department": "测试学院", "college": "测试学院", "cohort": "2024", "major": "测试专业X", "module": "专业选修课", "source_title": "测试培养方案", "page": 1, "source_row": 1, "evidence": {"chunk_id": "test-plan-1"}, "field_verification": {"module": _field_verification(source_sha256=CURRICULUM_SHA, chunk_id="test-plan-1", page=1, row=2, cell="A2", span="专业选修课"), "credits": _field_verification(source_sha256=CURRICULUM_SHA, chunk_id="test-plan-1", page=1, row=2, cell="B2", span="3学分"), "semester": _field_verification(source_sha256=CURRICULUM_SHA, chunk_id="test-plan-1", page=1, row=2, cell="C2", span="第1学期开设"), "nature": _field_verification(source_sha256=CURRICULUM_SHA, chunk_id="test-plan-1", page=1, row=2, cell="D2", span="选修")}},
            {"code": "TST201", "name": "测试系统", "credits": 4, "nature": "选修", "semester": "1", "department": "测试学院", "college": "测试学院", "cohort": "2024", "major": "测试专业Y", "module": "专业选修课", "source_title": "测试培养方案", "page": 2, "source_row": 1, "evidence": {"chunk_id": "test-plan-2"}, "field_verification": {"module": _field_verification(source_sha256=CURRICULUM_SHA, chunk_id="test-plan-2", page=2, row=2, cell="A2", span="专业选修课"), "credits": _field_verification(source_sha256=CURRICULUM_SHA, chunk_id="test-plan-2", page=2, row=2, cell="B2", span="4学分"), "semester": _field_verification(source_sha256=CURRICULUM_SHA, chunk_id="test-plan-2", page=2, row=2, cell="C2", span="第1学期开设"), "nature": _field_verification(source_sha256=CURRICULUM_SHA, chunk_id="test-plan-2", page=2, row=2, cell="D2", span="选修")}},
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
                "original_title": "test-curriculum.pdf",
                "corrected_title": "测试培养方案",
                "decision": "include",
                "reviewer": "canonical-fixture",
                "method": "manual",
                "reviewed_at": "2026-01-01T00:00:00+00:00",
            }
        )
        writer.writerow(
            {
                "original_title": "test-policy.pdf",
                "corrected_title": "测试学校政策",
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
