from __future__ import annotations

from pathlib import Path
import unittest

from contracts import CHUNK_FIELDS
from ingest.chunk import build_chunks
from ingest.models import DocumentElement, ParsedDocument, SourceRecord


class ChunkingTests(unittest.TestCase):
    def _source(self) -> SourceRecord:
        return SourceRecord(
            file="school/policy.txt",
            doc_title="本科课程管理办法",
            level="校级",
            college="全校",
            cohort="不限",
            year=2026,
            status="现行",
            page_url="https://jwc.swufe.edu.cn/policy",
            file_url="https://jwc.swufe.edu.cn/policy.txt",
            collected_at="2026-07-14",
        )

    def test_articles_split_and_tables_remain_atomic(self) -> None:
        parsed = ParsedDocument(
            Path("policy.txt"),
            [
                DocumentElement("heading", "第一章 总则"),
                DocumentElement("heading", "第一条 适用范围"),
                DocumentElement("paragraph", "本办法适用于全日制本科生。"),
                DocumentElement("heading", "第二条 学分要求"),
                DocumentElement("paragraph", "学生应修满规定学分。"),
                DocumentElement("table", "| 课程 | 学分 |\n| --- | --- |\n| 人工智能 | 3 |"),
            ],
        )
        chunks = build_chunks(parsed, self._source(), chunk_max_len=220)
        self.assertTrue(any(chunk["article"].endswith("第一条") for chunk in chunks))
        table = next(chunk for chunk in chunks if chunk["is_table"])
        self.assertIn("| 人工智能 | 3 |", table["text"])
        self.assertEqual(set(table), set(CHUNK_FIELDS))
        self.assertEqual(len({chunk["chunk_id"] for chunk in chunks}), len(chunks))

    def test_long_clause_splits_at_budget_with_stable_ids(self) -> None:
        parsed = ParsedDocument(
            Path("policy.txt"),
            [DocumentElement("paragraph", "第一条 " + "本条规定适用于学生。" * 80)],
        )
        first = build_chunks(parsed, self._source(), chunk_max_len=200)
        second = build_chunks(parsed, self._source(), chunk_max_len=200)
        self.assertGreater(len(first), 2)
        self.assertEqual(
            [item["chunk_id"] for item in first],
            [item["chunk_id"] for item in second],
        )
        self.assertTrue(all(len(item["text"]) <= 200 for item in first))

    def test_numbered_items_become_independent_evidence(self) -> None:
        parsed = ParsedDocument(
            Path("policy.txt"),
            [
                DocumentElement("heading", "注意事项"),
                DocumentElement("paragraph", "1. 须在开考前2小时申请。\n2 · 未获批准按旷考处理。"),
            ],
        )
        chunks = build_chunks(parsed, self._source())
        self.assertEqual([item["article"] for item in chunks], ["注意事项 / 第1项", "注意事项 / 第2项"])

    def test_pdf_table_uses_page_label_instead_of_previous_heading(self) -> None:
        parsed = ParsedDocument(
            Path("policy.pdf"),
            [
                DocumentElement("heading", "三、培养目标", page=1),
                DocumentElement("table", "| 专业 | 学分 |\n| --- | --- |\n| 计算机 | 151 |", page=1),
            ],
        )
        table = next(item for item in build_chunks(parsed, self._source()) if item["is_table"])
        self.assertEqual(table["article"], "第1页表格")


if __name__ == "__main__":
    unittest.main()
