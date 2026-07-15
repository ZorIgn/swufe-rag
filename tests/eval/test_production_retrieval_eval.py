from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from eval.production_retrieval_eval import evaluate


class FakeRetriever:
    def retrieve(self, query, top_k=5, college=None, cohort=None):
        return [
            {
                "chunk_id": "plan_001",
                "text": "专业选修课模块选修不低于 8 学分课程。",
                "doc_title": "计算机类专业人才培养方案（2024级）",
                "article": "专业课程板块",
                "level": "院级",
                "college": "计算机与人工智能学院",
                "cohort": "2024",
                "year": 2024,
                "status": "现行",
                "page_url": "https://jwc.swufe.edu.cn/page",
                "file_url": "https://jwc.swufe.edu.cn/file.pdf",
                "is_table": False,
                "score": 0.8,
            }
        ]


class ProductionRetrievalEvalTests(unittest.TestCase):
    def test_report_checks_answer_terms_inside_top_five_evidence(self) -> None:
        cases = [
            {
                "id": "case_1",
                "question": "2024级专业选修课至少修多少学分？",
                "college": "计算机与人工智能学院",
                "cohort": "2024",
                "expected_docs": ["计算机类专业人才培养方案（2024级）"],
                "answer_must_contain": ["不低于 8 学分"],
                "should_refuse": False,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
            report = evaluate(path, retriever=FakeRetriever())
        self.assertEqual(report["retrieval_recall_at_5"], 1.0)
        self.assertEqual(report["evidence_support_at_5"], 1.0)
        self.assertTrue(report["rows"][0]["evidence_support_at_5"])


if __name__ == "__main__":
    unittest.main()
