from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from eval.production_generation_eval import evaluate


def chunk() -> dict:
    return {
        "chunk_id": "real_001",
        "text": "缓考申请最迟应在课程开考前 2 小时提交。",
        "doc_title": "缓考规定",
        "article": "第一条",
        "level": "校级",
        "college": "全校",
        "cohort": "不限",
        "year": 2026,
        "status": "现行",
        "page_url": "https://jwc.swufe.edu.cn/page",
        "file_url": "https://jwc.swufe.edu.cn/file.pdf",
        "is_table": False,
        "score": 0.8,
    }


class FakeRetriever:
    def retrieve(self, query, top_k=5, college=None, cohort=None):
        return [chunk()]


class FakeGeneration:
    client = type("Client", (), {"model_spec": "fake-live-model"})()

    def answer(self, query, chunks):
        return {
            "answer_md": "应在课程开考前 2 小时提交[1]。",
            "citations": [
                {
                    "marker": 1,
                    "chunk_id": "real_001",
                    "doc_title": "缓考规定",
                    "article": "第一条",
                    "quote": "缓考申请最迟应在课程开考前 2 小时提交。",
                    "page_url": "https://jwc.swufe.edu.cn/page",
                    "file_url": "https://jwc.swufe.edu.cn/file.pdf",
                }
            ],
            "refused": False,
        }


class ProductionGenerationEvalTests(unittest.TestCase):
    def test_injected_pipeline_scores_support_and_exact_citations(self) -> None:
        cases = [
            {
                "id": "case_1",
                "question": "缓考最迟什么时候提交？",
                "college": "计算机与人工智能学院",
                "cohort": "2024",
                "category": "缓考",
                "answer_must_contain": ["开考前 2 小时"],
                "should_refuse": False,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
            report = evaluate(
                path,
                retriever=FakeRetriever(),
                generation=FakeGeneration(),
            )
        self.assertEqual(report["model"], "fake-live-model")
        self.assertEqual(report["refusal_accuracy"], 1.0)
        self.assertEqual(report["answer_support_accuracy"], 1.0)
        self.assertEqual(report["citation_integrity_accuracy"], 1.0)

    def test_limit_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "limit must be positive"):
            evaluate(
                "eval/real_dev_queries.json",
                limit=0,
                retriever=FakeRetriever(),
                generation=FakeGeneration(),
            )


if __name__ == "__main__":
    unittest.main()
