from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from contracts import RETRIEVED_CHUNK_FIELDS
from retrieval.embed import HashingEncoder
from retrieval.index import build_index, load_index
from retrieval.pipeline import AdvancedRetriever, RetrievalTuning
from retrieval.query import analyze_query
from retrieval.retriever import HybridRetriever


FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "chunks.jsonl"


class PreferReplacementReranker:
    def score(self, query: str, documents: list[str]) -> np.ndarray:
        return np.asarray(
            [1.0 if "课程替代" in document else 0.0 for document in documents],
            dtype=np.float32,
        )


class AdvancedRetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.encoder = HashingEncoder(512)
        build_index(
            FIXTURE_PATH,
            cls.temporary.name,
            cls.encoder,
            allow_test_backend=True,
        )
        bundle = load_index(
            FIXTURE_PATH,
            cls.temporary.name,
            cls.encoder,
            allow_test_backend=True,
        )
        cls.core = HybridRetriever(bundle, cls.encoder)
        cls.pipeline = AdvancedRetriever(cls.core)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_domain_query_expansion_is_deterministic(self) -> None:
        analysis = analyze_query("我挂科了还能保研吗")
        self.assertIn("不及格", analysis.expanded)
        self.assertIn("重修", analysis.expanded)
        self.assertIn("推免", analysis.expanded)
        self.assertIn("推荐免试", analysis.expanded)

    def test_colloquial_query_finds_policy_clause(self) -> None:
        results = self.pipeline.retrieve(
            "挂科还能保研吗",
            college="计算机与人工智能学院",
            cohort="2023",
        )
        self.assertEqual(results[0]["chunk_id"], "fixture_it_recommend_013")

    def test_title_and_article_terms_affect_second_stage_ranking(self) -> None:
        results = self.pipeline.retrieve(
            "第五条资格与成绩",
            college="计算机与人工智能学院",
            cohort="2023",
        )
        self.assertEqual(results[0]["chunk_id"], "fixture_it_recommend_013")

    def test_course_code_exact_signal_is_preserved(self) -> None:
        results = self.pipeline.retrieve(
            "CS205是什么课",
            college="计算机与人工智能学院",
            cohort="2023",
        )
        self.assertEqual(results[0]["chunk_id"], "fixture_it_table_011")

    def test_pluggable_reranker_can_reorder_candidate_window(self) -> None:
        tuning = RetrievalTuning(
            candidate_k=20,
            dense_weight=0.05,
            lexical_weight=0.05,
            rerank_weight=0.85,
            rank_prior_weight=0.05,
        )
        pipeline = AdvancedRetriever(
            self.core, reranker=PreferReplacementReranker(), tuning=tuning
        )
        result = pipeline.retrieve("学校规定", top_k=1)[0]
        self.assertEqual(result["chunk_id"], "fixture_school_credit_004")

    def test_scope_and_output_contract_remain_frozen(self) -> None:
        results = self.pipeline.retrieve(
            "金融学院推免综合成绩",
            top_k=10,
            college="计算机与人工智能学院",
            cohort="2023",
        )
        self.assertTrue(results)
        for result in results:
            self.assertEqual(set(result), set(RETRIEVED_CHUNK_FIELDS))
            self.assertNotEqual(result["college"], "金融学院")
            self.assertEqual(result["status"], "现行")
            self.assertIn(result["cohort"], {"不限", "2023"})


if __name__ == "__main__":
    unittest.main()
