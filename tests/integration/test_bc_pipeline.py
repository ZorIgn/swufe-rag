from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from contracts import ANSWER_FIELDS, CITATION_FIELDS, RETRIEVED_CHUNK_FIELDS
from generation.service import GenerationService
from retrieval.embed import HashingEncoder
from retrieval.index import build_index, load_index
from retrieval.retriever import HybridRetriever
from tests.generation.helpers import FakeClient


FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "chunks.jsonl"


class BCPipelineTests(unittest.TestCase):
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
        cls.retriever = HybridRetriever(bundle, cls.encoder)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_policy_fact_flows_from_retrieval_to_traceable_answer(self) -> None:
        chunks = self.retriever.retrieve(
            "不得有不及格课程记录70%30%",
            college="计算机与人工智能学院",
            cohort="2023",
        )
        self.assertEqual(chunks[0]["chunk_id"], "fixture_it_recommend_013")
        client = FakeClient(
            [
                "本科阶段不得有不及格课程记录[1]。"
                "综合成绩由学业成绩70%和综合素质30%构成[1]。"
            ]
        )
        result = GenerationService(client).answer("挂过科还能申请推免吗", chunks)
        self.assertFalse(result["refused"])
        self.assertEqual(result["citations"][0]["chunk_id"], chunks[0]["chunk_id"])
        self.assertIn(result["citations"][0]["quote"], chunks[0]["text"])

    def test_course_table_number_is_grounded(self) -> None:
        chunks = self.retriever.retrieve(
            "CS205机器学习导论",
            college="计算机与人工智能学院",
            cohort="2023",
        )
        client = FakeClient(
            ["机器学习导论课程代码为CS205，学分为3学分，类别为专业选修[1]。"]
        )
        result = GenerationService(client).answer("CS205是什么课", chunks)
        self.assertFalse(result["refused"])
        self.assertEqual(result["citations"][0]["chunk_id"], "fixture_it_table_011")

    def test_school_and_college_sources_can_be_combined(self) -> None:
        chunks = self.retriever.retrieve(
            "推免应届毕业生不及格课程记录",
            college="计算机与人工智能学院",
            cohort="2023",
        )
        ids = [chunk["chunk_id"] for chunk in chunks]
        school_marker = ids.index("fixture_school_recommend_005") + 1
        college_marker = ids.index("fixture_it_recommend_013") + 1
        client = FakeClient(
            [
                f"申请人应为应届毕业生[{school_marker}]。"
                f"本科阶段不得有不及格课程记录[{college_marker}]。"
            ]
        )
        result = GenerationService(client).answer("推免资格是什么", chunks)
        self.assertFalse(result["refused"])
        self.assertEqual(
            {citation["chunk_id"] for citation in result["citations"]},
            {"fixture_school_recommend_005", "fixture_it_recommend_013"},
        )

    def test_cross_college_source_cannot_reach_generation(self) -> None:
        chunks = self.retriever.retrieve(
            "金融学院推免综合成绩80%科研竞赛20%",
            college="计算机与人工智能学院",
            cohort="2023",
        )
        self.assertNotIn("金融学院", {chunk["college"] for chunk in chunks})
        self.assertNotIn(
            "fixture_fin_recommend_019", {chunk["chunk_id"] for chunk in chunks}
        )

    def test_out_of_domain_query_is_refused_before_llm(self) -> None:
        chunks = self.retriever.retrieve("食堂几点关门", top_k=5)
        self.assertLess(chunks[0]["score"], 0.35)
        client = FakeClient(["不应被使用"])
        result = GenerationService(client).answer("食堂几点关门", chunks)
        self.assertTrue(result["refused"])
        self.assertEqual(client.calls, [])

    def test_pipeline_keeps_exact_public_shapes(self) -> None:
        chunks = self.retriever.retrieve("重修成绩如实记载", top_k=1)
        self.assertEqual(set(chunks[0]), set(RETRIEVED_CHUNK_FIELDS))
        result = GenerationService(
            FakeClient(["重修成绩如实记载[1]。"])
        ).answer("重修成绩怎么记载", chunks)
        self.assertEqual(set(result), set(ANSWER_FIELDS))
        self.assertEqual(set(result["citations"][0]), set(CITATION_FIELDS))


if __name__ == "__main__":
    unittest.main()

