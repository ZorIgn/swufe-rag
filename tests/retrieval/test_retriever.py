from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from contracts import RETRIEVED_CHUNK_FIELDS
from retrieval.embed import HashingEncoder
from retrieval.index import build_index, load_index
from retrieval.retriever import HybridRetriever


FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "chunks.jsonl"


class RetrieverTests(unittest.TestCase):
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

    def test_course_code_is_found_by_hybrid_search(self) -> None:
        results = self.retriever.retrieve(
            "CS205 机器学习导论",
            college="计算机与人工智能学院",
            cohort="2023",
        )
        self.assertEqual(results[0]["chunk_id"], "fixture_it_table_011")

    def test_scope_filter_has_no_cross_college_or_history_pollution(self) -> None:
        results = self.retriever.retrieve(
            "推免综合成绩学业成绩科研竞赛",
            top_k=20,
            college="计算机与人工智能学院",
            cohort="2023",
        )
        self.assertTrue(results)
        for result in results:
            self.assertEqual(result["status"], "现行")
            self.assertTrue(
                result["level"] == "校级"
                or result["college"] == "计算机与人工智能学院"
            )
            self.assertIn(result["cohort"], {"不限", "2023"})
        self.assertNotIn("fixture_fin_recommend_019", {r["chunk_id"] for r in results})
        self.assertNotIn("fixture_it_recommend_old_014", {r["chunk_id"] for r in results})

    def test_filter_is_applied_before_top_k(self) -> None:
        result = self.retriever.retrieve(
            "金融学院推免综合成绩80%科研竞赛20%",
            top_k=1,
            college="计算机与人工智能学院",
            cohort="2023",
        )[0]
        self.assertNotEqual(result["college"], "金融学院")

    def test_output_has_exact_contract_fields(self) -> None:
        result = self.retriever.retrieve("重修成绩", top_k=1)[0]
        self.assertEqual(set(result), set(RETRIEVED_CHUNK_FIELDS))
        self.assertIsInstance(result["score"], float)

    def test_invalid_arguments_are_rejected(self) -> None:
        for query, top_k in (("", 5), ("valid", 0), ("valid", 51)):
            with self.subTest(query=query, top_k=top_k):
                with self.assertRaises(ValueError):
                    self.retriever.retrieve(query, top_k)

    def test_twenty_fixture_queries_reach_top_five_target(self) -> None:
        cases = [
            ("重修成绩如实记载", "fixture_school_status_001", None, None),
            ("完成培养方案申请毕业", "fixture_school_status_002", None, None),
            ("转专业第一学年名额", "fixture_school_transfer_003", None, None),
            ("课程替代开课学院审核", "fixture_school_credit_004", None, None),
            ("推免应届毕业生申请范围", "fixture_school_recommend_005", None, None),
            ("推荐公开公平公正综合评价", "fixture_school_recommend_006", None, None),
            ("2023毕业165学分24", "fixture_it_py2023_007", "计算机与人工智能学院", "2023"),
            ("2024毕业168实践30", "fixture_it_py2024_008", "计算机与人工智能学院", "2024"),
            ("2025毕业170人工智能12", "fixture_it_py2025_009", "计算机与人工智能学院", "2025"),
            ("CS101程序设计基础4学分", "fixture_it_table_010", "计算机与人工智能学院", "2023"),
            ("CS205机器学习导论", "fixture_it_table_011", "计算机与人工智能学院", "2023"),
            ("AI101人工智能导论", "fixture_it_table_012", "计算机与人工智能学院", "2024"),
            ("不得有不及格课程记录70%30%", "fixture_it_recommend_013", "计算机与人工智能学院", "2023"),
            ("金融2023毕业160学分", "fixture_fin_py2023_015", "金融学院", "2023"),
            ("金融2024量化金融10学分", "fixture_fin_py2024_016", "金融学院", "2024"),
            ("金融2025专业实践8学分", "fixture_fin_py2025_017", "金融学院", "2025"),
            ("FIN201公司金融3学分", "fixture_fin_table_018", "金融学院", "2023"),
            ("金融推免学业80%科研竞赛20%", "fixture_fin_recommend_019", "金融学院", "2023"),
            ("辅修学分不得重复计入主修", "fixture_school_minor_021", None, None),
            ("休学健康原因", "fixture_school_leave_022", None, None),
        ]
        hits = 0
        for query, expected, college, cohort in cases:
            results = self.retriever.retrieve(query, 5, college, cohort)
            hits += expected in {item["chunk_id"] for item in results}
        self.assertGreaterEqual(hits / len(cases), 0.8)


if __name__ == "__main__":
    unittest.main()

