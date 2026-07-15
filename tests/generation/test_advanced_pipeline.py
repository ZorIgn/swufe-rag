from __future__ import annotations

import unittest

from contracts import CitationValidationError
from generation.context import ContextBuilder
from generation.grounding import StrictGroundingValidator, normalize_citation_formats
from generation.pipeline import AdvancedGenerationService, EvidenceGate
from generation.prompts import REFUSAL_TEXT
from tests.generation.helpers import FakeClient, retrieved


class AdvancedGenerationTests(unittest.TestCase):
    def test_context_builder_respects_total_and_per_chunk_budget(self) -> None:
        chunks = [
            retrieved("fixture_it_table_010"),
            retrieved("fixture_it_table_011"),
            retrieved("fixture_it_recommend_013"),
        ]
        builder = ContextBuilder(
            max_context_chars=1200, max_chunk_chars=420, min_chunk_chars=120
        )
        context, items = builder.build("CS205机器学习导论", chunks)
        self.assertLessEqual(len(context), 1200)
        self.assertTrue(items)
        self.assertIn("CS205", context)
        self.assertTrue(all(len(item.excerpt) <= 420 for item in items))

    def test_common_malformed_citations_are_normalized_locally(self) -> None:
        answer = "申请人应为应届毕业生【１】。不得有不及格记录[1, 2]。"
        normalized = normalize_citation_formats(answer)
        self.assertEqual(normalized, "申请人应为应届毕业生[1]。不得有不及格记录[1][2]。")

    def test_citations_must_be_at_sentence_end(self) -> None:
        validator = StrictGroundingValidator()
        chunk = retrieved("fixture_it_recommend_013")
        with self.assertRaisesRegex(CitationValidationError, "end of the sentence"):
            validator.validate("根据[1]本科阶段不得有不及格课程记录。", [chunk])

    def test_numeric_match_without_semantic_support_is_rejected(self) -> None:
        validator = StrictGroundingValidator()
        chunk = retrieved("fixture_it_table_011")
        with self.assertRaisesRegex(CitationValidationError, "support"):
            validator.validate("食堂在3点关门[1]。", [chunk])

    def test_more_than_four_citations_is_rejected(self) -> None:
        validator = StrictGroundingValidator()
        chunks = [retrieved("fixture_school_recommend_005") for _ in range(5)]
        with self.assertRaisesRegex(CitationValidationError, "more than four"):
            validator.validate("申请人应为应届毕业生[1][2][3][4][5]。", chunks)

    def test_refusal_without_terminal_period_is_canonicalized(self) -> None:
        client = FakeClient([REFUSAL_TEXT.rstrip("。")])
        service = AdvancedGenerationService(client)
        result = service.answer(
            "未知政策", [retrieved("fixture_it_recommend_013", score=0.8)]
        )
        self.assertTrue(result["refused"])
        self.assertEqual(result["answer_md"], REFUSAL_TEXT)

    def test_exact_course_code_cannot_bypass_low_dense_score(self) -> None:
        chunk = retrieved("fixture_it_table_011", score=0.2)
        self.assertFalse(EvidenceGate().sufficient("CS205是什么课", [chunk]))

    def test_high_dense_score_cannot_bypass_missing_temporal_subject(self) -> None:
        chunk = retrieved("fixture_school_recommend_005", score=0.8)
        self.assertFalse(
            EvidenceGate().sufficient("食堂晚上几点关门？", [chunk])
        )
        self.assertFalse(
            EvidenceGate().sufficient(
                "博士研究生中期考核什么时候进行？", [chunk]
            )
        )
        self.assertFalse(
            EvidenceGate().sufficient("校园网密码忘了怎么办？", [chunk])
        )

    def test_matching_temporal_subject_passes_entity_gate(self) -> None:
        chunk = retrieved("fixture_school_recommend_005", score=0.8)
        chunk["text"] += " 缓考申请最迟应在开考前两小时提交。"
        self.assertTrue(
            EvidenceGate().sufficient("缓考申请最迟什么时候提交？", [chunk])
        )

    def test_cohort_specific_advice_requires_same_cohort_evidence(self) -> None:
        school = retrieved("fixture_school_recommend_005", score=0.8)
        school["text"] += " 专业选修课至少修满8学分。"
        self.assertFalse(
            EvidenceGate().sufficient(
                "2023级计算机科学与技术专业选修课还差多少学分？",
                [school],
            )
        )
        school["level"] = "院级"
        school["college"] = "计算机与人工智能学院"
        school["cohort"] = "2023"
        self.assertTrue(
            EvidenceGate().sufficient(
                "2023级计算机科学与技术专业选修课还差多少学分？",
                [school],
            )
        )

    def test_grouped_citation_is_fixed_without_llm_repair(self) -> None:
        school = retrieved("fixture_school_recommend_005")
        college = retrieved("fixture_it_recommend_013")
        client = FakeClient(
            ["申请人应为应届毕业生且本科阶段不得有不及格课程记录[1,2]。"]
        )
        result = AdvancedGenerationService(client).answer(
            "推免资格", [school, college]
        )
        self.assertFalse(result["refused"])
        self.assertEqual(len(client.calls), 1)
        self.assertEqual([c["marker"] for c in result["citations"]], [1, 2])


if __name__ == "__main__":
    unittest.main()
