from __future__ import annotations

import unittest

from swufe_rag.routing.router import HybridRouter
from swufe_rag.routing.schemas import RouteContext


class BrokenClassifier:
    def classify(self, question, context):
        raise RuntimeError("classifier unavailable")


class FixedClassifier:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def classify(self, question, context):
        school = self.mode == "school_rag"
        return {
            "mode": self.mode,
            "requires_school_facts": school,
            "intent": "school_general" if school else "general_chat",
            "college": None,
            "cohort": None,
            "policy_year": None,
            "rewritten_query": question,
            "search_terms": [],
            "confidence": 0.8,
        }


class HybridRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = HybridRouter(
            known_colleges=("计算机与人工智能学院", "金融学院")
        )

    def test_general_questions_default_to_general_chat(self) -> None:
        for question in (
            "什么是注意力机制",
            "帮我写快速排序",
            "我最近学习压力很大",
            "帮我润色一封邮件",
            "大学课程应该怎么学习",
        ):
            with self.subTest(question=question):
                self.assertEqual(self.router.route(question).mode, "general_chat")

    def test_school_facts_always_use_school_rag(self) -> None:
        for question in (
            "2024级计算机类要修多少学分",
            "挂科后还能推免吗",
            "校园网密码忘了怎么办",
            "西财教务处通知原文在哪里",
            "Kaggle比赛能不能算保研加分",
        ):
            with self.subTest(question=question):
                self.assertEqual(self.router.route(question).mode, "school_rag")

    def test_classifier_failure_defaults_general_but_not_for_school_fact(self) -> None:
        router = HybridRouter(BrokenClassifier())
        self.assertEqual(router.route("今天天气不错").mode, "general_chat")
        self.assertEqual(router.route("选课有什么规定").mode, "school_rag")

    def test_follow_up_inherits_school_topic_and_scope(self) -> None:
        context = RouteContext(
            last_mode="school_rag",
            last_intent="promotion",
            last_college="计算机与人工智能学院",
            last_cohort="2023",
            last_rewritten_query="挂科后还能推免吗",
        )
        decision = self.router.route("那重修通过以后呢？", context=context)
        self.assertEqual(decision.mode, "school_rag")
        self.assertEqual(decision.intent, "promotion")
        self.assertEqual(decision.college, "计算机与人工智能学院")
        self.assertEqual(decision.cohort, "2023")
        self.assertIn("挂科后还能推免吗", decision.rewritten_query)

    def test_explicit_topic_switch_leaves_school_context(self) -> None:
        context = RouteContext(
            last_mode="school_rag",
            last_intent="promotion",
            last_rewritten_query="挂科后还能推免吗",
        )
        self.assertEqual(
            self.router.route("不说这个了，给我写代码", context=context).mode,
            "general_chat",
        )

    def test_classifier_cannot_release_explicit_school_fact(self) -> None:
        router = HybridRouter(FixedClassifier("general_chat"))
        self.assertEqual(router.route("西财转专业有什么条件").mode, "school_rag")

    def test_year_fields_are_disambiguated(self) -> None:
        cohort = self.router.route("2024级培养方案有哪些专业选修")
        self.assertEqual(cohort.cohort, "2024")
        self.assertIsNone(cohort.policy_year)
        policy = self.router.route("2024年的推免细则是什么")
        self.assertEqual(policy.policy_year, 2024)


if __name__ == "__main__":
    unittest.main()
