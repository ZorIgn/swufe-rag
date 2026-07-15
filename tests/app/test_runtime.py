from __future__ import annotations

import unittest

from app.runtime import DEMO_CHUNKS, build_demo_runtime, build_review_runtime


class DemoRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = build_demo_runtime()

    def test_options_are_derived_from_demo_chunks(self) -> None:
        options = self.runtime.options()
        self.assertEqual(options["mode"], "demo")
        self.assertEqual(options["chunk_count"], 24)
        self.assertIn("计算机与人工智能学院", options["colleges"])
        self.assertIn("金融学院", options["colleges"])
        self.assertEqual(options["cohorts"], ["2023", "2024", "2025"])

    def test_debug_answer_contains_trace_and_retrieval_details(self) -> None:
        result = self.runtime.debug_ask(
            "CS205是什么课，多少学分？",
            college="计算机与人工智能学院",
            cohort="2023",
        )
        self.assertFalse(result["refused"])
        self.assertEqual(result["mode"], "demo")
        self.assertGreaterEqual(result["latency_ms"], 0)
        self.assertEqual(result["retrieved"][0]["chunk_id"], "fixture_it_table_011")
        self.assertEqual(result["citations"][0]["chunk_id"], "fixture_it_table_011")

    def test_formal_answer_does_not_leak_debug_mode(self) -> None:
        result = self.runtime.ask(
            "CS205是什么课，多少学分？",
            college="计算机与人工智能学院",
            cohort="2023",
        )
        self.assertEqual(
            set(result),
            {"answer_md", "citations", "refused", "retrieved", "latency_ms"},
        )
        self.assertNotIn("mode", result)

    def test_out_of_domain_question_is_refused(self) -> None:
        result = self.runtime.ask(
            "食堂晚上几点关门？",
            college="计算机与人工智能学院",
            cohort="2023",
        )
        self.assertTrue(result["refused"])
        self.assertEqual(result["citations"], [])

    def test_source_lookup_returns_exact_full_chunk(self) -> None:
        source = self.runtime.source("fixture_it_table_011")
        self.assertIsNotNone(source)
        self.assertIn("CS205", source["text"])
        self.assertNotIn("score", source)
        self.assertIsNone(self.runtime.source("does-not-exist"))


class ReviewRuntimeTests(unittest.TestCase):
    def test_review_mode_requires_an_explicit_chunk_file_and_keeps_contract(self) -> None:
        runtime = build_review_runtime(DEMO_CHUNKS)
        self.assertEqual(runtime.options()["mode"], "review")
        result = runtime.ask(
            "CS205是什么课，多少学分？",
            college="计算机与人工智能学院",
            cohort="2023",
        )
        self.assertEqual(
            set(result),
            {"answer_md", "citations", "refused", "retrieved", "latency_ms"},
        )
        self.assertEqual(runtime.source("fixture_it_table_011")["chunk_id"], "fixture_it_table_011")


if __name__ == "__main__":
    unittest.main()
