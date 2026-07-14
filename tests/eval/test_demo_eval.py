from __future__ import annotations

import unittest

from eval.demo_eval import evaluate


class DemoEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = evaluate()

    def test_retrieval_and_scope_gates(self) -> None:
        self.assertEqual(self.report["case_count"], 20)
        self.assertGreaterEqual(self.report["retrieval_recall_at_5"], 0.8)
        self.assertEqual(self.report["scope_pollution_count"], 0)

    def test_refusal_gate(self) -> None:
        self.assertGreaterEqual(self.report["refusal_accuracy"], 0.9)


if __name__ == "__main__":
    unittest.main()
