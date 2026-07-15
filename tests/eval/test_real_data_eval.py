from __future__ import annotations

import unittest

from eval.real_data_eval import DEFAULT_CASES, evaluate


class RealDataEvaluationTests(unittest.TestCase):
    def test_real_review_set_has_required_scope_and_contract_metrics(self) -> None:
        report = evaluate(DEFAULT_CASES, "data/chunks.jsonl")
        self.assertEqual(report["case_count"], 20)
        self.assertEqual(report["scope_pollution_count"], 0)
        self.assertGreaterEqual(report["retrieval_recall_at_5"], 0.8)
        self.assertGreaterEqual(report["answer_support_accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()
