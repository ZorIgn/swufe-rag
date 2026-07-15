from __future__ import annotations

import unittest

from eval.hybrid_route_eval import evaluate_route_cases


class HybridRouteEvalTests(unittest.TestCase):
    def test_one_hundred_case_route_gate(self) -> None:
        report = evaluate_route_cases()
        self.assertEqual(report["case_count"], 100)
        self.assertLessEqual(report["general_false_block_rate"], 0.02)
        self.assertEqual(report["school_fact_to_general_count"], 0)
        self.assertGreaterEqual(report["follow_up_accuracy"], 0.95)
        self.assertEqual(report["errors"], [])


if __name__ == "__main__":
    unittest.main()
