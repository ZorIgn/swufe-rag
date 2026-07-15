"""Deterministic 100-case evaluation for the mixed-dialogue router."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from swufe_rag.routing.router import HybridRouter
from swufe_rag.routing.schemas import RouteContext


DEFAULT_CASES = Path(__file__).with_name("hybrid_route_queries.json")


def evaluate_route_cases(path: str | Path = DEFAULT_CASES) -> dict[str, Any]:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    router = HybridRouter(
        known_colleges=("计算机与人工智能学院", "金融学院")
    )
    errors = []
    counts = {"general_chat": 0, "school_rag": 0, "follow_up": 0}
    correct = {"general_chat": 0, "school_rag": 0, "follow_up": 0}
    for case in cases:
        expected = case["mode"]
        category = "follow_up" if "context" in case else expected
        counts[category] += 1
        context = RouteContext(**case.get("context", {}))
        actual = router.route(case["question"], context=context).mode
        if actual == expected:
            correct[category] += 1
        else:
            errors.append(
                {"id": case["id"], "expected": expected, "actual": actual}
            )
    return {
        "case_count": len(cases),
        "general_false_block_rate": 1
        - correct["general_chat"] / max(counts["general_chat"], 1),
        "school_fact_to_general_count": counts["school_rag"]
        - correct["school_rag"],
        "follow_up_accuracy": correct["follow_up"] / max(counts["follow_up"], 1),
        "errors": errors,
    }


def main() -> None:
    print(json.dumps(evaluate_route_cases(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
