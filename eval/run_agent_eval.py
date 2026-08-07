"""Run typed-agent dev or holdout evaluation against released question data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.factory import build_runtime


def _f1(expected: set[str], observed: set[str]) -> float:
    if not expected and not observed:
        return 1.0
    precision = len(expected & observed) / max(1, len(observed))
    recall = len(expected & observed) / max(1, len(expected))
    return 2 * precision * recall / max(1e-12, precision + recall)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.questions.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise SystemExit("evaluation questions must be a non-empty JSON array")
    runtime = build_runtime(args.database)
    outcomes: list[dict[str, object]] = []
    try:
        for row in rows:
            if not isinstance(row, dict) or not str(row.get("question") or "").strip():
                raise SystemExit("each evaluation row needs a question")
            answer, state = runtime.ask(str(row["question"]))
            plan_types = {operation.type for operation in state.plan.operations} if state.plan else set()
            expected_tools = set(str(value) for value in row.get("expected_tools", ()))
            outcomes.append(
                {
                    "id": row.get("id"),
                    "intent_expected": row.get("intent"),
                    "intent_observed": state.normalized_query.intent if state.normalized_query else None,
                    "intent_correct": row.get("intent") in {None, state.normalized_query.intent if state.normalized_query else None},
                    "plan_exact": plan_types == expected_tools if "expected_tools" in row else None,
                    "tool_precision": len(plan_types & expected_tools) / max(1, len(plan_types)) if "expected_tools" in row else None,
                    "tool_recall": len(plan_types & expected_tools) / max(1, len(expected_tools)) if "expected_tools" in row else None,
                    "refused": answer.refused,
                    "clarified": bool(answer.clarification),
                    "scope_pollution": bool(state.normalized_query and state.normalized_query.warnings and "conflicts" in " ".join(state.normalized_query.warnings)),
                }
            )
    finally:
        runtime.repository.close()
    intent_correct = [bool(item["intent_correct"]) for item in outcomes]
    report = {
        "question_count": len(outcomes),
        "intent_accuracy": sum(intent_correct) / max(1, len(intent_correct)),
        "plan_exact_match": sum(bool(item["plan_exact"]) for item in outcomes if item["plan_exact"] is not None) / max(1, sum(item["plan_exact"] is not None for item in outcomes)),
        "tool_precision": sum(float(item["tool_precision"]) for item in outcomes if item["tool_precision"] is not None) / max(1, sum(item["tool_precision"] is not None for item in outcomes)),
        "tool_recall": sum(float(item["tool_recall"]) for item in outcomes if item["tool_recall"] is not None) / max(1, sum(item["tool_recall"] is not None for item in outcomes)),
        "refusal_count": sum(bool(item["refused"]) for item in outcomes),
        "clarification_count": sum(bool(item["clarified"]) for item in outcomes),
        "outcomes": outcomes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()