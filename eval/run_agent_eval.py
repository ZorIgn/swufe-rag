"""Run labeled typed-agent evaluation and fail closed on missed quality gates."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from agent.factory import build_runtime
from agent.state import AgentState
from eval.metrics import binary_classification_metrics
from evidence.models import FinalAnswer


@dataclass(frozen=True)
class Outcome:
    identifier: str | None
    intent_expected: str | None
    intent_observed: str | None
    intent_correct: bool
    expected_tools: tuple[str, ...] | None
    observed_tools: tuple[str, ...]
    plan_exact: bool | None
    tool_precision: float | None
    tool_recall: float | None
    expected_answer_contains: tuple[str, ...] | None
    answer_contains_correct: bool | None
    expected_safe_rejection: bool | None
    safe_rejection: bool
    refused: bool
    clarified: bool
    scope_pollution: bool

    def as_json(self) -> dict[str, object]:
        return {
            "id": self.identifier,
            "intent_expected": self.intent_expected,
            "intent_observed": self.intent_observed,
            "intent_correct": self.intent_correct,
            "expected_tools": list(self.expected_tools) if self.expected_tools is not None else None,
            "observed_tools": list(self.observed_tools),
            "plan_exact": self.plan_exact,
            "tool_precision": self.tool_precision,
            "tool_recall": self.tool_recall,
            "expected_answer_contains": (
                list(self.expected_answer_contains)
                if self.expected_answer_contains is not None
                else None
            ),
            "answer_contains_correct": self.answer_contains_correct,
            "expected_safe_rejection": self.expected_safe_rejection,
            "safe_rejection": self.safe_rejection,
            "refused": self.refused,
            "clarified": self.clarified,
            "scope_pollution": self.scope_pollution,
        }


class _EvaluationRuntime(Protocol):
    def ask(self, question: str) -> tuple[FinalAnswer, AgentState]: ...


def _probability(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("threshold must be a number in [0, 1]") from exc
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("threshold must be in [0, 1]")
    return parsed


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _label_string(row: dict[str, object], field: str, index: int) -> str | None:
    value = row.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"evaluation row {index}: {field} must be a non-empty string")
    return value.strip()


def _label_string_list(
    row: dict[str, object], field: str, index: int, *, allow_empty: bool
) -> tuple[str, ...] | None:
    if field not in row:
        return None
    value = row[field]
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise SystemExit(f"evaluation row {index}: {field} must be a list of non-empty strings")
    values = tuple(item.strip() for item in value)
    if not allow_empty and not values:
        raise SystemExit(f"evaluation row {index}: {field} must not be empty")
    if len(set(values)) != len(values):
        raise SystemExit(f"evaluation row {index}: {field} must not contain duplicates")
    return values


def _label_bool(row: dict[str, object], field: str, index: int) -> bool | None:
    if field not in row:
        return None
    value = row[field]
    if not isinstance(value, bool):
        raise SystemExit(f"evaluation row {index}: {field} must be a boolean")
    return value


def _minimum_gate(
    name: str, actual: float | None, threshold: float, sample_count: int
) -> dict[str, object]:
    if actual is None:
        return {
            "name": name,
            "operator": ">=",
            "threshold": threshold,
            "actual": None,
            "sample_count": sample_count,
            "status": "not_applicable",
            "passed": True,
        }
    return {
        "name": name,
        "operator": ">=",
        "threshold": threshold,
        "actual": actual,
        "sample_count": sample_count,
        "status": "measured",
        "passed": actual >= threshold,
    }


def _maximum_gate(
    name: str, actual: float | None, threshold: float, sample_count: int
) -> dict[str, object]:
    if actual is None:
        return {
            "name": name,
            "operator": "<=",
            "threshold": threshold,
            "actual": None,
            "sample_count": sample_count,
            "status": "not_applicable",
            "passed": True,
        }
    return {
        "name": name,
        "operator": "<=",
        "threshold": threshold,
        "actual": actual,
        "sample_count": sample_count,
        "status": "measured",
        "passed": actual <= threshold,
    }


def _outcome(row: dict[str, object], index: int, runtime: _EvaluationRuntime) -> Outcome:
    question = _label_string(row, "question", index)
    if question is None:
        raise SystemExit(f"evaluation row {index}: question is required")
    expected_intent = _label_string(row, "intent", index)
    expected_tools = _label_string_list(row, "expected_tools", index, allow_empty=True)
    expected_answer_contains = _label_string_list(
        row, "expected_answer_contains", index, allow_empty=False
    )
    expected_safe_rejection = _label_bool(row, "expected_safe_rejection", index)
    identifier = _label_string(row, "id", index)

    answer, state = runtime.ask(question)
    normalized = state.normalized_query
    observed_intent = str(normalized.intent) if normalized is not None else None
    observed_tools = tuple(
        sorted(str(operation.type) for operation in state.plan.operations)
    ) if state.plan else ()
    observed_tool_set = set(observed_tools)
    expected_tool_set = set(expected_tools) if expected_tools is not None else set()
    plan_exact = observed_tool_set == expected_tool_set if expected_tools is not None else None
    if expected_tools is None:
        tool_precision = None
        tool_recall = None
    elif not observed_tool_set and not expected_tool_set:
        # A deliberate out-of-corpus rejection with no tool calls is exactly
        # correct; it is not a zero-precision/zero-recall plan.
        tool_precision = 1.0
        tool_recall = 1.0
    else:
        tool_precision = len(observed_tool_set & expected_tool_set) / max(
            1, len(observed_tool_set)
        )
        tool_recall = len(observed_tool_set & expected_tool_set) / max(1, len(expected_tool_set))
    refused = bool(answer.refused)
    clarified = bool(answer.clarification)
    safe_rejection = refused or clarified
    answer_text = str(answer.answer_md)
    warnings = tuple(str(value) for value in normalized.warnings) if normalized else ()
    return Outcome(
        identifier=identifier,
        intent_expected=expected_intent,
        intent_observed=observed_intent,
        intent_correct=expected_intent is None or expected_intent == observed_intent,
        expected_tools=expected_tools,
        observed_tools=observed_tools,
        plan_exact=plan_exact,
        tool_precision=tool_precision,
        tool_recall=tool_recall,
        expected_answer_contains=expected_answer_contains,
        answer_contains_correct=(
            all(value in answer_text for value in expected_answer_contains)
            if expected_answer_contains is not None
            else None
        ),
        expected_safe_rejection=expected_safe_rejection,
        safe_rejection=safe_rejection,
        refused=refused,
        clarified=clarified,
        scope_pollution=any("conflict" in warning.lower() for warning in warnings),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retrieval-mode", choices=("lexical", "hybrid"))
    parser.add_argument("--min-intent-accuracy", type=_probability, default=1.0)
    parser.add_argument("--min-plan-exact-match", type=_probability, default=1.0)
    parser.add_argument("--min-tool-precision", type=_probability, default=1.0)
    parser.add_argument("--min-tool-recall", type=_probability, default=1.0)
    parser.add_argument("--min-answer-containment", type=_probability, default=1.0)
    parser.add_argument("--min-safe-rejection-f1", type=_probability, default=1.0)
    parser.add_argument("--max-scope-pollution-rate", type=_probability, default=0.0)
    args = parser.parse_args()

    raw_rows = json.loads(args.questions.read_text(encoding="utf-8"))
    if not isinstance(raw_rows, list) or not raw_rows:
        raise SystemExit("evaluation questions must be a non-empty JSON array")
    rows: list[dict[str, object]] = []
    for index, raw_row in enumerate(raw_rows, start=1):
        if not isinstance(raw_row, dict):
            raise SystemExit(f"evaluation row {index} must be a JSON object")
        rows.append(dict(raw_row))

    if args.retrieval_mode:
        os.environ["SWUFE_RETRIEVAL_MODE"] = args.retrieval_mode
    runtime = build_runtime(args.database)
    try:
        outcomes = [_outcome(row, index, runtime) for index, row in enumerate(rows, start=1)]
    finally:
        runtime.repository.close()

    plan_values = [outcome.plan_exact for outcome in outcomes if outcome.plan_exact is not None]
    precision_values = [
        outcome.tool_precision for outcome in outcomes if outcome.tool_precision is not None
    ]
    recall_values = [outcome.tool_recall for outcome in outcomes if outcome.tool_recall is not None]
    answer_values = [
        outcome.answer_contains_correct
        for outcome in outcomes
        if outcome.answer_contains_correct is not None
    ]
    rejection_pairs = [
        (outcome.expected_safe_rejection, outcome.safe_rejection)
        for outcome in outcomes
        if outcome.expected_safe_rejection is not None
    ]
    rejection_metrics = (
        binary_classification_metrics(
            [expected for expected, _observed in rejection_pairs],
            [observed for _expected, observed in rejection_pairs],
        )
        if rejection_pairs
        else None
    )
    intent_accuracy = _mean([float(outcome.intent_correct) for outcome in outcomes])
    plan_exact_match = _mean([float(value) for value in plan_values])
    tool_precision = _mean(precision_values)
    tool_recall = _mean(recall_values)
    answer_containment = _mean([float(value) for value in answer_values])
    safe_rejection_f1 = rejection_metrics["f1"] if rejection_metrics is not None else None
    scope_pollution_rate = _mean([float(outcome.scope_pollution) for outcome in outcomes])

    gates = {
        "intent_accuracy": _minimum_gate(
            "intent_accuracy", intent_accuracy, args.min_intent_accuracy, len(outcomes)
        ),
        "plan_exact_match": _minimum_gate(
            "plan_exact_match", plan_exact_match, args.min_plan_exact_match, len(plan_values)
        ),
        "tool_precision": _minimum_gate(
            "tool_precision", tool_precision, args.min_tool_precision, len(precision_values)
        ),
        "tool_recall": _minimum_gate(
            "tool_recall", tool_recall, args.min_tool_recall, len(recall_values)
        ),
        "answer_containment": _minimum_gate(
            "answer_containment",
            answer_containment,
            args.min_answer_containment,
            len(answer_values),
        ),
        "safe_rejection_f1": _minimum_gate(
            "safe_rejection_f1",
            safe_rejection_f1,
            args.min_safe_rejection_f1,
            len(rejection_pairs),
        ),
        "scope_pollution_rate": _maximum_gate(
            "scope_pollution_rate",
            scope_pollution_rate,
            args.max_scope_pollution_rate,
            len(outcomes),
        ),
    }
    passed = all(bool(gate["passed"]) for gate in gates.values())
    report = {
        "question_count": len(outcomes),
        "intent_accuracy": intent_accuracy,
        "plan_exact_match": plan_exact_match,
        "tool_precision": tool_precision,
        "tool_recall": tool_recall,
        "answer_containment": answer_containment,
        "safe_rejection": rejection_metrics,
        "scope_pollution_rate": scope_pollution_rate,
        "refusal_count": sum(outcome.refused for outcome in outcomes),
        "clarification_count": sum(outcome.clarified for outcome in outcomes),
        "gates": gates,
        "passed": passed,
        "outcomes": [outcome.as_json() for outcome in outcomes],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if not passed:
        failed = ", ".join(name for name, gate in gates.items() if not gate["passed"])
        raise SystemExit(f"agent evaluation quality gate failed: {failed}")


if __name__ == "__main__":
    main()
