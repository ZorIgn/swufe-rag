"""Run labeled typed-agent evaluation and fail closed on missed quality gates."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast, get_type_hints

from pydantic import BaseModel, TypeAdapter, ValidationError

from agent.factory import build_runtime
from agent.session import InMemoryTTLSessionStore
from agent.state import AgentState
from eval.candidate_release import (
    CandidateEvaluationError,
    load_candidate_evaluation_context,
)
from eval.holdout import HoldoutContractError, load_holdout_manifest
from eval.metrics import binary_classification_metrics, metric_gate
from eval.promotion_policy import (
    AGENT_EVALUATION_CONFIG,
    PROMOTION_POLICY_SHA256,
    PROMOTION_POLICY_VERSION,
    REPORT_CONTRACT_VERSION,
    PromotionPolicyError,
    validate_agent_report,
)
from evidence.models import FinalAnswer
from query.schemas import (
    AuditCompletedCoursesOperation,
    CheckCurriculumFeasibilityOperation,
    CompareProgramsOperation,
    GetCourseDetailOperation,
    GetGraduationRequirementsOperation,
    GetModuleRequirementsOperation,
    ListCoursesBeforeSemesterOperation,
    ListCoursesOperation,
    ListUnavoidableCoursesOperation,
    ResolveSourceOperation,
    RetrievePolicyOperation,
)

_OPERATION_MODELS: dict[str, type[BaseModel]] = {
    "list_courses": ListCoursesOperation,
    "get_course_detail": GetCourseDetailOperation,
    "get_graduation_requirements": GetGraduationRequirementsOperation,
    "get_module_requirements": GetModuleRequirementsOperation,
    "audit_completed_courses": AuditCompletedCoursesOperation,
    "list_courses_before_semester": ListCoursesBeforeSemesterOperation,
    "list_unavoidable_courses": ListUnavoidableCoursesOperation,
    "check_curriculum_feasibility": CheckCurriculumFeasibilityOperation,
    "retrieve_policy": RetrievePolicyOperation,
    "compare_programs": CompareProgramsOperation,
    "resolve_source": ResolveSourceOperation,
}


@dataclass(frozen=True)
class ExpectedOperation:
    """Frozen expected operation signature used by the plan contract."""

    operation_type: str
    args: dict[str, object]
    depends_on: tuple[str, ...]

    def as_json(self) -> dict[str, object]:
        return {
            "type": self.operation_type,
            "args": self.args,
            "depends_on": list(self.depends_on),
        }


@dataclass(frozen=True)
class Outcome:
    identifier: str
    intent_expected: str
    intent_observed: str | None
    intent_correct: bool
    expected_operations: tuple[ExpectedOperation, ...]
    observed_operations: tuple[dict[str, object], ...]
    expected_tools: tuple[str, ...]
    observed_tools: tuple[str, ...]
    plan_exact: bool
    tool_precision: float
    tool_recall: float
    expected_answer_contains: tuple[str, ...]
    answer_contains_correct: bool
    expected_safe_rejection: bool
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
            "expected_operations": [item.as_json() for item in self.expected_operations],
            "observed_operations": list(self.observed_operations),
            "expected_tools": list(self.expected_tools),
            "observed_tools": list(self.observed_tools),
            "plan_exact": self.plan_exact,
            "tool_precision": self.tool_precision,
            "tool_recall": self.tool_recall,
            "expected_answer_contains": (
                list(self.expected_answer_contains)
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


def _required_label_string(row: dict[str, object], field: str, index: int) -> str:
    value = _label_string(row, field, index)
    if value is None:
        raise SystemExit(f"evaluation row {index}: {field} is required")
    return value


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


def _json_value(value: object) -> object:
    """Convert Pydantic/JSON-compatible values into stable plain values."""

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _typed_args(operation_type: str, value: object, index: int) -> dict[str, object]:
    if operation_type not in _OPERATION_MODELS:
        raise SystemExit(f"evaluation row {index}: unknown operation type: {operation_type}")
    if not isinstance(value, dict):
        raise SystemExit(f"evaluation row {index}: expected_operations args must be an object")
    operation_model = _OPERATION_MODELS[operation_type]
    args_type = get_type_hints(operation_model)["args"]
    try:
        parsed = TypeAdapter(args_type).validate_python(value)
    except ValidationError as exc:
        raise SystemExit(
            f"evaluation row {index}: invalid typed args for {operation_type}: {exc}"
        ) from exc
    return cast(dict[str, object], _json_value(parsed))


def _expected_operations(
    row: dict[str, object], index: int
) -> tuple[ExpectedOperation, ...]:
    """Require operation-level labels, including typed args and dependencies."""

    if "expected_operations" not in row:
        raise SystemExit(
            f"evaluation row {index}: expected_operations is required; "
            "legacy expected_tools-only rows are not fail-closed"
        )
    value = row["expected_operations"]
    if not isinstance(value, list):
        raise SystemExit(f"evaluation row {index}: expected_operations must be a list")
    operations: list[ExpectedOperation] = []
    for operation_index, raw in enumerate(value, start=1):
        if not isinstance(raw, dict):
            raise SystemExit(
                f"evaluation row {index} operation {operation_index} must be an object"
            )
        operation_type = raw.get("type")
        if not isinstance(operation_type, str) or not operation_type.strip():
            raise SystemExit(
                f"evaluation row {index} operation {operation_index}: type is required"
            )
        operation_type = operation_type.strip()
        depends_on = raw.get("depends_on")
        if not isinstance(depends_on, list) or any(
            not isinstance(item, str) or not item.strip() for item in depends_on
        ):
            raise SystemExit(
                f"evaluation row {index} operation {operation_index}: "
                "depends_on must be a list of non-empty strings"
            )
        dependency_values = tuple(str(item).strip() for item in depends_on)
        if len(set(dependency_values)) != len(dependency_values):
            raise SystemExit(
                f"evaluation row {index} operation {operation_index}: depends_on has duplicates"
            )
        operations.append(
            ExpectedOperation(
                operation_type=operation_type,
                args=_typed_args(operation_type, raw.get("args"), index),
                depends_on=dependency_values,
            )
        )
    return tuple(operations)


def _validate_row_contract(row: dict[str, object], index: int) -> None:
    """Validate all required labels before constructing a runtime."""

    _required_label_string(row, "id", index)
    _required_label_string(row, "question", index)
    _required_label_string(row, "intent", index)
    _expected_operations(row, index)
    if _label_string_list(row, "expected_answer_contains", index, allow_empty=True) is None:
        raise SystemExit(f"evaluation row {index}: expected_answer_contains is required")
    if _label_bool(row, "expected_safe_rejection", index) is None:
        raise SystemExit(f"evaluation row {index}: expected_safe_rejection is required")


def _minimum_gate(
    name: str, actual: float | None, threshold: float, sample_count: int
) -> dict[str, object]:
    return metric_gate(
        name,
        actual,
        threshold,
        sample_count=sample_count,
        operator=">=",
    )


def _maximum_gate(
    name: str, actual: float | None, threshold: float, sample_count: int
) -> dict[str, object]:
    return metric_gate(
        name,
        actual,
        threshold,
        sample_count=sample_count,
        operator="<=",
    )


def _observed_operations(state: AgentState) -> tuple[dict[str, object], ...]:
    """Serialize the actual typed plan, resolving dependency ids to types."""

    if state.plan is None:
        return ()
    operation_types = {
        operation.operation_id: str(operation.type) for operation in state.plan.operations
    }
    values: list[dict[str, object]] = []
    for operation in state.plan.operations:
        args = _json_value(operation.args)
        values.append(
            {
                "type": str(operation.type),
                "args": cast(dict[str, object], args),
                "depends_on": [
                    operation_types.get(dependency, dependency)
                    for dependency in operation.depends_on
                ],
                "operation_id": operation.operation_id,
            }
        )
    return tuple(values)


def _outcome(row: dict[str, object], index: int, runtime: _EvaluationRuntime) -> Outcome:
    question = _required_label_string(row, "question", index)
    expected_intent = _required_label_string(row, "intent", index)
    expected_operations = _expected_operations(row, index)
    expected_answer_contains = _label_string_list(
        row, "expected_answer_contains", index, allow_empty=True
    )
    if expected_answer_contains is None:
        raise SystemExit(f"evaluation row {index}: expected_answer_contains is required")
    expected_safe_rejection = _label_bool(row, "expected_safe_rejection", index)
    if expected_safe_rejection is None:
        raise SystemExit(f"evaluation row {index}: expected_safe_rejection is required")
    identifier = _required_label_string(row, "id", index)

    answer, state = runtime.ask(question)
    normalized = state.normalized_query
    observed_intent = str(normalized.intent) if normalized is not None else None
    observed_operations = _observed_operations(state)
    observed_tools = tuple(str(item["type"]) for item in observed_operations)
    expected_tools = tuple(item.operation_type for item in expected_operations)
    observed_signature = tuple(
        {
            "type": item["type"],
            "args": item["args"],
            "depends_on": item["depends_on"],
        }
        for item in observed_operations
    )
    expected_signature = tuple(item.as_json() for item in expected_operations)
    plan_exact = observed_signature == expected_signature
    observed_tool_set = set(observed_tools)
    expected_tool_set = set(expected_tools)
    if not observed_tool_set and not expected_tool_set:
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
        intent_correct=expected_intent == observed_intent,
        expected_operations=expected_operations,
        observed_operations=observed_operations,
        expected_tools=expected_tools,
        observed_tools=observed_tools,
        plan_exact=plan_exact,
        tool_precision=tool_precision,
        tool_recall=tool_recall,
        expected_answer_contains=expected_answer_contains,
        answer_contains_correct=all(value in answer_text for value in expected_answer_contains),
        expected_safe_rejection=expected_safe_rejection,
        safe_rejection=safe_rejection,
        refused=refused,
        clarified=clarified,
        scope_pollution=any("conflict" in warning.lower() for warning in warnings),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate-release-manifest",
        type=Path,
        help="verified candidate entry; derives DB/index/commit and enables promotion evidence",
    )
    parser.add_argument(
        "--holdout-manifest",
        type=Path,
        help="frozen manifest whose hashes and dataset labels must cover --questions",
    )
    parser.add_argument("--dataset-version", required=False)
    parser.add_argument(
        "--allow-test-fixture-dataset-mismatch",
        action="store_true",
        help=(
            "only for checked-in test_fixture holdouts; records that the runtime "
            "database is intentionally different from the fixture dataset"
        ),
    )
    parser.add_argument("--model-id", default="deterministic-agent")
    parser.add_argument("--artifact-id", default="none")
    parser.add_argument("--commit", default=os.environ.get("GITHUB_SHA", "working-tree"))
    parser.add_argument("--retrieval-mode", choices=("lexical", "hybrid"))
    parser.add_argument("--min-intent-accuracy", type=_probability, default=1.0)
    parser.add_argument("--min-plan-exact-match", type=_probability, default=1.0)
    parser.add_argument("--min-tool-precision", type=_probability, default=1.0)
    parser.add_argument("--min-tool-recall", type=_probability, default=1.0)
    parser.add_argument("--min-answer-containment", type=_probability, default=1.0)
    parser.add_argument("--min-safe-rejection-f1", type=_probability, default=1.0)
    parser.add_argument("--max-scope-pollution-rate", type=_probability, default=0.0)
    args = parser.parse_args()

    candidate = None
    promotion_eligible = args.candidate_release_manifest is not None
    if promotion_eligible:
        if args.holdout_manifest is None:
            raise SystemExit("candidate evaluation requires --holdout-manifest")
        if (
            args.database is not None
            or args.questions is not None
            or args.dataset_version is not None
            or args.retrieval_mode is not None
            or args.artifact_id != "none"
            or args.allow_test_fixture_dataset_mismatch
        ):
            raise SystemExit(
                "candidate evaluation derives database/questions/version/mode/artifact; "
                "diagnostic overrides are forbidden"
            )
        try:
            candidate = load_candidate_evaluation_context(
                args.candidate_release_manifest,
                args.holdout_manifest,
            )
        except CandidateEvaluationError as exc:
            raise SystemExit(f"candidate evaluation contract failed: {exc}") from exc
        holdout = candidate.holdout
        questions_path = holdout.root / holdout.inputs["agent_cases"].path
        database_path = candidate.database_path
        expected_dataset_version: str | None = holdout.dataset_version
    else:
        if args.database is None or args.questions is None:
            raise SystemExit(
                "diagnostic evaluation requires --database and --questions"
            )
        database_path = args.database
        questions_path = args.questions
        holdout = None
        if args.holdout_manifest is not None:
            try:
                holdout = load_holdout_manifest(args.holdout_manifest)
                holdout.verify_role("agent_cases", questions_path)
            except HoldoutContractError as exc:
                raise SystemExit(f"holdout contract failed: {exc}") from exc
            if args.dataset_version and args.dataset_version != holdout.dataset_version:
                raise SystemExit(
                    "--dataset-version does not match holdout manifest: "
                    f"{args.dataset_version!r} != {holdout.dataset_version!r}"
                )
        expected_dataset_version = args.dataset_version or (
            holdout.dataset_version if holdout is not None else None
        )

    raw_rows = json.loads(questions_path.read_text(encoding="utf-8"))
    if not isinstance(raw_rows, list) or not raw_rows:
        raise SystemExit("evaluation questions must be a non-empty JSON array")
    rows: list[dict[str, object]] = []
    for index, raw_row in enumerate(raw_rows, start=1):
        if not isinstance(raw_row, dict):
            raise SystemExit(f"evaluation row {index} must be a JSON object")
        rows.append(dict(raw_row))
    if holdout is not None:
        holdout.verify_role_count("agent_cases", len(rows))
    for index, row in enumerate(rows, start=1):
        _validate_row_contract(row, index)

    if candidate is not None:
        runtime = build_runtime(
            release_bundle=candidate.runtime_bundle,
            session_store=InMemoryTTLSessionStore(
                dataset_version=candidate.holdout.dataset_version
            ),
        )
    else:
        if args.retrieval_mode:
            os.environ["SWUFE_RETRIEVAL_MODE"] = args.retrieval_mode
        runtime = build_runtime(database_path)
    try:
        runtime_dataset_version = runtime.repository.metadata().get("dataset_version")
        runtime_mode = runtime.options().get("retrieval_mode")
        if candidate is not None:
            ready, reasons = runtime.readiness()
            if not ready or reasons or runtime_mode != "hybrid":
                raise SystemExit(
                    "candidate runtime is not promotion-ready: "
                    f"mode={runtime_mode!r}, reasons={list(reasons)}"
                )
        dataset_mismatch = (
            expected_dataset_version is not None and runtime_dataset_version != expected_dataset_version
        )
        if dataset_mismatch and not (
            args.allow_test_fixture_dataset_mismatch
            and holdout is not None
            and holdout.status == "test_fixture"
        ):
            raise SystemExit(
                "runtime dataset_version does not match frozen evaluation data: "
                f"runtime={runtime_dataset_version!r}, expected={expected_dataset_version!r}"
            )
        outcomes = [_outcome(row, index, runtime) for index, row in enumerate(rows, start=1)]
    finally:
        runtime.repository.close()

    plan_values = [outcome.plan_exact for outcome in outcomes]
    precision_values = [outcome.tool_precision for outcome in outcomes]
    recall_values = [outcome.tool_recall for outcome in outcomes]
    answer_values = [outcome.answer_contains_correct for outcome in outcomes]
    rejection_pairs = [
        (outcome.expected_safe_rejection, outcome.safe_rejection)
        for outcome in outcomes
    ]
    rejection_metrics = binary_classification_metrics(
        [expected for expected, _observed in rejection_pairs],
        [observed for _expected, observed in rejection_pairs],
    )
    intent_accuracy = _mean([float(outcome.intent_correct) for outcome in outcomes])
    plan_exact_match = _mean([float(value) for value in plan_values])
    tool_precision = _mean(precision_values)
    tool_recall = _mean(recall_values)
    answer_containment = _mean([float(value) for value in answer_values])
    safe_rejection_f1 = rejection_metrics["f1"]
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
    if candidate is not None:
        evaluation_config = dict(AGENT_EVALUATION_CONFIG)
        provenance: dict[str, object] = {
            "release_subject": candidate.release_subject,
            "holdout": candidate.holdout.release_lock(),
            "runtime": candidate.runtime_provenance(),
            "evaluator_git": candidate.evaluator_git,
        }
    else:
        evaluation_config = {
            "retrieval_mode": runtime_mode,
            "reranker_min_score": None,
            "session_backend": "environment",
        }
        provenance = {
            "release_subject": None,
            "holdout": holdout.provenance() if holdout is not None else None,
            "runtime": {
                "dataset_version": runtime_dataset_version,
                "retrieval_mode": runtime_mode,
            },
            "evaluator_git": {
                "commit": args.commit,
                "dirty": None,
            },
            "diagnostic_input": str(questions_path),
            "diagnostic_artifact_label": args.artifact_id,
            "test_fixture_dataset_mismatch_allowed": dataset_mismatch,
        }
    report = {
        "report_contract_version": REPORT_CONTRACT_VERSION,
        "promotion_policy_version": PROMOTION_POLICY_VERSION,
        "promotion_policy_sha256": PROMOTION_POLICY_SHA256,
        "promotion_eligible": promotion_eligible,
        "evaluation_config": evaluation_config,
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
        "provenance": provenance,
        "outcomes": [outcome.as_json() for outcome in outcomes],
    }
    if promotion_eligible:
        try:
            validate_agent_report(report, expected_question_count=len(outcomes))
        except PromotionPolicyError as exc:
            raise SystemExit(f"agent report is not promotion-eligible: {exc}") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if not passed:
        failed = ", ".join(name for name, gate in gates.items() if not gate["passed"])
        raise SystemExit(f"agent evaluation quality gate failed: {failed}")


if __name__ == "__main__":
    main()
