from __future__ import annotations

from collections.abc import Callable
from time import monotonic, sleep
from typing import cast

from agent.policies import RuntimePolicy
from agent.registry import RegisteredTool, ToolRegistry
from agent.tools import PlanExecutor, ToolExecutionContext
from evidence.models import CoverageComponent, CoverageReport, EvidencePacket
from query.schemas import (
    ALL_OPERATION_TYPES,
    ExecutionPlan,
    ListCoursesArgs,
    ListCoursesOperation,
    NormalizedQuery,
    Operation,
)

ToolCallback = Callable[..., EvidencePacket]


def _operation(operation_id: str, depends_on: tuple[str, ...] = ()) -> ListCoursesOperation:
    return ListCoursesOperation(
        operation_id=operation_id,
        depends_on=depends_on,
        args=ListCoursesArgs(cohort=2024, program_id="fixture-program"),
    )


def _plan(*operations: Operation) -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="executor-plan",
        query=NormalizedQuery(
            raw_question="fixture question",
            intent="course_query",
            information_scope="curriculum",
        ),
        operations=operations,
    )


def _packet(operation_id: str) -> EvidencePacket:
    return EvidencePacket(
        packet_id=f"packet-{operation_id}",
        coverage=CoverageReport(
            components=(
                CoverageComponent(
                    operation_id=operation_id,
                    tool_name="academic.list_courses",
                    kind="course_set",
                    complete=True,
                    expected_count=1,
                    returned_count=1,
                ),
            )
        ),
    )


def _registry(callback: ToolCallback, *, tool_timeout_seconds: float) -> ToolRegistry:
    registry = ToolRegistry()

    def ignored(_: Operation) -> EvidencePacket:
        return EvidencePacket(packet_id="ignored")

    for operation_type in sorted(ALL_OPERATION_TYPES):
        execute = callback if operation_type == "list_courses" else ignored
        registry.register(
            RegisteredTool(
                name=f"test.{operation_type}",
                operation_type=operation_type,
                read_only=True,
                timeout_seconds=tool_timeout_seconds,
                execute=cast(Callable[[Operation], EvidencePacket], execute),
            )
        )
    return registry


def _executor(callback: ToolCallback, *, timeout: float = 1.0) -> PlanExecutor:
    return PlanExecutor(
        _registry(callback, tool_timeout_seconds=timeout),
        RuntimePolicy(max_tool_calls=8, tool_timeout_seconds=timeout),
    )


def test_failed_dependency_never_runs_and_every_operation_has_typed_result() -> None:
    calls: list[str] = []

    def execute(operation: Operation) -> EvidencePacket:
        calls.append(operation.operation_id)
        if operation.operation_id == "upstream":
            raise RuntimeError("fixture failure")
        return _packet(operation.operation_id)

    result = _executor(execute).execute(
        _plan(
            _operation("upstream"),
            _operation("blocked", ("upstream",)),
            _operation("independent"),
        )
    )

    outcomes = {item.operation_id: item for item in result.execution_results}
    assert set(outcomes) == {"upstream", "blocked", "independent"}
    assert outcomes["upstream"].status == "failed"
    assert outcomes["blocked"].status == "dependency_failed"
    assert outcomes["independent"].status == "success"
    assert "blocked" not in calls
    assert tuple(item.operation_id for item in result.coverage.components) == ("independent",)


def test_dependency_cycle_is_skipped_without_executing_a_tool() -> None:
    calls: list[str] = []

    def execute(operation: Operation) -> EvidencePacket:
        calls.append(operation.operation_id)
        return _packet(operation.operation_id)

    result = _executor(execute).execute(
        _plan(_operation("one", ("two",)), _operation("two", ("one",)))
    )

    outcomes = {item.operation_id: item for item in result.execution_results}
    assert calls == []
    assert {item.status for item in outcomes.values()} == {"skipped"}
    assert {item.error_code for item in outcomes.values()} == {
        "invalid_operation_dependency_cycle"
    }


def test_plan_deadline_uses_wait_and_does_not_reset_for_dependency_waves() -> None:
    def execute(operation: Operation) -> EvidencePacket:
        if operation.operation_id == "slow":
            sleep(0.20)
        return _packet(operation.operation_id)

    started_at = monotonic()
    result = _executor(execute, timeout=0.03).execute(
        _plan(_operation("slow"), _operation("downstream", ("slow",)))
    )
    elapsed = monotonic() - started_at

    outcomes = {item.operation_id: item for item in result.execution_results}
    assert elapsed < 0.15
    assert outcomes["slow"].status == "timeout"
    assert outcomes["slow"].error_code == "plan_deadline_exceeded"
    assert outcomes["downstream"].status == "dependency_failed"


def test_context_uses_successful_prior_packets_and_coverage_order_is_stable() -> None:
    seen_contexts: list[ToolExecutionContext] = []

    def execute(operation: Operation, *, context: ToolExecutionContext) -> EvidencePacket:
        if operation.operation_id == "dependent":
            seen_contexts.append(context)
        return _packet(operation.operation_id)

    result = _executor(execute).execute(
        _plan(_operation("zeta"), _operation("alpha"), _operation("dependent", ("alpha",)))
    )

    assert len(seen_contexts) == 1
    context = seen_contexts[0]
    assert context.plan_id == "executor-plan"
    assert tuple(packet.packet_id for packet in context.dependency_packets) == ("packet-alpha",)
    assert tuple(packet.packet_id for packet in context.prior_packets) == ("packet-alpha",)
    assert tuple(item.operation_id for item in result.coverage.components) == (
        "alpha",
        "dependent",
        "zeta",
    )
    assert tuple(item.operation_id for item in result.execution_results) == (
        "alpha",
        "dependent",
        "zeta",
    )