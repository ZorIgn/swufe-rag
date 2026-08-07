"""DAG execution with bounded, typed, read-only tool calls.

The executor is intentionally the only place that turns individual tool
invocations into plan-level execution state. In particular, it never treats a
failed dependency as completed: downstream work can start only after every
dependency has succeeded.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from enum import Enum
from inspect import Parameter, signature
from time import monotonic
from typing import TYPE_CHECKING, Literal, Protocol, cast

from agent.policies import RuntimePolicy
from agent.registry import RegisteredTool, ToolRegistry
from evidence.models import CoverageComponent, CoverageReport, EvidencePacket, ToolExecutionResult
from evidence.provenance import stable_id
from query.schemas import (
    ALL_OPERATION_TYPES,
    CheckCurriculumFeasibilityOperation,
    ExecutionPlan,
    ListCoursesBeforeSemesterOperation,
    ListUnavoidableCoursesOperation,
    Operation,
)

if TYPE_CHECKING:
    from academic.tools import AcademicTools


@dataclass(frozen=True)
class ToolFailure:
    operation_id: str
    code: str
    retryable: bool


@dataclass(frozen=True)
class ToolExecutionContext:
    """Evidence available to a context-aware tool invocation.

    ``prior_packets`` contains every successful packet from an earlier DAG wave,
    sorted by operation ID. ``dependency_packets`` is the corresponding
    deterministic subset for direct dependencies.
    """

    plan_id: str
    operation_id: str
    prior_packets: tuple[EvidencePacket, ...]
    dependency_packets: tuple[EvidencePacket, ...]


class _ExecutionState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class _RunningTool:
    operation: Operation
    started_at: float
    deadline: float


class _KeywordContextTool(Protocol):
    def __call__(
        self, operation: Operation, *, context: ToolExecutionContext
    ) -> EvidencePacket: ...


def _unique(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _component_key(component: CoverageComponent) -> str:
    """Provide a stable tie-breaker for invalid duplicate coverage entries."""

    return component.model_dump_json()


def _merge(
    packets_by_operation: dict[str, EvidencePacket],
    execution_results: dict[str, ToolExecutionResult],
    plan_id: str,
    executor_warnings: tuple[str, ...] = (),
) -> EvidencePacket:
    """Merge successful packet payloads in operation-ID order.

    Coverage is collected per operation rather than selected from whichever
    concurrent future completed last.
    """

    ordered_packets = tuple(
        (operation_id, packets_by_operation[operation_id])
        for operation_id in sorted(packets_by_operation)
    )
    facts = {fact.fact_id: fact for _, packet in ordered_packets for fact in packet.facts}
    evidence = {item.evidence_id: item for _, packet in ordered_packets for item in packet.evidence}
    components: dict[str, CoverageComponent] = {}
    duplicate_components: list[str] = []
    for _, packet in ordered_packets:
        for component in sorted(packet.coverage.components, key=_component_key):
            previous = components.get(component.operation_id)
            if previous is None:
                components[component.operation_id] = component
            elif previous != component:
                components[component.operation_id] = min(previous, component, key=_component_key)
                duplicate_components.append(component.operation_id)
    warnings = _unique(
        [
            *executor_warnings,
            *(warning for _, packet in ordered_packets for warning in packet.warnings),
            *(
                f"duplicate_coverage_component:{operation_id}"
                for operation_id in sorted(set(duplicate_components))
            ),
        ]
    )
    conflicts = _unique(
        [conflict for _, packet in ordered_packets for conflict in packet.conflicts]
    )
    return EvidencePacket(
        packet_id=stable_id("packet", plan_id),
        facts=tuple(facts[fact_id] for fact_id in sorted(facts)),
        evidence=tuple(evidence[evidence_id] for evidence_id in sorted(evidence)),
        coverage=CoverageReport(
            components=tuple(components[operation_id] for operation_id in sorted(components))
        ),
        execution_results=tuple(
            execution_results[operation_id] for operation_id in sorted(execution_results)
        ),
        conflicts=conflicts,
        warnings=warnings,
    )


class PlanExecutor:
    """Execute a typed DAG through registered, read-only tools only."""

    def __init__(self, registry: ToolRegistry, policy: RuntimePolicy) -> None:
        self.registry = registry
        self.policy = policy
        if self.registry.operation_types() != ALL_OPERATION_TYPES:
            missing = ALL_OPERATION_TYPES - self.registry.operation_types()
            extra = self.registry.operation_types() - ALL_OPERATION_TYPES
            raise ValueError(
                f"planner/executor mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
            )

    def execute(self, plan: ExecutionPlan) -> EvidencePacket:
        """Execute ``plan`` within one deadline and return typed outcomes.

        The plan deadline starts before scheduling work and is not reset for
        each DAG wave. Timed-out Python threads cannot be force-killed, so the
        executor cancels queued futures and returns without waiting for running
        work at shutdown.
        """

        operations = tuple(plan.operations)
        plan_started_at = monotonic()
        execution_results: dict[str, ToolExecutionResult] = {}
        packets_by_operation: dict[str, EvidencePacket] = {}
        warnings: list[str] = []
        states = {operation.operation_id: _ExecutionState.PENDING for operation in operations}
        operations_by_id = {operation.operation_id: operation for operation in operations}

        def record(
            operation: Operation,
            *,
            state: _ExecutionState,
            status: Literal["success", "timeout", "failed", "dependency_failed", "skipped"],
            error_code: str | None = None,
            started_at: float | None = None,
        ) -> None:
            states[operation.operation_id] = state
            elapsed = 0.0 if started_at is None else max(0.0, (monotonic() - started_at) * 1000)
            execution_results[operation.operation_id] = ToolExecutionResult(
                operation_id=operation.operation_id,
                tool_name=operation.tool_name,
                status=status,
                latency_ms=elapsed,
                error_code=error_code,
            )

        if len(operations) > self.policy.max_tool_calls:
            warnings.append("max_tool_calls_exceeded")
            for operation in sorted(operations, key=lambda value: value.operation_id):
                record(
                    operation,
                    state=_ExecutionState.SKIPPED,
                    status="skipped",
                    error_code="max_tool_calls_exceeded",
                )
            return _merge(packets_by_operation, execution_results, plan.plan_id, tuple(warnings))

        if len(operations_by_id) != len(operations):
            warnings.append("duplicate_operation_id")
            for operation in sorted(operations, key=lambda value: value.operation_id):
                record(
                    operation,
                    state=_ExecutionState.FAILED,
                    status="failed",
                    error_code="duplicate_operation_id",
                )
            return _merge(packets_by_operation, execution_results, plan.plan_id, tuple(warnings))

        plan_deadline = plan_started_at + max(0.0, self.policy.tool_timeout_seconds)
        worker_count = max(1, min(len(operations) or 1, self.policy.max_tool_calls))
        executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="academic-tool")
        running: dict[Future[EvidencePacket], _RunningTool] = {}
        try:
            while True:
                pending = tuple(
                    operation
                    for operation in operations
                    if states[operation.operation_id] is _ExecutionState.PENDING
                )
                if not pending and not running:
                    break

                # A downstream operation is not ready merely because a parent
                # finished. It is terminally blocked unless every parent
                # succeeded.
                for operation in sorted(pending, key=lambda value: value.operation_id):
                    dependency_states = [
                        states.get(dependency) for dependency in operation.depends_on
                    ]
                    if any(state is None for state in dependency_states):
                        record(
                            operation,
                            state=_ExecutionState.SKIPPED,
                            status="dependency_failed",
                            error_code="unknown_dependency",
                        )
                    elif any(
                        state in {_ExecutionState.FAILED, _ExecutionState.SKIPPED}
                        for state in dependency_states
                    ):
                        record(
                            operation,
                            state=_ExecutionState.SKIPPED,
                            status="dependency_failed",
                            error_code="dependency_failed",
                        )

                now = monotonic()
                if now >= plan_deadline:
                    for future, started in sorted(
                        running.items(), key=lambda item: item[1].operation.operation_id
                    ):
                        future.cancel()
                        record(
                            started.operation,
                            state=_ExecutionState.FAILED,
                            status="timeout",
                            error_code="plan_deadline_exceeded",
                            started_at=started.started_at,
                        )
                    running.clear()
                    for operation in sorted(
                        (
                            value
                            for value in operations
                            if states[value.operation_id] is _ExecutionState.PENDING
                        ),
                        key=lambda value: value.operation_id,
                    ):
                        record(
                            operation,
                            state=_ExecutionState.SKIPPED,
                            status="skipped",
                            error_code="plan_deadline_exceeded",
                        )
                    break

                ready = tuple(
                    operation
                    for operation in operations
                    if states[operation.operation_id] is _ExecutionState.PENDING
                    and all(
                        states.get(dependency) is _ExecutionState.SUCCEEDED
                        for dependency in operation.depends_on
                    )
                )
                for operation in sorted(ready, key=lambda value: value.operation_id):
                    registered = self.registry.for_operation(operation)
                    started_at = monotonic()
                    if started_at >= plan_deadline:
                        # The next loop records any still-pending work as
                        # skipped under the same plan-level deadline.
                        break
                    tool_deadline = min(
                        plan_deadline,
                        started_at + max(0.0, registered.timeout_seconds),
                    )
                    if tool_deadline <= started_at:
                        record(
                            operation,
                            state=_ExecutionState.FAILED,
                            status="timeout",
                            error_code="tool_timeout_exceeded",
                        )
                        continue
                    context = self._context_for(
                        plan, operation, packets_by_operation, operations_by_id
                    )
                    future = executor.submit(self._execute_one, operation, context)
                    running[future] = _RunningTool(
                        operation=operation,
                        started_at=started_at,
                        deadline=tool_deadline,
                    )
                    states[operation.operation_id] = _ExecutionState.RUNNING

                if not running:
                    unresolved = tuple(
                        operation
                        for operation in operations
                        if states[operation.operation_id] is _ExecutionState.PENDING
                    )
                    if unresolved:
                        warnings.append("invalid_operation_dependency_cycle")
                        for operation in sorted(unresolved, key=lambda value: value.operation_id):
                            record(
                                operation,
                                state=_ExecutionState.SKIPPED,
                                status="skipped",
                                error_code="invalid_operation_dependency_cycle",
                            )
                    continue

                now = monotonic()
                next_deadline = min(
                    plan_deadline,
                    *(started.deadline for started in running.values()),
                )
                completed, _ = wait(
                    tuple(running),
                    timeout=max(0.0, next_deadline - now),
                    return_when=FIRST_COMPLETED,
                )
                for future in sorted(
                    completed, key=lambda value: running[value].operation.operation_id
                ):
                    started = running.pop(future)
                    try:
                        packet = future.result()
                        if not isinstance(packet, EvidencePacket):
                            raise TypeError("registered tool did not return EvidencePacket")
                    except Exception:
                        record(
                            started.operation,
                            state=_ExecutionState.FAILED,
                            status="failed",
                            error_code="tool_execution_failed",
                            started_at=started.started_at,
                        )
                    else:
                        packets_by_operation[started.operation.operation_id] = packet
                        record(
                            started.operation,
                            state=_ExecutionState.SUCCEEDED,
                            status="success",
                            started_at=started.started_at,
                        )

                now = monotonic()
                for future, started in sorted(
                    tuple(running.items()), key=lambda item: item[1].operation.operation_id
                ):
                    if now >= started.deadline:
                        running.pop(future)
                        future.cancel()
                        record(
                            started.operation,
                            state=_ExecutionState.FAILED,
                            status="timeout",
                            error_code=(
                                "plan_deadline_exceeded"
                                if now >= plan_deadline
                                else "tool_timeout_exceeded"
                            ),
                            started_at=started.started_at,
                        )
        finally:
            # Do not let a timed-out callback block the agent response while it
            # unwinds. Queued work is cancelled; a running Python thread cannot
            # be forcibly stopped and is ignored after its typed timeout result.
            executor.shutdown(wait=False, cancel_futures=True)

        return _merge(packets_by_operation, execution_results, plan.plan_id, tuple(warnings))

    @staticmethod
    def _context_for(
        plan: ExecutionPlan,
        operation: Operation,
        packets_by_operation: dict[str, EvidencePacket],
        operations_by_id: dict[str, Operation],
    ) -> ToolExecutionContext:
        ancestors: set[str] = set()
        pending = list(operation.depends_on)
        while pending:
            dependency_id = pending.pop()
            if dependency_id in ancestors:
                continue
            ancestors.add(dependency_id)
            parent = operations_by_id.get(dependency_id)
            if parent is not None:
                pending.extend(parent.depends_on)
        prior_packets = tuple(
            packets_by_operation[operation_id]
            for operation_id in sorted(ancestors)
            if operation_id in packets_by_operation
        )
        dependency_packets = tuple(
            packets_by_operation[operation_id]
            for operation_id in sorted(operation.depends_on)
            if operation_id in packets_by_operation
        )
        return ToolExecutionContext(
            plan_id=plan.plan_id,
            operation_id=operation.operation_id,
            prior_packets=prior_packets,
            dependency_packets=dependency_packets,
        )

    def _execute_one(self, operation: Operation, context: ToolExecutionContext) -> EvidencePacket:
        callback = self.registry.for_operation(operation).execute
        context_style = self._context_style(callback)
        if context_style == "positional":
            contextual = cast(Callable[[Operation, ToolExecutionContext], EvidencePacket], callback)
            return contextual(operation, context)
        if context_style == "keyword":
            contextual_keyword = cast(_KeywordContextTool, callback)
            return contextual_keyword(operation, context=context)
        return callback(operation)

    @staticmethod
    def _context_style(
        callback: Callable[[Operation], EvidencePacket],
    ) -> Literal["none", "positional", "keyword"]:
        try:
            parameters = tuple(signature(callback).parameters.values())
        except (TypeError, ValueError):
            return "none"
        if any(parameter.kind is Parameter.VAR_POSITIONAL for parameter in parameters):
            return "positional"
        context_parameter = next(
            (
                parameter
                for parameter in parameters
                if parameter.name in {"context", "execution_context"}
            ),
            None,
        )
        if context_parameter is not None:
            if context_parameter.kind is Parameter.KEYWORD_ONLY:
                return "keyword"
            return "positional"
        positional = tuple(
            parameter
            for parameter in parameters
            if parameter.kind in {Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD}
        )
        if len(positional) >= 2:
            return "positional"
        if any(parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters):
            return "keyword"
        return "none"


def standard_registry(academic: AcademicTools, policy: RuntimePolicy) -> ToolRegistry:
    registry = ToolRegistry()
    timeout = policy.tool_timeout_seconds
    registry.register(
        RegisteredTool(
            "academic.list_courses", "list_courses", True, timeout, academic.list_courses
        )
    )
    registry.register(
        RegisteredTool(
            "academic.get_course", "get_course_detail", True, timeout, academic.get_course_detail
        )
    )
    registry.register(
        RegisteredTool(
            "academic.get_requirements",
            "get_graduation_requirements",
            True,
            timeout,
            academic.get_graduation_requirements,
        )
    )
    registry.register(
        RegisteredTool(
            "academic.get_module_requirements",
            "get_module_requirements",
            True,
            timeout,
            academic.get_module_requirements,
        )
    )
    registry.register(
        RegisteredTool(
            "academic.audit_progress",
            "audit_completed_courses",
            True,
            timeout,
            academic.audit_completed_courses,
        )
    )
    registry.register(
        RegisteredTool(
            "academic.compare_programs",
            "compare_programs",
            True,
            timeout,
            academic.compare_programs,
        )
    )
    registry.register(
        RegisteredTool("policy.search", "retrieve_policy", True, timeout, academic.retrieve_policy)
    )
    registry.register(
        RegisteredTool("source.resolve", "resolve_source", True, timeout, academic.resolve_source)
    )

    def before(operation: ListCoursesBeforeSemesterOperation) -> EvidencePacket:
        records = academic.repository.list_courses(
            cohort=operation.args.cohort,
            program_id=operation.args.program_id,
            natures=operation.args.course_natures,
        )
        selected = tuple(
            record
            for record in records
            if record.semester[:1].isdigit()
            and int(record.semester[:1]) < operation.args.deadline_semester
        )
        return academic._courses_packet(
            selected,
            program_id=operation.args.program_id,
            filters=("before_semester",),
            operation_id=operation.operation_id,
        )

    def unavoidable(operation: ListUnavoidableCoursesOperation) -> EvidencePacket:
        records = academic.repository.list_courses(
            cohort=operation.args.cohort, program_id=operation.args.program_id
        )
        selected = tuple(
            record
            for record in records
            if record.semester[:1].isdigit()
            and int(record.semester[:1]) > operation.args.after_semester
            and ("必修" in (record.nature or "") or "实践" in record.module_name)
        )
        return academic._courses_packet(
            selected,
            program_id=operation.args.program_id,
            filters=("unavoidable_after_semester",),
            operation_id=operation.operation_id,
        )

    def feasibility(operation: CheckCurriculumFeasibilityOperation) -> EvidencePacket:
        records = academic.repository.list_courses(
            cohort=operation.args.cohort, program_id=operation.args.program_id
        )
        unavoidable_records = tuple(
            record
            for record in records
            if record.semester[:1].isdigit()
            and int(record.semester[:1]) >= operation.args.deadline_semester
            and ("必修" in (record.nature or "") or "实践" in record.module_name)
        )
        packet = academic._courses_packet(
            unavoidable_records,
            program_id=operation.args.program_id,
            filters=("feasibility",),
            operation_id=operation.operation_id,
        )
        warning = (
            "curriculum_feasibility:infeasible"
            if unavoidable_records
            else "curriculum_feasibility:feasible"
        )
        return packet.model_copy(update={"warnings": (*packet.warnings, warning)})

    registry.register(
        RegisteredTool(
            "academic.list_courses_before_semester",
            "list_courses_before_semester",
            True,
            timeout,
            before,
        )
    )
    registry.register(
        RegisteredTool(
            "academic.list_unavoidable_courses",
            "list_unavoidable_courses",
            True,
            timeout,
            unavoidable,
        )
    )
    registry.register(
        RegisteredTool(
            "academic.check_curriculum_feasibility",
            "check_curriculum_feasibility",
            True,
            timeout,
            feasibility,
        )
    )
    return registry


__all__ = ["PlanExecutor", "ToolExecutionContext", "ToolFailure", "standard_registry"]
