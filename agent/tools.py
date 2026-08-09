"""DAG execution with bounded, typed, read-only tool calls.

The executor is intentionally the only place that turns individual tool
invocations into plan-level execution state. In particular, it never treats a
failed dependency as completed: downstream work can start only after every
dependency has succeeded.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from enum import Enum
from inspect import Parameter, signature
from time import monotonic
from typing import TYPE_CHECKING, Literal, cast

from agent.policies import RuntimePolicy
from agent.registry import ContextualTool, RegisteredTool, ToolExecutionContext, ToolRegistry
from evidence.models import (
    CoverageComponent,
    CoverageReport,
    DerivedFact,
    Evidence,
    EvidencePacket,
    Fact,
    ToolExecutionResult,
)
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
    # Dict insertion order preserves the ranking emitted inside each tool
    # packet while still deduplicating stable identifiers.  Sorting by hash-like
    # fact/evidence IDs here used to destroy retrieval order, so the policy
    # synthesizer could quote an arbitrary lower-ranked chunk as its Top-1.
    facts: dict[str, Fact | DerivedFact] = {}
    evidence: dict[str, Evidence] = {}
    for _, packet in ordered_packets:
        for fact in packet.facts:
            facts.setdefault(fact.fact_id, fact)
        for item in packet.evidence:
            evidence.setdefault(item.evidence_id, item)
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
        facts=tuple(facts.values()),
        evidence=tuple(evidence.values()),
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
            contextual_keyword = cast(ContextualTool[Operation], callback)
            return contextual_keyword(operation, context=context)
        standard = cast(Callable[[Operation], EvidencePacket], callback)
        return standard(operation)

    @staticmethod
    def _context_style(
        callback: Callable[[Operation], EvidencePacket] | ContextualTool[Operation],
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


_SEMESTER_NUMBER = re.compile(r"^\s*(\d+)")


def _semester_number(value: object) -> int | None:
    match = _SEMESTER_NUMBER.match(str(value or ""))
    return int(match.group(1)) if match else None


def _fact_value(
    values: dict[str, Fact | DerivedFact], predicate: str
) -> object | None:
    fact = values.get(predicate)
    return fact.value if fact is not None else None


def _string_fact_value(
    values: dict[str, Fact | DerivedFact], predicate: str, *, default: str = ""
) -> str:
    value = _fact_value(values, predicate)
    return default if value is None else str(value)


def _numeric_fact(
    values: dict[str, Fact | DerivedFact], predicate: str
) -> Fact | DerivedFact | None:
    fact = values.get(predicate)
    return fact if fact is not None and isinstance(fact.value, (int, float)) else None


def _fact_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"fact value is not numeric: {value!r}")


def _course_groups(
    packets: tuple[EvidencePacket, ...],
) -> dict[str, dict[str, Fact | DerivedFact]]:
    groups: dict[str, dict[str, Fact | DerivedFact]] = {}
    for packet in packets:
        for fact in packet.facts:
            if fact.type == "course":
                groups.setdefault(fact.subject, {})[fact.predicate] = fact
    return groups


def _pack_course_groups(
    *,
    operation_id: str,
    tool_name: str,
    source_packets: tuple[EvidencePacket, ...],
    groups: dict[str, dict[str, Fact | DerivedFact]],
    selected_subjects: set[str],
    warning_codes: tuple[str, ...] = (),
) -> EvidencePacket:
    facts = tuple(
        fact
        for subject in sorted(selected_subjects)
        for fact in groups.get(subject, {}).values()
    )
    evidence_ids = {evidence_id for fact in facts for evidence_id in fact.evidence_ids}
    evidence_by_id = {
        evidence.evidence_id: evidence
        for packet in source_packets
        for evidence in packet.evidence
    }
    source_components = tuple(
        component
        for packet in source_packets
        for component in packet.coverage.components
        if component.kind == "course_set"
    )
    complete = bool(source_components) and all(component.complete for component in source_components)
    trusted = bool(source_components) and all(
        component.trusted_evidence is not False for component in source_components
    )
    reason_values: list[str] = []
    if not source_components:
        reason_values.append("course_dependency_missing")
    if not complete:
        reason_values.append("course_dependency_incomplete")
    if not selected_subjects:
        reason_values.append("empty_result")
    reasons = tuple(reason_values)
    return EvidencePacket(
        packet_id=stable_id("packet", operation_id),
        facts=facts,
        evidence=tuple(
            evidence_by_id[evidence_id]
            for evidence_id in sorted(evidence_ids)
            if evidence_id in evidence_by_id
        ),
        coverage=CoverageReport(
            components=(
                CoverageComponent(
                    operation_id=operation_id,
                    tool_name=tool_name,
                    kind="course_set",
                    complete=complete,
                    expected_count=len(selected_subjects),
                    returned_count=len(selected_subjects),
                    scope_matched=True,
                    version_resolved=True,
                    conflict_free=True,
                    trusted_evidence=trusted,
                    reasons=reasons,
                ),
            )
        ),
        warnings=warning_codes,
    )


def _completed_course_subjects(packets: tuple[EvidencePacket, ...]) -> set[str]:
    return {
        fact.subject
        for packet in packets
        if any(component.kind == "audit" for component in packet.coverage.components)
        for fact in packet.facts
        if fact.type == "course"
    }


def _is_mandatory_course(values: dict[str, Fact | DerivedFact]) -> bool:
    nature = _string_fact_value(values, "nature").lower()
    module = _string_fact_value(values, "module").lower()
    return "必修" in nature or "required" in nature or "实践" in module or "practice" in module


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

    def before(
        operation: ListCoursesBeforeSemesterOperation, *, context: ToolExecutionContext
    ) -> EvidencePacket:
        catalog_packets = tuple(
            packet
            for packet in context.dependency_packets
            if any(component.kind == "course_set" for component in packet.coverage.components)
        )
        groups = _course_groups(catalog_packets)
        completed = _completed_course_subjects(context.dependency_packets)
        unresolved = {
            subject
            for subject, values in groups.items()
            if subject not in completed
            and _semester_number(_fact_value(values, "semester"))
            is None
        }
        selected = {
            subject
            for subject, values in groups.items()
            if subject not in completed
            and (
                semester := _semester_number(_fact_value(values, "semester"))
            )
            is not None
            and semester < operation.args.deadline_semester
        }
        return _pack_course_groups(
            operation_id=operation.operation_id,
            tool_name="academic.list_courses_before_semester",
            source_packets=catalog_packets,
            groups=groups,
            selected_subjects=selected,
            warning_codes=("course_semester_unresolved",) if unresolved else (),
        )

    def unavoidable(
        operation: ListUnavoidableCoursesOperation, *, context: ToolExecutionContext
    ) -> EvidencePacket:
        catalog_packets = tuple(
            packet
            for packet in context.dependency_packets
            if any(component.kind == "course_set" for component in packet.coverage.components)
        )
        groups = _course_groups(catalog_packets)
        completed = _completed_course_subjects(context.dependency_packets)
        unresolved_mandatory = {
            subject
            for subject, values in groups.items()
            if subject not in completed
            and _is_mandatory_course(values)
            and _semester_number(_fact_value(values, "semester"))
            is None
        }
        # deadline_semester is an exclusive boundary: "before semester 7"
        # means semesters 1-6 are usable and semester 7 itself is already late.
        selected = {
            subject
            for subject, values in groups.items()
            if subject not in completed
            and _is_mandatory_course(values)
            and (
                semester := _semester_number(_fact_value(values, "semester"))
            )
            is not None
            and semester >= operation.args.after_semester
        }
        return _pack_course_groups(
            operation_id=operation.operation_id,
            tool_name="academic.list_unavoidable_courses",
            source_packets=catalog_packets,
            groups=groups,
            selected_subjects=selected,
            warning_codes=("mandatory_course_semester_unresolved",)
            if unresolved_mandatory
            else (),
        )

    def feasibility(
        operation: CheckCurriculumFeasibilityOperation, *, context: ToolExecutionContext
    ) -> EvidencePacket:
        audit_packet = next(
            (
                packet
                for packet in context.dependency_packets
                if any(component.kind == "audit" for component in packet.coverage.components)
            ),
            None,
        )
        before_packet = next(
            (
                packet
                for packet in context.dependency_packets
                if any(
                    component.tool_name == "academic.list_courses_before_semester"
                    for component in packet.coverage.components
                )
            ),
            None,
        )
        unavoidable_packet = next(
            (
                packet
                for packet in context.dependency_packets
                if any(
                    component.tool_name == "academic.list_unavoidable_courses"
                    for component in packet.coverage.components
                )
            ),
            None,
        )
        dependencies = tuple(
            packet
            for packet in (audit_packet, before_packet, unavoidable_packet)
            if packet is not None
        )
        remaining_facts = tuple(
            fact
            for fact in (() if audit_packet is None else audit_packet.facts)
            if fact.predicate == "remaining_credits" and isinstance(fact.value, (int, float))
        )
        completed_ids = {
            str(fact.value)
            for fact in (() if audit_packet is None else audit_packet.facts)
            if fact.type == "course" and fact.predicate == "course_id"
        }
        requested_completed_ids = set(operation.args.completed_course_ids)
        before_groups = _course_groups(() if before_packet is None else (before_packet,))
        blocker_groups = _course_groups(
            () if unavoidable_packet is None else (unavoidable_packet,)
        )

        facts: list[Fact | DerivedFact] = []
        evidence_by_id = {
            evidence.evidence_id: evidence
            for packet in dependencies
            for evidence in packet.evidence
        }
        reasons: list[Fact] = []
        shortfall_found = False
        uncertainty_found = False

        if len(dependencies) != 3 or not remaining_facts:
            uncertainty_found = True
            reasons.append(
                Fact(
                    fact_id=stable_id("fact", operation.operation_id, "missing_dependencies"),
                    type="diagnostic",
                    subject="培养方案可行性",
                    predicate="feasibility_reason",
                    value="缺少完整的培养方案要求或依赖计算结果。",
                    derivation="tool_result",
                )
            )
        if not requested_completed_ids.issubset(completed_ids):
            uncertainty_found = True
            reasons.append(
                Fact(
                    fact_id=stable_id("fact", operation.operation_id, "completed_id_mismatch"),
                    type="diagnostic",
                    subject="培养方案可行性",
                    predicate="feasibility_reason",
                    value="部分已修课程未出现在上游学业审计结果中。",
                    derivation="tool_result",
                )
            )

        for subject, values in sorted(blocker_groups.items()):
            name = _string_fact_value(values, "name", default=subject)
            semester = _semester_number(_fact_value(values, "semester"))
            blocker_support = tuple(values.values())
            reasons.append(
                Fact(
                    fact_id=stable_id("fact", operation.operation_id, subject, "late_mandatory"),
                    type="progress",
                    subject=name,
                    predicate="feasibility_reason",
                    value=(
                        f"{name}安排在第{semester}学期且尚未完成，"
                        f"不满足第{operation.args.deadline_semester}学期开始前完成的目标。"
                    ),
                    source_record_ids=tuple(
                        dict.fromkeys(
                            record_id
                            for fact in blocker_support
                            for record_id in fact.source_record_ids
                        )
                    ),
                    evidence_ids=tuple(
                        dict.fromkeys(
                            evidence_id
                            for fact in blocker_support
                            for evidence_id in fact.evidence_ids
                        )
                    ),
                    derivation="tool_result",
                )
            )

        remaining_by_module = {fact.subject: fact for fact in remaining_facts}
        before_by_module: dict[str, list[dict[str, Fact | DerivedFact]]] = {}
        for values in before_groups.values():
            module_fact = values.get("module")
            if module_fact is not None:
                before_by_module.setdefault(str(module_fact.value), []).append(values)

        for module, remaining_fact in sorted(remaining_by_module.items()):
            gap = _fact_float(remaining_fact.value)
            if gap <= 0:
                continue
            candidates = before_by_module.get(module, [])
            credit_facts = tuple(
                credit_fact
                for values in candidates
                if (credit_fact := _numeric_fact(values, "credits")) is not None
            )
            available = sum((_fact_float(fact.value) for fact in credit_facts), 0.0)
            missing_credit = any(values.get("credits") is None for values in candidates)
            credit_support: tuple[Fact | DerivedFact, ...] = (remaining_fact, *credit_facts)
            support_evidence = tuple(
                dict.fromkeys(
                    evidence_id for fact in credit_support for evidence_id in fact.evidence_ids
                )
            )
            support_records = tuple(
                dict.fromkeys(
                    record_id for fact in credit_support for record_id in fact.source_record_ids
                )
            )
            if available + 1e-9 < gap and missing_credit:
                uncertainty_found = True
                reasons.append(
                    Fact(
                        fact_id=stable_id("fact", operation.operation_id, module, "credits_unknown"),
                        type="diagnostic",
                        subject=module,
                        predicate="feasibility_reason",
                        value=f"{module}尚差{gap:g}学分，但边界前课程存在未标注学分，无法判断是否足够。",
                        derivation="tool_result",
                    )
                )
            elif available + 1e-9 < gap:
                shortfall_found = True
                reasons.append(
                    Fact(
                        fact_id=stable_id("fact", operation.operation_id, module, "credit_shortfall"),
                        type="progress",
                        subject=module,
                        predicate="feasibility_reason",
                        value=(
                            f"{module}尚差{gap:g}学分，但第{operation.args.deadline_semester}"
                            f"学期开始前列明的未修课程仅有{available:g}学分。"
                        ),
                        source_record_ids=support_records,
                        evidence_ids=support_evidence,
                        derivation="tool_result",
                    )
                )
            else:
                reasons.append(
                    Fact(
                        fact_id=stable_id("fact", operation.operation_id, module, "capacity"),
                        type="progress",
                        subject=module,
                        predicate="feasibility_reason",
                        value=(
                            f"{module}尚差{gap:g}学分，边界前列明的未修课程共有"
                            f"{available:g}学分，可覆盖该最低学分差额。"
                        ),
                        source_record_ids=support_records,
                        evidence_ids=support_evidence,
                        derivation="tool_result",
                    )
                )

        if any(
            warning in {"course_semester_unresolved", "mandatory_course_semester_unresolved"}
            for packet in dependencies
            for warning in packet.warnings
        ):
            uncertainty_found = True
            reasons.append(
                Fact(
                    fact_id=stable_id("fact", operation.operation_id, "semester_unknown"),
                    type="diagnostic",
                    subject="培养方案可行性",
                    predicate="feasibility_reason",
                    value="仍有课程缺少可解析的开课学期，无法完成边界判断。",
                    derivation="tool_result",
                )
            )

        if blocker_groups or shortfall_found:
            status = "infeasible"
            status_text = "不可行（按培养方案结构）"
        elif uncertainty_found:
            status = "insufficient_data"
            status_text = "信息不足，无法判断"
        else:
            status = "feasible"
            status_text = "可行（仅按培养方案结构）"
            if all(_fact_float(fact.value) <= 0 for fact in remaining_facts):
                reasons.append(
                    Fact(
                        fact_id=stable_id("fact", operation.operation_id, "all_requirements_met"),
                        type="diagnostic",
                        subject="培养方案可行性",
                        predicate="feasibility_reason",
                        value="所有已结构化模块的最低学分差额均为0，且未发现尚未完成的边界后必修或实践课程。",
                        derivation="tool_result",
                    )
                )

        status_fact = Fact(
            fact_id=stable_id("fact", operation.operation_id, "status"),
            type="decision",
            subject="培养方案可行性",
            predicate="feasibility_status",
            value=status_text,
            derivation="tool_result",
        )
        facts.extend((status_fact, *reasons))
        used_evidence_ids = {evidence_id for fact in facts for evidence_id in fact.evidence_ids}
        return EvidencePacket(
            packet_id=stable_id("packet", operation.operation_id),
            facts=tuple(facts),
            evidence=tuple(
                evidence_by_id[evidence_id]
                for evidence_id in sorted(used_evidence_ids)
                if evidence_id in evidence_by_id
            ),
            coverage=CoverageReport(
                components=(
                    CoverageComponent(
                        operation_id=operation.operation_id,
                        tool_name="academic.check_curriculum_feasibility",
                        kind="course_set",
                        complete=True,
                        expected_count=len(remaining_facts),
                        returned_count=len(remaining_facts),
                        scope_matched=True,
                        version_resolved=True,
                        conflict_free=True,
                        trusted_evidence=all(
                            component.trusted_evidence is not False
                            for packet in dependencies
                            for component in packet.coverage.components
                        ),
                        reasons=("feasibility_insufficient_data",)
                        if status == "insufficient_data"
                        else (),
                    ),
                )
            ),
            warnings=(f"curriculum_feasibility:{status}",),
        )

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
