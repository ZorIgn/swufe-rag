"""DAG execution with bounded, typed, read-only tool calls."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from academic.tools import AcademicTools
from agent.policies import RuntimePolicy
from agent.registry import RegisteredTool, ToolRegistry
from evidence.models import Coverage, EvidencePacket
from evidence.provenance import stable_id
from query.schemas import (
    ALL_OPERATION_TYPES,
    CheckCurriculumFeasibilityOperation,
    ExecutionPlan,
    ListCoursesBeforeSemesterOperation,
    ListUnavoidableCoursesOperation,
    Operation,
)


@dataclass(frozen=True)
class ToolFailure:
    operation_id: str
    code: str
    retryable: bool


def _merge(packets: list[EvidencePacket], plan_id: str) -> EvidencePacket:
    facts = {fact.fact_id: fact for packet in packets for fact in packet.facts}
    evidence = {item.evidence_id: item for packet in packets for item in packet.evidence}
    warnings = tuple(value for packet in packets for value in packet.warnings)
    results = tuple(value for packet in packets for value in packet.tool_results)
    conflicts = tuple(value for packet in packets for value in packet.conflicts)
    coverage = next((packet.coverage for packet in reversed(packets) if packet.coverage != Coverage()), Coverage())
    return EvidencePacket(packet_id=stable_id("packet", plan_id), facts=tuple(facts.values()), evidence=tuple(evidence.values()), coverage=coverage, warnings=warnings, conflicts=conflicts, tool_results=results)


class PlanExecutor:
    """Executes a plan only through registered operations; no arbitrary functions."""

    def __init__(self, registry: ToolRegistry, policy: RuntimePolicy) -> None:
        self.registry = registry
        self.policy = policy
        if self.registry.operation_types() != ALL_OPERATION_TYPES:
            missing = ALL_OPERATION_TYPES - self.registry.operation_types()
            extra = self.registry.operation_types() - ALL_OPERATION_TYPES
            raise ValueError(f"planner/executor mismatch: missing={sorted(missing)}, extra={sorted(extra)}")

    def execute(self, plan: ExecutionPlan) -> EvidencePacket:
        if len(plan.operations) > self.policy.max_tool_calls:
            return EvidencePacket(packet_id=stable_id("packet", plan.plan_id), warnings=("max_tool_calls_exceeded",))
        pending = {operation.operation_id: operation for operation in plan.operations}
        completed: set[str] = set()
        packets: list[EvidencePacket] = []
        while pending:
            ready = [operation for operation in pending.values() if set(operation.depends_on).issubset(completed)]
            if not ready:
                return _merge(packets + [EvidencePacket(packet_id=stable_id("packet", plan.plan_id, "cycle"), warnings=("invalid_operation_dependency_cycle",))], plan.plan_id)
            with ThreadPoolExecutor(max_workers=min(len(ready), self.policy.max_tool_calls)) as pool:
                futures = {pool.submit(self._execute_one, operation): operation for operation in ready}
                for future in as_completed(futures):
                    operation = futures[future]
                    try:
                        packets.append(future.result(timeout=self.policy.tool_timeout_seconds))
                    except Exception:
                        packets.append(EvidencePacket(packet_id=stable_id("packet", operation.operation_id, "error"), warnings=(f"tool_error:{operation.type}",), tool_results=(operation.tool_name,)))
                    completed.add(operation.operation_id)
                    pending.pop(operation.operation_id, None)
        return _merge(packets, plan.plan_id)

    def _execute_one(self, operation: Operation) -> EvidencePacket:
        return self.registry.for_operation(operation).execute(operation)


def standard_registry(academic: AcademicTools, policy: RuntimePolicy) -> ToolRegistry:
    registry = ToolRegistry()
    timeout = policy.tool_timeout_seconds
    registry.register(RegisteredTool("academic.list_courses", "list_courses", True, timeout, academic.list_courses))
    registry.register(RegisteredTool("academic.get_course", "get_course_detail", True, timeout, academic.get_course_detail))
    registry.register(RegisteredTool("academic.get_requirements", "get_graduation_requirements", True, timeout, academic.get_graduation_requirements))
    registry.register(RegisteredTool("academic.get_module_requirements", "get_module_requirements", True, timeout, academic.get_module_requirements))
    registry.register(RegisteredTool("academic.audit_progress", "audit_completed_courses", True, timeout, academic.audit_completed_courses))
    registry.register(RegisteredTool("academic.compare_programs", "compare_programs", True, timeout, academic.compare_programs))
    registry.register(RegisteredTool("policy.search", "retrieve_policy", True, timeout, academic.retrieve_policy))
    registry.register(RegisteredTool("source.resolve", "resolve_source", True, timeout, academic.resolve_source))

    def before(operation: ListCoursesBeforeSemesterOperation) -> EvidencePacket:
        records = academic.repository.list_courses(cohort=operation.args.cohort, program_id=operation.args.program_id, natures=operation.args.course_natures)
        selected = tuple(record for record in records if record.semester[:1].isdigit() and int(record.semester[:1]) < operation.args.deadline_semester)
        return academic._courses_packet(selected, program_id=operation.args.program_id, filters=("before_semester",))

    def unavoidable(operation: ListUnavoidableCoursesOperation) -> EvidencePacket:
        records = academic.repository.list_courses(cohort=operation.args.cohort, program_id=operation.args.program_id)
        selected = tuple(record for record in records if record.semester[:1].isdigit() and int(record.semester[:1]) > operation.args.after_semester and ("必修" in (record.nature or "") or "实践" in record.module_name))
        return academic._courses_packet(selected, program_id=operation.args.program_id, filters=("unavoidable_after_semester",))

    def feasibility(operation: CheckCurriculumFeasibilityOperation) -> EvidencePacket:
        records = academic.repository.list_courses(cohort=operation.args.cohort, program_id=operation.args.program_id)
        unavoidable_records = tuple(record for record in records if record.semester[:1].isdigit() and int(record.semester[:1]) >= operation.args.deadline_semester and ("必修" in (record.nature or "") or "实践" in record.module_name))
        packet = academic._courses_packet(unavoidable_records, program_id=operation.args.program_id, filters=("feasibility",))
        warning = "curriculum_feasibility:infeasible" if unavoidable_records else "curriculum_feasibility:feasible"
        return packet.model_copy(update={"warnings": (*packet.warnings, warning), "tool_results": ("academic.audit_progress",)})

    registry.register(RegisteredTool("academic.list_courses_before_semester", "list_courses_before_semester", True, timeout, before))
    registry.register(RegisteredTool("academic.list_unavoidable_courses", "list_unavoidable_courses", True, timeout, unavoidable))
    registry.register(RegisteredTool("academic.check_curriculum_feasibility", "check_curriculum_feasibility", True, timeout, feasibility))
    return registry


__all__ = ["PlanExecutor", "ToolFailure", "standard_registry"]
