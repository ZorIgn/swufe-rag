from __future__ import annotations

from typing import Literal

from agent.coverage_gate import CoverageGate
from evidence.models import (
    CoverageComponent,
    CoverageKind,
    CoverageReport,
    EvidencePacket,
    ToolExecutionResult,
)
from query.planner import build_plan
from query.schemas import NormalizedQuery, Operation


def _query(
    *,
    outputs: tuple[str, ...],
    information_scope: str = "curriculum",
    unsupported_outputs: tuple[str, ...] = (),
) -> NormalizedQuery:
    return NormalizedQuery(
        raw_question="coverage lineage fixture",
        intent="course_query",
        requested_outputs=outputs,  # type: ignore[arg-type]
        cohort=2024,
        program_ids=("program-x",),
        program_names=("program-x",),
        course_ids=("course-x",),
        policy_topics=("转专业",),
        information_scope=information_scope,  # type: ignore[arg-type]
        unsupported_outputs=unsupported_outputs,  # type: ignore[arg-type]
    )


def _result(
    operation: Operation,
    *,
    status: Literal["success", "timeout", "failed", "dependency_failed", "skipped"] = "success",
) -> ToolExecutionResult:
    return ToolExecutionResult(
        operation_id=operation.operation_id,
        tool_name=operation.tool_name,
        status=status,
        latency_ms=0.1,
    )


def _component(
    operation: Operation,
    *,
    kind: CoverageKind = "course_set",
    tool_name: str | None = None,
    trusted: bool = True,
) -> CoverageComponent:
    return CoverageComponent(
        operation_id=operation.operation_id,
        tool_name=tool_name or operation.tool_name,
        kind=kind,
        complete=True,
        authoritative=True if kind == "policy" else None,
        scope_matched=True,
        version_resolved=True,
        conflict_free=True,
        trusted_evidence=trusted,
    )


def test_same_kind_component_cannot_satisfy_a_different_output_operation() -> None:
    query = _query(outputs=("course_list", "course_detail"))
    plan = build_plan(query)
    list_operation, detail_operation = plan.operations
    packet = EvidencePacket(
        packet_id="packet",
        execution_results=(_result(list_operation), _result(detail_operation)),
        coverage=CoverageReport(components=(_component(list_operation),)),
    )

    decision = CoverageGate().evaluate(query, plan, packet)

    assert decision.sufficient
    assert [(item.output, item.status) for item in decision.output_statuses] == [
        ("course_list", "fulfilled"),
        ("course_detail", "missing_data"),
    ]
    detail = decision.output_statuses[1]
    assert any("coverage_component_missing" in reason for reason in detail.reasons)


def test_component_tool_must_match_the_contracted_operation() -> None:
    query = _query(outputs=("course_list",))
    plan = build_plan(query)
    operation = plan.operations[0]
    packet = EvidencePacket(
        packet_id="packet",
        execution_results=(_result(operation),),
        coverage=CoverageReport(
            components=(
                _component(operation, tool_name="academic.get_course"),
            )
        ),
    )

    decision = CoverageGate().evaluate(query, plan, packet)

    assert not decision.sufficient
    assert decision.output_statuses[0].status == "refused"
    assert any(
        "coverage_tool_mismatch" in reason
        for reason in decision.output_statuses[0].reasons
    )


def test_duplicate_coverage_for_one_operation_fails_packet_integrity() -> None:
    query = _query(outputs=("course_list",))
    plan = build_plan(query)
    operation = plan.operations[0]
    component = _component(operation)
    packet = EvidencePacket(
        packet_id="packet",
        execution_results=(_result(operation),),
        coverage=CoverageReport(components=(component, component)),
    )

    decision = CoverageGate().evaluate(query, plan, packet)

    assert not decision.sufficient
    assert f"duplicate_coverage_component:{operation.operation_id}" in decision.reasons


def test_untrusted_course_component_is_refused_before_synthesis() -> None:
    query = _query(outputs=("course_list",))
    plan = build_plan(query)
    operation = plan.operations[0]
    packet = EvidencePacket(
        packet_id="packet",
        execution_results=(_result(operation),),
        coverage=CoverageReport(components=(_component(operation, trusted=False),)),
    )

    decision = CoverageGate().evaluate(query, plan, packet)

    assert not decision.sufficient
    assert decision.output_statuses[0].status == "refused"
    assert "course_set_untrusted_evidence" in decision.output_statuses[0].reasons


def test_safe_policy_output_survives_an_independent_unsupported_output() -> None:
    query = _query(
        outputs=("course_list", "policy_explanation"),
        information_scope="actual_offerings",
        unsupported_outputs=("course_list",),
    )
    plan = build_plan(query)
    policy_operation = plan.operations[0]
    packet = EvidencePacket(
        packet_id="packet",
        execution_results=(_result(policy_operation),),
        coverage=CoverageReport(
            components=(_component(policy_operation, kind="policy"),)
        ),
    )

    decision = CoverageGate().evaluate(query, plan, packet)

    assert decision.sufficient
    assert [(item.output, item.status) for item in decision.output_statuses] == [
        ("course_list", "unsupported"),
        ("policy_explanation", "fulfilled"),
    ]


def test_failed_producer_cannot_be_hidden_by_a_complete_component() -> None:
    query = _query(outputs=("course_list",))
    plan = build_plan(query)
    operation = plan.operations[0]
    packet = EvidencePacket(
        packet_id="packet",
        execution_results=(_result(operation, status="failed"),),
        coverage=CoverageReport(components=(_component(operation),)),
    )

    decision = CoverageGate().evaluate(query, plan, packet)

    assert not decision.sufficient
    assert decision.output_statuses[0].status == "missing_data"
    assert f"operation_failed:{operation.operation_id}" in decision.output_statuses[0].reasons


def test_all_unavailable_outputs_are_not_answer_eligible() -> None:
    query = _query(
        outputs=("course_list",),
        information_scope="actual_offerings",
        unsupported_outputs=("course_list",),
    )
    plan = build_plan(query)

    decision = CoverageGate().evaluate(
        query,
        plan,
        EvidencePacket(packet_id="packet"),
    )

    assert not decision.sufficient
    assert decision.output_statuses[0].status == "unsupported"
