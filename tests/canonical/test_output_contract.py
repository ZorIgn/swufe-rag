from __future__ import annotations

from agent.coverage_gate import CoverageGate
from evidence.models import CoverageComponent, CoverageReport, EvidencePacket, ToolExecutionResult
from query.planner import build_plan
from query.schemas import NormalizedQuery


def _query(
    *,
    outputs: tuple[str, ...],
    intent: str = "course_query",
    program_ids: tuple[str, ...] = ("program-x",),
    information_scope: str = "curriculum",
    unsupported_outputs: tuple[str, ...] = (),
) -> NormalizedQuery:
    return NormalizedQuery(
        raw_question="fixture output contract",
        intent=intent,  # type: ignore[arg-type]
        requested_outputs=outputs,  # type: ignore[arg-type]
        cohort=2024,
        program_ids=program_ids,
        program_names=tuple(program_ids),
        information_scope=information_scope,  # type: ignore[arg-type]
        unsupported_outputs=unsupported_outputs,  # type: ignore[arg-type]
    )


def test_course_and_policy_outputs_plan_both_typed_operations() -> None:
    query = _query(outputs=("course_list", "policy_explanation"))
    query = query.model_copy(update={"policy_topics": ("转专业",)})

    plan = build_plan(query)

    assert [operation.type for operation in plan.operations] == [
        "list_courses",
        "retrieve_policy",
    ]
    assert plan.operations[1].args.topics == ("转专业",)
    assert [(item.output, item.status) for item in plan.output_contract] == [
        ("course_list", "fulfilled"),
        ("policy_explanation", "fulfilled"),
    ]


def test_module_and_course_outputs_are_not_dropped() -> None:
    query = _query(
        outputs=("module_requirements", "course_list"),
        intent="module_requirements",
    )

    plan = build_plan(query)

    assert [operation.type for operation in plan.operations] == [
        "get_module_requirements",
        "list_courses",
    ]
    assert all(item.status == "fulfilled" for item in plan.output_contract)


def test_actual_offerings_is_explicitly_unsupported_and_never_plans_curriculum_sql() -> None:
    query = _query(
        outputs=("course_list",),
        information_scope="actual_offerings",
        unsupported_outputs=("course_list",),
    )

    plan = build_plan(query)

    assert plan.operations == ()
    assert plan.output_contract[0].status == "unsupported"
    assert "actual_offerings_not_supported" in plan.output_contract[0].reasons


def test_coverage_gate_reports_per_output_status() -> None:
    query = _query(outputs=("course_list",))
    plan = build_plan(query)
    operation = plan.operations[0]
    packet = EvidencePacket(
        packet_id="packet",
        execution_results=(
            ToolExecutionResult(
                operation_id=operation.operation_id,
                tool_name=operation.tool_name,
                status="success",
                latency_ms=0.1,
            ),
        ),
        coverage=CoverageReport(
            components=(
                CoverageComponent(
                    operation_id=operation.operation_id,
                    tool_name=operation.tool_name,
                    kind="course_set",
                    complete=True,
                    trusted_evidence=True,
                ),
            )
        ),
    )

    decision = CoverageGate().evaluate(query, plan, packet)

    assert decision.sufficient
    assert [(item.output, item.status) for item in decision.output_statuses] == [
        ("course_list", "fulfilled")
    ]


def test_multiple_program_course_list_is_not_silently_scoped_to_first_program() -> None:
    query = _query(
        outputs=("course_list",),
        program_ids=("program-x", "program-y"),
        unsupported_outputs=("course_list",),
    )

    plan = build_plan(query)

    assert plan.operations == ()
    assert plan.output_contract[0].status == "unsupported"
