"""Pure construction of bounded, typed tool DAGs."""

from __future__ import annotations

import re

from evidence.provenance import stable_id
from query.schemas import (
    AuditCompletedCoursesArgs,
    AuditCompletedCoursesOperation,
    CheckCurriculumFeasibilityArgs,
    CheckCurriculumFeasibilityOperation,
    CompareProgramsArgs,
    CompareProgramsOperation,
    ExecutionPlan,
    GetCourseDetailArgs,
    GetCourseDetailOperation,
    GetGraduationRequirementsArgs,
    GetGraduationRequirementsOperation,
    GetModuleRequirementsArgs,
    GetModuleRequirementsOperation,
    ListCoursesArgs,
    ListCoursesBeforeSemesterArgs,
    ListCoursesBeforeSemesterOperation,
    ListCoursesOperation,
    ListUnavoidableCoursesArgs,
    ListUnavoidableCoursesOperation,
    NormalizedQuery,
    Operation,
    OutputContract,
    RequestedOutput,
    RetrievePolicyArgs,
    RetrievePolicyOperation,
)


def _operation_id(query: NormalizedQuery, operation_type: str) -> str:
    return stable_id("op", query.raw_question, operation_type)


_POLICY_MARKERS = (
    "政策",
    "规定",
    "办法",
    "条件",
    "免修",
    "推免",
    "保研",
    "转专业",
    "学籍",
    "考试",
)


# Capabilities are deliberately named after coverage contracts, not after one
# particular tool.  This lets a composite request share a typed operation (for
# example, the curriculum catalog query used by both ``course_list`` and
# ``course_plan``) without silently dropping an output.
_OUTPUT_CAPABILITIES: dict[RequestedOutput, tuple[str, ...]] = {
    "course_list": ("course_set",),
    "course_detail": ("course_set",),
    "module_requirements": ("requirements",),
    "policy_explanation": ("policy",),
    "graduation_requirements": ("requirements",),
    "progress_audit": ("progress_audit",),
    "comparison": ("comparison",),
    "course_plan": ("requirements", "progress_audit", "course_set"),
    "feasibility": ("requirements", "progress_audit", "course_set"),
}

_OPERATION_CAPABILITIES: dict[str, frozenset[str]] = {
    "list_courses": frozenset({"course_set"}),
    # The planning catalog uses a distinct operation id but the same typed
    # capability as an ordinary course list.
    "list_courses_before_semester": frozenset({"course_set", "curriculum_plan"}),
    "list_unavoidable_courses": frozenset({"course_set", "curriculum_plan"}),
    "get_course_detail": frozenset({"course_set"}),
    "get_graduation_requirements": frozenset({"requirements"}),
    "get_module_requirements": frozenset({"requirements"}),
    "audit_completed_courses": frozenset({"progress_audit"}),
    "check_curriculum_feasibility": frozenset({"course_set"}),
    "retrieve_policy": frozenset({"policy"}),
    "compare_programs": frozenset({"comparison"}),
}

# Producer operations are explicit per output.  Coverage validation uses these
# exact IDs rather than accepting any component with the same broad kind.
_OUTPUT_OPERATION_TYPES: dict[RequestedOutput, frozenset[str]] = {
    "course_list": frozenset({"list_courses"}),
    "course_detail": frozenset({"get_course_detail"}),
    "module_requirements": frozenset({"get_module_requirements"}),
    "policy_explanation": frozenset({"retrieve_policy"}),
    "graduation_requirements": frozenset({"get_graduation_requirements"}),
    "progress_audit": frozenset(
        {"get_graduation_requirements", "audit_completed_courses"}
    ),
    "comparison": frozenset({"compare_programs"}),
    "course_plan": frozenset(
        {
            "get_graduation_requirements",
            "list_courses",
            "audit_completed_courses",
            "list_courses_before_semester",
            "list_unavoidable_courses",
            "check_curriculum_feasibility",
        }
    ),
    "feasibility": frozenset(
        {
            "get_graduation_requirements",
            "list_courses",
            "audit_completed_courses",
            "list_courses_before_semester",
            "list_unavoidable_courses",
            "check_curriculum_feasibility",
        }
    ),
}

_OUTPUT_PRIORITY: dict[RequestedOutput, int] = {
    "module_requirements": 10,
    "graduation_requirements": 20,
    "progress_audit": 30,
    "course_list": 40,
    "course_detail": 45,
    "comparison": 50,
    "course_plan": 60,
    "feasibility": 60,
    "policy_explanation": 70,
}


def _policy_question(question: str) -> str:
    """Keep structured clauses from diluting a composite policy search."""

    clauses = [
        re.sub(r"^(?:另外|此外|同时|以及|并且|再问)+", "", value).strip(" ：:,，")
        for value in re.split(r"[？?。；;，,\n]+|另外|此外", question)
    ]
    policy_clauses = [
        clause for clause in clauses if clause and any(marker in clause for marker in _POLICY_MARKERS)
    ]
    normalized: list[str] = []
    for clause in policy_clauses:
        if (
            "免修" in clause
            and "条件" not in clause
            and any(marker in clause for marker in ("规定", "政策", "办法"))
        ):
            subject = clause.split("免修", 1)[0].strip()
            if subject:
                clause = f"{subject}达到什么条件可以免修"
        normalized.append(clause)
    return "；".join(normalized) or question


def _policy_operation(
    query: NormalizedQuery, *, operation_id: str, depends_on: tuple[str, ...] = ()
) -> RetrievePolicyOperation:
    return RetrievePolicyOperation(
        operation_id=operation_id,
        depends_on=depends_on,
        args=RetrievePolicyArgs(
            question=_policy_question(query.raw_question),
            cohort=query.cohort,
            program_ids=query.program_ids,
            college_ids=query.college_ids,
            as_of=query.policy_as_of,
            topics=query.policy_topics,
        ),
    )


def _composite_operations(query: NormalizedQuery) -> tuple[Operation, ...]:
    """Plan the structured-plus-policy composite requested by the user."""

    required = {"module_requirements", "course_list", "policy_explanation"}
    if not required.issubset(query.requested_outputs):
        return ()
    if query.cohort is None or not query.program_ids:
        return ()
    program_id = query.program_ids[0]
    return (
        GetModuleRequirementsOperation(
            operation_id=_operation_id(query, "get_module_requirements"),
            args=GetModuleRequirementsArgs(
                cohort=query.cohort, program_id=program_id, module_ids=query.module_ids
            ),
        ),
        ListCoursesOperation(
            operation_id=_operation_id(query, "list_courses"),
            args=ListCoursesArgs(
                cohort=query.cohort,
                program_id=program_id,
                semesters=query.semesters,
                course_natures=query.course_natures,
                module_ids=query.module_ids,
                course_ids=query.course_ids,
            ),
        ),
        _policy_operation(query, operation_id=_operation_id(query, "retrieve_policy")),
    )


def _planning_operations(query: NormalizedQuery) -> tuple[Operation, ...]:
    """Use a small, meaningful DAG for planning and feasibility questions."""

    assert query.cohort is not None and query.program_ids and query.deadline_semester is not None
    program_id = query.program_ids[0]
    requirement_id = _operation_id(query, "get_graduation_requirements")
    catalog_id = _operation_id(query, "list_curriculum_courses")
    audit_id = _operation_id(query, "audit_completed_courses")
    before_id = _operation_id(query, "list_courses_before_semester")
    unavoidable_id = _operation_id(query, "list_unavoidable_courses")
    feasibility_id = _operation_id(query, "check_curriculum_feasibility")
    return (
        GetGraduationRequirementsOperation(
            operation_id=requirement_id,
            args=GetGraduationRequirementsArgs(cohort=query.cohort, program_id=program_id),
        ),
        ListCoursesOperation(
            operation_id=catalog_id,
            args=ListCoursesArgs(cohort=query.cohort, program_id=program_id),
        ),
        AuditCompletedCoursesOperation(
            operation_id=audit_id,
            depends_on=(requirement_id, catalog_id),
            args=AuditCompletedCoursesArgs(
                cohort=query.cohort,
                program_id=program_id,
                completed_course_ids=query.completed_course_ids,
            ),
        ),
        ListCoursesBeforeSemesterOperation(
            operation_id=before_id,
            depends_on=(catalog_id, audit_id),
            args=ListCoursesBeforeSemesterArgs(
                cohort=query.cohort,
                program_id=program_id,
                deadline_semester=query.deadline_semester,
                course_natures=query.course_natures,
            ),
        ),
        ListUnavoidableCoursesOperation(
            operation_id=unavoidable_id,
            depends_on=(catalog_id, audit_id),
            args=ListUnavoidableCoursesArgs(
                cohort=query.cohort,
                program_id=program_id,
                after_semester=query.deadline_semester,
            ),
        ),
        CheckCurriculumFeasibilityOperation(
            operation_id=feasibility_id,
            depends_on=(audit_id, before_id, unavoidable_id),
            args=CheckCurriculumFeasibilityArgs(
                cohort=query.cohort,
                program_id=program_id,
                deadline_semester=query.deadline_semester,
                completed_course_ids=query.completed_course_ids,
            ),
        ),
    )


def _operations_for_output(query: NormalizedQuery, output: RequestedOutput) -> tuple[Operation, ...]:
    """Build the smallest typed operation set for one requested output."""

    if output == "policy_explanation":
        return (
            _policy_operation(query, operation_id=_operation_id(query, "retrieve_policy")),
        )
    if query.cohort is None or not query.program_ids:
        return ()
    program_id = query.program_ids[0]
    if output == "course_list":
        return (
            ListCoursesOperation(
                operation_id=_operation_id(query, "list_courses"),
                args=ListCoursesArgs(
                    cohort=query.cohort,
                    program_id=program_id,
                    semesters=query.semesters,
                    course_natures=query.course_natures,
                    module_ids=query.module_ids,
                    course_ids=query.course_ids,
                ),
            ),
        )
    if output == "course_detail":
        if not query.course_ids and not query.course_codes:
            return ()
        return (
            GetCourseDetailOperation(
                operation_id=_operation_id(query, "get_course_detail"),
                args=GetCourseDetailArgs(
                    cohort=query.cohort,
                    program_id=program_id,
                    course_id=query.course_ids[0] if query.course_ids else None,
                    course_code=query.course_codes[0] if query.course_codes else None,
                ),
            ),
        )
    if output == "graduation_requirements":
        return (
            GetGraduationRequirementsOperation(
                operation_id=_operation_id(query, "get_graduation_requirements"),
                args=GetGraduationRequirementsArgs(cohort=query.cohort, program_id=program_id),
            ),
        )
    if output == "module_requirements":
        return (
            GetModuleRequirementsOperation(
                operation_id=_operation_id(query, "get_module_requirements"),
                args=GetModuleRequirementsArgs(
                    cohort=query.cohort,
                    program_id=program_id,
                    module_ids=query.module_ids,
                ),
            ),
        )
    if output == "progress_audit":
        requirements_id = _operation_id(query, "get_graduation_requirements")
        return (
            GetGraduationRequirementsOperation(
                operation_id=requirements_id,
                args=GetGraduationRequirementsArgs(cohort=query.cohort, program_id=program_id),
            ),
            AuditCompletedCoursesOperation(
                operation_id=_operation_id(query, "audit_completed_courses"),
                depends_on=(requirements_id,),
                args=AuditCompletedCoursesArgs(
                    cohort=query.cohort,
                    program_id=program_id,
                    completed_course_ids=query.completed_course_ids,
                ),
            ),
        )
    if output == "comparison":
        if len(query.program_ids) < 2:
            return ()
        dimensions = query.comparison_dimensions or ("module_requirements", "course_sets")
        return (
            CompareProgramsOperation(
                operation_id=_operation_id(query, "compare_programs"),
                args=CompareProgramsArgs(
                    cohort=query.cohort,
                    program_ids=query.program_ids,
                    dimensions=dimensions,
                ),
            ),
        )
    return ()


def _dedupe_operations(operations: tuple[Operation, ...]) -> tuple[Operation, ...]:
    seen: set[str] = set()
    values: list[Operation] = []
    for operation in operations:
        if operation.operation_id in seen:
            continue
        seen.add(operation.operation_id)
        values.append(operation)
    return tuple(values)


def _missing_fields_for_output(
    query: NormalizedQuery, output: RequestedOutput
) -> tuple[str, ...]:
    """Return only the fields needed by one requested output.

    The normalized query retains a union of all missing fields for clarification
    UX.  Planning must be more precise: a missing program for a course output
    must not suppress an independently answerable global policy output.
    """

    if output == "policy_explanation":
        return ()

    missing: list[str] = []
    if query.cohort is None:
        missing.append("cohort")
    if not query.program_ids:
        missing.append("program")

    if output == "comparison" and len(query.program_ids) < 2:
        missing.append("at_least_two_programs")
    if output in {
        "course_list",
        "course_detail",
        "module_requirements",
        "graduation_requirements",
        "progress_audit",
        "course_plan",
        "feasibility",
    } and len(query.program_ids) > 1:
        missing.append("single_program_scope")
    if output == "course_detail" and not query.course_ids and not query.course_codes:
        missing.append("course")
    if output == "module_requirements" and "module" in query.missing_fields:
        missing.append("module")
    if output in {"progress_audit", "feasibility"} and (
        "completed_courses" in query.missing_fields
        or query.unmatched_completed_courses
        or query.ambiguous_completed_courses
    ):
        missing.append("completed_courses")
    if output in {"course_plan", "feasibility"} and query.deadline_semester is None:
        missing.append("deadline_semester")
    return tuple(dict.fromkeys(missing))


def _output_contracts(
    query: NormalizedQuery,
    operations: tuple[Operation, ...],
) -> tuple[OutputContract, ...]:
    """Create per-output producer contracts before execution starts."""

    planned_types = {operation.type for operation in operations}
    contracts: list[OutputContract] = []
    for output in query.requested_outputs:
        capabilities = _OUTPUT_CAPABILITIES.get(output, ())
        required_types = _OUTPUT_OPERATION_TYPES.get(output, frozenset())
        output_operations = tuple(
            operation.operation_id
            for operation in operations
            if operation.type in required_types
        )
        if output in query.unsupported_outputs:
            reason = (
                "actual_offerings_not_supported"
                if query.information_scope == "actual_offerings"
                else "output_scope_or_entity_unsupported"
            )
            contracts.append(
                OutputContract(
                    output=output,
                    capabilities=capabilities,
                    status="unsupported",
                    operation_ids=(),
                    reasons=(reason,),
                )
            )
            continue

        missing = _missing_fields_for_output(query, output)
        if missing:
            contracts.append(
                OutputContract(
                    output=output,
                    capabilities=capabilities,
                    status="missing_data",
                    operation_ids=(),
                    reasons=tuple(f"missing:{field}" for field in missing),
                )
            )
            continue

        fulfilled = bool(required_types) and required_types.issubset(planned_types)
        contracts.append(
            OutputContract(
                output=output,
                capabilities=capabilities,
                status="fulfilled" if fulfilled else "missing_data",
                operation_ids=output_operations,
                reasons=() if fulfilled else ("producer_operation_not_planned",),
            )
        )
    return tuple(contracts)

def _empty_plan(query: NormalizedQuery) -> ExecutionPlan:
    return ExecutionPlan(
        plan_id=stable_id("plan", query.raw_question),
        query=query,
        operations=(),
        output_contract=_output_contracts(query, ()),
    )


def build_plan(query: NormalizedQuery) -> ExecutionPlan:
    """Return a DAG-ready plan. No operation accepts free-form SQL or code."""

    if query.intent == "general":
        return _empty_plan(query)

    requested = tuple(
        sorted(
            dict.fromkeys(query.requested_outputs),
            key=lambda output: _OUTPUT_PRIORITY.get(output, 100),
        )
    )
    planning_requested = any(
        output in {"course_plan", "feasibility"}
        and output not in query.unsupported_outputs
        and not _missing_fields_for_output(query, output)
        for output in requested
    )
    operation_values: list[Operation] = []
    if planning_requested:
        operation_values.extend(_planning_operations(query))

    planning_reuses = {
        "course_list",
        "graduation_requirements",
        "progress_audit",
    }
    for output in requested:
        if output in query.unsupported_outputs:
            continue
        if _missing_fields_for_output(query, output):
            continue
        if output in {"course_plan", "feasibility"}:
            continue
        if planning_requested and output in planning_reuses:
            continue
        operation_values.extend(_operations_for_output(query, output))

    operations = _dedupe_operations(tuple(operation_values))
    return ExecutionPlan(
        plan_id=stable_id(
            "plan", query.raw_question, *[value.operation_id for value in operations]
        ),
        query=query,
        operations=operations,
        output_contract=_output_contracts(query, operations),
    )


__all__ = ["build_plan"]
