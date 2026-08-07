"""Pure construction of bounded, typed tool DAGs."""

from __future__ import annotations

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
    RetrievePolicyArgs,
    RetrievePolicyOperation,
)


def _operation_id(query: NormalizedQuery, operation_type: str) -> str:
    return stable_id("op", query.raw_question, operation_type)


def _policy_operation(
    query: NormalizedQuery, *, operation_id: str, depends_on: tuple[str, ...] = ()
) -> RetrievePolicyOperation:
    return RetrievePolicyOperation(
        operation_id=operation_id,
        depends_on=depends_on,
        args=RetrievePolicyArgs(
            question=query.raw_question,
            cohort=query.cohort,
            program_ids=query.program_ids,
            college_ids=query.college_ids,
            as_of=query.policy_as_of,
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
    audit_id = _operation_id(query, "audit_completed_courses")
    before_id = _operation_id(query, "list_courses_before_semester")
    unavoidable_id = _operation_id(query, "list_unavoidable_courses")
    feasibility_id = _operation_id(query, "check_curriculum_feasibility")
    return (
        GetGraduationRequirementsOperation(
            operation_id=requirement_id,
            args=GetGraduationRequirementsArgs(cohort=query.cohort, program_id=program_id),
        ),
        AuditCompletedCoursesOperation(
            operation_id=audit_id,
            depends_on=(requirement_id,),
            args=AuditCompletedCoursesArgs(
                cohort=query.cohort,
                program_id=program_id,
                completed_course_ids=query.completed_course_ids,
            ),
        ),
        ListCoursesBeforeSemesterOperation(
            operation_id=before_id,
            depends_on=(audit_id,),
            args=ListCoursesBeforeSemesterArgs(
                cohort=query.cohort,
                program_id=program_id,
                deadline_semester=query.deadline_semester,
                course_natures=query.course_natures,
            ),
        ),
        ListUnavoidableCoursesOperation(
            operation_id=unavoidable_id,
            depends_on=(before_id,),
            args=ListUnavoidableCoursesArgs(
                cohort=query.cohort,
                program_id=program_id,
                after_semester=query.deadline_semester,
            ),
        ),
        CheckCurriculumFeasibilityOperation(
            operation_id=feasibility_id,
            depends_on=(audit_id, unavoidable_id),
            args=CheckCurriculumFeasibilityArgs(
                cohort=query.cohort,
                program_id=program_id,
                deadline_semester=query.deadline_semester,
                completed_course_ids=query.completed_course_ids,
            ),
        ),
    )


def build_plan(query: NormalizedQuery) -> ExecutionPlan:
    """Return a DAG-ready plan. No operation accepts free-form SQL or code."""

    if query.missing_fields or query.intent == "general":
        return ExecutionPlan(
            plan_id=stable_id("plan", query.raw_question), query=query, operations=()
        )
    operations: tuple[Operation, ...]
    composite = _composite_operations(query)
    if composite:
        operations = composite
    elif query.intent == "policy":
        operations = (
            _policy_operation(query, operation_id=_operation_id(query, "retrieve_policy")),
        )
    elif query.intent == "compare_programs":
        assert query.cohort is not None
        dimensions = query.comparison_dimensions or (
            "graduation_min_credits",
            "module_requirements",
            "course_sets",
        )
        operations = (
            CompareProgramsOperation(
                operation_id=_operation_id(query, "compare_programs"),
                args=CompareProgramsArgs(
                    cohort=query.cohort, program_ids=query.program_ids, dimensions=dimensions
                ),
            ),
        )
    else:
        assert query.cohort is not None and query.program_ids
        program_id = query.program_ids[0]
        if query.intent == "graduation_requirements":
            operations = (
                GetGraduationRequirementsOperation(
                    operation_id=_operation_id(query, "get_graduation_requirements"),
                    args=GetGraduationRequirementsArgs(cohort=query.cohort, program_id=program_id),
                ),
            )
        elif query.intent == "module_requirements":
            operations = (
                GetModuleRequirementsOperation(
                    operation_id=_operation_id(query, "get_module_requirements"),
                    args=GetModuleRequirementsArgs(
                        cohort=query.cohort, program_id=program_id, module_ids=query.module_ids
                    ),
                ),
            )
        elif query.intent == "progress_audit":
            requirements_id = _operation_id(query, "get_graduation_requirements")
            operations = (
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
        elif query.intent in {"course_planning", "curriculum_feasibility"}:
            operations = _planning_operations(query)
        elif query.intent == "course_detail" and (query.course_ids or query.course_codes):
            operations = (
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
        else:
            sql_operation = ListCoursesOperation(
                operation_id=_operation_id(query, "list_courses"),
                args=ListCoursesArgs(
                    cohort=query.cohort,
                    program_id=program_id,
                    semesters=query.semesters,
                    course_natures=query.course_natures,
                    module_ids=query.module_ids,
                    course_ids=query.course_ids,
                ),
            )
            operations = (
                (
                    sql_operation,
                    _policy_operation(query, operation_id=_operation_id(query, "retrieve_policy")),
                )
                if query.information_scope == "policy"
                else (sql_operation,)
            )
    return ExecutionPlan(
        plan_id=stable_id(
            "plan", query.raw_question, *[value.operation_id for value in operations]
        ),
        query=query,
        operations=operations,
    )


__all__ = ["build_plan"]
