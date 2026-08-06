"""Pure construction of ordered typed tool plans."""

from __future__ import annotations

from evidence.provenance import stable_id
from query.schemas import (
    AuditCompletedCoursesArgs,
    AuditCompletedCoursesOperation,
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
    ListCoursesOperation,
    NormalizedQuery,
    Operation,
    RetrievePolicyArgs,
    RetrievePolicyOperation,
)


def _operation_id(query: NormalizedQuery, operation_type: str) -> str:
    return stable_id("op", query.raw_question, operation_type)


def _composite_operations(query: NormalizedQuery) -> tuple[Operation, ...]:
    """Plan the generic structured-plus-policy composite requested by the user."""

    required = {"module_requirements", "course_list", "policy_explanation"}
    if not required.issubset(query.requested_outputs):
        return ()
    if query.cohort is None or not query.program_ids:
        return ()
    program_id = query.program_ids[0]
    return (
        GetModuleRequirementsOperation(
            operation_id=_operation_id(query, "get_module_requirements"),
            args=GetModuleRequirementsArgs(cohort=query.cohort, program_id=program_id, module_ids=query.module_ids),
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
        RetrievePolicyOperation(
            operation_id=_operation_id(query, "retrieve_policy"),
            args=RetrievePolicyArgs(question=query.raw_question, cohort=query.cohort, program_ids=query.program_ids),
        ),
    )

def build_plan(query: NormalizedQuery) -> ExecutionPlan:
    """Return a DAG-ready plan. Independent tools have no dependencies."""
    if query.missing_fields or query.intent == "general":
        return ExecutionPlan(plan_id=stable_id("plan", query.raw_question), query=query, operations=())
    operations: tuple[Operation, ...]
    composite = _composite_operations(query)
    if composite:
        operations = composite
    elif query.intent == "policy":
        operations = (RetrievePolicyOperation(
            operation_id=_operation_id(query, "retrieve_policy"),
            args=RetrievePolicyArgs(question=query.raw_question, cohort=query.cohort, program_ids=query.program_ids),
        ),)
    elif query.intent == "compare_programs":
        assert query.cohort is not None
        operations = (CompareProgramsOperation(
            operation_id=_operation_id(query, "compare_programs"),
            args=CompareProgramsArgs(cohort=query.cohort, program_ids=query.program_ids),
        ),)
    else:
        assert query.cohort is not None and query.program_ids
        program_id = query.program_ids[0]
        if query.intent == "graduation_requirements":
            operations = (GetGraduationRequirementsOperation(
                operation_id=_operation_id(query, "get_graduation_requirements"),
                args=GetGraduationRequirementsArgs(cohort=query.cohort, program_id=program_id),
            ),)
        elif query.intent == "module_requirements":
            operations = (GetModuleRequirementsOperation(
                operation_id=_operation_id(query, "get_module_requirements"),
                args=GetModuleRequirementsArgs(cohort=query.cohort, program_id=program_id, module_ids=query.module_ids),
            ),)
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
                        completed_course_codes=query.completed_courses,
                    ),
                ),
            )
        elif query.intent == "course_detail" and (query.course_ids or query.course_codes):
            operations = (GetCourseDetailOperation(
                operation_id=_operation_id(query, "get_course_detail"),
                args=GetCourseDetailArgs(
                    cohort=query.cohort,
                    program_id=program_id,
                    course_id=query.course_ids[0] if query.course_ids else None,
                    course_code=query.course_codes[0] if query.course_codes else None,
                ),
            ),)
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
            if query.information_scope == "policy":
                operations = (sql_operation, RetrievePolicyOperation(
                    operation_id=_operation_id(query, "retrieve_policy"),
                    args=RetrievePolicyArgs(question=query.raw_question, cohort=query.cohort, program_ids=query.program_ids),
                ))
            else:
                operations = (sql_operation,)
    return ExecutionPlan(
        plan_id=stable_id("plan", query.raw_question, *[value.operation_id for value in operations]),
        query=query,
        operations=operations,
    )


__all__ = ["build_plan"]
