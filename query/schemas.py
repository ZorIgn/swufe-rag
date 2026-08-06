"""Versioned schemas for the sole production planning path."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AcademicStage(StrictModel):
    year: int = Field(ge=1, le=4)
    term: Literal["spring", "autumn", "summer"] | None = None


Intent = Literal[
    "course_query",
    "course_detail",
    "graduation_requirements",
    "module_requirements",
    "progress_audit",
    "policy",
    "compare_programs",
    "general",
]

StageTerm = Literal["spring", "autumn", "summer"]
CourseNature = Literal["required", "elective", "free_elective"]
RequestedOutput = Literal[
    "course_list",
    "course_detail",
    "module_requirements",
    "policy_explanation",
    "graduation_requirements",
    "progress_audit",
    "comparison",
]
InformationScope = Literal["curriculum", "actual_offerings", "policy", "unknown"]


class UnderstandingDraft(StrictModel):
    schema_version: Literal["understanding-1"] = "understanding-1"
    intent: Literal[
        "course_query",
        "course_detail",
        "graduation_requirements",
        "module_requirements",
        "progress_audit",
        "policy",
        "compare_programs",
        "general",
    ]
    requested_outputs: tuple[RequestedOutput, ...] = ()
    program_mentions: tuple[str, ...] = ()
    cohort: int | None = Field(default=None, ge=2010, le=2100)
    current_stage: AcademicStage | None = None
    target_semesters: tuple[int, ...] = ()
    course_mentions: tuple[str, ...] = ()
    course_codes: tuple[str, ...] = ()
    module_mentions: tuple[str, ...] = ()
    course_natures: tuple[Literal["required", "elective", "free_elective"], ...] = ()
    completed_courses: tuple[str, ...] = ()
    information_scope: Literal["curriculum", "actual_offerings", "policy", "unknown"] = "unknown"
    parser: Literal["deterministic", "llm"] = "deterministic"
    failure_reason: (
        Literal["invalid_json", "schema_error", "missing_constraint", "conflict", "provider_error"]
        | None
    ) = None


class ResolvedEntity(StrictModel):
    entity_type: Literal["program", "course", "module"]
    canonical_id: str
    canonical_name: str
    confidence: float = Field(ge=0.0, le=1.0)


class NormalizedQuery(StrictModel):
    schema_version: Literal["normalized-1"] = "normalized-1"
    raw_question: str = Field(min_length=1, max_length=4000)
    intent: Intent
    requested_outputs: tuple[RequestedOutput, ...] = ()
    cohort: int | None = Field(default=None, ge=2010, le=2100)
    program_ids: tuple[str, ...] = ()
    program_names: tuple[str, ...] = ()
    course_ids: tuple[str, ...] = ()
    course_codes: tuple[str, ...] = ()
    module_ids: tuple[str, ...] = ()
    semesters: tuple[int, ...] = ()
    course_natures: tuple[Literal["required", "elective", "free_elective"], ...] = ()
    completed_courses: tuple[str, ...] = ()
    information_scope: Literal["curriculum", "actual_offerings", "policy", "unknown"]
    missing_fields: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class ListCoursesArgs(StrictModel):
    cohort: int = Field(ge=2010, le=2100)
    program_id: str
    semesters: tuple[int, ...] = ()
    course_natures: tuple[Literal["required", "elective", "free_elective"], ...] = ()
    module_ids: tuple[str, ...] = ()
    course_ids: tuple[str, ...] = ()


class GetCourseDetailArgs(StrictModel):
    cohort: int = Field(ge=2010, le=2100)
    program_id: str | None = None
    course_id: str | None = None
    course_code: str | None = None


class GetGraduationRequirementsArgs(StrictModel):
    cohort: int = Field(ge=2010, le=2100)
    program_id: str


class GetModuleRequirementsArgs(StrictModel):
    cohort: int = Field(ge=2010, le=2100)
    program_id: str
    module_ids: tuple[str, ...] = ()


class AuditCompletedCoursesArgs(StrictModel):
    cohort: int = Field(ge=2010, le=2100)
    program_id: str
    completed_course_ids: tuple[str, ...] = ()
    completed_course_codes: tuple[str, ...] = ()


class ListCoursesBeforeSemesterArgs(StrictModel):
    cohort: int = Field(ge=2010, le=2100)
    program_id: str
    deadline_semester: int = Field(ge=1, le=8)
    course_natures: tuple[Literal["required", "elective", "free_elective"], ...] = ()


class ListUnavoidableCoursesArgs(StrictModel):
    cohort: int = Field(ge=2010, le=2100)
    program_id: str
    after_semester: int = Field(ge=1, le=8)


class CheckCurriculumFeasibilityArgs(StrictModel):
    cohort: int = Field(ge=2010, le=2100)
    program_id: str
    deadline_semester: int = Field(ge=1, le=8)
    completed_course_ids: tuple[str, ...] = ()


class RetrievePolicyArgs(StrictModel):
    question: str = Field(min_length=1, max_length=4000)
    cohort: int | None = Field(default=None, ge=2010, le=2100)
    program_ids: tuple[str, ...] = ()
    as_of: str | None = None
    topics: tuple[str, ...] = ()


class CompareProgramsArgs(StrictModel):
    cohort: int = Field(ge=2010, le=2100)
    program_ids: tuple[str, ...] = Field(min_length=2)
    dimensions: tuple[
        Literal[
            "graduation_min_credits",
            "module_requirements",
            "course_sets",
            "required_courses",
            "practice_requirements",
        ],
        ...,
    ] = ("graduation_min_credits", "module_requirements", "course_sets")


class ResolveSourceArgs(StrictModel):
    chunk_id: str


class OperationBase(StrictModel):
    operation_id: str
    tool_name: str
    depends_on: tuple[str, ...] = ()
    required_evidence: tuple[str, ...] = ()
    optional: bool = False


class ListCoursesOperation(OperationBase):
    type: Literal["list_courses"] = "list_courses"
    tool_name: Literal["academic.list_courses"] = "academic.list_courses"
    args: ListCoursesArgs


class GetCourseDetailOperation(OperationBase):
    type: Literal["get_course_detail"] = "get_course_detail"
    tool_name: Literal["academic.get_course"] = "academic.get_course"
    args: GetCourseDetailArgs


class GetGraduationRequirementsOperation(OperationBase):
    type: Literal["get_graduation_requirements"] = "get_graduation_requirements"
    tool_name: Literal["academic.get_requirements"] = "academic.get_requirements"
    args: GetGraduationRequirementsArgs


class GetModuleRequirementsOperation(OperationBase):
    type: Literal["get_module_requirements"] = "get_module_requirements"
    tool_name: Literal["academic.get_module_requirements"] = "academic.get_module_requirements"
    args: GetModuleRequirementsArgs


class AuditCompletedCoursesOperation(OperationBase):
    type: Literal["audit_completed_courses"] = "audit_completed_courses"
    tool_name: Literal["academic.audit_progress"] = "academic.audit_progress"
    args: AuditCompletedCoursesArgs


class ListCoursesBeforeSemesterOperation(OperationBase):
    type: Literal["list_courses_before_semester"] = "list_courses_before_semester"
    tool_name: Literal["academic.list_courses"] = "academic.list_courses"
    args: ListCoursesBeforeSemesterArgs


class ListUnavoidableCoursesOperation(OperationBase):
    type: Literal["list_unavoidable_courses"] = "list_unavoidable_courses"
    tool_name: Literal["academic.list_courses"] = "academic.list_courses"
    args: ListUnavoidableCoursesArgs


class CheckCurriculumFeasibilityOperation(OperationBase):
    type: Literal["check_curriculum_feasibility"] = "check_curriculum_feasibility"
    tool_name: Literal["academic.audit_progress"] = "academic.audit_progress"
    args: CheckCurriculumFeasibilityArgs


class RetrievePolicyOperation(OperationBase):
    type: Literal["retrieve_policy"] = "retrieve_policy"
    tool_name: Literal["policy.search"] = "policy.search"
    args: RetrievePolicyArgs


class CompareProgramsOperation(OperationBase):
    type: Literal["compare_programs"] = "compare_programs"
    tool_name: Literal["academic.compare_programs"] = "academic.compare_programs"
    args: CompareProgramsArgs


class ResolveSourceOperation(OperationBase):
    type: Literal["resolve_source"] = "resolve_source"
    tool_name: Literal["source.resolve"] = "source.resolve"
    args: ResolveSourceArgs


Operation = Annotated[
    ListCoursesOperation
    | GetCourseDetailOperation
    | GetGraduationRequirementsOperation
    | GetModuleRequirementsOperation
    | AuditCompletedCoursesOperation
    | ListCoursesBeforeSemesterOperation
    | ListUnavoidableCoursesOperation
    | CheckCurriculumFeasibilityOperation
    | RetrievePolicyOperation
    | CompareProgramsOperation
    | ResolveSourceOperation,
    Field(discriminator="type"),
]


class ExecutionPlan(StrictModel):
    plan_id: str
    query: NormalizedQuery
    operations: tuple[Operation, ...]
    rationale: tuple[str, ...] = ()


ALL_OPERATION_TYPES = frozenset(
    {
        "list_courses",
        "get_course_detail",
        "get_graduation_requirements",
        "get_module_requirements",
        "audit_completed_courses",
        "list_courses_before_semester",
        "list_unavoidable_courses",
        "check_curriculum_feasibility",
        "retrieve_policy",
        "compare_programs",
        "resolve_source",
    }
)


__all__ = [name for name in globals() if name.endswith(("Args", "Operation"))] + [
    "ALL_OPERATION_TYPES",
    "AcademicStage",
    "CourseNature",
    "ExecutionPlan",
    "InformationScope",
    "Intent",
    "NormalizedQuery",
    "Operation",
    "ResolvedEntity",
    "UnderstandingDraft",
    "StageTerm",
    "RequestedOutput",
]
