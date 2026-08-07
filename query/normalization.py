"""Data-driven entity resolution and trusted request-scope normalization."""

from __future__ import annotations

from typing import Literal, Protocol

from query.context import RequestContext
from query.schemas import NormalizedQuery, ResolvedEntity, UnderstandingDraft


class EntityResolver(Protocol):
    def resolve_program(self, mention: str, cohort: int | None = None) -> ResolvedEntity | None: ...

    def resolve_program_candidates(
        self, mention: str, cohort: int | None = None
    ) -> tuple[ResolvedEntity, ...]: ...

    def resolve_course(
        self, mention: str, cohort: int | None = None, program_id: str | None = None
    ) -> ResolvedEntity | None: ...

    def resolve_course_candidates(
        self, mention: str, cohort: int | None = None, program_id: str | None = None
    ) -> tuple[ResolvedEntity, ...]: ...

    def resolve_module(
        self, mention: str, program_id: str | None = None
    ) -> ResolvedEntity | None: ...

    def resolve_module_candidates(
        self, mention: str, program_id: str | None = None
    ) -> tuple[ResolvedEntity, ...]: ...

    def course_mentions_in_text(
        self, text: str, cohort: int | None = None, program_id: str | None = None
    ) -> tuple[tuple[str, tuple[ResolvedEntity, ...]], ...]: ...

    def courses_in_text(
        self, text: str, cohort: int | None = None, program_id: str | None = None
    ) -> tuple[ResolvedEntity, ...]: ...

    def modules_in_text(
        self, text: str, program_id: str | None = None
    ) -> tuple[ResolvedEntity, ...]: ...

    def college_ids_for_programs(self, program_ids: tuple[str, ...]) -> tuple[str, ...]: ...


def _append_unique(values: list[ResolvedEntity], value: ResolvedEntity) -> None:
    if value.canonical_id not in {item.canonical_id for item in values}:
        values.append(value)


def _resolved_or_ambiguous(
    resolver: EntityResolver,
    kind: Literal["program", "course", "module"],
    mention: str,
    cohort: int | None,
    program_id: str | None,
) -> tuple[ResolvedEntity | None, bool]:
    if kind == "program":
        candidates = resolver.resolve_program_candidates(mention, cohort)
    elif kind == "course":
        candidates = resolver.resolve_course_candidates(mention, cohort, program_id)
    else:
        candidates = resolver.resolve_module_candidates(mention, program_id)
    return (candidates[0], False) if len(candidates) == 1 else (None, bool(candidates))


def normalize(
    draft: UnderstandingDraft,
    question: str,
    resolver: EntityResolver,
    *,
    context: RequestContext | None = None,
    inherited_program_id: str | None = None,
    inherited_cohort: int | None = None,
) -> NormalizedQuery:
    """Resolve aliases from the scoped database, never from name-specific code.

    Values supplied by :class:`RequestContext` travel as typed data and are not
    concatenated into the natural-language question. Explicit scope wins over a
    parser guess; a disagreement is surfaced as a warning rather than silently
    changing the request.
    """

    request_context = context or RequestContext()
    warnings: list[str] = []
    if request_context.cohort is not None and draft.cohort not in {None, request_context.cohort}:
        warnings.append("explicit_cohort_overrides_understanding")
    cohort = request_context.cohort or draft.cohort or inherited_cohort

    mentions = list(draft.program_mentions)
    if request_context.major:
        mentions.insert(0, request_context.major)
    programs: list[ResolvedEntity] = []
    ambiguous_programs: list[str] = []
    for mention in dict.fromkeys(mentions):
        resolved, ambiguous = _resolved_or_ambiguous(resolver, "program", mention, cohort, None)
        if resolved:
            _append_unique(programs, resolved)
        elif ambiguous:
            ambiguous_programs.append(mention)
        else:
            warnings.append(f"program_unmatched:{mention}")
    if not programs and inherited_program_id:
        resolved = resolver.resolve_program(inherited_program_id, cohort)
        if resolved:
            _append_unique(programs, resolved)
    primary_program = programs[0].canonical_id if len(programs) == 1 else None

    college_ids = list(
        resolver.college_ids_for_programs(tuple(item.canonical_id for item in programs))
    )
    if request_context.college:
        if request_context.college not in college_ids:
            college_ids.insert(0, request_context.college)
        derived = set(
            resolver.college_ids_for_programs(tuple(item.canonical_id for item in programs))
        )
        if derived and request_context.college not in derived:
            warnings.append("explicit_college_conflicts_with_program")

    courses: list[ResolvedEntity] = []
    ambiguous_course_mentions: list[str] = []
    for mention in dict.fromkeys((*draft.course_mentions, *draft.course_codes)):
        resolved, ambiguous = _resolved_or_ambiguous(
            resolver, "course", mention, cohort, primary_program
        )
        if resolved:
            _append_unique(courses, resolved)
        elif ambiguous:
            ambiguous_course_mentions.append(mention)
        else:
            warnings.append(f"course_unmatched:{mention}")

    modules: list[ResolvedEntity] = []
    ambiguous_module_mentions: list[str] = []
    for mention in dict.fromkeys(draft.module_mentions):
        resolved, ambiguous = _resolved_or_ambiguous(
            resolver, "module", mention, cohort, primary_program
        )
        if resolved:
            _append_unique(modules, resolved)
        elif ambiguous:
            ambiguous_module_mentions.append(mention)
        else:
            warnings.append(f"module_unmatched:{mention}")
    if primary_program:
        for entity in resolver.modules_in_text(question, primary_program):
            _append_unique(modules, entity)

    completed_mentions = list(request_context.completed_course_mentions)
    completed_mentions.extend(draft.completed_courses)
    # For a progress or feasibility question, resolve aliases from the database
    # after program/cohort scope is known. The parser never contains a course-name
    # special case.
    if draft.intent in {"progress_audit", "curriculum_feasibility"}:
        completed_mentions.extend(draft.course_codes)
    completed: list[ResolvedEntity] = []
    unmatched_completed: list[str] = []
    ambiguous_completed: list[str] = []
    for mention in dict.fromkeys(item.strip() for item in completed_mentions if item.strip()):
        resolved, ambiguous = _resolved_or_ambiguous(
            resolver, "course", mention, cohort, primary_program
        )
        if resolved:
            _append_unique(completed, resolved)
        elif ambiguous:
            ambiguous_completed.append(mention)
        else:
            unmatched_completed.append(mention)
    if draft.intent in {"progress_audit", "curriculum_feasibility"} and primary_program:
        for alias, candidates in resolver.course_mentions_in_text(
            question, cohort, primary_program
        ):
            if len(candidates) == 1:
                _append_unique(completed, candidates[0])
            elif alias not in ambiguous_completed:
                ambiguous_completed.append(alias)

    intent = draft.intent
    if (
        intent in {"course_query", "course_detail"}
        and not programs
        and not draft.target_semesters
        and not draft.course_natures
        and not draft.course_codes
    ):
        intent = "policy"
    structured = {
        "course_query",
        "course_detail",
        "graduation_requirements",
        "module_requirements",
        "progress_audit",
        "compare_programs",
        "course_planning",
        "curriculum_feasibility",
    }
    missing: list[str] = []
    if intent in structured and cohort is None:
        missing.append("cohort")
    if intent in structured and not programs:
        missing.append("program")
    if intent == "compare_programs" and len(programs) < 2:
        missing.append("at_least_two_programs")
    if intent in {"progress_audit", "curriculum_feasibility"} and not completed:
        missing.append("completed_courses")
    if unmatched_completed:
        missing.append("completed_courses")
    if ambiguous_completed:
        missing.append("completed_courses")
    if ambiguous_programs:
        missing.append("program")
    if intent in {"course_planning", "curriculum_feasibility"} and draft.deadline_semester is None:
        missing.append("deadline_semester")
    if intent == "course_detail" and ambiguous_course_mentions:
        missing.append("course")
    if intent == "module_requirements" and ambiguous_module_mentions:
        missing.append("module")
    if draft.information_scope == "actual_offerings":
        warnings.append("当前数据描述培养方案，而非实时开课或名额；实际选课以教务系统为准。")

    return NormalizedQuery(
        raw_question=question,
        intent=intent,
        requested_outputs=draft.requested_outputs,
        cohort=cohort,
        program_ids=tuple(item.canonical_id for item in programs),
        program_names=tuple(item.canonical_name for item in programs),
        college_ids=tuple(dict.fromkeys(college_ids)),
        policy_as_of=request_context.as_of,
        course_ids=tuple(item.canonical_id for item in courses),
        course_codes=tuple(draft.course_codes),
        module_ids=tuple(item.canonical_id for item in modules),
        semesters=draft.target_semesters,
        course_natures=draft.course_natures,
        completed_course_ids=tuple(item.canonical_id for item in completed),
        unmatched_completed_courses=tuple(dict.fromkeys(unmatched_completed)),
        ambiguous_completed_courses=tuple(dict.fromkeys(ambiguous_completed)),
        deadline_semester=draft.deadline_semester,
        comparison_dimensions=draft.comparison_dimensions,
        information_scope=draft.information_scope,
        missing_fields=tuple(dict.fromkeys(missing)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


__all__ = ["EntityResolver", "normalize"]
