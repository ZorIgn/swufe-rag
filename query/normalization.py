"""Data-driven entity resolution and scope normalization."""

from __future__ import annotations

from typing import Protocol

from query.schemas import NormalizedQuery, ResolvedEntity, UnderstandingDraft


class EntityResolver(Protocol):
    def resolve_program(self, mention: str, cohort: int | None = None) -> ResolvedEntity | None: ...

    def resolve_course(
        self, mention: str, cohort: int | None = None, program_id: str | None = None
    ) -> ResolvedEntity | None: ...

    def resolve_module(self, mention: str, program_id: str | None = None) -> ResolvedEntity | None: ...


def normalize(
    draft: UnderstandingDraft,
    question: str,
    resolver: EntityResolver,
    *,
    inherited_program_id: str | None = None,
    inherited_cohort: int | None = None,
) -> NormalizedQuery:
    """Resolve aliases from the repository, never from hand-written program tuples."""
    cohort = draft.cohort or inherited_cohort
    programs: list[ResolvedEntity] = []
    for mention in draft.program_mentions:
        resolved = resolver.resolve_program(mention, cohort)
        if resolved and resolved.canonical_id not in {item.canonical_id for item in programs}:
            programs.append(resolved)
    if not programs and inherited_program_id:
        resolved = resolver.resolve_program(inherited_program_id, cohort)
        if resolved:
            programs.append(resolved)
    primary = programs[0].canonical_id if len(programs) == 1 else None
    courses: list[ResolvedEntity] = []
    for mention in (*draft.course_mentions, *draft.course_codes):
        resolved = resolver.resolve_course(mention, cohort, primary)
        if resolved and resolved.canonical_id not in {item.canonical_id for item in courses}:
            courses.append(resolved)
    modules: list[ResolvedEntity] = []
    for mention in draft.module_mentions:
        resolved = resolver.resolve_module(mention, primary)
        if resolved and resolved.canonical_id not in {item.canonical_id for item in modules}:
            modules.append(resolved)
    intent = draft.intent
    if intent in {"course_query", "course_detail"} and not programs and not draft.target_semesters and not draft.course_natures and not draft.course_codes:
        intent = "policy"
    structured = {
        "course_query", "course_detail", "graduation_requirements", "module_requirements",
        "progress_audit", "compare_programs",
    }
    missing: list[str] = []
    if intent in structured and cohort is None:
        missing.append("cohort")
    if intent in structured and not programs:
        missing.append("program")
    if draft.intent == "compare_programs" and len(programs) < 2:
        missing.append("at_least_two_programs")
    warnings: list[str] = []
    if draft.information_scope == "actual_offerings":
        warnings.append("当前数据描述培养方案，而非实时开课或名额；实际选课以教务系统为准。")
    if draft.intent == "progress_audit" and not draft.completed_courses:
        warnings.append("未提供已修课程，无法完成个性化学分核算。")
    return NormalizedQuery(
        raw_question=question,
        intent=intent,
        requested_outputs=draft.requested_outputs,
        cohort=cohort,
        program_ids=tuple(item.canonical_id for item in programs),
        program_names=tuple(item.canonical_name for item in programs),
        course_ids=tuple(item.canonical_id for item in courses),
        course_codes=tuple(draft.course_codes),
        module_ids=tuple(item.canonical_id for item in modules),
        semesters=draft.target_semesters,
        course_natures=draft.course_natures,
        completed_courses=draft.completed_courses,
        information_scope=draft.information_scope,
        missing_fields=tuple(missing),
        warnings=tuple(warnings),
    )


__all__ = ["EntityResolver", "normalize"]
