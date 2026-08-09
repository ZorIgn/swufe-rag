"""Structured academic-audit adapter over the canonical request pipeline."""

from __future__ import annotations

from agent.orchestrator import AgentRuntime
from agent.state import AgentState
from evidence.models import FinalAnswer
from query.context import RequestContext


def audit(
    runtime: AgentRuntime,
    *,
    cohort: int,
    major: str,
    completed_courses: tuple[str, ...] = (),
    session_id: str | None = None,
) -> tuple[FinalAnswer, AgentState]:
    """Resolve every completed course before the normal planner runs."""

    context = RequestContext(
        cohort=cohort,
        major=major,
        session_id=session_id,
        completed_course_mentions=completed_courses,
    )
    return runtime.ask("已修课程的学业完成度审计", context=context)


__all__ = ["audit"]
