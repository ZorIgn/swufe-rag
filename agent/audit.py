"""Structured academic-audit request adapter over the canonical tool pipeline."""

from __future__ import annotations

from uuid import uuid4

from agent.orchestrator import AgentRuntime
from agent.state import AgentState, AgentStatus
from evidence.models import FinalAnswer
from generation.renderer import render
from query.schemas import NormalizedQuery


def audit(
    runtime: AgentRuntime,
    *,
    cohort: int,
    major: str,
    completed_courses: tuple[str, ...] = (),
    session_id: str | None = None,
) -> tuple[FinalAnswer, AgentState]:
    """Use the exact planner/executor/validator used by natural-language `/ask`."""
    resolved = runtime.repository.resolve_program(major, cohort)
    state = AgentState(request_id=uuid4().hex, session_id=session_id, raw_question=f"{cohort}级{major}学业审计")
    if resolved is None:
        answer = FinalAnswer(answer_md="请提供数据集中存在的入学年级和专业。", claims=(), citations=(), clarification="专业未解析")
        state.status = AgentStatus.FINISH
        state.answer = answer
        return answer, state
    query = NormalizedQuery(
        raw_question=state.raw_question, intent="progress_audit", cohort=cohort,
        program_ids=(resolved.canonical_id,), program_names=(resolved.canonical_name,),
        completed_courses=completed_courses, information_scope="curriculum",
    )
    state.normalized_query = query
    state.status = AgentStatus.PLAN
    plan = runtime._deps.planner(query)  # noqa: SLF001 - same canonical package adapter
    state.plan = plan
    state.status = AgentStatus.EXECUTE
    packet = runtime._deps.executor.execute(plan)  # noqa: SLF001
    state.evidence = packet
    state.status = AgentStatus.SYNTHESIZE
    answer = runtime._deps.synthesizer.synthesize(query, packet)  # noqa: SLF001
    state.status = AgentStatus.VALIDATE
    state.answer = render(runtime._deps.validator.validate(answer, packet))  # noqa: SLF001
    state.status = AgentStatus.FINISH
    return state.answer, state


__all__ = ["audit"]
