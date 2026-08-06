"""The only production Evidence-Grounded Academic Agent runtime."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from academic.database import AcademicRepository
from agent.interfaces import (
    AnswerSynthesizer,
    AnswerValidator,
    ExecutionPlanner,
    QueryNormalizer,
    QuestionUnderstanding,
    SessionStore,
    ToolExecutor,
)
from agent.registry import ToolRegistry
from agent.state import AgentState, AgentStatus
from agent.tracing import InMemoryTracer
from evidence.models import Coverage, EvidencePacket, FinalAnswer
from evidence.provenance import stable_id
from generation.renderer import render
from generation.synthesizer import LLMClaimSynthesizer
from generation.synthesizer import StructuredModel as SynthesisModel
from query.schemas import ExecutionPlan, RetrievePolicyArgs, RetrievePolicyOperation
from query.understanding import QuestionUnderstanding as StructuredQuestionUnderstanding


@dataclass(frozen=True)
class RuntimeDependencies:
    understanding: QuestionUnderstanding
    normalizer: QueryNormalizer
    planner: ExecutionPlanner
    executor: ToolExecutor
    synthesizer: AnswerSynthesizer
    validator: AnswerValidator
    sessions: SessionStore
    tracer: InMemoryTracer
    repository: AcademicRepository
    max_validation_retries: int = 1


def _merge_packets(left: EvidencePacket, right: EvidencePacket, packet_id: str) -> EvidencePacket:
    facts = {item.fact_id: item for item in (*left.facts, *right.facts)}
    evidence = {item.evidence_id: item for item in (*left.evidence, *right.evidence)}
    return EvidencePacket(packet_id=packet_id, facts=tuple(facts.values()), evidence=tuple(evidence.values()), coverage=right.coverage if right.coverage != Coverage() else left.coverage, warnings=(*left.warnings, *right.warnings), conflicts=(*left.conflicts, *right.conflicts), tool_results=(*left.tool_results, *right.tool_results))


class AgentRuntime:
    """Explicitly injected bounded state machine; it does not patch imports or globals."""

    def __init__(self, dependencies: RuntimeDependencies) -> None:
        self._deps = dependencies

    @property
    def repository(self) -> AcademicRepository:
        return self._deps.repository

    def options(self) -> dict[str, object]:
        registry = getattr(self._deps.executor, "registry", None)
        tool_names = registry.tool_names() if isinstance(registry, ToolRegistry) else ()
        return {**self.repository.options(), "agent": "bounded-evidence-grounded", "tool_names": tool_names}

    def source(self, chunk_id: str) -> dict[str, object] | None:
        stored = self.repository.source(chunk_id)
        if stored is None:
            return None
        return {
            "chunk_id": stored["chunk_id"], "text": stored["text"], "doc_title": stored["title"],
            "article": stored.get("article") or "", "page_url": stored.get("page_url") or "",
            "file_url": stored.get("file_url") or "", "physical_page": stored.get("physical_page"),
        }

    def ask(self, question: str, *, session_id: str | None = None, model: SynthesisModel | None = None) -> tuple[FinalAnswer, AgentState]:
        if not question.strip() or len(question) > 4000:
            raise ValueError("question must contain 1–4000 characters")
        understanding = StructuredQuestionUnderstanding(model) if model is not None else self._deps.understanding
        synthesizer = LLMClaimSynthesizer(model, fallback=self._deps.synthesizer) if model is not None else self._deps.synthesizer
        state = AgentState(request_id=uuid4().hex, session_id=session_id, raw_question=question.strip())
        previous = self._deps.sessions.get(session_id) if session_id else None
        with self._deps.tracer.start("understanding", request_id=state.request_id):
            draft = understanding.understand(state.raw_question)
            if not draft.program_mentions:
                detected = self.repository.programs_in_text(state.raw_question, draft.cohort)
                if detected:
                    draft = draft.model_copy(update={"program_mentions": tuple(item.canonical_name for item in detected)})
            state.understanding = draft
        state.transition(AgentStatus.NORMALIZE)
        inherited_program = str(previous.get("program_id")) if previous and previous.get("program_id") else None
        stored_cohort = previous.get("cohort") if previous else None
        inherited_cohort = stored_cohort if isinstance(stored_cohort, int) and not isinstance(stored_cohort, bool) else None
        with self._deps.tracer.start("normalization", request_id=state.request_id):
            normalized = self._deps.normalizer(draft, state.raw_question, inherited_program_id=inherited_program, inherited_cohort=inherited_cohort)
            state.normalized_query = normalized
        if normalized.missing_fields:
            state.transition(AgentStatus.CLARIFY)
            answer = self._deps.synthesizer.synthesize(normalized, EvidencePacket(packet_id=stable_id("packet", state.request_id)))
            state.answer = self._deps.validator.validate(answer, EvidencePacket(packet_id=stable_id("packet", state.request_id)))
            state.transition(AgentStatus.FINISH)
            return state.answer, state
        state.transition(AgentStatus.PLAN)
        with self._deps.tracer.start("planning", request_id=state.request_id):
            plan = self._deps.planner(normalized)
            state.plan = plan
        if not plan.operations:
            state.transition(AgentStatus.CLARIFY)
            answer = FinalAnswer(answer_md="该请求不需要或不适合调用校务知识工具。", claims=(), citations=(), clarification="请改为具体的培养方案或校务规定问题。")
            state.answer = answer
            state.transition(AgentStatus.FINISH)
            return answer, state
        state.transition(AgentStatus.EXECUTE)
        with self._deps.tracer.start("execution", request_id=state.request_id, plan_id=plan.plan_id):
            packet = self._deps.executor.execute(plan)
            state.tool_calls = [operation.tool_name for operation in plan.operations]
            state.tool_results = list(packet.tool_results)
            state.evidence = packet
        answer = self._synthesize_validate(state, packet, synthesizer)
        if answer.refused and state.retry_count < self._deps.max_validation_retries and not any(operation.type == "retrieve_policy" for operation in plan.operations):
            state.transition(AgentStatus.TARGETED_RETRIEVAL)
            state.retry_count += 1
            retry_plan = ExecutionPlan(
                plan_id=stable_id("plan", plan.plan_id, "targeted-retrieval"), query=normalized,
                operations=(RetrievePolicyOperation(operation_id=stable_id("op", plan.plan_id, "targeted-retrieval"), args=RetrievePolicyArgs(question=normalized.raw_question, cohort=normalized.cohort, program_ids=normalized.program_ids)),),
            )
            retry_packet = self._deps.executor.execute(retry_plan)
            packet = _merge_packets(packet, retry_packet, stable_id("packet", plan.plan_id, "retry"))
            state.evidence = packet
            answer = self._synthesize_validate(state, packet, synthesizer)
        state.answer = answer
        state.transition(AgentStatus.FINISH)
        if session_id:
            self._deps.sessions.put(session_id, {
                "program_id": normalized.program_ids[0] if normalized.program_ids else None,
                "cohort": normalized.cohort,
                "last_intent": normalized.intent,
                "last_tool_results": list(packet.tool_results),
            })
        return answer, state

    def _synthesize_validate(self, state: AgentState, packet: EvidencePacket, synthesizer: AnswerSynthesizer) -> FinalAnswer:
        state.transition(AgentStatus.SYNTHESIZE)
        with self._deps.tracer.start("synthesis", request_id=state.request_id):
            answer = synthesizer.synthesize(state.normalized_query, packet)  # type: ignore[arg-type]
        state.transition(AgentStatus.VALIDATE)
        with self._deps.tracer.start("validation", request_id=state.request_id):
            validated = self._deps.validator.validate(answer, packet)
        state.claims = list(validated.claims)
        state.validation = [claim.validation for claim in validated.claims]
        return render(validated)


__all__ = ["AgentRuntime", "RuntimeDependencies"]
