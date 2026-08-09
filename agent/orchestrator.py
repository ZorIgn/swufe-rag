"""The sole production Evidence-Grounded Academic Agent runtime."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

from academic.database import AcademicRepository
from agent.coverage_gate import CoverageDecision, CoverageGate
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
from agent.repair import RepairPlanner
from agent.state import AgentState, AgentStatus
from agent.tracing import InMemoryTracer
from evidence.models import CoverageComponent, CoverageReport, EvidencePacket, FinalAnswer
from evidence.provenance import stable_id
from generation.renderer import render
from generation.synthesizer import LLMClaimSynthesizer
from generation.synthesizer import StructuredModel as SynthesisModel
from query.context import RequestContext
from query.schemas import ExecutionPlan, NormalizedQuery
from query.understanding import QuestionUnderstanding as StructuredQuestionUnderstanding


@dataclass(frozen=True)
class RuntimeDependencies:
    understanding: QuestionUnderstanding
    normalizer: QueryNormalizer
    planner: ExecutionPlanner
    executor: ToolExecutor
    synthesizer: AnswerSynthesizer
    validator: AnswerValidator
    coverage_gate: CoverageGate
    repair_planner: RepairPlanner
    sessions: SessionStore
    tracer: InMemoryTracer
    repository: AcademicRepository
    retrieval_mode: str
    readiness: Callable[[], tuple[bool, tuple[str, ...]]]
    max_validation_retries: int = 1


def _merge_packets(left: EvidencePacket, right: EvidencePacket, packet_id: str) -> EvidencePacket:
    """Merge packets by stable identifiers, never concurrent completion order."""

    facts = {item.fact_id: item for item in (*left.facts, *right.facts)}
    evidence = {item.evidence_id: item for item in (*left.evidence, *right.evidence)}
    components: dict[str, CoverageComponent] = {}
    for component in (*left.coverage.components, *right.coverage.components):
        previous = components.get(component.operation_id)
        if previous is None or component.model_dump_json() < previous.model_dump_json():
            components[component.operation_id] = component
    outcomes = {
        item.operation_id: item for item in (*left.execution_results, *right.execution_results)
    }
    return EvidencePacket(
        packet_id=packet_id,
        facts=tuple(facts[key] for key in sorted(facts)),
        evidence=tuple(evidence[key] for key in sorted(evidence)),
        coverage=CoverageReport(components=tuple(components[key] for key in sorted(components))),
        execution_results=tuple(outcomes[key] for key in sorted(outcomes)),
        warnings=tuple(dict.fromkeys((*left.warnings, *right.warnings))),
        conflicts=tuple(dict.fromkeys((*left.conflicts, *right.conflicts))),
    )


class AgentRuntime:
    """Bounded, explicit state machine; it never executes arbitrary model output."""

    def __init__(self, dependencies: RuntimeDependencies) -> None:
        self._deps = dependencies

    @property
    def repository(self) -> AcademicRepository:
        return self._deps.repository

    def readiness(self) -> tuple[bool, tuple[str, ...]]:
        return self._deps.readiness()

    def options(self) -> dict[str, object]:
        registry = getattr(self._deps.executor, "registry", None)
        tool_names = registry.tool_names() if isinstance(registry, ToolRegistry) else ()
        ready, reasons = self.readiness()
        return {
            **self.repository.options(),
            "agent": "bounded-evidence-grounded",
            "tool_names": tool_names,
            "retrieval_mode": self._deps.retrieval_mode,
            "readiness": {"ready": ready, "reasons": reasons},
        }

    def source(self, chunk_id: str) -> dict[str, object] | None:
        stored = self.repository.source(chunk_id)
        if stored is None:
            return None
        return {
            "chunk_id": stored["chunk_id"],
            "text": stored["text"],
            "doc_title": stored["title"],
            "article": stored.get("article") or "",
            "page_url": stored.get("page_url") or "",
            "file_url": stored.get("file_url") or "",
            "physical_page": stored.get("physical_page"),
        }

    def ask(
        self,
        question: str,
        *,
        context: RequestContext | None = None,
        model: SynthesisModel | None = None,
    ) -> tuple[FinalAnswer, AgentState]:
        if not question.strip() or len(question) > 4000:
            raise ValueError("question must contain 1–4000 characters")
        request_context = context or RequestContext()
        session_id = request_context.session_id
        state = AgentState(
            request_id=uuid4().hex,
            session_id=session_id,
            raw_question=question.strip(),
        )
        understanding = (
            StructuredQuestionUnderstanding(model)
            if model is not None
            else self._deps.understanding
        )
        synthesizer = (
            LLMClaimSynthesizer(model, fallback=self._deps.synthesizer)
            if model is not None
            else self._deps.synthesizer
        )
        previous = self._deps.sessions.get(session_id) if session_id else None
        if session_id:
            self._deps.tracer.increment("session_hit_total" if previous else "session_miss_total")

        with self._deps.tracer.start("understanding", request_id=state.request_id):
            draft = understanding.understand(state.raw_question)
            if not draft.program_mentions and not request_context.major:
                detected = self.repository.programs_in_text(state.raw_question, draft.cohort)
                if detected:
                    draft = draft.model_copy(
                        update={"program_mentions": tuple(item.canonical_name for item in detected)}
                    )
            state.understanding = draft
        state.transition(AgentStatus.NORMALIZE)
        inherited_program = (
            str(previous.get("program_id")) if previous and previous.get("program_id") else None
        )
        stored_cohort = previous.get("cohort") if previous else None
        inherited_cohort = (
            stored_cohort
            if isinstance(stored_cohort, int) and not isinstance(stored_cohort, bool)
            else None
        )
        with self._deps.tracer.start("normalization", request_id=state.request_id):
            normalized = self._deps.normalizer(
                draft,
                state.raw_question,
                context=request_context,
                inherited_program_id=inherited_program,
                inherited_cohort=inherited_cohort,
            )
            state.normalized_query = normalized
        self._deps.tracer.increment("request_total", intent=normalized.intent)
        if normalized.missing_fields:
            return self._clarify(state, normalized)

        state.transition(AgentStatus.PLAN)
        with self._deps.tracer.start(
            "planning", request_id=state.request_id, intent=normalized.intent
        ):
            plan = self._deps.planner(normalized)
            state.plan = plan
        if not plan.operations:
            state.transition(AgentStatus.CLARIFY)
            answer = FinalAnswer(
                answer_md="该请求不需要或不适合调用校务知识工具。",
                claims=(),
                citations=(),
                clarification="请改为具体的培养方案或校务规定问题。",
            )
            state.answer = answer
            self._deps.tracer.increment("clarification_total", intent=normalized.intent)
            state.transition(AgentStatus.FINISH)
            return answer, state

        state.transition(AgentStatus.EXECUTE)
        packet = self._execute(state, plan)
        while True:
            state.transition(AgentStatus.COVERAGE_CHECK)
            decision = self._deps.coverage_gate.evaluate(normalized, plan, packet)
            if decision.sufficient:
                break
            self._deps.tracer.increment("coverage_failure_total", intent=normalized.intent)
            repair_plan = self._deps.repair_planner.plan(
                normalized, plan, decision, retry_count=state.repair_count
            )
            if repair_plan is not None:
                state.transition(AgentStatus.REPAIR_PLAN)
                state.repair_count += 1
                self._deps.tracer.increment("repair_attempt_total", intent=normalized.intent)
                state.transition(AgentStatus.REPAIR_EXECUTE)
                repair_packet = self._execute(state, repair_plan)
                packet = _merge_packets(
                    packet, repair_packet, stable_id("packet", plan.plan_id, "repair")
                )
                state.evidence = packet
                continue
            return self._coverage_stop(state, normalized, packet, decision)

        answer = self._synthesize_validate(state, packet, synthesizer)
        if answer.refused and state.regeneration_count < self._deps.max_validation_retries:
            validation_reasons = {
                reason for item in answer.claims for reason in item.validation.reasons
            }
            if validation_reasons & {
                "citation_not_supporting_claim",
                "claim_not_entailed_by_fact",
                "predicate_polarity_conflict",
            }:
                state.transition(AgentStatus.REGENERATE)
                state.regeneration_count += 1
                # Reword from the same packet only; this never launches a new tool call.
                answer = self._validate_same_packet(state, packet, self._deps.synthesizer)
        state.answer = answer
        if answer.refused:
            state.transition(AgentStatus.REFUSE)
            self._deps.tracer.increment("refusal_total", intent=normalized.intent)
        state.transition(AgentStatus.FINISH)
        if session_id:
            self._deps.sessions.put(
                session_id,
                {
                    "program_id": normalized.program_ids[0] if normalized.program_ids else None,
                    "cohort": normalized.cohort,
                    "last_intent": normalized.intent,
                    "last_tool_results": [
                        item.model_dump(mode="json") for item in packet.execution_results
                    ],
                },
            )
        return answer, state

    def _execute(self, state: AgentState, plan: ExecutionPlan) -> EvidencePacket:
        with self._deps.tracer.start(
            "execution", request_id=state.request_id, plan_id=plan.plan_id
        ):
            packet = self._deps.executor.execute(plan)
        state.tool_calls.extend(operation.tool_name for operation in plan.operations)
        state.tool_results.extend(packet.execution_results)
        state.evidence = packet
        for result in packet.execution_results:
            self._deps.tracer.increment(
                "tool_call_total", tool_name=result.tool_name, status=result.status
            )
            self._deps.tracer.increment(
                "tool_latency_ms",
                value=result.latency_ms,
                tool_name=result.tool_name,
                status=result.status,
            )
            if result.status == "timeout":
                self._deps.tracer.increment("tool_timeout_total", tool_name=result.tool_name)
            elif result.status != "success":
                self._deps.tracer.increment(
                    "tool_failure_total", tool_name=result.tool_name, status=result.status
                )
        return packet

    def _clarify(
        self, state: AgentState, query: NormalizedQuery
    ) -> tuple[FinalAnswer, AgentState]:
        state.transition(AgentStatus.CLARIFY)
        answer = self._deps.synthesizer.synthesize(
            query,
            EvidencePacket(packet_id=stable_id("packet", state.request_id)),
        )
        state.answer = answer
        self._deps.tracer.increment("clarification_total")
        state.transition(AgentStatus.FINISH)
        return answer, state

    def _coverage_stop(
        self,
        state: AgentState,
        query: NormalizedQuery,
        packet: EvidencePacket,
        decision: CoverageDecision,
    ) -> tuple[FinalAnswer, AgentState]:
        clarification = None
        if any(
            reason in decision.reasons
            for reason in ("completed_course_unmatched", "completed_course_ambiguous")
        ):
            clarification = "已修课程中存在未匹配或有歧义的名称，请提供课程代码或更完整的课程名称。"
        if clarification is not None:
            state.transition(AgentStatus.CLARIFY)
            self._deps.tracer.increment("clarification_total")
            answer = FinalAnswer(
                answer_md=clarification,
                claims=(),
                citations=(),
                clarification=clarification,
            )
        else:
            state.transition(AgentStatus.REFUSE)
            self._deps.tracer.increment("refusal_total")
            answer = FinalAnswer(
                answer_md="当前证据覆盖不足、范围不匹配或来源状态无法满足该问题的回答条件，因此不返回未经验证的学校事实。",
                claims=(),
                citations=packet.evidence,
                refused=True,
            )
        state.answer = answer
        state.transition(AgentStatus.FINISH)
        return answer, state

    def _synthesize_validate(
        self, state: AgentState, packet: EvidencePacket, synthesizer: AnswerSynthesizer
    ) -> FinalAnswer:
        state.transition(AgentStatus.SYNTHESIZE)
        with self._deps.tracer.start("synthesis", request_id=state.request_id):
            answer = synthesizer.synthesize(self._normalized_query(state), packet)
        return self._validate_same_packet(state, packet, answer=answer)

    @staticmethod
    def _normalized_query(state: AgentState) -> NormalizedQuery:
        query = state.normalized_query
        if query is None:
            raise RuntimeError("agent state has no normalized query")
        return query

    def _validate_same_packet(
        self,
        state: AgentState,
        packet: EvidencePacket,
        synthesizer: AnswerSynthesizer | None = None,
        *,
        answer: FinalAnswer | None = None,
    ) -> FinalAnswer:
        if answer is None:
            assert synthesizer is not None
            answer = synthesizer.synthesize(self._normalized_query(state), packet)
        state.transition(AgentStatus.VALIDATE)
        with self._deps.tracer.start("validation", request_id=state.request_id):
            validated = self._deps.validator.validate(answer, packet)
        state.claims = list(validated.claims)
        state.validation = [claim.validation for claim in validated.claims]
        for claim in validated.claims:
            if not claim.validation.passed:
                self._deps.tracer.increment("validation_failure_total")
                if any("citation" in reason for reason in claim.validation.reasons):
                    self._deps.tracer.increment("citation_failure_total")
        return render(validated)


__all__ = ["AgentRuntime", "RuntimeDependencies"]
