"""Evidence-first orchestration driven by a strict :class:`QueryPlan`.

The language model is allowed to understand a question and phrase verified
facts.  It never selects SQL text, table names, URLs, or school facts.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from academic_audit.database import AcademicDatabase
from academic_audit.structured_executor_v2 import (
    clarification as plan_clarification,
    execute as execute_structured,
    resolve_major,
)
from contracts import CitationValidationError
from generation.prompts import REFUSAL_TEXT
from generation.structured_presenter import StructuredAnswerPresenter
from swufe_rag.orchestration import HybridRuntime, _source_appendix, _summary
from swufe_rag.query_plan import QueryPlan, QuestionPlanner
from swufe_rag.routing.schemas import RouteDecision


@dataclass(frozen=True)
class RuntimeCapabilities:
    real_question_understanding: bool = False
    real_rag_generation: bool = False
    real_general_generation: bool = False
    real_structured_generation: bool = False
    model: str | None = None


def _decision(plan: QueryPlan) -> RouteDecision:
    school = plan.domain == "school"
    return RouteDecision(
        mode="school_rag" if school else "general_chat",
        requires_school_facts=school,
        intent=plan.intent,
        college=plan.college,
        cohort=str(plan.cohort) if plan.cohort is not None else None,
        policy_year=None,
        rewritten_query=plan.normalized_query,
        search_terms=tuple(
            value
            for value in (
                plan.major,
                str(plan.cohort) if plan.cohort is not None else None,
                plan.course_name,
                *plan.course_nature,
            )
            if value
        ),
        confidence=plan.confidence,
    )


def _stage() -> dict[str, bool]:
    return {
        "question_understanding": False,
        "sql_execution": False,
        "rag_retrieval": False,
        "answer_generation": False,
        "fact_validation": False,
    }


class QueryPlanRuntime(HybridRuntime):
    """Execute a validated plan with SQL-first exact facts and RAG fallback."""

    def __init__(
        self,
        *,
        planner: QuestionPlanner,
        presenter: StructuredAnswerPresenter,
        academic_db: AcademicDatabase,
        capabilities: RuntimeCapabilities | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.planner = planner
        self.presenter = presenter
        self.academic_db = academic_db
        self.capabilities = capabilities or RuntimeCapabilities()
        self._last_plans: dict[str, QueryPlan] = {}

    @classmethod
    def from_base(
        cls,
        base: HybridRuntime,
        *,
        planner: QuestionPlanner | None = None,
        presenter: StructuredAnswerPresenter | None = None,
        academic_db: AcademicDatabase,
        capabilities: RuntimeCapabilities | None = None,
    ) -> "QueryPlanRuntime":
        return cls(
            planner=planner or QuestionPlanner(),
            presenter=presenter or StructuredAnswerPresenter(),
            academic_db=academic_db,
            capabilities=capabilities,
            router=base.router,
            school_retrieve=base.school_retrieve,
            school_answer=base.school_answer,
            general_chat=base.general_chat,
            metadata_db=base.metadata_db,
            sessions=base.sessions,
            runtime_mode=f"{base.mode}+query-plan-v2",
            runtime_info=getattr(base, "runtime_info", {}),
        )

    def _inherited(self, session_id: str | None) -> tuple[str | None, int | None]:
        if not session_id:
            return None, None
        prior = self._last_plans.get(session_id)
        if prior is None:
            return None, None
        return prior.major, prior.cohort

    def _retrieve_rag(
        self,
        plan: QueryPlan,
        question: str,
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        terms = [plan.normalized_query, question, plan.major or ""]
        terms.extend(plan.course_nature)
        query = " ".join(dict.fromkeys(value for value in terms if value))
        # Complete curriculum books are registered as school-level documents.
        # A college filter would therefore hide the exact table pages we need.
        return self.school_retrieve(
            query,
            top_k=max(top_k, 12),
            college=None,
            cohort=str(plan.cohort) if plan.cohort is not None else None,
            policy_year=None,
            topic=None,
        )

    def handle_question(
        self,
        question: str,
        *,
        college: str | None = None,
        cohort: str | None = None,
        session_id: str | None = None,
        top_k: int = 12,
        include_route_debug: bool = False,
    ) -> dict[str, Any]:
        total_started = time.perf_counter()
        timings: dict[str, float] = {}
        stages = _stage()
        state = self.sessions.get(session_id)
        inherited_major, inherited_cohort = self._inherited(session_id)

        started = time.perf_counter()
        plan = self.planner.plan(
            question,
            college=college,
            cohort=cohort,
            inherited_major=inherited_major,
            inherited_cohort=inherited_cohort or state.last_cohort,
        )
        timings["question_understanding_ms"] = round(
            (time.perf_counter() - started) * 1000, 2
        )
        stages["question_understanding"] = plan.parser == "llm"
        decision = _decision(plan)

        sql_coverage: bool | None = None
        fallback: str | None = None
        execution_path = plan.tool
        answer_generation_error: str | None = None

        if plan.domain == "general":
            started = time.perf_counter()
            payload = self._general(question, decision, state)
            timings["answer_generation_ms"] = round(
                (time.perf_counter() - started) * 1000, 2
            )
            stages["answer_generation"] = self.capabilities.real_general_generation
            execution_path = "general_llm"
        else:
            clarify = plan_clarification(plan)
            if clarify:
                payload = self._clarification(decision, clarify)
                execution_path = "clarify"
            else:
                structured = None
                if plan.requires_sql and plan.cohort is not None:
                    resolution = resolve_major(
                        self.academic_db, plan.cohort, plan.major
                    )
                    if resolution.status == "ambiguous":
                        choices = "、".join(resolution.candidates[:12])
                        payload = self._clarification(
                            decision,
                            f"已识别到年级，但专业名称对应多个培养方案：{choices}。请指定具体专业。",
                        )
                        execution_path = "clarify"
                    else:
                        sql_coverage = resolution.status == "covered"
                        if sql_coverage:
                            started = time.perf_counter()
                            structured = execute_structured(
                                plan,
                                question,
                                metadata_db=self.metadata_db,
                                db=self.academic_db,
                            )
                            timings["sql_execution_ms"] = round(
                                (time.perf_counter() - started) * 1000, 2
                            )
                            stages["sql_execution"] = True
                        if structured is not None:
                            canonical = structured.answer["answer_md"]
                            started = time.perf_counter()
                            wording, called, answer_generation_error = self.presenter.present(
                                question,
                                canonical,
                                structured.answer["citations"],
                            )
                            timings["answer_generation_ms"] = round(
                                (time.perf_counter() - started) * 1000, 2
                            )
                            stages["answer_generation"] = called
                            stages["fact_validation"] = called
                            structured.answer["answer_md"] = wording
                            payload = {
                                "mode": "school_rag",
                                **structured.answer,
                                "retrieved": [
                                    _summary(chunk) for chunk in structured.chunks
                                ],
                                "official_links": [],
                                "route": decision,
                            }
                            execution_path = (
                                "sql+llm" if called else "sql+deterministic"
                            )
                        else:
                            # Complete user scope + missing SQL row is a coverage or
                            # extraction issue.  It is never reported as missing major.
                            fallback = "table_rag"
                            execution_path = "sql->table_rag"
                if "payload" not in locals() or payload.get("route") is not decision:
                    started = time.perf_counter()
                    chunks = self._retrieve_rag(plan, question, top_k=top_k)
                    timings["rag_retrieval_ms"] = round(
                        (time.perf_counter() - started) * 1000, 2
                    )
                    stages["rag_retrieval"] = True
                    if not chunks:
                        payload = self._insufficient(decision, retrieved=[])
                    else:
                        started = time.perf_counter()
                        raw = self.school_answer(plan.normalized_query, chunks)
                        timings["answer_generation_ms"] = round(
                            (time.perf_counter() - started) * 1000, 2
                        )
                        stages["answer_generation"] = (
                            self.capabilities.real_rag_generation
                        )
                        if raw["refused"] or raw["answer_md"] == REFUSAL_TEXT:
                            payload = self._insufficient(
                                decision, retrieved=chunks
                            )
                        else:
                            try:
                                answer = self.binder.bind(raw, chunks)
                                stages["fact_validation"] = True
                            except CitationValidationError:
                                payload = self._insufficient(
                                    decision, retrieved=chunks
                                )
                            else:
                                payload = {
                                    "mode": "school_rag",
                                    **answer,
                                    "retrieved": [
                                        _summary(chunk) for chunk in chunks
                                    ],
                                    "official_links": [],
                                    "route": decision,
                                }
                    if fallback is None:
                        execution_path = "rag+llm" if stages["answer_generation"] else "rag+deterministic"

        state.record_route(question, decision)
        if session_id:
            self._last_plans[session_id] = plan
        latency = round((time.perf_counter() - total_started) * 1000, 2)
        timings["total_ms"] = latency
        payload["latency_ms"] = latency
        if (
            payload.get("mode") == "school_rag"
            and not payload.get("refused")
            and payload.get("citations")
        ):
            payload["answer_md"] += _source_appendix(payload["citations"])
        payload["execution_path"] = execution_path
        payload["llm_called"] = any(
            stages[name]
            for name in ("question_understanding", "answer_generation")
        )
        payload["llm_stages"] = stages
        payload["model"] = self.capabilities.model if payload["llm_called"] else None
        payload["query_plan"] = plan.to_dict()
        payload["sql_coverage"] = sql_coverage
        payload["fallback"] = fallback
        payload["answer_generation_error"] = answer_generation_error
        payload["timings"] = timings
        if include_route_debug:
            payload["route"] = decision.to_dict()
        else:
            payload.pop("route", None)
        return payload

    def options(self) -> dict[str, Any]:
        value = super().options()
        value.update(self.academic_db.options())
        value["orchestration"] = "query-plan-v2"
        value["llm_capabilities"] = {
            "question_understanding": self.capabilities.real_question_understanding,
            "rag_generation": self.capabilities.real_rag_generation,
            "structured_generation": self.capabilities.real_structured_generation,
            "general_generation": self.capabilities.real_general_generation,
            "model": self.capabilities.model,
        }
        return value


__all__ = ["QueryPlanRuntime", "RuntimeCapabilities"]
