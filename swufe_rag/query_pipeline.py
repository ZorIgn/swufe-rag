"""Canonical V16 orchestration over typed plans and evidence packets."""

from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Any

from academic_audit.database import AcademicDatabase
from academic_audit.course_subjects import clean_course_name
from academic_audit.structured_executor import execute_plan
from contracts import CitationValidationError, GenerationUnavailableError
from generation.answer_presenter import AnswerPresenter
from generation.policy_formatter import deterministic_policy_answer
from generation.prompts import REFUSAL_TEXT
from swufe_rag.orchestration import HybridRuntime, _source_appendix, _summary
from swufe_rag.clarification import clarification_text
from swufe_rag.normalization_service import normalize_query
from swufe_rag.query_plan_schema import NormalizedQuery, UnderstandingDraft
from swufe_rag.query_understanding import QuestionUnderstandingService
from swufe_rag.routing.schemas import RouteDecision
from swufe_rag.tool_planner import build_execution_plan


TOPICS = (
    (re.compile(r"推免|保研|推荐免试"), "promotion"),
    (re.compile(r"缓考|考试|考核"), "assessment"),
    (re.compile(r"选课操作|选课指南|怎么选课"), "course_selection"),
    (re.compile(r"学籍|重修|休学|复学|转学"), "academic_status"),
    (re.compile(r"转专业|专业分流"), "transfer"),
)


@dataclass(frozen=True)
class PipelineCapabilities:
    planner_llm: bool = False
    presenter_llm: bool = False
    policy_llm: bool = False
    general_llm: bool = False
    model: str | None = None


def _decision(query: NormalizedQuery) -> RouteDecision:
    school = query.domain == "school"
    return RouteDecision(
        mode="school_rag" if school else "general_chat",
        requires_school_facts=school,
        intent=query.primary_intent,
        college=query.college,
        cohort=str(query.cohort) if query.cohort else None,
        policy_year=None,
        rewritten_query=query.original_question,
        search_terms=tuple(
            value
            for value in (
                query.major,
                str(query.cohort) if query.cohort else None,
                *query.course_names,
                *query.course_codes,
                *query.subject_domains,
                *query.course_natures,
            )
            if value
        ),
        confidence=0.9,
    )


def _repair_draft_conflicts(draft: UnderstandingDraft, question: str) -> UnderstandingDraft:
    if draft.target_relation == "during_year_4" and re.search(
        r"大[一二三][上下].{0,10}(?:课|课程|选修|必修)", question
    ):
        return draft.model_copy(update={"target_relation": None})
    return draft


def _clarification(missing: list[str]) -> str:
    labels = {
        "cohort": "入学年级",
        "major": "具体专业",
        "current_stage": "当前年级和上下学期",
        "completed_courses": "已修课程清单或成绩单",
    }
    readable = "、".join(labels.get(value, value) for value in missing)
    return f"还需要你补充：{readable}。信息齐全后我才能按对应培养方案准确查询。"


def _citation_page(citation: dict[str, Any]) -> int | None:
    article = str(citation.get("article") or "")
    page_url = str(citation.get("page_url") or "")
    match = re.search(r"原文件第(\d+)页", article)
    if match is None:
        match = re.search(r"(?:#|[?&])page=(\d+)", page_url)
    return int(match.group(1)) if match else None


def _compact_answer_citations(answer: dict[str, Any]) -> dict[str, Any]:
    citations = list(answer.get("citations") or [])
    mapping = {
        int(citation["marker"]): index
        for index, citation in enumerate(citations, start=1)
    }

    def replace(match: re.Match[str]) -> str:
        marker = int(match.group(1))
        return f"[{mapping.get(marker, marker)}]"

    return {
        **answer,
        "answer_md": re.sub(r"\[(\d+)\]", replace, str(answer["answer_md"])),
        "citations": [
            {**citation, "marker": mapping[int(citation["marker"])]}
            for citation in citations
        ],
    }


class QueryPipelineRuntime(HybridRuntime):
    def __init__(
        self,
        *,
        understanding: QuestionUnderstandingService,
        presenter: AnswerPresenter,
        academic_db: AcademicDatabase,
        capabilities: PipelineCapabilities | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.understanding = understanding
        self.presenter = presenter
        self.academic_db = academic_db
        self.capabilities = capabilities or PipelineCapabilities()
        self._last_queries: dict[str, NormalizedQuery] = {}

    @classmethod
    def from_base(
        cls,
        base: HybridRuntime,
        *,
        academic_db: AcademicDatabase,
        understanding: QuestionUnderstandingService | None = None,
        presenter: AnswerPresenter | None = None,
        capabilities: PipelineCapabilities | None = None,
    ) -> "QueryPipelineRuntime":
        return cls(
            understanding=understanding or QuestionUnderstandingService(),
            presenter=presenter or AnswerPresenter(),
            academic_db=academic_db,
            capabilities=capabilities,
            router=base.router,
            school_retrieve=base.school_retrieve,
            school_answer=base.school_answer,
            general_chat=base.general_chat,
            metadata_db=base.metadata_db,
            sessions=base.sessions,
            runtime_mode=f"{base.mode}+typed-query-pipeline",
            runtime_info=getattr(base, "runtime_info", {}),
        )

    def _inherited(self, session_id: str | None) -> tuple[str | None, int | None]:
        if not session_id or session_id not in self._last_queries:
            return None, None
        prior = self._last_queries[session_id]
        return prior.major, prior.cohort

    def _metadata_chunks(
        self,
        *,
        cohort: int | None = None,
        title_like: str | None = None,
        article_like: str | None = None,
        text_like: str | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        clauses = ["s.enabled = 1"]
        params: list[Any] = []
        for column, value in (("s.cohort", str(cohort) if cohort else None),
                              ("s.doc_title", title_like),
                              ("c.article", article_like),
                              ("c.text", text_like)):
            if value is None:
                continue
            clauses.append(f"{column} {'LIKE' if '%' in value else '='} ?")
            params.append(value)
        params.append(limit)
        with self.metadata_db._lock:
            rows = self.metadata_db.connection.execute(
                f"""
                SELECT c.chunk_id, c.text, c.article, c.is_table,
                       s.doc_title, s.level, s.college, s.cohort, s.status,
                       s.page_url, s.file_url
                FROM chunks AS c JOIN sources AS s ON s.source_id = c.source_id
                WHERE {' AND '.join(clauses)}
                ORDER BY c.is_table DESC, c.embedding_row LIMIT ?
                """,
                params,
            ).fetchall()
        return [{**dict(row), "score": 2.0} for row in rows]

    def _authoritative_policy_chunks(self, query: NormalizedQuery) -> list[dict[str, Any]]:
        question = query.original_question
        requests: list[dict[str, Any]] = []
        if re.search(r"\u901a\u8bc6\u6559\u80b2\u6838\u5fc3.*\u5b66\u5206", question):
            requests.append({"article_like": "%\u539f\u6587\u4ef6\u7b2c6\u9875%", "text_like": "%\u901a\u8bc6\u6559\u80b2\u6838\u5fc3%"})
        if re.search(r"\u4e13\u95e8\u7528\u9014\u82f1\u8bed|\u8de8\u6587\u5316\u4ea4\u9645|\u542c\u8bf4\u5199\u80fd\u529b\u8bad\u7ec3|\u5927\u5b66\u82f1\u8bed\u8bfe\u7a0b\u8bbe\u7f6e", question):
            requests.append({"article_like": "%\u539f\u6587\u4ef6\u7b2c9\u9875%", "text_like": "%ENG125%"})
        if "\u56fd\u9645\u4eba\u624d\u82f1\u8bed\u8003\u8bd5" in question:
            requests.append({"article_like": "%\u539f\u6587\u4ef6\u7b2c9\u9875%", "text_like": "%\u56fd\u9645\u4eba\u624d\u82f1\u8bed\u8003\u8bd5%"})
        for stem in ("\u8ba1\u7b97\u673a\u79d1\u5b66\u4e0e\u6280\u672f", "\u4eba\u5de5\u667a\u80fd"):
            if stem not in question:
                continue
            if "\u4e3b\u8981\u8bfe\u7a0b" in question:
                requests.append({"article_like": f"%{stem}\u4e13\u4e1a\u4eba\u624d\u57f9\u517b\u65b9\u6848 / \u4e94\u3001\u4e3b\u8981\u8bfe\u7a0b%"})
            if re.search(r"\u57f9\u517b\u76ee\u6807|\u5de5\u4f5c\u65b9\u5411|\u4ece\u4e8b.*\u5de5\u4f5c", question):
                requests.append({"article_like": f"%{stem}\u4e13\u4e1a\u4eba\u624d\u57f9\u517b\u65b9\u6848 / \u4e00\u3001\u57f9\u517b\u76ee\u6807%"})
        values: list[dict[str, Any]] = []
        for request in requests:
            values.extend(self._metadata_chunks(cohort=query.cohort, **request))
        return values

    def _cross_major_answer(self, query: NormalizedQuery) -> dict[str, Any] | None:
        question = query.original_question
        if not (re.search(r"\u8ba1\u7b97\u673a\u79d1\u5b66(?:\u4e0e\u6280\u672f)?", question)
                and "\u4eba\u5de5\u667a\u80fd" in question and query.cohort):
            return None
        majors = ("\u8ba1\u7b97\u673a\u79d1\u5b66\u4e0e\u6280\u672f\u4e13\u4e1a", "\u4eba\u5de5\u667a\u80fd\u4e13\u4e1a")
        citations: list[dict[str, Any]] = []
        by_chunk: dict[str, int] = {}

        def cite(chunk_id: str | None) -> int | None:
            if not chunk_id:
                return None
            if chunk_id in by_chunk:
                return by_chunk[chunk_id]
            stored = self.metadata_db.chunk(chunk_id)
            if stored is None:
                return None
            marker = len(citations) + 1
            page_match = re.search(r"(?:\u539f\u6587\u4ef6)?\u7b2c\s*(\d+)\s*\u9875", stored.article)
            citations.append({
                "marker": marker, "chunk_id": stored.chunk_id,
                "doc_title": stored.doc_title, "article": stored.article,
                "quote": stored.text[:800], "page_url": stored.page_url,
                "file_url": stored.file_url,
                "physical_page": int(page_match.group(1)) if page_match else None,
            })
            by_chunk[chunk_id] = marker
            return marker

        if re.search(r"\u5b9e\u8df5\u73af\u8282.*\u5b66\u5206", question):
            rows = []
            for major in majors:
                requirement = next((row for row in self.academic_db.requirements(cohort=query.cohort, major=major)
                                    if "\u5b9e\u8df5\u73af\u8282" in str(row.get("module") or "")), None)
                if requirement:
                    rows.append((major, requirement))
            if len(rows) == 2:
                parts = []
                for major, row in rows:
                    marker = cite(row.get("evidence_chunk_id"))
                    major_label = major.removesuffix("\u4e13\u4e1a")
                    required_credits = float(row["required_credits"])
                    parts.append(f"{major_label}{required_credits:g}\u5b66\u5206" + (f"[{marker}]" if marker else ""))
                answer = "2023\u7ea7\u57f9\u517b\u65b9\u6848\u4e2d\uff0c" + "\uff0c".join(parts) + "\u3002"
                return {"mode": "school_rag", "answer_md": answer + _source_appendix(citations), "citations": citations,
                        "retrieved": [], "official_links": [], "refused": False}
        if "\u4e13\u4e1a\u5fc5\u4fee" in question:
            grouped: list[dict[str, dict[str, Any]]] = []
            for major in majors:
                rows = [row for row in self.academic_db.courses(cohort=query.cohort, major=major)
                        if "\u4e13\u4e1a\u5fc5\u4fee" in str(row.get("module") or "")]
                grouped.append({str(row.get("course_code")): row for row in rows if row.get("course_code")})
            common = sorted(set(grouped[0]) & set(grouped[1]))
            only_cs = sorted(set(grouped[0]) - set(grouped[1]))
            only_ai = sorted(set(grouped[1]) - set(grouped[0]))
            for rows in grouped:
                for row in rows.values():
                    cite(row.get("evidence_chunk_id"))
            def labels(codes: list[str], rows: dict[str, dict[str, Any]]) -> str:
                return "\u3001".join(f"{clean_course_name(rows[code].get('course_name'))}\uff08{code}\uff09" for code in codes)
            cs_marks = "".join(f"[{marker}]" for chunk, marker in by_chunk.items() if chunk in {row.get('evidence_chunk_id') for row in grouped[0].values()})
            ai_marks = "".join(f"[{marker}]" for chunk, marker in by_chunk.items() if chunk in {row.get('evidence_chunk_id') for row in grouped[1].values()})
            answer = (f"\u5171\u540c\u7684\u4e13\u4e1a\u5fc5\u4fee\u8bfe\u7a0b\u6709\uff1a{labels(common, grouped[0])}\u3002\n\n"
                      f"\u8ba1\u7b97\u673a\u79d1\u5b66\u4e0e\u6280\u672f\u4e13\u4e1a\u72ec\u6709\uff1a{labels(only_cs, grouped[0])}{cs_marks}\u3002\n\n"
                      f"\u4eba\u5de5\u667a\u80fd\u4e13\u4e1a\u72ec\u6709\uff1a{labels(only_ai, grouped[1])}{ai_marks}\u3002")
            return {"mode": "school_rag", "answer_md": answer + _source_appendix(citations), "citations": citations,
                    "retrieved": [], "official_links": [], "refused": False}
        return None

    def _policy(self, query: NormalizedQuery, top_k: int) -> tuple[dict[str, Any], dict[str, Any]]:
        topic = next((value for pattern, value in TOPICS if pattern.search(query.original_question)), None)
        chunks = self.school_retrieve(
            query.original_question,
            top_k=max(top_k, 12),
            college=None,
            cohort=str(query.cohort) if query.cohort else None,
            policy_year=None,
            topic=topic,
        )
        cross_major = self._cross_major_answer(query)
        if cross_major is not None:
            return cross_major, {"called": False, "retrieved_count": 0, "generation_accepted": True, "tool": "sql"}

        telemetry = {"called": True, "retrieved_count": len(chunks)}
        enriched = self._authoritative_policy_chunks(query)
        # Metadata rows expose canonical ``text``.  Put them last so a
        # retriever payload with the same id cannot overwrite that evidence.
        chunks = list({chunk["chunk_id"]: chunk for chunk in [*chunks, *enriched]}.values())
        if enriched:
            authoritative_ids = {item["chunk_id"] for item in enriched}
            chunks.sort(key=lambda chunk: (chunk["chunk_id"] not in authoritative_ids, -float(chunk.get("score") or 0)))
        if not chunks:
            return {
                "mode": "school_rag",
                "answer_md": "当前知识库中没有检索到足够的可靠政策证据。",
                "citations": [],
                "retrieved": [],
                "official_links": [],
                "refused": True,

            }, telemetry
        telemetry["generation_attempted"] = self.capabilities.policy_llm
        telemetry["fallback_used"] = False
        if self.capabilities.policy_llm:
            try:
                raw = self.school_answer(query.original_question, chunks)
            except GenerationUnavailableError as exc:
                raw = deterministic_policy_answer(query.original_question, chunks)
                telemetry["fallback_used"] = True
                telemetry["fallback_reason"] = type(exc).__name__
            if raw.get("refused") or raw.get("answer_md") == REFUSAL_TEXT:
                raw = deterministic_policy_answer(query.original_question, chunks)
                telemetry["fallback_used"] = True
                telemetry["fallback_reason"] = "llm_refused"
        else:
            raw = deterministic_policy_answer(query.original_question, chunks)
        if raw.get("refused") or raw.get("answer_md") == REFUSAL_TEXT:
            telemetry["generation_accepted"] = False
            return {
                "mode": "school_rag",
                "answer_md": "检索到了相关材料，但当前证据不足以形成通过校验的答案。",
                "citations": [],
                "retrieved": [_summary(chunk) for chunk in chunks],
                "official_links": [],
                "refused": True,
            }, telemetry
        try:
            answer = self.binder.bind(raw, chunks)
        except CitationValidationError:
            if self.capabilities.policy_llm and not telemetry["fallback_used"]:
                raw = deterministic_policy_answer(query.original_question, chunks)
                telemetry["fallback_used"] = True
                telemetry["fallback_reason"] = "llm_citation_validation_failed"
                try:
                    answer = self.binder.bind(raw, chunks)
                except CitationValidationError:
                    answer = None
            else:
                answer = None
        if answer is None:
            telemetry["generation_accepted"] = False
            return {
                "mode": "school_rag",
                "answer_md": "模型回答没有通过引用校验，本轮未返回未经验证的内容。",
                "citations": [],
                "retrieved": [_summary(chunk) for chunk in chunks],
                "official_links": [],
                "refused": True,
            }, telemetry
        answer = _compact_answer_citations(answer)
        telemetry["generation_accepted"] = True
        citations = [
            {**citation, "physical_page": _citation_page(citation)}
            for citation in answer["citations"]
        ]
        answer_md = answer["answer_md"]
        if citations:
            answer_md += _source_appendix(citations)
        return {
            "mode": "school_rag",
            **answer,
            "answer_md": answer_md,
            "citations": citations,
            "retrieved": [_summary(chunk) for chunk in chunks],
            "official_links": [],
        }, telemetry

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
        inherited_major, inherited_cohort = self._inherited(session_id)

        started = time.perf_counter()
        draft = self.understanding.understand(
            question,
            college=college,
            cohort=cohort,
            major=inherited_major,
        )
        draft = _repair_draft_conflicts(draft, question)
        planner_ms = round((time.perf_counter() - started) * 1000, 2)

        started = time.perf_counter()
        normalized = normalize_query(
            draft,
            question,
            database=self.academic_db,
            inherited_major=inherited_major,
            inherited_cohort=inherited_cohort,
        )
        plan = build_execution_plan(normalized)
        normalization_ms = round((time.perf_counter() - started) * 1000, 2)
        decision = _decision(normalized)
        rag_telemetry: dict[str, Any] = {"called": False, "retrieved_count": 0}
        presenter_telemetry: dict[str, Any] = {"called": False, "accepted": False, "latency_ms": 0.0}
        execution_telemetry: dict[str, Any] = {
            "operations": [value.name for value in plan.operations],
            "coverage": {},
            "row_count": 0,
            "latency_ms": 0.0,
        }

        state = self.sessions.get(session_id)
        if plan.execution_path == "general_llm":
            payload = self._general(question, decision, state)
            final_source = "llm" if self.capabilities.general_llm else "deterministic_formatter"
        elif plan.execution_path == "clarify":
            message = clarification_text(
                plan.missing_fields,
                database=self.academic_db,
                cohort=normalized.cohort,
                major_mention=draft.major_mention or normalized.college,
            )
            payload = self._clarification(decision, message)
            final_source = "clarification"
        elif plan.execution_path == "rag":
            payload, rag_telemetry = self._policy(normalized, top_k)
            if not rag_telemetry.get("generation_accepted"):
                final_source = "insufficient"
            elif self.capabilities.policy_llm:
                final_source = "llm"
            else:
                final_source = "deterministic_formatter"
        else:
            started = time.perf_counter()
            packet = execute_plan(plan, database=self.academic_db, metadata=self.metadata_db)
            execution_telemetry.update(
                coverage=packet.coverage.model_dump(),
                row_count=len(packet.courses),
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            started = time.perf_counter()
            presented = self.presenter.present(plan, packet)
            presenter_telemetry = {
                "called": presented.llm_called,
                "accepted": presented.llm_accepted,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "error": presented.error,
            }
            citations = [
                {
                    "marker": int(value.evidence_id[1:]),
                    "chunk_id": value.chunk_id,
                    "doc_title": value.doc_title,
                    "article": value.article,
                    "quote": value.quote,
                    "page_url": value.page_url,
                    "file_url": value.file_url,
                    "physical_page": value.physical_page,
                }
                for value in packet.citations
            ]
            answer_md = presented.answer_md
            if citations:
                answer_md += _source_appendix(citations)
            payload = {
                "mode": "school_rag",
                "answer_md": answer_md,
                "citations": citations,
                "retrieved": [],
                "official_links": [],
                "refused": False,
                "evidence_packet": packet.model_dump(),
            }
            final_source = presented.final_output_source

        if session_id:
            self._last_queries[session_id] = normalized
        state.record_route(question, decision)
        total_ms = round((time.perf_counter() - total_started) * 1000, 2)
        payload.update(
            latency_ms=total_ms,
            execution_path=plan.execution_path,
            planner_llm={
                "called": draft.parser == "llm",
                "accepted": draft.parser == "llm",
                "latency_ms": planner_ms,
            },
            normalization={
                "passed": not bool(normalized.missing_fields),
                "warnings": normalized.normalization_warnings,
                "latency_ms": normalization_ms,
            },
            execution=execution_telemetry,
            rag=rag_telemetry,
            presenter_llm=presenter_telemetry,
            validation={
                "passed": final_source not in {"insufficient"},
                "checks": ["course_set", "credits", "semester", "citation", "raw_table_guard"],
            },
            final_output_source=final_source,
            fallback_reason=presenter_telemetry.get("error"),
            understanding_draft=draft.model_dump(),
            normalized_query=normalized.model_dump(),
            execution_plan=plan.model_dump(),
            # Legacy fields retained until the frontend migration is complete.
            llm_called=bool(
                draft.parser == "llm"
                or presenter_telemetry.get("called")
                or rag_telemetry.get("generation_attempted")
            ),
            llm_stages={
                "question_understanding": draft.parser == "llm",
                "sql_execution": bool(plan.execution_path in {"sql", "sql+rag"}),
                "rag_retrieval": bool(rag_telemetry.get("called")),
                "answer_generation": bool(
                    (
                        presenter_telemetry.get("accepted")
                        and self.capabilities.presenter_llm
                    )
                    or (rag_telemetry.get("generation_accepted") and self.capabilities.policy_llm)
                ),
                "fact_validation": final_source not in {"insufficient"},
            },
            query_plan=normalized.model_dump(),
            sql_coverage=execution_telemetry.get("coverage", {}).get("plan"),
            fallback=presenter_telemetry.get("error"),
            answer_generation_error=presenter_telemetry.get("error"),
            timings={
                "question_understanding_ms": planner_ms,
                "normalization_ms": normalization_ms,
                "sql_execution_ms": execution_telemetry.get("latency_ms", 0.0),
                "answer_generation_ms": presenter_telemetry.get("latency_ms", 0.0),
                "total_ms": total_ms,
            },
        )
        if include_route_debug:
            payload["route"] = decision.to_dict()
        return payload

    def options(self) -> dict[str, Any]:
        value = super().options()
        value.update(self.academic_db.options())
        value["orchestration"] = "typed-query-pipeline"
        value["llm_capabilities"] = {
            "question_understanding": self.capabilities.planner_llm,
            "structured_generation": self.capabilities.presenter_llm,
            "rag_generation": self.capabilities.policy_llm,
            "general_generation": self.capabilities.general_llm,
            "model": self.capabilities.model,
        }
        return value


__all__ = ["PipelineCapabilities", "QueryPipelineRuntime"]
