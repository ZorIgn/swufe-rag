"""V16 structured answer rendering and optional evidence-bound explanation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from academic_audit.semesters import semester_values
from generation.fact_validator import validate_explanation
from generation.llm import LLMClient
from swufe_rag.evidence import CourseFact, EvidencePacket
from swufe_rag.query_plan_schema import ExecutionPlan


ANSWER_SYSTEM_PROMPT = """你是教务事实表达器，只输出 JSON。
你只能使用 evidence_packet 中的事实生成简短总结、解释、警告或澄清问题。
先针对用户原问题给出自然、清楚的直接回答，再解释证据；不要只复述数据库字段。
若问题是规划或策略，必须区分“可选范围”和“必须完成”，给出可执行步骤，并说明缺少成绩单或实时开课数据造成的边界。
课程查询结果默认是符合条件的候选范围；除非 evidence_packet 有带引用的明确要求，不得声称所有候选都必须修完。
summary 与 explanations 应共同形成连贯回答，避免僵硬模板、无标点长句和中英文表头残片。
每条包含学校事实的 explanation 必须绑定支持它的 evidence_ids。
禁止新增或修改课程、代码、学分、学期、课程性质、模块、结论、网址和来源。
完整课程表由程序渲染，你不得重写或枚举整张课程表。
必须区分培养方案安排与实际开课，不得输出 OCR 表格原文。
JSON 字段只能是 summary、explanations、warnings、clarification_question。
explanations 的每项只能包含 text 和 evidence_ids。"""
ANSWER_SYSTEM_PROMPT += """
Credits are isolated by curriculum module. Never claim that credits from a
free-elective, general-education, compulsory, or practice module can fill a
professional-direction module deficit (or vice versa) unless the evidence
packet explicitly states that substitution rule. Derived arithmetic must use
only the provided records and module requirements.
"""


@dataclass(frozen=True)
class PresentedAnswer:
    answer_md: str
    llm_called: bool
    llm_accepted: bool
    final_output_source: str
    error: str | None = None


def _marker(evidence_id: str | None) -> str:
    if not evidence_id or not evidence_id.startswith("E"):
        return ""
    return f"[{evidence_id[1:]}]"


def _course_table(courses: list[CourseFact]) -> str:
    if not courses:
        return ""
    lines = [
        "| 课程代码 | 课程名称 | 学分 | 学期 | 性质 | 模块 |",
        "|---|---|---:|---:|---|---|",
    ]
    for course in courses:
        lines.append(
            f"| {course.code} | {course.name}{_marker(course.evidence_id)} | "
            f"{course.credits:g} | {course.semester} | {course.nature} | {course.module} |"
        )
    return "\n".join(lines)

from generation.course_renderer import course_table as _course_table


def _by_ids(packet: EvidencePacket, ids: list[str]) -> list[CourseFact]:
    wanted = set(ids)
    return [course for course in packet.courses if course.record_id in wanted]


def _credit_scope_summary(plan: ExecutionPlan, courses: list[CourseFact]) -> str | None:
    query = plan.query
    if "credit_total" not in query.requested_outputs or not query.target_semesters:
        return None
    targets = set(query.target_semesters)
    exact_terms = {frozenset({value}) for value in targets}
    exact_required = [
        course for course in courses
        if semester_values(course.semester) in exact_terms
        and "\u5fc5\u4fee" in course.nature
    ]
    flexible = [
        course for course in courses
        if len(semester_values(course.semester)) > 1
        and targets.intersection(semester_values(course.semester))
    ]
    exact_electives = [
        course for course in courses
        if semester_values(course.semester) in exact_terms
        and "\u5fc5\u4fee" not in course.nature
    ]
    total = sum(course.credits for course in exact_required)
    label = "\u3001".join(f"\u7b2c{value}\u5b66\u671f" for value in sorted(targets))
    body = f"{label}\u4e2d\uff0c\u5f00\u8bfe\u5b66\u671f\u53ea\u6807\u6ce8\u5728\u8be5\u8303\u56f4\u5185\u4e14\u6027\u8d28\u4e3a\u5fc5\u4fee\u7684\u8bfe\u7a0b\uff0c\u5b66\u5206\u5408\u8ba1\u4e3a **{total:g}\u5b66\u5206**\u3002"
    if flexible or exact_electives:
        body += (f"\u53e6\u6709{len(flexible)}\u95e8\u8de8\u5b66\u671f\u8bfe\u7a0b\u548c{len(exact_electives)}\u95e8\u8be5\u5b66\u671f\u9009\u4fee\u8bfe\uff1b"
                 "\u57f9\u517b\u65b9\u6848\u6ca1\u6709\u5c06\u8fd9\u4e9b\u5b66\u5206\u552f\u4e00\u5206\u644a\u5230\u67d0\u4e00\u5b66\u671f\uff0c\u56e0\u6b64\u4e0d\u80fd\u91cd\u590d\u8ba1\u5165\u5f97\u51fa\u201c\u552f\u4e00\u5b66\u671f\u603b\u5b66\u5206\u201d\u3002")
    return body

def _course_detail_summary(courses: list[CourseFact], question: str) -> str:
    values: list[str] = []
    include_hours = "\u5b66\u65f6" in question
    include_department = bool(re.search(r"\u54ea\u4e2a\u5b66\u9662|\u5f00\u8bfe\u5b66\u9662|\u7531.*\u5b66\u9662", question))
    for course in courses:
        parts = [
            f"{course.credits:g}\u5b66\u5206",
            f"\u5f00\u8bfe\u5b66\u671f\u4e3a{course.semester}",
            f"\u6027\u8d28\u4e3a{course.nature}",
            f"\u5c5e\u4e8e{course.module}",
        ]
        if include_hours:
            parts.append(f"\u603b\u5b66\u65f6{course.total_hours:g}" if course.total_hours is not None else "\u603b\u5b66\u65f6\u672a\u6807\u6ce8")
            parts.append(f"\u8bfe\u5802\u5b66\u65f6{course.teaching_hours:g}" if course.teaching_hours is not None else "\u8bfe\u5802\u5b66\u65f6\u672a\u6807\u6ce8")
            parts.append(f"\u5b9e\u8df5\u5b66\u65f6{course.practice_hours:g}" if course.practice_hours is not None else "\u5b9e\u8df5\u5b66\u65f6\u672a\u6807\u6ce8")
        if include_department:
            department = course.department or "\u672a\u6807\u6ce8"
            parts.append(f"\u5f00\u8bfe\u5b66\u9662\u4e3a{department}")
        values.append(f"{course.name}\uff08{course.code}\uff09\uff1a" + "\uff0c".join(parts) + _marker(course.evidence_id) + "\u3002")
    if len(values) == 1:
        return values[0]
    return "\n".join(f"- {value}" for value in values)


def deterministic_body(plan: ExecutionPlan, packet: EvidencePacket) -> str:
    query = plan.query
    scope = f"{query.cohort}级{query.major}" if query.cohort and query.major else "当前查询"

    sections: list[str] = []
    show_hours = bool(re.search(r"学时", query.original_question))
    show_department = bool(re.search(r"哪个学院|开课学院|由.*学院", query.original_question))

    minimum = next(
        (fact for fact in packet.facts if fact.get("field") == "graduation_min_credits"),
        None,
    )
    module_only = bool(query.course_modules and "module_breakdown" in query.requested_outputs)
    if minimum and not module_only:
        marker = _marker(str(minimum.get("evidence_id") or ""))
        sections.append(f"{scope}的毕业最低学分为 **{float(minimum['value']):g} 学分**{marker}。")

    visible_requirements = [
        value for value in packet.requirements
        if value.evidence_id is not None
    ]
    if query.course_modules:
        visible_requirements = [
            value for value in packet.requirements
            if any(module in value.module for module in query.course_modules)
        ]
    if visible_requirements:
        lines = [
            "### 培养方案模块要求",
            "| 模块 | 最低学分 | 说明 |",
            "|---|---:|---|",
        ]
        for requirement in visible_requirements:
            required = (
                f"{requirement.required_credits:g}"
                if requirement.required_credits is not None
                else "未明确提取"
            )
            lines.append(
                f"| {requirement.module} | {required} | "
                f"{requirement.rule_text or '—'}{_marker(requirement.evidence_id)} |"
            )
        sections.append("\n".join(lines))

    for result in packet.operation_results:
        name = result.get("operation")
        record_ids = list(result.get("record_ids") or [])
        courses = _by_ids(packet, record_ids)
        if name == "list_courses":
            if courses and re.search(r"实践学时.*最多|最多.*实践学时", query.original_question):
                highest = max((value.practice_hours or 0) for value in courses)
                courses = [value for value in courses if (value.practice_hours or 0) == highest]
                sections.append(f"实践学时最多的是以下课程（{highest:g} 学时）：")
            if courses:
                semester_text = "、".join(str(value) for value in query.target_semesters)
                heading = (
                    f"### {scope}第{semester_text}学期培养方案课程"
                    if semester_text
                    else f"### {scope}培养方案课程"
                )
                credit_summary = _credit_scope_summary(plan, courses)
                if credit_summary:
                    sections.append(credit_summary)
                sections.append(heading + "\n\n" + _course_table(courses, include_hours=show_hours, include_department=show_department))
            elif result.get("status") == "classification_incomplete":
                if query.information_scope == "actual_offerings":
                    target = "、".join(str(value) for value in query.target_semesters)
                    label = f"第{target}学期" if target else "目标学期"
                    sections.append(
                        f"培养方案中没有检出安排在{label}且符合主题条件的课程。"
                        "这只说明培养方案没有这样安排，不代表实时选课系统一定没有可选课程。"
                    )
                else:
                    sections.append(
                        "当前结构化课程中未检出符合主题条件的课程，但主题分类尚未完成全量审计，"
                        "因此不能把这一结果解释为实际开课数量为 0。"
                    )
            else:
                sections.append("按当前完整查询条件，结构化培养方案中没有匹配课程。")
        elif name == "get_course_detail" and courses:
            summary = _course_detail_summary(courses, query.original_question)
            sections.append(summary)
            sections.append(f"### {scope}课程信息\n\n" + _course_table(courses, include_hours=show_hours, include_department=show_department))
        elif name == "list_courses_before_semester":
            deadline = result.get("deadline_semester")
            if courses:
                note = (
                    f"以下是培养方案中安排在第 1—{int(deadline) - 1} 学期且符合条件的课程。"
                    "若查询的是选修课，它们是可选范围，不代表每门都必须修读；最终以模块最低学分要求为准。"
                )
                sections.append("### 大四前培养方案课程范围\n\n" + note + "\n\n" + _course_table(courses))
            else:
                sections.append("在截止学期之前没有查询到符合全部条件的结构化课程记录。")
        elif name == "list_unavoidable_courses_after_semester" and courses:
            sections.append(
                "### 截止学期后仍安排的必修或实践环节\n\n" + _course_table(courses)
            )
        elif name == "audit_completed_courses":
            assumed_ids = list(result.get("assumed_scope_record_ids") or [])
            if assumed_ids:
                assumed_credits = float(result.get("assumed_scope_credits") or 0)
                sections.append(
                    "### 按范围声明进行的条件核算\n\n"
                    f"根据你提供的范围声明，系统匹配到 **{len(assumed_ids)} 门、{assumed_credits:g} 学分**的课程。"
                    "本结果暂按这些课程均已通过并获得学分计算；如果只是选课但尚未通过，实际获得学分需要以成绩单为准。"
                )

            modules = result.get("modules") or []
            if modules:
                lines = [
                    "### 学分进度核算",
                    "| 模块 | 要求学分 | 已匹配学分 | 剩余学分 | 核验状态 |",
                    "|---|---:|---:|---:|---|",
                ]
                for module in modules:
                    required = module.get("required_credits")
                    remaining = module.get("remaining_credits")
                    completed = module.get("completed_credits")
                    if module.get("completed_by_unverified_claim"):
                        status = "用户声明，未核验"
                    elif module.get("completion_known"):
                        status = "按已修课程匹配"
                    else:
                        status = "缺少成绩单，无法核算"
                    lines.append(
                        f"| {module['module']} | {required if required is not None else '—'} | "
                        f"{completed if completed is not None else '—'} | {remaining if remaining is not None else '—'} | "
                        f"{status} |"
                    )
                sections.append("\n".join(lines))

    feasible = packet.audit.get("feasibility")
    if feasible:
        status = feasible.get("curriculum_feasibility")
        label = {
            "feasible": "培养方案层面可行",
            "infeasible": "无法做到大四完全没有教学活动",
            "insufficient_input": "需要补充已修课程后才能完成个性化判断",
        }.get(status, "尚不能判断")
        sections.append(
            f"### 可行性结论\n\n**{label}。**{feasible.get('reason', '')}\n\n"
            f"> 数据边界：{feasible.get('data_boundary', '')}"
        )

    if packet.warnings:
        sections.append("### 需要注意\n\n" + "\n".join(f"- {value}" for value in packet.warnings))
    if query.information_scope == "actual_offerings":
        sections.append(
            "> 当前回答依据培养方案，不是实时选课目录。实际是否开课、能否选中，需要以对应学期教务系统为准。"
        )
    if not sections:
        return "当前结构化证据不足，无法在不猜测的前提下回答。"
    return "\n\n".join(sections)


def _render_llm_draft(draft: dict[str, Any]) -> str:
    parts: list[str] = []
    summary = str(draft.get("summary") or "").strip()
    if summary:
        parts.append(summary)
    for item in draft.get("explanations") or []:
        text = str(item.get("text") or "").strip()
        markers = "".join(_marker(value) for value in item.get("evidence_ids") or [])
        if text:
            parts.append(text + markers)
    warnings = [str(value).strip() for value in draft.get("warnings") or [] if str(value).strip()]
    if warnings:
        parts.append("\n".join(f"- {value}" for value in warnings))
    clarification = str(draft.get("clarification_question") or "").strip()
    if clarification:
        parts.append(clarification)
    return "\n\n".join(parts)


class AnswerPresenter:
    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client

    def present(self, plan: ExecutionPlan, packet: EvidencePacket) -> PresentedAnswer:
        canonical = deterministic_body(plan, packet)
        if self.client is None:
            return PresentedAnswer(canonical, False, False, "deterministic_formatter")
        prompt_packet = packet.model_dump()
        prompt_packet["requirements"] = [
            value for value in prompt_packet.get("requirements", [])
            if value.get("evidence_id")
        ]
        for citation in prompt_packet.get("citations", []):
            citation.pop("quote", None)
            citation.pop("page_url", None)
            citation.pop("file_url", None)
        prompt = json.dumps(
            {
                "question": plan.query.original_question,
                "normalized_query": plan.query.model_dump(),
                "evidence_packet": prompt_packet,
                "output_schema": {
                    "summary": "string",
                    "explanations": [{"text": "string", "evidence_ids": ["E1"]}],
                    "warnings": ["string"],
                    "clarification_question": "string|null",
                },
            },
            ensure_ascii=False,
        )
        try:
            raw = self.client.generate(ANSWER_SYSTEM_PROMPT, prompt).strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            draft = json.loads(raw)
        except Exception:
            return PresentedAnswer(canonical, True, False, "deterministic_formatter", "generation_failed")
        if not isinstance(draft, dict):
            return PresentedAnswer(canonical, True, False, "deterministic_formatter", "invalid_json")
        valid, error = validate_explanation(draft, packet, plan)
        if not valid:
            return PresentedAnswer(canonical, True, False, "deterministic_formatter", error)
        explanation = _render_llm_draft(draft)
        answer = explanation + "\n\n" + canonical if explanation else canonical
        return PresentedAnswer(answer, True, True, "llm")


__all__ = ["AnswerPresenter", "PresentedAnswer", "deterministic_body"]
