"""Production planner: tolerant JSON normalization, strict fields, catalog repair."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from academic_audit.database import AcademicDatabase
from generation.llm import LLMClient
from swufe_rag.query_plan import (
    PLANNER_SYSTEM_PROMPT,
    QueryPlan,
    _json_object,
)
from swufe_rag.query_plan_catalog import (
    CatalogAwareQuestionPlanner,
    STUDENT_SCHOOL_RE,
)


PLAN_FIELDS = {
    "domain",
    "intent",
    "college",
    "major",
    "cohort",
    "semester",
    "course_nature",
    "course_name",
    "requires_sql",
    "requires_rag",
    "missing_fields",
    "normalized_query",
    "confidence",
}
NATURE_ALIASES = {
    "必修课": "必修",
    "选修课": "选修",
    "专业方向课": "专业方向课程",
    "专业选修课": "专业方向课程",
    "自选课": "自由选修",
    "自由选修课": "自由选修",
}
MODULE_LIST_RE = re.compile(
    r"实践环节(?:课|课程)|实践课程中|专业方向(?:课|课程).*(?:包含|有哪些)"
)
SEMESTER_TOTAL_RE = re.compile(r"(?:第?[一二1]学期|大一|两个学期).*(?:总学分|多少学分)")
CROSS_COMPARE_RE = re.compile(r"(?:和|与).*(?:相同点|不同点|比较|分别是多少)")
PROGRAM_REQUIREMENT_RE = re.compile(
    r"计划学制|最长修业|毕业最低|毕业后授予|专业准入|专业准出|培养目标|"
    r"毕业要求|(?:需要|至少|最低)修(?:满|读)?多少.*学分|"
    r"(?:通识教育|大学科基础|专业必修|专业方向|自由选修|实践环节).*需要.*学分"
)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def normalize_model_mapping(raw: Mapping[str, Any]) -> dict[str, Any]:
    extra = set(raw) - PLAN_FIELDS
    if extra:
        raise ValueError(f"query plan contains forbidden fields: {sorted(extra)}")
    value = dict(raw)
    value["semester"] = _as_list(value.get("semester"))
    natures = []
    for item in _as_list(value.get("course_nature")):
        clean = NATURE_ALIASES.get(str(item).strip(), str(item).strip())
        if clean:
            natures.append(clean)
    value["course_nature"] = natures
    value["missing_fields"] = _as_list(value.get("missing_fields"))
    cohort = value.get("cohort")
    if isinstance(cohort, str) and re.fullmatch(r"\d{2}", cohort):
        year = int(cohort)
        value["cohort"] = 2000 + year if year <= 80 else 1900 + year
    value.setdefault("confidence", 0.8)
    return value


def _specialize(plan: QueryPlan, question: str) -> QueryPlan:
    values = plan.to_dict()
    values.pop("tool", None)
    values.pop("parser", None)
    if CROSS_COMPARE_RE.search(question):
        values.update(
            intent="school_requirement",
            requires_sql=False,
            requires_rag=True,
            missing_fields=[],
        )
    elif MODULE_LIST_RE.search(question):
        values.update(intent="course_list", requires_sql=True, requires_rag=False)
        values["missing_fields"] = [
            item for item in values["missing_fields"] if item != "semester"
        ]
    elif SEMESTER_TOTAL_RE.search(question) and values.get("semester"):
        values.update(intent="course_list", requires_sql=True, requires_rag=False)
        values["missing_fields"] = [
            item for item in values["missing_fields"] if item != "semester"
        ]
    elif PROGRAM_REQUIREMENT_RE.search(question):
        values.update(
            intent="school_requirement",
            requires_sql=False,
            requires_rag=True,
        )
        values["missing_fields"] = [
            item for item in values["missing_fields"] if item not in {"semester", "completed_courses"}
        ]
    return QueryPlan.from_mapping(values, question=question, parser=plan.parser)


class ProductionQuestionPlanner(CatalogAwareQuestionPlanner):
    def __init__(self, database: AcademicDatabase, client: LLMClient | None = None) -> None:
        super().__init__(database, client)
        self.offline = CatalogAwareQuestionPlanner(database)

    def plan(self, question: str, **scope: Any) -> QueryPlan:
        if self.client is None:
            return _specialize(self.offline.plan(question, **scope), question)
        context = {
            "question": question,
            "explicit_college": scope.get("college"),
            "explicit_cohort": scope.get("cohort"),
            "inherited_major": scope.get("inherited_major"),
            "inherited_cohort": scope.get("inherited_cohort"),
        }
        try:
            raw_text = self.client.generate(
                PLANNER_SYSTEM_PROMPT, json.dumps(context, ensure_ascii=False)
            )
            values = normalize_model_mapping(_json_object(raw_text))
            if scope.get("college"):
                values["college"] = scope["college"]
            if scope.get("cohort") and str(scope["cohort"]).isdigit():
                values["cohort"] = int(scope["cohort"])
            plan = QueryPlan.from_mapping(values, question=question, parser="llm")
            if plan.domain == "general" and STUDENT_SCHOOL_RE.search(question):
                raise ValueError("school-like question may not be downgraded to general")
            major = (
                plan.major
                or scope.get("inherited_major")
                or self._catalog_major(question, plan.cohort)
            )
            plan = self._repair(plan, question=question, major=major)
            return _specialize(plan, question)
        except Exception:
            return _specialize(self.offline.plan(question, **scope), question)


__all__ = [
    "ProductionQuestionPlanner",
    "normalize_model_mapping",
]
