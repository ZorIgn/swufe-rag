"""Final course-aware QueryPlan refinements for the full-school catalog."""

from __future__ import annotations

import re
from typing import Any

from academic_audit.database import AcademicDatabase
from academic_audit.structured_executor_v2 import resolve_major
from generation.llm import LLMClient
from swufe_rag.query_plan import QueryPlan
from swufe_rag.query_plan_catalog_v3 import ProductionQuestionPlanner


SCHOOL_POLICY_RE = re.compile(
    r"通常从第几个学期|专门用途英语模块|跨文化交际模块|实践教学.*占比|"
    r"有哪些主要课程|取得学分需要.*证明"
)
FILTERED_LIST_RE = re.compile(
    r"(?:需要修|有哪些|哪些).*?(?:数学|程序设计|思想政治|思政|体育|军事|实践环节|实践课程).*?课"
)
PRACTICE_MAX_RE = re.compile(r"实践学时.*(?:最多|最高)|(?:最多|最高).*实践学时")


def _compact(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", value).lower()


class CourseAwareQuestionPlanner(ProductionQuestionPlanner):
    def __init__(self, database: AcademicDatabase, client: LLMClient | None = None) -> None:
        super().__init__(database, client)

    def _course_names(self, plan: QueryPlan, question: str) -> list[str]:
        if plan.cohort is None or not plan.major:
            return []
        resolution = resolve_major(self.database, plan.cohort, plan.major)
        if resolution.status != "covered" or resolution.major is None:
            return []
        target = _compact(question)
        values: list[str] = []
        for row in self.database.courses(cohort=plan.cohort, major=resolution.major):
            name = str(row["course_name"])
            compact = _compact(name)
            if len(compact) >= 2 and compact in target:
                values.append(name)
        return list(dict.fromkeys(sorted(values, key=lambda item: -len(_compact(item)))))

    def plan(self, question: str, **scope: Any) -> QueryPlan:
        plan = super().plan(question, **scope)
        values = plan.to_dict()
        values.pop("tool", None)
        values.pop("parser", None)
        if SCHOOL_POLICY_RE.search(question):
            values.update(
                intent="school_requirement",
                requires_sql=False,
                requires_rag=True,
                course_name=None,
                missing_fields=[],
            )
            if "英语模块" in question:
                values["major"] = None
        elif FILTERED_LIST_RE.search(question):
            values.update(
                intent="course_list",
                requires_sql=True,
                requires_rag=False,
                course_name=None,
            )
            # A filtered all-program list (for example all practice courses)
            # intentionally has no semester predicate.
            values["missing_fields"] = [
                item for item in values["missing_fields"] if item != "semester"
            ]
        elif PRACTICE_MAX_RE.search(question):
            values.update(
                intent="course_detail",
                requires_sql=True,
                requires_rag=False,
                course_name=None,
            )
            values["missing_fields"] = [
                item for item in values["missing_fields"] if item != "semester"
            ]
        else:
            names = self._course_names(plan, question)
            if names:
                values.update(
                    intent="course_detail",
                    requires_sql=True,
                    requires_rag=False,
                    course_name="、".join(names),
                )
                values["missing_fields"] = [
                    item for item in values["missing_fields"] if item != "semester"
                ]
        return QueryPlan.from_mapping(values, question=question, parser=plan.parser)


__all__ = ["CourseAwareQuestionPlanner"]
