"""Course-name tolerant planner for parenthesized curriculum labels."""

from __future__ import annotations

import re

from academic_audit.database import AcademicDatabase
from academic_audit.structured_executor_v2 import resolve_major
from generation.llm import LLMClient
from swufe_rag.query_plan import QueryPlan
from swufe_rag.query_plan_catalog_v5 import CourseAwareQuestionPlanner as BasePlanner


def _compact(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", value).lower()


class CourseAwareQuestionPlanner(BasePlanner):
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
            full = _compact(name)
            base = _compact(re.split(r"[(（]", name, maxsplit=1)[0])
            if full in target or (len(base) >= 2 and base in target):
                values.append(name)
        return list(dict.fromkeys(sorted(values, key=lambda item: -len(_compact(item)))))


__all__ = ["CourseAwareQuestionPlanner"]
