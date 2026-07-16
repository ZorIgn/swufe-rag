"""Normalized exact-course matching for OCR and curriculum naming variants."""

from __future__ import annotations

import re

from academic_audit.database import AcademicDatabase
from academic_audit.structured_executor_v2 import resolve_major
from generation.llm import LLMClient
from swufe_rag.query_plan import QueryPlan
from swufe_rag.query_plan_catalog_v6 import CourseAwareQuestionPlanner as BasePlanner


def _compact(value: str) -> str:
    value = (
        value.replace("Ⅰ", "I")
        .replace("Ⅱ", "II")
        .replace("Ⅲ", "III")
        .replace("Ⅳ", "IV")
    )
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", value).lower()


def _variants(value: str) -> set[str]:
    full = _compact(value)
    base = _compact(re.split(r"[(（]", value, maxsplit=1)[0])
    values = {full, base}
    values.update(item.replace("学科", "") for item in tuple(values))
    return {item for item in values if len(item) >= 2}


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
            if any(variant in target for variant in _variants(name)):
                values.append(name)
        return list(dict.fromkeys(sorted(values, key=lambda item: -len(_compact(item)))))


__all__ = ["CourseAwareQuestionPlanner"]
