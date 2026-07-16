"""Catalog-aware fail-closed question planning for offline and LLM modes."""

from __future__ import annotations

import re
from typing import Any

from academic_audit.database import AcademicDatabase
from generation.llm import LLMClient
from swufe_rag.query_plan import QueryPlan, QuestionPlanner, deterministic_plan


STUDENT_SCHOOL_RE = re.compile(
    r"(?:\d{2}|20\d{2})级|大[一二三四](?:上|下)?|第?[1-8一二三四五六七八]学期|"
    r"选修|必修|课程|课表|学分|专业|学院|培养方案|毕业|推免|保研|免修|"
    r"缓考|重修|转专业|辅修|教务|考试|学位"
)


def _compact(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", value).lower()


class CatalogAwareQuestionPlanner:
    """Use the model for schema filling, then repair scope against the catalog."""

    def __init__(self, database: AcademicDatabase, client: LLMClient | None = None) -> None:
        self.database = database
        self.client = client
        self.model_planner = QuestionPlanner(client)

    def _catalog_major(self, question: str, cohort: int | None) -> str | None:
        clauses = ["is_primary = 1"]
        params: list[Any] = []
        if cohort is not None:
            clauses.append("cohort = ?")
            params.append(cohort)
        rows = self.database.connection.execute(
            f"SELECT DISTINCT major FROM course_offerings WHERE {' AND '.join(clauses)}",
            params,
        ).fetchall()
        text = _compact(question)
        matches: list[str] = []
        for row in rows:
            major = str(row["major"])
            stem = _compact(major.removesuffix("专业"))
            if len(stem) >= 2 and stem in text:
                matches.append(major)
        if not matches:
            resolved = self.database.resolve_major(question, cohort)
            return resolved
        return max(matches, key=lambda value: len(_compact(value)))

    @staticmethod
    def _repair(plan: QueryPlan, *, question: str, major: str | None) -> QueryPlan:
        values = plan.to_dict()
        values.pop("tool", None)
        values.pop("parser", None)
        if major:
            values["major"] = major
        missing = list(values["missing_fields"])
        if values["requires_sql"]:
            required = ["cohort", "major"]
            if values["intent"] == "course_list":
                required.append("semester")
            present = {
                "cohort": values["cohort"] is not None,
                "major": bool(values["major"]),
                "semester": bool(values["semester"]),
            }
            missing = [name for name in missing if not present.get(name, False)]
            missing.extend(name for name in required if not present[name] and name not in missing)
        values["missing_fields"] = missing
        return QueryPlan.from_mapping(values, question=question, parser=plan.parser)

    def plan(self, question: str, **scope: Any) -> QueryPlan:
        if self.client is None:
            enriched = question
            if STUDENT_SCHOOL_RE.search(question):
                enriched += " 课程"
            plan = deterministic_plan(enriched, **scope)
        else:
            plan = self.model_planner.plan(question, **scope)
            if plan.parser != "llm" and STUDENT_SCHOOL_RE.search(question):
                plan = deterministic_plan(question + " 课程", **scope)
        major = plan.major or self._catalog_major(question, plan.cohort)
        return self._repair(plan, question=question, major=major)


__all__ = ["CatalogAwareQuestionPlanner", "STUDENT_SCHOOL_RE"]
