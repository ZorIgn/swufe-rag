"""Final structured executor with question-sized evidence claims."""

from __future__ import annotations

import re
from typing import Any

from academic_audit.database import AcademicDatabase
from academic_audit.structured_executor_v2 import StructuredExecution, database, resolve_major
from academic_audit.structured_executor_v3 import (
    MAX_PRACTICE_RE,
    TOTAL_CREDIT_RE,
    _candidate_course_names,
    _deduplicate,
    _filter_records,
    _repair_evidence,
)
from academic_audit.structured_qa import _clean_course_name, _ground
from storage.metadata_db import MetadataDB
from swufe_rag.query_plan import QueryPlan


def _hours(record: dict[str, Any]) -> str:
    values = []
    for field, label in (
        ("weekly_hours", "周学时"),
        ("total_hours", "总学时"),
        ("teaching_hours", "课堂学时"),
        ("practice_hours", "实践学时"),
    ):
        value = record.get(field)
        if value is not None:
            values.append(f"{label}{float(value):g}")
    return "，".join(values)


def _lines(records: list[dict[str, Any]], question: str) -> list[str]:
    """Keep each cited claim limited to fields the student requested."""

    values: list[str] = []
    wants_hours = bool(re.search(r"学时|课时", question))
    wants_department = "学院" in question
    for record in records:
        name = _clean_course_name(str(record.get("course_name") or ""))
        code = str(record.get("course_code") or "未标注")
        credits = float(record.get("credits") or 0)
        semester = str(record.get("semester") or "未标注")
        nature = str(record.get("course_nature") or "未标注")
        module = str(record.get("module") or "未标注")
        suffix = f"，{_hours(record)}" if wants_hours else ""
        if wants_department and record.get("department"):
            suffix += f"，开课学院为{record['department']}"
        values.append(
            f"- 第{semester}学期：{code} {name}，{credits:g}学分，"
            f"{nature}，属于{module}{suffix}[{{marker}}]。"
        )
    return values


def execute(
    plan: QueryPlan,
    question: str,
    *,
    metadata_db: MetadataDB,
    db: AcademicDatabase | None = None,
) -> StructuredExecution | None:
    if not plan.requires_sql or plan.cohort is None:
        return None
    db = db or database()
    resolution = resolve_major(db, plan.cohort, plan.major)
    if resolution.status != "covered" or resolution.major is None:
        return None
    major = resolution.major
    all_rows = db.courses(cohort=plan.cohort, major=major)

    if plan.intent == "course_list":
        elective = (
            True
            if any(value in {"选修", "专业方向课程", "自由选修"} for value in plan.course_nature)
            else False if "必修" in plan.course_nature else None
        )
        records = db.courses(
            cohort=plan.cohort,
            major=major,
            semesters=tuple(str(value) for value in plan.semester),
            elective=elective,
        )
        records = _filter_records(records, question)
    elif plan.intent == "course_detail":
        names = _candidate_course_names(plan, question, all_rows)
        if names:
            records = []
            for name in names:
                if re.fullmatch(r"[A-Z]{2,5}\d{3}", name, re.I):
                    records.extend(db.courses(cohort=plan.cohort, major=major, code=name))
                else:
                    records.extend(db.courses(cohort=plan.cohort, major=major, name=name))
        elif MAX_PRACTICE_RE.search(question):
            records = _filter_records(all_rows, question)
        else:
            return None
    else:
        return None

    records = _deduplicate(records)
    if not records:
        return None
    records = _repair_evidence(records, metadata_db)
    grounded = _ground(_lines(records, question), records, metadata_db)
    if grounded is None:
        return None
    answer, chunks = grounded
    scope = "、".join(f"第{value}学期" for value in plan.semester)
    if plan.intent == "course_list":
        label = "选修/专业方向课程" if plan.course_nature else "课程"
        heading = f"{plan.cohort}级{major}{scope}的{label}共{len(records)}门："
        if TOTAL_CREDIT_RE.search(question):
            total = sum(float(row.get("credits") or 0) for row in records)
            heading += f"合计{total:g}学分。"
    else:
        heading = f"{plan.cohort}级{major}的课程信息如下："
    answer["answer_md"] = heading + "\n" + answer["answer_md"]
    return StructuredExecution(answer, chunks, records, major)


__all__ = ["execute"]
