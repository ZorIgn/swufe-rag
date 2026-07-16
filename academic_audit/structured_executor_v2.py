"""Parameter-bound SQL execution for validated :class:`QueryPlan` objects."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

from academic_audit.database import AcademicDatabase
from academic_audit.structured_qa import _clean_course_name, _course_name_from_question, _ground
from contracts import RetrievedChunk
from storage.metadata_db import MetadataDB
from swufe_rag.query_plan import QueryPlan


DEFAULT_DATABASE = Path(__file__).parents[1] / "data" / "academic_v2.sqlite3"


def _compact(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", value).lower().removesuffix("专业")


@lru_cache(maxsize=2)
def database(path: str = str(DEFAULT_DATABASE)) -> AcademicDatabase:
    return AcademicDatabase(path)


@dataclass(frozen=True)
class MajorResolution:
    status: str
    major: str | None
    candidates: tuple[str, ...] = ()


@dataclass
class StructuredExecution:
    answer: dict[str, Any]
    chunks: list[RetrievedChunk]
    records: list[dict[str, Any]]
    major: str
    sql_coverage: bool = True


def resolve_major(
    db: AcademicDatabase, cohort: int, requested: str | None
) -> MajorResolution:
    if not requested:
        return MajorResolution("missing", None)
    rows = db.connection.execute(
        """
        SELECT DISTINCT major FROM course_offerings
        WHERE is_primary = 1 AND cohort = ? ORDER BY major
        """,
        (cohort,),
    ).fetchall()
    majors = [str(row["major"]) for row in rows]
    target = _compact(requested)
    exact = [major for major in majors if _compact(major) == target]
    if len(exact) == 1:
        return MajorResolution("covered", exact[0], tuple(exact))
    candidates = [
        major
        for major in majors
        if target in _compact(major) or _compact(major) in target
    ]
    if len(candidates) == 1:
        return MajorResolution("covered", candidates[0], tuple(candidates))
    if candidates:
        return MajorResolution("ambiguous", None, tuple(candidates))
    return MajorResolution("uncovered", None)


def clarification(plan: QueryPlan) -> str | None:
    if not plan.missing_fields:
        return None
    labels = {
        "college": "学院",
        "major": "专业",
        "cohort": "入学年级",
        "semester": "学期",
        "completed_courses": "已修课程",
    }
    values = "、".join(labels[value] for value in plan.missing_fields)
    return f"请补充{values}。这些条件会影响培养方案查询结果。"


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


def _course_lines(records: list[dict[str, Any]]) -> list[str]:
    lines = []
    for record in records:
        hours = _hours(record)
        detail = f"，{hours}" if hours else ""
        lines.append(
            f"- 第{record.get('semester')}学期：{record.get('course_code') or '未标注'} "
            f"{_clean_course_name(str(record['course_name']))}，"
            f"{float(record.get('credits') or 0):g}学分，"
            f"{record.get('course_nature') or '未标注'}，"
            f"{record.get('module') or '未标注'}{detail}[{{marker}}]。"
        )
    return lines


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

    records: list[dict[str, Any]] = []
    if plan.intent == "course_list":
        elective = (
            True
            if any(value in {"选修", "专业方向课程", "自由选修"} for value in plan.course_nature)
            else False
            if "必修" in plan.course_nature
            else None
        )
        records = db.courses(
            cohort=plan.cohort,
            major=major,
            semesters=tuple(str(value) for value in plan.semester),
            elective=elective,
        )
    elif plan.intent == "course_detail":
        all_courses = db.courses(cohort=plan.cohort, major=major)
        code = (
            plan.course_name.upper()
            if plan.course_name and re.fullmatch(r"[A-Z]{2,5}\d{3}", plan.course_name, re.I)
            else None
        )
        name = None if code else (
            plan.course_name or _course_name_from_question(question, all_courses)
        )
        records = db.courses(
            cohort=plan.cohort,
            major=major,
            code=code,
            name=name,
        )
        if records:
            records = [records[0]]
    else:
        return None
    if not records:
        return None

    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        key = (
            str(record.get("course_code") or record["course_name"]),
            str(record.get("semester") or ""),
            str(record.get("module") or ""),
        )
        unique.setdefault(key, record)
    records = list(unique.values())
    lines = _course_lines(records)
    grounded = _ground(lines, records, metadata_db)
    if grounded is None:
        return None
    answer, chunks = grounded
    if plan.intent == "course_list":
        scope = "、".join(f"第{value}学期" for value in plan.semester)
        label = "选修课程" if plan.course_nature else "课程"
        answer["answer_md"] = (
            f"{plan.cohort}级{major}{scope}的{label}共{len(records)}门：\n"
            + answer["answer_md"]
        )
    else:
        record = records[0]
        answer["answer_md"] = (
            f"{plan.cohort}级{major}的"
            + _clean_course_name(str(record["course_name"]))
            + "信息如下：\n"
            + answer["answer_md"]
        )
    return StructuredExecution(answer, chunks, records, major)


__all__ = [
    "MajorResolution",
    "StructuredExecution",
    "clarification",
    "database",
    "execute",
    "resolve_major",
]
