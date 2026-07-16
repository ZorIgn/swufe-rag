"""Evidence-repaired structured curriculum execution.

This module is intentionally read-only.  QueryPlan values are mapped to fixed
Python predicates and the parameter-bound methods on :class:`AcademicDatabase`;
no model text is ever executed as SQL.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from academic_audit.database import AcademicDatabase
from academic_audit.structured_executor_v2 import (
    StructuredExecution,
    database,
    resolve_major,
)
from academic_audit.structured_qa import _clean_course_name, _ground
from storage.metadata_db import MetadataDB
from swufe_rag.query_plan import QueryPlan


PRACTICE_RE = re.compile(r"实践环节|实践课程|实训|实验")
MATH_RE = re.compile(r"数学课程|哪些数学|数学课")
PROGRAMMING_RE = re.compile(r"程序设计课程|程序设计课")
POLITICS_RE = re.compile(r"思想政治|思政")
PE_RE = re.compile(r"体育课程|体育课")
MILITARY_RE = re.compile(r"军事教育|军事课程|军事课")
MAX_PRACTICE_RE = re.compile(r"实践学时.*(?:最多|最高)|(?:最多|最高).*实践学时")
TOTAL_CREDIT_RE = re.compile(r"总共.*学分|总学分|合计.*学分|一共.*学分")


def _compact(value: object) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", str(value or "")).lower()


def _deduplicate(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        key = (
            str(record.get("course_code") or record.get("course_name") or ""),
            str(record.get("semester") or ""),
            str(record.get("module") or ""),
        )
        unique.setdefault(key, record)
    return list(unique.values())


def _filter_records(
    records: list[dict[str, Any]], question: str
) -> list[dict[str, Any]]:
    """Apply only developer-owned semantic filters to SQL rows."""

    def searchable(row: dict[str, Any]) -> str:
        return " ".join(
            str(row.get(field) or "")
            for field in ("module", "course_name", "course_code", "department")
        )

    if PRACTICE_RE.search(question):
        records = [
            row
            for row in records
            if "实践" in str(row.get("module") or "")
            or any(term in str(row.get("course_name") or "") for term in ("实践", "实训", "实验", "实习", "论文"))
        ]
    if MATH_RE.search(question):
        records = [
            row
            for row in records
            if re.search(r"数学|代数|微积分|概率|统计", searchable(row))
        ]
    if PROGRAMMING_RE.search(question):
        records = [
            row
            for row in records
            if re.search(r"程序设计|编程|程序语言", searchable(row))
        ]
    if POLITICS_RE.search(question):
        records = [
            row
            for row in records
            if re.search(r"马克思|思想|政治|形势与政策|近现代史|毛泽东|习近平", searchable(row))
        ]
    if PE_RE.search(question):
        records = [
            row
            for row in records
            if str(row.get("course_code") or "").upper().startswith("PED")
            or "体育" in searchable(row)
        ]
    if MILITARY_RE.search(question):
        records = [
            row
            for row in records
            if str(row.get("course_code") or "").upper().startswith("MTT")
            or "军事" in searchable(row)
        ]
    if MAX_PRACTICE_RE.search(question) and records:
        maximum = max(float(row.get("practice_hours") or 0) for row in records)
        records = [
            row
            for row in records
            if float(row.get("practice_hours") or 0) == maximum
        ]
    return records


def _candidate_course_names(
    plan: QueryPlan, question: str, rows: list[dict[str, Any]]
) -> list[str]:
    values: list[str] = []
    if plan.course_name:
        values.extend(re.split(r"[、,，/；;]+", plan.course_name))
    target = _compact(question)
    for row in rows:
        name = _clean_course_name(str(row.get("course_name") or ""))
        if len(_compact(name)) >= 2 and _compact(name) in target:
            values.append(name)
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def _repair_evidence(
    records: list[dict[str, Any]], metadata_db: MetadataDB
) -> list[dict[str, Any]]:
    """Bind a course row to the matching chunk on its authoritative PDF page.

    Older full-book chunks can have a stale *chapter label* even though their
    page and table body are correct.  The relational record's source page is
    authoritative, so evidence is selected by document + physical page +
    course code/name instead of trusting a stale catalog chunk id.
    """

    repaired: list[dict[str, Any]] = []
    for original in records:
        record = dict(original)
        code = str(record.get("course_code") or "").strip()
        name = _clean_course_name(str(record.get("course_name") or ""))
        current = metadata_db.chunk(str(record.get("evidence_chunk_id") or ""))
        page_token = f"第{int(record['source_page'])}页"
        current_ok = bool(
            current
            and page_token in current.article
            and (not code or code in current.text)
            and (not name or _compact(name) in _compact(current.text))
        )
        if not current_ok:
            rows = metadata_db.connection.execute(
                """
                SELECT chunk_id, text FROM chunks
                WHERE enabled = 1 AND doc_title = ? AND article LIKE ?
                ORDER BY is_table DESC, embedding_row
                """,
                (str(record.get("doc_title") or ""), f"%{page_token}%"),
            ).fetchall()
            ranked = sorted(
                rows,
                key=lambda row: (
                    int(bool(code and code in row["text"])),
                    int(bool(name and _compact(name) in _compact(row["text"]))),
                    -len(row["text"]),
                ),
                reverse=True,
            )
            if ranked and (
                (code and code in ranked[0]["text"])
                or (name and _compact(name) in _compact(ranked[0]["text"]))
            ):
                record["evidence_chunk_id"] = str(ranked[0]["chunk_id"])
        repaired.append(record)
    return repaired


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


def _lines(records: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for record in records:
        name = _clean_course_name(str(record.get("course_name") or ""))
        code = str(record.get("course_code") or "未标注")
        credits = float(record.get("credits") or 0)
        semester = str(record.get("semester") or "未标注")
        nature = str(record.get("course_nature") or "未标注")
        module = str(record.get("module") or "未标注")
        hours = _hours(record)
        suffix = f"，{hours}" if hours else ""
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
                    records.extend(
                        db.courses(cohort=plan.cohort, major=major, code=name)
                    )
                else:
                    records.extend(
                        db.courses(cohort=plan.cohort, major=major, name=name)
                    )
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
    grounded = _ground(_lines(records), records, metadata_db)
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
