"""Review-gated extraction of curriculum-course drafts from parsed tables.

This module deliberately stops before the production catalog/database boundary.
It turns parser-owned Markdown table cells into a schema-shaped, traceable
*draft*, quarantines every unknown or ambiguous row, and only an explicit
review ledger may promote a record to ``verified``.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import cast

from ingest.models import DocumentElement, ParsedDocument, SourceRecord
from ingest.parse import extraction_quality_ledger, normalize_text

CATALOG_DRAFT_VERSION = "catalog-draft/v1"
_COURSE_FIELDS = ("code", "name", "credits", "semester", "nature", "module")
_CATALOG_DOC_TYPES = frozenset({"curriculum", "course_catalog"})
_CREDIT_RE = re.compile(r"^(?P<value>\d+(?:\.\d+)?)\s*(?:学分)?$", re.IGNORECASE)
_SEMESTER_RE = re.compile(r"^(?:第\s*)?(?P<value>\d{1,2})(?:\s*学期)?$")
_CODE_RE = re.compile(r"^[A-Za-z0-9_-]{2,64}$")
_MARKDOWN_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")


@dataclass(frozen=True)
class _FieldSpec:
    name: str
    aliases: frozenset[str]


def _header_key(value: str) -> str:
    return re.sub(r"[\s_\-—–/()（）【】\[\]：:]+", "", normalize_text(value).lower())


_COURSE_SCHEMA = (
    _FieldSpec(
        "code",
        frozenset(
            _header_key(value)
            for value in ("课程代码", "课程编号", "课程号", "课程序号", "course code", "code")
        ),
    ),
    _FieldSpec(
        "name",
        frozenset(
            _header_key(value)
            for value in ("课程名称", "课程名", "course name", "course title", "name")
        ),
    ),
    _FieldSpec(
        "credits",
        frozenset(_header_key(value) for value in ("学分", "课程学分", "credits", "credit")),
    ),
    _FieldSpec(
        "semester",
        frozenset(
            _header_key(value)
            for value in ("开课学期", "建议修读学期", "修读学期", "学期", "semester", "term")
        ),
    ),
    _FieldSpec(
        "nature",
        frozenset(_header_key(value) for value in ("课程性质", "课程属性", "性质", "nature")),
    ),
    _FieldSpec(
        "module",
        frozenset(
            _header_key(value) for value in ("课程模块", "模块", "模块名称", "所属模块", "module")
        ),
    ),
)


class CatalogExtractionError(ValueError):
    """Raised when catalog extraction cannot safely produce a draft."""


class CatalogReviewError(ValueError):
    """Raised when an append-only catalog review entry is invalid."""


@dataclass(frozen=True)
class _Cell:
    text: str
    char_start: int
    char_end: int
    row: int
    cell: int


@dataclass(frozen=True)
class _Table:
    element_index: int
    table_index: int
    page: int | None
    headers: tuple[_Cell, ...]
    rows: tuple[tuple[_Cell, ...], ...]


def _markdown_cells(line: str, *, offset: int, row: int) -> tuple[_Cell, ...]:
    """Parse a standard Markdown row while preserving source-text offsets."""

    boundaries = [0]
    escaped = False
    for index, character in enumerate(line):
        if character == "|" and not escaped:
            boundaries.append(index + 1)
        escaped = character == "\\" and not escaped
        if character != "\\":
            escaped = False
    if boundaries[-1] != len(line):
        boundaries.append(len(line))

    values: list[tuple[int, int]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        values.append((start, end - 1 if end > start and line[end - 1] == "|" else end))
    if line.lstrip().startswith("|") and values:
        values = values[1:]

    cells: list[_Cell] = []
    for cell_number, (start, end) in enumerate(values, start=1):
        raw = line[start:end]
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        normalized = normalize_text(raw.strip().replace("\\|", "|").replace("\\\\", "\\"))
        cells.append(
            _Cell(
                text=normalized,
                char_start=offset + start + left,
                char_end=offset + start + right,
                row=row,
                cell=cell_number,
            )
        )
    return tuple(cells)


def _table_from_markdown(
    element: DocumentElement, *, element_index: int, table_index: int
) -> _Table | None:
    lines: list[tuple[int, str]] = []
    offset = 0
    for raw_line in element.text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        if line.strip().startswith("|"):
            lines.append((offset, line))
        offset += len(raw_line)
    if len(lines) < 2:
        return None
    headers = _markdown_cells(lines[0][1], offset=lines[0][0], row=1)
    separator = _markdown_cells(lines[1][1], offset=lines[1][0], row=2)
    if (
        not headers
        or len(headers) != len(separator)
        or not all(
            _MARKDOWN_SEPARATOR_RE.fullmatch(cell.text.replace(" ", "")) for cell in separator
        )
    ):
        return None
    rows: list[tuple[_Cell, ...]] = []
    for row_number, (line_offset, line) in enumerate(lines[2:], start=3):
        cells = _markdown_cells(line, offset=line_offset, row=row_number)
        if cells:
            rows.append(cells)
    return _Table(
        element_index=element_index,
        table_index=table_index,
        page=element.page,
        headers=headers,
        rows=tuple(rows),
    )


def _source_context(source: SourceRecord) -> dict[str, object]:
    return {
        "source_file": source.file,
        "source_sha256": source.source_sha256,
        "source_authenticity_status": source.authenticity_status,
        "doc_title": source.doc_title,
        "doc_type": source.doc_type,
        "cohort": source.cohort,
        "college": source.college,
    }


def _lineage(
    cell: _Cell,
    *,
    source: SourceRecord,
    table: _Table,
    field: str,
) -> dict[str, object]:
    return {
        **_source_context(source),
        "field": field,
        "element_index": table.element_index,
        "page": table.page,
        "table": table.table_index,
        "row": cell.row,
        "cell": cell.cell,
        "char_span": {
            "start": cell.char_start,
            "end": cell.char_end,
            "coordinate_system": "unicode_codepoint_in_table_markdown",
        },
        "raw_value": cell.text,
    }


def _quarantine(
    *,
    reason: str,
    source: SourceRecord,
    table: _Table | None = None,
    row: tuple[_Cell, ...] | None = None,
    details: Mapping[str, object] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {"reason": reason, **_source_context(source)}
    if table is not None:
        result.update(
            {
                "element_index": table.element_index,
                "page": table.page,
                "table": table.table_index,
            }
        )
    if row is not None:
        result["row"] = row[0].row if row else None
        result["raw_row"] = [cell.text for cell in row]
        result["row_lineage"] = [
            _lineage(cell, source=source, table=table, field="raw")
            for cell in row
            if table is not None
        ]
    if details:
        result["details"] = dict(details)
    return result


def _column_map(table: _Table) -> tuple[dict[str, int], list[dict[str, object]]]:
    by_field: dict[str, list[int]] = defaultdict(list)
    for index, header in enumerate(table.headers):
        key = _header_key(header.text)
        for spec in _COURSE_SCHEMA:
            if key in spec.aliases:
                by_field[spec.name].append(index)
    mapping: dict[str, int] = {}
    issues: list[dict[str, object]] = []
    for spec in _COURSE_SCHEMA:
        positions = by_field.get(spec.name, [])
        if not positions:
            issues.append({"reason": "missing_required_column", "field": spec.name})
        elif len(positions) > 1:
            issues.append(
                {
                    "reason": "ambiguous_column",
                    "field": spec.name,
                    "header_cells": [position + 1 for position in positions],
                }
            )
        else:
            mapping[spec.name] = positions[0]
    return mapping, issues


def _normalise_field(field: str, value: str) -> object:
    text = normalize_text(value)
    if not text:
        raise ValueError("empty_value")
    if field == "code":
        code = re.sub(r"\s+", "", text).upper()
        if not _CODE_RE.fullmatch(code):
            raise ValueError("invalid_course_code")
        return code
    if field == "name":
        if len(text) > 200:
            raise ValueError("course_name_too_long")
        return text
    if field == "credits":
        match = _CREDIT_RE.fullmatch(text)
        if match is None:
            raise ValueError("invalid_credits")
        credits = float(match.group("value"))
        if not 0 < credits <= 50:
            raise ValueError("credits_out_of_range")
        return credits
    if field == "semester":
        match = _SEMESTER_RE.fullmatch(text)
        if match is None:
            raise ValueError("invalid_semester")
        semester = int(match.group("value"))
        if not 1 <= semester <= 16:
            raise ValueError("semester_out_of_range")
        return str(semester)
    if field in {"nature", "module"}:
        if len(text) > 200:
            raise ValueError(f"{field}_too_long")
        return text
    raise ValueError(f"unsupported_field:{field}")


def _quality_for_page(
    quality_ledger: Iterable[Mapping[str, object]], page: int | None
) -> dict[str, object]:
    fallback: dict[str, object] | None = None
    for entry in quality_ledger:
        value = dict(entry)
        if value.get("page") == page:
            return value
        if value.get("page") is None:
            fallback = value
    return fallback or {
        "page": page,
        "status": "review_required",
        "table_status": "not_recorded",
        "ocr_status": "not_recorded",
        "critical": False,
        "issues": ["quality_ledger_missing"],
        "warnings": [],
    }


def _critical_table_failures(
    quality_ledger: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    return [
        dict(entry)
        for entry in quality_ledger
        if entry.get("table_status") == "failed" and bool(entry.get("critical"))
    ]


def _course_record(
    *,
    source: SourceRecord,
    table: _Table,
    row: tuple[_Cell, ...],
    columns: Mapping[str, int],
    quality: Mapping[str, object],
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    if len(row) != len(table.headers):
        return None, _quarantine(
            reason="ragged_table_row",
            source=source,
            table=table,
            row=row,
            details={"header_count": len(table.headers), "cell_count": len(row)},
        )
    values: dict[str, object] = {}
    field_lineage: dict[str, list[dict[str, object]]] = {}
    for field in _COURSE_FIELDS:
        cell = row[columns[field]]
        try:
            values[field] = _normalise_field(field, cell.text)
        except ValueError as exc:
            return None, _quarantine(
                reason=str(exc),
                source=source,
                table=table,
                row=row,
                details={
                    "field": field,
                    "lineage": _lineage(cell, source=source, table=table, field=field),
                },
            )
        field_lineage[field] = [_lineage(cell, source=source, table=table, field=field)]
    identifier_payload = "|".join(
        [source.file, str(table.page), str(table.table_index), str(row[0].row), str(values["code"])]
    )
    record_id = "course_" + sha256(identifier_payload.encode("utf-8")).hexdigest()[:16]
    return (
        {
            "record_id": record_id,
            **values,
            "source": _source_context(source),
            "table_location": {
                "element_index": table.element_index,
                "page": table.page,
                "table": table.table_index,
                "row": row[0].row,
            },
            "field_lineage": field_lineage,
            "extraction_quality": str(quality.get("status") or "review_required"),
            "extraction_warnings": [
                str(warning)
                for warning in cast(list[object], quality.get("warnings", []))
                if isinstance(warning, str) and warning.strip()
            ],
            "review_status": "review_required",
            "field_verification": {field: "review_required" for field in _COURSE_FIELDS},
        },
        None,
    )


def _quarantine_conflicts(
    courses: list[dict[str, object]], *, source: SourceRecord
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Quarantine every same-code row whose canonical values disagree."""

    by_code: dict[str, list[dict[str, object]]] = defaultdict(list)
    for course in courses:
        by_code[str(course["code"])].append(course)
    accepted: list[dict[str, object]] = []
    quarantined: list[dict[str, object]] = []
    for code, grouped in by_code.items():
        if len(grouped) == 1:
            accepted.extend(grouped)
            continue
        conflicts = [
            field
            for field in ("name", "credits", "semester", "nature", "module")
            if len(
                {
                    json.dumps(course[field], ensure_ascii=False, sort_keys=True)
                    for course in grouped
                }
            )
            > 1
        ]
        if not conflicts:
            accepted.extend(grouped)
            continue
        for course in grouped:
            quarantined.append(
                {
                    "reason": "conflicting_duplicate_course",
                    **_source_context(source),
                    "record_id": course["record_id"],
                    "code": code,
                    "conflict_fields": conflicts,
                    "table_location": course["table_location"],
                    "field_lineage": course["field_lineage"],
                }
            )
    return accepted, quarantined


def _elements_and_quality(
    document: ParsedDocument | Iterable[DocumentElement],
    *,
    source: SourceRecord,
    quality_ledger: Iterable[Mapping[str, object]] | None,
) -> tuple[list[DocumentElement], list[dict[str, object]]]:
    if isinstance(document, ParsedDocument):
        elements = list(document.elements)
        generated_quality = extraction_quality_ledger(document)
    else:
        elements = list(document)
        generated_quality = extraction_quality_ledger(
            ParsedDocument(path=Path(source.file), elements=elements)
        )
    return elements, [dict(entry) for entry in (quality_ledger or generated_quality)]


def extract_catalog_draft(
    document: ParsedDocument | Iterable[DocumentElement],
    *,
    source: SourceRecord,
    quality_ledger: Iterable[Mapping[str, object]] | None = None,
    fail_on_critical_table_failure: bool = True,
) -> dict[str, object]:
    """Extract a schema-shaped, non-verified curriculum catalog draft.

    There is intentionally no path from parser output to a verified record in
    this function.  Ambiguous headers, malformed rows, invalid values, and
    conflicting duplicate courses enter ``quarantine`` with their source
    coordinates intact.
    """

    elements, quality = _elements_and_quality(
        document, source=source, quality_ledger=quality_ledger
    )
    if source.doc_type not in _CATALOG_DOC_TYPES:
        table_count = sum(element.kind == "table" for element in elements)
        type_quarantine = [
            {
                "reason": "source_document_type_not_catalog",
                **_source_context(source),
                "details": {"allowed_doc_types": sorted(_CATALOG_DOC_TYPES)},
            }
        ]
        return {
            "schema_version": CATALOG_DRAFT_VERSION,
            "review_status": "review_required",
            "source": _source_context(source),
            "quality_ledger": quality,
            "courses": [],
            "quarantine": type_quarantine,
            "counts": {
                "table_count": table_count,
                "course_draft_count": 0,
                "quarantine_count": len(type_quarantine),
                "verified_course_count": 0,
            },
            "course_schema": list(_COURSE_FIELDS),
        }
    critical_failures = _critical_table_failures(quality)
    if critical_failures and fail_on_critical_table_failure:
        pages = ", ".join(str(item.get("page")) for item in critical_failures)
        raise CatalogExtractionError(
            "catalog extraction blocked by critical table extraction failure on page(s): " + pages
        )

    courses: list[dict[str, object]] = []
    quarantine: list[dict[str, object]] = []
    table_index = 0
    for element_index, element in enumerate(elements, start=1):
        if element.kind != "table":
            continue
        table_index += 1
        table = _table_from_markdown(element, element_index=element_index, table_index=table_index)
        if table is None:
            quarantine.append(
                _quarantine(
                    reason="malformed_markdown_table",
                    source=source,
                    details={
                        "element_index": element_index,
                        "page": element.page,
                        "table": table_index,
                    },
                )
            )
            continue
        columns, issues = _column_map(table)
        if issues:
            quarantine.append(
                _quarantine(
                    reason="unmapped_or_ambiguous_columns",
                    source=source,
                    table=table,
                    details={"issues": issues, "headers": [cell.text for cell in table.headers]},
                )
            )
            continue
        page_quality = _quality_for_page(quality, table.page)
        for row in table.rows:
            course, rejected = _course_record(
                source=source,
                table=table,
                row=row,
                columns=columns,
                quality=page_quality,
            )
            if course is not None:
                courses.append(course)
            if rejected is not None:
                quarantine.append(rejected)
    courses, conflicts = _quarantine_conflicts(courses, source=source)
    quarantine.extend(conflicts)
    return {
        "schema_version": CATALOG_DRAFT_VERSION,
        "review_status": "review_required",
        "source": _source_context(source),
        "quality_ledger": quality,
        "courses": courses,
        "quarantine": quarantine,
        "counts": {
            "table_count": table_index,
            "course_draft_count": len(courses),
            "quarantine_count": len(quarantine),
            "verified_course_count": 0,
        },
        "course_schema": list(_COURSE_FIELDS),
    }


def combine_catalog_drafts(drafts: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Combine source drafts without changing their review state or lineage."""

    values = [dict(draft) for draft in drafts]
    courses = [course for draft in values for course in cast(list[object], draft.get("courses", []))]
    quarantine = [item for draft in values for item in cast(list[object], draft.get("quarantine", []))]
    source_quality = [
        {"source": draft.get("source"), "entries": draft.get("quality_ledger", [])}
        for draft in values
    ]
    return {
        "schema_version": CATALOG_DRAFT_VERSION,
        "review_status": "review_required",
        "sources": [draft.get("source") for draft in values],
        "quality_ledger": source_quality,
        "courses": courses,
        "quarantine": quarantine,
        "counts": {
            "source_count": len(values),
            "course_draft_count": len(courses),
            "quarantine_count": len(quarantine),
            "verified_course_count": 0,
        },
        "course_schema": list(_COURSE_FIELDS),
    }


def _require_review_text(entry: Mapping[str, object], field: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise CatalogReviewError(f"review entry {field} must be a non-empty string")
    return value.strip()


def _reviewed_at(value: str) -> str:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CatalogReviewError("review entry reviewed_at must be ISO-8601") from exc
    return value


def _ensure_no_review_conflicts(courses: Iterable[dict[str, object]]) -> None:
    """Fail closed when a reviewer edit would recreate a quarantined conflict."""

    by_code: dict[str, list[dict[str, object]]] = defaultdict(list)
    for course in courses:
        by_code[str(course.get("code") or "")].append(course)
    for code, grouped in by_code.items():
        if not code or len(grouped) < 2:
            continue
        conflict_fields = [
            field
            for field in ("name", "credits", "semester", "nature", "module")
            if len(
                {
                    json.dumps(course.get(field), ensure_ascii=False, sort_keys=True)
                    for course in grouped
                }
            )
            > 1
        ]
        if conflict_fields:
            record_ids = ", ".join(sorted(str(course.get("record_id")) for course in grouped))
            raise CatalogReviewError(
                "review ledger creates conflicting course code "
                f"{code} for records {record_ids}: {', '.join(conflict_fields)}"
            )


def apply_review_ledger(
    draft: Mapping[str, object], ledger: Iterable[Mapping[str, object]]
) -> dict[str, object]:
    """Apply explicit reviewer decisions and emit field-level diffs.

    Review entries are processed in supplied order and retained verbatim in a
    normalized append-only ledger.  A reviewer must explicitly approve or edit
    a record before it becomes ``verified``; untouched drafts remain
    ``review_required``.
    """

    result = cast(dict[str, object], json.loads(json.dumps(dict(draft), ensure_ascii=False)))
    courses = result.get("courses")
    if not isinstance(courses, list):
        raise CatalogReviewError("catalog draft courses must be a list")
    by_id = {
        str(course.get("record_id")): course
        for course in courses
        if isinstance(course, dict) and isinstance(course.get("record_id"), str)
    }
    if len(by_id) != len(courses):
        raise CatalogReviewError("catalog draft courses need unique record_id values")
    review_entries: list[dict[str, object]] = []
    diffs: list[dict[str, object]] = []
    reviewed_out: set[str] = set()
    quarantine = result.setdefault("quarantine", [])
    if not isinstance(quarantine, list):
        raise CatalogReviewError("catalog draft quarantine must be a list")

    for raw_entry in ledger:
        entry = dict(raw_entry)
        record_id = _require_review_text(entry, "record_id")
        reviewer = _require_review_text(entry, "reviewer")
        reviewed_at = _reviewed_at(_require_review_text(entry, "reviewed_at"))
        decision = _require_review_text(entry, "decision").lower()
        if decision not in {"approve", "edit", "quarantine"}:
            raise CatalogReviewError("review entry decision must be approve, edit, or quarantine")
        course = by_id.get(record_id)
        if course is None:
            raise CatalogReviewError(f"review entry references unknown record_id: {record_id}")
        if record_id in reviewed_out:
            raise CatalogReviewError(
                f"review entry repeats terminal decision for record_id: {record_id}"
            )
        if str(course.get("extraction_quality")) == "failed" and decision != "quarantine":
            raise CatalogReviewError("failed extraction quality may only be quarantined")

        updates_raw = entry.get("field_updates", {})
        if not isinstance(updates_raw, Mapping):
            raise CatalogReviewError("review entry field_updates must be an object")
        unknown_fields = sorted(set(updates_raw) - set(_COURSE_FIELDS))
        if unknown_fields:
            raise CatalogReviewError(
                "review entry has unsupported field_updates: " + ", ".join(unknown_fields)
            )
        if decision == "approve" and updates_raw:
            raise CatalogReviewError("approve entries must not include field_updates; use edit")
        if decision == "edit" and not updates_raw:
            raise CatalogReviewError("edit entries require at least one field_update")

        changes: list[dict[str, object]] = []
        if decision == "edit":
            lineage = course.get("field_lineage")
            if not isinstance(lineage, dict):
                raise CatalogReviewError("course draft lacks field_lineage")
            for field, raw_value in updates_raw.items():
                if not isinstance(raw_value, str):
                    raise CatalogReviewError(f"review field_updates.{field} must be a string")
                before = course[field]
                try:
                    after = _normalise_field(str(field), raw_value)
                except ValueError as exc:
                    raise CatalogReviewError(
                        f"review field_updates.{field} is invalid: {exc}"
                    ) from exc
                course[field] = after
                field_events = lineage.get(field)
                if not isinstance(field_events, list):
                    raise CatalogReviewError(f"course draft lacks {field} lineage")
                field_events.append(
                    {
                        "origin": "reviewer",
                        "reviewer": reviewer,
                        "reviewed_at": reviewed_at,
                        "prior_value": before,
                        "review_value": after,
                    }
                )
                changes.append({"field": field, "before": before, "after": after})

        normalized_entry: dict[str, object] = {
            "record_id": record_id,
            "decision": decision,
            "reviewer": reviewer,
            "reviewed_at": reviewed_at,
            "field_updates": dict(updates_raw),
        }
        if decision == "quarantine":
            course["review_status"] = "quarantined"
            quarantine.append(
                {
                    "reason": "reviewer_quarantine",
                    "record_id": record_id,
                    "reviewer": reviewer,
                    "reviewed_at": reviewed_at,
                    "course": course,
                }
            )
            reviewed_out.add(record_id)
        else:
            course["review_status"] = "verified"
            course["field_verification"] = {field: "verified" for field in _COURSE_FIELDS}
            reviewed_out.add(record_id)
        review_entries.append(normalized_entry)
        diffs.append(
            {
                "record_id": record_id,
                "decision": decision,
                "reviewer": reviewer,
                "reviewed_at": reviewed_at,
                "changes": changes,
            }
        )

    active_courses: list[dict[str, object]] = [
        course
        for course in courses
        if isinstance(course, dict) and course.get("review_status") != "quarantined"
    ]
    result["courses"] = active_courses
    _ensure_no_review_conflicts(active_courses)
    result["review_ledger"] = review_entries
    result["review_diff"] = diffs
    counts = result.setdefault("counts", {})
    if not isinstance(counts, dict):
        raise CatalogReviewError("catalog draft counts must be an object")
    counts["course_draft_count"] = len(active_courses)
    counts["quarantine_count"] = len(quarantine)
    counts["verified_course_count"] = sum(
        course.get("review_status") == "verified"
        for course in active_courses
        if isinstance(course, dict)
    )
    result["review_status"] = (
        "verified"
        if active_courses and counts["verified_course_count"] == len(active_courses)
        else "review_required"
    )
    return result


def load_review_ledger(path: str | Path) -> list[dict[str, object]]:
    """Read an append-only JSONL review ledger with no implicit defaults."""

    ledger_path = Path(path)
    if not ledger_path.is_file():
        raise FileNotFoundError(ledger_path)
    entries: list[dict[str, object]] = []
    for line_number, line in enumerate(
        ledger_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CatalogReviewError(f"review ledger line {line_number} is not valid JSON") from exc
        if not isinstance(value, dict):
            raise CatalogReviewError(f"review ledger line {line_number} must be an object")
        entries.append(dict(value))
    return entries


__all__ = [
    "CATALOG_DRAFT_VERSION",
    "CatalogExtractionError",
    "CatalogReviewError",
    "apply_review_ledger",
    "combine_catalog_drafts",
    "extract_catalog_draft",
    "load_review_ledger",
]
