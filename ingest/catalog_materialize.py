"""Fail-closed adapter from reviewed catalog drafts to the canonical DB catalog.

The adapter is deliberately not an extractor.  It accepts only explicit
program/module scaffolds and explicit per-record evidence mappings, so it never
guesses a program, module, chunk ID, or field span from nearby text.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from copy import deepcopy
from typing import cast

from ingest.catalog import CATALOG_DRAFT_VERSION

ADAPTER_VERSION = "catalog-materialize/v1"
_COURSE_FIELDS = ("module", "credits", "semester", "nature")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CREDIT_SPAN_RE = re.compile(
    r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?:学分|credits?)?\s*$", re.IGNORECASE
)
_SEMESTER_SPAN_RE = re.compile(
    r"^\s*(?:第\s*)?(?P<value>\d{1,2})\s*(?:学期|semester|term)?\s*$", re.IGNORECASE
)
_TEXT_FIELD_LABELS = {
    "module": r"课程模块|模块名称|所属模块|模块|module",
    "nature": r"课程性质|课程属性|性质|nature",
}


class CatalogMaterializationError(ValueError):
    """Raised when a materialization input is structurally unsafe or strict-fails."""


class _CourseFailure(Exception):
    """A record-local failure which becomes an explicit quarantine row."""

    def __init__(self, reason: str, details: Mapping[str, object]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = dict(details)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise CatalogMaterializationError(f"{name} must be a JSON object")
    return {str(key): item for key, item in value.items()}


def _object_list(value: object, *, name: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise CatalogMaterializationError(f"{name} must be a list")
    values: list[dict[str, object]] = []
    for index, item in enumerate(value, start=1):
        values.append(_mapping(item, name=f"{name}[{index}]"))
    return values


def _required_text(row: Mapping[str, object], field: str, *, context: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise CatalogMaterializationError(f"{context}.{field} must be a non-empty string")
    return value.strip()


def _required_integer(row: Mapping[str, object], field: str, *, context: str) -> int:
    value = row.get(field)
    if isinstance(value, bool):
        raise CatalogMaterializationError(f"{context}.{field} must be a positive integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
    else:
        raise CatalogMaterializationError(f"{context}.{field} must be a positive integer")
    if parsed <= 0:
        raise CatalogMaterializationError(f"{context}.{field} must be a positive integer")
    return parsed


def _required_sha256(value: object, *, context: str) -> str:
    digest = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        raise CatalogMaterializationError(f"{context} must be a lowercase SHA-256 digest")
    return digest


def _normalised(value: object) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", str(value or "")).lower()


def _canonical_number(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _field_span_covers_value(field: str, value: object, span: str) -> bool:
    """Require a field-semantic witness, not a value-shaped substring.

    A number alone cannot prove its column semantics: ``3`` is both a possible
    credit and a possible semester.  Credits therefore accept only a pure
    number or an explicit credit unit; semesters only a pure number or an
    explicit semester unit.  Text fields require normalized equality, with a
    narrowly defined explicit field-label form for hand-authored ledgers.
    """

    normalized_value = _normalised(value)
    normalized_span = _normalised(span)
    if not normalized_value or not normalized_span:
        return False
    if field == "credits":
        match = _CREDIT_SPAN_RE.fullmatch(span)
        if match is None or not isinstance(value, (str, int, float)):
            return False
        try:
            return float(match.group("value")) == float(value)
        except (TypeError, ValueError):
            return False
    if field == "semester":
        expected = str(value).strip()
        if not expected.isdigit():
            return False
        match = _SEMESTER_SPAN_RE.fullmatch(span)
        return match is not None and match.group("value") == expected
    if normalized_value == normalized_span:
        return True
    label = _TEXT_FIELD_LABELS.get(field)
    if label is None:
        return False
    match = re.fullmatch(rf"\s*(?:{label})\s*[:：]\s*(?P<value>.+?)\s*", span, re.IGNORECASE)
    return match is not None and _normalised(match.group("value")) == normalized_value


def _input_courses(reviewed_draft: Mapping[str, object]) -> list[dict[str, object]]:
    courses = _object_list(reviewed_draft.get("courses", []), name="reviewed_draft.courses")
    identifiers: set[str] = set()
    for index, course in enumerate(courses, start=1):
        record_id = _required_text(course, "record_id", context=f"reviewed_draft.courses[{index}]")
        if record_id in identifiers:
            raise CatalogMaterializationError(
                f"reviewed_draft.courses has duplicate record_id: {record_id!r}"
            )
        identifiers.add(record_id)
    return courses


def _review_decisions(
    reviewed_draft: Mapping[str, object],
    courses: list[dict[str, object]],
    upstream_quarantine: list[dict[str, object]],
) -> dict[str, str]:
    """Authenticate active records against the append-only review ledger.

    ``review_status`` alone is intentionally insufficient: it is an output
    field and could otherwise be hand-authored without any reviewer event.
    A quarantined record may remain in the ledger, but it is never active.
    """

    schema_version = _required_text(
        reviewed_draft, "schema_version", context="reviewed_draft"
    )
    if schema_version != CATALOG_DRAFT_VERSION:
        raise CatalogMaterializationError(
            "reviewed_draft.schema_version must equal "
            f"{CATALOG_DRAFT_VERSION!r}, got {schema_version!r}"
        )
    entries = _object_list(
        reviewed_draft.get("review_ledger", []), name="reviewed_draft.review_ledger"
    )
    active_ids = {
        _required_text(course, "record_id", context="reviewed_draft.courses") for course in courses
    }
    upstream_ids = {
        str(item.get("record_id") or "").strip()
        for item in upstream_quarantine
        if str(item.get("record_id") or "").strip()
    }
    decisions: dict[str, str] = {}
    for position, entry in enumerate(entries, start=1):
        context = f"reviewed_draft.review_ledger[{position}]"
        record_id = _required_text(entry, "record_id", context=context)
        decision = _required_text(entry, "decision", context=context).lower()
        _required_text(entry, "reviewer", context=context)
        _required_text(entry, "reviewed_at", context=context)
        if decision not in {"approve", "edit", "quarantine"}:
            raise CatalogMaterializationError(f"{context}.decision is not a terminal review decision")
        if record_id not in active_ids | upstream_ids:
            raise CatalogMaterializationError(
                f"{context} references an unknown record_id: {record_id!r}"
            )
        if record_id in decisions:
            raise CatalogMaterializationError(
                f"reviewed_draft.review_ledger repeats terminal decision for {record_id!r}"
            )
        decisions[record_id] = decision
    return decisions


def _scaffold_plans(scaffold: Mapping[str, object]) -> tuple[list[dict[str, object]], dict[tuple[str, str, str], dict[str, object]]]:
    plans = _object_list(scaffold.get("plans"), name="plan_scaffold.plans")
    index: dict[tuple[str, str, str], dict[str, object]] = {}
    output: list[dict[str, object]] = []
    for position, plan in enumerate(plans, start=1):
        context = f"plan_scaffold.plans[{position}]"
        college = _required_text(plan, "college", context=context)
        cohort = _required_text(plan, "cohort", context=context)
        if not (cohort.isdigit() and len(cohort) == 4):
            raise CatalogMaterializationError(f"{context}.cohort must be a four-digit year")
        major = _required_text(plan, "major", context=context)
        _required_text(plan, "source_title", context=context)
        modules = _object_list(plan.get("modules"), name=f"{context}.modules")
        module_names: set[str] = set()
        for module_index, module in enumerate(modules, start=1):
            module_name = _required_text(
                module, "name", context=f"{context}.modules[{module_index}]"
            )
            if module_name in module_names:
                raise CatalogMaterializationError(
                    f"{context}.modules has duplicate module name: {module_name!r}"
                )
            module_names.add(module_name)
        key = (college, cohort, major)
        if key in index:
            raise CatalogMaterializationError(f"plan_scaffold.plans has duplicate scope: {key!r}")
        normalized = deepcopy(plan)
        normalized.update({"college": college, "cohort": cohort, "major": major, "modules": modules})
        index[key] = normalized
        output.append(normalized)
    if not output:
        raise CatalogMaterializationError("plan_scaffold.plans must not be empty")
    return output, index


def _index_records(
    payload: Mapping[str, object], *, key: str, context: str
) -> dict[str, dict[str, object]]:
    rows = _object_list(payload.get(key, []), name=f"{context}.{key}")
    values: dict[str, dict[str, object]] = {}
    for position, row in enumerate(rows, start=1):
        record_id = _required_text(row, "record_id", context=f"{context}.{key}[{position}]")
        if record_id in values:
            raise CatalogMaterializationError(
                f"{context}.{key} has duplicate record_id: {record_id!r}"
            )
        values[record_id] = row
    return values


def _course_source(
    course: Mapping[str, object], *, context: str
) -> tuple[str, str, dict[str, object]]:
    source = _mapping(course.get("source"), name=f"{context}.source")
    source_title = _required_text(source, "doc_title", context=f"{context}.source")
    source_sha256 = _required_sha256(
        source.get("source_sha256"), context=f"{context}.source.source_sha256"
    )
    return source_title, source_sha256, source


def _reviewed_field_status(course: Mapping[str, object], field: str) -> None:
    """Require the draft's own reviewer decision before evidence can promote it."""

    verification = _mapping(
        course.get("field_verification"), name="reviewed course.field_verification"
    )
    raw_status = verification.get(field)
    if isinstance(raw_status, str):
        status = raw_status.strip().lower()
    elif isinstance(raw_status, Mapping):
        status = str(
            raw_status.get("verification_status") or raw_status.get("status") or ""
        ).strip().lower()
    else:
        status = ""
    if status != "verified":
        raise _CourseFailure(
            "course_field_not_explicitly_approved",
            {"field": field, "verification_status": status or None},
        )


def _source_lineages(
    course: Mapping[str, object], field: str, *, context: str
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return immutable source locations and reviewer amendments separately.

    The extraction layer retains the physical cell in every field lineage.  A
    reviewer correction is intentionally a separate event, never a fabricated
    replacement location; materialization therefore keeps both in its audit
    trail and requires one of them to justify the final canonical value.
    """

    field_lineage = _mapping(course.get("field_lineage"), name=f"{context}.field_lineage")
    entries = field_lineage.get(field)
    if not isinstance(entries, list):
        raise _CourseFailure("missing_draft_field_lineage", {"field": field})
    physical: list[dict[str, object]] = []
    reviewer: list[dict[str, object]] = []
    for position, value in enumerate(entries, start=1):
        if not isinstance(value, Mapping):
            raise _CourseFailure(
                "invalid_draft_field_lineage", {"field": field, "entry": position}
            )
        entry = {str(key): item for key, item in value.items()}
        if entry.get("source_sha256"):
            physical.append(entry)
        elif entry.get("origin") == "reviewer":
            reviewer.append({"entry": position, **entry})
        else:
            raise _CourseFailure(
                "invalid_draft_field_lineage", {"field": field, "entry": position}
            )
    if not physical:
        raise _CourseFailure("missing_draft_field_lineage", {"field": field})
    return physical, reviewer


def _values_equivalent(left: object, right: object) -> bool:
    """Compare review values without treating 3 and 3.0 as different facts."""

    if (
        not isinstance(left, bool)
        and not isinstance(right, bool)
        and isinstance(left, (int, float))
        and isinstance(right, (int, float))
    ):
        return float(left) == float(right)
    return _normalised(left) == _normalised(right)


def _assignment(
    assignment: Mapping[str, object],
    *,
    course: Mapping[str, object],
    source: Mapping[str, object],
    plans: Mapping[tuple[str, str, str], Mapping[str, object]],
) -> dict[str, object]:
    record_id = _required_text(course, "record_id", context="reviewed course")
    context = f"plan_scaffold.course_assignments[{record_id}]"
    college = _required_text(assignment, "college", context=context)
    cohort = _required_text(assignment, "cohort", context=context)
    major = _required_text(assignment, "major", context=context)
    module = _required_text(assignment, "module", context=context)
    source_title = _required_text(assignment, "source_title", context=context)
    raw_department = assignment.get("department")
    if raw_department is None:
        department: str | None = None
    elif isinstance(raw_department, str):
        department = raw_department.strip() or None
    else:
        raise CatalogMaterializationError(f"{context}.department must be a string when supplied")
    source_college = _required_text(source, "college", context="reviewed course.source")
    source_cohort = _required_text(source, "cohort", context="reviewed course.source")
    source_doc_type = _required_text(source, "doc_type", context="reviewed course.source").lower()
    source_authenticity = _required_text(
        source, "source_authenticity_status", context="reviewed course.source"
    ).lower()
    if source_college != college or source_cohort != cohort:
        raise _CourseFailure(
            "source_scope_mismatch",
            {
                "source_college": source_college,
                "source_cohort": source_cohort,
                "assignment_college": college,
                "assignment_cohort": cohort,
            },
        )
    if source_doc_type not in {"curriculum", "course_catalog"}:
        raise _CourseFailure(
            "source_document_type_not_catalog", {"doc_type": source_doc_type}
        )
    if source_authenticity != "verified":
        raise _CourseFailure(
            "source_not_authenticity_verified",
            {"source_authenticity_status": source_authenticity},
        )
    plan = plans.get((college, cohort, major))
    if plan is None:
        raise _CourseFailure(
            "missing_plan_mapping",
            {"college": college, "cohort": cohort, "major": major},
        )
    plan_source_title = _required_text(plan, "source_title", context=f"{context}.resolved_plan")
    if source_title != plan_source_title:
        raise _CourseFailure(
            "assignment_source_title_not_plan",
            {
                "assignment_source_title": source_title,
                "plan_source_title": plan_source_title,
            },
        )
    modules = _object_list(plan.get("modules"), name=f"{context}.resolved_plan.modules")
    if module not in {_required_text(item, "name", context=f"{context}.resolved_plan.module") for item in modules}:
        raise _CourseFailure("assignment_module_not_in_plan", {"module": module})
    if str(course.get("module") or "").strip() != module:
        raise _CourseFailure(
            "assignment_module_mismatch",
            {"draft_module": course.get("module"), "assignment_module": module},
        )
    draft_title, _draft_sha, _source = _course_source(course, context="reviewed course")
    if draft_title != source_title:
        raise _CourseFailure(
            "assignment_source_title_mismatch",
            {"draft_source_title": draft_title, "assignment_source_title": source_title},
        )
    return {
        "college": college,
        "cohort": cohort,
        "major": major,
        "module": module,
        "source_title": source_title,
        "department": department,
    }


def _field_verification(
    *,
    course: Mapping[str, object],
    evidence_mapping: Mapping[str, object],
    source_sha256: str,
    source_title: str,
    context: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    mapping_title = _required_text(evidence_mapping, "source_title", context=context)
    if mapping_title != source_title:
        raise _CourseFailure(
            "evidence_source_title_mismatch",
            {"assignment_source_title": source_title, "evidence_source_title": mapping_title},
        )
    mapping_sha = _required_sha256(
        evidence_mapping.get("source_sha256"), context=f"{context}.source_sha256"
    )
    if mapping_sha != source_sha256:
        raise _CourseFailure(
            "evidence_source_hash_mismatch",
            {"draft_source_sha256": source_sha256, "mapping_source_sha256": mapping_sha},
        )
    evidence = _mapping(evidence_mapping.get("evidence"), name=f"{context}.evidence")
    chunk_id = _required_text(evidence, "chunk_id", context=f"{context}.evidence")
    page = _required_integer(evidence_mapping, "page", context=context)
    source_row = _required_integer(evidence_mapping, "source_row", context=context)
    fields = _mapping(evidence_mapping.get("fields"), name=f"{context}.fields")
    result: dict[str, object] = {}
    audit: dict[str, object] = {}
    for field in _COURSE_FIELDS:
        _reviewed_field_status(course, field)
        field_entry = fields.get(field)
        if not isinstance(field_entry, Mapping):
            raise _CourseFailure("evidence_mapping_missing_field", {"field": field})
        field_context = f"{context}.fields.{field}"
        verification_status = _required_text(
            field_entry, "verification_status", context=field_context
        ).lower()
        if verification_status != "verified":
            raise _CourseFailure(
                "evidence_field_not_verified",
                {"field": field, "verification_status": verification_status},
            )
        lineage = _mapping(field_entry.get("lineage"), name=f"{field_context}.lineage")
        lineage_sha = _required_sha256(
            lineage.get("source_sha256"), context=f"{field_context}.lineage.source_sha256"
        )
        lineage_chunk_id = _required_text(lineage, "chunk_id", context=f"{field_context}.lineage")
        lineage_page = _required_integer(lineage, "page", context=f"{field_context}.lineage")
        lineage_row = _required_integer(lineage, "row", context=f"{field_context}.lineage")
        cell = _required_text(lineage, "cell", context=f"{field_context}.lineage")
        span = _required_text(lineage, "span", context=f"{field_context}.lineage")
        if (
            lineage_sha != source_sha256
            or lineage_chunk_id != chunk_id
            or lineage_page != page
            or lineage_row != source_row
        ):
            raise _CourseFailure(
                "evidence_field_record_mismatch",
                {"field": field, "chunk_id": lineage_chunk_id, "page": lineage_page, "row": lineage_row},
            )
        physical_lineages, reviewer_lineages = _source_lineages(
            course, field, context="reviewed course"
        )
        matching_draft_lineages: list[dict[str, object]] = []
        physical_locations: set[tuple[str, int, int, str, str]] = set()
        for draft_lineage in physical_lineages:
            draft_context = f"reviewed course.field_lineage.{field}"
            draft_sha = _required_sha256(
                draft_lineage.get("source_sha256"), context=draft_context
            )
            draft_page = _required_integer(draft_lineage, "page", context=draft_context)
            draft_row = _required_integer(draft_lineage, "row", context=draft_context)
            draft_cell = _required_text(
                {"cell": str(draft_lineage.get("cell") or "")},
                "cell",
                context=draft_context,
            )
            physical_locations.add(
                (
                    draft_sha,
                    draft_page,
                    draft_row,
                    draft_cell,
                    _normalised(draft_lineage.get("raw_value")),
                )
            )
            if (
                draft_sha == lineage_sha
                and draft_page == lineage_page
                and draft_row == lineage_row
                and draft_cell == cell
            ):
                matching_draft_lineages.append(draft_lineage)
        if len(physical_locations) != 1:
            raise _CourseFailure(
                "ambiguous_draft_field_lineage",
                {
                    "field": field,
                    "locations": [
                        {
                            "source_sha256": source_hash,
                            "page": page_number,
                            "row": row_number,
                            "cell": cell_ref,
                            "raw_value": raw_value,
                        }
                        for source_hash, page_number, row_number, cell_ref, raw_value in sorted(
                            physical_locations
                        )
                    ],
                },
            )
        if not matching_draft_lineages:
            raise _CourseFailure(
                "evidence_field_lineage_mismatch",
                {
                    "field": field,
                    "draft_locations": [
                        {
                            "page": entry.get("page"),
                            "row": entry.get("row"),
                            "cell": entry.get("cell"),
                        }
                        for entry in physical_lineages
                    ],
                    "mapping": {"page": lineage_page, "row": lineage_row, "cell": cell},
                },
            )
        value = _canonical_number(course.get(field))
        if not _field_span_covers_value(field, value, span):
            raise _CourseFailure(
                "evidence_span_does_not_cover_field_value",
                {"field": field, "value": value, "span": span},
            )
        raw_source_covers_value = any(
            _field_span_covers_value(field, value, str(entry.get("raw_value") or ""))
            for entry in matching_draft_lineages
        )
        invalid_reviewer_entries = [
            entry
            for entry in reviewer_lineages
            if not _values_equivalent(entry.get("review_value"), value)
        ]
        if invalid_reviewer_entries:
            raise _CourseFailure(
                "reviewer_lineage_value_mismatch",
                {
                    "field": field,
                    "value": value,
                    "reviewer_values": [
                        entry.get("review_value") for entry in invalid_reviewer_entries
                    ],
                },
            )
        reviewer_confirms_value = any(
            _values_equivalent(entry.get("review_value"), value)
            for entry in reviewer_lineages
            if entry.get("origin") == "reviewer"
        )
        if not raw_source_covers_value and not reviewer_confirms_value:
            raise _CourseFailure(
                "draft_lineage_does_not_cover_field_value",
                {
                    "field": field,
                    "value": value,
                    "source_values": [entry.get("raw_value") for entry in matching_draft_lineages],
                    "reviewer_values": [entry.get("review_value") for entry in reviewer_lineages],
                },
            )
        result[field] = {
            "verification_status": "verified",
            "lineage": {
                "source_sha256": lineage_sha,
                "chunk_id": chunk_id,
                "page": lineage_page,
                "row": lineage_row,
                "cell": cell,
                "span": span,
            },
        }
        audit[field] = {
            "draft_lineages": matching_draft_lineages,
            "reviewer_lineages": reviewer_lineages,
            "evidence_lineage": cast(Mapping[str, object], result[field])["lineage"],
        }
    return {"chunk_id": chunk_id}, {
        "page": page,
        "source_row": source_row,
        "field_lineage": audit,
    }, result


def _materialize_course(
    *,
    course: Mapping[str, object],
    assignment: Mapping[str, object] | None,
    evidence_mapping: Mapping[str, object] | None,
    review_decision: str | None,
    plans: Mapping[tuple[str, str, str], Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    record_id = _required_text(course, "record_id", context="reviewed course")
    if str(course.get("review_status") or "").strip().lower() != "verified":
        raise _CourseFailure(
            "course_not_explicitly_approved",
            {"review_status": course.get("review_status")},
        )
    if assignment is None:
        raise _CourseFailure("missing_plan_mapping", {"kind": "assignment_missing"})
    if evidence_mapping is None:
        raise _CourseFailure("missing_evidence_mapping", {})
    if review_decision not in {"approve", "edit"}:
        raise _CourseFailure(
            "course_missing_review_ledger_entry", {"review_decision": review_decision}
        )
    try:
        source_title, source_sha256, source = _course_source(course, context="reviewed course")
        resolved_assignment = _assignment(
            assignment, course=course, source=source, plans=plans
        )
    except CatalogMaterializationError as exc:
        raise _CourseFailure("invalid_plan_or_source_mapping", {"error": str(exc)}) from exc
    evidence, audit, field_verification = _field_verification(
        course=course,
        evidence_mapping=evidence_mapping,
        source_sha256=source_sha256,
        source_title=source_title,
        context=f"evidence_mappings[{record_id}]",
    )
    code = _required_text(course, "code", context="reviewed course")
    name = _required_text(course, "name", context="reviewed course")
    nature = _required_text(course, "nature", context="reviewed course")
    semester = _required_text(course, "semester", context="reviewed course")
    credits = _canonical_number(course.get("credits"))
    if isinstance(credits, bool) or not isinstance(credits, (int, float)) or credits <= 0:
        raise _CourseFailure("invalid_course_credits", {"credits": course.get("credits")})
    materialized = {
        "code": code,
        "name": name,
        "credits": credits,
        "nature": nature,
        "semester": semester,
        "department": resolved_assignment["department"],
        "college": resolved_assignment["college"],
        "cohort": resolved_assignment["cohort"],
        "major": resolved_assignment["major"],
        "module": resolved_assignment["module"],
        "source_title": resolved_assignment["source_title"],
        "page": audit["page"],
        "source_row": audit["source_row"],
        "evidence": evidence,
        "field_verification": field_verification,
    }
    return materialized, {
        "record_id": record_id,
        "review_decision": review_decision,
        "assignment": resolved_assignment,
        "source_sha256": source_sha256,
        "evidence": evidence,
        "lineage": audit["field_lineage"],
    }


def materialize_catalog(
    reviewed_draft: Mapping[str, object],
    plan_scaffold: Mapping[str, object],
    evidence_mapping_ledger: Mapping[str, object],
    *,
    fail_on_quarantine: bool = False,
    input_file_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Compile explicitly reviewed courses into ``build_database`` catalog input.

    Required inputs are deliberately separate:

    * ``reviewed_draft`` is a ``catalog-draft/v1`` document emitted by the
      public table extractor and finalized by its append-only reviewer ledger;
    * ``plan_scaffold`` owns declared program scope and reviewed module
      requirements, plus an explicit ``course_assignments`` record for every
      active draft ``record_id``;
    * ``evidence_mapping_ledger`` owns one explicit chunk/page/row mapping and
      the four strict field witnesses for every active draft ``record_id``.

    No fallback looks at a chunk's text, title similarity, or another record's
    mapping.  Missing or conflicting record-local inputs are quarantined;
    malformed global ledgers fail before any caller can publish the result.

    Module requirements are intentionally copied only from ``plan_scaffold``.
    This adapter neither extracts nor reviews requirement totals; their
    evidence and field verification must arrive from a separate reviewed plan
    extraction before ``build_database`` can promote them.
    """

    draft = _mapping(reviewed_draft, name="reviewed_draft")
    scaffold = _mapping(plan_scaffold, name="plan_scaffold")
    ledger = _mapping(evidence_mapping_ledger, name="evidence_mapping_ledger")
    catalog_version = _required_text(scaffold, "catalog_version", context="plan_scaffold")
    plans, plan_index = _scaffold_plans(scaffold)
    assignments = _index_records(
        scaffold, key="course_assignments", context="plan_scaffold"
    )
    evidence_mappings = _index_records(
        ledger, key="mappings", context="evidence_mapping_ledger"
    )
    courses = _input_courses(draft)
    upstream_quarantine = _object_list(draft.get("quarantine", []), name="reviewed_draft.quarantine")
    review_decisions = _review_decisions(draft, courses, upstream_quarantine)
    active_course_ids = {
        _required_text(course, "record_id", context="reviewed course") for course in courses
    }
    upstream_record_ids = {
        str(item.get("record_id") or "").strip()
        for item in upstream_quarantine
        if str(item.get("record_id") or "").strip()
    }
    unknown_assignment_ids = set(assignments) - active_course_ids - upstream_record_ids
    unknown_mapping_ids = set(evidence_mappings) - active_course_ids - upstream_record_ids
    if unknown_assignment_ids:
        raise CatalogMaterializationError(
            "plan_scaffold.course_assignments has orphan record_id values: "
            + ", ".join(sorted(unknown_assignment_ids))
        )
    if unknown_mapping_ids:
        raise CatalogMaterializationError(
            "evidence_mapping_ledger.mappings has orphan record_id values: "
            + ", ".join(sorted(unknown_mapping_ids))
        )
    materialized: list[dict[str, object]] = []
    audit_records: list[dict[str, object]] = []
    quarantine: list[dict[str, object]] = []
    for course in courses:
        record_id = _required_text(course, "record_id", context="reviewed course")
        try:
            canonical, audit = _materialize_course(
                course=course,
                assignment=assignments.get(record_id),
                evidence_mapping=evidence_mappings.get(record_id),
                review_decision=review_decisions.get(record_id),
                plans=plan_index,
            )
        except _CourseFailure as exc:
            quarantine.append(
                {
                    "record_id": record_id,
                    "reason": exc.reason,
                    "details": exc.details,
                    "review_status": course.get("review_status"),
                }
            )
            continue
        materialized.append(canonical)
        audit_records.append(audit)

    counts = {
        "input_courses": len(courses),
        "materialized_courses": len(materialized),
        "quarantined_courses": len(quarantine),
        "upstream_quarantine_records": len(upstream_quarantine),
        "assignment_records": len(assignments),
        "evidence_mapping_records": len(evidence_mappings),
        "assignment_records_for_upstream_quarantine": len(
            set(assignments) & upstream_record_ids
        ),
        "evidence_mapping_records_for_upstream_quarantine": len(
            set(evidence_mappings) & upstream_record_ids
        ),
    }
    if counts["input_courses"] != counts["materialized_courses"] + counts["quarantined_courses"]:
        raise CatalogMaterializationError(f"course count conservation violated: {counts!r}")
    if fail_on_quarantine and (quarantine or upstream_quarantine):
        details = ", ".join(
            f"{item.get('record_id', '<upstream>')}:{item.get('reason', 'upstream_quarantine')}"
            for item in [*quarantine, *upstream_quarantine]
        )
        raise CatalogMaterializationError("materialization blocked by quarantine: " + details)

    semantic_hashes = {
        "reviewed_draft_sha256": _hash(draft),
        "review_ledger_sha256": _hash(draft.get("review_ledger", [])),
        "plan_scaffold_sha256": _hash(scaffold),
        "evidence_mapping_sha256": _hash(ledger),
    }
    file_hashes = {
        str(key): _required_sha256(value, context=f"input_file_hashes.{key}")
        for key, value in (input_file_hashes or {}).items()
    }
    return {
        "catalog_version": catalog_version,
        "plans": plans,
        "courses": materialized,
        "materialization": {
            "adapter_version": ADAPTER_VERSION,
            "input_hashes": semantic_hashes,
            "input_file_hashes": file_hashes,
            "counts": counts,
            "records": audit_records,
            "quarantine": quarantine,
            "upstream_quarantine": upstream_quarantine,
            "module_requirements_boundary": {
                "owner": "plan_scaffold",
                "message": "Module requirements are copied from the explicit scaffold; this adapter does not extract or verify requirement totals.",
            },
        },
    }


__all__ = ["ADAPTER_VERSION", "CatalogMaterializationError", "materialize_catalog"]
