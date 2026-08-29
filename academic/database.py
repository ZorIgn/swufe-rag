"""Canonical, read-only academic repository with entity aliases and provenance.

The builder converts the versioned curriculum catalog and source registry into a
single SQLite projection.  Runtime queries use fixed parameterized statements;
no model-generated SQL is ever accepted.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any, Literal, cast

from evidence.provenance import PARSER_VERSION, stable_id
from ingest.contracts import (
    DOC_TYPES,
    EXTRACTION_QUALITY_STATUSES,
    POLICY_DOCUMENT_TYPES,
    POLICY_TOPICS,
    SOURCE_AUTHENTICITY_STATUSES,
)
from query.schemas import ResolvedEntity
from retrieval.scope import policy_scope_matches

ROOT = Path(__file__).parents[1]
DEFAULT_DATABASE = ROOT / "data" / "academic.sqlite3"
DEFAULT_CATALOG = ROOT / "data" / "curriculum_catalog.json"
DEFAULT_SOURCES = ROOT / "data" / "sources.csv"
DEFAULT_CHUNKS = ROOT / "data" / "chunks.jsonl"
DEFAULT_ALIAS_CONFIG = ROOT / "config" / "entity_aliases.json"
DEFAULT_SOURCE_REVIEW = ROOT / "data" / "source_review.csv"
DEFAULT_EVIDENCE_REVIEW = ROOT / "data" / "evidence_review.csv"


SCHEMA_VERSION = "2"
REVIEW_STATUSES = frozenset({"verified", "review_required", "unverified"})
VERIFIED_SOURCE_REVIEW_DECISIONS = frozenset(
    {"include", "include_ocr", "include_converted", "include_split"}
)
VERIFIED_EVIDENCE_REVIEW_DECISIONS = frozenset({"verified", "include"})
STRICT_COURSE_FIELDS = ("module", "credits", "semester", "nature")
STRICT_REQUIREMENT_FIELDS = ("module", "required_credits")


class DataIntegrityError(ValueError):
    """Raised when released catalog data cannot be mapped unambiguously."""


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE sources (
  source_id TEXT PRIMARY KEY, title TEXT NOT NULL, level TEXT NOT NULL,
  college_id TEXT NOT NULL, cohort TEXT NOT NULL, authority_level INTEGER NOT NULL,
  published_at TEXT, effective_from TEXT, effective_to TEXT, supersedes_source_id TEXT,
  status TEXT NOT NULL, page_url TEXT, file_url TEXT, source_sha256 TEXT,
  collected_at TEXT, UNIQUE(title, cohort, file_url)
);
CREATE TABLE source_taxonomy (
  source_id TEXT PRIMARY KEY REFERENCES sources(source_id),
  doc_type TEXT NOT NULL,
  topics_json TEXT NOT NULL
);
CREATE INDEX idx_source_taxonomy_type ON source_taxonomy(doc_type);
CREATE TABLE source_authenticity (
  source_id TEXT PRIMARY KEY REFERENCES sources(source_id),
  declared_sha256 TEXT,
  observed_sha256 TEXT,
  authenticity_status TEXT NOT NULL,
  review_decision TEXT
);
CREATE INDEX idx_source_authenticity_status ON source_authenticity(authenticity_status);
CREATE TABLE programs (
  program_id TEXT PRIMARY KEY, canonical_name TEXT NOT NULL, college_id TEXT NOT NULL,
  cohort INTEGER NOT NULL, source_id TEXT NOT NULL REFERENCES sources(source_id),
  UNIQUE(canonical_name, college_id, cohort)
);
CREATE TABLE program_aliases (
  alias TEXT NOT NULL, normalized_alias TEXT NOT NULL, program_id TEXT NOT NULL REFERENCES programs(program_id),
  PRIMARY KEY(alias, program_id)
);
CREATE INDEX idx_program_alias_normalized ON program_aliases(normalized_alias);
CREATE TABLE modules (
  module_id TEXT PRIMARY KEY, program_id TEXT NOT NULL REFERENCES programs(program_id),
  canonical_name TEXT NOT NULL, UNIQUE(program_id, canonical_name)
);
CREATE TABLE module_aliases (
  alias TEXT NOT NULL, normalized_alias TEXT NOT NULL, module_id TEXT NOT NULL REFERENCES modules(module_id),
  PRIMARY KEY(alias, module_id)
);
CREATE TABLE courses (
  course_id TEXT PRIMARY KEY, canonical_code TEXT, canonical_name TEXT NOT NULL,
  UNIQUE(canonical_code, canonical_name)
);
CREATE TABLE course_aliases (
  alias TEXT NOT NULL, normalized_alias TEXT NOT NULL, course_id TEXT NOT NULL REFERENCES courses(course_id),
  PRIMARY KEY(alias, course_id)
);
CREATE INDEX idx_course_alias_normalized ON course_aliases(normalized_alias);
CREATE TABLE source_sections (
  chunk_id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(source_id),
  article TEXT, text TEXT NOT NULL, physical_page INTEGER, is_table INTEGER NOT NULL,
  parser_version TEXT NOT NULL, extracted_at TEXT NOT NULL, confidence REAL NOT NULL,
  review_status TEXT NOT NULL
);
CREATE INDEX idx_section_source ON source_sections(source_id, physical_page);
CREATE TABLE section_extraction_quality (
  chunk_id TEXT PRIMARY KEY REFERENCES source_sections(chunk_id),
  extraction_quality TEXT NOT NULL,
  warnings_json TEXT NOT NULL
);
CREATE INDEX idx_section_quality_status ON section_extraction_quality(extraction_quality);
CREATE TABLE program_courses (
  record_id TEXT PRIMARY KEY, program_id TEXT NOT NULL REFERENCES programs(program_id),
  module_id TEXT NOT NULL REFERENCES modules(module_id), course_id TEXT NOT NULL REFERENCES courses(course_id),
  course_nature TEXT, semester TEXT, credits REAL, weekly_hours REAL, total_hours REAL,
  teaching_hours REAL, practice_hours REAL, department TEXT, source_id TEXT NOT NULL REFERENCES sources(source_id),
  source_page INTEGER, source_row INTEGER, chunk_id TEXT,
  parser_version TEXT NOT NULL, confidence REAL NOT NULL, review_status TEXT NOT NULL
);
CREATE INDEX idx_program_courses_scope ON program_courses(program_id, semester, course_nature, module_id);
CREATE INDEX idx_program_courses_course ON program_courses(course_id, program_id);
CREATE UNIQUE INDEX idx_program_course_canonical ON program_courses(program_id, module_id, course_id, semester);
CREATE TABLE requirements (
  record_id TEXT PRIMARY KEY, program_id TEXT NOT NULL REFERENCES programs(program_id),
  module_id TEXT NOT NULL REFERENCES modules(module_id), required_credits REAL, listed_credits REAL,
  rule_text TEXT NOT NULL, source_id TEXT NOT NULL REFERENCES sources(source_id), source_page INTEGER,
  chunk_id TEXT, parser_version TEXT NOT NULL,
  confidence REAL NOT NULL, review_status TEXT NOT NULL
);
CREATE INDEX idx_requirements_program ON requirements(program_id, module_id);
CREATE TABLE field_verifications (
  entity_type TEXT NOT NULL,
  record_id TEXT NOT NULL,
  field_name TEXT NOT NULL,
  source_id TEXT NOT NULL REFERENCES sources(source_id),
  source_sha256 TEXT,
  chunk_id TEXT REFERENCES source_sections(chunk_id),
  physical_page INTEGER,
  table_row INTEGER,
  cell_ref TEXT,
  text_span TEXT,
  verification_status TEXT NOT NULL,
  lineage_json TEXT NOT NULL,
  PRIMARY KEY(entity_type, record_id, field_name)
);
CREATE INDEX idx_field_verification_record ON field_verifications(entity_type, record_id);
"""


def _normalized(value: object) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", str(value or "")).lower()


def _sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _combined_sha(*values: str) -> str:
    """Fingerprint evidence content together with its independent review ledgers."""

    return sha256("\n".join(values).encode("utf-8")).hexdigest()


def _semester(value: object) -> str:
    normalized = str(value or "").strip()
    return "" if normalized in {"", "未标注", "待定", "—", "-"} else normalized


def _page(value: object) -> int | None:
    matched = re.search(r"(?:原文件)?第\s*(\d+)\s*页", str(value or ""))
    return int(matched.group(1)) if matched else None


def _source_id(title: str, cohort: str, file_url: str) -> str:
    return stable_id("src", title, cohort, file_url)


def _program_id(name: str, college: str, cohort: int) -> str:
    return stable_id("program", name, college, cohort)


def _module_id(program_id: str, name: str) -> str:
    return stable_id("module", program_id, name)


def _course_id(code: str | None, name: str) -> str:
    return stable_id("course", (code or "").upper(), name)


def _read_aliases(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {"program_aliases": {}, "module_aliases": {}, "course_aliases": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    return {
        key: {str(alias): str(target) for alias, target in dict(value.get(key, {})).items()}
        for key in ("program_aliases", "module_aliases", "course_aliases")
    }


def _source_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _source_scope(value: object) -> str:
    """Return a source's explicit cohort scope."""

    scope = str(value or "不限").strip()
    return scope or "不限"


def _controlled_doc_type(value: object) -> str:
    """Return an explicit source type; missing legacy metadata stays unknown."""

    doc_type = str(value or "unknown").strip().lower() or "unknown"
    if doc_type not in DOC_TYPES:
        raise DataIntegrityError(
            f"invalid doc_type: {doc_type!r}; expected one of {sorted(DOC_TYPES)!r}"
        )
    return doc_type


def _controlled_topics(value: object) -> tuple[str, ...]:
    """Parse registry topics without using document text as a fallback."""

    if value is None:
        return ()
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return ()
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            decoded = raw.split(",")
    else:
        decoded = value
    if not isinstance(decoded, (list, tuple)) or any(
        not isinstance(topic, str) or not topic.strip() for topic in decoded
    ):
        raise DataIntegrityError("topics must be a comma-separated string or JSON string list")
    topics = tuple(topic.strip().lower() for topic in decoded)
    invalid = sorted(set(topics) - POLICY_TOPICS)
    if invalid:
        raise DataIntegrityError(f"invalid topics: {invalid!r}")
    if len(set(topics)) != len(topics):
        raise DataIntegrityError("source registry topics must not contain duplicates")
    return topics


def _declared_sha256(value: object) -> str | None:
    digest = str(value or "").strip().lower()
    if not digest:
        return None
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise DataIntegrityError("source_sha256 must be a lowercase SHA-256 hex digest")
    return digest


def _source_authenticity_status(value: object, *, has_digest: bool) -> str:
    status = str(value or "").strip().lower() or (
        "review_required" if has_digest else "unverified"
    )
    if status not in SOURCE_AUTHENTICITY_STATUSES:
        raise DataIntegrityError(
            "invalid authenticity_status: "
            f"{status!r}; expected one of {sorted(SOURCE_AUTHENTICITY_STATUSES)!r}"
        )
    if status == "verified" and not has_digest:
        raise DataIntegrityError("verified source authenticity requires source_sha256")
    return status


def _review_status(value: object, *, default: str = "unverified") -> str:
    status = str(value or default).strip()
    if status not in REVIEW_STATUSES:
        raise DataIntegrityError(
            f"invalid review_status: {status!r}; expected one of {sorted(REVIEW_STATUSES)!r}"
        )
    return status


@dataclass(frozen=True)
class SourceReview:
    """One reviewer-owned source decision from the independent audit ledger.

    ``reviewer``, ``method``, and ``reviewed_at`` are deliberately optional so
    the released ledger can add accountability fields without changing the
    build contract.  Only the decision itself controls evidence trust.
    """

    original_title: str
    corrected_title: str
    decision: str
    reviewer: str | None = None
    method: str | None = None
    reviewed_at: str | None = None


@dataclass(frozen=True)
class EvidenceReview:
    """A reviewer-owned decision for one precise evidence chunk.

    This ledger exists for vetted slices of an otherwise quarantined source.
    Optional scope and audit fields keep the file forward-compatible without
    allowing source chunks to self-promote through their JSON payload.
    """

    chunk_id: str
    decision: str
    scope: str | None = None
    reviewer: str | None = None
    method: str | None = None
    reviewed_at: str | None = None


@dataclass(frozen=True)
class SourceTrust:
    """The source-owned authenticity and taxonomy state used by the builder."""

    source_sha256: str | None
    authenticity_status: str
    doc_type: str
    topics: tuple[str, ...]


@dataclass(frozen=True)
class ReconciledRows:
    """Deterministic duplicate accounting for one catalog entity type."""

    rows: tuple[dict[str, object], ...]
    input_count: int
    accepted_count: int
    exact_duplicate_count: int
    quarantined_count: int = 0

    def as_metadata(self) -> dict[str, int]:
        values = {
            "input": self.input_count,
            "accepted": self.accepted_count,
            "exact_duplicates": self.exact_duplicate_count,
            "quarantined": self.quarantined_count,
        }
        if values["input"] != (
            values["accepted"] + values["exact_duplicates"] + values["quarantined"]
        ):
            raise DataIntegrityError(f"reconciliation count conservation violated: {values!r}")
        return values


def _title_key(value: object) -> str:
    """Normalize a title or source filename for conservative ledger matching."""

    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        return ""
    return _normalized(Path(raw).stem)


def _optional_review_value(row: dict[str, str | None], *names: str) -> str | None:
    for name in names:
        value = str(row.get(name) or "").strip()
        if value:
            return value
    return None


def _load_source_reviews(path: Path | None) -> tuple[SourceReview, ...]:
    """Read a source-review ledger without making optional audit fields required."""

    if path is None or not path.is_file():
        return ()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"original_title", "corrected_title", "decision"}
        actual = set(reader.fieldnames or ())
        if not required.issubset(actual):
            missing = ", ".join(sorted(required - actual))
            raise DataIntegrityError(f"source review ledger missing required columns: {missing}")
        reviews: list[SourceReview] = []
        for line_number, raw in enumerate(reader, start=2):
            row = {str(key): value for key, value in raw.items() if key is not None}
            original_title = str(row.get("original_title") or "").strip()
            corrected_title = str(row.get("corrected_title") or "").strip()
            decision = str(row.get("decision") or "").strip().lower()
            if not original_title and not corrected_title:
                raise DataIntegrityError(
                    f"source review ledger row {line_number} has no original_title or corrected_title"
                )
            if not decision:
                raise DataIntegrityError(
                    f"source review ledger row {line_number} has no decision"
                )
            reviews.append(
                SourceReview(
                    original_title=original_title,
                    corrected_title=corrected_title,
                    decision=decision,
                    reviewer=_optional_review_value(row, "reviewer"),
                    method=_optional_review_value(row, "method", "review_method"),
                    reviewed_at=_optional_review_value(
                        row, "reviewed_at", "reviewed_on", "reviewed_date", "timestamp"
                    ),
                )
            )
    return tuple(reviews)


def _load_evidence_reviews(path: Path | None) -> dict[str, EvidenceReview]:
    """Read explicit chunk-level decisions from an independent audit ledger."""

    if path is None or not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"chunk_id", "decision"}
        actual = set(reader.fieldnames or ())
        if not required.issubset(actual):
            missing = ", ".join(sorted(required - actual))
            raise DataIntegrityError(f"evidence review ledger missing required columns: {missing}")
        reviews: dict[str, EvidenceReview] = {}
        for line_number, raw in enumerate(reader, start=2):
            row = {str(key): value for key, value in raw.items() if key is not None}
            chunk_id = str(row.get("chunk_id") or "").strip()
            decision = str(row.get("decision") or "").strip().lower()
            if not chunk_id:
                raise DataIntegrityError(f"evidence review ledger row {line_number} has no chunk_id")
            if not decision:
                raise DataIntegrityError(f"evidence review ledger row {line_number} has no decision")
            if chunk_id in reviews:
                raise DataIntegrityError(f"duplicate chunk_id in evidence review ledger: {chunk_id!r}")
            reviews[chunk_id] = EvidenceReview(
                chunk_id=chunk_id,
                decision=decision,
                scope=_optional_review_value(row, "scope"),
                reviewer=_optional_review_value(row, "reviewer"),
                method=_optional_review_value(row, "method", "review_method"),
                reviewed_at=_optional_review_value(
                    row, "reviewed_at", "reviewed_on", "reviewed_date", "timestamp"
                ),
            )
    return reviews


def _review_for_source(
    source: dict[str, str], reviews: tuple[SourceReview, ...]
) -> SourceReview | None:
    """Return an unambiguous trusted review for one registered source.

    The reviewer ledger may correct a title, so both its original and corrected
    forms are considered.  A filename basename is used only as a fallback
    identifier.  Any conflicting match fails closed instead of picking an
    arbitrary review decision.
    """

    title_key = _title_key(source.get("doc_title"))
    basename_key = _title_key(source.get("file"))
    matches = tuple(
        review
        for review in reviews
        if title_key
        in {_title_key(review.original_title), _title_key(review.corrected_title)}
        or (
            basename_key
            and basename_key
            in {_title_key(review.original_title), _title_key(review.corrected_title)}
        )
    )
    decisions = {review.decision for review in matches}
    if len(decisions) != 1:
        return None
    review = matches[0]
    return review if review.decision in VERIFIED_SOURCE_REVIEW_DECISIONS else None


def _section_review_status(
    raw_status: object,
    source_authenticity: str,
    extraction_quality: str,
    evidence_review: EvidenceReview | None,
) -> str:
    """Keep source authenticity, extraction quality, and review distinct.

    A reviewer may attest that a URL is an authentic school source, but that
    cannot certify a failed OCR/table extraction.  Likewise a raw chunk's
    ``verified`` assertion is never enough to promote it.
    """

    claimed = _review_status(raw_status)
    if extraction_quality == "failed":
        return "unverified"
    if source_authenticity != "verified" or extraction_quality != "verified":
        return "unverified" if claimed == "unverified" else "review_required"

    if evidence_review is not None:
        return (
            "verified"
            if evidence_review.decision in VERIFIED_EVIDENCE_REVIEW_DECISIONS
            else "unverified"
        )
    # The source review establishes authenticity, while the independently
    # recorded extraction quality establishes that this section was extracted
    # acceptably.  Field values still require their own lineage/review below.
    if source_authenticity == "verified" and extraction_quality == "verified":
        return "verified"
    # An external chunk can request a review, but cannot mark itself verified.
    return "review_required" if claimed == "verified" else claimed


def _structured_review_status(
    evidence: object,
    source_id: str,
    section_statuses: dict[str, tuple[str, str, str]],
) -> str:
    """Derive structured-record trust from the exact evidence it references."""

    if not isinstance(evidence, dict):
        return "review_required"
    chunk_id = str(evidence.get("chunk_id") or "").strip()
    if not chunk_id:
        return "review_required"
    stored = section_statuses.get(chunk_id)
    if stored is None or stored[0] != source_id:
        return "unverified"
    return stored[1]


def _evidence_chunk_id(evidence: object) -> str | None:
    if not isinstance(evidence, dict):
        return None
    value = str(evidence.get("chunk_id") or "").strip()
    return value or None


def _extraction_quality(value: object) -> str:
    quality = str(value or "review_required").strip().lower() or "review_required"
    if quality not in EXTRACTION_QUALITY_STATUSES:
        raise DataIntegrityError(
            "invalid extraction_quality: "
            f"{quality!r}; expected one of {sorted(EXTRACTION_QUALITY_STATUSES)!r}"
        )
    return quality


def _extraction_warnings(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise DataIntegrityError("extraction_warnings must be a list of non-empty strings")
    return tuple(item.strip() for item in value)


def _load_sections(
    chunk_file: Path,
    source_ids: dict[tuple[str, str], str],
    source_trusts: dict[str, SourceTrust],
    evidence_reviews: dict[str, EvidenceReview],
) -> tuple[
    list[tuple[object, ...]],
    list[tuple[object, ...]],
    dict[str, tuple[str, str, str]],
]:
    """Materialize source sections and their ledger-derived trust state."""

    sections: list[tuple[object, ...]] = []
    qualities: list[tuple[object, ...]] = []
    section_statuses: dict[str, tuple[str, str, str]] = {}
    if not chunk_file.is_file():
        return sections, qualities, section_statuses
    with chunk_file.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise DataIntegrityError(f"chunk line {line_number} must be a JSON object")
            title = str(value["doc_title"]).strip()
            chunk_cohort = _source_scope(value.get("cohort"))
            source_id = _source_for(title, chunk_cohort, source_ids)
            chunk_id = str(value["chunk_id"]).strip()
            if not chunk_id:
                raise DataIntegrityError(f"chunk line {line_number} has an empty chunk_id")
            if chunk_id in section_statuses:
                raise DataIntegrityError(f"duplicate chunk_id in chunk file: {chunk_id!r}")
            trust = source_trusts[source_id]
            chunk_doc_type = value.get("doc_type")
            if chunk_doc_type is not None and _controlled_doc_type(chunk_doc_type) != trust.doc_type:
                raise DataIntegrityError(
                    "chunk doc_type must exactly match its registered source taxonomy: "
                    f"chunk_id={chunk_id!r}"
                )
            chunk_topics = value.get("topics")
            if chunk_topics is not None and _controlled_topics(chunk_topics) != trust.topics:
                raise DataIntegrityError(
                    "chunk topics must exactly match its registered source taxonomy: "
                    f"chunk_id={chunk_id!r}"
                )
            chunk_sha256 = _declared_sha256(value.get("source_sha256"))
            if chunk_sha256 is not None and chunk_sha256 != trust.source_sha256:
                raise DataIntegrityError(
                    "chunk source_sha256 does not match registered source bytes: "
                    f"chunk_id={chunk_id!r}"
                )
            extraction_quality = _extraction_quality(value.get("extraction_quality"))
            warnings = _extraction_warnings(value.get("extraction_warnings"))
            review_status = _section_review_status(
                value.get("review_status"),
                trust.authenticity_status,
                extraction_quality,
                evidence_reviews.get(chunk_id),
            )
            section_statuses[chunk_id] = (source_id, review_status, extraction_quality)
            sections.append(
                (
                    chunk_id,
                    source_id,
                    value.get("article"),
                    value["text"],
                    _page(value.get("article")),
                    int(bool(value.get("is_table"))),
                    PARSER_VERSION,
                    datetime.now(timezone.utc).isoformat(),
                    0.8,
                    review_status,
                )
            )
            qualities.append((chunk_id, extraction_quality, json.dumps(warnings, ensure_ascii=False)))
    return sections, qualities, section_statuses


def _source_index(rows: Iterable[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    exact: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        title = str(row.get("doc_title") or "").strip()
        if not title:
            raise DataIntegrityError("source registry contains an empty doc_title")
        key = (title, _source_scope(row.get("cohort")))
        if key in exact:
            raise DataIntegrityError(
                "source registry contains duplicate title/cohort scope: "
                f"title={title!r}, cohort={key[1]!r}"
            )
        exact[key] = row
    return exact


def _canonical_value(value: object) -> object:
    """Create a comparison-only representation independent of input order."""

    if isinstance(value, dict):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            if not str(key).startswith("__catalog_")
        }
    if isinstance(value, (list, tuple)):
        normalized = [_canonical_value(item) for item in value]
        # Catalog lists are semantic collections.  Sorting their canonical
        # JSON prevents ordering alone from deciding whether two rows conflict.
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _canonical_payload(row: dict[str, object]) -> str:
    return json.dumps(
        _canonical_value(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _input_row_label(row: dict[str, object], ordinal: int) -> object:
    return row.get("__catalog_input_row__", ordinal)


def _field_differences(entries: list[tuple[int, dict[str, object]]]) -> dict[str, object]:
    """Return precise, serializable field deltas for a conflict diagnostic."""

    fields = sorted(
        {
            str(field)
            for _ordinal, row in entries
            for field in row
            if not str(field).startswith("__catalog_")
        }
    )
    differences: dict[str, object] = {}
    for field in fields:
        values: list[dict[str, object]] = []
        signatures: set[str] = set()
        for ordinal, row in entries:
            value = row.get(field)
            signature = json.dumps(
                _canonical_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            signatures.add(signature)
            values.append({"input_row": _input_row_label(row, ordinal), "value": value})
        if len(signatures) > 1:
            differences[field] = values
    return differences


def _reconcile_catalog_rows(
    entity: str,
    raw_rows: object,
    *,
    key_for: Any,
) -> ReconciledRows:
    """Deduplicate exact catalog inputs and fail before SQLite sees conflicts.

    SQLite uniqueness violations used to be hidden by ``INSERT OR IGNORE``.
    This reconciliation is intentionally run before any business-table write,
    and selects canonical ordering by key/payload rather than input order.
    """

    if raw_rows is None:
        values: list[object] = []
    elif isinstance(raw_rows, list):
        values = raw_rows
    else:
        raise DataIntegrityError(f"catalog {entity} must be a list")

    indexed: list[tuple[int, dict[str, object]]] = []
    for ordinal, value in enumerate(values, start=1):
        if not isinstance(value, dict):
            raise DataIntegrityError(f"catalog {entity} input row {ordinal} must be an object")
        indexed.append((ordinal, {str(key): item for key, item in value.items()}))

    groups: dict[tuple[object, ...], list[tuple[int, dict[str, object]]]] = {}
    for ordinal, row in indexed:
        try:
            raw_key = key_for(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise DataIntegrityError(
                f"catalog {entity} input row {_input_row_label(row, ordinal)!r} has no canonical key: {exc}"
            ) from exc
        key = tuple(raw_key) if isinstance(raw_key, tuple) else (raw_key,)
        groups.setdefault(key, []).append((ordinal, row))

    accepted: list[tuple[tuple[object, ...], str, dict[str, object]]] = []
    exact_duplicates = 0
    for key, entries in groups.items():
        payloads: dict[str, list[tuple[int, dict[str, object]]]] = {}
        for ordinal, row in entries:
            payloads.setdefault(_canonical_payload(row), []).append((ordinal, row))
        if len(payloads) != 1:
            labels = [_input_row_label(row, ordinal) for ordinal, row in entries]
            differences = _field_differences(entries)
            raise DataIntegrityError(
                f"catalog {entity} canonical-key conflict; key={key!r}; "
                f"input_rows={labels!r}; field_differences="
                f"{json.dumps(differences, ensure_ascii=False, sort_keys=True)}"
            )
        payload, equivalent = next(iter(payloads.items()))
        exact_duplicates += len(equivalent) - 1
        # Stable selection and stable output ordering make the result wholly
        # independent of source-file row order.
        selected = min(
            equivalent,
            key=lambda entry: str(_input_row_label(entry[1], entry[0])),
        )[1]
        accepted.append((key, payload, selected))

    accepted.sort(key=lambda item: (repr(item[0]), item[1]))
    rows = tuple(
        {
            key: value
            for key, value in row.items()
            if not key.startswith("__catalog_")
        }
        for _key, _payload, row in accepted
    )
    result = ReconciledRows(
        rows=rows,
        input_count=len(indexed),
        accepted_count=len(rows),
        exact_duplicate_count=exact_duplicates,
    )
    result.as_metadata()
    return result


def _registered_source_path(root: Path, row: dict[str, str]) -> Path:
    """Resolve a source-registry path under an explicitly selected raw root."""

    raw_name = str(row.get("file") or "").strip().replace("\\", "/")
    if not raw_name:
        raise DataIntegrityError("source registry contains an empty file")
    candidate = root.joinpath(*[part for part in raw_name.split("/") if part])
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise DataIntegrityError(f"source file escapes source_root: {raw_name!r}") from exc
    return candidate


def _materialize_sources(
    connection: sqlite3.Connection,
    rows: list[dict[str, str]],
    source_root: Path,
    reviews: tuple[SourceReview, ...],
    *,
    require_source_files: bool,
) -> tuple[dict[tuple[str, str], str], dict[str, SourceTrust]]:
    ids: dict[tuple[str, str], str] = {}
    values: list[tuple[object, ...]] = []
    taxonomy_values: list[tuple[object, ...]] = []
    authenticity_values: list[tuple[object, ...]] = []
    trusts: dict[str, SourceTrust] = {}
    for row in rows:
        title = str(row.get("doc_title") or "").strip()
        cohort = _source_scope(row.get("cohort"))
        url = str(row.get("file_url") or "")
        key = (title, cohort)
        if key in ids:
            raise DataIntegrityError(
                "source registry contains duplicate title/cohort scope: "
                f"title={title!r}, cohort={cohort!r}"
            )
        identifier = _source_id(title, cohort, url)
        ids[key] = identifier
        local = _registered_source_path(source_root, row)
        declared_sha256 = _declared_sha256(row.get("source_sha256"))
        if require_source_files and not local.is_file():
            raise DataIntegrityError(
                "registered source does not exist under explicit source_root: "
                f"title={title!r}, file={row.get('file')!r}, source_root={str(source_root)!r}"
            )
        if require_source_files and declared_sha256 is None:
            raise DataIntegrityError(
                "registered source under explicit source_root requires source_sha256: "
                f"title={title!r}, file={row.get('file')!r}"
            )
        observed_sha256 = _sha(local)
        if declared_sha256 and observed_sha256 and declared_sha256 != observed_sha256:
            raise DataIntegrityError(
                "registered source_sha256 does not match source bytes: "
                f"title={title!r}, file={row.get('file')!r}"
            )
        configured_authenticity = _source_authenticity_status(
            row.get("authenticity_status"), has_digest=declared_sha256 is not None
        )
        source_review = _review_for_source(row, reviews)
        if configured_authenticity == "verified" and source_review is not None:
            authenticity_status = "verified"
        elif configured_authenticity == "unverified" or not (declared_sha256 or observed_sha256):
            authenticity_status = "unverified"
        else:
            authenticity_status = "review_required"
        source_sha256 = observed_sha256 or declared_sha256
        doc_type = _controlled_doc_type(row.get("doc_type"))
        topics = _controlled_topics(row.get("topics"))
        trusts[identifier] = SourceTrust(
            source_sha256=source_sha256,
            authenticity_status=authenticity_status,
            doc_type=doc_type,
            topics=topics,
        )
        year = row.get("year") or row.get("cohort") or ""
        fallback_authority = 2 if row.get("level") == "校级" else 1
        try:
            authority = max(1, int(str(row.get("authority_level") or fallback_authority)))
        except ValueError:
            authority = fallback_authority
        published_at = str(row.get("published_at") or year).strip() or None
        effective_from = str(row.get("effective_from") or published_at or "").strip() or None
        effective_to = str(row.get("effective_to") or "").strip() or None
        supersedes = str(row.get("supersedes_source_id") or "").strip() or None
        values.append(
            (
                identifier,
                title,
                row.get("level", ""),
                row.get("college", ""),
                cohort,
                authority,
                published_at,
                effective_from,
                effective_to,
                supersedes,
                row.get("status", "历史"),
                row.get("page_url"),
                url,
                source_sha256,
                row.get("collected_at"),
            )
        )
        taxonomy_values.append((identifier, doc_type, json.dumps(topics, ensure_ascii=False)))
        authenticity_values.append(
            (
                identifier,
                declared_sha256,
                observed_sha256,
                authenticity_status,
                source_review.decision if source_review is not None else None,
            )
        )
    connection.executemany(
        "INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values
    )
    connection.executemany("INSERT INTO source_taxonomy VALUES (?, ?, ?)", taxonomy_values)
    connection.executemany(
        "INSERT INTO source_authenticity VALUES (?, ?, ?, ?, ?)", authenticity_values
    )
    return ids, trusts


def _source_for(
    title: str,
    cohort: int | str | None,
    source_ids: dict[tuple[str, str], str],
) -> str:
    source_title = str(title).strip()
    scope = _source_scope(cohort)
    found = source_ids.get((source_title, scope))
    if found:
        return found
    universal = source_ids.get((source_title, "不限"))
    if universal:
        return universal
    available_scopes = sorted(
        registered_scope
        for registered_title, registered_scope in source_ids
        if registered_title == source_title
    )
    raise DataIntegrityError(
        "catalog/chunk source is not registered for its cohort or the universal scope: "
        f"title={source_title!r}, cohort={scope!r}, available={available_scopes!r}"
    )


def _lineage_integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _field_metadata(row: dict[str, object], field: str) -> tuple[object, str]:
    """Read a field-owned verification entry without accepting a bare claim.

    ``field_verification`` is the current contract.  ``field_lineage`` is
    accepted as a migration aid, but it still has to carry an explicit review
    status before it can promote a field.
    """

    verification = row.get("field_verification")
    lineage = row.get("field_lineage")
    entry: object = None
    if isinstance(verification, dict):
        entry = verification.get(field)
    if entry is None and isinstance(lineage, dict):
        entry = lineage.get(field)
    if not isinstance(entry, dict):
        return {}, "review_required"
    nested_lineage = entry.get("lineage")
    value = nested_lineage if isinstance(nested_lineage, dict) else entry
    status = str(entry.get("verification_status") or entry.get("status") or "").strip().lower()
    if status not in REVIEW_STATUSES:
        status = "review_required"
    return value, status


def _decimal(value: object) -> Decimal | None:
    """Return a finite decimal without accepting booleans or malformed values."""

    if isinstance(value, bool):
        return None
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _numeric_tokens(value: str) -> tuple[Decimal, ...]:
    """Read standalone Arabic numerals, never a prefix of a larger number."""

    values: list[Decimal] = []
    for token in re.findall(r"(?<![0-9.])[0-9]+(?:\.[0-9]+)?(?![0-9.])", value):
        parsed = _decimal(token)
        if parsed is not None:
            values.append(parsed)
    return tuple(values)


def _span_covers_field_value(field: str, exact_value: object, text_span: str | None) -> bool:
    """Validate a field-owned evidence span without cross-field numeric matches.

    A source/table review says that a physical extraction is usable; it does
    not make a number in any nearby text evidence for every structured field.
    Numeric fields therefore require an exact numeric token plus their own
    semantic unit (or a bare table cell), while textual fields require an exact
    normalized cell value.  This deliberately fails closed for prose-like
    spans that cannot be attributed to one field.
    """

    if exact_value is None or text_span is None:
        return exact_value is None
    span = text_span.strip()
    if not span:
        return False

    if field in {"credits", "required_credits"}:
        expected = _decimal(exact_value)
        if expected is None or re.search(r"(?:第\s*)?[0-9]+\s*(?:学期|semester\b)", span, re.I):
            return False
        labelled = re.findall(
            r"(?<![0-9.])([0-9]+(?:\.[0-9]+)?)\s*(?:学分|credits?\b)",
            span,
            re.I,
        )
        values = tuple(value for token in labelled if (value := _decimal(token)) is not None)
        if values:
            return expected in values
        # A table cell can legitimately contain only the number.  Treat any
        # prose or a number with a different unit as insufficient evidence.
        bare_values = _numeric_tokens(span)
        return (
            len(bare_values) == 1
            and bare_values[0] == expected
            and re.fullmatch(r"\s*[0-9]+(?:\.[0-9]+)?\s*", span) is not None
        )

    if field == "semester":
        expected = _decimal(exact_value)
        if expected is None or re.search(r"[0-9]+(?:\.[0-9]+)?\s*(?:学分|credits?\b)", span, re.I):
            return False
        semantic = re.findall(r"(?:第\s*)?([0-9]+)\s*(?:学期|semester\b)", span, re.I)
        values = tuple(value for token in semantic if (value := _decimal(token)) is not None)
        if values:
            return expected in values
        # As above, accept a bare cell only on exact equality; do not allow
        # "1" to be proved by a substring of "10".
        bare_values = _numeric_tokens(span)
        return (
            len(bare_values) == 1
            and bare_values[0] == expected
            and re.fullmatch(r"\s*[0-9]+\s*", span) is not None
        )

    return _normalized(exact_value) == _normalized(span)


def _field_verification_rows(
    *,
    entity_type: str,
    record_id: str,
    source_id: str,
    source_trust: SourceTrust,
    evidence: object,
    section_statuses: dict[str, tuple[str, str, str]],
    strict_fields: tuple[str, ...],
    catalog_row: dict[str, object],
    field_values: dict[str, object],
    expected_page: object = None,
) -> tuple[str, list[tuple[object, ...]]]:
    """Materialize reviewed field lineage and derive the row trust status.

    A structured record is ``verified`` only when each strict field carries a
    source hash, page, row/cell/span locator, explicit field review decision,
    and points at the same reviewed evidence chunk used by the record.  This
    prevents a whole-source review from silently promoting OCR-derived numbers.
    """

    evidence_chunk_id = _evidence_chunk_id(evidence)
    base_status = _structured_review_status(evidence, source_id, section_statuses)
    expected_page_number = _lineage_integer(expected_page)
    rows: list[tuple[object, ...]] = []
    statuses: list[str] = []
    for field in strict_fields:
        lineage, requested_status = _field_metadata(catalog_row, field)
        lineage_dict = lineage if isinstance(lineage, dict) else {}
        try:
            lineage_sha = _declared_sha256(lineage_dict.get("source_sha256"))
        except DataIntegrityError:
            lineage_sha = None
        chunk_id = str(lineage_dict.get("chunk_id") or "").strip() or None
        page = _lineage_integer(lineage_dict.get("page"))
        table_row = _lineage_integer(lineage_dict.get("row"))
        cell_ref = str(lineage_dict.get("cell") or "").strip() or None
        text_span = str(lineage_dict.get("span") or "").strip() or None
        locator_present = table_row is not None or cell_ref is not None or text_span is not None
        exact_value = field_values.get(field)
        span_covers_value = _span_covers_field_value(field, exact_value, text_span)
        matching_page = expected_page_number is None or page == expected_page_number
        valid_lineage = (
            source_trust.source_sha256 is not None
            and lineage_sha == source_trust.source_sha256
            and evidence_chunk_id is not None
            and chunk_id == evidence_chunk_id
            and page is not None
            and page > 0
            and locator_present
            and text_span is not None
            and matching_page
            and span_covers_value
        )
        if (
            requested_status == "verified"
            and valid_lineage
            and base_status == "verified"
            and source_trust.authenticity_status == "verified"
        ):
            status = "verified"
        elif base_status == "unverified":
            status = "unverified"
        else:
            status = "review_required"
        statuses.append(status)
        rows.append(
            (
                entity_type,
                record_id,
                field,
                source_id,
                lineage_sha,
                chunk_id,
                page,
                table_row,
                cell_ref,
                text_span,
                status,
                json.dumps(_canonical_value(lineage_dict), ensure_ascii=False, sort_keys=True),
            )
        )

    if base_status == "unverified":
        return "unverified", rows
    if base_status == "verified" and statuses and all(status == "verified" for status in statuses):
        return "verified", rows
    return "review_required", rows


def build_database(
    output: str | Path = DEFAULT_DATABASE,
    *,
    catalog_path: str | Path = DEFAULT_CATALOG,
    sources_path: str | Path = DEFAULT_SOURCES,
    chunks_path: str | Path = DEFAULT_CHUNKS,
    aliases_path: str | Path = DEFAULT_ALIAS_CONFIG,
    source_review_path: str | Path | None = DEFAULT_SOURCE_REVIEW,
    evidence_review_path: str | Path | None = DEFAULT_EVIDENCE_REVIEW,
    source_root: str | Path | None = None,
) -> dict[str, object]:
    """Build a new immutable SQLite projection; generated output is not Git data.

    Catalog identity reconciliation happens before SQLite is opened for business
    rows.  Thus a uniqueness conflict can never be hidden by insertion order
    or by SQLite's conflict handling.
    """
    target, catalog_file, source_file, chunk_file = map(
        Path, (output, catalog_path, sources_path, chunks_path)
    )
    source_review_file = Path(source_review_path) if source_review_path is not None else None
    evidence_review_file = Path(evidence_review_path) if evidence_review_path is not None else None
    raw_catalog = json.loads(catalog_file.read_text(encoding="utf-8"))
    if not isinstance(raw_catalog, dict):
        raise DataIntegrityError("catalog root must be a JSON object")
    catalog = {str(key): value for key, value in raw_catalog.items()}
    source_rows = _source_rows(source_file)
    _source_index(source_rows)
    aliases = _read_aliases(Path(aliases_path))
    source_reviews = _load_source_reviews(source_review_file)
    evidence_reviews = _load_evidence_reviews(evidence_review_file)
    plans = _reconcile_catalog_rows(
        "plans",
        catalog.get("plans", []),
        key_for=lambda row: (
            str(row["major"]),
            str(row["college"]),
            int(str(row["cohort"])),
        ),
    )
    course_offerings = _reconcile_catalog_rows(
        "course_offerings",
        catalog.get("courses", []),
        key_for=lambda row: (
            str(row["major"]),
            str(row["college"]),
            int(str(row["cohort"])),
            str(row["module"]),
            str(row.get("code") or "").upper(),
            str(row["name"]),
            _semester(row.get("semester")),
        ),
    )
    raw_source_root = Path(source_root) if source_root is not None else ROOT / "data" / "raw"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(SCHEMA)
        source_ids, source_trusts = _materialize_sources(
            connection,
            source_rows,
            raw_source_root,
            source_reviews,
            require_source_files=source_root is not None,
        )
        sections, section_qualities, section_statuses = _load_sections(
            chunk_file, source_ids, source_trusts, evidence_reviews
        )
        connection.executemany(
            "INSERT INTO source_sections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", sections
        )
        connection.executemany(
            "INSERT INTO section_extraction_quality VALUES (?, ?, ?)", section_qualities
        )

        program_inputs: list[dict[str, object]] = []
        requirement_inputs: list[dict[str, object]] = []
        module_entity_inputs: list[dict[str, object]] = []
        for plan_index, plan in enumerate(plans.rows, start=1):
            cohort = int(str(plan["cohort"]))
            program_id = _program_id(str(plan["major"]), str(plan["college"]), cohort)
            program_inputs.append(
                {
                    "program_id": program_id,
                    "major": plan["major"],
                    "college": plan["college"],
                    "cohort": cohort,
                    "source_title": plan["source_title"],
                    "__catalog_input_row__": f"plans[{plan_index}]",
                }
            )
            raw_modules = plan.get("modules", [])
            if not isinstance(raw_modules, list):
                raise DataIntegrityError(f"catalog plans[{plan_index}].modules must be a list")
            for module_index, raw_module in enumerate(raw_modules, start=1):
                if not isinstance(raw_module, dict):
                    raise DataIntegrityError(
                        f"catalog plans[{plan_index}].modules[{module_index}] must be an object"
                    )
                module = {str(key): value for key, value in raw_module.items()}
                module_name = str(module.get("name") or "").strip()
                if not module_name:
                    raise DataIntegrityError(
                        f"catalog plans[{plan_index}].modules[{module_index}] has an empty name"
                    )
                module_entity_inputs.append(
                    {
                        "program_id": program_id,
                        "module": module_name,
                        "__catalog_input_row__": f"plans[{plan_index}].modules[{module_index}]",
                    }
                )
                requirement_inputs.append(
                    {
                        **module,
                        "program_id": program_id,
                        "module": module_name,
                        "source_title": plan["source_title"],
                        "cohort": cohort,
                        "__catalog_input_row__": f"plans[{plan_index}].modules[{module_index}]",
                    }
                )

        for course_index, course in enumerate(course_offerings.rows, start=1):
            cohort = int(str(course["cohort"]))
            program_id = _program_id(str(course["major"]), str(course["college"]), cohort)
            module_name = str(course.get("module") or "").strip()
            if not module_name:
                raise DataIntegrityError(f"catalog courses[{course_index}] has an empty module")
            module_entity_inputs.append(
                {
                    "program_id": program_id,
                    "module": module_name,
                    "__catalog_input_row__": f"courses[{course_index}]",
                }
            )

        programs = _reconcile_catalog_rows(
            "programs",
            program_inputs,
            key_for=lambda row: (str(row["program_id"]),),
        )
        modules = _reconcile_catalog_rows(
            "modules",
            module_entity_inputs,
            key_for=lambda row: (str(row["program_id"]), str(row["module"])),
        )
        requirements = _reconcile_catalog_rows(
            "requirements",
            requirement_inputs,
            key_for=lambda row: (str(row["program_id"]), str(row["module"])),
        )

        program_rows: list[tuple[object, ...]] = []
        alias_rows: set[tuple[str, str, str]] = set()
        program_alias_input_count = 0
        module_rows: list[tuple[str, str, str]] = []
        module_map: dict[tuple[str, str], str] = {}
        program_ids: set[str] = set()
        for program in programs.rows:
            program_id = str(program["program_id"])
            source_id = _source_for(program["source_title"], int(program["cohort"]), source_ids)
            program_ids.add(program_id)
            program_rows.append(
                (
                    program_id,
                    program["major"],
                    program["college"],
                    program["cohort"],
                    source_id,
                )
            )
            values = {str(program["major"]), str(program["major"]).removesuffix("专业")}
            values.update(
                alias
                for alias, target_name in aliases["program_aliases"].items()
                if target_name == program["major"]
            )
            program_alias_input_count += len(values)
            alias_rows.update(
                (alias, _normalized(alias), program_id) for alias in values if _normalized(alias)
            )
        for module in modules.rows:
            program_id = str(module["program_id"])
            module_name = str(module["module"])
            module_id = _module_id(program_id, module_name)
            module_map[(program_id, module_name)] = module_id
            module_rows.append((module_id, program_id, module_name))

        requirement_rows: list[tuple[object, ...]] = []
        field_verification_rows: list[tuple[object, ...]] = []
        quarantined_requirement_count = 0
        for module in requirements.rows:
            program_id = str(module["program_id"])
            module_name = str(module["module"])
            source_id = _source_for(module["source_title"], int(module["cohort"]), source_ids)
            evidence = module.get("evidence")
            page = _page(evidence.get("article") if isinstance(evidence, dict) else None)
            chunk_id = _evidence_chunk_id(evidence)
            record_id = stable_id("req", program_id, module_name)
            review_status, field_rows = _field_verification_rows(
                entity_type="requirement",
                record_id=record_id,
                source_id=source_id,
                source_trust=source_trusts[source_id],
                evidence=evidence,
                section_statuses=section_statuses,
                strict_fields=STRICT_REQUIREMENT_FIELDS,
                catalog_row=module,
                field_values={
                    "module": module_name,
                    "required_credits": module.get("required_credits"),
                },
                expected_page=page,
            )
            field_verification_rows.extend(field_rows)
            # Untraceable or source-mismatched requirements remain in the
            # explicit quarantine accounting rather than entering production.
            if chunk_id is None or review_status == "unverified":
                quarantined_requirement_count += 1
                continue
            requirement_rows.append(
                (
                    record_id,
                    program_id,
                    module_map[(program_id, module_name)],
                    module.get("required_credits"),
                    module.get("listed_credits"),
                    module.get("rule_text") or "",
                    source_id,
                    page,
                    chunk_id,
                    PARSER_VERSION,
                    0.9 if evidence else 0.6,
                    review_status,
                )
            )

        connection.executemany("INSERT INTO programs VALUES (?, ?, ?, ?, ?)", program_rows)
        connection.executemany("INSERT INTO program_aliases VALUES (?, ?, ?)", sorted(alias_rows))
        connection.executemany("INSERT INTO modules VALUES (?, ?, ?)", module_rows)
        connection.executemany(
            "INSERT INTO requirements VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", requirement_rows
        )

        module_alias_rows: set[tuple[str, str, str]] = set()
        module_alias_input_count = 0
        for (_program_key, module_name), module_id in module_map.items():
            module_alias_rows.add((module_name, _normalized(module_name), module_id))
            module_alias_input_count += 1
            for alias, target_name in aliases["module_aliases"].items():
                if _normalized(target_name) in _normalized(module_name) or _normalized(
                    module_name
                ) in _normalized(target_name):
                    module_alias_rows.add((alias, _normalized(alias), module_id))
                    module_alias_input_count += 1
        connection.executemany("INSERT INTO module_aliases VALUES (?, ?, ?)", sorted(module_alias_rows))

        course_entity_inputs = [
            {
                "code": str(course.get("code") or "").upper() or None,
                "name": str(course["name"]),
                "__catalog_input_row__": f"courses[{course_index}]",
            }
            for course_index, course in enumerate(course_offerings.rows, start=1)
        ]
        courses = _reconcile_catalog_rows(
            "course_entities",
            course_entity_inputs,
            key_for=lambda row: (str(row.get("code") or ""), str(row["name"])),
        )
        course_rows: list[tuple[str, str | None, str]] = [
            (
                _course_id(
                    str(course.get("code") or "").upper() or None,
                    str(course["name"]),
                ),
                str(course.get("code") or "").upper() or None,
                str(course["name"]),
            )
            for course in courses.rows
        ]
        course_alias_rows: set[tuple[str, str, str]] = set()
        course_alias_input_count = 0
        offering_rows: list[tuple[object, ...]] = []
        for course in course_offerings.rows:
            cohort = int(str(course["cohort"]))
            program_id = _program_id(str(course["major"]), str(course["college"]), cohort)
            if program_id not in program_ids:
                raise DataIntegrityError(
                    "catalog course references a program without a reconciled plan: "
                    f"major={course['major']!r}, college={course['college']!r}, cohort={cohort!r}"
                )
            course_module_id = module_map[(program_id, str(course["module"]))]
            code = str(course.get("code") or "").upper() or None
            name = str(course["name"])
            course_id = _course_id(code, name)
            course_alias_rows.add((name, _normalized(name), course_id))
            course_alias_input_count += 1
            if code:
                course_alias_rows.add((code, _normalized(code), course_id))
                course_alias_input_count += 1
            for alias, target_name in aliases["course_aliases"].items():
                if target_name == name:
                    course_alias_rows.add((alias, _normalized(alias), course_id))
                    course_alias_input_count += 1
            evidence = course.get("evidence")
            source_id = _source_for(course["source_title"], cohort, source_ids)
            record_id = stable_id(
                "offering",
                program_id,
                course_module_id,
                course_id,
                course.get("semester"),
                course.get("page"),
                course.get("source_row"),
            )
            review_status, field_rows = _field_verification_rows(
                entity_type="program_course",
                record_id=record_id,
                source_id=source_id,
                source_trust=source_trusts[source_id],
                evidence=evidence,
                section_statuses=section_statuses,
                strict_fields=STRICT_COURSE_FIELDS,
                catalog_row=course,
                field_values={
                    "module": course.get("module"),
                    "credits": course.get("credits"),
                    "semester": _semester(course.get("semester")),
                    "nature": course.get("nature"),
                },
                expected_page=course.get("page"),
            )
            field_verification_rows.extend(field_rows)
            offering_rows.append(
                (
                    record_id,
                    program_id,
                    course_module_id,
                    course_id,
                    course.get("nature"),
                    _semester(course.get("semester")),
                    course.get("credits"),
                    course.get("weekly_hours"),
                    course.get("total_hours"),
                    course.get("teaching_hours"),
                    course.get("practice_hours"),
                    course.get("department"),
                    source_id,
                    course.get("page"),
                    course.get("source_row"),
                    _evidence_chunk_id(evidence),
                    PARSER_VERSION,
                    0.95 if evidence else 0.7,
                    review_status,
                )
            )
        connection.executemany("INSERT INTO courses VALUES (?, ?, ?)", course_rows)
        connection.executemany("INSERT INTO course_aliases VALUES (?, ?, ?)", sorted(course_alias_rows))
        connection.executemany(
            "INSERT INTO program_courses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            offering_rows,
        )
        connection.executemany(
            "INSERT INTO field_verifications VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            field_verification_rows,
        )
        chunks_sha256 = _sha(chunk_file) or ""
        source_review_sha256 = (_sha(source_review_file) or "") if source_review_file else ""
        evidence_review_sha256 = (
            (_sha(evidence_review_file) or "") if evidence_review_file else ""
        )
        def counts(*, input_count: int, accepted: int, duplicates: int, quarantined: int = 0) -> dict[str, int]:
            result = {
                "input": input_count,
                "accepted": accepted,
                "exact_duplicates": duplicates,
                "quarantined": quarantined,
            }
            if result["input"] != result["accepted"] + result["exact_duplicates"] + result["quarantined"]:
                raise DataIntegrityError(f"reconciliation count conservation violated: {result!r}")
            return result

        reconciliation_counts = {
            "plans": plans.as_metadata(),
            "programs": programs.as_metadata(),
            "modules": modules.as_metadata(),
            "requirements": counts(
                input_count=requirements.input_count,
                accepted=len(requirement_rows),
                duplicates=requirements.exact_duplicate_count,
                quarantined=quarantined_requirement_count,
            ),
            "course_offerings": course_offerings.as_metadata(),
            "course_entities": courses.as_metadata(),
            "program_aliases": counts(
                input_count=program_alias_input_count,
                accepted=len(alias_rows),
                duplicates=max(
                    0,
                    program_alias_input_count - len(alias_rows),
                ),
            ),
            "module_aliases": counts(
                input_count=module_alias_input_count,
                accepted=len(module_alias_rows),
                duplicates=max(0, module_alias_input_count - len(module_alias_rows)),
            ),
            "course_aliases": counts(
                input_count=course_alias_input_count,
                accepted=len(course_alias_rows),
                duplicates=max(0, course_alias_input_count - len(course_alias_rows)),
            ),
        }
        field_status_counts = {
            status: sum(row[10] == status for row in field_verification_rows)
            for status in REVIEW_STATUSES
        }
        source_authenticity_counts = {
            status: sum(trust.authenticity_status == status for trust in source_trusts.values())
            for status in REVIEW_STATUSES
        }
        extraction_quality_counts = {
            status: sum(row[1] == status for row in section_qualities)
            for status in EXTRACTION_QUALITY_STATUSES
        }
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "parser_version": PARSER_VERSION,
            "dataset_version": str(catalog.get("catalog_version", "unknown")),
            "catalog_sha256": _sha(catalog_file) or "",
            "sources_sha256": _sha(source_file) or "",
            "chunks_sha256": chunks_sha256,
            "source_review_sha256": source_review_sha256,
            "evidence_review_sha256": evidence_review_sha256,
            "evidence_state_sha256": _combined_sha(
                chunks_sha256, source_review_sha256, evidence_review_sha256
            ),
            "reconciliation_contract": "input=accepted+exact_duplicates+quarantined",
            "reconciliation_counts": json.dumps(
                reconciliation_counts, ensure_ascii=False, sort_keys=True
            ),
            "source_authenticity_counts": json.dumps(
                source_authenticity_counts, ensure_ascii=False, sort_keys=True
            ),
            "extraction_quality_counts": json.dumps(
                extraction_quality_counts, ensure_ascii=False, sort_keys=True
            ),
            "field_verification_counts": json.dumps(
                field_status_counts, ensure_ascii=False, sort_keys=True
            ),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        connection.executemany("INSERT INTO metadata VALUES (?, ?)", metadata.items())
        connection.commit()
        report = {
            "program_count": connection.execute("SELECT count(*) FROM programs").fetchone()[0],
            "course_count": connection.execute("SELECT count(*) FROM courses").fetchone()[0],
            "offering_count": connection.execute("SELECT count(*) FROM program_courses").fetchone()[
                0
            ],
            "requirement_count": connection.execute("SELECT count(*) FROM requirements").fetchone()[
                0
            ],
            "chunk_count": connection.execute("SELECT count(*) FROM source_sections").fetchone()[0],
            "verified_chunk_count": connection.execute(
                "SELECT count(*) FROM source_sections WHERE review_status='verified'"
            ).fetchone()[0],
            **metadata,
            "quarantined_requirement_count": quarantined_requirement_count,
            "reconciliation_counts": reconciliation_counts,
            "field_verification_counts": field_status_counts,
        }
    finally:
        connection.close()
    temporary.replace(target)
    return {**report, "database_path": str(target.resolve())}


@dataclass(frozen=True)
class CourseRecord:
    record_id: str
    course_id: str
    code: str | None
    name: str
    credits: float | None
    semester: str
    nature: str | None
    module_id: str
    module_name: str
    department: str | None
    source_id: str
    source_page: int | None
    chunk_id: str | None


class AcademicRepository:
    """Thread-safe read-only access to the canonical projection."""

    def __init__(self, path: str | Path = DEFAULT_DATABASE) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"academic database not found: {self.path}; run python -m scripts.build_all"
            )
        self._connection = sqlite3.connect(
            f"file:{self.path.resolve().as_posix()}?mode=ro", uri=True, check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        self._lock = RLock()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cursor = self._connection.cursor()
            try:
                yield cursor
            finally:
                cursor.close()

    def _all(self, statement: str, values: Iterable[object] = ()) -> list[sqlite3.Row]:
        with self._cursor() as cursor:
            return cursor.execute(statement, tuple(values)).fetchall()

    def _one(self, statement: str, values: Iterable[object] = ()) -> sqlite3.Row | None:
        with self._cursor() as cursor:
            return cast(sqlite3.Row | None, cursor.execute(statement, tuple(values)).fetchone())

    def metadata(self) -> dict[str, str]:
        return {row["key"]: row["value"] for row in self._all("SELECT key, value FROM metadata")}

    def _has_table(self, name: str) -> bool:
        return self._one(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ) is not None

    def _legacy_untyped_taxonomy_compatibility(self) -> bool:
        """Allow only unversioned hand-built test databases to remain readable.

        A released legacy database has a schema-version metadata row and is
        intentionally fail-closed when it lacks the taxonomy migration.  The
        narrow compatibility branch exists for direct-SQL unit-test fixtures,
        which historically create ``SCHEMA`` but omit all metadata.
        """

        return self._one("SELECT 1 FROM metadata WHERE key='schema_version'") is None

    def evidence_readiness(self) -> tuple[bool, tuple[str, ...]]:
        """Check that verified evidence supports the agent's core curriculum work.

        A database can be syntactically complete while every chunk is still
        pending review.  Health readiness therefore requires one current
        active program whose full course and requirement projections are
        field-verified and tied to verified source sections from the same
        source, plus a verified current policy chunk that is not merely the
        provenance of one structured course or requirement.  This is
        deliberately stronger than checking file existence.
        """

        def count(statement: str) -> int:
            row = self._one(statement)
            return int(row[0]) if row is not None else 0

        verified_sections = count(
            """
            SELECT count(*)
            FROM source_sections ss
            JOIN source_authenticity sa ON sa.source_id=ss.source_id
            JOIN section_extraction_quality sq ON sq.chunk_id=ss.chunk_id
            WHERE ss.review_status='verified'
              AND sa.authenticity_status='verified'
              AND sq.extraction_quality='verified'
            """
        )
        verified_courses = count(
            """
            SELECT count(*)
            FROM program_courses pc
            JOIN source_sections ss
              ON ss.chunk_id=pc.chunk_id AND ss.source_id=pc.source_id
            JOIN source_authenticity sa ON sa.source_id=pc.source_id
            JOIN section_extraction_quality sq ON sq.chunk_id=pc.chunk_id
            WHERE pc.review_status='verified' AND ss.review_status='verified'
              AND sa.authenticity_status='verified'
              AND sq.extraction_quality='verified'
              AND (
                SELECT count(DISTINCT fv.field_name)
                FROM field_verifications fv
                WHERE fv.entity_type='program_course' AND fv.record_id=pc.record_id
                  AND fv.field_name IN ('module', 'credits', 'semester', 'nature')
                  AND fv.verification_status='verified'
              ) = 4
            """
        )
        verified_requirements = count(
            """
            SELECT count(*)
            FROM requirements r
            JOIN source_sections ss
              ON ss.chunk_id=r.chunk_id AND ss.source_id=r.source_id
            JOIN source_authenticity sa ON sa.source_id=r.source_id
            JOIN section_extraction_quality sq ON sq.chunk_id=r.chunk_id
            WHERE r.required_credits IS NOT NULL
              AND r.review_status='verified'
              AND ss.review_status='verified'
              AND sa.authenticity_status='verified'
              AND sq.extraction_quality='verified'
              AND (
                SELECT count(DISTINCT fv.field_name)
                FROM field_verifications fv
                WHERE fv.entity_type='requirement' AND fv.record_id=r.record_id
                  AND fv.field_name IN ('module', 'required_credits')
                  AND fv.verification_status='verified'
              ) = 2
            """
        )
        verified_policy_evidence = count(
            """
            SELECT count(*)
            FROM source_sections ss
            JOIN sources s ON s.source_id=ss.source_id
            JOIN source_taxonomy st ON st.source_id=s.source_id
            JOIN source_authenticity sa ON sa.source_id=s.source_id
            JOIN section_extraction_quality sq ON sq.chunk_id=ss.chunk_id
            WHERE ss.review_status='verified'
              AND s.status='现行'
              AND st.doc_type IN ('policy', 'notice', 'guide')
              AND sa.authenticity_status='verified'
              AND sq.extraction_quality='verified'
              AND NOT EXISTS (
                SELECT 1 FROM program_courses pc WHERE pc.chunk_id=ss.chunk_id
              )
              AND NOT EXISTS (
                SELECT 1 FROM requirements r WHERE r.chunk_id=ss.chunk_id
              )
            """
        )
        active_program_rows = self._all(
            """
            SELECT p.program_id
            FROM programs p
            JOIN sources ps ON ps.source_id=p.source_id
            WHERE ps.status='现行'
            ORDER BY p.program_id
            """
        )
        answerable_program_rows = self._all(
            """
            SELECT p.program_id
            FROM programs p
            JOIN sources ps ON ps.source_id=p.source_id
            WHERE ps.status='现行'
              AND EXISTS (
                SELECT 1 FROM source_authenticity psa
                WHERE psa.source_id=p.source_id
                  AND psa.authenticity_status='verified'
              )
              AND EXISTS (
                SELECT 1 FROM program_courses pc
                WHERE pc.program_id=p.program_id
              )
              AND NOT EXISTS (
                SELECT 1
                FROM program_courses pc
                WHERE pc.program_id=p.program_id
                  AND NOT (
                    pc.review_status='verified'
                    AND EXISTS (
                      SELECT 1
                      FROM source_sections ss
                      JOIN source_authenticity sa ON sa.source_id=ss.source_id
                      JOIN section_extraction_quality sq ON sq.chunk_id=ss.chunk_id
                      WHERE ss.chunk_id=pc.chunk_id
                        AND ss.source_id=pc.source_id
                        AND ss.review_status='verified'
                        AND sa.authenticity_status='verified'
                        AND sq.extraction_quality='verified'
                    )
                    AND (
                      SELECT count(DISTINCT fv.field_name)
                      FROM field_verifications fv
                      WHERE fv.entity_type='program_course'
                        AND fv.record_id=pc.record_id
                        AND fv.field_name IN ('module', 'credits', 'semester', 'nature')
                        AND fv.verification_status='verified'
                    ) = 4
                  )
              )
              AND EXISTS (
                SELECT 1 FROM requirements r
                WHERE r.program_id=p.program_id
              )
              AND NOT EXISTS (
                SELECT 1
                FROM requirements r
                WHERE r.program_id=p.program_id
                  AND NOT (
                    r.required_credits IS NOT NULL
                    AND r.review_status='verified'
                    AND EXISTS (
                      SELECT 1
                      FROM source_sections ss
                      JOIN source_authenticity sa ON sa.source_id=ss.source_id
                      JOIN section_extraction_quality sq ON sq.chunk_id=ss.chunk_id
                      WHERE ss.chunk_id=r.chunk_id
                        AND ss.source_id=r.source_id
                        AND ss.review_status='verified'
                        AND sa.authenticity_status='verified'
                        AND sq.extraction_quality='verified'
                    )
                    AND (
                      SELECT count(DISTINCT fv.field_name)
                      FROM field_verifications fv
                      WHERE fv.entity_type='requirement'
                        AND fv.record_id=r.record_id
                        AND fv.field_name IN ('module', 'required_credits')
                        AND fv.verification_status='verified'
                    ) = 2
                  )
              )
            ORDER BY p.program_id
            """
        )
        active_program_ids = {
            str(row["program_id"]) for row in active_program_rows
        }
        answerable_program_ids = {
            str(row["program_id"]) for row in answerable_program_rows
        }
        unanswerable_program_ids = sorted(
            active_program_ids - answerable_program_ids
        )
        reasons: list[str] = []
        if not verified_sections:
            reasons.append("verified_evidence_missing")
        if not verified_courses:
            reasons.append("verified_course_evidence_missing")
        if not verified_requirements:
            reasons.append("verified_requirement_evidence_missing")
        if not verified_policy_evidence:
            reasons.append("verified_policy_evidence_missing")
        if not active_program_ids:
            reasons.append("active_program_missing")
        if not answerable_program_ids:
            reasons.append("core_business_unanswerable")
        if unanswerable_program_ids:
            reasons.append("program_readiness_incomplete")
            reasons.extend(
                f"program_unanswerable:{program_id}"
                for program_id in unanswerable_program_ids
            )
        return not reasons, tuple(reasons)

    def options(self) -> dict[str, object]:
        rows = self._all(
            "SELECT cohort, canonical_name, college_id FROM programs ORDER BY cohort, canonical_name"
        )
        values: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            values.setdefault(str(row["cohort"]), []).append(
                {
                    "id": row["canonical_name"],
                    "name": row["canonical_name"],
                    "college": row["college_id"],
                }
            )
        return {"dataset": self.metadata(), "programs_by_cohort": values}

    @staticmethod
    def _matched_entities(
        kind: Literal["program", "course", "module"], target: str, rows: list[sqlite3.Row]
    ) -> tuple[ResolvedEntity, ...]:
        exact = [row for row in rows if str(row["alias"]) == target]
        matches = exact or [
            row
            for row in rows
            if str(row["alias"]) and (str(row["alias"]) in target or target in str(row["alias"]))
        ]
        confidence = 1.0 if exact else 0.85
        values: dict[str, ResolvedEntity] = {}
        for row in matches:
            identifier = str(row["id"])
            values.setdefault(
                identifier,
                ResolvedEntity(
                    entity_type=kind,
                    canonical_id=identifier,
                    canonical_name=str(row["name"]),
                    confidence=confidence,
                ),
            )
        return tuple(
            sorted(values.values(), key=lambda item: (item.canonical_name, item.canonical_id))
        )

    def resolve_program_candidates(
        self, mention: str, cohort: int | None = None
    ) -> tuple[ResolvedEntity, ...]:
        target = _normalized(mention)
        if not target:
            return ()
        rows = self._all(
            """
            SELECT p.program_id AS id, p.canonical_name AS name, a.normalized_alias AS alias
            FROM program_aliases a JOIN programs p ON p.program_id=a.program_id
            WHERE (? IS NULL OR p.cohort=?)
            """,
            (cohort, cohort),
        )
        return self._matched_entities("program", target, rows)

    def resolve_program(self, mention: str, cohort: int | None = None) -> ResolvedEntity | None:
        direct = self._one(
            "SELECT program_id, canonical_name FROM programs WHERE program_id=? AND (? IS NULL OR cohort=?)",
            (mention, cohort, cohort),
        )
        if direct:
            return ResolvedEntity(
                entity_type="program",
                canonical_id=str(direct["program_id"]),
                canonical_name=str(direct["canonical_name"]),
                confidence=1.0,
            )
        candidates = self.resolve_program_candidates(mention, cohort)
        return candidates[0] if len(candidates) == 1 else None

    def resolve_course_candidates(
        self, mention: str, cohort: int | None = None, program_id: str | None = None
    ) -> tuple[ResolvedEntity, ...]:
        target = _normalized(mention)
        if not target:
            return ()
        rows = self._all(
            """
            SELECT DISTINCT c.course_id AS id, c.canonical_name AS name, a.normalized_alias AS alias
            FROM course_aliases a
            JOIN courses c ON c.course_id=a.course_id
            JOIN program_courses pc ON pc.course_id=c.course_id
            JOIN programs p ON p.program_id=pc.program_id
            WHERE (? IS NULL OR p.cohort=?)
              AND (? IS NULL OR pc.program_id=?)
            """,
            (cohort, cohort, program_id, program_id),
        )
        return self._matched_entities("course", target, rows)

    def resolve_course(
        self, mention: str, cohort: int | None = None, program_id: str | None = None
    ) -> ResolvedEntity | None:
        candidates = self.resolve_course_candidates(mention, cohort, program_id)
        return candidates[0] if len(candidates) == 1 else None

    def resolve_module_candidates(
        self, mention: str, program_id: str | None = None
    ) -> tuple[ResolvedEntity, ...]:
        target = _normalized(mention)
        if not target:
            return ()
        rows = self._all(
            """
            SELECT DISTINCT m.module_id AS id, m.canonical_name AS name, a.normalized_alias AS alias
            FROM module_aliases a JOIN modules m ON m.module_id=a.module_id
            WHERE (? IS NULL OR m.program_id=?)
            """,
            (program_id, program_id),
        )
        return self._matched_entities("module", target, rows)

    def resolve_module(self, mention: str, program_id: str | None = None) -> ResolvedEntity | None:
        candidates = self.resolve_module_candidates(mention, program_id)
        return candidates[0] if len(candidates) == 1 else None

    def course_mentions_in_text(
        self, text: str, cohort: int | None = None, program_id: str | None = None
    ) -> tuple[tuple[str, tuple[ResolvedEntity, ...]], ...]:
        """Return every in-text course alias with its scoped candidate set.

        Unlike ``courses_in_text``, this preserves ambiguity so the normalizer
        can request a clarification instead of silently dropping a course name.
        """

        normalized_text = _normalized(text)
        if not normalized_text:
            return ()
        rows = self._all(
            """
            SELECT DISTINCT a.normalized_alias AS alias
            FROM course_aliases a
            JOIN program_courses pc ON pc.course_id=a.course_id
            JOIN programs p ON p.program_id=pc.program_id
            WHERE (? IS NULL OR p.cohort=?)
              AND (? IS NULL OR pc.program_id=?)
            """,
            (cohort, cohort, program_id, program_id),
        )
        values: list[tuple[int, str, tuple[ResolvedEntity, ...]]] = []
        for alias in {str(row["alias"]) for row in rows if len(str(row["alias"])) >= 2}:
            position = normalized_text.find(alias)
            if position < 0:
                continue
            candidates = self.resolve_course_candidates(alias, cohort, program_id)
            if candidates:
                values.append((position, alias, candidates))
        return tuple(
            (alias, candidates)
            for _position, alias, candidates in sorted(
                values, key=lambda item: (item[0], -len(item[1]), item[1])
            )
        )

    def courses_in_text(
        self, text: str, cohort: int | None = None, program_id: str | None = None
    ) -> tuple[ResolvedEntity, ...]:
        normalized_text = _normalized(text)
        if not normalized_text:
            return ()
        rows = self._all(
            """
            SELECT DISTINCT a.normalized_alias AS alias
            FROM course_aliases a
            JOIN program_courses pc ON pc.course_id=a.course_id
            JOIN programs p ON p.program_id=pc.program_id
            WHERE (? IS NULL OR p.cohort=?)
              AND (? IS NULL OR pc.program_id=?)
            """,
            (cohort, cohort, program_id, program_id),
        )
        matches: list[tuple[int, int, ResolvedEntity]] = []
        for alias in {str(row["alias"]) for row in rows if len(str(row["alias"])) >= 2}:
            position = normalized_text.find(alias)
            if position < 0:
                continue
            candidates = self.resolve_course_candidates(alias, cohort, program_id)
            if len(candidates) == 1:
                matches.append((position, -len(alias), candidates[0]))
        values: dict[str, ResolvedEntity] = {}
        for _position, _length, value in sorted(
            matches, key=lambda item: (item[0], item[1], item[2].canonical_id)
        ):
            values.setdefault(value.canonical_id, value)
        return tuple(values.values())

    def modules_in_text(
        self, text: str, program_id: str | None = None
    ) -> tuple[ResolvedEntity, ...]:
        normalized_text = _normalized(text)
        if not normalized_text:
            return ()
        rows = self._all(
            """
            SELECT DISTINCT a.normalized_alias AS alias
            FROM module_aliases a JOIN modules m ON m.module_id=a.module_id
            WHERE (? IS NULL OR m.program_id=?)
            """,
            (program_id, program_id),
        )
        matches: list[tuple[int, int, ResolvedEntity]] = []
        for alias in {str(row["alias"]) for row in rows if len(str(row["alias"])) >= 2}:
            position = normalized_text.find(alias)
            if position < 0:
                continue
            candidates = self.resolve_module_candidates(alias, program_id)
            if len(candidates) == 1:
                matches.append((position, -len(alias), candidates[0]))
        values: dict[str, ResolvedEntity] = {}
        for _position, _length, value in sorted(
            matches, key=lambda item: (item[0], item[1], item[2].canonical_id)
        ):
            values.setdefault(value.canonical_id, value)
        return tuple(values.values())

    def programs_in_text(self, text: str, cohort: int | None = None) -> tuple[ResolvedEntity, ...]:
        normalized_text = _normalized(text)
        rows = self._all(
            "SELECT DISTINCT a.alias, p.canonical_name FROM program_aliases a "
            "JOIN programs p ON p.program_id=a.program_id"
            + (" WHERE p.cohort=?" if cohort else ""),
            (cohort,) if cohort else (),
        )
        matches: list[tuple[int, int, ResolvedEntity]] = []
        for row in rows:
            alias = _normalized(str(row["alias"]))
            position = normalized_text.find(alias)
            if position < 0:
                continue
            canonical_name = _normalized(str(row["canonical_name"]))
            canonical_stem = canonical_name.removesuffix("专业")
            if (
                canonical_name != canonical_stem
                and alias == canonical_stem
                and f"{canonical_stem}专业" not in normalized_text
            ):
                # Do not infer 英语专业 from 大学英语, or similarly overload a
                # bare canonical stem that is also ordinary academic wording.
                continue
            resolved = self.resolve_program(str(row["alias"]), cohort)
            if resolved:
                matches.append((position, -len(alias), resolved))
        values: dict[str, ResolvedEntity] = {}
        for _position, _length, value in sorted(
            matches, key=lambda item: (item[0], item[1], item[2].canonical_id)
        ):
            values.setdefault(value.canonical_id, value)
        return tuple(values.values())

    def list_courses(
        self,
        *,
        cohort: int,
        program_id: str | None = None,
        semesters: tuple[int, ...] = (),
        natures: tuple[str, ...] = (),
        module_ids: tuple[str, ...] = (),
        course_ids: tuple[str, ...] = (),
    ) -> tuple[CourseRecord, ...]:
        clauses = ["p.cohort=?"]
        params: list[object] = [cohort]
        if program_id is not None:
            clauses.append("pc.program_id=?")
            params.append(program_id)
        if semesters:
            placeholders = ",".join("?" for _ in semesters)
            clauses.append(f"CAST(substr(pc.semester, 1, 1) AS INTEGER) IN ({placeholders})")
            params.extend(semesters)
        if natures:
            conditions = []
            for nature in natures:
                if nature == "elective":
                    conditions.append(
                        "(pc.course_nature LIKE '%选修%' OR m.canonical_name LIKE '%方向%')"
                    )
                elif nature == "free_elective":
                    conditions.append("m.canonical_name LIKE '%自由选修%'")
                else:
                    conditions.append("pc.course_nature LIKE '%必修%'")
            clauses.append("(" + " OR ".join(conditions) + ")")
        if module_ids:
            placeholders = ",".join("?" for _ in module_ids)
            clauses.append(f"pc.module_id IN ({placeholders})")
            params.extend(module_ids)
        if course_ids:
            placeholders = ",".join("?" for _ in course_ids)
            clauses.append(f"pc.course_id IN ({placeholders})")
            params.extend(course_ids)
        rows = self._all(
            f"""
            SELECT pc.record_id, pc.course_id, c.canonical_code AS code, c.canonical_name AS name, pc.credits, pc.semester,
                   pc.course_nature AS nature, pc.module_id, m.canonical_name AS module_name, pc.department,
                   pc.source_id, pc.source_page, pc.chunk_id
            FROM program_courses pc JOIN programs p ON p.program_id=pc.program_id
            JOIN courses c ON c.course_id=pc.course_id JOIN modules m ON m.module_id=pc.module_id
            WHERE {" AND ".join(clauses)}
            ORDER BY CAST(substr(pc.semester, 1, 1) AS INTEGER), m.canonical_name, c.canonical_code, c.canonical_name
        """,
            params,
        )
        return tuple(CourseRecord(**dict(row)) for row in rows)

    def requirements(
        self, *, cohort: int, program_id: str, module_ids: tuple[str, ...] = ()
    ) -> list[dict[str, object]]:
        clauses = ["p.cohort=?", "r.program_id=?"]
        params: list[object] = [cohort, program_id]
        if module_ids:
            placeholders = ",".join("?" for _ in module_ids)
            clauses.append(f"r.module_id IN ({placeholders})")
            params.extend(module_ids)
        return [
            dict(row)
            for row in self._all(
                f"""
            SELECT r.*, m.canonical_name AS module_name FROM requirements r JOIN programs p ON p.program_id=r.program_id
            JOIN modules m ON m.module_id=r.module_id WHERE {" AND ".join(clauses)} ORDER BY m.canonical_name
        """,
                params,
            )
        ]

    def source(self, chunk_id: str) -> dict[str, object] | None:
        row = self._one(
            """
            SELECT ss.*, s.title, s.page_url, s.file_url, s.source_sha256, s.effective_from, s.effective_to, s.authority_level, s.status, s.cohort, s.college_id
            FROM source_sections ss JOIN sources s ON s.source_id=ss.source_id WHERE ss.chunk_id=?
        """,
            (chunk_id,),
        )
        return dict(row) if row else None

    def college_ids_for_programs(self, program_ids: tuple[str, ...]) -> tuple[str, ...]:
        """Return the deterministic college scope represented by program ids."""

        if not program_ids:
            return ()
        placeholders = ",".join("?" for _ in program_ids)
        rows = self._all(
            f"SELECT program_id, college_id FROM programs WHERE program_id IN ({placeholders})",
            program_ids,
        )
        college_by_program = {str(row["program_id"]): str(row["college_id"]) for row in rows}
        return tuple(
            dict.fromkeys(
                college_by_program[program_id]
                for program_id in program_ids
                if program_id in college_by_program
            )
        )

    def program_ids_for_colleges(
        self, college_ids: tuple[str, ...], cohort: int | None = None
    ) -> tuple[str, ...]:
        """Return programs in requested colleges, optionally constrained to a cohort."""

        if not college_ids:
            return ()
        placeholders = ",".join("?" for _ in college_ids)
        clauses = [f"college_id IN ({placeholders})"]
        values: list[object] = list(college_ids)
        if cohort is not None:
            clauses.append("cohort=?")
            values.append(cohort)
        rows = self._all(
            "SELECT program_id FROM programs WHERE "
            + " AND ".join(clauses)
            + " ORDER BY program_id",
            values,
        )
        return tuple(str(row["program_id"]) for row in rows)

    def retrieval_documents(self) -> tuple[dict[str, object], ...]:
        """Expose only explicitly taxonomy-approved policy evidence.

        The index backing the policy route should never contain curriculum or
        course-catalog passages.  Taxonomy is source-registry data, not a
        title/text heuristic.  The only exception is an unversioned direct-SQL
        compatibility database used by historical unit tests; released legacy
        artifacts remain fail-closed.
        """

        has_taxonomy = self._has_table("source_taxonomy")
        has_authenticity = self._has_table("source_authenticity")
        has_quality = self._has_table("section_extraction_quality")
        legacy_untyped = self._legacy_untyped_taxonomy_compatibility()
        taxonomy_join = (
            "LEFT JOIN source_taxonomy st ON st.source_id=s.source_id"
            if has_taxonomy
            else ""
        )
        authenticity_join = (
            "LEFT JOIN source_authenticity sa ON sa.source_id=s.source_id"
            if has_authenticity
            else ""
        )
        quality_join = (
            "LEFT JOIN section_extraction_quality sq ON sq.chunk_id=ss.chunk_id"
            if has_quality
            else ""
        )
        doc_type = "COALESCE(st.doc_type, 'unknown')" if has_taxonomy else "'unknown'"
        topics = "COALESCE(st.topics_json, '[]')" if has_taxonomy else "'[]'"
        authenticity = (
            "COALESCE(sa.authenticity_status, 'unverified')"
            if has_authenticity
            else "'unverified'"
        )
        quality = (
            "COALESCE(sq.extraction_quality, 'review_required')"
            if has_quality
            else "'review_required'"
        )
        policy_type_sql = ", ".join(repr(item) for item in sorted(POLICY_DOCUMENT_TYPES))
        policy_filter = "" if legacy_untyped else f"WHERE {doc_type} IN ({policy_type_sql})"
        rows = self._all(
            f"""
            SELECT ss.chunk_id, ss.text, ss.source_id, ss.article, ss.physical_page, ss.parser_version, ss.extracted_at, ss.confidence,
                   ss.review_status, s.title, s.page_url, s.file_url, s.college_id,
                   s.cohort, s.authority_level, s.effective_from, s.effective_to,
                   s.status, s.supersedes_source_id, s.source_sha256,
                   {doc_type} AS doc_type, {topics} AS topics_json,
                   {authenticity} AS source_authenticity, {quality} AS extraction_quality
            FROM source_sections ss
            JOIN sources s ON s.source_id=ss.source_id
            {taxonomy_join}
            {authenticity_join}
            {quality_join}
            {policy_filter}
            ORDER BY ss.chunk_id
            """
        )
        program_rows = self._all(
            "SELECT source_id, program_id FROM programs ORDER BY source_id, program_id"
        )
        grouped: dict[str, list[str]] = {}
        for program_row in program_rows:
            grouped.setdefault(str(program_row["source_id"]), []).append(
                str(program_row["program_id"])
            )
        programs_by_source = {source_id: tuple(values) for source_id, values in grouped.items()}
        documents: list[dict[str, object]] = []
        for row in rows:
            try:
                decoded_topics = json.loads(str(row["topics_json"] or "[]"))
            except json.JSONDecodeError:
                decoded_topics = []
            topics_value = (
                tuple(topic for topic in decoded_topics if isinstance(topic, str))
                if isinstance(decoded_topics, list)
                else ()
            )
            document: dict[str, object] = {
                "chunk_id": str(row["chunk_id"]),
                "text": str(row["text"]),
                "source_id": str(row["source_id"]),
                "title": str(row["title"]),
                "article": str(row["article"] or ""),
                "physical_page": row["physical_page"],
                "page_url": str(row["page_url"] or "") or None,
                "file_url": str(row["file_url"] or "") or None,
                "review_status": str(row["review_status"]),
                "college_id": str(row["college_id"] or ""),
                "cohort": str(row["cohort"]),
                "authority_level": int(row["authority_level"]),
                "effective_from": str(row["effective_from"] or "") or None,
                "effective_to": str(row["effective_to"] or "") or None,
                "status": str(row["status"]),
                "supersedes_source_id": str(row["supersedes_source_id"] or "") or None,
                "source_sha256": str(row["source_sha256"] or "") or None,
                "parser_version": str(row["parser_version"]),
                "extracted_at": str(row["extracted_at"]),
                "confidence": float(row["confidence"]),
                "doc_type": str(row["doc_type"]),
                "topics": topics_value,
                "source_authenticity": str(row["source_authenticity"]),
                "extraction_quality": str(row["extraction_quality"]),
            }
            program_ids = programs_by_source.get(str(row["source_id"]), ())
            if program_ids:
                document["program_ids"] = program_ids
            documents.append(document)
        return tuple(documents)

    @staticmethod
    def _effective_boundary(value: object, *, end: bool) -> str | None:
        normalized = str(value or "").strip()
        if not normalized:
            return None
        if re.fullmatch(r"\d{4}", normalized):
            return f"{normalized}-12-31" if end else f"{normalized}-01-01"
        return normalized

    def scoped_policy_documents(
        self,
        *,
        cohort: int | None = None,
        program_ids: tuple[str, ...] = (),
        college_ids: tuple[str, ...] = (),
        as_of: str | None = None,
    ) -> tuple[dict[str, object], ...]:
        """Filter retrieval documents by explicit cohort, program and college scope."""

        effective_date = self._policy_as_of(as_of)
        requested_colleges = tuple(
            dict.fromkeys((*college_ids, *self.college_ids_for_programs(program_ids)))
        )
        scoped: list[dict[str, object]] = []
        for document in self.retrieval_documents():
            if str(document.get("doc_type") or "unknown") not in POLICY_DOCUMENT_TYPES:
                # ``retrieval_documents`` already enforces this for a released
                # projection.  Keep a second hard guard so a compatibility
                # source cannot leak an untyped curriculum document through a
                # scoped policy call.
                continue
            if not policy_scope_matches(
                document,
                cohort=cohort,
                program_ids=program_ids,
                college_ids=requested_colleges,
            ):
                continue
            if as_of is None and str(document["status"]) != "现行":
                continue
            effective_from = self._effective_boundary(document.get("effective_from"), end=False)
            effective_to = self._effective_boundary(document.get("effective_to"), end=True)
            if effective_from is not None and effective_from > effective_date:
                continue
            if effective_to is not None and effective_to < effective_date:
                continue
            scoped.append(document)
        return tuple(scoped)

    @staticmethod
    def _policy_as_of(as_of: str | None) -> str:
        """Normalize the version boundary used by policy retrieval."""

        if as_of is None:
            return datetime.now(timezone.utc).date().isoformat()
        value = as_of.strip()
        if re.fullmatch(r"\d{4}", value):
            return f"{value}-12-31"
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
        except ValueError as exc:
            raise ValueError("as_of must be an ISO date or four-digit year") from exc

    @staticmethod
    def _policy_scope_key(row: dict[str, object]) -> str:
        article = re.sub(r"(?:原文件)?第\s*\d+\s*页", "", str(row.get("article") or ""))
        return _normalized(article) or _normalized(row.get("title"))

    @staticmethod
    def _policy_value_signature(text: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Extract comparable policy values without relying on an LLM."""

        value = str(text or "")
        numbers = tuple(
            sorted(
                {f"{float(item):g}" for item in re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", value)}
            )
        )
        codes = tuple(
            sorted({item.upper() for item in re.findall(r"\b[A-Z]{2,6}\d{2,4}\b", value, re.I)})
        )
        return numbers, codes

    @classmethod
    def policy_conflicts(cls, rows: list[dict[str, object]]) -> tuple[str, ...]:
        """Report incompatible values from equally authoritative active sources.

        Different prose is not itself a contradiction. A conflict requires the
        same college/cohort/article scope and different numeric or course-code
        values, which keeps the detector deterministic and fail-closed.
        """

        grouped: dict[tuple[int, str, str, str], list[dict[str, object]]] = {}
        for row in rows:
            scope_key = cls._policy_scope_key(row)
            if not scope_key:
                continue
            key = (
                int(str(row["authority_level"])),
                str(row.get("college_id") or ""),
                str(row.get("cohort") or ""),
                scope_key,
            )
            grouped.setdefault(key, []).append(row)

        conflicts: list[str] = []
        for _key, values in grouped.items():
            # A source may contribute multiple chunks to one section; compare
            # only its best matching section with another source.
            by_source: dict[str, dict[str, object]] = {}
            for row in values:
                by_source.setdefault(str(row["source_id"]), row)
            sources = list(by_source.values())
            for index, left in enumerate(sources):
                left_signature = cls._policy_value_signature(left.get("text"))
                if not any(left_signature):
                    continue
                for right in sources[index + 1 :]:
                    right_signature = cls._policy_value_signature(right.get("text"))
                    if not any(right_signature) or left_signature == right_signature:
                        continue
                    conflicts.append(
                        "同等权威来源冲突："
                        f"{left['title']}（{left['chunk_id']}）与"
                        f"{right['title']}（{right['chunk_id']}）"
                    )
        return tuple(conflicts)

    def policy_candidates(
        self,
        query: str,
        cohort: int | None = None,
        limit: int = 40,
        *,
        as_of: str | None = None,
    ) -> list[dict[str, object]]:
        runs = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", query)
        terms: list[str] = []
        for run in runs:
            if re.fullmatch(r"[\u4e00-\u9fff]+", run):
                for size in range(2, min(6, len(run) + 1)):
                    terms.extend(run[index : index + size] for index in range(len(run) - size + 1))
            elif len(run) > 1:
                terms.append(run)
        effective_date = self._policy_as_of(as_of)
        if not terms:
            return []
        where = " OR ".join("ss.text LIKE ?" for _ in terms)
        params: list[object] = [f"%{term}%" for term in terms]
        scope = ""
        if cohort is not None:
            scope = " AND (s.cohort=? OR s.cohort='不限')"
            params.append(str(cohort))

        # The default path only exposes sources currently in force. Historical
        # records become eligible only for an explicit as_of query.
        status_scope = "" if as_of is not None else " AND s.status='现行'"
        newer_status_scope = "" if as_of is not None else " AND newer.status='现行'"
        source_from = "CASE WHEN length(s.effective_from)=4 THEN s.effective_from || '-01-01' ELSE s.effective_from END"
        source_to = "CASE WHEN length(s.effective_to)=4 THEN s.effective_to || '-12-31' ELSE s.effective_to END"
        newer_from = "CASE WHEN length(newer.effective_from)=4 THEN newer.effective_from || '-01-01' ELSE newer.effective_from END"
        newer_to = "CASE WHEN length(newer.effective_to)=4 THEN newer.effective_to || '-12-31' ELSE newer.effective_to END"
        has_taxonomy = self._has_table("source_taxonomy")
        legacy_untyped = self._legacy_untyped_taxonomy_compatibility()
        taxonomy_join = (
            "LEFT JOIN source_taxonomy st ON st.source_id=s.source_id"
            if has_taxonomy
            else ""
        )
        taxonomy_column = "COALESCE(st.doc_type, 'unknown')" if has_taxonomy else "'unknown'"
        policy_type_sql = ", ".join(repr(item) for item in sorted(POLICY_DOCUMENT_TYPES))
        taxonomy_scope = (
            "" if legacy_untyped else f" AND {taxonomy_column} IN ({policy_type_sql})"
        )
        rows = self._all(
            f"""
            SELECT ss.*, s.title, s.page_url, s.file_url, s.authority_level, s.status, s.cohort, s.college_id,
                   s.published_at, s.effective_from, s.effective_to, s.source_sha256,
                   {taxonomy_column} AS doc_type
            FROM source_sections ss JOIN sources s ON s.source_id=ss.source_id
            {taxonomy_join}
            WHERE ({where}){scope}{status_scope}{taxonomy_scope}
              AND ({source_from} IS NULL OR {source_from}='' OR {source_from} <= ?)
              AND ({source_to} IS NULL OR {source_to}='' OR {source_to} >= ?)
              AND NOT EXISTS (
                  SELECT 1 FROM sources newer
                  WHERE newer.supersedes_source_id=s.source_id
                    AND newer.authority_level >= s.authority_level{newer_status_scope}
                    AND ({newer_from} IS NULL OR {newer_from}='' OR {newer_from} <= ?)
                    AND ({newer_to} IS NULL OR {newer_to}='' OR {newer_to} >= ?)
              )
            ORDER BY s.authority_level DESC,
                     COALESCE({source_from}, s.published_at, '') DESC,
                     ss.physical_page
            LIMIT ?
            """,
            [
                *params,
                effective_date,
                effective_date,
                effective_date,
                effective_date,
                max(limit * 12, 240),
            ],
        )
        candidates = [dict(row) for row in rows]

        def lexical_score(row: dict[str, object]) -> int:
            text = str(row["text"])
            return sum(text.count(term) * len(term) * len(term) for term in set(terms))

        def version_rank(row: dict[str, object]) -> int:
            value = str(row.get("effective_from") or row.get("published_at") or "")
            return int(re.sub(r"\D", "", value) or 0)

        candidates.sort(
            key=lambda row: (
                -lexical_score(row),
                -int(row["authority_level"]),
                -version_rank(row),
                row["physical_page"] or 0,
            )
        )
        return candidates[:limit]

    def compare_programs(
        self,
        *,
        cohort: int,
        program_ids: tuple[str, ...],
        dimensions: tuple[str, ...] = (
            "graduation_min_credits",
            "module_requirements",
            "course_sets",
        ),
    ) -> dict[str, object]:
        """Compute only the comparison dimensions requested by the typed plan."""

        requested = set(dimensions)
        program_names = {
            str(row["program_id"]): str(row["canonical_name"])
            for row in self._all(
                "SELECT program_id, canonical_name FROM programs WHERE program_id IN ("
                + ",".join("?" for _ in program_ids)
                + ")",
                program_ids,
            )
        }
        result: dict[str, object] = {"programs": program_names}
        needs_courses = bool(
            requested.intersection({"course_sets", "required_courses", "practice_requirements"})
        )
        groups: dict[str, list[CourseRecord]] = {}
        if needs_courses:
            groups = {
                program_id: list(self.list_courses(cohort=cohort, program_id=program_id))
                for program_id in program_ids
            }
        if "course_sets" in requested:
            code_sets = {
                program_id: {course.code or course.name for course in values}
                for program_id, values in groups.items()
            }
            intersection = set.intersection(*code_sets.values()) if code_sets else set()
            result["intersection"] = sorted(intersection)
            result["only_in_each"] = {
                program_id: sorted(values - intersection)
                for program_id, values in code_sets.items()
            }
        if "required_courses" in requested:
            # Preserve the producer scope. A cross-program union cannot answer
            # which requirement belongs to which program and previously made a
            # comparison look complete while discarding its central relation.
            result["required_courses_by_program"] = {
                program_id: sorted(
                    {
                        course.code or course.name
                        for course in values
                        if "必修" in (course.nature or "")
                    }
                )
                for program_id, values in groups.items()
            }
        if "practice_requirements" in requested:
            result["practice_requirements_by_program"] = {
                program_id: sorted(
                    {
                        course.code or course.name
                        for course in values
                        if "实践" in (course.nature or "")
                        or "实践" in course.module_name
                    }
                )
                for program_id, values in groups.items()
            }
        if "module_requirements" in requested or "graduation_min_credits" in requested:
            module_values: dict[str, list[dict[str, object]]] = {}
            for program_id in program_ids:
                module_values[program_id] = self.requirements(cohort=cohort, program_id=program_id)
            result["module_requirements"] = module_values
            # A catalog-level total is only canonical when an explicit rule row
            # states it. Sums are exposed separately and never mislabeled.
            explicit_totals: dict[str, float] = {}
            for program_id, rows in module_values.items():
                for row in rows:
                    rule_text = str(row.get("rule_text") or "")
                    if any(token in rule_text for token in ("毕业最低", "毕业总", "总学分")):
                        value = row.get("required_credits")
                        if value is not None:
                            explicit_totals[program_id] = float(str(value))
                            break
            if explicit_totals:
                result["graduation_min_credits"] = explicit_totals
        return result


__all__ = [
    "AcademicRepository",
    "CourseRecord",
    "DataIntegrityError",
    "DEFAULT_DATABASE",
    "build_database",
]
