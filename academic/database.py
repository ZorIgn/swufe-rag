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
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Literal, cast

from evidence.provenance import PARSER_VERSION, stable_id
from query.schemas import ResolvedEntity

ROOT = Path(__file__).parents[1]
DEFAULT_DATABASE = ROOT / "data" / "academic.sqlite3"
DEFAULT_CATALOG = ROOT / "data" / "curriculum_catalog.json"
DEFAULT_SOURCES = ROOT / "data" / "sources.csv"
DEFAULT_CHUNKS = ROOT / "data" / "chunks.jsonl"
DEFAULT_ALIAS_CONFIG = ROOT / "config" / "entity_aliases.json"
DEFAULT_SOURCE_REVIEW = ROOT / "data" / "source_review.csv"
DEFAULT_EVIDENCE_REVIEW = ROOT / "data" / "evidence_review.csv"


SCHEMA_VERSION = "1"
REVIEW_STATUSES = frozenset({"verified", "review_required", "unverified"})
VERIFIED_SOURCE_REVIEW_DECISIONS = frozenset(
    {"include", "include_ocr", "include_converted", "include_split"}
)
VERIFIED_EVIDENCE_REVIEW_DECISIONS = frozenset({"verified", "include"})


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
    source_review: SourceReview | None,
    evidence_review: EvidenceReview | None,
) -> str:
    """Use the ledger, never a chunk's self-assertion, to promote verification."""

    if evidence_review is not None:
        return (
            "verified"
            if evidence_review.decision in VERIFIED_EVIDENCE_REVIEW_DECISIONS
            else "unverified"
        )
    if source_review is not None:
        return "verified"
    claimed = _review_status(raw_status)
    # An external chunk can request a review, but cannot mark itself verified.
    return "review_required" if claimed == "verified" else claimed


def _structured_review_status(
    evidence: object,
    source_id: str,
    section_statuses: dict[str, tuple[str, str]],
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


def _load_sections(
    chunk_file: Path,
    source_ids: dict[tuple[str, str], str],
    source_rows: list[dict[str, str]],
    reviews: tuple[SourceReview, ...],
    evidence_reviews: dict[str, EvidenceReview],
) -> tuple[list[tuple[object, ...]], dict[str, tuple[str, str]]]:
    """Materialize source sections and their ledger-derived trust state."""

    reviews_by_source_id = {
        source_ids[(str(row.get("doc_title") or "").strip(), _source_scope(row.get("cohort")))]: _review_for_source(row, reviews)
        for row in source_rows
    }
    sections: list[tuple[object, ...]] = []
    section_statuses: dict[str, tuple[str, str]] = {}
    if not chunk_file.is_file():
        return sections, section_statuses
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
            review_status = _section_review_status(
                value.get("review_status"),
                reviews_by_source_id.get(source_id),
                evidence_reviews.get(chunk_id),
            )
            section_statuses[chunk_id] = (source_id, review_status)
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
    return sections, section_statuses


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


def _materialize_sources(
    connection: sqlite3.Connection, rows: list[dict[str, str]], root: Path
) -> dict[tuple[str, str], str]:
    ids: dict[tuple[str, str], str] = {}
    values: list[tuple[object, ...]] = []
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
        local = root / "data" / row["file"]
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
                _sha(local),
                row.get("collected_at"),
            )
        )
    connection.executemany(
        "INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values
    )
    return ids


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


def build_database(
    output: str | Path = DEFAULT_DATABASE,
    *,
    catalog_path: str | Path = DEFAULT_CATALOG,
    sources_path: str | Path = DEFAULT_SOURCES,
    chunks_path: str | Path = DEFAULT_CHUNKS,
    aliases_path: str | Path = DEFAULT_ALIAS_CONFIG,
    source_review_path: str | Path | None = DEFAULT_SOURCE_REVIEW,
    evidence_review_path: str | Path | None = DEFAULT_EVIDENCE_REVIEW,
) -> dict[str, object]:
    """Build a new immutable SQLite projection; generated output is not Git data."""
    target, catalog_file, source_file, chunk_file = map(
        Path, (output, catalog_path, sources_path, chunks_path)
    )
    source_review_file = Path(source_review_path) if source_review_path is not None else None
    evidence_review_file = Path(evidence_review_path) if evidence_review_path is not None else None
    catalog = json.loads(catalog_file.read_text(encoding="utf-8"))
    source_rows = _source_rows(source_file)
    _source_index(source_rows)
    aliases = _read_aliases(Path(aliases_path))
    source_reviews = _load_source_reviews(source_review_file)
    evidence_reviews = _load_evidence_reviews(evidence_review_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(SCHEMA)
        source_ids = _materialize_sources(connection, source_rows, ROOT)
        sections, section_statuses = _load_sections(
            chunk_file, source_ids, source_rows, source_reviews, evidence_reviews
        )
        connection.executemany(
            "INSERT INTO source_sections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", sections
        )
        program_rows: list[tuple[object, ...]] = []
        alias_rows: set[tuple[str, str, str]] = set()
        module_rows: list[tuple[str, str, str]] = []
        module_map: dict[tuple[str, str], str] = {}
        requirement_rows: list[tuple[object, ...]] = []
        quarantined_requirement_count = 0
        for plan in catalog.get("plans", []):
            cohort = int(plan["cohort"])
            source_id = _source_for(plan["source_title"], cohort, source_ids)
            program_id = _program_id(plan["major"], plan["college"], cohort)
            program_rows.append((program_id, plan["major"], plan["college"], cohort, source_id))
            values = {str(plan["major"]), str(plan["major"]).removesuffix("专业")}
            values.update(
                alias
                for alias, target_name in aliases["program_aliases"].items()
                if target_name == plan["major"]
            )
            alias_rows.update(
                (alias, _normalized(alias), program_id) for alias in values if _normalized(alias)
            )
            for module in plan.get("modules", []):
                module_name = str(module["name"])
                module_id = _module_id(program_id, module_name)
                module_map[(program_id, module_name)] = module_id
                module_rows.append((module_id, program_id, module_name))
                evidence = module.get("evidence")
                page = _page(evidence.get("article") if isinstance(evidence, dict) else None)
                chunk_id = _evidence_chunk_id(evidence)
                review_status = _structured_review_status(
                    evidence, source_id, section_statuses
                )
                # Untraceable requirements remain in the source catalog's
                # review queue, not in the production projection.  Keeping
                # thousands of rows with no evidence made the release verifier
                # fail while also inviting downstream tools to treat them as
                # partially usable requirements.
                if chunk_id is None or review_status == "unverified":
                    quarantined_requirement_count += 1
                    continue
                requirement_rows.append(
                    (
                        stable_id("req", program_id, module_name),
                        program_id,
                        module_id,
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
        connection.executemany(
            "INSERT OR IGNORE INTO programs VALUES (?, ?, ?, ?, ?)", program_rows
        )
        connection.executemany(
            "INSERT OR IGNORE INTO program_aliases VALUES (?, ?, ?)", sorted(alias_rows)
        )
        connection.executemany("INSERT OR IGNORE INTO modules VALUES (?, ?, ?)", module_rows)
        module_alias_rows: set[tuple[str, str, str]] = set()
        for (_program_key, module_name), module_id in module_map.items():
            module_alias_rows.add((module_name, _normalized(module_name), module_id))
            for alias, target_name in aliases["module_aliases"].items():
                if _normalized(target_name) in _normalized(module_name) or _normalized(
                    module_name
                ) in _normalized(target_name):
                    module_alias_rows.add((alias, _normalized(alias), module_id))
        connection.executemany(
            "INSERT OR IGNORE INTO module_aliases VALUES (?, ?, ?)", sorted(module_alias_rows)
        )
        connection.executemany(
            "INSERT OR IGNORE INTO requirements VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            requirement_rows,
        )
        course_rows: list[tuple[str, str | None, str]] = []
        course_alias_rows: set[tuple[str, str, str]] = set()
        offering_rows: list[tuple[object, ...]] = []
        for course in catalog.get("courses", []):
            cohort = int(course["cohort"])
            program_id = _program_id(course["major"], course["college"], cohort)
            course_module_id = module_map.get((program_id, course["module"]))
            if course_module_id is None:
                course_module_id = _module_id(program_id, course["module"])
                module_map[(program_id, course["module"])] = course_module_id
                connection.execute(
                    "INSERT OR IGNORE INTO modules VALUES (?, ?, ?)",
                    (course_module_id, program_id, course["module"]),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO module_aliases VALUES (?, ?, ?)",
                    (course["module"], _normalized(course["module"]), course_module_id),
                )
            code = str(course.get("code") or "").upper() or None
            name = str(course["name"])
            course_id = _course_id(code, name)
            course_rows.append((course_id, code, name))
            course_alias_rows.add((name, _normalized(name), course_id))
            if code:
                course_alias_rows.add((code, _normalized(code), course_id))
            for alias, target_name in aliases["course_aliases"].items():
                if target_name == name:
                    course_alias_rows.add((alias, _normalized(alias), course_id))
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
                    _structured_review_status(evidence, source_id, section_statuses),
                )
            )
        connection.executemany("INSERT OR IGNORE INTO courses VALUES (?, ?, ?)", course_rows)
        connection.executemany(
            "INSERT OR IGNORE INTO course_aliases VALUES (?, ?, ?)", sorted(course_alias_rows)
        )
        connection.executemany(
            "INSERT OR IGNORE INTO program_courses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            offering_rows,
        )
        chunks_sha256 = _sha(chunk_file) or ""
        source_review_sha256 = (_sha(source_review_file) or "") if source_review_file else ""
        evidence_review_sha256 = (
            (_sha(evidence_review_file) or "") if evidence_review_file else ""
        )
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

    def evidence_readiness(self) -> tuple[bool, tuple[str, ...]]:
        """Check that verified evidence supports the agent's core curriculum work.

        A database can be syntactically complete while every chunk is still
        pending review.  Health readiness therefore requires one current
        program with both a verified course offering and a verified structured
        requirement, each tied to a verified source section from the same
        source, plus a verified current policy chunk that is not merely the
        provenance of one structured course or requirement.  This is
        deliberately stronger than checking file existence.
        """

        def count(statement: str) -> int:
            row = self._one(statement)
            return int(row[0]) if row is not None else 0

        verified_sections = count(
            "SELECT count(*) FROM source_sections WHERE review_status='verified'"
        )
        verified_courses = count(
            """
            SELECT count(*)
            FROM program_courses pc
            JOIN source_sections ss
              ON ss.chunk_id=pc.chunk_id AND ss.source_id=pc.source_id
            WHERE pc.review_status='verified' AND ss.review_status='verified'
            """
        )
        verified_requirements = count(
            """
            SELECT count(*)
            FROM requirements r
            JOIN source_sections ss
              ON ss.chunk_id=r.chunk_id AND ss.source_id=r.source_id
            WHERE r.required_credits IS NOT NULL
              AND r.review_status='verified'
              AND ss.review_status='verified'
            """
        )
        verified_policy_evidence = count(
            """
            SELECT count(*)
            FROM source_sections ss
            JOIN sources s ON s.source_id=ss.source_id
            WHERE ss.review_status='verified'
              AND s.status='现行'
              AND NOT EXISTS (
                SELECT 1 FROM program_courses pc WHERE pc.chunk_id=ss.chunk_id
              )
              AND NOT EXISTS (
                SELECT 1 FROM requirements r WHERE r.chunk_id=ss.chunk_id
              )
            """
        )
        answerable_programs = count(
            """
            SELECT count(*)
            FROM programs p
            JOIN sources ps ON ps.source_id=p.source_id
            WHERE ps.status='现行'
              AND EXISTS (
                SELECT 1
                FROM program_courses pc
                JOIN source_sections ss
                  ON ss.chunk_id=pc.chunk_id AND ss.source_id=pc.source_id
                WHERE pc.program_id=p.program_id
                  AND pc.review_status='verified'
                  AND ss.review_status='verified'
              )
              AND EXISTS (
                SELECT 1
                FROM requirements r
                JOIN source_sections ss
                  ON ss.chunk_id=r.chunk_id AND ss.source_id=r.source_id
                WHERE r.program_id=p.program_id
                  AND r.required_credits IS NOT NULL
                  AND r.review_status='verified'
                  AND ss.review_status='verified'
              )
            """
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
        if not answerable_programs:
            reasons.append("core_business_unanswerable")
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
        """Expose source sections with all metadata required by policy retrieval.

        The provider is database-backed and data-driven: a section is associated
        with program ids only when its source is the authoritative source for
        one or more catalog programs.  School-wide documents intentionally omit
        ``program_ids`` rather than inheriting a guessed program scope.
        """

        rows = self._all(
            """
            SELECT ss.chunk_id, ss.text, ss.source_id, ss.article, ss.physical_page, ss.parser_version, ss.extracted_at, ss.confidence,
                   ss.review_status, s.title, s.page_url, s.file_url, s.college_id,
                   s.cohort, s.authority_level, s.effective_from, s.effective_to,
                   s.status, s.supersedes_source_id, s.source_sha256
            FROM source_sections ss JOIN sources s ON s.source_id=ss.source_id
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
        requested_programs = set(program_ids)
        requested_colleges = set(college_ids)
        requested_colleges.update(self.college_ids_for_programs(program_ids))
        scoped: list[dict[str, object]] = []
        for document in self.retrieval_documents():
            document_cohort = str(document["cohort"])
            if cohort is not None and document_cohort not in {"不限", str(cohort)}:
                continue
            if as_of is None and str(document["status"]) != "现行":
                continue
            effective_from = self._effective_boundary(document.get("effective_from"), end=False)
            effective_to = self._effective_boundary(document.get("effective_to"), end=True)
            if effective_from is not None and effective_from > effective_date:
                continue
            if effective_to is not None and effective_to < effective_date:
                continue
            document_college = str(document.get("college_id") or "")
            if requested_colleges and document_college not in {
                "",
                "全校",
                "校级",
                *requested_colleges,
            }:
                continue
            raw_program_ids = document.get("program_ids", ())
            document_programs = (
                set(raw_program_ids)
                if isinstance(raw_program_ids, tuple)
                and all(isinstance(program_id, str) for program_id in raw_program_ids)
                else set()
            )
            if (
                requested_programs
                and document_programs
                and not document_programs.intersection(requested_programs)
            ):
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
        rows = self._all(
            f"""
            SELECT ss.*, s.title, s.page_url, s.file_url, s.authority_level, s.status, s.cohort, s.college_id,
                   s.published_at, s.effective_from, s.effective_to, s.source_sha256
            FROM source_sections ss JOIN sources s ON s.source_id=ss.source_id
            WHERE ({where}){scope}{status_scope}
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
            result["required_course_difference"] = sorted(
                {
                    course.code or course.name
                    for values in groups.values()
                    for course in values
                    if "必修" in (course.nature or "")
                }
            )
        if "practice_requirements" in requested:
            result["practice_requirements"] = sorted(
                {
                    course.name
                    for values in groups.values()
                    for course in values
                    if "实践" in (course.nature or "") or "实践" in course.module_name
                }
            )
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
