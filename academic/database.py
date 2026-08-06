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

from evidence.provenance import PARSER_VERSION, stable_id
from query.schemas import ResolvedEntity

ROOT = Path(__file__).parents[1]
DEFAULT_DATABASE = ROOT / "data" / "academic.sqlite3"
DEFAULT_CATALOG = ROOT / "data" / "curriculum_catalog.json"
DEFAULT_SOURCES = ROOT / "data" / "sources.csv"
DEFAULT_CHUNKS = ROOT / "data" / "chunks.jsonl"
DEFAULT_ALIAS_CONFIG = ROOT / "config" / "entity_aliases.json"


SCHEMA_VERSION = "1"
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


def _source_index(
    rows: Iterable[dict[str, str]],
) -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, dict[str, str]]]:
    exact: dict[tuple[str, str], dict[str, str]] = {}
    title_only: dict[str, dict[str, str]] = {}
    for row in rows:
        exact[(row["doc_title"], row["cohort"])] = row
        title_only.setdefault(row["doc_title"], row)
    return exact, title_only


def _materialize_sources(
    connection: sqlite3.Connection, rows: list[dict[str, str]], root: Path
) -> dict[tuple[str, str], str]:
    ids: dict[tuple[str, str], str] = {}
    values: list[tuple[object, ...]] = []
    for row in rows:
        title, cohort, url = row["doc_title"], row["cohort"], row["file_url"]
        identifier = _source_id(title, cohort, url)
        ids[(title, cohort)] = identifier
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
    cohort: int,
    source_ids: dict[tuple[str, str], str],
    title_rows: dict[str, dict[str, str]],
) -> str:
    found = source_ids.get((title, str(cohort)))
    if found:
        return found
    row = title_rows.get(title)
    if row:
        return source_ids[(row["doc_title"], row["cohort"])]
    raise ValueError(f"catalog source is not registered: {title!r}, cohort={cohort}")


def build_database(
    output: str | Path = DEFAULT_DATABASE,
    *,
    catalog_path: str | Path = DEFAULT_CATALOG,
    sources_path: str | Path = DEFAULT_SOURCES,
    chunks_path: str | Path = DEFAULT_CHUNKS,
    aliases_path: str | Path = DEFAULT_ALIAS_CONFIG,
) -> dict[str, object]:
    """Build a new immutable SQLite projection; generated output is not Git data."""
    target, catalog_file, source_file, chunk_file = map(
        Path, (output, catalog_path, sources_path, chunks_path)
    )
    catalog = json.loads(catalog_file.read_text(encoding="utf-8"))
    source_rows = _source_rows(source_file)
    source_by_title_cohort, source_by_title = _source_index(source_rows)
    aliases = _read_aliases(Path(aliases_path))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(SCHEMA)
        source_ids = _materialize_sources(connection, source_rows, ROOT)
        program_rows: list[tuple[object, ...]] = []
        alias_rows: set[tuple[str, str, str]] = set()
        module_rows: list[tuple[str, str, str]] = []
        module_map: dict[tuple[str, str], str] = {}
        requirement_rows: list[tuple[object, ...]] = []
        for plan in catalog.get("plans", []):
            cohort = int(plan["cohort"])
            source_id = _source_for(plan["source_title"], cohort, source_ids, source_by_title)
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
                evidence = module.get("evidence") or {}
                page = _page(evidence.get("article"))
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
                        evidence.get("chunk_id"),
                        PARSER_VERSION,
                        0.9 if evidence else 0.6,
                        "verified" if evidence else "review_required",
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
            module_id = module_map.get((program_id, course["module"]))
            if module_id is None:
                module_id = _module_id(program_id, course["module"])
                module_map[(program_id, course["module"])] = module_id
                connection.execute(
                    "INSERT OR IGNORE INTO modules VALUES (?, ?, ?)",
                    (module_id, program_id, course["module"]),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO module_aliases VALUES (?, ?, ?)",
                    (course["module"], _normalized(course["module"]), module_id),
                )
            code = str(course.get("code") or "").upper() or None
            name = str(course["name"])
            course_id = _course_id(code, name)
            course_rows.append((course_id, code, name))
            course_alias_rows.add((name, _normalized(name), course_id))
            for alias, target_name in aliases["course_aliases"].items():
                if target_name == name:
                    course_alias_rows.add((alias, _normalized(alias), course_id))
            evidence = course.get("evidence") or {}
            source_id = _source_for(course["source_title"], cohort, source_ids, source_by_title)
            record_id = stable_id(
                "offering",
                program_id,
                module_id,
                course_id,
                course.get("semester"),
                course.get("page"),
                course.get("source_row"),
            )
            offering_rows.append(
                (
                    record_id,
                    program_id,
                    module_id,
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
                    evidence.get("chunk_id"),
                    PARSER_VERSION,
                    0.95 if evidence else 0.7,
                    "verified" if evidence else "review_required",
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
        if chunk_file.is_file():
            sections: list[tuple[object, ...]] = []
            with chunk_file.open(encoding="utf-8") as stream:
                for line in stream:
                    value = json.loads(line)
                    title, cohort = str(value["doc_title"]), str(value.get("cohort") or "不限")
                    source_id = source_ids.get((title, cohort))
                    if source_id is None:
                        source_id = _source_for(
                            title,
                            int(cohort) if cohort.isdigit() else 0,
                            source_ids,
                            source_by_title,
                        )
                    sections.append(
                        (
                            value["chunk_id"],
                            source_id,
                            value.get("article"),
                            value["text"],
                            _page(value.get("article")),
                            int(bool(value.get("is_table"))),
                            PARSER_VERSION,
                            datetime.now(timezone.utc).isoformat(),
                            0.8,
                            "unverified",
                        )
                    )
            connection.executemany(
                "INSERT OR IGNORE INTO source_sections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                sections,
            )
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "parser_version": PARSER_VERSION,
            "dataset_version": str(catalog.get("catalog_version", "unknown")),
            "catalog_sha256": _sha(catalog_file) or "",
            "sources_sha256": _sha(source_file) or "",
            "chunks_sha256": _sha(chunk_file) or "",
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
            **metadata,
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
            return cursor.execute(statement, tuple(values)).fetchone()

    def metadata(self) -> dict[str, str]:
        return {row["key"]: row["value"] for row in self._all("SELECT key, value FROM metadata")}

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

    def _resolve(
        self, kind: str, mention: str, cohort: int | None, program_id: str | None = None
    ) -> ResolvedEntity | None:
        target = _normalized(mention)
        if not target:
            return None
        if kind == "program":
            statement = """
                SELECT p.program_id AS id, p.canonical_name AS name, a.normalized_alias AS alias
                FROM program_aliases a JOIN programs p ON p.program_id=a.program_id
                WHERE (? IS NULL OR p.cohort=?) ORDER BY length(a.normalized_alias) DESC
            """
            rows = self._all(statement, (cohort, cohort))
        elif kind == "course":
            statement = """
                SELECT c.course_id AS id, c.canonical_name AS name, a.normalized_alias AS alias
                FROM course_aliases a JOIN courses c ON c.course_id=a.course_id
                WHERE ? IS NULL OR EXISTS (SELECT 1 FROM program_courses pc JOIN programs p ON p.program_id=pc.program_id WHERE pc.course_id=c.course_id AND p.cohort=? AND (? IS NULL OR p.program_id=?))
                ORDER BY length(a.normalized_alias) DESC
            """
            rows = self._all(statement, (cohort, cohort, program_id, program_id))
        else:
            statement = """
                SELECT m.module_id AS id, m.canonical_name AS name, a.normalized_alias AS alias
                FROM module_aliases a JOIN modules m ON m.module_id=a.module_id
                WHERE ? IS NULL OR m.program_id=? ORDER BY length(a.normalized_alias) DESC
            """
            rows = self._all(statement, (program_id, program_id))
        for row in rows:
            alias = str(row["alias"])
            if alias == target or alias in target or target in alias:
                return ResolvedEntity(
                    entity_type=kind,
                    canonical_id=row["id"],
                    canonical_name=row["name"],
                    confidence=1.0 if alias == target else 0.85,
                )
        return None

    def resolve_program(self, mention: str, cohort: int | None = None) -> ResolvedEntity | None:
        direct = self._one(
            "SELECT program_id, canonical_name FROM programs WHERE program_id=?", (mention,)
        )
        if direct:
            return ResolvedEntity(
                entity_type="program",
                canonical_id=direct["program_id"],
                canonical_name=direct["canonical_name"],
                confidence=1.0,
            )
        return self._resolve("program", mention, cohort)

    def resolve_course(
        self, mention: str, cohort: int | None = None, program_id: str | None = None
    ) -> ResolvedEntity | None:
        return self._resolve("course", mention, cohort, program_id)

    def resolve_module(self, mention: str, program_id: str | None = None) -> ResolvedEntity | None:
        return self._resolve("module", mention, None, program_id)

    def programs_in_text(self, text: str, cohort: int | None = None) -> tuple[ResolvedEntity, ...]:
        rows = self._all(
            "SELECT DISTINCT alias FROM program_aliases"
            + (
                " a JOIN programs p ON p.program_id=a.program_id WHERE p.cohort=?" if cohort else ""
            ),
            (cohort,) if cohort else (),
        )
        values: list[ResolvedEntity] = []
        for row in rows:
            resolved = self.resolve_program(str(row["alias"]), cohort)
            if (
                resolved
                and _normalized(str(row["alias"])) in _normalized(text)
                and resolved.canonical_id not in {item.canonical_id for item in values}
            ):
                values.append(resolved)
        return tuple(values)

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
            SELECT ss.*, s.title, s.page_url, s.file_url, s.source_sha256, s.effective_from, s.effective_to
            FROM source_sections ss JOIN sources s ON s.source_id=ss.source_id WHERE ss.chunk_id=?
        """,
            (chunk_id,),
        )
        return dict(row) if row else None

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
                int(row["authority_level"]),
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

    def compare_programs(self, *, cohort: int, program_ids: tuple[str, ...]) -> dict[str, object]:
        groups: dict[str, list[CourseRecord]] = {
            program_id: list(self.list_courses(cohort=cohort, program_id=program_id))
            for program_id in program_ids
        }
        program_names = {
            row["program_id"]: row["canonical_name"]
            for row in self._all(
                "SELECT program_id, canonical_name FROM programs WHERE program_id IN ("
                + ",".join("?" for _ in program_ids)
                + ")",
                program_ids,
            )
        }
        code_sets = {
            program_id: {course.code or course.name for course in values}
            for program_id, values in groups.items()
        }
        intersection = set.intersection(*code_sets.values()) if code_sets else set()
        only = {
            program_id: sorted(values - intersection) for program_id, values in code_sets.items()
        }
        minimums = {
            program_id: sum(
                float(row["required_credits"] or 0)
                for row in self.requirements(cohort=cohort, program_id=program_id)
            )
            for program_id in program_ids
        }
        return {
            "programs": program_names,
            "intersection": sorted(intersection),
            "only_in_each": only,
            "numeric_difference": minimums,
        }


__all__ = ["AcademicRepository", "CourseRecord", "DEFAULT_DATABASE", "build_database"]
