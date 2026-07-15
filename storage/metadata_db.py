"""SQLite-backed source scope, chunk identity and official-link binding."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Iterable, Sequence
from urllib.parse import urlparse

from contracts import ContractError, KnowledgeChunk
from ingest.sources import SOURCE_FIELDS, validate_source_row
from retrieval.index import file_sha256, load_chunks
from storage.migrations import apply_migrations


@dataclass(frozen=True)
class OfficialLink:
    source_id: str
    title: str
    page_url: str
    file_url: str


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: str
    source_id: str
    text: str
    doc_title: str
    article: str
    level: str
    college: str
    cohort: str
    year: int
    status: str
    topic: str
    page_url: str
    file_url: str
    is_table: bool
    embedding_row: int


def _official_swufe_url(value: str) -> bool:
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    return parsed.scheme in {"http", "https"} and (
        host == "swufe.edu.cn" or host.endswith(".swufe.edu.cn")
    )


def infer_topic(title: str, text: str = "") -> str:
    searchable = title + "\n" + text[:1000]
    rules = (
        (("推荐免试", "推免", "保研"), "promotion"),
        (("培养方案", "课程设置"), "curriculum"),
        (("选课操作", "选课指南"), "course_selection"),
        (("转专业", "专业分流"), "transfer"),
        (("学籍", "重修", "休学", "转学"), "academic_status"),
        (("考试", "缓考", "考核"), "assessment"),
        (("学分认定", "课程免修", "辅修"), "credit"),
        (("毕业论文", "毕业设计"), "thesis"),
    )
    for terms, topic in rules:
        if any(term in searchable for term in terms):
            return topic
    return "school_policy"


def _source_key(values: Sequence[object]) -> str:
    return "\x1f".join(str(value) for value in values)


def _source_id(source_key: str) -> str:
    return "src_" + sha256(source_key.encode("utf-8")).hexdigest()[:24]


class MetadataDB:
    """Owns the SQL allow-list used before ranking and after generation.

    User or model text is never interpolated into SQL.  The only dynamic SQL
    fragments below are fixed, developer-owned predicates; all values use
    bound parameters.
    """

    def __init__(self, database: str | Path = ":memory:") -> None:
        self.database = str(database)
        self.connection = sqlite3.connect(
            self.database, check_same_thread=False
        )
        self.connection.row_factory = sqlite3.Row
        self._lock = RLock()
        with self._lock:
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.execute("PRAGMA busy_timeout = 5000")
            apply_migrations(self.connection)

    @classmethod
    def from_chunks(
        cls,
        chunks: Sequence[KnowledgeChunk],
        *,
        database: str | Path = ":memory:",
        trusted_by_default: bool = False,
    ) -> "MetadataDB":
        instance = cls(database)
        instance.rebuild_from_chunks(
            chunks, trusted_by_default=trusted_by_default
        )
        return instance

    @classmethod
    def from_files(
        cls,
        *,
        sources_path: str | Path = "data/sources.csv",
        chunks_path: str | Path = "data/chunks.jsonl",
        database: str | Path = "data/metadata.sqlite3",
    ) -> "MetadataDB":
        sources_file = Path(sources_path)
        chunks_file = Path(chunks_path)
        if not sources_file.is_file():
            raise ContractError(f"source registry does not exist: {sources_file}")
        chunks = load_chunks(chunks_file)
        fingerprint = {
            "sources_sha256": file_sha256(sources_file),
            "chunks_sha256": file_sha256(chunks_file),
        }
        instance = cls(database)
        if instance._fingerprint_matches(fingerprint, len(chunks)):
            return instance
        instance.rebuild_from_registry(sources_file, chunks, fingerprint)
        return instance

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def _fingerprint_matches(self, fingerprint: dict[str, str], count: int) -> bool:
        with self._lock:
            values = {
                row["key"]: row["value"]
                for row in self.connection.execute(
                    "SELECT key, value FROM schema_meta"
                )
            }
            chunk_count = self.connection.execute(
                "SELECT count(*) FROM chunks"
            ).fetchone()[0]
        return (
            chunk_count == count
            and values.get("sources_sha256") == fingerprint["sources_sha256"]
            and values.get("chunks_sha256") == fingerprint["chunks_sha256"]
        )

    @staticmethod
    def _load_registry(path: Path) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != SOURCE_FIELDS:
                raise ContractError(
                    "source registry header must exactly match the canonical schema"
                )
            for line_number, raw in enumerate(reader, start=2):
                record = validate_source_row(raw, line_number=line_number)
                records.append(
                    {
                        "source_key": record.file,
                        "doc_title": record.doc_title,
                        "level": record.level,
                        "college": record.college,
                        "cohort": record.cohort,
                        "year": record.year,
                        "status": record.status,
                        "page_url": record.page_url,
                        "file_url": record.file_url,
                    }
                )
        return records

    @staticmethod
    def _signature(item: dict[str, object] | KnowledgeChunk) -> tuple[object, ...]:
        return tuple(
            item[key]
            for key in (
                "doc_title",
                "level",
                "college",
                "cohort",
                "year",
                "status",
                "page_url",
                "file_url",
            )
        )

    def rebuild_from_registry(
        self,
        sources_path: Path,
        chunks: Sequence[KnowledgeChunk],
        fingerprint: dict[str, str],
    ) -> None:
        registry = self._load_registry(sources_path)
        by_signature = {self._signature(item): item for item in registry}
        if len(by_signature) != len(registry):
            raise ContractError("source registry contains duplicate source metadata")
        unmatched = [
            chunk["chunk_id"]
            for chunk in chunks
            if self._signature(chunk) not in by_signature
        ]
        if unmatched:
            preview = ", ".join(unmatched[:5])
            raise ContractError(
                f"chunks are not bound to a registered trusted source: {preview}"
            )

        source_rows = []
        source_ids: dict[tuple[object, ...], str] = {}
        source_text: dict[tuple[object, ...], str] = {}
        for chunk in chunks:
            signature = self._signature(chunk)
            source_text.setdefault(signature, chunk["text"])
        for item in registry:
            signature = self._signature(item)
            key = str(item["source_key"])
            identifier = _source_id(key)
            source_ids[signature] = identifier
            source_rows.append(
                (
                    identifier,
                    key,
                    item["doc_title"],
                    item["level"],
                    item["college"],
                    item["cohort"],
                    item["year"],
                    item["status"],
                    infer_topic(str(item["doc_title"]), source_text.get(signature, "")),
                    item["page_url"],
                    item["file_url"],
                    1,
                    1,
                )
            )
        chunk_rows = [
            (
                chunk["chunk_id"],
                source_ids[self._signature(chunk)],
                chunk["article"],
                chunk["text"],
                int(chunk["is_table"]),
                index,
            )
            for index, chunk in enumerate(chunks)
        ]
        self._replace_all(source_rows, chunk_rows, fingerprint)

    def rebuild_from_chunks(
        self,
        chunks: Sequence[KnowledgeChunk],
        *,
        trusted_by_default: bool,
    ) -> None:
        grouped: dict[tuple[object, ...], list[KnowledgeChunk]] = {}
        for chunk in chunks:
            grouped.setdefault(self._signature(chunk), []).append(chunk)
        source_rows = []
        source_ids: dict[tuple[object, ...], str] = {}
        for signature, items in grouped.items():
            chunk = items[0]
            key = _source_key(signature)
            identifier = _source_id(key)
            source_ids[signature] = identifier
            trusted = trusted_by_default or (
                _official_swufe_url(chunk["page_url"])
                and _official_swufe_url(chunk["file_url"])
            )
            source_rows.append(
                (
                    identifier,
                    key,
                    chunk["doc_title"],
                    chunk["level"],
                    chunk["college"],
                    chunk["cohort"],
                    chunk["year"],
                    chunk["status"],
                    infer_topic(chunk["doc_title"], chunk["text"]),
                    chunk["page_url"],
                    chunk["file_url"],
                    int(trusted),
                    1,
                )
            )
        chunk_rows = [
            (
                chunk["chunk_id"],
                source_ids[self._signature(chunk)],
                chunk["article"],
                chunk["text"],
                int(chunk["is_table"]),
                index,
            )
            for index, chunk in enumerate(chunks)
        ]
        self._replace_all(source_rows, chunk_rows, {})

    def _replace_all(
        self,
        source_rows: Iterable[tuple[object, ...]],
        chunk_rows: Iterable[tuple[object, ...]],
        fingerprint: dict[str, str],
    ) -> None:
        with self._lock, self.connection:
            self.connection.execute("DELETE FROM chunks")
            self.connection.execute("DELETE FROM sources")
            self.connection.execute(
                "DELETE FROM schema_meta WHERE key IN ('sources_sha256', 'chunks_sha256')"
            )
            self.connection.executemany(
                """
                INSERT INTO sources(
                    source_id, source_key, doc_title, level, college, cohort,
                    year, status, topic, page_url, file_url, trusted, enabled
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                list(source_rows),
            )
            self.connection.executemany(
                """
                INSERT INTO chunks(
                    chunk_id, source_id, article, text, is_table, embedding_row
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                list(chunk_rows),
            )
            for key, value in fingerprint.items():
                self.connection.execute(
                    "INSERT OR REPLACE INTO schema_meta(key, value) VALUES(?, ?)",
                    (key, value),
                )

    @staticmethod
    def _scope_values(
        college: str | None,
        cohort: str | None,
        policy_year: int | None,
        topic: str | None,
    ) -> tuple[str | None, str | None, int | None, str | None]:
        clean_college = college.strip() if isinstance(college, str) else None
        clean_cohort = cohort.strip() if isinstance(cohort, str) else None
        clean_topic = topic.strip() if isinstance(topic, str) else None
        if college is not None and not clean_college:
            raise ValueError("college must be None or a non-empty string")
        if cohort is not None and not clean_cohort:
            raise ValueError("cohort must be None or a non-empty string")
        if policy_year is not None and not 1900 <= policy_year <= 2100:
            raise ValueError("policy_year must be between 1900 and 2100")
        if topic is not None and not clean_topic:
            raise ValueError("topic must be None or a non-empty string")
        return clean_college, clean_cohort, policy_year, clean_topic

    def candidate_rows(
        self,
        *,
        college: str | None = None,
        cohort: str | None = None,
        policy_year: int | None = None,
        topic: str | None = None,
    ) -> list[int]:
        college, cohort, policy_year, topic = self._scope_values(
            college, cohort, policy_year, topic
        )
        sql = """
            SELECT c.embedding_row
            FROM chunks AS c
            JOIN sources AS s ON s.source_id = c.source_id
            WHERE s.enabled = 1
              AND s.trusted = 1
              AND ((? IS NULL AND s.status = '现行')
                   OR (? IS NOT NULL AND s.year = ?))
              AND (? IS NULL OR s.level = '校级' OR s.college = ?)
              AND (? IS NULL OR s.cohort = '不限' OR s.cohort = ?)
              AND (? IS NULL OR s.topic = ?)
            ORDER BY c.embedding_row
        """
        params = (
            policy_year,
            policy_year,
            policy_year,
            college,
            college,
            cohort,
            cohort,
            topic,
            topic,
        )
        with self._lock:
            return [
                int(row[0]) for row in self.connection.execute(sql, params).fetchall()
            ]

    def chunk(
        self, chunk_id: str, *, require_trusted: bool = True
    ) -> StoredChunk | None:
        trusted_clause = "AND s.trusted = 1 AND s.enabled = 1" if require_trusted else ""
        with self._lock:
            row = self.connection.execute(
                f"""
                SELECT c.chunk_id, c.source_id, c.text, s.doc_title, c.article,
                       s.level, s.college, s.cohort, s.year, s.status, s.topic,
                       s.page_url, s.file_url, c.is_table, c.embedding_row
                FROM chunks AS c
                JOIN sources AS s ON s.source_id = c.source_id
                WHERE c.chunk_id = ? {trusted_clause}
                """,
                (chunk_id,),
            ).fetchone()
        if row is None:
            return None
        return StoredChunk(
            chunk_id=row["chunk_id"],
            source_id=row["source_id"],
            text=row["text"],
            doc_title=row["doc_title"],
            article=row["article"],
            level=row["level"],
            college=row["college"],
            cohort=row["cohort"],
            year=int(row["year"]),
            status=row["status"],
            topic=row["topic"],
            page_url=row["page_url"],
            file_url=row["file_url"],
            is_table=bool(row["is_table"]),
            embedding_row=int(row["embedding_row"]),
        )

    def official_links(
        self,
        *,
        college: str | None = None,
        cohort: str | None = None,
        topic: str | None = None,
        policy_year: int | None = None,
        limit: int = 3,
    ) -> list[OfficialLink]:
        if not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20")
        college, cohort, policy_year, topic = self._scope_values(
            college, cohort, policy_year, topic
        )
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT source_id, doc_title, page_url, file_url
                FROM sources
                WHERE enabled = 1 AND trusted = 1
                  AND ((? IS NULL AND status = '现行')
                       OR (? IS NOT NULL AND year = ?))
                  AND (? IS NULL OR level = '校级' OR college = ?)
                  AND (? IS NULL OR cohort = '不限' OR cohort = ?)
                  AND (? IS NULL OR topic = ?)
                ORDER BY CASE WHEN level = '院级' THEN 0 ELSE 1 END,
                         year DESC, doc_title
                LIMIT ?
                """,
                (
                    policy_year,
                    policy_year,
                    policy_year,
                    college,
                    college,
                    cohort,
                    cohort,
                    topic,
                    topic,
                    limit,
                ),
            ).fetchall()
        return [
            OfficialLink(
                source_id=row["source_id"],
                title=row["doc_title"],
                page_url=row["page_url"],
                file_url=row["file_url"],
            )
            for row in rows
        ]

    def known_colleges(self) -> tuple[str, ...]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT DISTINCT college FROM sources
                WHERE trusted = 1 AND enabled = 1 AND level = '院级'
                ORDER BY college
                """
            ).fetchall()
        return tuple(row[0] for row in rows)

    def known_cohorts(self) -> tuple[str, ...]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT DISTINCT cohort FROM sources
                WHERE trusted = 1 AND enabled = 1 AND cohort <> '不限'
                ORDER BY cohort
                """
            ).fetchall()
        return tuple(row[0] for row in rows)

    def set_source_state(
        self,
        source_id: str,
        *,
        trusted: bool | None = None,
        enabled: bool | None = None,
    ) -> None:
        assignments: list[str] = []
        params: list[object] = []
        if trusted is not None:
            assignments.append("trusted = ?")
            params.append(int(trusted))
        if enabled is not None:
            assignments.append("enabled = ?")
            params.append(int(enabled))
        if not assignments:
            return
        params.append(source_id)
        with self._lock, self.connection:
            cursor = self.connection.execute(
                f"UPDATE sources SET {', '.join(assignments)} WHERE source_id = ?",
                params,
            )
            if cursor.rowcount != 1:
                raise KeyError(source_id)

    def integrity_report(self) -> dict[str, int]:
        checks = {
            "sources": "SELECT count(*) FROM sources",
            "chunks": "SELECT count(*) FROM chunks",
            "orphan_chunks": """
                SELECT count(*) FROM chunks c
                LEFT JOIN sources s ON s.source_id = c.source_id
                WHERE s.source_id IS NULL
            """,
            "eligible_untrusted": """
                SELECT count(*) FROM chunks c
                JOIN sources s ON s.source_id = c.source_id
                WHERE s.trusted = 0 AND s.enabled = 1 AND s.status = '现行'
                  AND c.embedding_row IN (
                      SELECT c2.embedding_row FROM chunks c2
                      JOIN sources s2 ON s2.source_id = c2.source_id
                      WHERE s2.trusted = 1 AND s2.enabled = 1 AND s2.status = '现行'
                  )
            """,
        }
        with self._lock:
            return {
                name: int(self.connection.execute(sql).fetchone()[0])
                for name, sql in checks.items()
            }


__all__ = [
    "MetadataDB",
    "OfficialLink",
    "StoredChunk",
    "infer_topic",
]
