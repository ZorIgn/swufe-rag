"""SQLite schema used as the hard trust boundary for school facts."""

from __future__ import annotations

import sqlite3


SCHEMA_VERSION = 1

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    source_key TEXT NOT NULL UNIQUE,
    doc_title TEXT NOT NULL CHECK(length(trim(doc_title)) > 0),
    level TEXT NOT NULL CHECK(level IN ('校级', '院级')),
    college TEXT NOT NULL CHECK(length(trim(college)) > 0),
    cohort TEXT NOT NULL CHECK(
        cohort = '不限'
        OR (length(cohort) = 4 AND cohort NOT GLOB '*[^0-9]*')
    ),
    year INTEGER NOT NULL CHECK(year BETWEEN 1900 AND 2100),
    status TEXT NOT NULL CHECK(status IN ('现行', '历史')),
    topic TEXT NOT NULL CHECK(length(trim(topic)) > 0),
    page_url TEXT NOT NULL CHECK(
        lower(page_url) GLOB 'http://*/*'
        OR lower(page_url) GLOB 'https://*/*'
    ),
    file_url TEXT NOT NULL CHECK(
        lower(file_url) GLOB 'http://*/*'
        OR lower(file_url) GLOB 'https://*/*'
    ),
    trusted INTEGER NOT NULL DEFAULT 0 CHECK(trusted IN (0, 1)),
    enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0, 1)),
    CHECK((level = '校级' AND college = '全校')
          OR (level = '院级' AND college <> '全校')),
    UNIQUE(doc_title, level, college, cohort, year, page_url, file_url)
) STRICT;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    article TEXT NOT NULL CHECK(length(trim(article)) > 0),
    text TEXT NOT NULL CHECK(length(trim(text)) > 0),
    is_table INTEGER NOT NULL DEFAULT 0 CHECK(is_table IN (0, 1)),
    embedding_row INTEGER NOT NULL UNIQUE CHECK(embedding_row >= 0),
    FOREIGN KEY(source_id) REFERENCES sources(source_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
) STRICT;

CREATE INDEX IF NOT EXISTS idx_sources_scope
ON sources(enabled, trusted, status, level, college, cohort, year, topic);

CREATE INDEX IF NOT EXISTS idx_chunks_source
ON chunks(source_id);
"""


def apply_migrations(connection: sqlite3.Connection) -> None:
    """Create or verify the current schema in one transaction."""

    connection.executescript(SCHEMA_SQL)
    row = connection.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()
    if row is not None and int(row[0]) != SCHEMA_VERSION:
        raise RuntimeError(
            f"unsupported metadata schema version: {row[0]} (expected {SCHEMA_VERSION})"
        )
    connection.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    connection.commit()


__all__ = ["SCHEMA_VERSION", "SCHEMA_SQL", "apply_migrations"]
