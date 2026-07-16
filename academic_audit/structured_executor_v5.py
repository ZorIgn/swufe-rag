"""Structured executor with correct metadata source join."""

from __future__ import annotations

from typing import Any

import academic_audit.structured_executor_v4 as base
from academic_audit.structured_qa import _clean_course_name
from storage.metadata_db import MetadataDB


def _compact(value: object) -> str:
    import re

    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", str(value or "")).lower()


def _repair_evidence(
    records: list[dict[str, Any]], metadata_db: MetadataDB
) -> list[dict[str, Any]]:
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
                SELECT c.chunk_id, c.text
                FROM chunks AS c
                JOIN sources AS s ON s.source_id = c.source_id
                WHERE s.enabled = 1 AND s.doc_title = ? AND c.article LIKE ?
                ORDER BY c.is_table DESC, c.embedding_row
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


# v4 resolves this global at call time.  Install the schema-correct binder once
# during module import, before the runtime starts accepting concurrent requests.
base._repair_evidence = _repair_evidence
execute = base.execute


__all__ = ["execute"]
