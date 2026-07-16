"""Integrity and field-completeness checks for the full-school SQLite store."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3


def main() -> None:
    path = Path("data/academic_v2.sqlite3")
    connection = sqlite3.connect(path)

    def scalar(sql: str):
        return connection.execute(sql).fetchone()[0]

    report = {
        "integrity": scalar("PRAGMA integrity_check"),
        "foreign_key_errors": connection.execute("PRAGMA foreign_key_check").fetchall(),
        "null_course_evidence": scalar(
            "SELECT count(*) FROM course_offerings WHERE evidence_chunk_id IS NULL"
        ),
        "duplicate_course_rows": scalar(
            "SELECT count(*) FROM course_offerings WHERE is_primary=0"
        ),
        "null_total_hours": scalar(
            "SELECT count(*) FROM course_offerings WHERE total_hours IS NULL"
        ),
        "unmarked_semester": scalar(
            """
            SELECT count(*) FROM course_offerings
            WHERE semester IS NULL OR semester='' OR semester='未标注'
            """
        ),
        "cohorts": [
            {"cohort": row[0], "plans": row[1], "course_rows": row[2]}
            for row in connection.execute(
                """
                SELECT cohort, count(DISTINCT major), count(*)
                FROM course_offerings WHERE is_primary=1
                GROUP BY cohort ORDER BY cohort
                """
            )
        ],
    }
    output = Path("analysis-output/full-system-v2/database-quality.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
