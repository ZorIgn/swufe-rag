"""Fail fast on canonical data integrity violations."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

CHECKS = {
    "duplicate_source": "SELECT count(*) FROM (SELECT title, cohort, file_url, count(*) n FROM sources GROUP BY title, cohort, file_url HAVING n > 1)",
    "orphan_program_course": "SELECT count(*) FROM program_courses pc LEFT JOIN programs p ON p.program_id=pc.program_id WHERE p.program_id IS NULL",
    "orphan_provenance": "SELECT count(*) FROM program_courses pc LEFT JOIN sources s ON s.source_id=pc.source_id WHERE s.source_id IS NULL",
    "invalid_course_code": "SELECT count(*) FROM courses WHERE canonical_code IS NOT NULL AND canonical_code <> '' AND canonical_code NOT GLOB '[A-Za-z]*[0-9]*'",
    "invalid_credits": "SELECT count(*) FROM program_courses WHERE credits IS NOT NULL AND credits <= 0",
    "invalid_semesters": "SELECT count(*) FROM program_courses WHERE semester <> '' AND NOT (semester GLOB '[1-8]' OR semester GLOB '[1-8]-[1-8]' OR semester GLOB '[Ss][1-3]' OR semester GLOB '[Ss][1-3]-[1-8]' OR semester GLOB '[Ss][1-3]-[Ss][1-3]')",
    "invalid_program_relation": "SELECT count(*) FROM requirements r LEFT JOIN programs p ON p.program_id=r.program_id WHERE p.program_id IS NULL",
    "duplicate_canonical_course": "SELECT count(*) FROM (SELECT program_id, module_id, course_id, semester, count(*) n FROM program_courses GROUP BY program_id, module_id, course_id, semester HAVING n > 1)",
    "requirement_without_evidence": "SELECT count(*) FROM requirements WHERE required_credits IS NOT NULL AND (chunk_id IS NULL OR chunk_id='')",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=Path("data/academic.sqlite3"))
    parser.add_argument("--allow-unverified-requirements", action="store_true")
    args = parser.parse_args()
    connection = sqlite3.connect(args.database)
    try:
        results = {name: int(connection.execute(sql).fetchone()[0]) for name, sql in CHECKS.items()}
    finally:
        connection.close()
    if args.allow_unverified_requirements:
        results["requirement_without_evidence"] = 0
    print(json.dumps(results, ensure_ascii=False, indent=2))
    failures = {name: value for name, value in results.items() if value}
    if failures:
        raise SystemExit(f"dataset verification failed: {failures}")


if __name__ == "__main__":
    main()
