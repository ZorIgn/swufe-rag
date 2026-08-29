"""Fail fast on canonical data-integrity and evidence-trust violations.

The verifier deliberately opens SQLite in read-only URI mode.  Verification is
not allowed to create an empty database merely because an operator typed a
wrong path, and it validates reconciliation metadata in addition to rows that
survived materialisation.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


class DatasetVerificationError(RuntimeError):
    """A deterministic, operator-actionable dataset verification failure."""


CHECKS = {
    "duplicate_source": "SELECT count(*) FROM (SELECT title, cohort, file_url, count(*) n FROM sources GROUP BY title, cohort, file_url HAVING n > 1)",
    "orphan_program_course": "SELECT count(*) FROM program_courses pc LEFT JOIN programs p ON p.program_id=pc.program_id WHERE p.program_id IS NULL",
    "orphan_provenance": "SELECT count(*) FROM program_courses pc LEFT JOIN sources s ON s.source_id=pc.source_id WHERE s.source_id IS NULL",
    "invalid_course_code": "SELECT count(*) FROM courses WHERE canonical_code IS NOT NULL AND canonical_code <> '' AND canonical_code NOT GLOB '[A-Za-z]*[0-9]*'",
    "invalid_credits": "SELECT count(*) FROM program_courses WHERE credits IS NOT NULL AND credits <= 0",
    "invalid_semesters": "SELECT count(*) FROM program_courses WHERE semester <> '' AND NOT (semester GLOB '[1-8]' OR semester GLOB '[1-8]-[1-8]' OR semester GLOB '[Ss][1-3]' OR semester GLOB '[Ss][1-3]-[1-8]' OR semester GLOB '[Ss][1-3]-[Ss][1-3]')",
    "invalid_program_relation": "SELECT count(*) FROM requirements r LEFT JOIN programs p ON p.program_id=r.program_id WHERE p.program_id IS NULL",
    "duplicate_canonical_course": "SELECT count(*) FROM (SELECT program_id, module_id, course_id, semester, count(*) n FROM program_courses GROUP BY program_id, module_id, course_id, semester HAVING n > 1)",
    "invalid_review_status": "SELECT count(*) FROM source_sections WHERE review_status NOT IN ('verified','review_required','unverified')",
    "invalid_structured_review_status": "SELECT count(*) FROM (SELECT review_status FROM program_courses UNION ALL SELECT review_status FROM requirements) WHERE review_status NOT IN ('verified','review_required','unverified')",
    "verified_program_course_without_evidence": "SELECT count(*) FROM program_courses pc LEFT JOIN source_sections ss ON ss.chunk_id=pc.chunk_id AND ss.source_id=pc.source_id WHERE pc.review_status='verified' AND (pc.chunk_id IS NULL OR ss.chunk_id IS NULL OR ss.review_status <> 'verified')",
    "program_course_evidence_status_mismatch": "SELECT count(*) FROM program_courses pc LEFT JOIN source_sections ss ON ss.chunk_id=pc.chunk_id WHERE pc.chunk_id IS NOT NULL AND (ss.chunk_id IS NULL OR ss.source_id <> pc.source_id OR ss.review_status <> pc.review_status)",
    "verified_requirement_without_evidence": "SELECT count(*) FROM requirements r LEFT JOIN source_sections ss ON ss.chunk_id=r.chunk_id AND ss.source_id=r.source_id WHERE r.required_credits IS NOT NULL AND r.review_status='verified' AND (r.chunk_id IS NULL OR ss.chunk_id IS NULL OR ss.review_status <> 'verified')",
    "review_required_requirement": "SELECT count(*) FROM requirements r JOIN source_sections ss ON ss.chunk_id=r.chunk_id AND ss.source_id=r.source_id WHERE r.required_credits IS NOT NULL AND r.review_status='review_required' AND ss.review_status='review_required'",
    "review_required_requirement_without_evidence": "SELECT count(*) FROM requirements r LEFT JOIN source_sections ss ON ss.chunk_id=r.chunk_id AND ss.source_id=r.source_id WHERE r.required_credits IS NOT NULL AND r.review_status='review_required' AND (r.chunk_id IS NULL OR ss.chunk_id IS NULL OR ss.review_status <> 'review_required')",
    "unverified_requirement": "SELECT count(*) FROM requirements r WHERE r.required_credits IS NOT NULL AND r.review_status='unverified'",
    "requirement_evidence_status_mismatch": "SELECT count(*) FROM requirements r LEFT JOIN source_sections ss ON ss.chunk_id=r.chunk_id WHERE r.chunk_id IS NOT NULL AND (ss.chunk_id IS NULL OR ss.source_id <> r.source_id OR ss.review_status <> r.review_status)",
    "orphan_requirement_evidence": "SELECT count(*) FROM requirements r LEFT JOIN source_sections ss ON ss.chunk_id=r.chunk_id AND ss.source_id=r.source_id WHERE r.chunk_id IS NOT NULL AND ss.chunk_id IS NULL",
}


_TRUST_CHECKS = {
    "verified_section_trust_mismatch": """
        SELECT count(*)
        FROM source_sections ss
        LEFT JOIN source_authenticity sa ON sa.source_id=ss.source_id
        LEFT JOIN section_extraction_quality sq ON sq.chunk_id=ss.chunk_id
        WHERE ss.review_status='verified'
          AND (sa.authenticity_status IS NULL OR sa.authenticity_status <> 'verified'
               OR sq.extraction_quality IS NULL OR sq.extraction_quality <> 'verified')
    """,
    "invalid_source_taxonomy": """
        SELECT count(*)
        FROM source_taxonomy
        WHERE doc_type NOT IN ('policy','notice','guide','curriculum','course_catalog','unknown')
    """,
    "verified_field_without_complete_lineage": """
        SELECT count(*)
        FROM field_verifications
        WHERE verification_status='verified'
          AND (source_id='' OR source_sha256 IS NULL OR source_sha256=''
               OR chunk_id IS NULL OR physical_page IS NULL OR table_row IS NULL
               OR cell_ref IS NULL OR cell_ref='' OR text_span IS NULL OR text_span='')
    """,
    "verified_course_without_all_verified_fields": """
        SELECT count(*)
        FROM program_courses pc
        WHERE pc.review_status='verified'
          AND (
              (SELECT count(*) FROM field_verifications fv
               WHERE fv.entity_type='program_course' AND fv.record_id=pc.record_id
                 AND fv.field_name IN ('module','credits','semester','nature')
                 AND fv.verification_status='verified') <> 4
          )
    """,
    "verified_requirement_without_all_verified_fields": """
        SELECT count(*)
        FROM requirements r
        WHERE r.required_credits IS NOT NULL AND r.review_status='verified'
          AND (
              (SELECT count(*) FROM field_verifications fv
               WHERE fv.entity_type='requirement' AND fv.record_id=r.record_id
                 AND fv.field_name IN ('module','required_credits')
                 AND fv.verification_status='verified') <> 2
          )
    """,
}


def _database_uri(path: Path) -> str:
    return f"file:{path.resolve().as_posix()}?mode=ro&immutable=1"


def _has_table(connection: sqlite3.Connection, table: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _metadata(connection: sqlite3.Connection) -> dict[str, str]:
    if not _has_table(connection, "metadata"):
        return {}
    return {
        str(key): str(value)
        for key, value in connection.execute("SELECT key, value FROM metadata").fetchall()
    }


def _reconciliation_failures(metadata: dict[str, str]) -> int:
    """Validate that the pre-insert accounting remains a conservation law."""

    if metadata.get("reconciliation_contract") != "input=accepted+exact_duplicates+quarantined":
        return 1
    raw = metadata.get("reconciliation_counts")
    if raw is None:
        return 1
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError:
        return 1
    if not isinstance(parsed, dict) or not parsed:
        return 1
    for value in parsed.values():
        if not isinstance(value, dict):
            return 1
        try:
            input_count = int(value["input"])
            accepted = int(value["accepted"])
            duplicates = int(value["exact_duplicates"])
            quarantined = int(value["quarantined"])
        except (KeyError, TypeError, ValueError):
            return 1
        if min(input_count, accepted, duplicates, quarantined) < 0:
            return 1
        if input_count != accepted + duplicates + quarantined:
            return 1
    return 0


def verify_database(database: str | Path) -> dict[str, int]:
    """Run row, trust and pre-insert reconciliation checks without mutation."""

    path = Path(database)
    if not path.is_file():
        raise DatasetVerificationError(f"dataset database is missing: {path}")
    try:
        connection = sqlite3.connect(_database_uri(path), uri=True)
    except sqlite3.Error as exc:
        raise DatasetVerificationError(f"cannot open dataset database read-only: {path}") from exc
    try:
        results = {name: int(connection.execute(sql).fetchone()[0]) for name, sql in CHECKS.items()}
        required_tables = {"source_authenticity", "section_extraction_quality", "field_verifications", "source_taxonomy"}
        if required_tables.issubset(
            {
                str(row[0])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
        ):
            results.update(
                {
                    name: int(connection.execute(sql).fetchone()[0])
                    for name, sql in _TRUST_CHECKS.items()
                }
            )
        else:
            results["released_schema_trust_tables_missing"] = 1
        results["reconciliation_metadata_invalid"] = _reconciliation_failures(_metadata(connection))
        return results
    except sqlite3.Error as exc:
        raise DatasetVerificationError("dataset schema cannot satisfy verification queries") from exc
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=Path("data/academic.sqlite3"))
    parser.add_argument(
        "--allow-review-required-requirements",
        action="store_true",
        help="allow requirements whose provenance is explicitly review_required; never allows unverified data",
    )
    args = parser.parse_args()
    try:
        results = verify_database(args.database)
    except DatasetVerificationError as exc:
        raise SystemExit(f"dataset verification failed: {exc}") from exc
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    allowed = {"review_required_requirement"} if args.allow_review_required_requirements else set()
    failures = {name: value for name, value in results.items() if value and name not in allowed}
    if failures:
        raise SystemExit(f"dataset verification failed: {failures}")


if __name__ == "__main__":
    main()
