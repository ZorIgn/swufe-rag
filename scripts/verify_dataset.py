"""Fail fast on canonical data-integrity and evidence-trust violations."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=Path("data/academic.sqlite3"))
    parser.add_argument(
        "--allow-review-required-requirements",
        action="store_true",
        help="allow requirements whose provenance is explicitly review_required; never allows unverified data",
    )
    args = parser.parse_args()
    with sqlite3.connect(args.database) as connection:
        results = {name: int(connection.execute(sql).fetchone()[0]) for name, sql in CHECKS.items()}
    print(json.dumps(results, ensure_ascii=False, indent=2))
    allowed = (
        {"review_required_requirement"}
        if args.allow_review_required_requirements
        else set()
    )
    failures = {name: value for name, value in results.items() if value and name not in allowed}
    if failures:
        raise SystemExit(f"dataset verification failed: {failures}")


if __name__ == "__main__":
    main()
