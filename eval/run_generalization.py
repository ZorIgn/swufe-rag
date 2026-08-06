"""Database-driven generalization test; no hard-coded major or question case."""

from __future__ import annotations

import argparse
import json
import random

from academic.database import AcademicRepository
from agent.factory import build_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="data/academic.sqlite3")
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()
    repository = AcademicRepository(args.database)
    rows = repository._all("SELECT program_id, canonical_name, cohort FROM programs ORDER BY program_id")  # noqa: SLF001 - evaluation truth query
    selected = random.Random(20260807).sample(rows, min(args.samples, len(rows)))
    runtime = build_runtime(args.database)
    outcomes = []
    for row in selected:
        courses = repository.list_courses(cohort=int(row["cohort"]), program_id=str(row["program_id"]))
        if not courses:
            continue
        course = courses[0]
        answer, state = runtime.ask(f"{row['cohort']}级{row['canonical_name']}{course.name}是多少学分？")
        outcomes.append({"program_id": row["program_id"], "course_id": course.course_id, "expected_code": course.code, "passed": not answer.refused and (course.code or "") in answer.answer_md, "operations": [operation.type for operation in state.plan.operations] if state.plan else []})
    report = {"sample_count": len(outcomes), "passed": sum(item["passed"] for item in outcomes), "outcomes": outcomes}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["passed"] != report["sample_count"]:
        raise SystemExit("generalization test failed")


if __name__ == "__main__":
    main()
