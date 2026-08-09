"""Database-driven, targeted-course generalization evaluation."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

from academic.database import AcademicRepository
from agent.factory import build_runtime


def _probability(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("threshold must be a number in [0, 1]") from exc
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("threshold must be in [0, 1]")
    return parsed


def _credit_label(value: object) -> str:
    try:
        return f"{float(str(value)):g} 学分"
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"course credits must be numeric for generalization evaluation: {value!r}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=Path("data/academic.sqlite3"))
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--retrieval-mode", choices=("lexical", "hybrid"))
    parser.add_argument(
        "--output", type=Path, default=Path("eval/reports/generalization.json")
    )
    parser.add_argument("--min-pass-rate", type=_probability, default=1.0)
    args = parser.parse_args()
    if args.samples < 1:
        raise SystemExit("--samples must be at least one")

    repository = AcademicRepository(args.database)
    try:
        rows = repository._all(  # noqa: SLF001 - evaluation truth query
            "SELECT program_id, canonical_name, cohort FROM programs ORDER BY program_id"
        )
        candidates: list[tuple[Any, Any]] = []
        for row in rows:
            courses = repository.list_courses(
                cohort=int(row["cohort"]), program_id=str(row["program_id"])
            )
            # A generalization probe must name one concrete course code.  A
            # broad course listing can otherwise contain the expected code and
            # create a false positive without exercising course-detail lookup.
            course = next(
                (
                    value
                    for value in courses
                    if str(value.code or "").strip() and str(value.name or "").strip()
                ),
                None,
            )
            if course is not None:
                candidates.append((row, course))
    finally:
        repository.close()

    if len(candidates) < args.samples:
        raise SystemExit(
            "generalization evaluation requires "
            f"{args.samples} eligible program/course probes, found {len(candidates)}"
        )
    selected = random.Random(20260807).sample(candidates, args.samples)

    if args.retrieval_mode:
        os.environ["SWUFE_RETRIEVAL_MODE"] = args.retrieval_mode
    runtime = build_runtime(args.database)
    outcomes: list[dict[str, object]] = []
    try:
        for row, course in selected:
            cohort = int(row["cohort"])
            program_id = str(row["program_id"])
            program_name = str(row["canonical_name"])
            course_code = str(course.code)
            course_name = str(course.name)
            expected_credit = _credit_label(course.credits)
            question = f"{cohort}级{program_name}（{course_code}）是多少学分？"
            answer, state = runtime.ask(question)
            operations = (
                tuple(str(operation.type) for operation in state.plan.operations)
                if state.plan
                else ()
            )
            detail_lookup = (
                len(state.plan.operations) == 1
                and state.plan.operations[0].type == "get_course_detail"
                and str(getattr(state.plan.operations[0].args, "course_code", "")) == course_code
                if state.plan
                else False
            )
            answer_contains = all(
                token in str(answer.answer_md)
                for token in (course_name, course_code, expected_credit)
            )
            passed = not answer.refused and detail_lookup and answer_contains
            outcomes.append(
                {
                    "program_id": program_id,
                    "course_id": str(course.course_id),
                    "question": question,
                    "expected_name": course_name,
                    "expected_code": course_code,
                    "expected_credits": expected_credit,
                    "operations": list(operations),
                    "targeted_detail_lookup": detail_lookup,
                    "answer_contains_target": answer_contains,
                    "refused": bool(answer.refused),
                    "passed": passed,
                }
            )
    finally:
        runtime.repository.close()

    passed_count = sum(bool(item["passed"]) for item in outcomes)
    pass_rate = passed_count / len(outcomes)
    report = {
        "requested_samples": args.samples,
        "eligible_samples": len(candidates),
        "sample_count": len(outcomes),
        "passed_count": passed_count,
        "pass_rate": pass_rate,
        "threshold": args.min_pass_rate,
        "passed": pass_rate >= args.min_pass_rate,
        "outcomes": outcomes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(
            "generalization quality gate failed: "
            f"pass_rate={pass_rate:.3f} < threshold={args.min_pass_rate:.3f}"
        )


if __name__ == "__main__":
    main()
