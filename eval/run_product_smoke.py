"""Run the product-level questions that gate a full data release."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent.factory import build_runtime


@dataclass(frozen=True)
class Case:
    identifier: str
    question: str
    expected: Literal["answer", "clarify", "refuse"]
    operations: tuple[str, ...] = ()
    contains: tuple[str, ...] = ()


CASES = (
    Case(
        "semester-electives",
        "2023级人工智能专业第6学期有哪些选修课？",
        "answer",
        ("list_courses",),
    ),
    Case(
        "module-minimum",
        "2024级网络空间安全专业的专业选修模块最低要修多少学分？",
        "answer",
        ("get_module_requirements",),
        ("8 学分",),
    ),
    Case(
        "readme-course-unscoped",
        "离散数学多少学分，在哪个学期开设？",
        "clarify",
    ),
    Case(
        "readme-course-scoped",
        "2024级人工智能专业的离散数学多少学分，在哪个学期开设？",
        "answer",
        ("get_course_detail",),
        ("离散数学", "3 学分"),
    ),
    Case(
        "english-exemption",
        "大学英语达到什么条件可以免修？",
        "answer",
        ("retrieve_policy",),
    ),
    Case(
        "sql-rag-composite",
        "2024级人工智能专业的专业选修课最低学分和课程有哪些？另外大学英语免修有什么规定？",
        "answer",
        ("get_module_requirements", "list_courses", "retrieve_policy"),
        ("8 学分", "算法交易", "大学外语"),
    ),
    Case(
        "progress-audit",
        "2024级人工智能专业，我已经修完算法交易，还差多少学分？",
        "answer",
        ("get_graduation_requirements", "audit_completed_courses"),
    ),
    Case(
        "graduation-feasibility",
        "2024级人工智能专业，我已经修完算法交易，大四前能毕业吗？",
        "answer",
        (
            "list_courses",
            "get_graduation_requirements",
            "audit_completed_courses",
            "list_courses_before_semester",
            "list_unavoidable_courses",
            "check_curriculum_feasibility",
        ),
        ("结论",),
    ),
    Case(
        "program-comparison",
        "2024级人工智能专业和网络空间安全专业的课程有什么区别？",
        "answer",
        ("compare_programs",),
        ("独有课程",),
    ),
    Case("out-of-corpus", "火星殖民规定是什么？", "refuse", ("retrieve_policy",)),
)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=Path("data/academic.sqlite3"))
    parser.add_argument("--retrieval-mode", choices=("lexical", "hybrid"))
    args = parser.parse_args()
    if args.retrieval_mode:
        os.environ["SWUFE_RETRIEVAL_MODE"] = args.retrieval_mode
    runtime = build_runtime(args.database)
    outcomes: list[dict[str, object]] = []
    try:
        for case in CASES:
            answer, state = runtime.ask(case.question)
            observed_operations = tuple(
                operation.type for operation in state.plan.operations
            ) if state.plan else ()
            observed = (
                "clarify"
                if answer.clarification
                else "refuse"
                if answer.refused
                else "answer"
            )
            validation_reasons = sorted(
                {
                    reason
                    for claim in answer.claims
                    for reason in claim.validation.reasons
                }
            )
            outcome_matches = observed == case.expected
            operations_match = not case.operations or set(observed_operations) == set(
                case.operations
            )
            content_matches = all(value in answer.answer_md for value in case.contains)
            passed = outcome_matches and operations_match and content_matches
            outcomes.append(
                {
                    "id": case.identifier,
                    "question": case.question,
                    "expected": case.expected,
                    "observed": observed,
                    "intent": state.normalized_query.intent if state.normalized_query else None,
                    "operations": observed_operations,
                    "claim_count": len(answer.claims),
                    "validation_reasons": validation_reasons,
                    "outcome_matches": outcome_matches,
                    "operations_match": operations_match,
                    "content_matches": content_matches,
                    "answer_preview": answer.answer_md[:300],
                    "passed": passed,
                }
            )
    finally:
        runtime.repository.close()
    passed_count = sum(bool(outcome["passed"]) for outcome in outcomes)
    report = {
        "case_count": len(outcomes),
        "passed_count": passed_count,
        "passed": passed_count == len(outcomes),
        "outcomes": outcomes,
    }
    print(json.dumps(report, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit("full-data product smoke failed")


if __name__ == "__main__":
    main()
