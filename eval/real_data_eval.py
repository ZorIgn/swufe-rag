"""Offline development audit over reviewed real chunks.

The review runtime deliberately uses a lightweight hashing encoder and an
extractive client.  Its metrics are useful for regression and Web/data QA, but
they do not replace the final BGE/LLM evaluation required by the project plan.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.runtime import build_review_runtime


DEFAULT_CASES = Path(__file__).with_name("real_dev_queries.json")


def evaluate(
    cases_path: str | Path = DEFAULT_CASES,
    chunks_path: str | Path = "data/chunks.jsonl",
) -> dict[str, Any]:
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    runtime = build_review_runtime(chunks_path)
    rows: list[dict[str, Any]] = []
    retrieval_hits = 0
    retrieval_total = 0
    scope_pollution_count = 0
    refusal_correct = 0
    answer_support_correct = 0
    answer_support_total = 0

    for case in cases:
        retrieved = runtime.retrieve(
            case["question"],
            top_k=5,
            college=case.get("college"),
            cohort=case.get("cohort"),
        )
        expected_docs = set(case.get("expected_docs", []))
        retrieved_docs = [item["doc_title"] for item in retrieved]
        retrieval_hit = not expected_docs or bool(expected_docs & set(retrieved_docs))
        if expected_docs:
            retrieval_total += 1
            retrieval_hits += int(retrieval_hit)

        polluted = any(
            (
                item["level"] == "院级"
                and case.get("college")
                and item["college"] != case["college"]
            )
            or (
                case.get("cohort")
                and item["cohort"] not in {"不限", case["cohort"]}
            )
            for item in retrieved
        )
        scope_pollution_count += int(polluted)

        result = runtime.ask(
            case["question"],
            top_k=5,
            college=case.get("college"),
            cohort=case.get("cohort"),
        )
        refusal_ok = bool(result["refused"]) == bool(case["should_refuse"])
        refusal_correct += int(refusal_ok)
        answer_terms = case.get("answer_must_contain", [])
        answer_support_ok = not answer_terms or all(
            term.replace(" ", "") in result["answer_md"].replace(" ", "")
            for term in answer_terms
        )
        if answer_terms:
            answer_support_total += 1
            answer_support_correct += int(answer_support_ok)
        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "retrieved_docs": retrieved_docs,
                "retrieval_hit": retrieval_hit,
                "scope_pollution": polluted,
                "refused": result["refused"],
                "refusal_correct": refusal_ok,
                "answer_support_correct": answer_support_ok,
                "answer_md": result["answer_md"],
            }
        )

    return {
        "runtime": "offline-review-not-production-bge",
        "case_count": len(cases),
        "retrieval_recall_at_5": retrieval_hits / max(retrieval_total, 1),
        "scope_pollution_count": scope_pollution_count,
        "refusal_accuracy": refusal_correct / max(len(cases), 1),
        "answer_support_accuracy": answer_support_correct
        / max(answer_support_total, 1),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.cases, args.chunks), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
