"""Deterministic retrieval/refusal audit for the bundled Demo knowledge base."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.runtime import build_demo_runtime


DEFAULT_CASES = Path(__file__).parents[1] / "demo" / "queries.json"


def evaluate(cases_path: str | Path = DEFAULT_CASES) -> dict[str, Any]:
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    runtime = build_demo_runtime()
    rows: list[dict[str, Any]] = []
    retrieval_hits = 0
    retrieval_total = 0
    scope_pollution = 0
    refusal_correct = 0
    for case in cases:
        results = runtime.retrieve(
            case["question"],
            top_k=5,
            college=case.get("college"),
            cohort=case.get("cohort"),
        )
        ids = [item["chunk_id"] for item in results]
        expected = set(case["expected_chunk_ids"])
        hit = not expected or bool(expected & set(ids))
        if expected:
            retrieval_total += 1
            retrieval_hits += int(hit)
        polluted = any(
            item["level"] != "校级"
            and case.get("college")
            and item["college"] != case["college"]
            for item in results
        )
        scope_pollution += int(polluted)
        answer = runtime.ask(
            case["question"],
            top_k=5,
            college=case.get("college"),
            cohort=case.get("cohort"),
        )
        refusal_ok = bool(answer["refused"]) == bool(case["should_refuse"])
        refusal_correct += int(refusal_ok)
        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "retrieved": ids,
                "retrieval_hit": hit,
                "scope_pollution": polluted,
                "refusal_correct": refusal_ok,
                "refused": answer["refused"],
            }
        )
    return {
        "case_count": len(cases),
        "retrieval_recall_at_5": retrieval_hits / max(retrieval_total, 1),
        "scope_pollution_count": scope_pollution,
        "refusal_accuracy": refusal_correct / max(len(cases), 1),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    args = parser.parse_args()
    print(json.dumps(evaluate(args.cases), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
