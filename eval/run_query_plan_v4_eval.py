"""Run the 100-question context-aware benchmark on runtime v6."""

from __future__ import annotations

import json

from app.runtime_v6 import build_local_query_plan_runtime
from eval.run_query_plan_v2_eval import CASES, OUTPUT, _row, _summary


def main() -> None:
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    runtime = build_local_query_plan_runtime()
    rows = []
    for index, case in enumerate(cases, 1):
        question = case["question"]
        scope = case["scope"]
        if scope in {"计算机科学与技术专业", "人工智能专业"} and scope not in question:
            question = f"{scope} {question}"
        try:
            result = runtime.handle_question(
                question,
                cohort="2023",
                session_id=f"benchmark-v4-{case['id']}",
            )
            rows.append(_row(case, result))
        except Exception as exc:
            rows.append(
                {
                    **case,
                    "expected_tool": "unknown",
                    "actual_tool": "error",
                    "tool_match": False,
                    "execution_path": "error",
                    "query_plan": None,
                    "sql_coverage": None,
                    "fallback": None,
                    "refused": True,
                    "clarified": False,
                    "citation_count": 0,
                    "source_pages": [],
                    "has_page_citation": False,
                    "latency_ms": None,
                    "answer_md": f"{type(exc).__name__}: {exc}",
                }
            )
        print(f"{index:03d}/100 q{case['id']:03d}", flush=True)
    summary = _summary(rows)
    target = OUTPUT.parent / "query-plan-v4-eval"
    target.mkdir(parents=True, exist_ok=True)
    (target / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (target / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
