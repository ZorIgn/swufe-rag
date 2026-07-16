"""End-to-end final-answer audit over the user's 100 curriculum questions."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any

from app.runtime_v5 import build_local_query_plan_runtime


ROOT = Path(__file__).parents[1]
CASES = ROOT / "eval" / "curriculum_2023_100.json"
OUTPUT = ROOT / "analysis-output" / "query-plan-v2-eval"
EXPECTED_SQL = {
    *range(46, 61),
    *range(61, 80),
    81,
    82,
    83,
    84,
    85,
    94,
    96,
    97,
    98,
}


def _source_pages(answer: str) -> list[int]:
    return [int(value) for value in re.findall(r"原文件第(\d+)页", answer)]


def _row(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    case_id = int(case["id"])
    path = str(result.get("execution_path") or "")
    expected = "sql" if case_id in EXPECTED_SQL else "rag"
    actual = "sql" if path.startswith("sql") else "rag" if path.startswith("rag") else path
    citations = result.get("citations") or []
    return {
        **case,
        "expected_tool": expected,
        "actual_tool": actual,
        "tool_match": actual == expected,
        "execution_path": path,
        "query_plan": result.get("query_plan"),
        "sql_coverage": result.get("sql_coverage"),
        "fallback": result.get("fallback"),
        "refused": bool(result.get("refused")),
        "clarified": path == "clarify",
        "citation_count": len(citations),
        "source_pages": _source_pages(str(result.get("answer_md") or "")),
        "has_page_citation": bool(_source_pages(str(result.get("answer_md") or ""))),
        "latency_ms": result.get("latency_ms"),
        "answer_md": result.get("answer_md"),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)

    def bucket(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(items),
            "answered": sum(not item["refused"] and not item["clarified"] for item in items),
            "refused": sum(item["refused"] for item in items),
            "clarified": sum(item["clarified"] for item in items),
            "with_page_citation": sum(item["has_page_citation"] for item in items),
            "tool_match": sum(item["tool_match"] for item in items),
            "paths": dict(Counter(item["execution_path"] for item in items)),
        }

    return {
        **bucket(rows),
        "by_group": {name: bucket(items) for name, items in groups.items()},
        "refused_ids": [item["id"] for item in rows if item["refused"]],
        "clarified_ids": [item["id"] for item in rows if item["clarified"]],
        "tool_mismatch_ids": [item["id"] for item in rows if not item["tool_match"]],
    }


def main() -> None:
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    runtime = build_local_query_plan_runtime()
    rows = []
    for index, case in enumerate(cases, 1):
        try:
            result = runtime.handle_question(
                case["question"],
                cohort="2023",
                session_id=f"benchmark-{case['id']}",
            )
            rows.append(_row(case, result))
        except Exception as exc:
            rows.append(
                {
                    **case,
                    "expected_tool": "sql" if int(case["id"]) in EXPECTED_SQL else "rag",
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
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
