"""Two-case real-provider smoke test without persisting the API key."""

from __future__ import annotations

import json
import os
from pathlib import Path

from app.runtime_v4 import (
    build_local_query_plan_runtime,
    build_request_query_plan_runtime,
)


def main() -> None:
    key = os.environ.get("SWUFE_TEST_API_KEY", "").strip()
    if not key:
        raise RuntimeError("SWUFE_TEST_API_KEY is required")
    local = build_local_query_plan_runtime()
    runtime = build_request_query_plan_runtime(local, key)
    rows = []
    for index, question in enumerate(
        (
            "23级人工智能大三下有什么选修课？",
            "2023级公共外语一共要修多少学分？",
        )
    ):
        result = runtime.handle_question(question, session_id=f"live-smoke-{index}")
        rows.append(
            {
                key: result.get(key)
                for key in (
                    "mode",
                    "answer_md",
                    "refused",
                    "latency_ms",
                    "execution_path",
                    "llm_called",
                    "llm_stages",
                    "model",
                    "query_plan",
                    "sql_coverage",
                    "fallback",
                    "answer_generation_error",
                    "timings",
                )
            }
        )
    if not all(row["llm_called"] for row in rows):
        raise RuntimeError("real-provider smoke did not invoke the LLM on every case")
    output = Path("analysis-output/full-system-v2/live-query-plan-smoke.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
