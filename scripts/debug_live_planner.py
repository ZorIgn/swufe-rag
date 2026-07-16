"""Inspect the real planner JSON and strict-schema error without storing a key."""

from __future__ import annotations

import json
import os

from app.runtime import _load_config
from generation.llm import OpenAICompatibleClient
from swufe_rag.query_plan import PLANNER_SYSTEM_PROMPT, QueryPlan, _json_object


def main() -> None:
    key = os.environ.get("SWUFE_TEST_API_KEY", "").strip()
    if not key:
        raise RuntimeError("SWUFE_TEST_API_KEY is required")
    config = _load_config("config.advanced.yaml")["generation"]
    client = OpenAICompatibleClient(
        str(config.get("llm", "deepseek-chat")),
        api_key=key,
        temperature=0,
        max_retries=0,
        timeout_seconds=60,
    )
    question = "23级人工智能大三下有什么选修课？"
    context = {
        "question": question,
        "explicit_college": None,
        "explicit_cohort": None,
        "inherited_major": None,
        "inherited_cohort": None,
    }
    raw_text = client.generate(
        PLANNER_SYSTEM_PROMPT, json.dumps(context, ensure_ascii=False)
    )
    error = None
    parsed = None
    try:
        parsed = QueryPlan.from_mapping(
            _json_object(raw_text), question=question, parser="llm"
        ).to_dict()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    print(
        json.dumps(
            {"raw": raw_text, "parsed": parsed, "error": error},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
