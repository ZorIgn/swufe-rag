"""Verify that the production planner accepts real provider JSON."""

from __future__ import annotations

import json
import os

from academic_audit.database import AcademicDatabase
from app.runtime import _load_config
from generation.llm import OpenAICompatibleClient
from swufe_rag.query_plan_catalog_v3 import ProductionQuestionPlanner


def main() -> None:
    key = os.environ.get("SWUFE_TEST_API_KEY", "").strip()
    config = _load_config("config.advanced.yaml")["generation"]
    client = OpenAICompatibleClient(
        str(config.get("llm", "deepseek-chat")),
        api_key=key,
        temperature=0,
        max_retries=0,
        timeout_seconds=60,
    )
    planner = ProductionQuestionPlanner(
        AcademicDatabase("data/academic_v2.sqlite3"), client
    )
    plans = [
        planner.plan(question).to_dict()
        for question in (
            "23级人工智能大三下有什么选修课？",
            "2023级公共外语一共要修多少学分？",
        )
    ]
    if not all(plan["parser"] == "llm" for plan in plans):
        raise RuntimeError("production planner still fell back from real JSON")
    print(json.dumps(plans, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
