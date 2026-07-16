"""Real-model smoke test without persisting or printing the supplied API key."""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

from app.server_v9 import app


QUESTIONS = [
    "人工智能专业2023级大三下有哪些选修课？",
    "金融学专业2023级第一学期有哪些课程？",
    "西南财经大学推荐免试研究生的申请条件是什么？",
    "2023级公共外语一共要修多少学分？",
]


def main() -> None:
    key = os.environ.get("SWUFE_TEST_KEY", "").strip()
    if not key:
        raise RuntimeError("SWUFE_TEST_KEY is required")
    client = TestClient(app)
    rows = []
    for index, question in enumerate(QUESTIONS, 1):
        response = client.post(
            "/ask",
            headers={"X-LLM-API-Key": key},
            json={"question": question, "cohort": "2023", "session_id": f"live-v9-{index}"},
        )
        payload = response.json()
        row = {
            "question": question,
            "status_code": response.status_code,
            "execution_path": payload.get("execution_path"),
            "llm_called": payload.get("llm_called"),
            "llm_stages": payload.get("llm_stages"),
            "model": payload.get("model"),
            "parser": (payload.get("query_plan") or {}).get("parser"),
            "tool": (payload.get("query_plan") or {}).get("tool"),
            "citation_count": len(payload.get("citations") or []),
            "sources": [
                {"doc_title": item["doc_title"], "article": item["article"]}
                for item in payload.get("citations") or []
            ],
            "answer_generation_error": payload.get("answer_generation_error"),
            "timings": payload.get("timings"),
            "answer_md": payload.get("answer_md"),
        }
        rows.append(row)
        print(json.dumps({k: row[k] for k in row if k != "answer_md"}, ensure_ascii=False))
    output = Path("analysis-output/full-system-v2/live-v9-smoke.json")
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
