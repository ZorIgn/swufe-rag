"""Cross-college and non-curriculum source smoke audit."""

from __future__ import annotations

import json
from pathlib import Path

from app.runtime_v8 import build_local_query_plan_runtime


CASES = [
    {"id": "finance", "question": "金融学专业2023级第一学期有哪些课程？", "expected": "sql"},
    {"id": "accounting", "question": "会计学（注册会计师方向）专业2023级第一学期有哪些课程？", "expected": "sql"},
    {"id": "law", "question": "法学专业2022级第二学期有哪些课程？", "expected": "sql"},
    {"id": "statistics", "question": "统计学专业2021级第三学期有哪些课程？", "expected": "sql"},
    {"id": "language", "question": "商务英语专业2020级第一学期有哪些课程？", "expected": "sql"},
    {"id": "administration", "question": "行政管理专业2019级第四学期有哪些课程？", "expected": "sql"},
    {"id": "insurance", "question": "保险学专业2018级第五学期有哪些课程？", "expected": "sql"},
    {"id": "clarify_college", "question": "会计学院第一学期有什么课？", "expected": "clarify"},
    {"id": "promotion", "question": "西南财经大学推荐免试研究生的申请条件是什么？", "expected": "rag"},
    {"id": "promotion_score", "question": "学校推免细则中的综合成绩如何计算？", "expected": "rag"},
    {"id": "defer_exam", "question": "西南财经大学学生申请缓考需要满足什么条件？", "expected": "rag"},
    {"id": "course_selection", "question": "本科生选课操作指南说明了哪些选课步骤？", "expected": "rag"},
]


def main() -> None:
    runtime = build_local_query_plan_runtime()
    rows = []
    for case in CASES:
        result = runtime.handle_question(
            case["question"], cohort="2023", session_id=f"full-school-{case['id']}"
        )
        plan = result["query_plan"]
        actual = plan["tool"]
        clarified = result["execution_path"] == "clarify"
        citations = result.get("citations", [])
        passed = (
            (case["expected"] == "clarify" and clarified)
            or (
                actual == case["expected"]
                and not result.get("refused")
                and bool(citations)
            )
        )
        rows.append(
            {
                **case,
                "passed": passed,
                "actual": actual,
                "execution_path": result["execution_path"],
                "parser": plan["parser"],
                "citation_count": len(citations),
                "sources": [
                    {"doc_title": item["doc_title"], "article": item["article"]}
                    for item in citations
                ],
                "answer_md": result["answer_md"],
            }
        )
        print(case["id"], "PASS" if passed else "FAIL", actual, result["execution_path"])
    output = Path("analysis-output/full-system-v2/full-school-smoke.json")
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n": len(rows), "passed": sum(row["passed"] for row in rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
