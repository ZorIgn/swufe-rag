"""Audit raw-file, vector/policy, page, and structured-course coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sqlite3
from typing import Any


FULL_BOOK_RE = re.compile(r"^school/[0-9]{2}级培养方案\.pdf$", re.I)


def _category(key: str) -> str:
    key = key.replace("\\", "/")
    if FULL_BOOK_RE.fullmatch(key):
        return "full_curriculum"
    if "/training/" in "/" + key:
        return "split_curriculum"
    return "policy_guide"


def _pdf_pages(path: Path) -> int | None:
    if path.suffix.lower() != ".pdf":
        return None
    try:
        from pypdf import PdfReader

        return len(PdfReader(path).pages)
    except Exception:
        try:
            import fitz

            with fitz.open(path) as document:
                return int(document.page_count)
        except Exception:
            return None


def audit(
    database: str | Path = "data/academic_v2.sqlite3",
    *,
    raw_dir: str | Path = "data/raw",
) -> dict[str, Any]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT s.source_id, s.source_key, s.doc_title, s.local_path,
               count(DISTINCT p.chunk_id) AS chunks,
               count(DISTINCT CASE WHEN p.is_primary=1 THEN p.chunk_id END) AS primary_chunks,
               count(DISTINCT p.source_page) AS indexed_pages,
               count(DISTINCT c.id) AS course_rows
        FROM document_sources AS s
        LEFT JOIN policy_chunks AS p ON p.source_id=s.source_id
        LEFT JOIN course_offerings AS c ON c.source_id=s.source_id
        GROUP BY s.source_id ORDER BY s.source_key
        """
    ).fetchall()
    sources: list[dict[str, Any]] = []
    for row in rows:
        path = Path(row["local_path"]) if row["local_path"] else None
        sources.append(
            {
                "source_id": row["source_id"],
                "source_key": row["source_key"].replace("\\", "/"),
                "doc_title": row["doc_title"],
                "category": _category(row["source_key"]),
                "extension": Path(row["source_key"]).suffix.lower(),
                "raw_exists": bool(path and path.is_file()),
                "pdf_pages": _pdf_pages(path) if path and path.is_file() else None,
                "indexed_pages": int(row["indexed_pages"]),
                "chunks": int(row["chunks"]),
                "primary_chunks": int(row["primary_chunks"]),
                "course_rows": int(row["course_rows"]),
            }
        )
    raw_root = Path(raw_dir)
    actual = {
        path.relative_to(raw_root).as_posix()
        for path in raw_root.rglob("*")
        if path.is_file()
    }
    registered = {item["source_key"] for item in sources}
    categories: dict[str, dict[str, int]] = {}
    for name in ("full_curriculum", "split_curriculum", "policy_guide"):
        scoped = [item for item in sources if item["category"] == name]
        categories[name] = {
            "sources": len(scoped),
            "chunks": sum(item["chunks"] for item in scoped),
            "primary_chunks": sum(item["primary_chunks"] for item in scoped),
            "course_rows": sum(item["course_rows"] for item in scoped),
            "pdf_pages": sum(item["pdf_pages"] or 0 for item in scoped),
        }
    database_counts = {
        table: int(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
        for table in (
            "document_sources",
            "policy_chunks",
            "course_offerings",
            "program_requirements",
        )
    }
    connection.close()
    promotion = [
        item
        for item in sources
        if "推荐免试" in item["doc_title"] or "推免" in item["doc_title"]
    ]
    return {
        "database": str(Path(database).resolve()),
        "database_counts": database_counts,
        "categories": categories,
        "registered_source_count": len(sources),
        "raw_file_count": len(actual),
        "missing_raw_files": sorted(registered - actual),
        "unregistered_raw_files": sorted(actual - registered),
        "zero_chunk_sources": [item["source_key"] for item in sources if item["chunks"] == 0],
        "zero_course_full_books": [
            item["source_key"]
            for item in sources
            if item["category"] == "full_curriculum" and item["course_rows"] == 0
        ],
        "promotion_sources": promotion,
        "sources": sources,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 全量知识库覆盖审计",
        "",
        f"- 登记来源：{report['registered_source_count']}",
        f"- 原始文件：{report['raw_file_count']}",
        f"- policy_chunks：{report['database_counts']['policy_chunks']}",
        f"- course_offerings：{report['database_counts']['course_offerings']}",
        f"- 推免/保研来源：{len(report['promotion_sources'])}",
        f"- 缺失原文件：{len(report['missing_raw_files'])}",
        f"- 未登记原文件：{len(report['unregistered_raw_files'])}",
        f"- 零知识块来源：{len(report['zero_chunk_sources'])}",
        "",
        "| 类别 | 文件数 | 知识块 | 去重主块 | 课程行 | PDF页数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "full_curriculum": "八本完整培养方案",
        "split_curriculum": "计智拆分培养方案",
        "policy_guide": "制度、推免与操作指南",
    }
    for key, label in labels.items():
        item = report["categories"][key]
        lines.append(
            f"| {label} | {item['sources']} | {item['chunks']} | {item['primary_chunks']} | {item['course_rows']} | {item['pdf_pages']} |"
        )
    lines.extend(["", "## 推免/保研材料", ""])
    for item in report["promotion_sources"]:
        lines.append(
            f"- {item['doc_title']}：{item['chunks']} 块（去重主块 {item['primary_chunks']}）"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="data/academic_v2.sqlite3")
    parser.add_argument("--output", default="analysis-output/full-system-v2")
    args = parser.parse_args()
    report = audit(args.database)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "coverage.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "coverage.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("database_counts", "categories", "registered_source_count", "raw_file_count", "missing_raw_files", "unregistered_raw_files", "zero_chunk_sources", "zero_course_full_books")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
