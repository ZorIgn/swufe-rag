"""Conservative full-book section-label repair.

Unlike the first draft, shared 2024 professional-category tables are left
untouched because one physical page legitimately represents several majors.
"""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import re
from typing import Any


CHUNKS = Path("data/chunks.jsonl")
CATALOG = Path("data/curriculum_catalog_v2.json")
REPORT = Path("analysis-output/full-system-v2/scope-label-repair.json")
PAGE_RE = re.compile(r"(?:原文件)?第(\d+)页")


def _page_range(value: str) -> tuple[int, int] | None:
    values = [int(item) for item in re.findall(r"\d+", value)]
    return (min(values), max(values)) if values else None


def _intervals(
    catalog: dict[str, Any],
) -> tuple[dict[str, list[tuple[int, int, str | None]]], list[dict[str, Any]]]:
    grouped: dict[str, dict[tuple[int, int], set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for plan in catalog["plans"]:
        title = str(plan["source_title"])
        if "完整总册" not in title:
            continue
        pages = _page_range(str(plan.get("source_pages") or ""))
        if pages:
            grouped[title][pages].add(str(plan["major"]))

    result: dict[str, list[tuple[int, int, str | None]]] = {}
    ambiguous: list[dict[str, Any]] = []
    for title, ranges in grouped.items():
        previous_end: int | None = None
        values: list[tuple[int, int, str | None]] = []
        for index, ((course_start, course_end), majors) in enumerate(sorted(ranges.items())):
            start = course_start if index == 0 or previous_end is None else previous_end + 1
            major = next(iter(majors)) if len(majors) == 1 else None
            if major is None:
                ambiguous.append(
                    {
                        "doc_title": title,
                        "pages": f"{start}-{course_end}",
                        "majors": sorted(majors),
                    }
                )
            if start <= course_end:
                values.append((start, course_end, major))
            previous_end = max(previous_end or 0, course_end)
        result[title] = values
    return result, ambiguous


def _scope(
    intervals: list[tuple[int, int, str | None]], page: int
) -> str | None:
    values = [major for start, end, major in intervals if start <= page <= end]
    return values[0] if len(values) == 1 else None


def _article(article: str, major: str) -> str:
    label = f"{major}人才培养方案"
    parts = article.split(" / ")
    if len(parts) == 1 and PAGE_RE.search(article):
        return f"{label} / {article}"
    parts[0] = label
    return " / ".join(parts)


def main() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    intervals, ambiguous = _intervals(catalog)
    temporary = CHUNKS.with_suffix(".jsonl.scope.tmp")
    changed = 0
    by_title: dict[str, int] = defaultdict(int)
    row_count = 0
    with CHUNKS.open("r", encoding="utf-8") as source, temporary.open(
        "w", encoding="utf-8", newline="\n"
    ) as target:
        for line in source:
            if not line.strip():
                continue
            chunk = json.loads(line)
            row_count += 1
            title = str(chunk["doc_title"])
            match = PAGE_RE.search(str(chunk["article"]))
            major = _scope(intervals.get(title, []), int(match.group(1))) if match else None
            if major:
                old = str(chunk["article"])
                new = _article(old, major)
                if old != new:
                    old_prefix = f"《{title}》{old}"
                    new_prefix = f"《{title}》{new}"
                    text = str(chunk["text"])
                    if not text.startswith(old_prefix):
                        raise RuntimeError(f"chunk prefix mismatch: {chunk['chunk_id']}")
                    chunk["article"] = new
                    chunk["text"] = new_prefix + text[len(old_prefix) :]
                    changed += 1
                    by_title[title] += 1
            target.write(json.dumps(chunk, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, CHUNKS)
    report = {
        "chunk_rows": row_count,
        "changed_scope_labels": changed,
        "books_changed": dict(sorted(by_title.items())),
        "ambiguous_ranges_preserved": ambiguous,
        "body_content_changed": False,
        "physical_pages_changed": False,
        "requires_embedding_refresh": bool(changed),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
