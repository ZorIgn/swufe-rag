"""Restore and fill semester fields only when the original PDF row states one."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

from academic_audit.full_catalog import COURSE_CODE_RE


SEMESTER_RE = re.compile(
    r"(?:[1-8]|S[1-4])(?:[-—–](?:[1-8]|S[1-4]))?",
    re.I,
)


def _compact_cell(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).upper()


def repair(
    catalog_path: str | Path = "data/curriculum_catalog_v2.json",
    *,
    baseline_database: str | Path = "data/academic_v2.sqlite3",
    raw_dir: str | Path = "data/raw",
) -> dict[str, Any]:
    target = Path(catalog_path)
    catalog = json.loads(target.read_text(encoding="utf-8"))
    connection = sqlite3.connect(baseline_database)
    connection.row_factory = sqlite3.Row
    baseline = connection.execute(
        """
        SELECT c.cohort,c.major,c.module,c.course_code,c.source_page,s.doc_title
        FROM course_offerings AS c JOIN document_sources AS s USING(source_id)
        WHERE c.semester IS NULL OR c.semester='' OR c.semester='未标注'
        """
    ).fetchall()
    connection.close()
    keys = {
        (
            str(row["cohort"]),
            row["major"],
            row["module"],
            str(row["course_code"] or "").upper(),
            int(row["source_page"]),
            row["doc_title"],
        )
        for row in baseline
    }
    affected: list[dict[str, Any]] = []
    for course in catalog["courses"]:
        key = (
            str(course["cohort"]),
            course["major"],
            course["module"],
            str(course.get("code") or "").upper(),
            int(course["page"]),
            course["source_title"],
        )
        if key in keys:
            course["semester"] = "未标注"
            affected.append(course)

    pages: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for course in affected:
        pages[(course["source_title"], int(course["page"]))].append(course)
    by_title = {
        f"西南财经大学{year}级本科人才培养方案（完整总册）":
        Path(raw_dir) / "school" / f"{str(year)[-2:]}级培养方案.pdf"
        for year in range(2017, 2025)
    }
    extracted: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    import pdfplumber

    documents: dict[str, Any] = {}
    try:
        for (title, page), scoped in sorted(pages.items()):
            document = documents.get(title)
            if document is None:
                document = pdfplumber.open(by_title[title])
                documents[title] = document
            wanted = {str(course.get("code") or "").upper() for course in scoped}
            for table in document.pages[page - 1].extract_tables() or []:
                for row in table:
                    cells = [_compact_cell(cell) for cell in row]
                    code = next(
                        (
                            match.group(0)
                            for cell in cells
                            if (match := COURSE_CODE_RE.search(cell))
                        ),
                        None,
                    )
                    if code not in wanted:
                        continue
                    values = [cell for cell in cells if SEMESTER_RE.fullmatch(cell)]
                    for value in values:
                        extracted[(title, page, code)].add(value)
    finally:
        for document in documents.values():
            document.close()

    repaired = 0
    ambiguous = 0
    for course in affected:
        values = extracted.get(
            (course["source_title"], int(course["page"]), str(course.get("code") or "").upper()),
            set(),
        )
        if len(values) == 1:
            course["semester"] = next(iter(values))
            repaired += 1
        elif len(values) > 1:
            ambiguous += 1
    remaining = [course for course in affected if course.get("semester") == "未标注"]
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    report = {
        "baseline_unmarked": len(affected),
        "pdf_row_repaired": repaired,
        "ambiguous_pdf_rows": ambiguous,
        "remaining_without_declared_semester": len(remaining),
        "remaining_by_cohort": {
            cohort: sum(str(course["cohort"]) == cohort for course in remaining)
            for cohort in sorted({str(course["cohort"]) for course in remaining})
        },
        "remaining_by_module": dict(
            sorted(
                {
                    module: sum(course["module"] == module for course in remaining)
                    for module in {course["module"] for course in remaining}
                }.items(),
                key=lambda item: -item[1],
            )
        ),
    }
    output = Path("analysis-output/full-system-v2/semester-pdf-repair.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    print(json.dumps(repair(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
