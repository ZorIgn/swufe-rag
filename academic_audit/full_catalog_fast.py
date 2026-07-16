"""Linear-time runner for :mod:`academic_audit.full_catalog`.

PDF extraction workers skip the legacy O(courses*chunks) evidence lookup.
After all books finish, one streaming pass binds every course row to a trusted
table chunk by (document, physical page, course code).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import date
import json
from pathlib import Path
from typing import Any

import academic_audit.full_catalog as base
from academic_audit.catalog import _constraint_rules, _source_hash


_ORIGINAL_EXTRACT = base._extract_book


def _extract_without_evidence(
    source: dict[str, str], raw_dir: str, chunks_path: str
) -> dict[str, Any]:
    # This assignment is process-local. Evidence is bound once in the parent.
    base._evidence = lambda *args, **kwargs: None  # type: ignore[assignment]
    base._following_evidence = lambda *args, **kwargs: []  # type: ignore[assignment]
    return _ORIGINAL_EXTRACT(source, raw_dir, chunks_path)


def _evidence_index(chunks_path: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    values: dict[tuple[str, int, str], dict[str, Any]] = {}
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            chunk = json.loads(line)
            if not chunk["is_table"] or "完整总册" not in chunk["doc_title"]:
                continue
            page = base._page(chunk["article"])
            if page is None:
                continue
            for code in base.COURSE_CODE_RE.findall(chunk["text"].upper()):
                key = (chunk["doc_title"], page, code)
                candidate = {
                    "chunk_id": chunk["chunk_id"],
                    "doc_title": chunk["doc_title"],
                    "article": chunk["article"],
                    "quote": chunk["text"][:500],
                    "page_url": chunk["page_url"],
                    "file_url": chunk["file_url"],
                }
                current = values.get(key)
                if current is None or len(candidate["quote"]) < len(current["quote"]):
                    values[key] = candidate
    return values


def build_full_catalog_fast(
    *,
    sources_path: str | Path = "data/sources.csv",
    raw_dir: str | Path = "data/raw",
    chunks_path: str | Path = "data/chunks.jsonl",
    workers: int = 4,
) -> dict[str, Any]:
    sources_file = Path(sources_path)
    raw_root = Path(raw_dir)
    chunks_file = Path(chunks_path)
    with sources_file.open("r", encoding="utf-8-sig", newline="") as handle:
        sources = list(csv.DictReader(handle))
    books = [
        source
        for source in sources
        if base.BOOK_RE.search(source["file"].replace("\\", "/"))
    ]
    parts: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(books)))) as pool:
        futures = {
            pool.submit(
                _extract_without_evidence, source, str(raw_root), str(chunks_file)
            ): source
            for source in books
        }
        for future in as_completed(futures):
            parts.append(future.result())

    evidence = _evidence_index(chunks_file)
    courses: list[dict[str, Any]] = []
    for part in parts:
        for course in part["courses"]:
            course["evidence"] = evidence.get(
                (course["source_title"], int(course["page"]), course["code"])
            )
            courses.append(course)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for course in courses:
        grouped[(course["cohort"], course["major"])].append(course)

    modules_by_cohort: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for part in parts:
        modules_by_cohort[part["cohort"]].extend(part["modules"])

    plans: list[dict[str, Any]] = []
    for (cohort, major), plan_courses in sorted(grouped.items()):
        plan_modules: list[dict[str, Any]] = []
        for module_name in sorted({course["module"] for course in plan_courses}):
            module_courses = [
                course for course in plan_courses if course["module"] == module_name
            ]
            catalog_credits = round(
                sum(float(course.get("credits") or 0) for course in module_courses), 2
            )
            module = next(
                (
                    value
                    for value in modules_by_cohort[cohort]
                    if value["name"] == module_name
                ),
                {
                    "name": module_name,
                    "required_credits": None,
                    "listed_credits": None,
                    "rule_text": "",
                    "evidence": None,
                    "supporting_evidence": [],
                    "source_title": plan_courses[0]["source_title"],
                },
            )
            required = module.get("required_credits")
            if required is None:
                required = module.get("listed_credits")
            if required is None and module_courses and all(
                "必修" in course["nature"] for course in module_courses
            ):
                required = catalog_credits
            plan_modules.append(
                {
                    **module,
                    "required_credits": required,
                    "catalog_credits": catalog_credits,
                    "course_count": len(module_courses),
                    "constraints": _constraint_rules(
                        module.get("rule_text", ""), module_courses, major
                    ),
                }
            )
        pages = [int(course["page"]) for course in plan_courses]
        plans.append(
            {
                "college": Counter(
                    course["college"] for course in plan_courses
                ).most_common(1)[0][0],
                "cohort": cohort,
                "major": major,
                "source_title": plan_courses[0]["source_title"],
                "course_count": len(plan_courses),
                "source_pages": f"{min(pages)}-{max(pages)}",
                "rag_ready": True,
                "sql_ready": True,
                "modules": plan_modules,
            }
        )

    coverage: list[dict[str, Any]] = []
    for cohort in sorted({part["cohort"] for part in parts}):
        part = next(value for value in parts if value["cohort"] == cohort)
        scoped_plans = [plan for plan in plans if plan["cohort"] == cohort]
        scoped_courses = [course for course in courses if course["cohort"] == cohort]
        coverage.append(
            {
                "cohort": cohort,
                "physical_pages": part["physical_pages"],
                "table_pages": part["table_pages"],
                "plan_count": len(scoped_plans),
                "course_rows": len(scoped_courses),
                "hours_complete_rows": sum(
                    course.get("total_hours") is not None for course in scoped_courses
                ),
                "plans_without_courses": [],
            }
        )

    source_paths = [
        sources_file,
        chunks_file,
        *[raw_root / source["file"] for source in books],
    ]
    return {
        "catalog_version": "2.0",
        "generated_at": date.today().isoformat(),
        "source_sha256": _source_hash(source_paths),
        "plan_count": len(plans),
        "course_count": len(courses),
        "coverage": coverage,
        "plans": plans,
        "courses": sorted(
            courses,
            key=lambda course: (
                course["cohort"],
                course["major"],
                course["page"],
                course["source_row"],
                course["code"],
            ),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/curriculum_catalog_v2.json")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = build_full_catalog_fast(workers=args.workers)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(target)
    print(
        json.dumps(
            {key: result[key] for key in ("plan_count", "course_count", "coverage")},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

