"""Repair the article-based 2024 curriculum extraction and normalize owners.

The 2017--2023 books print an explicit ``专业: ... 年级: ...`` header on
course-table pages.  The 2024 book instead groups majors into plan articles,
so its professional-code block must be parsed before table rows are assigned.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
from typing import Any, Iterable

import academic_audit.full_catalog as base
from academic_audit.catalog import _constraint_rules
from academic_audit.full_catalog_fast import _evidence_index


GENERIC_ARTICLE = "本科专业人才培养方案"
LABEL_RE = re.compile(
    r"学科门类|授予学位|专业代码|标准学制|计划学制|毕业|学分统计|实践教学"
)
CODE_RE = re.compile(r"(?<!\d)(\d{6}[A-Z]{0,3})\s*[-—]\s*", re.I)


def _stem(article: str) -> str:
    return re.sub(r"专业人才培养方案$|人才培养方案$", "", article).strip()


def article_majors(article: str, chunks: Iterable[dict[str, Any]]) -> list[str]:
    if article == GENERIC_ARTICLE:
        return []
    stem = _stem(article)
    is_category = bool(re.search(r"类(?:[（(][^）)]*[）)])?$", stem))
    if not is_category:
        stem = re.sub(
            r"^西南财经大学[—-]+电子科技大学联合学士学位",
            "",
            stem,
        )
        return [base._canonical_major(stem)]

    text = " ".join(
        chunk["text"]
        for chunk in chunks
        if base._article_root(chunk["article"]) == article
        and "专业类基本信息" in chunk["article"]
    )
    matches = list(CODE_RE.finditer(text))
    names: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[match.end() : end]
        label = LABEL_RE.search(value)
        if label:
            value = value[: label.start()]
        value = value.strip(" ,，、;/；")
        if value and re.search(r"[\u4e00-\u9fff]", value):
            names.append(base._canonical_major(value))
    return list(dict.fromkeys(names)) or [base._canonical_major(stem)]


def _normal(value: str) -> str:
    value = re.sub(r"\s+", "", value or "")
    value = re.sub(r"(?:School|Department).*$", "", value, flags=re.I)
    return value or "全校"


def _owner(courses: list[dict[str, Any]], fallback: str) -> str:
    counts = Counter(
        _normal(str(course.get("department") or ""))
        for course in courses
        if course.get("department")
    )
    if counts:
        return counts.most_common(1)[0][0]
    if "英语" in fallback or "商务英语" in fallback or "西班牙语" in fallback:
        return "经贸外语学院"
    return _normal(str(courses[0].get("college") or "全校")) if courses else "全校"


def _plans(courses: list[dict[str, Any]], modules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for course in courses:
        grouped[(str(course["cohort"]), course["major"])].append(course)
    results: list[dict[str, Any]] = []
    for (cohort, major), plan_courses in sorted(grouped.items()):
        owner = _owner(plan_courses, major)
        for course in plan_courses:
            course["college"] = owner
        plan_modules = []
        for module_name in sorted({course["module"] for course in plan_courses}):
            scoped = [course for course in plan_courses if course["module"] == module_name]
            catalog_credits = round(sum(float(course.get("credits") or 0) for course in scoped), 2)
            record = next(
                (item for item in modules if item["name"] == module_name),
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
            required = record.get("required_credits") or record.get("listed_credits")
            if required is None and scoped and all("必修" in course["nature"] for course in scoped):
                required = catalog_credits
            plan_modules.append(
                {
                    **record,
                    "required_credits": required,
                    "catalog_credits": catalog_credits,
                    "course_count": len(scoped),
                    "constraints": _constraint_rules(
                        record.get("rule_text", ""), scoped, major
                    ),
                }
            )
        pages = [int(course["page"]) for course in plan_courses]
        results.append(
            {
                "college": owner,
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
    return results


def repair(
    catalog_path: str | Path = "data/curriculum_catalog_v2.json",
    *,
    sources_path: str | Path = "data/sources.csv",
    chunks_path: str | Path = "data/chunks.jsonl",
    raw_dir: str | Path = "data/raw",
) -> dict[str, Any]:
    target = Path(catalog_path)
    catalog = json.loads(target.read_text(encoding="utf-8"))
    with Path(sources_path).open("r", encoding="utf-8-sig", newline="") as handle:
        source = next(row for row in csv.DictReader(handle) if row["file"].replace("\\", "/") == "school/24级培养方案.pdf")

    base._majors_from_basic_info = article_majors  # type: ignore[assignment]
    base._evidence = lambda *args, **kwargs: None  # type: ignore[assignment]
    base._following_evidence = lambda *args, **kwargs: []  # type: ignore[assignment]
    part = base._extract_book(source, str(raw_dir), str(chunks_path))
    evidence = _evidence_index(Path(chunks_path))
    courses_2024 = part["courses"]
    for course in courses_2024:
        course["evidence"] = evidence.get(
            (course["source_title"], int(course["page"]), course["code"])
        )

    courses = [course for course in catalog["courses"] if str(course["cohort"]) != "2024"]
    courses.extend(courses_2024)
    plans = [plan for plan in catalog["plans"] if str(plan["cohort"]) != "2024"]
    plans.extend(_plans(courses_2024, part["modules"]))

    # Normalize owner labels in all cohorts and use the dominant offering
    # department where the old extractor excluded the actual owner as a
    # "general" department (for example mathematics and foreign languages).
    by_plan: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for course in courses:
        by_plan[(str(course["cohort"]), course["major"])].append(course)
    for plan in plans:
        scoped = by_plan[(str(plan["cohort"]), plan["major"])]
        owner = _owner(scoped, plan["major"])
        plan["college"] = owner
        for course in scoped:
            course["college"] = owner

    catalog["plans"] = sorted(plans, key=lambda item: (str(item["cohort"]), item["major"]))
    catalog["courses"] = sorted(
        courses,
        key=lambda item: (
            str(item["cohort"]), item["major"], int(item["page"]), int(item["source_row"]), item["code"]
        ),
    )
    catalog["plan_count"] = len(catalog["plans"])
    catalog["course_count"] = len(catalog["courses"])
    coverage_by_cohort = {str(item["cohort"]): item for item in catalog["coverage"]}
    for cohort, row in coverage_by_cohort.items():
        scoped_courses = [course for course in courses if str(course["cohort"]) == cohort]
        row["plan_count"] = sum(str(plan["cohort"]) == cohort for plan in plans)
        row["course_rows"] = len(scoped_courses)
        row["hours_complete_rows"] = sum(course.get("total_hours") is not None for course in scoped_courses)
        row["plans_without_courses"] = []
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return {
        "plan_count": catalog["plan_count"],
        "course_count": catalog["course_count"],
        "coverage": catalog["coverage"],
        "cohort_2024_majors": [
            plan["major"] for plan in catalog["plans"] if str(plan["cohort"]) == "2024"
        ],
    }


def main() -> None:
    print(json.dumps(repair(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["article_majors", "repair"]
