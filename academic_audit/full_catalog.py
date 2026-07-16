"""Build a full-school relational curriculum catalog from the eight books.

The existing vector corpus is used only to recover trusted evidence chunk IDs
and page/article context.  Course facts are re-read from PDF table cells so a
row always carries cohort, major, module, hours, semester and physical page.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import date
import json
from pathlib import Path
import re
from typing import Any, Iterable

from academic_audit.catalog import (
    CATALOG_VERSION,
    MODULE_HEADING_RE,
    REQUIRED_CREDITS_RE,
    _clean,
    _compact,
    _constraint_rules,
    _course_name,
    _evidence,
    _following_evidence,
    _module_record,
    _source_hash,
    _total_credits,
    _zh_label,
)


COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,5}\d{3}\b")
PAGE_RE = re.compile(r"(?:原文件)?第(\d+)页")
BOOK_RE = re.compile(r"(?:^|/)school/(\d{2})级培养方案\.pdf$", re.I)
HEADER_MAJOR_RE = re.compile(
    r"专业\s*[:：]\s*(.{2,60}?)\s+年级\s*[:：]\s*(?:20)?\d{2}\s*级"
)
PLAN_TITLE_RE = re.compile(r"(.{2,80}?)\s*(?:20\d{2}\s*级)?本科?人才培养方案")
SEMESTER_RE = re.compile(r"(?:[1-8](?:\s*-\s*[1-8])?|S[1-4])", re.I)
GENERIC_ARTICLES = {
    "正文",
    "本科专业人才培养方案",
    "七、课程设置",
    "六、课程设置",
    "八、课程设置",
}
GENERAL_DEPARTMENTS = {
    "马克思主义学院",
    "外国语学院",
    "体育学院",
    "数学学院",
    "经济学院",
    "通识教育学院",
}


def _canonical_major(value: str) -> str:
    value = _clean(value).strip("：:，,。")
    value = re.sub(r"\s*20\d{2}\s*级.*$", "", value)
    value = re.sub(r"(?:本科)?人才培养方案$", "", value)
    if value.endswith(("专业", "班", "类")):
        return value
    return value + "专业"


def _article_root(article: str) -> str:
    return article.split(" / ", 1)[0].strip()


def _page(article: str) -> int | None:
    match = PAGE_RE.search(article)
    return int(match.group(1)) if match else None


def _load_book_chunks(chunks_path: Path, doc_title: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item["doc_title"] == doc_title:
                values.append(item)
    return values


def _page_articles(chunks: Iterable[dict[str, Any]]) -> dict[int, str]:
    grouped: dict[int, Counter[str]] = defaultdict(Counter)
    for chunk in chunks:
        page = _page(chunk["article"])
        if page is None:
            continue
        root = _article_root(chunk["article"])
        if re.fullmatch(r"第\d+页表格", root):
            continue
        weight = 4 if "培养方案" in root else 1
        grouped[page][root] += weight
    return {
        page: max(
            counts,
            key=lambda value: (
                counts[value],
                "培养方案" in value,
                value not in GENERIC_ARTICLES,
                len(value),
            ),
        )
        for page, counts in grouped.items()
        if counts
    }


def _table_pages(chunks: Iterable[dict[str, Any]]) -> set[int]:
    values: set[int] = set()
    for chunk in chunks:
        if not chunk["is_table"]:
            continue
        page = _page(chunk["article"])
        if page is not None:
            values.add(page)
    return values


def _majors_from_basic_info(article: str, chunks: Iterable[dict[str, Any]]) -> list[str]:
    text = " ".join(
        chunk["text"]
        for chunk in chunks
        if _article_root(chunk["article"]) == article
        and "专业类基本信息" in chunk["article"]
    )
    block_match = re.search(r"专业代码\s*[:：](.+?)(?:标准学制|计划学制)", text)
    block = block_match.group(1) if block_match else ""
    names: list[str] = []
    matches = list(re.finditer(r"\d{6}[A-Z]?\s*[-—]?", block))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(block)
        value = block[match.end() : end]
        for part in re.split(r"[/、，,]", value):
            cleaned = _clean(part)
            if cleaned and re.search(r"[\u4e00-\u9fff]", cleaned):
                names.append(_canonical_major(cleaned))
    if not names:
        title = re.sub(r"专业人才培养方案$", "", article)
        title = re.sub(r"人才培养方案$", "", title)
        if "类" not in title or "(" in title:
            names.append(_canonical_major(title))
    return list(dict.fromkeys(names))


def _article_major_map(chunks: list[dict[str, Any]]) -> dict[str, list[str]]:
    articles = {
        _article_root(chunk["article"])
        for chunk in chunks
        if "培养方案" in _article_root(chunk["article"])
    }
    return {
        article: _majors_from_basic_info(article, chunks)
        for article in articles
    }


def _float(value: str) -> float | None:
    value = value.replace(" ", "")
    if not re.fullmatch(r"\d+(?:\.\d+)?", value):
        return None
    return float(value)


def _parse_course(row: list[Any]) -> dict[str, Any] | None:
    cells = [_clean(cell) for cell in row]
    code_index = -1
    code = ""
    for index, cell in enumerate(cells):
        match = COURSE_CODE_RE.search(cell.upper())
        if match:
            code_index = index
            code = match.group(0)
            break
    if code_index < 0 or code_index + 1 >= len(cells):
        return None
    name = _course_name(cells[code_index + 1])
    if not name or "课程名称" in name:
        return None
    nature_index = next(
        (index for index, cell in enumerate(cells) if re.search(r"必修|限选|选修", cell)),
        -1,
    )
    if nature_index < 0:
        return None
    numbers = [
        value
        for value in (_float(cell) for cell in cells[code_index + 2 : nature_index])
        if value is not None
    ]
    if not numbers or not 0 < numbers[0] <= 30:
        return None
    padded = (numbers + [None] * 5)[:5]
    semester = "未标注"
    for cell in reversed(cells):
        compact = cell.replace(" ", "").upper()
        if re.fullmatch(SEMESTER_RE, compact):
            semester = compact
            break
    department = next(
        (_zh_label(cell) for cell in cells if "学院" in cell and "开课学院" not in cell),
        "",
    )
    return {
        "code": code,
        "name": name,
        "credits": padded[0],
        "weekly_hours": padded[1],
        "total_hours": padded[2],
        "teaching_hours": padded[3],
        "practice_hours": padded[4],
        "nature": _zh_label(cells[nature_index]),
        "semester": semester,
        "department": department,
    }


def _scoped_majors(module: str, plan_majors: list[str]) -> list[str]:
    compact_module = _compact(module)
    matches = [
        major
        for major in plan_majors
        if _compact(major.removesuffix("专业")) in compact_module
    ]
    return matches or plan_majors


def _college(courses: list[dict[str, Any]], article: str) -> str:
    match = re.search(r"[（(]([^）)]*学院)[）)]", article)
    if match:
        return match.group(1)
    counts = Counter(
        course["department"]
        for course in courses
        if course.get("department")
        and course["department"] not in GENERAL_DEPARTMENTS
    )
    return counts.most_common(1)[0][0] if counts else "全校"


def _extract_book(
    source: dict[str, str],
    raw_dir: str,
    chunks_path: str,
) -> dict[str, Any]:
    import pdfplumber

    path = Path(raw_dir) / source["file"]
    chunks = _load_book_chunks(Path(chunks_path), source["doc_title"])
    articles_by_page = _page_articles(chunks)
    table_pages = _table_pages(chunks)
    majors_by_article = _article_major_map(chunks)
    cohort = source["cohort"]
    courses: list[dict[str, Any]] = []
    modules: dict[tuple[str, str, str], dict[str, Any]] = {}
    current_article = ""
    current_majors: list[str] = []
    current_module = ""

    with pdfplumber.open(path) as pdf:
        for page_number in sorted(table_pages):
            if page_number < 1 or page_number > len(pdf.pages):
                continue
            page = pdf.pages[page_number - 1]
            text = page.extract_text() or ""
            header = HEADER_MAJOR_RE.search(text)
            article = articles_by_page.get(page_number, current_article)
            if header:
                detected = _canonical_major(header.group(1))
                if current_majors != [detected]:
                    current_module = ""
                current_majors = [detected]
            elif article and article != current_article:
                article_majors = majors_by_article.get(article, [])
                if article_majors:
                    current_majors = article_majors
                    current_module = ""
            if article:
                current_article = article
            if not current_majors:
                continue
            for table in page.extract_tables() or []:
                for row_index, row in enumerate(table, start=1):
                    cells = [_clean(cell) for cell in row]
                    joined = " | ".join(cells)
                    first = next((cell for cell in cells if cell), "")
                    parsed = _parse_course(row)
                    if parsed is None and MODULE_HEADING_RE.search(first):
                        current_module = _zh_label(first)
                        for major in _scoped_majors(current_module, current_majors):
                            _module_record(
                                modules,
                                cohort=cohort,
                                major=major,
                                name=current_module,
                                source=source,
                            )
                        continue
                    if not current_module:
                        continue
                    active = _scoped_majors(current_module, current_majors)
                    required_match = REQUIRED_CREDITS_RE.search(joined)
                    if required_match and parsed is None:
                        for major in active:
                            module = _module_record(
                                modules,
                                cohort=cohort,
                                major=major,
                                name=current_module,
                                source=source,
                            )
                            module["required_credits"] = float(required_match.group(1))
                            module["rule_text"] = _zh_label(joined)
                            evidence = _evidence(
                                chunks, source, page=page_number, needle=required_match.group(0)
                            ) or _evidence(chunks, source, page=page_number, needle="学分")
                            module["evidence"] = evidence
                            module["supporting_evidence"] = _following_evidence(chunks, evidence)
                        continue
                    total = _total_credits(row)
                    if total is not None and parsed is None:
                        for major in active:
                            _module_record(
                                modules,
                                cohort=cohort,
                                major=major,
                                name=current_module,
                                source=source,
                            )["listed_credits"] = total
                        continue
                    if parsed is None:
                        continue
                    evidence = _evidence(chunks, source, page=page_number, needle=parsed["code"])
                    for major in active:
                        courses.append(
                            {
                                **parsed,
                                "college": "全校",
                                "cohort": cohort,
                                "major": major,
                                "module": current_module,
                                "source_title": source["doc_title"],
                                "page": page_number,
                                "source_row": row_index,
                                "evidence": evidence,
                                "plan_article": current_article,
                            }
                        )

    unique: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for course in courses:
        key = (
            course["cohort"],
            course["major"],
            course["module"],
            course["code"],
            course["semester"],
        )
        unique.setdefault(key, course)
    courses = list(unique.values())
    by_major: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for course in courses:
        by_major[course["major"]].append(course)
    for major, values in by_major.items():
        article = Counter(value["plan_article"] for value in values).most_common(1)[0][0]
        college = _college(values, article)
        for value in values:
            value["college"] = college
    return {
        "cohort": cohort,
        "source": source,
        "courses": courses,
        "modules": list(modules.values()),
        "physical_pages": len(pdf.pages),
        "table_pages": len(table_pages),
    }


def build_full_catalog(
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
        if BOOK_RE.search(source["file"].replace("\\", "/"))
    ]
    parts: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(books)))) as pool:
        futures = {
            pool.submit(_extract_book, source, str(raw_root), str(chunks_file)): source
            for source in books
        }
        for future in as_completed(futures):
            parts.append(future.result())
    courses = [course for part in parts for course in part["courses"]]
    modules_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for part in parts:
        for module in part["modules"]:
            key = (part["cohort"], module.get("source_title", ""), module["name"])
            modules_by_key.setdefault(key, module)
    plans: list[dict[str, Any]] = []
    for (cohort, major), plan_courses in sorted(
        {
            (course["cohort"], course["major"]): [
                value
                for value in courses
                if value["cohort"] == course["cohort"] and value["major"] == course["major"]
            ]
            for course in courses
        }.items()
    ):
        module_names = sorted({course["module"] for course in plan_courses})
        plan_modules: list[dict[str, Any]] = []
        for module_name in module_names:
            module_courses = [value for value in plan_courses if value["module"] == module_name]
            catalog_credits = round(sum(float(value["credits"] or 0) for value in module_courses), 2)
            module = next(
                (
                    value
                    for part in parts
                    for value in part["modules"]
                    if value["name"] == module_name
                    and any(
                        course["cohort"] == cohort
                        and course["major"] == major
                        and course["module"] == module_name
                        for course in part["courses"]
                    )
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
            required = module.get("required_credits") or module.get("listed_credits")
            if required is None and module_courses and all("必修" in value["nature"] for value in module_courses):
                required = catalog_credits
            plan_modules.append(
                {
                    **module,
                    "required_credits": required,
                    "catalog_credits": catalog_credits,
                    "course_count": len(module_courses),
                    "constraints": _constraint_rules(module.get("rule_text", ""), module_courses, major),
                }
            )
        pages = [int(value["page"]) for value in plan_courses]
        plans.append(
            {
                "college": Counter(value["college"] for value in plan_courses).most_common(1)[0][0],
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
    coverage = []
    for cohort in sorted({part["cohort"] for part in parts}):
        scoped_plans = [plan for plan in plans if plan["cohort"] == cohort]
        scoped_courses = [course for course in courses if course["cohort"] == cohort]
        part = next(value for value in parts if value["cohort"] == cohort)
        coverage.append(
            {
                "cohort": cohort,
                "physical_pages": part["physical_pages"],
                "table_pages": part["table_pages"],
                "plan_count": len(scoped_plans),
                "course_rows": len(scoped_courses),
                "plans_without_courses": [],
            }
        )
    source_paths = [sources_file, chunks_file, *[raw_root / source["file"] for source in books]]
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
            key=lambda value: (
                value["cohort"], value["major"], value["page"], value["source_row"], value["code"]
            ),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/curriculum_catalog_v2.json")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = build_full_catalog(workers=args.workers)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps({key: result[key] for key in ("plan_count", "course_count", "coverage")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

