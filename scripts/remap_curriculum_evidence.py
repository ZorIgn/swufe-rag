"""Remap selected curriculum records from quarantined books to scoped sources.

Whole-school curriculum books remain quarantined.  This utility resolves the
same structured rows against already-ingested, program/college-scoped source
documents and writes a new catalog plus an exact chunk-review ledger.  It fails
closed if any selected course or module cannot be tied to a matching chunk.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def _mapping(value: str) -> tuple[tuple[str, str], str]:
    scope, separator, title = value.partition("=")
    cohort, scope_separator, program = scope.partition(":")
    if not separator or not scope_separator or not cohort.isdigit() or not program or not title:
        raise argparse.ArgumentTypeError(
            "mapping must use COHORT:PROGRAM=SCOPED_SOURCE_TITLE"
        )
    return (cohort, program.strip()), title.strip()


def _compact(value: object) -> str:
    text = str(value or "").replace("Ⅰ", "I").replace("Ⅱ", "II").replace("Ⅲ", "III").replace("Ⅳ", "IV")
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).lower()


def _number_present(text: str, value: object) -> bool:
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return True
    forms = {f"{number:g}", f"{number:.1f}"}
    return any(re.search(rf"(?<!\d){re.escape(form)}(?!\d)", text) for form in forms)


def _evidence(chunk: dict[str, object]) -> dict[str, object]:
    return {
        "chunk_id": chunk["chunk_id"],
        "doc_title": chunk.get("doc_title"),
        "article": chunk.get("article"),
        "quote": str(chunk.get("text") or "")[:500],
        "page_url": chunk.get("page_url"),
        "file_url": chunk.get("file_url"),
    }


def _course_chunk(
    course: dict[str, object], candidates: list[dict[str, object]]
) -> dict[str, object] | None:
    code = str(course.get("code") or "").strip()
    name = _compact(course.get("name"))
    scored: list[tuple[int, int, dict[str, object]]] = []
    for position, chunk in enumerate(candidates):
        text = str(chunk.get("text") or "")
        compact = _compact(text)
        if not code or code.lower() not in text.lower():
            continue
        name_present = bool(name and name in compact)
        credit_present = _number_present(text, course.get("credits"))
        if not name_present or not credit_present:
            continue
        semester = str(course.get("semester") or "").strip()
        semester_present = bool(semester and semester in text)
        score = 10 + 4 * int(not bool(chunk.get("is_table"))) + int(semester_present)
        scored.append((score, -position, chunk))
    return max(scored, key=lambda item: (item[0], item[1]))[2] if scored else None


def _module_chunk(
    module: dict[str, object], candidates: list[dict[str, object]]
) -> tuple[dict[str, object], float] | None:
    name = _compact(module.get("name"))
    # Drop outline numbering while retaining the business-specific phrase.
    key = re.sub(r"^[一二三四五六七八九十0-9]+", "", name)
    scored: list[tuple[int, int, dict[str, object], float]] = []
    for position, chunk in enumerate(candidates):
        text = str(chunk.get("text") or "")
        compact = _compact(f"{chunk.get('article') or ''} {text}")
        if not key or key not in compact:
            continue
        window_text = "\n".join(
            str(candidate.get("text") or "")
            for candidate in candidates[position : position + 30]
        )
        totals = re.findall(
            r"(?:合计|In\s*Total)[^0-9]{0,80}([0-9]+(?:\.[0-9]+)?)",
            window_text,
            re.I,
        )
        if not totals:
            continue
        total = float(totals[0])
        rule_text = str(module.get("rule_text") or "")
        rule_supported = not rule_text or _number_present(window_text, module.get("required_credits"))
        score = 8 + 4 * int(rule_supported) + 3 * int(bool(chunk.get("is_table")))
        scored.append((score, -position, chunk, total))
    if not scored:
        return None
    best = max(scored, key=lambda item: (item[0], item[1]))
    return (best[2], best[3]) if best[0] >= 12 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path("data/curriculum_catalog.json"))
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.jsonl"))
    parser.add_argument("--mapping", action="append", type=_mapping, required=True)
    parser.add_argument("--output-catalog", type=Path, required=True)
    parser.add_argument("--output-evidence-review", type=Path, required=True)
    args = parser.parse_args()

    mappings = dict(args.mapping)
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    chunks = [
        json.loads(line) for line in args.chunks.read_text(encoding="utf-8").splitlines() if line
    ]
    by_title: dict[str, list[dict[str, object]]] = {}
    for chunk in chunks:
        by_title.setdefault(str(chunk.get("doc_title") or ""), []).append(chunk)

    failures: list[str] = []
    reviewed_ids: set[str] = set()
    remapped_courses = 0
    remapped_modules = 0
    for course in catalog.get("courses", []):
        scope = (str(course.get("cohort")), str(course.get("major")))
        target_title = mappings.get(scope)
        if target_title is None:
            continue
        chunk = _course_chunk(course, by_title.get(target_title, []))
        if chunk is None:
            failures.append(f"course:{scope[0]}:{scope[1]}:{course.get('code')}:{course.get('name')}")
            continue
        course["source_title"] = target_title
        course["evidence"] = _evidence(chunk)
        reviewed_ids.add(str(chunk["chunk_id"]))
        remapped_courses += 1

    for plan in catalog.get("plans", []):
        scope = (str(plan.get("cohort")), str(plan.get("major")))
        target_title = mappings.get(scope)
        if target_title is None:
            continue
        plan["source_title"] = target_title
        for module in plan.get("modules", []):
            resolution = _module_chunk(module, by_title.get(target_title, []))
            if resolution is None:
                failures.append(f"module:{scope[0]}:{scope[1]}:{module.get('name')}")
                continue
            chunk, listed_credits = resolution
            if not str(module.get("rule_text") or "").strip():
                module["listed_credits"] = listed_credits
                module["required_credits"] = listed_credits
            module["evidence"] = _evidence(chunk)
            reviewed_ids.add(str(chunk["chunk_id"]))
            remapped_modules += 1

    report = {
        "scope_count": len(mappings),
        "remapped_course_count": remapped_courses,
        "remapped_module_count": remapped_modules,
        "reviewed_chunk_count": len(reviewed_ids),
        "failure_count": len(failures),
        "failures": failures[:100],
    }
    print(json.dumps(report, ensure_ascii=False))
    if failures:
        raise SystemExit("scoped curriculum evidence remap failed; outputs were not written")

    args.output_catalog.parent.mkdir(parents=True, exist_ok=True)
    args.output_catalog.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_evidence_review.parent.mkdir(parents=True, exist_ok=True)
    with args.output_evidence_review.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("chunk_id", "decision", "scope", "reviewer", "method", "reviewed_at"),
        )
        writer.writeheader()
        for chunk_id in sorted(reviewed_ids):
            writer.writerow(
                {
                    "chunk_id": chunk_id,
                    "decision": "verified",
                    "scope": " | ".join(
                        f"{cohort}:{program}" for cohort, program in sorted(mappings)
                    ),
                    "reviewer": "scoped-curriculum-remap",
                    "method": "code_name_credit_and_module_total_crosscheck",
                    "reviewed_at": "generated-during-release-audit",
                }
            )


if __name__ == "__main__":
    main()
