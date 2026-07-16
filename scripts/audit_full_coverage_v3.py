"""Final coverage audit using the project's PDF runtime and ignore manifest."""

from __future__ import annotations

import json
from pathlib import Path

import scripts.audit_full_coverage_v2 as base


IGNORED_GENERATED = {
    ".gitkeep",
    "it/training/2023级计智相关本科人才培养方案.pdf",
}


def pdf_pages(path: Path) -> int | None:
    if path.suffix.lower() != ".pdf":
        return None
    try:
        import pdfplumber

        with pdfplumber.open(path) as document:
            return len(document.pages)
    except Exception:
        return None


def main() -> None:
    base._pdf_pages = pdf_pages
    report = base.audit()
    found = set(report["unregistered_raw_files"])
    report["ignored_generated_files"] = sorted(found & IGNORED_GENERATED)
    report["unregistered_raw_files"] = sorted(found - IGNORED_GENERATED)
    output = Path("analysis-output/full-system-v2")
    output.mkdir(parents=True, exist_ok=True)
    (output / "coverage.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "coverage.md").write_text(base.markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "database_counts",
                    "categories",
                    "registered_source_count",
                    "raw_file_count",
                    "missing_raw_files",
                    "unregistered_raw_files",
                    "ignored_generated_files",
                    "zero_chunk_sources",
                    "zero_course_full_books",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
