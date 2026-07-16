"""Rebuild the full-school SQLite projection from versioned corpus files."""

from __future__ import annotations

import json

from academic_audit.database import build_database


def main() -> None:
    report = build_database(
        "data/academic_v2.sqlite3",
        catalog_path="data/curriculum_catalog_v2.json",
        sources_path="data/sources.csv",
        chunks_path="data/chunks.jsonl",
        raw_dir="data/raw",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
