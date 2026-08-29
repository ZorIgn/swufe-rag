"""Create a review-required catalog draft from public, registered source files.

The command records parser quality and field-level source locations.  It does
not assert that any private, unreleased, or school-specific PDF was validated;
the output remains a draft until an explicit reviewer ledger is applied.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ingest.pipeline import ingest_catalog_draft


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="catalog-draft JSON output")
    parser.add_argument(
        "--quality-ledger",
        type=Path,
        required=True,
        help="per-source/page extraction-quality JSON output",
    )
    parser.add_argument("--ocr-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--allow-critical-table-failures",
        action="store_true",
        help="write a review-required/quarantined draft instead of fail-closing on table extraction failure",
    )
    args = parser.parse_args()

    report = ingest_catalog_draft(
        args.sources,
        args.raw_dir,
        args.output,
        ocr_dir=args.ocr_dir,
        quality_ledger_path=args.quality_ledger,
        fail_on_critical_table_failure=not args.allow_critical_table_failures,
    )
    if args.report is not None:
        _atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
