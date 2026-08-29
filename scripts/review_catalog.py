"""Apply an explicit JSONL reviewer ledger to a catalog draft.

Only an ``approve`` or ``edit`` ledger event can mark a draft course verified.
The command emits a separate field-level diff so review edits cannot be hidden
inside a rewritten catalog file.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ingest.catalog import apply_review_ledger, load_review_ledger


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="reviewed catalog JSON output")
    parser.add_argument("--diff", type=Path, required=True, help="review diff JSON output")
    args = parser.parse_args()

    payload = json.loads(args.draft.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("catalog draft must be a JSON object")
    reviewed = apply_review_ledger(payload, load_review_ledger(args.ledger))
    _atomic_json(args.output, reviewed)
    _atomic_json(
        args.diff,
        {
            "schema_version": reviewed.get("schema_version"),
            "review_ledger": reviewed.get("review_ledger", []),
            "review_diff": reviewed.get("review_diff", []),
        },
    )
    counts = reviewed.get("counts", {})
    if not isinstance(counts, dict):
        raise SystemExit("reviewed catalog counts must be an object")
    print(
        json.dumps(
            {
                "course_count": counts.get("course_draft_count", 0),
                "verified_course_count": counts.get("verified_course_count", 0),
                "quarantine_count": counts.get("quarantine_count", 0),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
