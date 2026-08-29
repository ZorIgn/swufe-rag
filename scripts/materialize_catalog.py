"""Compile a reviewed catalog draft into the canonical database catalog shape.

The command is intentionally a compiler, not an extractor: program/module
membership and evidence chunk IDs must be supplied in explicit JSON inputs.
It never searches document text to fill in a missing relation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Support both the documented module form (``python -m scripts.materialize_catalog``)
# and a direct repository checkout invocation (``python scripts/materialize_catalog.py``).
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest.catalog_materialize import CatalogMaterializationError, materialize_catalog


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogMaterializationError(f"{label} is unreadable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CatalogMaterializationError(f"{label} must be a JSON object: {path}")
    return {str(key): item for key, item in value.items()}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    """Write a complete JSON file before atomically exposing it to the caller."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _quarantine_report(materialized: dict[str, object]) -> dict[str, object]:
    metadata = materialized.get("materialization")
    if not isinstance(metadata, dict):
        raise CatalogMaterializationError("materialization output lacks its audit metadata")
    return {
        "catalog_version": materialized.get("catalog_version"),
        "adapter_version": metadata.get("adapter_version"),
        "input_hashes": metadata.get("input_hashes", {}),
        "input_file_hashes": metadata.get("input_file_hashes", {}),
        "counts": metadata.get("counts", {}),
        "quarantine": metadata.get("quarantine", []),
        "upstream_quarantine": metadata.get("upstream_quarantine", []),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-draft", type=Path, required=True)
    parser.add_argument("--plan-scaffold", type=Path, required=True)
    parser.add_argument("--evidence-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="canonical curriculum_catalog.json")
    parser.add_argument(
        "--quarantine-report",
        type=Path,
        help="optional audit-only report for quarantined reviewed records",
    )
    parser.add_argument(
        "--fail-on-quarantine",
        action="store_true",
        help="do not write output when either adapter or upstream quarantine is present",
    )
    args = parser.parse_args(argv)
    if args.quarantine_report is not None and args.quarantine_report.resolve() == args.output.resolve():
        parser.error("--quarantine-report must not overwrite --output")

    try:
        reviewed_draft = _load_json_object(args.reviewed_draft, label="reviewed draft")
        plan_scaffold = _load_json_object(args.plan_scaffold, label="plan scaffold")
        evidence_mapping = _load_json_object(args.evidence_mapping, label="evidence mapping")
        materialized = materialize_catalog(
            reviewed_draft,
            plan_scaffold,
            evidence_mapping,
            fail_on_quarantine=args.fail_on_quarantine,
            input_file_hashes={
                "reviewed_draft_file_sha256": _sha256_file(args.reviewed_draft),
                "plan_scaffold_file_sha256": _sha256_file(args.plan_scaffold),
                "evidence_mapping_file_sha256": _sha256_file(args.evidence_mapping),
            },
        )
    except (CatalogMaterializationError, OSError) as exc:
        parser.error(str(exc))

    _atomic_json(args.output, materialized)
    if args.quarantine_report is not None:
        _atomic_json(args.quarantine_report, _quarantine_report(materialized))
    metadata = materialized["materialization"]
    assert isinstance(metadata, dict)
    print(
        json.dumps(
            {
                "catalog_version": materialized["catalog_version"],
                "output": str(args.output),
                "counts": metadata["counts"],
                "quarantine_report": str(args.quarantine_report)
                if args.quarantine_report is not None
                else None,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
