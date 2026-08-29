"""Module A orchestration and atomic ``chunks.jsonl`` delivery."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from ingest.catalog import combine_catalog_drafts, extract_catalog_draft
from ingest.chunk import build_chunks
from ingest.parse import SidecarOCRProvider, extraction_quality_ledger, parse_document
from ingest.sources import load_sources


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def ingest_sources(
    sources_path: str | Path,
    raw_dir: str | Path,
    output_path: str | Path,
    *,
    ocr_dir: str | Path | None = None,
    report_path: str | Path | None = None,
    chunk_max_len: int = 500,
) -> dict[str, Any]:
    records = load_sources(sources_path, raw_dir=raw_dir, require_files=True)
    ocr_provider = SidecarOCRProvider(ocr_dir) if ocr_dir is not None else None
    chunks = []
    source_reports: list[dict[str, Any]] = []
    for record in records:
        parsed = parse_document(record.resolve(raw_dir), ocr_provider=ocr_provider)
        source_chunks = build_chunks(parsed, record, chunk_max_len=chunk_max_len)
        quality = extraction_quality_ledger(parsed)
        chunks.extend(source_chunks)
        source_reports.append(
            {
                "file": record.file,
                "doc_title": record.doc_title,
                "pages": parsed.page_count,
                "elements": len(parsed.elements),
                "chunks": len(source_chunks),
                "table_chunks": sum(chunk["is_table"] for chunk in source_chunks),
                "warnings": parsed.warnings,
                "quality_ledger": quality,
                "quality_failed_page_count": sum(item["status"] == "failed" for item in quality),
            }
        )

    ids = [chunk["chunk_id"] for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("generated chunk_id values are not globally unique")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    os.replace(temporary, destination)

    report = {
        "source_count": len(records),
        "chunk_count": len(chunks),
        "table_chunk_count": sum(chunk["is_table"] for chunk in chunks),
        "ocr_source_count": sum(
            any(
                warning == "ocr_used" or warning.startswith("quality:ocr_used:")
                for warning in item["warnings"]
            )
            for item in source_reports
        ),
        "quality_failed_page_count": sum(
            int(item["quality_failed_page_count"]) for item in source_reports
        ),
        "sources": source_reports,
    }
    if report_path is not None:
        _atomic_json(Path(report_path), report)
    return report


def ingest_catalog_draft(
    sources_path: str | Path,
    raw_dir: str | Path,
    output_path: str | Path,
    *,
    ocr_dir: str | Path | None = None,
    quality_ledger_path: str | Path | None = None,
    fail_on_critical_table_failure: bool = True,
) -> dict[str, Any]:
    """Produce a review-required public catalog draft from registered sources.

    This is intentionally separate from ``ingest_sources``: the longstanding
    PDF-to-chunks path remains useful for policy prose, while catalog drafts
    require table completeness and a later reviewer decision before they can
    become structured facts.
    """

    records = load_sources(sources_path, raw_dir=raw_dir, require_files=True)
    ocr_provider = SidecarOCRProvider(ocr_dir) if ocr_dir is not None else None
    drafts: list[dict[str, object]] = []
    source_reports: list[dict[str, object]] = []
    for record in records:
        parsed = parse_document(record.resolve(raw_dir), ocr_provider=ocr_provider)
        quality = extraction_quality_ledger(parsed)
        draft = extract_catalog_draft(
            parsed,
            source=record,
            quality_ledger=quality,
            fail_on_critical_table_failure=fail_on_critical_table_failure,
        )
        drafts.append(draft)
        source_reports.append(
            {
                "file": record.file,
                "doc_title": record.doc_title,
                "pages": parsed.page_count,
                "quality_ledger": quality,
                "course_draft_count": cast(Mapping[str, object], draft["counts"])["course_draft_count"],
                "quarantine_count": cast(Mapping[str, object], draft["counts"])["quarantine_count"],
            }
        )
    combined = combine_catalog_drafts(drafts)
    _atomic_json(Path(output_path), combined)
    if quality_ledger_path is not None:
        _atomic_json(Path(quality_ledger_path), combined["quality_ledger"])
    return {
        "source_count": len(records),
        "course_draft_count": cast(Mapping[str, object], combined["counts"])["course_draft_count"],
        "quarantine_count": cast(Mapping[str, object], combined["counts"])["quarantine_count"],
        "verified_course_count": 0,
        "sources": source_reports,
    }
