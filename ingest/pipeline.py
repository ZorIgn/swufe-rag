"""Module A orchestration and atomic ``chunks.jsonl`` delivery."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ingest.chunk import build_chunks
from ingest.parse import SidecarOCRProvider, parse_document
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
            "ocr_used" in item["warnings"] for item in source_reports
        ),
        "sources": source_reports,
    }
    if report_path is not None:
        _atomic_json(Path(report_path), report)
    return report
