"""Strict validation for module A's canonical source registry."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path, PurePosixPath
import re
from urllib.parse import urlparse

from contracts import ContractError
from ingest.models import SourceRecord


SOURCE_FIELDS = (
    "file",
    "doc_title",
    "level",
    "college",
    "cohort",
    "year",
    "status",
    "page_url",
    "file_url",
    "collected_at",
)
SUPPORTED_SOURCE_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}


def _error(message: str, *, line: int, field: str | None = None) -> ContractError:
    return ContractError(message, line_number=line, field=field)


def _required(row: dict[str, str], field: str, line: int) -> str:
    value = row.get(field, "").strip()
    if not value:
        raise _error("must be a non-empty string", line=line, field=field)
    return value


def _validate_file(value: str, line: int) -> str:
    if "\\" in value or re.match(r"^[A-Za-z]:", value):
        raise _error(
            "must be a portable POSIX path relative to data/raw",
            line=line,
            field="file",
        )
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise _error(
            "must stay within data/raw and must not contain traversal segments",
            line=line,
            field="file",
        )
    if path.parts[0] in {"raw", "data"}:
        raise _error(
            "must be relative to data/raw (for example school/policy.pdf)",
            line=line,
            field="file",
        )
    if path.suffix.lower() not in SUPPORTED_SOURCE_SUFFIXES:
        raise _error(
            "unsupported format; convert DOC/ZIP inputs before registration",
            line=line,
            field="file",
        )
    return path.as_posix()


def _validate_url(value: str, field: str, line: int) -> str:
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or not host:
        raise _error("must be an absolute HTTP(S) URL", line=line, field=field)
    if host != "swufe.edu.cn" and not host.endswith(".swufe.edu.cn"):
        raise _error(
            "must point to the official swufe.edu.cn domain",
            line=line,
            field=field,
        )
    return value


def validate_source_row(row: dict[str, str], *, line_number: int) -> SourceRecord:
    file = _validate_file(_required(row, "file", line_number), line_number)
    doc_title = _required(row, "doc_title", line_number)
    level = _required(row, "level", line_number)
    if level not in {"校级", "院级"}:
        raise _error("must be one of: 校级, 院级", line=line_number, field="level")
    college = _required(row, "college", line_number)
    if level == "校级" and college != "全校":
        raise _error(
            "school-level sources must use college=全校",
            line=line_number,
            field="college",
        )
    if level == "院级" and college == "全校":
        raise _error(
            "college-level sources must name a specific college",
            line=line_number,
            field="college",
        )
    cohort = _required(row, "cohort", line_number)
    if cohort != "不限" and not re.fullmatch(r"\d{4}", cohort):
        raise _error(
            "must be a four-digit year or 不限",
            line=line_number,
            field="cohort",
        )
    raw_year = _required(row, "year", line_number)
    try:
        year = int(raw_year)
    except ValueError as exc:
        raise _error("must be a four-digit integer", line=line_number, field="year") from exc
    if year < 1900 or year > 2100:
        raise _error(
            "must be between 1900 and 2100", line=line_number, field="year"
        )
    status = _required(row, "status", line_number)
    if status not in {"现行", "历史"}:
        raise _error("must be one of: 现行, 历史", line=line_number, field="status")
    page_url = _validate_url(
        _required(row, "page_url", line_number), "page_url", line_number
    )
    file_url = _validate_url(
        _required(row, "file_url", line_number), "file_url", line_number
    )
    collected_at = _required(row, "collected_at", line_number)
    try:
        date.fromisoformat(collected_at)
    except ValueError as exc:
        raise _error(
            "must use ISO date format YYYY-MM-DD",
            line=line_number,
            field="collected_at",
        ) from exc
    return SourceRecord(
        file=file,
        doc_title=doc_title,
        level=level,
        college=college,
        cohort=cohort,
        year=year,
        status=status,
        page_url=page_url,
        file_url=file_url,
        collected_at=collected_at,
    )


def load_sources(
    path: str | Path,
    *,
    raw_dir: str | Path | None = None,
    require_files: bool = True,
) -> list[SourceRecord]:
    source_path = Path(path)
    if not source_path.is_file():
        raise ContractError(f"source registry does not exist: {source_path}")
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        actual = tuple(reader.fieldnames or ())
        if actual != SOURCE_FIELDS:
            missing = sorted(set(SOURCE_FIELDS) - set(actual))
            extra = sorted(set(actual) - set(SOURCE_FIELDS))
            details: list[str] = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if extra:
                details.append("unexpected: " + ", ".join(extra))
            raise ContractError(
                "source registry header must exactly match the canonical schema"
                + (" (" + "; ".join(details) + ")" if details else "")
            )
        records: list[SourceRecord] = []
        seen_files: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            record = validate_source_row(row, line_number=line_number)
            if record.file in seen_files:
                raise _error(
                    "duplicate source file", line=line_number, field="file"
                )
            seen_files.add(record.file)
            if require_files:
                if raw_dir is None:
                    raise ValueError("raw_dir is required when require_files=True")
                resolved = record.resolve(raw_dir)
                if not resolved.is_file():
                    raise _error(
                        f"registered source does not exist: {resolved}",
                        line=line_number,
                        field="file",
                    )
            records.append(record)
    if not records:
        raise ContractError("source registry is empty")
    return records
