"""Internal, typed records used by the module A ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


ElementKind = Literal["heading", "paragraph", "table"]


@dataclass(frozen=True)
class SourceRecord:
    """One validated row from the canonical ``data/sources.csv`` registry."""

    file: str
    doc_title: str
    level: str
    college: str
    cohort: str
    year: int
    status: str
    page_url: str
    file_url: str
    collected_at: str

    def resolve(self, raw_dir: str | Path) -> Path:
        return Path(raw_dir).joinpath(*self.file.split("/"))


@dataclass(frozen=True)
class DocumentElement:
    """A paragraph, heading, or whole table in source-document order."""

    kind: ElementKind
    text: str
    page: int | None = None


@dataclass
class ParsedDocument:
    """Parser output independent of the final frozen chunk contract."""

    path: Path
    elements: list[DocumentElement]
    page_count: int | None = None
    warnings: list[str] = field(default_factory=list)
