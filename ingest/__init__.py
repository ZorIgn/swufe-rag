"""Module A source validation, parsing, chunking, and delivery."""

from ingest.chunk import build_chunks
from ingest.parse import OCRRequiredError, SidecarOCRProvider, parse_document
from ingest.pipeline import ingest_sources
from ingest.sources import SOURCE_FIELDS, load_sources

__all__ = [
    "OCRRequiredError",
    "SOURCE_FIELDS",
    "SidecarOCRProvider",
    "build_chunks",
    "ingest_sources",
    "load_sources",
    "parse_document",
]

