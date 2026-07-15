"""Post-generation trust binding for school answers and citations."""

from __future__ import annotations

import re
from typing import Any

from contracts import (
    AnswerResult,
    CitationValidationError,
    RetrievedChunk,
    validate_answer_result,
)
from storage.metadata_db import MetadataDB


URL_RE = re.compile(r"https?://[^\s)\]}>]+", re.I)


class TrustedAnswerBinder:
    """Discard model-provided metadata and bind every citation through SQLite."""

    def __init__(self, metadata_db: MetadataDB) -> None:
        self.metadata_db = metadata_db

    def bind(
        self,
        raw_answer: dict[str, Any],
        retrieved: list[RetrievedChunk],
    ) -> AnswerResult:
        answer = validate_answer_result(raw_answer)
        if URL_RE.search(answer["answer_md"]):
            raise CitationValidationError(
                "school answer body must not contain a model-generated URL"
            )
        allowed = {chunk["chunk_id"] for chunk in retrieved}
        rebound = []
        seen_markers: set[int] = set()
        for raw in answer["citations"]:
            marker = raw.get("marker")
            chunk_id = raw.get("chunk_id")
            quote = raw.get("quote")
            if isinstance(marker, bool) or not isinstance(marker, int) or marker < 1:
                raise CitationValidationError("citation marker must be a positive integer")
            if marker in seen_markers:
                raise CitationValidationError("citation markers must be unique")
            seen_markers.add(marker)
            if not isinstance(chunk_id, str) or chunk_id not in allowed:
                raise CitationValidationError(
                    "citation chunk_id is outside this retrieval result"
                )
            if marker > len(retrieved) or retrieved[marker - 1]["chunk_id"] != chunk_id:
                raise CitationValidationError(
                    "citation marker does not match its retrieved chunk position"
                )
            stored = self.metadata_db.chunk(chunk_id)
            if stored is None:
                raise CitationValidationError(
                    "citation chunk_id is not an enabled trusted database row"
                )
            if not isinstance(quote, str) or not quote or quote not in stored.text:
                raise CitationValidationError(
                    "citation quote is not an exact database text substring"
                )
            rebound.append(
                {
                    "marker": marker,
                    "chunk_id": stored.chunk_id,
                    "doc_title": stored.doc_title,
                    "article": stored.article,
                    "quote": quote,
                    "page_url": stored.page_url,
                    "file_url": stored.file_url,
                }
            )
        if not answer["refused"] and not rebound:
            raise CitationValidationError("answered school response has no citations")
        return validate_answer_result(
            {
                "answer_md": answer["answer_md"],
                "citations": rebound,
                "refused": answer["refused"],
            }
        )


__all__ = ["TrustedAnswerBinder", "URL_RE"]
