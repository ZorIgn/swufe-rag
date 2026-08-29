"""Frozen public contracts shared by modules A, B, C and D."""

from __future__ import annotations

import re
from typing import Any, Literal, TypedDict
from urllib.parse import urlparse

CONTRACT_VERSION = "1.2"
REVIEW_STATUSES = frozenset({"verified", "review_required", "unverified"})

# Classification is deliberately a small, explicit vocabulary.  A policy
# retriever must never infer document type from a title or from chunk text: an
# unclassified document remains ``unknown`` and is not policy-eligible.
DOC_TYPES = frozenset(
    {"policy", "notice", "guide", "curriculum", "course_catalog", "unknown"}
)
POLICY_DOCUMENT_TYPES = frozenset({"policy", "notice", "guide"})
POLICY_TOPICS = frozenset(
    {
        "academic_status",
        "assessment",
        "course_selection",
        "enrollment",
        "exemption",
        "graduation",
        "transfer",
        "other",
        # Query understanding currently emits these Chinese controlled IDs.
        # Keep them in the registry contract so source metadata and retrieval
        # scope filtering use the same exact vocabulary.
        "转专业",
        "免修",
        "学籍",
        "考试",
        "推免",
        "毕业学位",
    }
)
SOURCE_AUTHENTICITY_STATUSES = frozenset({"verified", "review_required", "unverified"})
EXTRACTION_QUALITY_STATUSES = frozenset({"verified", "review_required", "failed"})


class ContractError(ValueError):
    """Raised when a value does not satisfy a frozen public contract."""

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        line_number: int | None = None,
        chunk_id: str | None = None,
    ) -> None:
        details: list[str] = []
        if line_number is not None:
            details.append(f"line={line_number}")
        if chunk_id:
            details.append(f"chunk_id={chunk_id}")
        if field:
            details.append(f"field={field}")
        suffix = f" ({', '.join(details)})" if details else ""
        super().__init__(message + suffix)
        self.field = field
        self.line_number = line_number
        self.chunk_id = chunk_id


class KnowledgeBaseNotReadyError(RuntimeError):
    """Raised when production chunks or retrieval artifacts are unavailable."""


class GenerationUnavailableError(RuntimeError):
    """Raised when the configured LLM provider cannot generate a response."""


class CitationValidationError(ValueError):
    """Raised internally when an answer contains unsupported citations."""


class _KnowledgeChunkRequired(TypedDict):
    chunk_id: str
    text: str
    doc_title: str
    article: str
    level: str
    college: str
    cohort: str
    year: int
    status: str
    page_url: str
    file_url: str
    is_table: bool
    review_status: Literal["verified", "review_required", "unverified"]


class KnowledgeChunk(_KnowledgeChunkRequired, total=False):
    """A chunk with optional, explicit trust and taxonomy extensions.

    The optional fields preserve ingestion compatibility with historical chunk
    exports.  The database builder treats their absence fail-closed as
    ``unknown`` document type and ``review_required`` extraction quality.
    """

    doc_type: Literal["policy", "notice", "guide", "curriculum", "course_catalog", "unknown"]
    topics: list[str]
    extraction_quality: Literal["verified", "review_required", "failed"]
    extraction_warnings: list[str]
    source_sha256: str


class RetrievedChunk(KnowledgeChunk):
    score: float


class Citation(TypedDict):
    marker: int
    chunk_id: str
    doc_title: str
    article: str
    quote: str
    page_url: str
    file_url: str


class AnswerResult(TypedDict):
    answer_md: str
    citations: list[Citation]
    refused: bool


CHUNK_FIELDS = (
    "chunk_id",
    "text",
    "doc_title",
    "article",
    "level",
    "college",
    "cohort",
    "year",
    "status",
    "page_url",
    "file_url",
    "is_table",
    "review_status",
)
OPTIONAL_CHUNK_FIELDS = (
    "doc_type",
    "topics",
    "extraction_quality",
    "extraction_warnings",
    "source_sha256",
)
RETRIEVED_CHUNK_FIELDS = CHUNK_FIELDS + ("score",)
CITATION_FIELDS = (
    "marker",
    "chunk_id",
    "doc_title",
    "article",
    "quote",
    "page_url",
    "file_url",
)
ANSWER_FIELDS = ("answer_md", "citations", "refused")


def _context(raw: dict[str, Any], line_number: int | None) -> dict[str, Any]:
    return {
        "line_number": line_number,
        "chunk_id": raw.get("chunk_id") if isinstance(raw.get("chunk_id"), str) else None,
    }


def _require_exact_keys(
    raw: dict[str, Any],
    expected: tuple[str, ...],
    *,
    optional: tuple[str, ...] = (),
    line_number: int | None = None,
) -> None:
    missing = sorted(set(expected) - set(raw))
    extra = sorted(set(raw) - set(expected) - set(optional))
    context = _context(raw, line_number)
    if missing:
        raise ContractError(
            f"missing required fields: {', '.join(missing)}", **context
        )
    if extra:
        raise ContractError(f"unexpected fields: {', '.join(extra)}", **context)


def _require_nonempty_string(
    raw: dict[str, Any], field: str, *, line_number: int | None = None
) -> str:
    value = raw[field]
    if not isinstance(value, str) or not value.strip():
        raise ContractError(
            "must be a non-empty string",
            field=field,
            **_context(raw, line_number),
        )
    return value.strip()


def _require_http_url(
    raw: dict[str, Any], field: str, *, line_number: int | None = None
) -> str:
    value = _require_nonempty_string(raw, field, line_number=line_number)
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ContractError(
            "must be an absolute HTTP(S) URL",
            field=field,
            **_context(raw, line_number),
        )
    return value


def validate_chunk(raw: Any, *, line_number: int | None = None) -> KnowledgeChunk:
    """Validate and return a normalized copy of a contract-1 knowledge chunk."""

    if not isinstance(raw, dict):
        raise ContractError("knowledge chunk must be a JSON object", line_number=line_number)
    _require_exact_keys(
        raw,
        CHUNK_FIELDS,
        optional=OPTIONAL_CHUNK_FIELDS,
        line_number=line_number,
    )

    result: dict[str, Any] = {}
    for field in ("chunk_id", "text", "doc_title", "article", "college"):
        result[field] = _require_nonempty_string(raw, field, line_number=line_number)

    level = _require_nonempty_string(raw, "level", line_number=line_number)
    if level not in {"校级", "院级"}:
        raise ContractError(
            "must be one of: 校级, 院级",
            field="level",
            **_context(raw, line_number),
        )
    result["level"] = level
    if level == "校级" and result["college"] != "全校":
        raise ContractError(
            "school-level chunks must use college=全校",
            field="college",
            **_context(raw, line_number),
        )

    cohort = _require_nonempty_string(raw, "cohort", line_number=line_number)
    if cohort != "不限" and not re.fullmatch(r"\d{4}", cohort):
        raise ContractError(
            "must be a four-digit year or 不限",
            field="cohort",
            **_context(raw, line_number),
        )
    result["cohort"] = cohort

    year = raw["year"]
    if isinstance(year, bool) or not isinstance(year, int) or not 1900 <= year <= 2100:
        raise ContractError(
            "must be an integer between 1900 and 2100",
            field="year",
            **_context(raw, line_number),
        )
    result["year"] = year

    status = _require_nonempty_string(raw, "status", line_number=line_number)
    if status not in {"现行", "历史"}:
        raise ContractError(
            "must be one of: 现行, 历史",
            field="status",
            **_context(raw, line_number),
        )
    result["status"] = status
    result["page_url"] = _require_http_url(raw, "page_url", line_number=line_number)
    result["file_url"] = _require_http_url(raw, "file_url", line_number=line_number)

    is_table = raw["is_table"]
    if not isinstance(is_table, bool):
        raise ContractError(
            "must be a boolean",
            field="is_table",
            **_context(raw, line_number),
        )
    result["is_table"] = is_table

    review_status = _require_nonempty_string(raw, "review_status", line_number=line_number)
    if review_status not in REVIEW_STATUSES:
        raise ContractError(
            "must be one of: verified, review_required, unverified",
            field="review_status",
            **_context(raw, line_number),
        )
    result["review_status"] = review_status

    if "doc_type" in raw:
        doc_type = _require_nonempty_string(raw, "doc_type", line_number=line_number).lower()
        if doc_type not in DOC_TYPES:
            raise ContractError(
                "must be one of: " + ", ".join(sorted(DOC_TYPES)),
                field="doc_type",
                **_context(raw, line_number),
            )
        result["doc_type"] = doc_type
    if "topics" in raw:
        topics = raw["topics"]
        if not isinstance(topics, list) or any(
            not isinstance(topic, str) or not topic.strip() for topic in topics
        ):
            raise ContractError(
                "must be a list of non-empty controlled topic names",
                field="topics",
                **_context(raw, line_number),
            )
        normalized_topics = [topic.strip().lower() for topic in topics]
        invalid_topics = sorted(set(normalized_topics) - POLICY_TOPICS)
        if invalid_topics:
            raise ContractError(
                "contains unsupported topic names: " + ", ".join(invalid_topics),
                field="topics",
                **_context(raw, line_number),
            )
        if len(set(normalized_topics)) != len(normalized_topics):
            raise ContractError(
                "must not contain duplicate topics",
                field="topics",
                **_context(raw, line_number),
            )
        result["topics"] = normalized_topics
    if "extraction_quality" in raw:
        extraction_quality = _require_nonempty_string(
            raw, "extraction_quality", line_number=line_number
        ).lower()
        if extraction_quality not in EXTRACTION_QUALITY_STATUSES:
            raise ContractError(
                "must be one of: " + ", ".join(sorted(EXTRACTION_QUALITY_STATUSES)),
                field="extraction_quality",
                **_context(raw, line_number),
            )
        result["extraction_quality"] = extraction_quality
    if "extraction_warnings" in raw:
        warnings = raw["extraction_warnings"]
        if not isinstance(warnings, list) or any(
            not isinstance(warning, str) or not warning.strip() for warning in warnings
        ):
            raise ContractError(
                "must be a list of non-empty strings",
                field="extraction_warnings",
                **_context(raw, line_number),
            )
        result["extraction_warnings"] = [warning.strip() for warning in warnings]
    if "source_sha256" in raw:
        source_sha256 = _require_nonempty_string(
            raw, "source_sha256", line_number=line_number
        ).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
            raise ContractError(
                "must be a lowercase SHA-256 hex digest",
                field="source_sha256",
                **_context(raw, line_number),
            )
        result["source_sha256"] = source_sha256
    return result  # type: ignore[return-value]


def validate_retrieved_chunk(raw: Any) -> RetrievedChunk:
    """Validate a contract-2 result and return a normalized copy."""

    if not isinstance(raw, dict):
        raise ContractError("retrieved chunk must be a dictionary")
    _require_exact_keys(raw, RETRIEVED_CHUNK_FIELDS, optional=OPTIONAL_CHUNK_FIELDS)
    base = validate_chunk(
        {key: raw[key] for key in CHUNK_FIELDS + OPTIONAL_CHUNK_FIELDS if key in raw}
    )
    score = raw["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ContractError("must be a numeric cosine similarity", field="score")
    return {**base, "score": float(score)}


def validate_answer_result(raw: Any) -> AnswerResult:
    """Validate the exact public contract-3 response shape."""

    if not isinstance(raw, dict):
        raise ContractError("answer result must be a dictionary")
    _require_exact_keys(raw, ANSWER_FIELDS)
    if not isinstance(raw["answer_md"], str) or not raw["answer_md"].strip():
        raise ContractError("must be a non-empty string", field="answer_md")
    if not isinstance(raw["refused"], bool):
        raise ContractError("must be a boolean", field="refused")
    if not isinstance(raw["citations"], list):
        raise ContractError("must be a list", field="citations")
    for index, citation in enumerate(raw["citations"]):
        if not isinstance(citation, dict):
            raise ContractError(f"citation {index} must be a dictionary")
        _require_exact_keys(citation, CITATION_FIELDS)
    return raw  # type: ignore[return-value]
