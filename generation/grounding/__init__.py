"""Strict sentence-level grounding, citation repair, and exact quote mapping."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from contracts import Citation, CitationValidationError, RetrievedChunk
from generation.prompts import REFUSAL_TEXT
from retrieval.query import lexical_tokens, normalize_query


MARKER_RE = re.compile(r"\[(\d+)\]")
GROUPED_MARKER_RE = re.compile(r"\[\s*(\d+(?:\s*[,，、]\s*\d+)+)\s*\]")
FULLWIDTH_MARKER_RE = re.compile(r"【\s*(\d+)\s*】")
SOURCE_MARKER_RE = re.compile(
    r"\[\s*(?:来源|source)\s*[:：]?\s*(\d+)\s*\]", re.I
)
NUMBER_RE = re.compile(
    r"(?<![A-Za-z])\d+(?:\.\d+)?%?|[A-Za-z]{2,}\d{2,4}", re.I
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s*|\n+")
PASSAGE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])|\n+")


def normalize_citation_formats(answer: str) -> str:
    normalized = unicodedata.normalize("NFKC", answer).strip()
    normalized = FULLWIDTH_MARKER_RE.sub(
        lambda match: f"[{int(match.group(1))}]", normalized
    )
    normalized = SOURCE_MARKER_RE.sub(
        lambda match: f"[{int(match.group(1))}]", normalized
    )

    def expand_group(match: re.Match[str]) -> str:
        values = re.split(r"\s*[,，、]\s*", match.group(1))
        return "".join(f"[{int(value)}]" for value in values)

    normalized = GROUPED_MARKER_RE.sub(expand_group, normalized)
    normalized = re.sub(r"(\[\d+\])(?:\s*\1)+", r"\1", normalized)
    return normalized


def _sentences(answer: str) -> list[str]:
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(answer) if part.strip()]


def _passages_with_offsets(text: str) -> list[tuple[str, int, int]]:
    passages: list[tuple[str, int, int]] = []
    start = 0
    for match in PASSAGE_SPLIT_RE.finditer(text):
        end = match.end()
        raw = text[start:end]
        stripped = raw.strip()
        if stripped:
            offset = raw.find(stripped)
            passages.append((stripped, start + offset, start + offset + len(stripped)))
        start = end
    raw = text[start:]
    stripped = raw.strip()
    if stripped:
        offset = raw.find(stripped)
        passages.append((stripped, start + offset, start + offset + len(stripped)))
    return passages or [(text, 0, len(text))]


def _plain_refusal(text: str) -> bool:
    left = unicodedata.normalize("NFKC", text).strip().rstrip("。.!！")
    right = unicodedata.normalize("NFKC", REFUSAL_TEXT).strip().rstrip("。.!！")
    return left == right


def _exempt(sentence: str) -> bool:
    cleaned = MARKER_RE.sub("", sentence).strip(" \t-*#>。.!！?？;；:：")
    return (
        not cleaned
        or _plain_refusal(cleaned)
        or cleaned.startswith(("建议咨询", "仅供参考", "以下条款仅供参考"))
        or cleaned.endswith(("如下", "如下所示"))
    )


def _citation_at_sentence_end(sentence: str) -> bool:
    stripped = re.sub(r"[。！？!?；;]+$", "", sentence.strip()).rstrip()
    return bool(re.search(r"(?:\[\d+\]){1,4}$", stripped))


def _best_support(claim: str, source: str) -> tuple[str, int, float, int, int]:
    claim_tokens = lexical_tokens(MARKER_RE.sub("", claim))
    best = (source, 0, 0.0, 0, len(source))
    for passage, start, end in _passages_with_offsets(source):
        passage_tokens = lexical_tokens(passage)
        overlap = len(claim_tokens & passage_tokens)
        coverage = overlap / len(claim_tokens) if claim_tokens else 0.0
        candidate = (passage, overlap, coverage, start, end)
        if (candidate[1], candidate[2], -len(candidate[0])) > (
            best[1],
            best[2],
            -len(best[0]),
        ):
            best = candidate
    return best


def _supporting_quote(claims: list[str], source: str, max_chars: int) -> str:
    spans = [_best_support(claim, source) for claim in claims]
    supported = [span for span in spans if span[1] > 0]
    if not supported:
        return source[:max_chars]
    start = min(span[3] for span in supported)
    end = max(span[4] for span in supported)
    if end - start <= max_chars:
        return source[start:end]
    best = max(supported, key=lambda span: (span[1], span[2], -len(span[0])))
    return best[0][:max_chars]


@dataclass(frozen=True)
class GroundingResult:
    answer: str
    citations: list[Citation]


class StrictGroundingValidator:
    def __init__(
        self,
        *,
        min_overlap_tokens: int = 2,
        min_claim_coverage: float = 0.16,
        max_citations_per_sentence: int = 4,
        max_quote_chars: int = 300,
    ) -> None:
        self.min_overlap_tokens = min_overlap_tokens
        self.min_claim_coverage = min_claim_coverage
        self.max_citations_per_sentence = max_citations_per_sentence
        self.max_quote_chars = max_quote_chars

    def _validate_sentence(
        self, sentence: str, chunks: list[RetrievedChunk]
    ) -> list[int]:
        if _exempt(sentence):
            return []
        markers = [int(value) for value in MARKER_RE.findall(sentence)]
        if not markers:
            raise CitationValidationError(f"factual sentence has no citation: {sentence}")
        if len(markers) > self.max_citations_per_sentence:
            raise CitationValidationError("a sentence cites more than four sources")
        if not _citation_at_sentence_end(sentence):
            raise CitationValidationError("citations must appear at the end of the sentence")
        if any(marker < 1 or marker > len(chunks) for marker in markers):
            raise CitationValidationError("citation marker is out of range")

        unique_markers = list(dict.fromkeys(markers))
        claim = MARKER_RE.sub("", sentence)
        claim_numbers = set(NUMBER_RE.findall(normalize_query(claim)))
        sources = [chunks[marker - 1]["text"] for marker in unique_markers]
        combined = normalize_query("\n".join(sources))
        missing = sorted(number for number in claim_numbers if number not in combined)
        if missing:
            raise CitationValidationError(
                f"numbers or codes are absent from cited source: {', '.join(missing)}"
            )

        claim_tokens = lexical_tokens(claim)
        union_tokens = lexical_tokens(combined)
        overlap = len(claim_tokens & union_tokens)
        coverage = overlap / len(claim_tokens) if claim_tokens else 0.0
        if overlap < self.min_overlap_tokens or coverage < self.min_claim_coverage:
            raise CitationValidationError("cited sources do not support the sentence")
        for marker, source in zip(unique_markers, sources):
            _, marker_overlap, _, _, _ = _best_support(claim, source)
            if marker_overlap == 0:
                raise CitationValidationError(
                    f"citation [{marker}] is unrelated to the sentence"
                )
        return unique_markers

    def validate(self, answer: str, chunks: list[RetrievedChunk]) -> GroundingResult:
        normalized = normalize_citation_formats(answer)
        if not normalized:
            raise CitationValidationError("answer is empty")
        if _plain_refusal(normalized):
            return GroundingResult(REFUSAL_TEXT, [])

        sentences = _sentences(normalized)
        marker_order: list[int] = []
        claims_by_marker: dict[int, list[str]] = {}
        for sentence in sentences:
            markers = self._validate_sentence(sentence, chunks)
            for marker in markers:
                if marker not in marker_order:
                    marker_order.append(marker)
                claims_by_marker.setdefault(marker, []).append(sentence)
        if not marker_order:
            raise CitationValidationError("answer contains no grounded citations")

        citations: list[Citation] = []
        for marker in marker_order:
            chunk = chunks[marker - 1]
            quote = _supporting_quote(
                claims_by_marker[marker], chunk["text"], self.max_quote_chars
            )
            if quote not in chunk["text"]:
                raise CitationValidationError("quote is not an exact source substring")
            citations.append(
                {
                    "marker": marker,
                    "chunk_id": chunk["chunk_id"],
                    "doc_title": chunk["doc_title"],
                    "article": chunk["article"],
                    "quote": quote,
                    "page_url": chunk["page_url"],
                    "file_url": chunk["file_url"],
                }
            )
        return GroundingResult(normalized, citations)


__all__ = [
    "GroundingResult",
    "StrictGroundingValidator",
    "normalize_citation_formats",
]
