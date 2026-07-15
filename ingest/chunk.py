"""Article-aware chunking that preserves tables as atomic evidence."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re

from contracts import KnowledgeChunk, validate_chunk
from ingest.models import DocumentElement, ParsedDocument, SourceRecord
from ingest.parse import join_wrapped_lines, normalize_text


_NUMERALS = "〇零一二三四五六七八九十百千万0-9"
_CHAPTER_RE = re.compile(rf"^\s*(第[{_NUMERALS}]+[章节])\s*(.*?)\s*$")
_ARTICLE_RE = re.compile(rf"^\s*(第[{_NUMERALS}]+条)\s*(.*?)\s*$")
_CN_HEADING_RE = re.compile(rf"^\s*([{_NUMERALS}]+、)\s*(.*?)\s*$")
_LIST_ITEM_RE = re.compile(r"^\s*(\d{1,3})\s*[.·、]\s*(.+)$")
_INLINE_BOUNDARY_RE = re.compile(
    rf"(?<=[。！？；;])(?=第[{_NUMERALS}]+(?:条|章|节))"
)
_SENTENCE_RE = re.compile(r".*?(?:[。！？；;]|\n+|$)", re.S)


@dataclass(frozen=True)
class _Segment:
    article: str
    text: str
    is_table: bool


def _logical_lines(text: str) -> list[str]:
    expanded = _INLINE_BOUNDARY_RE.sub("\n", normalize_text(text))
    return [line.strip() for line in expanded.splitlines() if line.strip()]


def _segments(elements: list[DocumentElement]) -> list[_Segment]:
    result: list[_Segment] = []
    buffer: list[str] = []
    chapter = ""
    article = "正文"
    base_article = "正文"

    def label() -> str:
        return " / ".join(item for item in (chapter, article) if item and item != "正文") or "正文"

    def flush() -> None:
        text = join_wrapped_lines("\n".join(buffer))
        if text:
            result.append(_Segment(label(), text, False))
        buffer.clear()

    for element in elements:
        if element.kind == "table":
            flush()
            table_article = f"第{element.page}页表格" if element.page else label()
            result.append(_Segment(table_article, element.text, True))
            continue
        for line in _logical_lines(element.text):
            chapter_match = _CHAPTER_RE.match(line)
            if chapter_match:
                flush()
                chapter = " ".join(part for part in chapter_match.groups() if part).strip()
                article = "正文"
                base_article = "正文"
                continue
            article_match = _ARTICLE_RE.match(line)
            if article_match:
                flush()
                article = article_match.group(1)
                base_article = article
                remainder = article_match.group(2).strip()
                if remainder:
                    buffer.append(remainder)
                continue
            heading_match = _CN_HEADING_RE.match(line)
            if element.kind == "heading" or heading_match:
                flush()
                article = line[:100]
                base_article = article
                continue
            list_match = _LIST_ITEM_RE.match(line)
            if list_match:
                flush()
                article = f"{base_article} / 第{list_match.group(1)}项"
                buffer.append(list_match.group(2).strip())
                continue
            buffer.append(line)
    flush()
    return result


def _split_body(text: str, limit: int) -> list[str]:
    if limit < 80:
        raise ValueError("chunk_max_len is too small for a useful prefixed chunk")
    if len(text) <= limit:
        return [text]
    sentences = [match.group(0).strip() for match in _SENTENCE_RE.finditer(text)]
    sentences = [sentence for sentence in sentences if sentence]
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > limit:
            if current:
                parts.append(current)
                current = ""
            parts.extend(
                sentence[start : start + limit]
                for start in range(0, len(sentence), limit)
            )
            continue
        candidate = sentence if not current else current + sentence
        if len(candidate) <= limit:
            current = candidate
        else:
            parts.append(current)
            current = sentence
    if current:
        parts.append(current)
    return parts


def build_chunks(
    document: ParsedDocument,
    source: SourceRecord,
    *,
    chunk_max_len: int = 500,
) -> list[KnowledgeChunk]:
    if chunk_max_len < 160:
        raise ValueError("chunk_max_len must be at least 160")
    source_key = f"{source.file}|{source.doc_title}|{source.year}"
    prefix_id = sha256(source_key.encode("utf-8")).hexdigest()[:12]
    chunks: list[KnowledgeChunk] = []

    for segment in _segments(document.elements):
        article = segment.article or "正文"
        if segment.is_table:
            bodies = [f"下表为{source.doc_title}中“{article}”的原表：\n{segment.text}"]
        else:
            prefix = f"《{source.doc_title}》{article}\n"
            bodies = _split_body(segment.text, chunk_max_len - len(prefix))
        for body in bodies:
            text = f"《{source.doc_title}》{article}\n{body}".strip()
            chunk: KnowledgeChunk = {
                "chunk_id": f"swufe_{prefix_id}_{len(chunks) + 1:04d}",
                "text": text,
                "doc_title": source.doc_title,
                "article": article,
                "level": source.level,
                "college": source.college,
                "cohort": source.cohort,
                "year": source.year,
                "status": source.status,
                "page_url": source.page_url,
                "file_url": source.file_url,
                "is_table": segment.is_table,
            }
            chunks.append(validate_chunk(chunk))
    if not chunks:
        raise ValueError(f"source produced no chunks: {source.file}")
    return chunks
