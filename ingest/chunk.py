"""Article-aware chunking with bounded, header-preserving table windows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from ingest.contracts import KnowledgeChunk, validate_chunk
from ingest.models import DocumentElement, ParsedDocument, SourceRecord
from ingest.parse import join_wrapped_lines, normalize_text

_NUMERALS = "〇零一二三四五六七八九十百千万0-9"
_CHAPTER_RE = re.compile(rf"^\s*(第[{_NUMERALS}]+[章节])\s*(.*?)\s*$")
_ARTICLE_RE = re.compile(rf"^\s*(第[{_NUMERALS}]+条)\s*(.*?)\s*$")
_CN_HEADING_RE = re.compile(rf"^\s*([{_NUMERALS}]+、)\s*(.*?)\s*$")
_LIST_ITEM_RE = re.compile(r"^\s*(\d{1,3})\s*[.·、]\s*(.+)$")
_PROGRAM_HEADING_RE = re.compile(
    r"^.{2,100}(?:专业|专业类|计算机类).{0,40}(?:本科)?人才培养方案$"
)
_PROGRAM_BODY_START_RE = re.compile(
    r"^西南财经大学(.{2,80}?(?:专业|专业类))人才培养"
)
_INLINE_BOUNDARY_RE = re.compile(
    rf"(?<=[。！？；;])(?=第[{_NUMERALS}]+(?:条|章|节))"
)
_SENTENCE_RE = re.compile(r".*?(?:[。！？；;]|\n+|$)", re.S)

# Parsing creates traceable evidence, but it is not a human verification step.
# A reviewer-owned ledger is the only path that may later promote a chunk.
GENERATED_REVIEW_STATUS: Literal["review_required"] = "review_required"


@dataclass(frozen=True)
class _Segment:
    article: str
    text: str
    is_table: bool
    page: int | None


def _logical_lines(text: str) -> list[str]:
    expanded = _INLINE_BOUNDARY_RE.sub("\n", normalize_text(text))
    return [line.strip() for line in expanded.splitlines() if line.strip()]


def _segments(elements: list[DocumentElement]) -> list[_Segment]:
    result: list[_Segment] = []
    buffer: list[str] = []
    chapter = ""
    article = "正文"
    base_article = "正文"
    current_page: int | None = None

    def label() -> str:
        return " / ".join(item for item in (chapter, article) if item and item != "正文") or "正文"

    def flush() -> None:
        text = join_wrapped_lines("\n".join(buffer))
        if text:
            result.append(_Segment(label(), text, False, current_page))
        buffer.clear()

    for element in elements:
        if element.page is not None and element.page != current_page:
            flush()
            current_page = element.page
        if element.kind == "table":
            flush()
            if element.page and chapter:
                table_article = f"{chapter} / 第{element.page}页表格"
            else:
                table_article = f"第{element.page}页表格" if element.page else label()
            result.append(_Segment(table_article, element.text, True, element.page))
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
            if _PROGRAM_HEADING_RE.match(line):
                flush()
                chapter = line[:100]
                article = "正文"
                base_article = "正文"
                continue
            program_body_match = _PROGRAM_BODY_START_RE.match(line)
            if program_body_match:
                program = program_body_match.group(1)
                if program not in chapter:
                    flush()
                    chapter = f"{program}人才培养方案"
                buffer.append(line)
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

def _split_table(markdown: str, limit: int) -> list[str]:
    """Split large Markdown tables on row boundaries and repeat the header."""

    intro = "原表：\n"
    if len(intro) + len(markdown) <= limit:
        return [intro + markdown]
    lines = [line for line in markdown.splitlines() if line.strip()]
    header_lines = (
        lines[:2]
        if len(lines) >= 2 and lines[0].lstrip().startswith("|")
        and lines[1].lstrip().startswith("|")
        else []
    )
    rows = lines[2:] if header_lines else lines
    header = "\n".join(header_lines)
    fixed = intro + (header + "\n" if header else "")
    if len(fixed) >= limit:
        fixed = intro
    row_budget = max(1, limit - len(fixed))
    groups: list[list[str]] = []
    current: list[str] = []
    current_len = len(fixed)
    for row in rows:
        needed = len(row) + (1 if current else 0)
        if len(row) > row_budget:
            if current:
                groups.append(current)
                current = []
                current_len = len(fixed)
            groups.extend(
                [[row[start : start + row_budget]]
                 for start in range(0, len(row), row_budget)]
            )
            continue
        if current and current_len + needed > limit:
            groups.append(current)
            current = []
            current_len = len(fixed)
        current.append(row)
        current_len += len(row) + (1 if len(current) > 1 else 0)
    if current:
        groups.append(current)
    if not groups:
        return _split_body(intro + markdown, limit)
    return [fixed + "\n".join(group) for group in groups]

def _page_url(source: SourceRecord, page: int | None) -> str:
    """Return an exact PDF page link when the source is the original file."""

    if page is None or "/training/" in f"/{source.file.replace(chr(92), '/')}":
        return source.page_url
    base_url = source.page_url if ".pdf" in source.page_url.lower() else source.file_url
    if ".pdf" not in base_url.lower():
        return source.page_url
    parsed = urlsplit(base_url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, f"page={page}"))


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
        page_label = f"原文件第{segment.page}页" if segment.page else ""
        if page_label and f"第{segment.page}页" not in article:
            article = f"{article} / {page_label}"
        prefix = f"《{source.doc_title}》{article}\n"
        if segment.is_table:
            bodies = _split_table(segment.text, chunk_max_len - len(prefix))
        else:
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
                "page_url": _page_url(source, segment.page),
                "file_url": source.file_url,
                "is_table": segment.is_table,
                "review_status": GENERATED_REVIEW_STATUS,
            }
            chunks.append(validate_chunk(chunk))
    if not chunks:
        raise ValueError(f"source produced no chunks: {source.file}")
    return chunks
