"""Query-focused context assembly with deterministic character budgets."""

from __future__ import annotations

from dataclasses import dataclass
import re

from contracts import RetrievedChunk
from retrieval.query import QueryAnalysis, analyze_query, lexical_tokens, normalize_query


PASSAGE_RE = re.compile(r"[^\n。！？!?；;]+[。！？!?；;]?|[^\n]+")


@dataclass(frozen=True)
class ContextItem:
    marker: int
    chunk: RetrievedChunk
    excerpt: str


class ContextBuilder:
    def __init__(
        self,
        *,
        max_context_chars: int = 7000,
        max_chunk_chars: int = 1600,
        min_chunk_chars: int = 240,
    ) -> None:
        if max_context_chars < 1000:
            raise ValueError("max_context_chars must be at least 1000")
        if not 100 <= min_chunk_chars <= max_chunk_chars:
            raise ValueError("chunk character limits are inconsistent")
        self.max_context_chars = max_context_chars
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars

    @staticmethod
    def _passage_score(analysis: QueryAnalysis, passage: str) -> tuple[float, int]:
        tokens = lexical_tokens(passage)
        coverage = (
            len(analysis.tokens & tokens) / len(analysis.tokens)
            if analysis.tokens
            else 0.0
        )
        normalized = normalize_query(passage).upper().replace("-", "").replace(" ", "")
        code_hits = sum(code in normalized for code in analysis.course_codes)
        article_hits = sum(article in passage for article in analysis.article_refs)
        number_hits = sum(number in passage for number in analysis.numbers)
        score = coverage + code_hits * 2.0 + article_hits * 1.2 + number_hits * 0.35
        return score, len(passage)

    def _excerpt(self, analysis: QueryAnalysis, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        passages = [match.group(0).strip() for match in PASSAGE_RE.finditer(text) if match.group(0).strip()]
        if not passages:
            return text[:limit]
        ranked = sorted(
            range(len(passages)),
            key=lambda index: (
                -self._passage_score(analysis, passages[index])[0],
                index,
            ),
        )
        chosen: list[int] = []
        used = 0
        separator_cost = len("\n…\n")
        for index in ranked:
            passage = passages[index]
            cost = len(passage) + (separator_cost if chosen else 0)
            if cost + used <= limit:
                chosen.append(index)
                used += cost
            if used >= min(limit, self.min_chunk_chars):
                break
        if not chosen:
            return passages[ranked[0]][:limit]
        return "\n…\n".join(passages[index] for index in sorted(chosen))

    @staticmethod
    def _header(marker: int, chunk: RetrievedChunk) -> str:
        return (
            f"<source id=\"{marker}\">\n"
            f"文档：《{chunk['doc_title']}》\n"
            f"条款：{chunk['article']}\n"
            f"适用：{chunk['level']} / {chunk['college']} / {chunk['cohort']} / {chunk['year']} / {chunk['status']}\n"
            "原文：\n"
        )

    def build(self, query: str, chunks: list[RetrievedChunk]) -> tuple[str, list[ContextItem]]:
        analysis = analyze_query(query)
        remaining = self.max_context_chars
        blocks: list[str] = []
        items: list[ContextItem] = []
        for marker, chunk in enumerate(chunks, start=1):
            header = self._header(marker, chunk)
            footer = "\n</source>"
            available = min(self.max_chunk_chars, remaining - len(header) - len(footer))
            if available < self.min_chunk_chars:
                break
            excerpt = self._excerpt(analysis, chunk["text"], available)
            block = header + excerpt + footer
            blocks.append(block)
            items.append(ContextItem(marker, chunk, excerpt))
            remaining -= len(block) + 2
        return "\n\n".join(blocks), items
