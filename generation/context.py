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
        number_hits = sum(
            number in passage
            for number in analysis.numbers
            if not re.fullmatch(r"(?:19|20)\d{2}", number)
        )
        intent = 0.0
        if re.search(r"最迟|什么时候|多久|时间", analysis.normalized) and re.search(
            r"\d|小时|工作日|日前|月前|周前|学期前|开考前", passage
        ):
            intent += 1.0
        if re.search(
            r"多少|几分|比例|学分|分数|怎么计算|怎么算", analysis.normalized
        ) and re.search(r"\d|%|％|×|=|学分|分", passage):
            intent += 0.8
        if re.search(r"怎么计算|怎么算", analysis.normalized) and re.search(
            r"%|％|×|=|构成|占比", passage
        ):
            intent += 1.0
        if re.search(r"进考场|进入考场|迟到", analysis.normalized) and re.search(
            r"未进入考场|取消.*考试资格", passage
        ):
            intent += 1.2
        if "什么时候" in analysis.normalized and re.search(
            r"期中|期末|学期|大一|大二", passage
        ):
            intent += 1.0
        if re.search(r"多少门|几门", analysis.normalized) and re.search(
            r"\d+\s*门", passage
        ):
            intent += 1.2
        if "最长学习年限" in analysis.normalized and re.search(
            r"最长为|最长学习年限.*\d", passage
        ):
            intent += 1.5
        header_only = len(passage) < 150 and (
            re.fullmatch(r"《[^》]+》[^。！？；;]*", passage)
            or re.fullmatch(r"下表为.{0,120}的原表[:：]", passage)
        )
        header_penalty = 2.0 if header_only else 0.0
        score = (
            coverage
            + code_hits * 2.0
            + article_hits * 1.2
            + number_hits * 0.35
            + intent
            - header_penalty
        )
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
