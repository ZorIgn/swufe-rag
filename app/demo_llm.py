"""Deterministic extractive LLM substitute for the isolated debug runtime."""

from __future__ import annotations

import re

from generation.prompts import REFUSAL_TEXT
from retrieval.query import analyze_query, lexical_tokens, normalize_query


SOURCE_RE = re.compile(
    r'<source id="(\d+)">.*?原文：\s*(.*?)\s*</source>', re.S
)
QUESTION_RE = re.compile(r"【(?:原)?问题】\s*(.*?)(?:\n\n|\Z)", re.S)
PASSAGE_RE = re.compile(r"[^\n。！？!?；;]+[。！？!?；;]?|\|[^\n]+\|")


class DemoGroundedClient:
    """Extracts one grounded passage; never used by the production runtime."""

    @staticmethod
    def _question(prompt: str) -> str:
        matches = QUESTION_RE.findall(prompt)
        return matches[-1].strip() if matches else ""

    @staticmethod
    def _score(question: str, passage: str) -> float:
        analysis = analyze_query(question)
        tokens = lexical_tokens(passage)
        overlap = len(analysis.tokens & tokens)
        coverage = overlap / len(analysis.tokens) if analysis.tokens else 0.0
        normalized = normalize_query(passage).upper().replace("-", "").replace(" ", "")
        exact = sum(code in normalized for code in analysis.course_codes) * 2.5
        exact += sum(article in passage for article in analysis.article_refs) * 1.5
        exact += sum(number in passage for number in analysis.numbers) * 0.3
        return coverage + exact

    @staticmethod
    def _cite(passage: str, marker: int) -> str:
        text = passage.strip()
        if not text:
            return REFUSAL_TEXT
        if text[-1] in "。！？!?；;":
            return f"{text[:-1]}[{marker}]{text[-1]}"
        return f"{text}[{marker}]。"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        question = self._question(user_prompt)
        candidates: list[tuple[float, int, str]] = []
        for marker_text, source in SOURCE_RE.findall(user_prompt):
            marker = int(marker_text)
            passages = [
                match.group(0).strip()
                for match in PASSAGE_RE.finditer(source)
                if match.group(0).strip()
            ] or [source.strip()]
            for passage in passages:
                candidates.append((self._score(question, passage), marker, passage))
        if not candidates:
            return REFUSAL_TEXT
        score, marker, passage = max(
            candidates, key=lambda item: (item[0], -item[1], len(item[2]))
        )
        if score < 0.12:
            return REFUSAL_TEXT
        return self._cite(passage, marker)
