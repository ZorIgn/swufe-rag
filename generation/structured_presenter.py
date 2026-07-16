"""LLM wording for SQL answers with a deterministic fact-preserving guard."""

from __future__ import annotations

import json
import re
from typing import Any

from generation.grounded_answer import URL_RE
from generation.llm import LLMClient


PRESENTER_SYSTEM_PROMPT = """你只负责把已经核验的教务查询结果整理成自然中文。
禁止增加、删除或修改课程、课程代码、学分、学期、课程性质和模块。
禁止生成网址。保留全部引用编号。若是课程列表，必须完整列出输入中的每条课程记录。
只输出最终回答正文，不要解释你的工作过程。"""

COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,5}\d{3}\b", re.I)
CITATION_RE = re.compile(r"\[(\d+)\]")
NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?")


def _without_citations(value: str) -> str:
    return CITATION_RE.sub("", value)


def validate_structured_wording(
    canonical: str,
    candidate: str,
    *,
    allowed_markers: set[int],
) -> bool:
    """Reject wording that loses or invents machine-checked facts."""

    if not candidate.strip() or URL_RE.search(candidate):
        return False
    canonical_codes = {value.upper() for value in COURSE_CODE_RE.findall(canonical)}
    candidate_codes = {value.upper() for value in COURSE_CODE_RE.findall(candidate)}
    if candidate_codes != canonical_codes:
        return False
    canonical_numbers = set(NUMBER_RE.findall(_without_citations(canonical)))
    candidate_numbers = set(NUMBER_RE.findall(_without_citations(candidate)))
    if candidate_numbers != canonical_numbers:
        return False
    markers = {int(value) for value in CITATION_RE.findall(candidate)}
    if markers and not markers <= allowed_markers:
        return False
    if allowed_markers and not markers:
        return False
    return True


class StructuredAnswerPresenter:
    def __init__(self, client: LLMClient | None = None, *, model: str | None = None) -> None:
        self.client = client
        self.model = model

    def present(
        self,
        question: str,
        canonical_answer: str,
        citations: list[dict[str, Any]],
    ) -> tuple[str, bool, str | None]:
        if self.client is None:
            return canonical_answer, False, None
        allowed = {int(item["marker"]) for item in citations}
        evidence = [
            {
                "evidence_id": int(item["marker"]),
                "doc_title": item["doc_title"],
                "article": item["article"],
                "quote": item["quote"],
            }
            for item in citations
        ]
        prompt = json.dumps(
            {
                "question": question,
                "canonical_answer": canonical_answer,
                "evidence": evidence,
            },
            ensure_ascii=False,
        )
        try:
            candidate = self.client.generate(PRESENTER_SYSTEM_PROMPT, prompt).strip()
        except Exception:
            return canonical_answer, False, "generation_failed"
        if not validate_structured_wording(
            canonical_answer, candidate, allowed_markers=allowed
        ):
            return canonical_answer, True, "fact_validation_failed"
        return candidate, True, None


__all__ = [
    "PRESENTER_SYSTEM_PROMPT",
    "StructuredAnswerPresenter",
    "validate_structured_wording",
]
