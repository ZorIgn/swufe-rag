"""Advanced grounded-answer pipeline preserving the frozen answer contract."""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any

from contracts import (
    AnswerResult,
    CitationValidationError,
    RetrievedChunk,
    validate_answer_result,
    validate_retrieved_chunk,
)
from generation.context import ContextBuilder
from generation.grounding import StrictGroundingValidator
from generation.llm import LLMClient, OpenAICompatibleClient
from generation.prompts import REFUSAL_TEXT
from retrieval.query import (
    analyze_query,
    entity_coverage,
)


ADVANCED_SYSTEM_PROMPT = f"""你是西南财经大学教务政策问答助手。可信度高于回答完整度。
严格遵守：
1. 只能依据 <source> 内的原文回答，不得使用模型记忆补充学校政策；
2. 每个可核查的政策事实、数字、条件、比较或结论都必须在句末标注[编号]，角标放在句末标点之前；
3. 一句话最多引用4个来源，引用必须支持整句话，不输出单独的参考文献列表；
4. 学分、比例、年份、课程代码、资格条件必须与原文逐字一致；
5. 多文件综合时要区分校级总则和学院细则，不得扩大适用学院、年级或状态；
6. 问题中的课程名、课程代码或明确实体未出现在资料中时必须拒答；
7. 资料不足或相互冲突时只回答“{REFUSAL_TEXT}”；
8. 只输出简洁 Markdown 正文，不输出 JSON、推理过程或虚构链接。"""


class EvidenceGate:
    def __init__(self, *, dense_threshold: float = 0.35) -> None:
        if not 0 <= dense_threshold <= 1:
            raise ValueError("dense_threshold must be between 0 and 1")
        self.dense_threshold = dense_threshold

    def sufficient(self, query: str, chunks: list[RetrievedChunk]) -> bool:
        if not chunks:
            return False
        if max(chunk["score"] for chunk in chunks) < self.dense_threshold:
            return False
        analysis = analyze_query(query)
        if not entity_coverage(analysis, chunks):
            return False
        return True


class AdvancedGenerationService:
    def __init__(
        self,
        client: LLMClient,
        *,
        refuse_th: float = 0.35,
        context_builder: ContextBuilder | None = None,
        validator: StrictGroundingValidator | None = None,
    ) -> None:
        self.client = client
        self.gate = EvidenceGate(dense_threshold=refuse_th)
        self.context_builder = context_builder or ContextBuilder()
        self.validator = validator or StrictGroundingValidator()

    @staticmethod
    def _refusal() -> AnswerResult:
        return validate_answer_result(
            {"answer_md": REFUSAL_TEXT, "citations": [], "refused": True}
        )

    @staticmethod
    def _query(query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be blank")
        return query.strip()

    @staticmethod
    def _user_prompt(query: str, context: str) -> str:
        return f"【参考资料】\n{context}\n\n【问题】\n{query}"

    @staticmethod
    def _repair_prompt(
        query: str, context: str, answer: str, error: str
    ) -> str:
        return (
            f"【参考资料】\n{context}\n\n【原问题】\n{query}\n\n"
            f"【未通过校验的回答】\n{answer}\n\n【校验错误】\n{error}\n\n"
            "只允许调整角标位置、删除无依据句子或改为资料中的原文表达；不得新增事实。"
            f"无法修复时只回答“{REFUSAL_TEXT}”"
        )

    def answer(self, query: str, chunks: list[dict[str, Any]]) -> AnswerResult:
        clean_query = self._query(query)
        if not isinstance(chunks, list):
            raise ValueError("chunks must be a list")
        validated = [validate_retrieved_chunk(chunk) for chunk in chunks]
        if not self.gate.sufficient(clean_query, validated):
            return self._refusal()

        context, items = self.context_builder.build(clean_query, validated)
        prompt_chunks = [item.chunk for item in items]
        if not prompt_chunks:
            return self._refusal()
        response = self.client.generate(
            ADVANCED_SYSTEM_PROMPT, self._user_prompt(clean_query, context)
        )
        try:
            grounded = self.validator.validate(response, prompt_chunks)
        except CitationValidationError as first_error:
            repaired = self.client.generate(
                ADVANCED_SYSTEM_PROMPT,
                self._repair_prompt(clean_query, context, response, str(first_error)),
            )
            try:
                grounded = self.validator.validate(repaired, prompt_chunks)
            except CitationValidationError:
                return self._refusal()
        if grounded.answer == REFUSAL_TEXT:
            return self._refusal()
        return validate_answer_result(
            {
                "answer_md": grounded.answer,
                "citations": grounded.citations,
                "refused": False,
            }
        )


def service_from_config(
    path: str | Path = "config.advanced.yaml",
) -> AdvancedGenerationService:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required; install requirements.txt") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    generation = config.get("generation", {})
    client = OpenAICompatibleClient(
        str(generation.get("llm", "deepseek-chat")),
        temperature=float(generation.get("temperature", 0)),
        max_retries=int(generation.get("max_retries", 2)),
        timeout_seconds=float(generation.get("request_timeout_seconds", 60)),
    )
    builder = ContextBuilder(
        max_context_chars=int(generation.get("max_context_chars", 7000)),
        max_chunk_chars=int(generation.get("max_chunk_chars", 1600)),
    )
    return AdvancedGenerationService(
        client,
        refuse_th=float(generation.get("refuse_th", 0.35)),
        context_builder=builder,
    )


_default_service: AdvancedGenerationService | None = None
_default_lock = RLock()


def configure_default(service: AdvancedGenerationService | None) -> None:
    global _default_service
    with _default_lock:
        _default_service = service


def _get_default() -> AdvancedGenerationService:
    global _default_service
    with _default_lock:
        if _default_service is None:
            _default_service = service_from_config()
        return _default_service


def answer(query: str, chunks: list[dict[str, Any]]) -> AnswerResult:
    return _get_default().answer(query, chunks)


__all__ = [
    "ADVANCED_SYSTEM_PROMPT",
    "AdvancedGenerationService",
    "EvidenceGate",
    "answer",
    "configure_default",
    "service_from_config",
]
