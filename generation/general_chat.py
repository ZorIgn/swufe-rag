"""Unrestricted general conversation kept separate from the school RAG prompt."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from generation.llm import LLMClient, OpenAICompatibleClient


GENERAL_CHAT_SYSTEM_PROMPT = """你是友好、自然的通用人工智能助手。
可以回答普通知识、编程、学习、写作、情绪交流和日常问题。
当前分支不负责西南财经大学的真实制度、校内事实或官方网址；路由器会把这类问题交给可信知识库。
直接回答用户，不要提及路由器、检索阈值或内部系统。"""


class GeneralChatService:
    def __init__(self, client: LLMClient, *, max_history_messages: int = 12) -> None:
        if not 0 <= max_history_messages <= 40:
            raise ValueError("max_history_messages must be between 0 and 40")
        self.client = client
        self.max_history_messages = max_history_messages

    def answer(
        self,
        question: str,
        history: Sequence[tuple[str, str]] = (),
    ) -> str:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must not be blank")
        recent = list(history)[-self.max_history_messages :]
        transcript = "\n".join(
            f"{('用户' if role == 'user' else '助手')}：{content}"
            for role, content in recent
            if role in {"user", "assistant"} and content.strip()
        )
        prompt = (
            (f"【最近的普通对话】\n{transcript}\n\n" if transcript else "")
            + f"【当前问题】\n{question.strip()}"
        )
        return self.client.generate(GENERAL_CHAT_SYSTEM_PROMPT, prompt).strip()


def service_from_config(
    path: str | Path = "config.advanced.yaml",
) -> GeneralChatService:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required; install requirements.txt") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    generation = config.get("generation", {})
    client = OpenAICompatibleClient(
        str(generation.get("llm", "deepseek-chat")),
        temperature=float(generation.get("general_temperature", 0.7)),
        max_retries=int(generation.get("max_retries", 2)),
        timeout_seconds=float(generation.get("request_timeout_seconds", 60)),
    )
    return GeneralChatService(client)


__all__ = ["GENERAL_CHAT_SYSTEM_PROMPT", "GeneralChatService", "service_from_config"]
