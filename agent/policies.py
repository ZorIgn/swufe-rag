"""Hard execution limits for the bounded agent."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimePolicy:
    max_tool_calls: int = 8
    tool_timeout_seconds: float = 15.0
    max_validation_retries: int = 1
    max_question_chars: int = 4000
