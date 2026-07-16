"""Production runtime with a grounded deterministic fallback for LLM refusal.

The model is still called for policy wording.  If both of its citation-repair
attempts fail strict validation, the request falls back to the already
grounded extractive answer instead of falsely claiming the corpus has no rule.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.runtime_v9 import (
    build_local_query_plan_runtime,
    build_request_query_plan_runtime as _build_request,
)
from swufe_rag.orchestration import HybridRuntime


def build_request_query_plan_runtime(
    base_runtime: HybridRuntime,
    api_key: str,
    *,
    config_path: str | Path = "config.advanced.yaml",
):
    deterministic_answer = base_runtime.school_answer
    runtime = _build_request(base_runtime, api_key, config_path=config_path)
    model_answer = runtime.school_answer

    def answer_with_grounded_fallback(
        query: str, chunks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        answer = model_answer(query, chunks)
        if answer.get("refused"):
            fallback = deterministic_answer(query, chunks)
            if not fallback.get("refused") and fallback.get("citations"):
                return fallback
        return answer

    runtime.school_answer = answer_with_grounded_fallback
    return runtime


__all__ = ["build_local_query_plan_runtime", "build_request_query_plan_runtime"]
