"""Production runtime with promotion-answer completeness validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.runtime_v14 import (
    PROMOTION_CONDITION_RE,
    build_local_query_plan_runtime as _build_local,
    build_request_query_plan_runtime as _build_request,
)
from generation.promotion_conditions import canonical, complete
from swufe_rag.orchestration import HybridRuntime


def _install_completion_guard(runtime: Any) -> Any:
    original = runtime.school_answer

    def answer(query: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        value = original(query, chunks)
        if PROMOTION_CONDITION_RE.search(query) and not complete(value):
            fallback = canonical(chunks)
            if fallback is not None:
                return fallback
        return value

    runtime.school_answer = answer
    return runtime


def build_local_query_plan_runtime(
    chunks_path: str | Path = "data/chunks.jsonl",
    *,
    sources_path: str | Path = "data/sources.csv",
    metadata_path: str | Path = "data/metadata.sqlite3",
    config_path: str | Path = "config.advanced.yaml",
    academic_database: str | Path = "data/academic_v2.sqlite3",
):
    return _install_completion_guard(
        _build_local(
            chunks_path,
            sources_path=sources_path,
            metadata_path=metadata_path,
            config_path=config_path,
            academic_database=academic_database,
        )
    )


def build_request_query_plan_runtime(
    base_runtime: HybridRuntime,
    api_key: str,
    *,
    config_path: str | Path = "config.advanced.yaml",
):
    return _install_completion_guard(
        _build_request(base_runtime, api_key, config_path=config_path)
    )


__all__ = ["build_local_query_plan_runtime", "build_request_query_plan_runtime"]
