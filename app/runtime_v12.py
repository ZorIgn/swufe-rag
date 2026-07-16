"""Production runtime with intent-specific promotion retrieval expansion."""

from __future__ import annotations

from pathlib import Path
import re
from types import MethodType
from typing import Any

from app.runtime_v11 import (
    build_local_query_plan_runtime as _build_local,
    build_request_query_plan_runtime as _build_request,
)
from swufe_rag.orchestration import HybridRuntime
from swufe_rag.query_plan import QueryPlan


PROMOTION_CONDITION_RE = re.compile(r"(?:推免|保研|推荐免试).*(?:条件|资格|要求)|(?:条件|资格|要求).*(?:推免|保研|推荐免试)")


def _install_expansion(runtime: Any) -> Any:
    original = runtime._retrieve_rag

    def retrieve_rag(
        self: Any,
        plan: QueryPlan,
        question: str,
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        expanded = question
        if PROMOTION_CONDITION_RE.search(question):
            expanded += " 推免生基本条件 学分条件 成绩条件 外语条件 第四条"
        return original(plan, expanded, top_k=top_k)

    runtime._retrieve_rag = MethodType(retrieve_rag, runtime)
    return runtime


def build_local_query_plan_runtime(
    chunks_path: str | Path = "data/chunks.jsonl",
    *,
    sources_path: str | Path = "data/sources.csv",
    metadata_path: str | Path = "data/metadata.sqlite3",
    config_path: str | Path = "config.advanced.yaml",
    academic_database: str | Path = "data/academic_v2.sqlite3",
):
    return _install_expansion(
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
    return _install_expansion(
        _build_request(base_runtime, api_key, config_path=config_path)
    )


__all__ = ["build_local_query_plan_runtime", "build_request_query_plan_runtime"]
