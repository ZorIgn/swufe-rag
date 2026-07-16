"""Production runtime with authoritative promotion-clause ordering."""

from __future__ import annotations

from pathlib import Path
from types import MethodType
from typing import Any

from app.runtime_v12 import (
    PROMOTION_CONDITION_RE,
    build_local_query_plan_runtime as _build_local,
    build_request_query_plan_runtime as _build_request,
)
from swufe_rag.orchestration import HybridRuntime
from swufe_rag.query_plan import QueryPlan


def _install_clause_priority(runtime: Any) -> Any:
    original = runtime._retrieve_rag

    def retrieve_rag(
        self: Any,
        plan: QueryPlan,
        question: str,
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not PROMOTION_CONDITION_RE.search(question):
            return original(plan, question, top_k=top_k)
        chunks = original(plan, question, top_k=max(top_k, 40))
        preferred = [
            chunk
            for chunk in chunks
            if "西南财经大学推荐免试研究生管理办法（2024年修订）"
            == chunk["doc_title"]
            and "第四条" in chunk["article"]
        ]
        if preferred:
            remainder = [chunk for chunk in chunks if chunk not in preferred]
            return preferred + remainder
        return chunks

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
    return _install_clause_priority(
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
    return _install_clause_priority(
        _build_request(base_runtime, api_key, config_path=config_path)
    )


__all__ = ["build_local_query_plan_runtime", "build_request_query_plan_runtime"]
