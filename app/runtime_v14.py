"""Production runtime with direct authoritative policy-clause retrieval."""

from __future__ import annotations

from pathlib import Path
from types import MethodType
from typing import Any

from app.runtime_v12 import (
    PROMOTION_CONDITION_RE,
    build_local_query_plan_runtime as _build_local,
    build_request_query_plan_runtime as _build_request,
)
from contracts import CHUNK_FIELDS
from swufe_rag.orchestration import HybridRuntime
from swufe_rag.query_plan import QueryPlan


PROMOTION_RULE = "西南财经大学推荐免试研究生管理办法（2024年修订）"


def _authoritative_chunks(runtime: Any) -> list[dict[str, Any]]:
    rows = runtime.metadata_db.connection.execute(
        """
        SELECT c.chunk_id
        FROM chunks AS c
        JOIN sources AS s ON s.source_id = c.source_id
        WHERE s.enabled = 1 AND s.trusted = 1
          AND s.doc_title = ? AND c.article LIKE '%第四条%'
        ORDER BY c.embedding_row
        """,
        (PROMOTION_RULE,),
    ).fetchall()
    values = []
    for row in rows:
        stored = runtime.metadata_db.chunk(str(row["chunk_id"]))
        if stored is None:
            continue
        item = {field: getattr(stored, field) for field in CHUNK_FIELDS}
        item["score"] = 1.0
        values.append(item)
    return values


def _install_authoritative_policy(runtime: Any) -> Any:
    original = runtime._retrieve_rag

    def retrieve_rag(
        self: Any,
        plan: QueryPlan,
        question: str,
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        chunks = original(plan, question, top_k=top_k)
        if not PROMOTION_CONDITION_RE.search(question):
            return chunks
        preferred = _authoritative_chunks(self)
        seen = {chunk["chunk_id"] for chunk in preferred}
        return preferred + [chunk for chunk in chunks if chunk["chunk_id"] not in seen]

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
    return _install_authoritative_policy(
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
    return _install_authoritative_policy(
        _build_request(base_runtime, api_key, config_path=config_path)
    )


__all__ = ["build_local_query_plan_runtime", "build_request_query_plan_runtime"]
