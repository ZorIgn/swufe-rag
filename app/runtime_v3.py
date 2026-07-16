"""Catalog-aware wrappers around the v2 runtime builders."""

from __future__ import annotations

from pathlib import Path

from app.runtime_v2 import (
    DEFAULT_ACADEMIC_DATABASE,
    build_local_query_plan_runtime as _build_local,
    build_request_query_plan_runtime as _build_request,
)
from swufe_rag.orchestration import HybridRuntime
from swufe_rag.query_plan_catalog import CatalogAwareQuestionPlanner


def build_local_query_plan_runtime(
    chunks_path: str | Path = "data/chunks.jsonl",
    *,
    sources_path: str | Path = "data/sources.csv",
    metadata_path: str | Path = "data/metadata.sqlite3",
    config_path: str | Path = "config.advanced.yaml",
    academic_database: str | Path = DEFAULT_ACADEMIC_DATABASE,
):
    runtime = _build_local(
        chunks_path,
        sources_path=sources_path,
        metadata_path=metadata_path,
        config_path=config_path,
        academic_database=academic_database,
    )
    runtime.planner = CatalogAwareQuestionPlanner(runtime.academic_db)
    return runtime


def build_request_query_plan_runtime(
    base_runtime: HybridRuntime,
    api_key: str,
    *,
    config_path: str | Path = "config.advanced.yaml",
):
    runtime = _build_request(base_runtime, api_key, config_path=config_path)
    runtime.planner = CatalogAwareQuestionPlanner(
        runtime.academic_db, runtime.planner.client
    )
    return runtime


__all__ = [
    "build_local_query_plan_runtime",
    "build_request_query_plan_runtime",
]
