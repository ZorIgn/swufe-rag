"""Offline-cache production runtime with an explicit loaded-index fingerprint."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from app.runtime import _runtime_fingerprint
from app.runtime_v3 import (
    build_local_query_plan_runtime as _build_local,
    build_request_query_plan_runtime as _build_request,
)
from swufe_rag.orchestration import HybridRuntime


def build_local_query_plan_runtime(
    chunks_path: str | Path = "data/chunks.jsonl",
    *,
    sources_path: str | Path = "data/sources.csv",
    metadata_path: str | Path = "data/metadata.sqlite3",
    config_path: str | Path = "config.advanced.yaml",
    academic_database: str | Path = "data/academic_v2.sqlite3",
):
    runtime = _build_local(
        chunks_path,
        sources_path=sources_path,
        metadata_path=metadata_path,
        config_path=config_path,
        academic_database=academic_database,
    )
    retriever = getattr(runtime.school_retrieve, "__self__", None)
    if retriever is not None:
        from app.runtime import _load_config

        config = _load_config(config_path)
        runtime.runtime_info = _runtime_fingerprint(
            retriever,
            chunks_path=chunks_path,
            sources_path=sources_path,
            metadata_path=metadata_path,
            config_path=config_path,
            artifacts_path=config.get("paths", {}).get("artifacts", "artifacts"),
        )
        runtime.runtime_info["academic_database"] = str(
            Path(academic_database).resolve()
        )
    return runtime


def build_request_query_plan_runtime(
    base_runtime: HybridRuntime,
    api_key: str,
    *,
    config_path: str | Path = "config.advanced.yaml",
):
    runtime = _build_request(base_runtime, api_key, config_path=config_path)
    runtime.runtime_info = getattr(base_runtime, "runtime_info", {})
    return runtime


__all__ = [
    "build_local_query_plan_runtime",
    "build_request_query_plan_runtime",
]
