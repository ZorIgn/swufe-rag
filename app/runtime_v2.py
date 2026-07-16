"""Builders for the QueryPlan -> SQL/RAG -> grounded wording runtime."""

from __future__ import annotations

from pathlib import Path

from academic_audit.database import AcademicDatabase
from app.runtime import _load_config, build_local_hybrid_runtime
from generation.context import ContextBuilder
from generation.general_chat import GeneralChatService
from generation.llm import OpenAICompatibleClient
from generation.pipeline import AdvancedGenerationService
from generation.structured_presenter import StructuredAnswerPresenter
from swufe_rag.orchestration import HybridRuntime
from swufe_rag.orchestration_v2 import QueryPlanRuntime, RuntimeCapabilities
from swufe_rag.query_plan import QuestionPlanner


DEFAULT_ACADEMIC_DATABASE = Path(__file__).parents[1] / "data" / "academic_v2.sqlite3"


def build_local_query_plan_runtime(
    chunks_path: str | Path = "data/chunks.jsonl",
    *,
    sources_path: str | Path = "data/sources.csv",
    metadata_path: str | Path = "data/metadata.sqlite3",
    config_path: str | Path = "config.advanced.yaml",
    academic_database: str | Path = DEFAULT_ACADEMIC_DATABASE,
) -> QueryPlanRuntime:
    base = build_local_hybrid_runtime(
        chunks_path,
        sources_path=sources_path,
        metadata_path=metadata_path,
        config_path=config_path,
    )
    return QueryPlanRuntime.from_base(
        base,
        planner=QuestionPlanner(),
        presenter=StructuredAnswerPresenter(),
        academic_db=AcademicDatabase(academic_database),
        capabilities=RuntimeCapabilities(),
    )


def build_request_query_plan_runtime(
    base_runtime: HybridRuntime,
    api_key: str,
    *,
    config_path: str | Path = "config.advanced.yaml",
) -> QueryPlanRuntime:
    """Create request-local model clients; the API key is never persisted."""

    if not isinstance(base_runtime, HybridRuntime):
        raise ValueError("request API keys require a HybridRuntime")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("X-LLM-API-Key must not be blank")
    config = _load_config(config_path)
    generation = config.get("generation", {})
    if not isinstance(generation, dict):
        raise ValueError("generation config must be a mapping")
    model = str(generation.get("llm", "deepseek-chat"))
    retries = int(generation.get("max_retries", 2))
    timeout = float(generation.get("request_timeout_seconds", 60))
    key = api_key.strip()

    planner_client = OpenAICompatibleClient(
        model,
        api_key=key,
        temperature=0,
        max_retries=retries,
        timeout_seconds=timeout,
    )
    answer_client = OpenAICompatibleClient(
        model,
        api_key=key,
        temperature=0,
        max_retries=retries,
        timeout_seconds=timeout,
    )
    general_client = OpenAICompatibleClient(
        model,
        api_key=key,
        temperature=float(generation.get("general_temperature", 0.7)),
        max_retries=retries,
        timeout_seconds=timeout,
    )
    grounded = AdvancedGenerationService(
        answer_client,
        refuse_th=float(generation.get("refuse_th", 0.35)),
        context_builder=ContextBuilder(
            max_context_chars=int(generation.get("max_context_chars", 7000)),
            max_chunk_chars=int(generation.get("max_chunk_chars", 1600)),
        ),
    )
    academic_db = (
        base_runtime.academic_db
        if isinstance(base_runtime, QueryPlanRuntime)
        else AcademicDatabase(DEFAULT_ACADEMIC_DATABASE)
    )
    runtime = QueryPlanRuntime(
        planner=QuestionPlanner(planner_client),
        presenter=StructuredAnswerPresenter(answer_client, model=model),
        academic_db=academic_db,
        capabilities=RuntimeCapabilities(
            real_question_understanding=True,
            real_rag_generation=True,
            real_general_generation=True,
            real_structured_generation=True,
            model=model,
        ),
        router=base_runtime.router,
        school_retrieve=base_runtime.school_retrieve,
        school_answer=grounded.answer,
        general_chat=GeneralChatService(general_client),
        metadata_db=base_runtime.metadata_db,
        sessions=base_runtime.sessions,
        runtime_mode=f"{base_runtime.mode}+request-llm-v2",
        runtime_info=getattr(base_runtime, "runtime_info", {}),
    )
    runtime.runtime_info = getattr(base_runtime, "runtime_info", {})
    return runtime


__all__ = [
    "DEFAULT_ACADEMIC_DATABASE",
    "build_local_query_plan_runtime",
    "build_request_query_plan_runtime",
]
