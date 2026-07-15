"""Team-facing B/C facade with frozen function signatures."""

from __future__ import annotations

from threading import RLock
from typing import Any

from contracts import AnswerResult, RetrievedChunk
from generation.pipeline import AdvancedGenerationService, service_from_config
from retrieval.pipeline import AdvancedRetriever


_retriever: AdvancedRetriever | None = None
_generation: AdvancedGenerationService | None = None
_lock = RLock()


def configure(
    *,
    retriever: AdvancedRetriever | None = None,
    generation: AdvancedGenerationService | None = None,
) -> None:
    """Inject both pipelines for tests or application startup."""

    global _retriever, _generation
    with _lock:
        _retriever = retriever
        _generation = generation


def _get_retriever() -> AdvancedRetriever:
    global _retriever
    with _lock:
        if _retriever is None:
            _retriever = AdvancedRetriever.from_artifacts()
        return _retriever


def _get_generation() -> AdvancedGenerationService:
    global _generation
    with _lock:
        if _generation is None:
            _generation = service_from_config()
        return _generation


def retrieve(
    query: str,
    top_k: int = 5,
    college: str | None = None,
    cohort: str | None = None,
) -> list[RetrievedChunk]:
    return _get_retriever().retrieve(query, top_k, college, cohort)


def retrieve_scoped(
    query: str,
    top_k: int = 5,
    college: str | None = None,
    cohort: str | None = None,
    *,
    policy_year: int | None = None,
    topic: str | None = None,
) -> list[RetrievedChunk]:
    """Additive school-orchestration entry point with SQL-bound scope.

    The frozen ``retrieve`` signature remains unchanged.  This entry point is
    used only by the mixed-dialogue layer when the validated router supplies a
    policy year or topic allow-list value.
    """

    return _get_retriever().retrieve_scoped(
        query,
        top_k=top_k,
        college=college,
        cohort=cohort,
        policy_year=policy_year,
        topic=topic,
    )


def answer(query: str, chunks: list[dict[str, Any]]) -> AnswerResult:
    return _get_generation().answer(query, chunks)
