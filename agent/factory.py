"""Explicit production composition root; no import-time patching."""

from __future__ import annotations

from pathlib import Path

from academic.database import AcademicRepository
from academic.tools import AcademicTools
from agent.orchestrator import AgentRuntime, RuntimeDependencies
from agent.otel import OpenTelemetryTracer
from agent.policies import RuntimePolicy
from agent.session import InMemoryTTLSessionStore
from agent.tools import PlanExecutor, standard_registry
from generation.synthesizer import DeterministicSynthesizer
from generation.validator import ClaimValidator
from query.normalization import normalize
from query.planner import build_plan
from query.schemas import NormalizedQuery, UnderstandingDraft
from query.understanding import QuestionUnderstanding, StructuredModel


def build_runtime(
    database_path: str | Path = "data/academic.sqlite3",
    *,
    model: StructuredModel | None = None,
    policy: RuntimePolicy | None = None,
) -> AgentRuntime:
    runtime_policy = policy or RuntimePolicy()
    repository = AcademicRepository(database_path)
    academic = AcademicTools(repository)
    registry = standard_registry(academic, runtime_policy)
    tracer = OpenTelemetryTracer()

    def normalizer(
        draft: UnderstandingDraft,
        question: str,
        *,
        inherited_program_id: str | None = None,
        inherited_cohort: int | None = None,
    ) -> NormalizedQuery:
        return normalize(
            draft,
            question,
            repository,
            inherited_program_id=inherited_program_id,
            inherited_cohort=inherited_cohort,
        )

    return AgentRuntime(RuntimeDependencies(
        understanding=QuestionUnderstanding(model), normalizer=normalizer, planner=build_plan,
        executor=PlanExecutor(registry, runtime_policy), synthesizer=DeterministicSynthesizer(),
        validator=ClaimValidator(), sessions=InMemoryTTLSessionStore(dataset_version=repository.metadata().get("dataset_version", "unknown")),
        tracer=tracer, repository=repository, max_validation_retries=runtime_policy.max_validation_retries,
    ))


__all__ = ["build_runtime"]
