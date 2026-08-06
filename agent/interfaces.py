"""Explicit dependency interfaces for :class:`agent.orchestrator.AgentRuntime`."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol

from evidence.models import EvidencePacket, FinalAnswer
from query.schemas import ExecutionPlan, NormalizedQuery, UnderstandingDraft


class QuestionUnderstanding(Protocol):
    def understand(self, question: str) -> UnderstandingDraft: ...


class QueryNormalizer(Protocol):
    def __call__(self, draft: UnderstandingDraft, question: str, *, inherited_program_id: str | None = None, inherited_cohort: int | None = None) -> NormalizedQuery: ...


class ExecutionPlanner(Protocol):
    def __call__(self, query: NormalizedQuery) -> ExecutionPlan: ...


class ToolExecutor(Protocol):
    def execute(self, plan: ExecutionPlan) -> EvidencePacket: ...


class Retriever(Protocol):
    def retrieve(self, question: str) -> EvidencePacket: ...


class AnswerSynthesizer(Protocol):
    def synthesize(self, query: NormalizedQuery, packet: EvidencePacket) -> FinalAnswer: ...


class AnswerValidator(Protocol):
    def validate(self, answer: FinalAnswer, packet: EvidencePacket) -> FinalAnswer: ...


class SessionStore(Protocol):
    def get(self, session_id: str) -> dict[str, object] | None: ...

    def put(self, session_id: str, value: dict[str, object]) -> None: ...


class Tracer(Protocol):
    def start(self, name: str, **attributes: object) -> AbstractContextManager[None]: ...
