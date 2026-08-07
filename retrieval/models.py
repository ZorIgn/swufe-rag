"""Typed contracts for scoped policy retrieval."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from retrieval.scoring import RetrievedCandidate


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PolicyRetrievalRequest(StrictModel):
    query: str = Field(min_length=1, max_length=4000)
    cohort: int | None = Field(default=None, ge=2010, le=2100)
    program_ids: tuple[str, ...] = ()
    college_ids: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    as_of: str | None = None
    top_k: int = Field(default=8, ge=1, le=100)


class PolicyRetrievalResult(StrictModel):
    candidates: tuple[RetrievedCandidate, ...] = ()
    review_candidates: tuple[RetrievedCandidate, ...] = ()
    scope_filtered_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    retrieval_mode: str
    warnings: tuple[str, ...] = ()


class PolicyRetriever(Protocol):
    """Production policy retriever, independent from SQL repository queries."""

    mode: str

    def retrieve(self, request: PolicyRetrievalRequest) -> PolicyRetrievalResult: ...

    def readiness(self) -> tuple[bool, tuple[str, ...]]: ...


__all__ = ["PolicyRetriever", "PolicyRetrievalRequest", "PolicyRetrievalResult"]
