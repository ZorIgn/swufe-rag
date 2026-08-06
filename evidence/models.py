"""Strict, provider-neutral evidence and claim schemas.

The runtime never passes untyped dictionaries between agent stages.  Facts and
their provenance are first-class objects, so an answer can be validated without
asking an LLM to reconstruct where a number or course code came from.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Provenance(StrictModel):
    record_id: str
    source_id: str
    chunk_id: str | None = None
    physical_page: int | None = Field(default=None, ge=1)
    parser_version: str
    source_sha256: str | None = None
    extracted_at: datetime
    effective_from: str | None = None
    effective_to: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    review_status: Literal["verified", "review_required", "unverified"]


class Evidence(StrictModel):
    evidence_id: str
    source_id: str
    chunk_id: str | None = None
    title: str
    article: str | None = None
    quote: str
    page_url: str | None = None
    file_url: str | None = None
    provenance: Provenance


FactValue = str | int | float | bool | list[str]


class Fact(StrictModel):
    fact_id: str
    type: str
    subject: str
    predicate: str
    value: FactValue
    unit: str | None = None
    source_record_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    derivation: Literal["observed", "retrieved", "tool_result", "derived"] = "observed"


class DerivedFact(Fact):
    derivation: Literal["derived"] = "derived"
    operator: Literal["sum", "difference", "intersection", "set_difference", "count"]
    input_fact_ids: tuple[str, ...]


class ProgramCoverage(StrictModel):
    requested_program_id: str | None = None
    resolved: bool = False
    dataset_complete: bool = False


class FieldCoverage(StrictModel):
    field: str
    covered: bool
    source_record_count: int = Field(ge=0)
    reason: str | None = None


class CourseSetCoverage(StrictModel):
    database_match_count: int = Field(ge=0)
    returned_count: int = Field(ge=0)
    truncated: bool = False
    filters_applied: tuple[str, ...] = ()
    dataset_complete: bool = False
    classification_complete: bool = True

    @property
    def complete(self) -> bool:
        return (
            self.database_match_count == self.returned_count
            and not self.truncated
            and self.dataset_complete
            and self.classification_complete
        )


class PolicyCoverage(StrictModel):
    support_sufficient: bool = False
    source_authoritative: bool = False
    scope_matched: bool = False
    version_resolved: bool = False
    conflict_free: bool = False


class Coverage(StrictModel):
    program: ProgramCoverage = Field(default_factory=ProgramCoverage)
    fields: tuple[FieldCoverage, ...] = ()
    course_set: CourseSetCoverage | None = None
    policy: PolicyCoverage | None = None


class ClaimDraft(StrictModel):
    claim_id: str
    text: str = Field(min_length=1)
    fact_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()


class ClaimValidation(StrictModel):
    claim_id: str
    passed: bool
    reasons: tuple[str, ...] = ()


class ClaimSpan(StrictModel):
    text: str
    fact_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    validation: ClaimValidation


class EvidencePacket(StrictModel):
    packet_id: str
    facts: tuple[Fact | DerivedFact, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    coverage: Coverage = Field(default_factory=Coverage)
    conflicts: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    tool_results: tuple[str, ...] = ()

    def fact(self, fact_id: str) -> Fact | DerivedFact | None:
        return next((value for value in self.facts if value.fact_id == fact_id), None)

    def evidence_by_id(self, evidence_id: str) -> Evidence | None:
        return next((value for value in self.evidence if value.evidence_id == evidence_id), None)


ValidationOutcome = Literal["pass", "insufficient_evidence", "missing_information"]


class FinalAnswer(StrictModel):
    answer_md: str
    claims: tuple[ClaimSpan, ...]
    citations: tuple[Evidence, ...]
    refused: bool = False
    clarification: str | None = None


__all__ = [
    "ClaimDraft",
    "ClaimSpan",
    "ClaimValidation",
    "CourseSetCoverage",
    "Coverage",
    "DerivedFact",
    "Evidence",
    "EvidencePacket",
    "Fact",
    "FieldCoverage",
    "FinalAnswer",
    "PolicyCoverage",
    "ProgramCoverage",
    "Provenance",
    "ValidationOutcome",
]
