"""Typed facts, provenance, coverage, and execution evidence.

The agent never moves an untyped dictionary between stages. Coverage is reported
per tool operation, so concurrent completion order cannot alter an
answer-eligibility decision.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceTrust(str, Enum):
    VERIFIED = "verified"
    REVIEW_REQUIRED = "review_required"
    UNVERIFIED = "unverified"


FactRole = Literal["factual", "non_factual", "metadata"]
ClaimComparator = Literal[
    "equals",
    "contains",
    "at_least",
    "at_most",
    "before",
    "after",
    "satisfies",
]


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
    review_status: EvidenceTrust


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


def _comparator_value_is_valid(comparator: ClaimComparator, value: FactValue) -> bool:
    if comparator in {"at_least", "at_most"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if comparator == "contains":
        return isinstance(value, list)
    if comparator in {"before", "after"}:
        return isinstance(value, (str, int, float)) and not isinstance(value, bool)
    return True


class Fact(StrictModel):
    fact_id: str
    type: str
    subject: str
    predicate: str
    value: FactValue
    comparator: ClaimComparator = "equals"
    unit: str | None = None
    source_record_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    # Facts are factual unless an author deliberately classifies them as
    # process metadata or a non-factual diagnostic.  This is deliberately not
    # inferred from ``type``: adding a new fact type must not accidentally
    # create an evidence bypass.
    role: FactRole = "factual"
    conditions: tuple[str, ...] = ()
    exceptions: tuple[str, ...] = ()
    scope: str | None = None
    temporal: str | None = None
    derivation: Literal["observed", "retrieved", "tool_result", "derived"] = "observed"

    @model_validator(mode="after")
    def comparator_matches_value(self) -> Fact:
        if not _comparator_value_is_valid(self.comparator, self.value):
            raise ValueError(
                f"comparator {self.comparator!r} is incompatible with fact value"
            )
        return self


class DerivedFact(Fact):
    derivation: Literal["derived"] = "derived"
    operator: Literal[
        "sum",
        "difference",
        "intersection",
        "set_difference",
        "count",
        "rule_evaluation",
        "threshold_check",
        "all_of",
        "any_of",
    ]
    input_fact_ids: tuple[str, ...]


class ClaimAtom(StrictModel):
    """One auditable semantic assertion inside a rendered claim.

    Text remains useful for human readers, but validators must bind each
    statement to an atom.  The atom makes numeric values, units, qualifiers,
    scope and temporal boundaries explicit rather than treating them as a bag
    of words found somewhere in the packet.
    """

    subject: str
    predicate: str
    comparator: ClaimComparator = "equals"
    value: FactValue
    unit: str | None = None
    conditions: tuple[str, ...] = ()
    exceptions: tuple[str, ...] = ()
    scope: str | None = None
    temporal: str | None = None
    fact_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def comparator_matches_value(self) -> ClaimAtom:
        if not _comparator_value_is_valid(self.comparator, self.value):
            raise ValueError(
                f"comparator {self.comparator!r} is incompatible with claim value"
            )
        return self


CoverageKind = Literal[
    "course_set",
    "requirement",
    "policy",
    "comparison",
    "audit",
    "source",
]


class CoverageComponent(StrictModel):
    operation_id: str
    tool_name: str
    kind: CoverageKind
    complete: bool
    expected_count: int | None = Field(default=None, ge=0)
    returned_count: int | None = Field(default=None, ge=0)
    truncated: bool = False
    authoritative: bool | None = None
    scope_matched: bool | None = None
    version_resolved: bool | None = None
    conflict_free: bool | None = None
    trusted_evidence: bool | None = None
    reasons: tuple[str, ...] = ()


class CoverageReport(StrictModel):
    """Deterministic per-operation coverage collection."""

    components: tuple[CoverageComponent, ...] = ()

    def for_operation(self, operation_id: str) -> CoverageComponent | None:
        return next((item for item in self.components if item.operation_id == operation_id), None)

    def for_kind(self, kind: CoverageKind) -> tuple[CoverageComponent, ...]:
        return tuple(item for item in self.components if item.kind == kind)


class ToolExecutionResult(StrictModel):
    operation_id: str
    tool_name: str
    status: Literal["success", "timeout", "failed", "dependency_failed", "skipped"]
    latency_ms: float = Field(ge=0.0)
    error_code: str | None = None


class ClaimDraft(StrictModel):
    claim_id: str
    text: str = Field(min_length=1)
    fact_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    atoms: tuple[ClaimAtom, ...] = ()


class ClaimValidation(StrictModel):
    claim_id: str
    passed: bool
    reasons: tuple[str, ...] = ()


class ClaimSpan(StrictModel):
    text: str
    fact_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    atoms: tuple[ClaimAtom, ...] = ()
    validation: ClaimValidation


class EvidencePacket(StrictModel):
    packet_id: str
    facts: tuple[Fact | DerivedFact, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    coverage: CoverageReport = Field(default_factory=CoverageReport)
    execution_results: tuple[ToolExecutionResult, ...] = ()
    conflicts: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

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
    "ClaimAtom",
    "ClaimComparator",
    "ClaimSpan",
    "ClaimValidation",
    "CoverageComponent",
    "CoverageKind",
    "CoverageReport",
    "DerivedFact",
    "Evidence",
    "EvidencePacket",
    "EvidenceTrust",
    "Fact",
    "FactRole",
    "FinalAnswer",
    "Provenance",
    "ToolExecutionResult",
    "ValidationOutcome",
]
