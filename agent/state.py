"""Bounded single-agent state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from evidence.models import (
    ClaimSpan,
    ClaimValidation,
    EvidencePacket,
    FinalAnswer,
    ToolExecutionResult,
)
from query.schemas import ExecutionPlan, NormalizedQuery, OutputContract, UnderstandingDraft


class AgentStatus(str, Enum):
    UNDERSTAND = "UNDERSTAND"
    NORMALIZE = "NORMALIZE"
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    COVERAGE_CHECK = "COVERAGE_CHECK"
    REPAIR_PLAN = "REPAIR_PLAN"
    REPAIR_EXECUTE = "REPAIR_EXECUTE"
    SYNTHESIZE = "SYNTHESIZE"
    VALIDATE = "VALIDATE"
    REGENERATE = "REGENERATE"
    CLARIFY = "CLARIFY"
    REFUSE = "REFUSE"
    FINISH = "FINISH"


ALLOWED_TRANSITIONS = {
    AgentStatus.UNDERSTAND: {AgentStatus.NORMALIZE},
    AgentStatus.NORMALIZE: {AgentStatus.PLAN, AgentStatus.CLARIFY},
    AgentStatus.PLAN: {AgentStatus.EXECUTE, AgentStatus.CLARIFY},
    AgentStatus.EXECUTE: {AgentStatus.COVERAGE_CHECK},
    AgentStatus.COVERAGE_CHECK: {
        AgentStatus.SYNTHESIZE,
        AgentStatus.REPAIR_PLAN,
        AgentStatus.CLARIFY,
        AgentStatus.REFUSE,
    },
    AgentStatus.REPAIR_PLAN: {AgentStatus.REPAIR_EXECUTE, AgentStatus.REFUSE},
    AgentStatus.REPAIR_EXECUTE: {AgentStatus.COVERAGE_CHECK},
    AgentStatus.SYNTHESIZE: {AgentStatus.VALIDATE},
    AgentStatus.VALIDATE: {AgentStatus.FINISH, AgentStatus.REGENERATE, AgentStatus.REFUSE},
    AgentStatus.REGENERATE: {AgentStatus.VALIDATE},
    AgentStatus.CLARIFY: {AgentStatus.FINISH},
    AgentStatus.REFUSE: {AgentStatus.FINISH},
    AgentStatus.FINISH: set(),
}


@dataclass
class AgentState:
    request_id: str
    session_id: str | None
    raw_question: str
    status: AgentStatus = AgentStatus.UNDERSTAND
    understanding: UnderstandingDraft | None = None
    normalized_query: NormalizedQuery | None = None
    plan: ExecutionPlan | None = None
    output_contracts: list[OutputContract] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    tool_results: list[ToolExecutionResult] = field(default_factory=list)
    evidence: EvidencePacket | None = None
    answer: FinalAnswer | None = None
    claims: list[ClaimSpan] = field(default_factory=list)
    validation: list[ClaimValidation] = field(default_factory=list)
    repair_count: int = 0
    regeneration_count: int = 0

    def transition(self, target: AgentStatus) -> None:
        if target not in ALLOWED_TRANSITIONS[self.status]:
            raise RuntimeError(f"invalid agent transition: {self.status} -> {target}")
        self.status = target
