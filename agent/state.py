"""Bounded single-agent state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from evidence.models import ClaimSpan, ClaimValidation, EvidencePacket, FinalAnswer
from query.schemas import ExecutionPlan, NormalizedQuery, UnderstandingDraft


class AgentStatus(str, Enum):
    UNDERSTAND = "UNDERSTAND"
    NORMALIZE = "NORMALIZE"
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    SYNTHESIZE = "SYNTHESIZE"
    VALIDATE = "VALIDATE"
    TARGETED_RETRIEVAL = "TARGETED_RETRIEVAL"
    CLARIFY = "CLARIFY"
    FINISH = "FINISH"


ALLOWED_TRANSITIONS = {
    AgentStatus.UNDERSTAND: {AgentStatus.NORMALIZE},
    AgentStatus.NORMALIZE: {AgentStatus.PLAN, AgentStatus.CLARIFY},
    AgentStatus.PLAN: {AgentStatus.EXECUTE, AgentStatus.CLARIFY},
    AgentStatus.EXECUTE: {AgentStatus.SYNTHESIZE},
    AgentStatus.SYNTHESIZE: {AgentStatus.VALIDATE},
    AgentStatus.VALIDATE: {AgentStatus.FINISH, AgentStatus.TARGETED_RETRIEVAL, AgentStatus.CLARIFY},
    AgentStatus.TARGETED_RETRIEVAL: {AgentStatus.SYNTHESIZE},
    AgentStatus.CLARIFY: {AgentStatus.FINISH},
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
    tool_calls: list[str] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)
    evidence: EvidencePacket | None = None
    answer: FinalAnswer | None = None
    claims: list[ClaimSpan] = field(default_factory=list)
    validation: list[ClaimValidation] = field(default_factory=list)
    retry_count: int = 0

    def transition(self, target: AgentStatus) -> None:
        if target not in ALLOWED_TRANSITIONS[self.status]:
            raise RuntimeError(f"invalid agent transition: {self.status} -> {target}")
        self.status = target
