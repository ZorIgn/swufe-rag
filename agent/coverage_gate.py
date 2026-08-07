"""Answer-eligibility gate over deterministic coverage components."""

from __future__ import annotations

from evidence.models import CoverageComponent, EvidencePacket, StrictModel
from query.schemas import ExecutionPlan, NormalizedQuery


class CoverageDecision(StrictModel):
    sufficient: bool
    repairable: bool
    reasons: tuple[str, ...] = ()
    missing_capabilities: tuple[str, ...] = ()


def _required_components(
    query: NormalizedQuery,
) -> tuple[tuple[str, str], ...]:
    """Return ``(kind, capability)`` pairs required by this answer type."""

    required: list[tuple[str, str]] = []

    def add(kind: str, capability: str) -> None:
        value = (kind, capability)
        if value not in required:
            required.append(value)

    if (
        query.intent in {"course_query", "course_detail"}
        or "course_list" in query.requested_outputs
    ):
        add("course_set", "course_set")
    if (
        query.intent in {"graduation_requirements", "module_requirements"}
        or "module_requirements" in query.requested_outputs
    ):
        add("requirement", "requirements")
    if query.intent == "progress_audit":
        add("audit", "progress_audit")
    if query.intent in {"course_planning", "curriculum_feasibility"}:
        add("requirement", "requirements")
        add("audit", "progress_audit")
        add("course_set", "curriculum_plan")
    if query.intent == "compare_programs":
        add("comparison", "comparison")
    if query.intent == "policy" or "policy_explanation" in query.requested_outputs:
        add("policy", "policy")
    return tuple(required)


def _valid_component(component: CoverageComponent, kind: str) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    if not component.complete:
        reasons.append(f"{kind}_incomplete")
    if component.truncated:
        reasons.append(f"{kind}_truncated")
    if kind in {"requirement", "policy"} and component.trusted_evidence is not True:
        reasons.append(f"{kind}_untrusted_evidence")
    if kind == "policy":
        if component.authoritative is not True:
            reasons.append("policy_not_authoritative")
        if component.scope_matched is not True:
            reasons.append("policy_scope_unmatched")
        if component.version_resolved is not True:
            reasons.append("policy_version_unresolved")
        if component.conflict_free is not True:
            reasons.append("policy_source_conflict")
    return not reasons, tuple(reasons)


class CoverageGate:
    """Fail closed before synthesis when required evidence is incomplete or unsafe."""

    def evaluate(
        self, query: NormalizedQuery, plan: ExecutionPlan, packet: EvidencePacket
    ) -> CoverageDecision:
        reasons: list[str] = []
        missing: list[str] = []
        outcomes = {item.operation_id: item for item in packet.execution_results}
        for operation in plan.operations:
            result = outcomes.get(operation.operation_id)
            if result is None:
                reasons.append(f"operation_not_reported:{operation.operation_id}")
            elif result.status != "success" and not operation.optional:
                reasons.append(f"operation_{result.status}:{operation.operation_id}")

        by_kind: dict[str, list[CoverageComponent]] = {}
        for component in packet.coverage.components:
            by_kind.setdefault(component.kind, []).append(component)
        for kind, capability in _required_components(query):
            components = by_kind.get(kind, [])
            if not components:
                missing.append(capability)
                reasons.append(f"coverage_missing:{capability}")
                continue
            for component in components:
                valid, component_reasons = _valid_component(component, kind)
                if not valid:
                    reasons.extend(component_reasons)

        if query.unmatched_completed_courses:
            reasons.append("completed_course_unmatched")
        if query.ambiguous_completed_courses:
            reasons.append("completed_course_ambiguous")
        # Only a retrieval support shortage is repairable. Scope/version conflict,
        # incomplete structured sets, and entity ambiguity must not trigger a random
        # second search.
        repairable = bool(
            ("policy" in missing or "policy_incomplete" in reasons)
            and not any(
                value in reasons
                for value in (
                    "policy_source_conflict",
                    "policy_version_unresolved",
                    "policy_scope_unmatched",
                    "policy_not_authoritative",
                    "policy_untrusted_evidence",
                )
            )
        )
        return CoverageDecision(
            sufficient=not reasons,
            repairable=repairable,
            reasons=tuple(dict.fromkeys(reasons)),
            missing_capabilities=tuple(dict.fromkeys(missing)),
        )


__all__ = ["CoverageDecision", "CoverageGate"]
