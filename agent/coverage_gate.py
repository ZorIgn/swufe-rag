"""Answer-eligibility gate over per-operation coverage contracts."""

from __future__ import annotations

from evidence.models import (
    CoverageComponent,
    CoverageKind,
    EvidencePacket,
    StrictModel,
    ToolExecutionResult,
)
from query.schemas import ExecutionPlan, NormalizedQuery, Operation, OutputContract


class CoverageDecision(StrictModel):
    sufficient: bool
    repairable: bool
    reasons: tuple[str, ...] = ()
    missing_capabilities: tuple[str, ...] = ()
    output_statuses: tuple[OutputContract, ...] = ()


_CAPABILITY_KINDS: dict[str, frozenset[CoverageKind]] = {
    "course_set": frozenset({"course_set"}),
    "requirements": frozenset({"requirement"}),
    "policy": frozenset({"policy"}),
    "progress_audit": frozenset({"audit"}),
    "comparison": frozenset({"comparison"}),
    "curriculum_plan": frozenset({"course_set"}),
}

_OPERATION_KIND: dict[str, CoverageKind] = {
    "list_courses": "course_set",
    "get_course_detail": "course_set",
    "get_graduation_requirements": "requirement",
    "get_module_requirements": "requirement",
    "audit_completed_courses": "audit",
    "list_courses_before_semester": "course_set",
    "list_unavoidable_courses": "course_set",
    "check_curriculum_feasibility": "course_set",
    "retrieve_policy": "policy",
    "compare_programs": "comparison",
    "resolve_source": "source",
}


def _valid_component(component: CoverageComponent) -> tuple[bool, tuple[str, ...]]:
    """Validate one producer's own coverage; never aggregate by broad kind."""

    kind = component.kind
    reasons: list[str] = []
    if not component.complete:
        reasons.append(f"{kind}_incomplete")
    if component.truncated:
        reasons.append(f"{kind}_truncated")
    if kind != "source" and component.trusted_evidence is not True:
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


def _severe(reasons: tuple[str, ...] | list[str]) -> bool:
    markers = (
        "authoritative",
        "conflict",
        "contract",
        "duplicate",
        "kind_mismatch",
        "not_in_plan",
        "scope",
        "tool_mismatch",
        "unexpected",
        "untrusted",
        "version",
    )
    return any(marker in reason for reason in reasons for marker in markers)


def _with_runtime_status(
    contract: OutputContract,
    *,
    status: str,
    reasons: tuple[str, ...],
) -> OutputContract:
    return contract.model_copy(update={"status": status, "reasons": reasons})


class CoverageGate:
    """Fail closed per requested output while permitting safe partial answers.

    Every capability is checked only against the concrete producer operation
    IDs named by its :class:`OutputContract`.  A successful operation of the
    same broad kind cannot satisfy a different requested output.
    """

    def evaluate(
        self, query: NormalizedQuery, plan: ExecutionPlan, packet: EvidencePacket
    ) -> CoverageDecision:
        global_reasons: list[str] = []
        reported_reasons: list[str] = []
        missing: list[str] = []

        operations_by_id: dict[str, Operation] = {}
        for operation in plan.operations:
            if operation.operation_id in operations_by_id:
                global_reasons.append(f"duplicate_plan_operation:{operation.operation_id}")
            else:
                operations_by_id[operation.operation_id] = operation

        outcomes_by_id: dict[str, list[ToolExecutionResult]] = {}
        for outcome in packet.execution_results:
            outcomes_by_id.setdefault(outcome.operation_id, []).append(outcome)
            if outcome.operation_id not in operations_by_id:
                global_reasons.append(
                    f"unexpected_execution_result:{outcome.operation_id}"
                )
        for operation_id, outcomes in outcomes_by_id.items():
            if len(outcomes) > 1:
                global_reasons.append(f"duplicate_execution_result:{operation_id}")

        components_by_id: dict[str, list[CoverageComponent]] = {}
        for component in packet.coverage.components:
            components_by_id.setdefault(component.operation_id, []).append(component)
            if component.operation_id not in operations_by_id:
                global_reasons.append(
                    f"unexpected_coverage_component:{component.operation_id}"
                )
        for operation_id, components in components_by_id.items():
            if len(components) > 1:
                global_reasons.append(f"duplicate_coverage_component:{operation_id}")
        global_reasons.extend(
            warning
            for warning in packet.warnings
            if warning.startswith("duplicate_coverage_component:")
        )

        contracts_by_output: dict[str, list[OutputContract]] = {}
        for contract in plan.output_contract:
            contracts_by_output.setdefault(contract.output, []).append(contract)
            if contract.output not in query.requested_outputs:
                global_reasons.append(f"unexpected_output_contract:{contract.output}")

        output_statuses: list[OutputContract] = []
        fulfilled_count = 0
        repairable_policy_shortage = False

        for output in tuple(dict.fromkeys(query.requested_outputs)):
            matches = contracts_by_output.get(output, [])
            if not matches:
                contract = OutputContract(
                    output=output,
                    status="missing_data",
                    reasons=("plan_output_contract_missing",),
                )
                output_statuses.append(contract)
                missing.append(output)
                reported_reasons.append(f"output_missing_data:{output}:plan_output_contract_missing")
                continue
            if len(matches) > 1:
                reason = f"duplicate_output_contract:{output}"
                global_reasons.append(reason)
                contract = matches[0]
                output_statuses.append(
                    _with_runtime_status(
                        contract, status="refused", reasons=(reason,)
                    )
                )
                missing.append(output)
                continue

            contract = matches[0]
            if contract.status in {"unsupported", "refused", "missing_data"}:
                output_statuses.append(contract)
                missing.append(output)
                reasons = contract.reasons or (f"output_{contract.status}:{output}",)
                reported_reasons.extend(
                    f"output_{contract.status}:{output}:{reason}" for reason in reasons
                )
                continue

            output_reasons: list[str] = []
            if not contract.operation_ids:
                output_reasons.append("producer_operation_missing")
            if len(set(contract.operation_ids)) != len(contract.operation_ids):
                output_reasons.append("duplicate_contract_operation_id")

            producer_components: list[tuple[CoverageComponent, bool]] = []
            for operation_id in contract.operation_ids:
                producer = operations_by_id.get(operation_id)
                if producer is None:
                    output_reasons.append(
                        f"producer_operation_not_in_plan:{operation_id}"
                    )
                    continue

                outcomes = outcomes_by_id.get(operation_id, [])
                if len(outcomes) != 1:
                    output_reasons.append(
                        f"operation_not_reported:{operation_id}"
                        if not outcomes
                        else f"duplicate_execution_result:{operation_id}"
                    )
                    continue
                outcome = outcomes[0]
                if outcome.tool_name != producer.tool_name:
                    output_reasons.append(
                        f"execution_tool_mismatch:{operation_id}"
                    )
                if outcome.status != "success":
                    output_reasons.append(
                        f"operation_{outcome.status}:{operation_id}"
                    )

                components = components_by_id.get(operation_id, [])
                if len(components) != 1:
                    output_reasons.append(
                        f"coverage_component_missing:{operation_id}"
                        if not components
                        else f"duplicate_coverage_component:{operation_id}"
                    )
                    continue
                component = components[0]
                if component.tool_name != producer.tool_name:
                    output_reasons.append(
                        f"coverage_tool_mismatch:{operation_id}"
                    )
                expected_kind = _OPERATION_KIND.get(producer.type)
                if expected_kind is None or component.kind != expected_kind:
                    output_reasons.append(
                        f"coverage_kind_mismatch:{operation_id}"
                    )
                valid, component_reasons = _valid_component(component)
                output_reasons.extend(component_reasons)
                producer_components.append((component, valid))

            for capability in contract.capabilities:
                accepted_kinds = _CAPABILITY_KINDS.get(capability)
                if not accepted_kinds:
                    output_reasons.append(
                        f"unknown_output_capability:{capability}"
                    )
                    continue
                candidates = [
                    valid
                    for component, valid in producer_components
                    if component.kind in accepted_kinds
                ]
                if not candidates:
                    output_reasons.append(
                        f"output_capability_missing:{capability}"
                    )
                elif not any(candidates):
                    output_reasons.append(
                        f"output_capability_unsatisfied:{capability}"
                    )

            unique_reasons = tuple(dict.fromkeys(output_reasons))
            if unique_reasons:
                status = "refused" if _severe(unique_reasons) else "missing_data"
                output_statuses.append(
                    _with_runtime_status(
                        contract, status=status, reasons=unique_reasons
                    )
                )
                missing.append(output)
                reported_reasons.extend(
                    f"output_{status}:{output}:{reason}" for reason in unique_reasons
                )
                if (
                    output == "policy_explanation"
                    and status == "missing_data"
                    and "policy_incomplete" in unique_reasons
                ):
                    repairable_policy_shortage = True
            else:
                fulfilled_count += 1
                output_statuses.append(
                    _with_runtime_status(contract, status="fulfilled", reasons=())
                )

        reasons = tuple(dict.fromkeys((*global_reasons, *reported_reasons)))
        sufficient = fulfilled_count > 0 and not global_reasons
        return CoverageDecision(
            sufficient=sufficient,
            repairable=(
                not sufficient
                and not global_reasons
                and repairable_policy_shortage
            ),
            reasons=reasons,
            missing_capabilities=tuple(dict.fromkeys(missing)),
            output_statuses=tuple(output_statuses),
        )


__all__ = ["CoverageDecision", "CoverageGate"]
