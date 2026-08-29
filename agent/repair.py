"""Bounded, reason-driven evidence repair planning."""

from __future__ import annotations

from agent.coverage_gate import CoverageDecision
from evidence.provenance import stable_id
from query.schemas import ExecutionPlan, NormalizedQuery, RetrievePolicyOperation


class RepairPlanner:
    """Create at most one targeted repair plan from an explicit gate decision."""

    max_repair_attempts = 1

    def plan(
        self,
        query: NormalizedQuery,
        original_plan: ExecutionPlan,
        decision: CoverageDecision,
        *,
        retry_count: int,
    ) -> ExecutionPlan | None:
        if retry_count >= self.max_repair_attempts or not decision.repairable:
            return None
        policy_operation = next(
            (
                operation
                for operation in original_plan.operations
                if isinstance(operation, RetrievePolicyOperation)
            ),
            None,
        )
        # A repair refines a policy retrieval that was already selected by the
        # typed planner. It never appends policy search merely because a draft
        # answer was refused.
        if policy_operation is None:
            return None
        widened = policy_operation.args.model_copy(
            update={"top_k": min(100, max(16, policy_operation.args.top_k * 2))}
        )
        operation = policy_operation.model_copy(
            update={
                "operation_id": stable_id("op", original_plan.plan_id, "policy-repair"),
                "depends_on": (),
                "args": widened,
            }
        )
        output_contract = tuple(
            contract.model_copy(
                update={
                    "operation_ids": (operation.operation_id,)
                    if contract.output == "policy_explanation"
                    else contract.operation_ids,
                    "status": "fulfilled"
                    if contract.output == "policy_explanation"
                    else contract.status,
                }
            )
            for contract in original_plan.output_contract
        )
        return ExecutionPlan(
            plan_id=stable_id("plan", original_plan.plan_id, "policy-repair"),
            query=query,
            operations=(operation,),
            output_contract=output_contract,
            rationale=("coverage_gate:policy_support_insufficient",),
        )


__all__ = ["RepairPlanner"]
