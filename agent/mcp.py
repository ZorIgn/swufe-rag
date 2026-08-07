"""MCP-compatible typed adapter governed by the same PlanExecutor as HTTP."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast, get_type_hints

from pydantic import BaseModel, TypeAdapter

from agent.policies import RuntimePolicy
from agent.registry import ToolRegistry
from agent.tools import PlanExecutor
from evidence.provenance import stable_id
from query.schemas import (
    ALL_OPERATION_TYPES,
    AuditCompletedCoursesOperation,
    CheckCurriculumFeasibilityOperation,
    CompareProgramsOperation,
    ExecutionPlan,
    GetCourseDetailOperation,
    GetGraduationRequirementsOperation,
    GetModuleRequirementsOperation,
    ListCoursesBeforeSemesterOperation,
    ListCoursesOperation,
    ListUnavoidableCoursesOperation,
    NormalizedQuery,
    Operation,
    ResolveSourceOperation,
    RetrievePolicyOperation,
)

_OPERATION_MODELS: dict[str, type[BaseModel]] = {
    "list_courses": ListCoursesOperation,
    "get_course_detail": GetCourseDetailOperation,
    "get_graduation_requirements": GetGraduationRequirementsOperation,
    "get_module_requirements": GetModuleRequirementsOperation,
    "audit_completed_courses": AuditCompletedCoursesOperation,
    "list_courses_before_semester": ListCoursesBeforeSemesterOperation,
    "list_unavoidable_courses": ListUnavoidableCoursesOperation,
    "check_curriculum_feasibility": CheckCurriculumFeasibilityOperation,
    "retrieve_policy": RetrievePolicyOperation,
    "compare_programs": CompareProgramsOperation,
    "resolve_source": ResolveSourceOperation,
}

_MCP_NAMES: dict[str, str] = {
    "list_courses": "list_courses",
    "get_course_detail": "get_course_detail",
    "get_graduation_requirements": "get_graduation_requirements",
    "get_module_requirements": "get_module_requirements",
    "audit_completed_courses": "audit_academic_progress",
    "list_courses_before_semester": "list_courses_before_semester",
    "list_unavoidable_courses": "list_unavoidable_courses",
    "check_curriculum_feasibility": "check_curriculum_feasibility",
    "retrieve_policy": "search_policy",
    "compare_programs": "compare_programs",
    "resolve_source": "resolve_source",
}


@dataclass(frozen=True)
class MCPToolDefinition:
    name: str
    operation_type: str
    description: str
    input_schema: dict[str, Any]
    side_effect_free: bool = True


class MCPAdapter:
    """Validate MCP arguments, create a one-operation plan, and execute it."""

    def __init__(self, executor: PlanExecutor | ToolRegistry) -> None:
        # Passing a runtime PlanExecutor is the production path. The registry
        # fallback is useful for isolated callers and still routes through a
        # newly constructed PlanExecutor rather than invoking a callback directly.
        if isinstance(executor, PlanExecutor):
            self.executor = executor
        else:
            self.executor = PlanExecutor(executor, RuntimePolicy())
        self.registry = self.executor.registry
        if self.registry.operation_types() != ALL_OPERATION_TYPES:
            raise ValueError("MCP registry does not cover the canonical operation set")
        self._names = {
            definition.name: definition.operation_type for definition in self.registry.definitions()
        }
        self._names.update(
            {operation_type: operation_type for operation_type in ALL_OPERATION_TYPES}
        )
        self._names.update({name: operation_type for operation_type, name in _MCP_NAMES.items()})

    def list_tools(self) -> tuple[MCPToolDefinition, ...]:
        definitions = {
            definition.operation_type: definition for definition in self.registry.definitions()
        }
        tools: list[MCPToolDefinition] = []
        for operation_type in sorted(_OPERATION_MODELS):
            operation_model = _OPERATION_MODELS[operation_type]
            args_type = get_type_hints(operation_model)["args"]
            tools.append(
                MCPToolDefinition(
                    name=_MCP_NAMES[operation_type],
                    operation_type=operation_type,
                    description=f"Read-only {definitions[operation_type].name}",
                    input_schema=TypeAdapter(args_type).json_schema(),
                )
            )
        return tuple(tools)

    def schema_names(self) -> tuple[str, ...]:
        return self.registry.tool_names()

    def call_tool(self, name: str, arguments: Mapping[str, object]) -> dict[str, object]:
        if not isinstance(arguments, Mapping):
            raise ValueError("MCP arguments must be an object")
        operation_type = self._names.get(name)
        if operation_type is None:
            raise ValueError(f"unknown MCP tool: {name}")
        operation_model = _OPERATION_MODELS[operation_type]
        canonical_arguments = json.dumps(
            dict(arguments), ensure_ascii=False, sort_keys=True, default=str
        )
        operation = cast(
            Operation,
            operation_model.model_validate(
                {
                    "operation_id": stable_id("mcp", operation_type, canonical_arguments),
                    "args": dict(arguments),
                }
            ),
        )
        query = NormalizedQuery(
            raw_question=f"MCP:{operation_type}",
            intent="general",
            information_scope="unknown",
        )
        plan = ExecutionPlan(
            plan_id=stable_id("mcp-plan", operation.operation_id),
            query=query,
            operations=(operation,),
        )
        packet = self.executor.execute(plan)
        return packet.model_dump(mode="json")


__all__ = ["MCPAdapter", "MCPToolDefinition"]
