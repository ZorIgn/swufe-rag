"""MCP-facing adapter generated from the same typed ToolRegistry as HTTP."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast, get_type_hints

from pydantic import BaseModel, TypeAdapter

from agent.registry import ToolRegistry
from evidence.provenance import stable_id
from query.schemas import (
    ALL_OPERATION_TYPES,
    AuditCompletedCoursesOperation,
    CheckCurriculumFeasibilityOperation,
    CompareProgramsOperation,
    GetCourseDetailOperation,
    GetGraduationRequirementsOperation,
    GetModuleRequirementsOperation,
    ListCoursesBeforeSemesterOperation,
    ListCoursesOperation,
    ListUnavoidableCoursesOperation,
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
    """Expose typed schemas and execute only operations in the shared registry."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        registered_types = registry.operation_types()
        if registered_types != ALL_OPERATION_TYPES:
            raise ValueError("MCP registry does not cover the canonical operation set")
        self._names = {
            definition.name: definition.operation_type for definition in registry.definitions()
        }
        self._names.update({operation_type: operation_type for operation_type in registered_types})
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
        """Return registry names for callers that use the HTTP tool identifiers."""

        return self.registry.tool_names()

    def call_tool(self, name: str, arguments: Mapping[str, object]) -> dict[str, object]:
        """Validate a tool call into a typed operation and return its packet."""

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
        packet = self.registry.for_operation(operation).execute(operation)
        return packet.model_dump(mode="json")


__all__ = ["MCPAdapter", "MCPToolDefinition"]
