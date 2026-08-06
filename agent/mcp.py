"""MCP-facing adapter generated from the same ToolRegistry as HTTP execution."""

from __future__ import annotations

from dataclasses import dataclass

from agent.registry import ToolRegistry


@dataclass(frozen=True)
class MCPToolDefinition:
    name: str
    description: str
    side_effect_free: bool = True


class MCPAdapter:
    """A small contract adapter; transport hosts can expose it as an MCP server."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def list_tools(self) -> tuple[MCPToolDefinition, ...]:
        names = {
            "policy.search": "search_policy",
            "academic.list_courses": "list_courses",
            "academic.get_course": "get_course_detail",
            "academic.get_requirements": "get_graduation_requirements",
            "academic.audit_progress": "audit_academic_progress",
            "academic.compare_programs": "compare_programs",
            "source.resolve": "resolve_source",
        }
        return tuple(MCPToolDefinition(name=names.get(tool.name, tool.name.replace(".", "_")), description=f"Read-only {tool.name}") for tool in self.registry.definitions())

    def schema_names(self) -> tuple[str, ...]:
        return self.registry.tool_names()


__all__ = ["MCPAdapter", "MCPToolDefinition"]
