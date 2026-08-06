"""Single registry shared by HTTP Agent and MCP adapter."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

from evidence.models import EvidencePacket
from query.schemas import Operation

T = TypeVar("T", bound=Operation, contravariant=True)


@dataclass(frozen=True)
class RegisteredTool(Generic[T]):
    name: str
    operation_type: str
    read_only: bool
    timeout_seconds: float
    execute: Callable[[T], EvidencePacket]


class ToolRegistry:
    def __init__(self) -> None:
        self._by_type: dict[str, RegisteredTool[Operation]] = {}
        self._by_name: dict[str, RegisteredTool[Operation]] = {}

    def register(self, tool: RegisteredTool[T]) -> None:
        if tool.operation_type in self._by_type or tool.name in self._by_name:
            raise ValueError(f"duplicate tool registration: {tool.name}")
        registered = cast(RegisteredTool[Operation], tool)
        self._by_type[tool.operation_type] = registered
        self._by_name[tool.name] = registered

    def for_operation(self, operation: Operation) -> RegisteredTool[Operation]:
        try:
            return self._by_type[operation.type]
        except KeyError as exc:
            raise ValueError(f"unsupported operation: {operation.type}") from exc

    def tool_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_name))

    def operation_types(self) -> frozenset[str]:
        return frozenset(self._by_type)

    def definitions(self) -> tuple[RegisteredTool[Operation], ...]:
        return tuple(self._by_name[name] for name in sorted(self._by_name))
