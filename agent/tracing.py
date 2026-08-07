"""Portable trace and metric hooks with optional OpenTelemetry integration."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class InMemoryTracer:
    spans: list[dict[str, object]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    metric_events: list[dict[str, object]] = field(default_factory=list)

    @contextmanager
    def start(self, name: str, **attributes: object) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        finally:
            self.spans.append(
                {
                    "name": name,
                    "latency_ms": round((perf_counter() - started) * 1000, 3),
                    **attributes,
                }
            )

    def increment(self, name: str, value: float = 1.0, **attributes: object) -> None:
        """Record a low-cardinality metric without accepting sensitive payloads."""

        self.metrics[name] = self.metrics.get(name, 0.0) + value
        safe = {
            key: item
            for key, item in attributes.items()
            if key in {"intent", "tool_name", "status", "retrieval_mode"}
        }
        self.metric_events.append({"name": name, "value": value, **safe})
