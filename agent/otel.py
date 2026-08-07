"""Optional OpenTelemetry bridge; safe no-op fallback for local/offline use."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from agent.tracing import InMemoryTracer


class OpenTelemetryTracer(InMemoryTracer):
    def __init__(self) -> None:
        super().__init__()
        self._tracer: object | None = None
        self._meter: object | None = None
        self._counters: dict[str, object] = {}
        try:
            from opentelemetry import metrics, trace
        except ImportError:
            return
        self._tracer = trace.get_tracer("swufe-rag")
        self._meter = metrics.get_meter("swufe-rag")

    @contextmanager
    def start(self, name: str, **attributes: object) -> Iterator[None]:
        with super().start(name, **attributes):
            if self._tracer is None:
                yield
                return
            # The SDK's span object is optional; all local traces still remain
            # observable through the parent context manager.
            with self._tracer.start_as_current_span(name) as span:  # type: ignore[union-attr]
                for key, value in attributes.items():
                    if key in {
                        "intent",
                        "tool_name",
                        "status",
                        "retrieval_mode",
                        "request_id",
                        "plan_id",
                    }:
                        span.set_attribute(key, str(value))
                yield

    def increment(self, name: str, value: float = 1.0, **attributes: object) -> None:
        super().increment(name, value, **attributes)
        if self._meter is None:
            return
        counter = self._counters.get(name)
        if counter is None:
            counter = self._meter.create_counter(name)  # type: ignore[union-attr]
            self._counters[name] = counter
        labels = {
            key: str(item)
            for key, item in attributes.items()
            if key in {"intent", "tool_name", "status", "retrieval_mode"}
        }
        counter.add(value, labels)  # type: ignore[union-attr]
