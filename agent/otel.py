"""Optional OpenTelemetry bridge; safe no-op fallback for local/offline use."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from agent.tracing import InMemoryTracer


class OpenTelemetryTracer(InMemoryTracer):
    def __init__(self) -> None:
        super().__init__()
        try:
            from opentelemetry import trace
        except ImportError:
            self._tracer = None
        else:  # pragma: no cover - optional deployment integration
            self._tracer = trace.get_tracer("swufe-rag")

    @contextmanager
    def start(self, name: str, **attributes: object) -> Iterator[None]:
        if self._tracer is None:
            with super().start(name, **attributes):
                yield
            return
        with self._tracer.start_as_current_span(name) as span:  # pragma: no cover
            for key, value in attributes.items():
                if "key" not in key.lower() and "secret" not in key.lower():
                    span.set_attribute(key, str(value))
            yield
