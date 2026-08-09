"""Optional OpenTelemetry bridge; safe no-op fallback for local/offline use."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from typing import Protocol, cast

from agent.tracing import InMemoryTracer


class _OtelSpan(Protocol):
    def set_attribute(self, key: str, value: str) -> None: ...


class _OtelTracer(Protocol):
    def start_as_current_span(self, name: str) -> AbstractContextManager[_OtelSpan]: ...


class _OtelCounter(Protocol):
    def add(self, value: float, attributes: Mapping[str, str] | None = None) -> None: ...


class _OtelMeter(Protocol):
    def create_counter(self, name: str) -> _OtelCounter: ...


class OpenTelemetryTracer(InMemoryTracer):
    def __init__(self) -> None:
        super().__init__()
        self._tracer: _OtelTracer | None = None
        self._meter: _OtelMeter | None = None
        self._counters: dict[str, _OtelCounter] = {}
        try:
            from opentelemetry import metrics, trace
        except ImportError:
            return
        # The optional OpenTelemetry packages are intentionally treated as an
        # external boundary.  Only the small surface used below is retained.
        self._tracer = cast(_OtelTracer, trace.get_tracer("swufe-rag"))
        self._meter = cast(_OtelMeter, metrics.get_meter("swufe-rag"))

    @contextmanager
    def start(self, name: str, **attributes: object) -> Iterator[None]:
        with super().start(name, **attributes):
            if self._tracer is None:
                yield
                return
            # The SDK's span object is optional; all local traces still remain
            # observable through the parent context manager.
            with self._tracer.start_as_current_span(name) as span:
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
            counter = self._meter.create_counter(name)
            self._counters[name] = counter
        labels = {
            key: str(item)
            for key, item in attributes.items()
            if key in {"intent", "tool_name", "status", "retrieval_mode"}
        }
        counter.add(value, labels)
