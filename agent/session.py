"""Bounded session stores; no provider credentials are ever persisted."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Any


@dataclass
class InMemoryTTLSessionStore:
    ttl_seconds: float = 1800
    max_messages: int = 12
    max_sessions: int = 10000
    dataset_version: str = "unknown"
    _values: dict[str, tuple[float, dict[str, object]]] = field(default_factory=dict)

    def _purge(self) -> None:
        now = monotonic()
        for key, (expires, _) in tuple(self._values.items()):
            if expires <= now:
                self._values.pop(key, None)
        while len(self._values) > self.max_sessions:
            self._values.pop(next(iter(self._values)))

    def get(self, session_id: str) -> dict[str, object] | None:
        self._purge()
        stored = self._values.get(session_id)
        if stored is None:
            return None
        expires, value = stored
        if expires <= monotonic() or value.get("dataset_version") != self.dataset_version:
            self._values.pop(session_id, None)
            return None
        return dict(value)

    def put(self, session_id: str, value: dict[str, object]) -> None:
        safe = {key: item for key, item in value.items() if key not in {"api_key", "client", "provider"}}
        safe["dataset_version"] = self.dataset_version
        messages = safe.get("messages")
        if isinstance(messages, list):
            safe["messages"] = messages[-self.max_messages:]
        self._values[session_id] = (monotonic() + self.ttl_seconds, safe)
        self._purge()


class RedisSessionStore:
    """Optional multi-worker implementation. Redis is imported only when requested."""

    def __init__(self, url: str, *, ttl_seconds: int, dataset_version: str) -> None:
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - optional deployment dependency
            raise RuntimeError("RedisSessionStore requires the redis extra") from exc
        self._client: Any = redis.Redis.from_url(url, decode_responses=True)
        self._ttl_seconds = ttl_seconds
        self._dataset_version = dataset_version

    def get(self, session_id: str) -> dict[str, object] | None:
        import json
        raw = self._client.get(f"academic-agent:{session_id}")
        if not raw:
            return None
        value = json.loads(raw)
        return value if value.get("dataset_version") == self._dataset_version else None

    def put(self, session_id: str, value: dict[str, object]) -> None:
        import json
        safe = {key: item for key, item in value.items() if key not in {"api_key", "client", "provider"}}
        safe["dataset_version"] = self._dataset_version
        self._client.setex(f"academic-agent:{session_id}", self._ttl_seconds, json.dumps(safe, ensure_ascii=False))
