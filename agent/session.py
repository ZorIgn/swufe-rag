"""Bounded session stores; provider credentials and raw tool output are never persisted."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from hashlib import sha256
from threading import Lock
from time import monotonic
from typing import Any

from storage.json_contract import StrictJSONError, loads_strict_json

_SENSITIVE_KEYS = frozenset({"api_key", "client", "provider"})
_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9:_-]{1,96}$")


class SessionStoreError(RuntimeError):
    """A configured session backend is unavailable or rejected unsafe state."""


def scoped_session_id(session_id: str, principal_id: str | None = None) -> str:
    """Bind a client handle to an authenticated principal before storage.

    ``None`` preserves the single-tenant development behaviour.  When an
    identity is configured, both pieces are hashed so user-controlled labels
    cannot become a Redis key namespace or disclose identity metadata.
    """

    if principal_id is None:
        return session_id
    principal_digest = sha256(principal_id.encode("utf-8")).hexdigest()
    session_digest = sha256(session_id.encode("utf-8")).hexdigest()
    return f"principal-{principal_digest}:session-{session_digest}"


def _encoded_session(
    value: dict[str, object],
    *,
    dataset_version: str,
    max_messages: int,
    max_payload_bytes: int,
) -> tuple[dict[str, object], str]:
    """Return a redacted, bounded JSON payload shared by both stores."""

    safe = {key: item for key, item in value.items() if key not in _SENSITIVE_KEYS}
    safe["dataset_version"] = dataset_version
    messages = safe.get("messages")
    if isinstance(messages, list):
        safe["messages"] = messages[-max_messages:]
    try:
        encoded = json.dumps(
            safe,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise SessionStoreError("session payload is not valid bounded JSON") from exc
    if len(encoded.encode("utf-8")) > max_payload_bytes:
        raise SessionStoreError("session payload exceeds the configured byte limit")
    return safe, encoded


@dataclass
class InMemoryTTLSessionStore:
    ttl_seconds: float = 1800
    max_messages: int = 12
    max_sessions: int = 10000
    max_payload_bytes: int = 16 * 1024
    dataset_version: str = "unknown"
    _values: dict[str, tuple[float, dict[str, object]]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.ttl_seconds <= 0:
            raise ValueError("session TTL must be positive")
        if self.max_messages <= 0 or self.max_sessions <= 0 or self.max_payload_bytes <= 0:
            raise ValueError("session bounds must be positive")

    def _purge(self) -> None:
        now = monotonic()
        for key, (expires, _) in tuple(self._values.items()):
            if expires <= now:
                self._values.pop(key, None)
        while len(self._values) > self.max_sessions:
            self._values.pop(next(iter(self._values)))

    def get(
        self, session_id: str, *, principal_id: str | None = None
    ) -> dict[str, object] | None:
        key = scoped_session_id(session_id, principal_id)
        with self._lock:
            self._purge()
            stored = self._values.get(key)
            if stored is None:
                return None
            expires, value = stored
            if expires <= monotonic() or value.get("dataset_version") != self.dataset_version:
                self._values.pop(key, None)
                return None
            return dict(value)

    def put(
        self,
        session_id: str,
        value: dict[str, object],
        *,
        principal_id: str | None = None,
    ) -> None:
        key = scoped_session_id(session_id, principal_id)
        safe, _ = _encoded_session(
            value,
            dataset_version=self.dataset_version,
            max_messages=self.max_messages,
            max_payload_bytes=self.max_payload_bytes,
        )
        with self._lock:
            self._values[key] = (monotonic() + self.ttl_seconds, safe)
            self._purge()


class RedisSessionStore:
    """Opt-in shared sessions for multi-worker deployments.

    This class does not provide distributed rate limiting or retrieval caching.
    A configured Redis backend is fail-closed: startup and I/O failures are
    surfaced rather than silently falling back to process-local state.
    """

    def __init__(
        self,
        url: str,
        *,
        ttl_seconds: int,
        dataset_version: str,
        key_namespace: str = "swufe-rag:sessions",
        max_messages: int = 12,
        max_payload_bytes: int = 16 * 1024,
        socket_timeout_seconds: float = 1.0,
        client: Any | None = None,
        healthcheck: bool = True,
    ) -> None:
        if not url.strip() and client is None:
            raise ValueError("Redis session URL is required")
        if ttl_seconds <= 0 or max_messages <= 0 or max_payload_bytes <= 0:
            raise ValueError("Redis session bounds must be positive")
        if socket_timeout_seconds <= 0:
            raise ValueError("Redis socket timeout must be positive")
        if not _NAMESPACE_RE.fullmatch(key_namespace):
            raise ValueError("Redis session namespace contains unsupported characters")
        if client is None:
            try:
                import redis
            except ImportError as exc:  # pragma: no cover - optional deployment dependency
                raise RuntimeError("RedisSessionStore requires the redis extra") from exc
            client = redis.Redis.from_url(
                url,
                decode_responses=True,
                socket_connect_timeout=socket_timeout_seconds,
                socket_timeout=socket_timeout_seconds,
                retry_on_timeout=False,
            )
        self._client: Any = client
        self._ttl_seconds = ttl_seconds
        self._dataset_version = dataset_version
        self._key_namespace = key_namespace
        self._max_messages = max_messages
        self._max_payload_bytes = max_payload_bytes
        if healthcheck:
            try:
                if self._client.ping() is not True:
                    raise SessionStoreError("Redis session backend health check failed")
            except SessionStoreError:
                raise
            except Exception as exc:
                raise SessionStoreError("Redis session backend is unavailable") from exc

    def _key(self, session_id: str, principal_id: str | None = None) -> str:
        return f"{self._key_namespace}:{scoped_session_id(session_id, principal_id)}"

    def _forget(self, key: str) -> None:
        try:
            self._client.delete(key)
        except Exception:
            # The caller already treats the value as absent.  A later TTL will
            # remove it even if this best-effort cleanup cannot reach Redis.
            return

    def get(
        self, session_id: str, *, principal_id: str | None = None
    ) -> dict[str, object] | None:
        key = self._key(session_id, principal_id)
        try:
            raw = self._client.get(key)
        except Exception as exc:
            raise SessionStoreError("Redis session read failed") from exc
        if raw is None:
            return None
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError:
                self._forget(key)
                return None
        if not isinstance(raw, str):
            self._forget(key)
            return None
        if len(raw.encode("utf-8")) > self._max_payload_bytes:
            self._forget(key)
            return None
        try:
            loaded = loads_strict_json(raw, label="Redis session payload")
        except StrictJSONError:
            self._forget(key)
            return None
        if not isinstance(loaded, dict):
            self._forget(key)
            return None
        value = {str(field): item for field, item in loaded.items()}
        if value.get("dataset_version") != self._dataset_version:
            self._forget(key)
            return None
        return value

    def put(
        self,
        session_id: str,
        value: dict[str, object],
        *,
        principal_id: str | None = None,
    ) -> None:
        _, encoded = _encoded_session(
            value,
            dataset_version=self._dataset_version,
            max_messages=self._max_messages,
            max_payload_bytes=self._max_payload_bytes,
        )
        try:
            self._client.setex(
                self._key(session_id, principal_id),
                self._ttl_seconds,
                encoded,
            )
        except Exception as exc:
            raise SessionStoreError("Redis session write failed") from exc


__all__ = [
    "InMemoryTTLSessionStore",
    "RedisSessionStore",
    "SessionStoreError",
    "scoped_session_id",
]
