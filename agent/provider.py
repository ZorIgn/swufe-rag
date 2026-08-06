"""Request-scoped, OpenAI-compatible structured-output provider.

The factory owns only endpoint and reliability policy.  A caller supplies an
API key for one request; the key is held by the short-lived model object and
is neither retained by the factory nor included in exception text.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ProviderError(RuntimeError):
    """A safe provider failure that deliberately omits credentials."""


@dataclass
class ProviderCircuitBreaker:
    """Small process-local circuit breaker keyed by provider configuration only."""

    failure_threshold: int = 3
    reset_after_seconds: float = 30.0
    _failure_count: int = 0
    _opened_at: float | None = None
    _lock: Lock = field(default_factory=Lock)

    def allow(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return True
            if monotonic() - self._opened_at >= self.reset_after_seconds:
                self._opened_at = None
                self._failure_count = 0
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                self._opened_at = monotonic()


@dataclass(frozen=True)
class RequestScopedModel:
    """A single-request model client implementing ``StructuredModel``.

    The OpenAI chat-completions wire format keeps the runtime provider-neutral
    without requiring an SDK or a global client carrying a user's credential.
    """

    api_key: str
    base_url: str
    model: str
    timeout_seconds: float
    max_tokens: int
    max_retries: int
    breaker: ProviderCircuitBreaker

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if not self.breaker.allow():
            raise ProviderError("provider circuit is temporarily open")
        endpoint = self.base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"
        payload = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            ensure_ascii=False,
        ).encode("utf-8")
        last_error: Exception | None = None
        for _attempt in range(self.max_retries + 1):
            try:
                request = Request(
                    endpoint,
                    data=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urlopen(request, timeout=self.timeout_seconds) as response:  # nosec B310 - configured provider URL
                    decoded = json.loads(response.read().decode("utf-8"))
                content = decoded["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise ProviderError("provider returned a non-text completion")
                self.breaker.record_success()
                return content
            except (HTTPError, URLError, TimeoutError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                last_error = exc
        self.breaker.record_failure()
        # Never interpolate URL headers, response bodies, or exception strings:
        # provider errors can reflect input in unexpected deployments.
        raise ProviderError("structured provider request failed") from last_error


@dataclass
class RequestModelFactory:
    """Creates ephemeral BYOK clients from environment-only provider settings."""

    base_url: str | None = None
    model: str | None = None
    timeout_seconds: float = 45.0
    max_tokens: int = 1200
    max_retries: int = 2
    breaker: ProviderCircuitBreaker = field(default_factory=ProviderCircuitBreaker)

    @classmethod
    def from_environment(cls) -> RequestModelFactory:
        return cls(
            base_url=os.getenv("SWUFE_LLM_BASE_URL") or None,
            model=os.getenv("SWUFE_LLM_MODEL") or None,
            timeout_seconds=float(os.getenv("SWUFE_LLM_TIMEOUT_SECONDS", "45")),
            max_tokens=int(os.getenv("SWUFE_LLM_MAX_TOKENS", "1200")),
            max_retries=int(os.getenv("SWUFE_LLM_MAX_RETRIES", "2")),
        )

    def create(self, api_key: str) -> RequestScopedModel:
        if not api_key or len(api_key) > 512:
            raise ProviderError("provider credential is invalid")
        if not self.base_url or not self.model:
            raise ProviderError("model provider endpoint is not configured")
        return RequestScopedModel(
            api_key=api_key,
            base_url=self.base_url,
            model=self.model,
            timeout_seconds=self.timeout_seconds,
            max_tokens=self.max_tokens,
            max_retries=self.max_retries,
            breaker=self.breaker,
        )


__all__ = ["ProviderCircuitBreaker", "ProviderError", "RequestModelFactory", "RequestScopedModel"]
