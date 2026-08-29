"""Hard execution and deployment limits for the bounded agent.

The application deliberately keeps these limits in a small, dependency-free
module.  This gives a deployment a single place to clamp values supplied by
environment variables, while leaving the choice of an identity provider and
the decision to deploy Redis to the operator.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


def _bounded_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    """Return a safe integer configuration value.

    A malformed or out-of-range environment value falls back to the safe
    default.  Falling back is useful for a development checkout (where an
    accidental shell value should not make the app impossible to import), and
    the upper bounds prevent configuration from disabling the application
    back-pressure entirely.
    """

    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    if value < minimum or value > maximum:
        return default
    return value


def _bounded_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    if value < minimum or value > maximum:
        return default
    return value


def _bounded_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class RuntimePolicy:
    max_tool_calls: int = 8
    tool_timeout_seconds: float = 15.0
    max_validation_retries: int = 1
    max_question_chars: int = 4000


@dataclass(frozen=True)
class Principal:
    """Trusted request identity supplied by an explicitly configured resolver."""

    subject: str
    tenant: str = "default"
    authenticated: bool = True
    roles: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return f"{self.tenant}:{self.subject}"

    @property
    def is_admin(self) -> bool:
        return "admin" in self.roles


@dataclass(frozen=True)
class DeploymentPolicy:
    """Application-level resource and request boundary configuration.

    This is intentionally local-process policy.  It is not a replacement for
    a gateway, a distributed limiter, OAuth/SSO, or a shared session store.
    Those choices remain explicit deployment decisions.
    """

    deployment_mode: Literal["local", "production"] = "local"
    require_authentication: bool = False
    debug_responses_enabled: bool = False
    allow_anonymous_sessions: bool = False
    request_max_bytes: int = 32 * 1024
    rate_limit_per_minute: int = 60
    rate_limit_max_keys: int = 10_000
    rate_limit_key_max_chars: int = 256
    max_concurrent_requests: int = 32
    request_queue_timeout_seconds: float = 0.25
    provider_timeout_seconds: float = 45.0
    provider_max_retries: int = 2
    provider_max_tokens: int = 1200

    @classmethod
    def from_environment(cls) -> DeploymentPolicy:
        configured_mode = os.getenv("SWUFE_DEPLOYMENT_MODE", "local").strip().lower()
        if configured_mode not in {"local", "production"}:
            raise ValueError("SWUFE_DEPLOYMENT_MODE must be local or production")
        mode: Literal["local", "production"] = (
            "production" if configured_mode == "production" else "local"
        )
        require_authentication = (
            mode == "production" or _bounded_bool("SWUFE_REQUIRE_AUTHENTICATION", False)
        )
        return cls(
            deployment_mode=mode,
            require_authentication=require_authentication,
            debug_responses_enabled=_bounded_bool(
                "SWUFE_ENABLE_DEBUG_RESPONSES", False
            ),
            allow_anonymous_sessions=(
                mode == "local"
                and _bounded_bool("SWUFE_ALLOW_ANONYMOUS_SESSIONS", False)
            ),
            request_max_bytes=_bounded_int(
                "SWUFE_REQUEST_MAX_BYTES", 32 * 1024, minimum=1, maximum=16 * 1024 * 1024
            ),
            rate_limit_per_minute=_bounded_int(
                "SWUFE_RATE_LIMIT_PER_MINUTE", 60, minimum=1, maximum=100_000
            ),
            rate_limit_max_keys=_bounded_int(
                "SWUFE_RATE_LIMIT_MAX_KEYS", 10_000, minimum=1, maximum=1_000_000
            ),
            rate_limit_key_max_chars=_bounded_int(
                "SWUFE_RATE_LIMIT_KEY_MAX_CHARS", 256, minimum=32, maximum=4096
            ),
            max_concurrent_requests=_bounded_int(
                "SWUFE_MAX_CONCURRENT_REQUESTS", 32, minimum=1, maximum=256
            ),
            request_queue_timeout_seconds=_bounded_float(
                "SWUFE_REQUEST_QUEUE_TIMEOUT_SECONDS", 0.25, minimum=0.0, maximum=30.0
            ),
            provider_timeout_seconds=_bounded_float(
                "SWUFE_LLM_TIMEOUT_SECONDS", 45.0, minimum=0.1, maximum=300.0
            ),
            provider_max_retries=_bounded_int(
                "SWUFE_LLM_MAX_RETRIES", 2, minimum=0, maximum=5
            ),
            provider_max_tokens=_bounded_int(
                "SWUFE_LLM_MAX_TOKENS", 1200, minimum=1, maximum=16_000
            ),
        )


__all__ = ["DeploymentPolicy", "Principal", "RuntimePolicy"]
