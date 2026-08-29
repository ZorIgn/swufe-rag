from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent.factory import _build_session_store, _resolve_release_paths
from agent.orchestrator import AgentRuntime
from agent.policies import DeploymentPolicy, Principal
from agent.provider import ProviderCircuitBreaker, RequestModelFactory
from agent.session import (
    InMemoryTTLSessionStore,
    RedisSessionStore,
    SessionStoreError,
    scoped_session_id,
)
from app.server.canonical import RateLimiter, create_app
from storage.release import ReleaseError


def test_session_store_is_thread_safe_and_principal_scoped() -> None:
    store = InMemoryTTLSessionStore(max_sessions=16)
    barrier = threading.Barrier(8)

    def write(index: int) -> None:
        barrier.wait()
        store.put("shared", {"worker": index}, principal_id=f"tenant-{index}")

    threads = [threading.Thread(target=write, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert all(
        store.get("shared", principal_id=f"tenant-{index}")
        == {"worker": index, "dataset_version": "unknown"}
        for index in range(8)
    )
    assert store.get("shared", principal_id="other") is None
    # The compatibility path intentionally remains unchanged for a local,
    # single-tenant development runtime.
    store.put("shared", {"worker": "anonymous"})
    assert store.get("shared") == {"worker": "anonymous", "dataset_version": "unknown"}
    assert scoped_session_id("shared", "tenant-a") != scoped_session_id("shared", "tenant-b")


def test_rate_limiter_is_bounded_and_thread_safe() -> None:
    limiter = RateLimiter(maximum=100, max_clients=4, max_key_chars=16)
    barrier = threading.Barrier(16)

    def hit(index: int) -> None:
        barrier.wait()
        limiter.allow("x" * (1000 + index))

    threads = [threading.Thread(target=hit, args=(index,)) for index in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(limiter._values) <= 4
    assert all(len(key) <= 80 for key in limiter._values)


def test_provider_breaker_isolated_by_principal_and_credential() -> None:
    factory = RequestModelFactory(
        base_url="https://provider.example/v1",
        model="test-model",
        breaker=ProviderCircuitBreaker(failure_threshold=1, reset_after_seconds=60),
    )
    tenant_a = factory.create("same-key", principal_id="tenant-a")
    tenant_b = factory.create("same-key", principal_id="tenant-b")
    credential_b = factory.create("different-key", principal_id="tenant-a")
    tenant_a.breaker.record_failure()
    assert tenant_a.breaker.allow() is False
    assert tenant_b.breaker.allow() is True
    assert credential_b.breaker.allow() is True
    assert "same-key" not in repr(tenant_a)


def test_deployment_policy_clamps_invalid_environment(monkeypatch) -> None:
    monkeypatch.setenv("SWUFE_REQUEST_MAX_BYTES", "999999999")
    monkeypatch.setenv("SWUFE_MAX_CONCURRENT_REQUESTS", "0")
    monkeypatch.setenv("SWUFE_LLM_TIMEOUT_SECONDS", "999999999")
    policy = DeploymentPolicy.from_environment()
    assert policy.request_max_bytes == 32 * 1024
    assert policy.max_concurrent_requests == 32
    assert policy.provider_timeout_seconds == 45.0


def test_deployment_policy_rejects_unknown_mode(monkeypatch) -> None:
    monkeypatch.setenv("SWUFE_DEPLOYMENT_MODE", "prodution")

    with pytest.raises(ValueError, match="must be local or production"):
        DeploymentPolicy.from_environment()

    with pytest.raises(ReleaseError, match="must be local or production"):
        _resolve_release_paths("data/academic.sqlite3")


def test_static_bearer_protects_debug_and_source_and_ignores_tenant_header(
    canonical_runtime, monkeypatch
) -> None:
    monkeypatch.setenv("SWUFE_STATIC_BEARER_TOKEN", "local-development-token")
    monkeypatch.setenv("SWUFE_STATIC_BEARER_ROLES", "admin")
    monkeypatch.setenv("SWUFE_ENABLE_DEBUG_RESPONSES", "true")
    monkeypatch.setenv("SWUFE_RATE_LIMIT_PER_MINUTE", "100")
    client = TestClient(create_app(canonical_runtime))

    assert client.get("/health/live").status_code == 200
    unauthenticated = client.get("/source/test-plan-1")
    assert unauthenticated.status_code == 401
    unauthenticated = client.post(
        "/ask", json={"question": "2024级X专业第1学期有哪些选修课？", "debug": True}
    )
    assert unauthenticated.status_code == 401

    headers = {
        "Authorization": "Bearer local-development-token",
        "X-tenant": "attacker-selected-tenant",
    }
    source = client.get("/source/test-plan-1", headers=headers)
    assert source.status_code == 200
    answer = client.post(
        "/ask",
        headers=headers,
        json={
            "question": "2024级X专业第1学期有哪些选修课？",
            "debug": True,
            "session_id": "shared",
        },
    )
    assert answer.status_code == 200
    assert "debug" in answer.json()
    assert "args" not in answer.json()["debug"]["plan"]["operations"][0]



def test_production_mode_requires_authentication_configuration(
    canonical_runtime,
    monkeypatch,
) -> None:
    monkeypatch.setenv("SWUFE_DEPLOYMENT_MODE", "production")
    monkeypatch.delenv("SWUFE_STATIC_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("SWUFE_AUTH_STATIC_BEARER", raising=False)

    with pytest.raises(RuntimeError, match="production deployment requires"):
        create_app(canonical_runtime)


def test_production_mode_rejects_a_weak_static_bearer(
    canonical_runtime,
    monkeypatch,
) -> None:
    monkeypatch.setenv("SWUFE_DEPLOYMENT_MODE", "production")
    monkeypatch.setenv("SWUFE_STATIC_BEARER_TOKEN", "weak-token")

    with pytest.raises(RuntimeError, match="at least 32 characters"):
        create_app(canonical_runtime)


def test_debug_requires_admin_even_when_server_opted_in(
    canonical_runtime,
    monkeypatch,
) -> None:
    monkeypatch.setenv("SWUFE_ENABLE_DEBUG_RESPONSES", "true")

    client = TestClient(
        create_app(
            canonical_runtime,
            principal_resolver=lambda _request: Principal(subject="ordinary-user"),
        )
    )
    response = client.post(
        "/ask",
        json={
            "question": "2024级X专业第1学期有哪些选修课？",
            "debug": True,
        },
    )

    assert response.status_code == 403
    assert response.json()["error_code"] == "debug_not_allowed"


def test_debug_requires_explicit_server_opt_in_even_for_admin(
    canonical_runtime,
    monkeypatch,
) -> None:
    monkeypatch.delenv("SWUFE_ENABLE_DEBUG_RESPONSES", raising=False)
    client = TestClient(
        create_app(
            canonical_runtime,
            principal_resolver=lambda _request: Principal(
                subject="admin-user",
                roles=("admin",),
            ),
        )
    )

    response = client.post(
        "/ask",
        json={
            "question": "2024级X专业第1学期有哪些选修课？",
            "debug": True,
        },
    )

    assert response.status_code == 403
    assert response.json()["error_code"] == "debug_not_allowed"


def test_anonymous_sessions_are_rejected_by_default(
    canonical_runtime,
    monkeypatch,
) -> None:
    monkeypatch.delenv("SWUFE_STATIC_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("SWUFE_AUTH_STATIC_BEARER", raising=False)
    monkeypatch.delenv("SWUFE_ALLOW_ANONYMOUS_SESSIONS", raising=False)
    client = TestClient(create_app(canonical_runtime))

    response = client.post(
        "/ask",
        json={
            "question": "2024级X专业第1学期有哪些选修课？",
            "session_id": "guessable-session",
        },
    )

    assert response.status_code == 403
    assert response.json()["error_code"] == "anonymous_session_not_allowed"

def test_trusted_resolver_separates_same_client_session(canonical_runtime, monkeypatch) -> None:
    monkeypatch.delenv("SWUFE_STATIC_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("SWUFE_AUTH_STATIC_BEARER", raising=False)

    def resolver(request) -> Principal | None:
        value = request.headers.get("X-Trusted-Test-Principal")
        return Principal(subject=value) if value else None

    client = TestClient(create_app(canonical_runtime, principal_resolver=resolver))
    question = {"question": "2024级X专业第1学期有哪些选修课？", "session_id": "same"}
    assert client.post("/ask", json=question).status_code == 401
    for subject in ("alice", "bob"):
        response = client.post(
            "/ask", headers={"X-Trusted-Test-Principal": subject}, json=question
        )
        assert response.status_code == 200
    keys = tuple(canonical_runtime._deps.sessions._values)
    assert len(keys) >= 2
    assert all(key.startswith("principal-") for key in keys)


def test_content_length_precheck_and_health_rate_limit_exemption(canonical_runtime, monkeypatch) -> None:
    monkeypatch.delenv("SWUFE_STATIC_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("SWUFE_AUTH_STATIC_BEARER", raising=False)
    monkeypatch.setenv("SWUFE_REQUEST_MAX_BYTES", "64")
    monkeypatch.setenv("SWUFE_RATE_LIMIT_PER_MINUTE", "1")
    client = TestClient(create_app(canonical_runtime))
    oversized = client.post(
        "/ask", headers={"Content-Length": "65"}, content=b"{}"
    )
    assert oversized.status_code == 413
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/live").status_code == 200



class _FakeRedisClient:
    def __init__(self, *, ping_result: bool = True) -> None:
        self.values: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.ping_result = ping_result

    def ping(self) -> bool:
        return self.ping_result

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.values[key] = value
        self.ttls[key] = ttl

    def delete(self, key: str) -> None:
        self.values.pop(key, None)
        self.ttls.pop(key, None)


def test_redis_session_store_is_bounded_redacted_and_dataset_scoped() -> None:
    client = _FakeRedisClient()
    store = RedisSessionStore(
        "",
        client=client,
        ttl_seconds=120,
        dataset_version="release-a",
        max_messages=2,
        max_payload_bytes=1024,
    )
    store.put(
        "shared",
        {
            "messages": ["one", "two", "three"],
            "api_key": "must-not-be-persisted",
            "program_id": "program-a",
        },
        principal_id="tenant-a",
    )

    assert all("must-not-be-persisted" not in value for value in client.values.values())
    value = store.get("shared", principal_id="tenant-a")
    assert value is not None
    assert value["messages"] == ["two", "three"]
    assert value["dataset_version"] == "release-a"
    assert store.get("shared", principal_id="tenant-b") is None

    stale_reader = RedisSessionStore(
        "",
        client=client,
        ttl_seconds=120,
        dataset_version="release-b",
        max_payload_bytes=1024,
    )
    assert stale_reader.get("shared", principal_id="tenant-a") is None

    with pytest.raises(SessionStoreError, match="byte limit"):
        store.put("too-large", {"value": "x" * 2000})


def test_redis_session_read_rejects_oversized_or_ambiguous_backend_values() -> None:
    client = _FakeRedisClient()
    store = RedisSessionStore(
        "",
        client=client,
        ttl_seconds=120,
        dataset_version="release-a",
        max_payload_bytes=256,
    )

    key = store._key("corrupt", "tenant-a")
    for raw in (
        '{"dataset_version":"release-a","dataset_version":"release-a"}',
        '{"dataset_version":"release-a","metric":NaN}',
        '{"dataset_version":"release-a","value":"' + ("x" * 512) + '"}',
    ):
        client.values[key] = raw
        assert store.get("corrupt", principal_id="tenant-a") is None
        assert key not in client.values


def test_redis_backend_selection_is_explicit_and_has_no_memory_fallback(
    monkeypatch,
) -> None:
    sentinel = InMemoryTTLSessionStore(dataset_version="sentinel")
    captured: dict[str, object] = {}

    def build_redis(url: str, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return sentinel

    monkeypatch.setenv("SWUFE_SESSION_BACKEND", "redis")
    monkeypatch.setenv("SWUFE_REDIS_URL", "redis://redis.internal:6379/3")
    monkeypatch.setattr("agent.factory.RedisSessionStore", build_redis)

    selected = _build_session_store("release-a")
    assert selected is sentinel
    assert captured["url"] == "redis://redis.internal:6379/3"
    assert captured["dataset_version"] == "release-a"

    monkeypatch.delenv("SWUFE_REDIS_URL")
    with pytest.raises(ValueError, match="SWUFE_REDIS_URL"):
        _build_session_store("release-a")


def test_production_deployment_cannot_enable_unattested_active_release(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("SWUFE_RELEASE_ROOT", str(tmp_path / "releases"))
    monkeypatch.setenv("SWUFE_DEPLOYMENT_MODE", "production")
    monkeypatch.setenv("SWUFE_ALLOW_UNATTESTED_ACTIVE", "1")

    with pytest.raises(ReleaseError, match="production deployment"):
        _resolve_release_paths("data/academic.sqlite3")


def test_redis_healthcheck_failure_is_fail_closed() -> None:
    with pytest.raises(SessionStoreError, match="health check failed"):
        RedisSessionStore(
            "",
            client=_FakeRedisClient(ping_result=False),
            ttl_seconds=120,
            dataset_version="release-a",
        )


def test_session_backend_failure_is_reported_as_503(
    canonical_runtime,
    monkeypatch,
) -> None:
    class FailingSessions:
        def get(self, session_id: str):
            raise SessionStoreError("backend unavailable")

        def put(self, session_id: str, value: dict[str, object]) -> None:
            raise SessionStoreError("backend unavailable")

    runtime = AgentRuntime(
        replace(canonical_runtime._deps, sessions=FailingSessions())
    )
    client = TestClient(
        create_app(
            runtime,
            principal_resolver=lambda _request: Principal(subject="session-test"),
        )
    )
    response = client.post(
        "/ask",
        json={
            "question": "2024级X专业第1学期有哪些选修课？",
            "session_id": "shared",
        },
    )

    assert response.status_code == 503
    assert response.json()["error_code"] == "session_store_unavailable"
