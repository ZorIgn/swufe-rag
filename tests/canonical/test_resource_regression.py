import asyncio
from threading import Event

from starlette.requests import Request
from starlette.responses import Response

from agent.policies import RuntimePolicy
from agent.session import InMemoryTTLSessionStore, RedisSessionStore
from agent.tools import BoundedToolPool, PlanExecutor
from app.server.canonical import create_app
from tests.canonical.test_plan_executor import _operation, _packet, _plan, _registry


def test_chunked_body_stops_on_first_over_limit_chunk(monkeypatch):
    monkeypatch.setenv("SWUFE_REQUEST_MAX_BYTES", "32768")
    application = create_app(runtime=object())
    guard = next(
        m.kwargs["dispatch"] for m in application.user_middleware if "dispatch" in m.kwargs
    )
    reads = 0

    async def receive():
        nonlocal reads
        reads += 1
        return {"type": "http.request", "body": b"x" * 16384, "more_body": reads < 64}

    async def downstream(request):
        raise AssertionError("Oversized body reached route")

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/ask",
        "headers": [],
        "client": ("198.51.100.1", 1),
        "server": ("test", 80),
        "scheme": "http",
        "query_string": b"",
    }
    result = asyncio.run(guard(Request(scope, receive), downstream))
    assert result.status_code == 413
    assert reads == 3


def test_body_timeout_returns_capacity_and_next_request_succeeds(monkeypatch):
    monkeypatch.setenv("SWUFE_BODY_TIMEOUT_SECONDS", "0.01")
    application = create_app(runtime=object())
    guard = next(
        m.kwargs["dispatch"] for m in application.user_middleware if "dispatch" in m.kwargs
    )
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/ask",
        "headers": [],
        "client": ("198.51.100.1", 1),
        "server": ("test", 80),
        "scheme": "http",
        "query_string": b"",
    }

    async def stalled():
        await asyncio.Event().wait()

    async def complete():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    async def downstream(request):
        assert await request.body() == b"{}"
        return Response(status_code=200)

    async def run():
        assert (await guard(Request(scope, stalled), downstream)).status_code == 408
        assert (await guard(Request(scope, complete), downstream)).status_code == 200

    asyncio.run(run())


def test_repeated_timeouts_keep_actual_tool_work_bounded_and_recover():
    release = Event()
    started = []

    def blocked(operation):
        started.append(operation.operation_id)
        release.wait(5)
        return _packet(operation.operation_id)

    class TrackingPool(BoundedToolPool):
        def __init__(self):
            super().__init__(2)
            self.finished = []

        def submit(self, *args):
            future = super().submit(*args)
            finished = Event()
            future.add_done_callback(lambda _: finished.set())
            self.finished.append(finished)
            return future

    pool = TrackingPool()
    executor = PlanExecutor(
        _registry(blocked, tool_timeout_seconds=0.02),
        RuntimePolicy(max_tool_calls=8, tool_timeout_seconds=0.02),
        pool=pool,
    )
    try:
        outcomes = [
            executor.execute(_plan(_operation(str(i)))).execution_results[0] for i in range(10)
        ]
        assert len(started) == 2
        assert [item.status for item in outcomes[:2]] == ["timeout", "timeout"]
        assert all(item.error_code == "tool_capacity_exceeded" for item in outcomes[2:])
        release.set()
        assert all(finished.wait(2) for finished in pool.finished)
        # Reuse the same process resource after actual futures complete.
        recovered = PlanExecutor(
            _registry(lambda operation: _packet(operation.operation_id), tool_timeout_seconds=1),
            RuntimePolicy(max_tool_calls=8, tool_timeout_seconds=1),
            pool=pool,
        )
        assert (
            recovered.execute(_plan(_operation("recovered"))).execution_results[0].status
            == "success"
        )
    finally:
        release.set()
        pool.shutdown()


def test_late_session_completion_does_not_overwrite_newer_request():
    store = InMemoryTTLSessionStore()
    first = store.begin("session")
    second = store.begin("session")
    store.put("session", {"program_id": "new"}, expected_token=second)
    store.put("session", {"program_id": "old"}, expected_token=first)
    assert store.get("session")["program_id"] == "new"


def test_redis_cas_across_two_store_instances():
    import os
    from uuid import uuid4

    import pytest

    url = os.getenv("SWUFE_TEST_REDIS_URL")
    if not url:
        pytest.skip("Set SWUFE_TEST_REDIS_URL for the Redis integration check")
    import redis

    client = redis.Redis.from_url(url, decode_responses=True)
    namespace = "regression:" + uuid4().hex
    first_store = RedisSessionStore(
        "", client=client, ttl_seconds=120, dataset_version="test", key_namespace=namespace
    )
    second_store = RedisSessionStore(
        "", client=client, ttl_seconds=120, dataset_version="test", key_namespace=namespace
    )
    try:
        old = first_store.begin("session")
        new = second_store.begin("session")
        second_store.put("session", {"program_id": "new"}, expected_token=new)
        first_store.put("session", {"program_id": "old"}, expected_token=old)
        assert first_store.get("session")["program_id"] == "new"
    finally:
        keys = list(client.scan_iter(namespace + ":*"))
        if keys:
            client.delete(*keys)
