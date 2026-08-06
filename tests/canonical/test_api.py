from __future__ import annotations

import json

from fastapi.testclient import TestClient

from agent.provider import ProviderError, RequestModelFactory
from app.server.canonical import create_app


def test_public_endpoints_share_one_runtime(canonical_runtime) -> None:
    client = TestClient(create_app(canonical_runtime))
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 200
    result = client.post(
        "/ask", json={"question": "2024级X专业第1学期有哪些选修课？", "debug": True}
    )
    assert result.status_code == 200
    payload = result.json()
    assert payload["debug"]["tool_calls"] == ["academic.list_courses"]
    source = client.get("/source/test-plan-1")
    assert source.status_code == 200
    assert "local_path" not in source.text


def test_api_rejects_body_secret_and_exposes_safe_error(canonical_runtime) -> None:
    client = TestClient(create_app(canonical_runtime))
    response = client.post("/ask", json={"question": "x", "api_key": "secret-value"})
    assert response.status_code == 422
    assert "secret-value" not in response.text


def test_provider_requires_explicit_endpoint_and_model() -> None:
    for factory in (
        RequestModelFactory(),
        RequestModelFactory(base_url="https://provider.example/v1"),
        RequestModelFactory(model="example-model"),
    ):
        try:
            factory.create("request-scoped-secret")
        except ProviderError:
            continue
        raise AssertionError("provider configuration must be explicit")


def test_academic_audit_uses_canonical_agent(canonical_runtime) -> None:
    client = TestClient(create_app(canonical_runtime))
    response = client.post(
        "/academic-audit",
        json={"cohort": 2024, "major": "测试专业X", "completed_courses": ["TST101"]},
    )
    assert response.status_code == 200
    assert response.json()["request_id"]
    assert response.json()["refused"] is False
    assert response.json()["citations"]


class _RecordingModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, _system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        payload = json.loads(user_prompt)
        if "schema" in payload:
            return json.dumps(
                {
                    "intent": "course_query",
                    "program_mentions": ["测试专业X"],
                    "cohort": 2024,
                    "target_semesters": [1],
                    "course_natures": ["elective"],
                    "information_scope": "curriculum",
                },
                ensure_ascii=False,
            )
        facts = payload["facts"]
        selected = [
            fact
            for fact in facts
            if fact["predicate"] in {"name", "code", "credits", "semester", "nature", "module"}
        ]
        return json.dumps(
            {
                "claims": [
                    {
                        "claim_id": "claim-1",
                        "text": "测试算法（TST101）：3 学分，第 1 学期开设。",
                        "fact_ids": [fact["fact_id"] for fact in selected],
                        "evidence_ids": sorted(
                            {identifier for fact in selected for identifier in fact["evidence_ids"]}
                        ),
                    }
                ]
            },
            ensure_ascii=False,
        )


def test_byok_model_is_request_scoped_and_never_stored(canonical_runtime) -> None:
    received: list[str] = []
    model = _RecordingModel()

    def factory(api_key: str) -> _RecordingModel:
        received.append(api_key)
        return model

    client = TestClient(create_app(canonical_runtime, request_model_factory=factory))
    response = client.post(
        "/ask",
        headers={"X-LLM-API-Key": "test-secret-key-123456"},
        json={"question": "2024级X专业第1学期有哪些选修课？", "session_id": "byok-test"},
    )

    assert response.status_code == 200
    assert response.json()["refused"] is False
    assert received == ["test-secret-key-123456"]
    assert model.calls == 2
    session = canonical_runtime._deps.sessions._values["byok-test"][1]
    assert "test-secret-key-123456" not in repr(session)
    assert "test-secret-key-123456" not in repr(canonical_runtime._deps.tracer.spans)
