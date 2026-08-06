"""Only production FastAPI surface for the bounded academic agent."""

from __future__ import annotations

import os
import re
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from starlette.requests import Request

from agent.audit import audit as run_audit
from agent.factory import build_runtime
from agent.orchestrator import AgentRuntime
from agent.provider import ProviderError, RequestModelFactory
from evidence.models import FinalAnswer
from query.understanding import StructuredModel


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AskRequest(StrictModel):
    question: str = Field(min_length=1, max_length=4000)
    college: str | None = None
    cohort: int | None = Field(default=None, ge=2010, le=2100)
    major: str | None = None
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    debug: bool = False


class CompletedCourse(StrictModel):
    code: str | None = None
    name: str | None = None


class AcademicAuditRequest(StrictModel):
    question: str | None = Field(default=None, min_length=1, max_length=4000)
    cohort: int | None = Field(default=None, ge=2010, le=2100)
    major: str | None = None
    completed_courses: tuple[str | CompletedCourse, ...] = ()
    session_id: str | None = Field(default=None, min_length=1, max_length=128)


class ErrorResponse(StrictModel):
    request_id: str
    error_code: str
    message: str
    retryable: bool


@dataclass
class RateLimiter:
    maximum: int
    window_seconds: float = 60.0
    max_clients: int = 10000
    _values: dict[str, list[float]] = field(default_factory=dict)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        for client, values in tuple(self._values.items()):
            active = [item for item in values if item > cutoff]
            if active:
                self._values[client] = active
            else:
                self._values.pop(client, None)
        while len(self._values) >= self.max_clients and key not in self._values:
            self._values.pop(next(iter(self._values)))
        values = [item for item in self._values.get(key, []) if item > now - self.window_seconds]
        if len(values) >= self.maximum:
            self._values[key] = values
            return False
        values.append(now)
        self._values[key] = values
        return True


def _public(answer: FinalAnswer, state: Any, *, debug: bool) -> dict[str, object]:
    value: dict[str, object] = {
        "answer_md": answer.answer_md,
        "citations": [item.model_dump(mode="json") for item in answer.citations],
        "claims": [item.model_dump(mode="json") for item in answer.claims],
        "refused": answer.refused,
        "clarification": answer.clarification,
        "request_id": state.request_id,
    }
    if debug:
        value["debug"] = {
            "status": state.status.value,
            "retry_count": state.retry_count,
            "tool_calls": state.tool_calls,
            "tool_results": state.tool_results,
            "plan": state.plan.model_dump(mode="json") if state.plan else None,
        }
    return value


def create_app(runtime: AgentRuntime | None = None, *, request_model_factory: RequestModelFactory | Callable[[str], StructuredModel] | None = None) -> Any:
    try:
        from fastapi import FastAPI, Header
        from fastapi.exceptions import RequestValidationError
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
    except ImportError as exc:  # pragma: no cover - package install error
        raise RuntimeError("install the web dependency group") from exc

    limit_bytes = int(os.getenv("SWUFE_REQUEST_MAX_BYTES", "32768"))
    limiter = RateLimiter(maximum=int(os.getenv("SWUFE_RATE_LIMIT_PER_MINUTE", "60")))
    model_factory = request_model_factory or RequestModelFactory.from_environment()
    state: dict[str, object] = {"runtime": runtime, "ready": runtime is not None}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if state["runtime"] is None:
            state["runtime"] = build_runtime(os.getenv("SWUFE_ACADEMIC_DATABASE", "data/academic.sqlite3"))
        state["ready"] = True
        try:
            yield
        finally:
            loaded = state.get("runtime")
            if isinstance(loaded, AgentRuntime):
                loaded.repository.close()
            state["ready"] = False

    application = FastAPI(title="Evidence-Grounded Academic Agent", version="1.0.0", lifespan=lifespan, redoc_url=None)

    def request_id_for(request: Request) -> str:
        candidate = request.headers.get("X-Request-ID", "")
        return candidate if re.fullmatch(r"[A-Za-z0-9-]{1,64}", candidate) else uuid4().hex
    allowed_origins = [item.strip() for item in os.getenv("SWUFE_CORS_ALLOW_ORIGINS", "").split(",") if item.strip()]
    if allowed_origins:
        application.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-LLM-API-Key"])

    @application.middleware("http")
    async def guard(request: Request, call_next: Any) -> Any:
        request_id = request_id_for(request)
        body = await request.body()
        if len(body) > limit_bytes:
            return JSONResponse(status_code=413, content=ErrorResponse(request_id=request_id, error_code="request_too_large", message="request body is too large", retryable=False).model_dump())
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            return JSONResponse(status_code=429, content=ErrorResponse(request_id=request_id, error_code="rate_limited", message="too many requests", retryable=True).model_dump())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @application.exception_handler(RequestValidationError)
    async def body_validation_error(request: Request, _: RequestValidationError) -> Any:
        return JSONResponse(status_code=422, content=ErrorResponse(request_id=request_id_for(request), error_code="invalid_schema", message="request schema is invalid", retryable=False).model_dump())

    @application.exception_handler(ValueError)
    async def validation_error(request: Request, _: ValueError) -> Any:
        return JSONResponse(status_code=400, content=ErrorResponse(request_id=request_id_for(request), error_code="invalid_request", message="request could not be processed", retryable=False).model_dump())

    @application.exception_handler(ProviderError)
    async def provider_error(request: Request, _: ProviderError) -> Any:
        return JSONResponse(status_code=503, content=ErrorResponse(request_id=request_id_for(request), error_code="provider_unavailable", message="the configured model provider is unavailable", retryable=True).model_dump())

    @application.exception_handler(Exception)
    async def internal_error(request: Request, _: Exception) -> Any:
        return JSONResponse(status_code=500, content=ErrorResponse(request_id=request_id_for(request), error_code="internal_error", message="request could not be completed", retryable=True).model_dump())

    def current() -> AgentRuntime:
        loaded = state.get("runtime")
        if not isinstance(loaded, AgentRuntime):
            raise ValueError("service not ready")
        return loaded

    def model_for_request(api_key: str | None) -> StructuredModel | None:
        if api_key is None:
            return None
        if isinstance(model_factory, RequestModelFactory):
            model: StructuredModel = model_factory.create(api_key)
        else:
            model = model_factory(api_key)
        del api_key
        return model

    def ask_with_model(question: str, *, session_id: str | None, api_key: str | None) -> tuple[FinalAnswer, object]:
        model = model_for_request(api_key)
        try:
            return current().ask(question, session_id=session_id, model=model)
        finally:
            # The short-lived client is the only object that held the header value.
            del model

    @application.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "live"}

    @application.get("/health/ready")
    def ready() -> Any:
        if not state["ready"]:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return {"status": "ready", "dataset": current().repository.metadata().get("dataset_version")}

    @application.get("/options")
    def options() -> dict[str, object]:
        return current().options()

    @application.post("/ask")
    def ask(request: AskRequest, x_llm_api_key: str | None = Header(default=None, alias="X-LLM-API-Key", max_length=512)) -> dict[str, object]:
        prefix = " ".join(value for value in (str(request.cohort) + "级" if request.cohort else None, request.major) if value)
        answer, agent_state = ask_with_model(f"{prefix} {request.question}".strip(), session_id=request.session_id, api_key=x_llm_api_key)
        return _public(answer, agent_state, debug=request.debug)

    @application.get("/source/{chunk_id}")
    def source(chunk_id: str, request: Request) -> Any:
        value = current().source(chunk_id)
        if value is None:
            return JSONResponse(status_code=404, content=ErrorResponse(request_id=request_id_for(request), error_code="source_not_found", message="source was not found", retryable=False).model_dump())
        return value

    @application.get("/academic-audit/options")
    def audit_options() -> dict[str, object]:
        return current().options()

    @application.post("/academic-audit")
    def academic_audit(request: AcademicAuditRequest) -> dict[str, object]:
        if request.question:
            answer, agent_state = current().ask(request.question, session_id=request.session_id)
        else:
            if request.cohort is None or not request.major:
                raise ValueError("cohort and major are required")
            courses = tuple(value if isinstance(value, str) else (value.code or value.name or "") for value in request.completed_courses)
            answer, agent_state = run_audit(current(), cohort=request.cohort, major=request.major, completed_courses=tuple(value for value in courses if value), session_id=request.session_id)
        return _public(answer, agent_state, debug=False)

    return application


__all__ = ["AcademicAuditRequest", "AskRequest", "ErrorResponse", "create_app"]
