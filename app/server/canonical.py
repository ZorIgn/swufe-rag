"""Only production FastAPI surface for the bounded academic agent."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import os
import re
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from starlette.requests import Request

from agent.audit import audit as run_audit
from agent.factory import build_runtime
from agent.orchestrator import AgentRuntime
from agent.policies import DeploymentPolicy, Principal
from agent.provider import ProviderError, RequestModelFactory
from agent.session import SessionStoreError, scoped_session_id
from evidence.models import FinalAnswer
from query.context import RequestContext
from query.understanding import StructuredModel

IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AskRequest(StrictModel):
    question: str = Field(min_length=1, max_length=4000)
    college: str | None = None
    cohort: int | None = Field(default=None, ge=2010, le=2100)
    major: str | None = None
    as_of: str | None = None
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    debug: bool = False


class CompletedCourse(StrictModel):
    code: str | None = None
    name: str | None = None


class AcademicAuditRequest(StrictModel):
    question: str | None = Field(default=None, min_length=1, max_length=4000)
    cohort: int | None = Field(default=None, ge=2010, le=2100)
    major: str | None = None
    college: str | None = None
    as_of: str | None = None
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
    max_key_chars: int = 256
    _values: dict[str, list[float]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def _key(self, key: str) -> str:
        value = str(key or "unknown")
        if len(value) <= self.max_key_chars:
            return value
        # A fixed-size digest prevents attacker-controlled headers/addresses
        # from making the in-memory dictionary grow by unbounded key length.
        return f"digest:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"

    def allow(self, key: str) -> bool:
        normalized = self._key(key)
        with self._lock:
            now = time.monotonic()
            cutoff = now - self.window_seconds
            for client, values in tuple(self._values.items()):
                active = [item for item in values if item > cutoff]
                if active:
                    self._values[client] = active
                else:
                    self._values.pop(client, None)
            capacity = max(1, self.max_clients)
            while len(self._values) >= capacity and normalized not in self._values:
                self._values.pop(next(iter(self._values)))
            values = [
                item for item in self._values.get(normalized, []) if item > now - self.window_seconds
            ]
            if len(values) >= self.maximum:
                self._values[normalized] = values
                return False
            values.append(now)
            self._values[normalized] = values
            return True


def _valid_principal(value: object) -> Principal | None:
    """Normalize trusted resolver output without accepting arbitrary headers."""

    if isinstance(value, Principal):
        principal = value
    elif isinstance(value, str):
        principal = Principal(subject=value)
    else:
        return None
    if (
        not principal.authenticated
        or not principal.subject
        or not principal.tenant
        or len(principal.subject) > 256
        or len(principal.tenant) > 128
        or any(ord(char) < 32 for char in principal.subject + principal.tenant)
        or any(
            not role
            or len(role) > 64
            or any(ord(char) < 32 for char in role)
            for role in principal.roles
        )
    ):
        return None
    return principal


def _static_bearer_configuration() -> tuple[str | None, str, tuple[str, ...]]:
    token = os.getenv("SWUFE_STATIC_BEARER_TOKEN") or os.getenv("SWUFE_AUTH_STATIC_BEARER")
    subject = os.getenv("SWUFE_STATIC_BEARER_PRINCIPAL", "static-bearer")
    roles = tuple(
        dict.fromkeys(
            role.strip()
            for role in os.getenv("SWUFE_STATIC_BEARER_ROLES", "").split(",")
            if role.strip()
        )
    )
    return token, subject, roles


def _trusted_proxy_networks() -> tuple[IPNetwork, ...]:
    configured = os.getenv("SWUFE_TRUSTED_PROXY_CIDRS", "")
    networks: list[IPNetwork] = []
    for item in configured.split(","):
        value = item.strip()
        if not value:
            continue
        try:
            networks.append(ipaddress.ip_network(value, strict=False))
        except ValueError:
            # An invalid proxy allow-list must not cause the app to trust a
            # forwarded address; it is ignored and the direct peer is used.
            continue
    return tuple(networks)


def _peer_is_trusted(peer: str, networks: tuple[IPNetwork, ...]) -> bool:
    try:
        address = ipaddress.ip_address(peer)
    except ValueError:
        return False
    return any(address in network for network in networks)


def _client_rate_key(
    request: Request,
    principal: Principal | None,
    trusted_proxies: tuple[IPNetwork, ...],
) -> str:
    if principal is not None and principal.authenticated:
        return f"principal:{principal.key}"
    peer = request.client.host if request.client else "unknown"
    # Forwarded headers are only consulted when the immediate peer belongs to
    # an operator-provided proxy allow-list.  X-tenant and similar client
    # headers are never used as identities or rate keys.
    if _peer_is_trusted(peer, trusted_proxies):
        forwarded = request.headers.get("X-Forwarded-For", "")
        candidate = forwarded.split(",", 1)[0].strip()
        try:
            ipaddress.ip_address(candidate)
        except ValueError:
            pass
        else:
            peer = candidate
    return f"peer:{peer}"


def _public(
    answer: FinalAnswer,
    state: Any,
    *,
    debug: bool,
    dataset_version: str | None = None,
) -> dict[str, object]:
    query = getattr(state, "normalized_query", None)
    contracts = getattr(state, "output_contracts", ())
    if not contracts:
        plan = getattr(state, "plan", None)
        contracts = getattr(plan, "output_contract", ())
    scope = getattr(query, "information_scope", None)
    limitations = list(getattr(query, "warnings", ()) or ())
    if scope == "actual_offerings":
        limitations.append("actual_offerings_not_supported")
    limitations.extend(
        reason
        for contract in contracts
        if getattr(contract, "status", None) != "fulfilled"
        for reason in getattr(contract, "reasons", ())
    )
    value: dict[str, object] = {
        "answer_md": answer.answer_md,
        "citations": [item.model_dump(mode="json") for item in answer.citations],
        "claims": [item.model_dump(mode="json") for item in answer.claims],
        "refused": answer.refused,
        "clarification": answer.clarification,
        "request_id": state.request_id,
        "data_scope": "none" if scope == "actual_offerings" else scope,
        "limitations": list(dict.fromkeys(limitations)),
        "as_of": getattr(query, "policy_as_of", None),
        "dataset_version": dataset_version,
        "output_statuses": [item.model_dump(mode="json") for item in contracts],
    }
    if debug:
        plan = state.plan
        value["debug"] = {
            "status": state.status.value,
            "repair_count": state.repair_count,
            "regeneration_count": state.regeneration_count,
            "tool_calls": state.tool_calls,
            "tool_results": [item.model_dump(mode="json") for item in state.tool_results],
            "plan": (
                None
                if plan is None
                else {
                    "plan_id": plan.plan_id,
                    "operations": [
                        {
                            "operation_id": operation.operation_id,
                            "type": operation.type,
                            "tool_name": operation.tool_name,
                            "depends_on": operation.depends_on,
                        }
                        for operation in plan.operations
                    ],
                    "output_contract": [
                        item.model_dump(mode="json")
                        for item in state.output_contracts
                    ],
                }
            ),
        }
    return value


def create_app(
    runtime: AgentRuntime | None = None,
    *,
    request_model_factory: RequestModelFactory | Callable[[str], StructuredModel] | None = None,
    principal_resolver: Callable[[Request], Principal | str | None] | None = None,
) -> Any:
    try:
        from fastapi import FastAPI, Header
        from fastapi.exceptions import RequestValidationError
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
    except ImportError as exc:  # pragma: no cover - package install error
        raise RuntimeError("install the web dependency group") from exc

    deployment_policy = DeploymentPolicy.from_environment()
    limit_bytes = deployment_policy.request_max_bytes
    limiter = RateLimiter(
        maximum=deployment_policy.rate_limit_per_minute,
        max_clients=deployment_policy.rate_limit_max_keys,
        max_key_chars=deployment_policy.rate_limit_key_max_chars,
    )
    trusted_proxies = _trusted_proxy_networks()
    static_token, static_subject, static_roles = _static_bearer_configuration()
    if static_token is not None and (
        len(static_token) > 512
        or static_token != static_token.strip()
        or any(ord(char) < 32 for char in static_token)
    ):
        raise RuntimeError("static bearer token format is invalid")
    if (
        deployment_policy.deployment_mode == "production"
        and static_token is not None
        and len(static_token) < 32
    ):
        raise RuntimeError(
            "production static bearer token must contain at least 32 characters"
        )
    auth_configured = principal_resolver is not None or bool(static_token)
    if deployment_policy.require_authentication and not auth_configured:
        raise RuntimeError(
            "production deployment requires a trusted principal resolver or "
            "SWUFE_STATIC_BEARER_TOKEN"
        )
    concurrency = asyncio.Semaphore(deployment_policy.max_concurrent_requests)
    model_factory = request_model_factory or RequestModelFactory.from_environment()
    state: dict[str, object] = {"runtime": runtime, "ready": runtime is not None}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if state["runtime"] is None:
            state["runtime"] = build_runtime(
                os.getenv("SWUFE_ACADEMIC_DATABASE", "data/academic.sqlite3")
            )
        state["ready"] = True
        try:
            yield
        finally:
            loaded = state.get("runtime")
            if isinstance(loaded, AgentRuntime):
                loaded.repository.close()
            state["ready"] = False

    application = FastAPI(
        title="Evidence-Grounded Academic Agent", version="0.1.0", lifespan=lifespan, redoc_url=None
    )

    def request_id_for(request: Request) -> str:
        candidate = request.headers.get("X-Request-ID", "")
        return candidate if re.fullmatch(r"[A-Za-z0-9-]{1,64}", candidate) else uuid4().hex

    def principal_for(request: Request) -> Principal | None:
        if principal_resolver is not None:
            try:
                return _valid_principal(principal_resolver(request))
            except Exception:  # pragma: no cover - deployment resolver boundary
                return None
        if not static_token:
            return None
        authorization = request.headers.get("Authorization", "")
        scheme, separator, candidate = authorization.partition(" ")
        if (
            not separator
            or scheme.lower() != "bearer"
            or not candidate
            or not hmac.compare_digest(candidate, static_token)
        ):
            return None
        return _valid_principal(
            Principal(subject=static_subject, roles=static_roles)
        )

    def auth_failure(request: Request) -> Any:
        response = JSONResponse(
            status_code=401,
            content=ErrorResponse(
                request_id=request_id_for(request),
                error_code="authentication_required",
                message="authentication is required",
                retryable=False,
            ).model_dump(),
        )
        response.headers["WWW-Authenticate"] = "Bearer"
        return response

    def authorization_failure(
        request: Request,
        *,
        error_code: str,
        message: str,
    ) -> Any:
        return JSONResponse(
            status_code=403,
            content=ErrorResponse(
                request_id=request_id_for(request),
                error_code=error_code,
                message=message,
                retryable=False,
            ).model_dump(),
        )

    allowed_origins = [
        item.strip()
        for item in os.getenv("SWUFE_CORS_ALLOW_ORIGINS", "").split(",")
        if item.strip()
    ]
    if allowed_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "X-LLM-API-Key"],
        )

    @application.middleware("http")
    async def guard(request: Request, call_next: Any) -> Any:
        request_id = request_id_for(request)
        is_health = request.url.path in {"/health/live", "/health/ready"}
        is_preflight = request.method == "OPTIONS"
        principal = None if (is_health or is_preflight) else principal_for(request)
        if auth_configured and not (is_health or is_preflight) and (
            principal is None or not principal.authenticated
        ):
            return auth_failure(request)
        request.state.principal = principal

        # Health probes are intentionally cheap and exempt from user request
        # rate/concurrency quotas.  Body checks still apply to all other paths.
        if not (is_health or is_preflight):
            raw_length = request.headers.get("content-length")
            if raw_length is not None:
                try:
                    declared_length = int(raw_length)
                except ValueError:
                    return JSONResponse(
                        status_code=400,
                        content=ErrorResponse(
                            request_id=request_id,
                            error_code="invalid_content_length",
                            message="content length is invalid",
                            retryable=False,
                        ).model_dump(),
                    )
                if declared_length < 0 or declared_length > limit_bytes:
                    return JSONResponse(
                        status_code=413,
                        content=ErrorResponse(
                            request_id=request_id,
                            error_code="request_too_large",
                            message="request body is too large",
                            retryable=False,
                        ).model_dump(),
                    )
            body = await request.body()
            if len(body) > limit_bytes:
                return JSONResponse(
                    status_code=413,
                    content=ErrorResponse(
                        request_id=request_id,
                        error_code="request_too_large",
                        message="request body is too large",
                        retryable=False,
                    ).model_dump(),
                )
            if not limiter.allow(_client_rate_key(request, principal, trusted_proxies)):
                return JSONResponse(
                    status_code=429,
                    content=ErrorResponse(
                        request_id=request_id,
                        error_code="rate_limited",
                        message="too many requests",
                        retryable=True,
                    ).model_dump(),
                )
            try:
                if deployment_policy.request_queue_timeout_seconds:
                    await asyncio.wait_for(
                        concurrency.acquire(), deployment_policy.request_queue_timeout_seconds
                    )
                else:
                    await concurrency.acquire()
            except TimeoutError:
                return JSONResponse(
                    status_code=503,
                    content=ErrorResponse(
                        request_id=request_id,
                        error_code="server_busy",
                        message="request concurrency limit reached",
                        retryable=True,
                    ).model_dump(),
                )
            try:
                response = await call_next(request)
            finally:
                concurrency.release()
        else:
            response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @application.exception_handler(RequestValidationError)
    async def body_validation_error(request: Request, _: RequestValidationError) -> Any:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                request_id=request_id_for(request),
                error_code="invalid_schema",
                message="request schema is invalid",
                retryable=False,
            ).model_dump(),
        )

    @application.exception_handler(ValueError)
    async def validation_error(request: Request, _: ValueError) -> Any:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                request_id=request_id_for(request),
                error_code="invalid_request",
                message="request could not be processed",
                retryable=False,
            ).model_dump(),
        )

    @application.exception_handler(SessionStoreError)
    async def session_store_error(request: Request, _: SessionStoreError) -> Any:
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                request_id=request_id_for(request),
                error_code="session_store_unavailable",
                message="session continuity is temporarily unavailable",
                retryable=True,
            ).model_dump(),
        )

    @application.exception_handler(ProviderError)
    async def provider_error(request: Request, _: ProviderError) -> Any:
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                request_id=request_id_for(request),
                error_code="provider_unavailable",
                message="the configured model provider is unavailable",
                retryable=True,
            ).model_dump(),
        )

    @application.exception_handler(Exception)
    async def internal_error(request: Request, _: Exception) -> Any:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                request_id=request_id_for(request),
                error_code="internal_error",
                message="request could not be completed",
                retryable=True,
            ).model_dump(),
        )

    def current() -> AgentRuntime:
        loaded = state.get("runtime")
        if not isinstance(loaded, AgentRuntime):
            raise ValueError("service not ready")
        return loaded

    def request_session_id(
        requested: str | None,
        http_request: Request,
        principal: Principal | None,
    ) -> str | None:
        if requested is None:
            return None
        if principal is not None and principal.authenticated:
            return scoped_session_id(requested, principal.key)
        anonymous_scope = _client_rate_key(
            http_request,
            None,
            trusted_proxies,
        )
        return scoped_session_id(requested, f"anonymous:{anonymous_scope}")

    def model_for_request(
        api_key: str | None, *, principal: Principal | None = None
    ) -> StructuredModel | None:
        if api_key is None:
            return None
        model: StructuredModel
        if isinstance(model_factory, RequestModelFactory):
            model = model_factory.create(
                api_key,
                principal_id=principal.key if principal and principal.authenticated else None,
            )
        else:
            model = model_factory(api_key)
        del api_key
        return model

    def ask_with_model(
        question: str,
        *,
        context: RequestContext,
        api_key: str | None,
        principal: Principal | None,
    ) -> tuple[FinalAnswer, object]:
        model = model_for_request(api_key, principal=principal)
        try:
            return current().ask(question, context=context, model=model)
        finally:
            # The short-lived client is the only object that held the header value.
            del model

    @application.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "live"}

    @application.get("/health/ready")
    def ready() -> Any:
        loaded = current()
        ready_state, reasons = loaded.readiness()
        if not ready_state:
            return JSONResponse(
                status_code=503, content={"status": "not_ready", "reasons": reasons}
            )
        return {
            "status": "ready",
            "dataset": loaded.repository.metadata().get("dataset_version"),
            "retrieval_mode": loaded.options().get("retrieval_mode"),
        }

    @application.get("/options")
    def options() -> dict[str, object]:
        return current().options()

    @application.post("/ask")
    def ask(
        request: AskRequest,
        http_request: Request,
        x_llm_api_key: str | None = Header(default=None, alias="X-LLM-API-Key", max_length=512),
    ) -> Any:
        raw_principal = getattr(http_request.state, "principal", None)
        principal = raw_principal if isinstance(raw_principal, Principal) else None
        if request.debug and (
            not deployment_policy.debug_responses_enabled
            or principal is None
            or not principal.authenticated
            or not principal.is_admin
        ):
            return authorization_failure(
                http_request,
                error_code="debug_not_allowed",
                message="debug responses require an authenticated admin and explicit server opt-in",
            )
        if (
            request.session_id
            and (principal is None or not principal.authenticated)
            and not deployment_policy.allow_anonymous_sessions
        ):
            return authorization_failure(
                http_request,
                error_code="anonymous_session_not_allowed",
                message="session continuity requires authentication",
            )
        session_id = request_session_id(
            request.session_id,
            http_request,
            principal,
        )
        context = RequestContext(
            cohort=request.cohort,
            college=request.college,
            major=request.major,
            as_of=request.as_of,
            session_id=session_id,
        )
        answer, agent_state = ask_with_model(
            request.question,
            context=context,
            api_key=x_llm_api_key,
            principal=principal,
        )
        return _public(
            answer,
            agent_state,
            debug=request.debug,
            dataset_version=str(current().repository.metadata().get("dataset_version") or "") or None,
        )

    @application.get("/source/{chunk_id}")
    def source(chunk_id: str, request: Request) -> Any:
        # The middleware has already required the configured principal.  Keep
        # this explicit endpoint check so a future router/middleware change
        # cannot accidentally make provenance export public in an authenticated
        # deployment.
        if auth_configured and not (
            isinstance(getattr(request.state, "principal", None), Principal)
            and request.state.principal.authenticated
        ):
            return auth_failure(request)
        value = current().source(chunk_id)
        if value is None:
            return JSONResponse(
                status_code=404,
                content=ErrorResponse(
                    request_id=request_id_for(request),
                    error_code="source_not_found",
                    message="source was not found",
                    retryable=False,
                ).model_dump(),
            )
        return value

    @application.get("/academic-audit/options")
    def audit_options() -> dict[str, object]:
        return current().options()

    @application.post("/academic-audit")
    def academic_audit(
        request: AcademicAuditRequest, http_request: Request
    ) -> Any:
        raw_principal = getattr(http_request.state, "principal", None)
        principal = raw_principal if isinstance(raw_principal, Principal) else None
        if (
            request.session_id
            and (principal is None or not principal.authenticated)
            and not deployment_policy.allow_anonymous_sessions
        ):
            return authorization_failure(
                http_request,
                error_code="anonymous_session_not_allowed",
                message="session continuity requires authentication",
            )
        session_id = request_session_id(
            request.session_id,
            http_request,
            principal,
        )
        if request.question:
            answer, agent_state = current().ask(
                request.question,
                context=RequestContext(
                    cohort=request.cohort,
                    college=request.college,
                    major=request.major,
                    as_of=request.as_of,
                    session_id=session_id,
                ),
                model=None,
            )
        else:
            if request.cohort is None or not request.major:
                raise ValueError("cohort and major are required")
            courses: list[str] = []
            for value in request.completed_courses:
                if isinstance(value, str):
                    if not value.strip():
                        raise ValueError("completed course mentions must not be empty")
                    courses.append(value.strip())
                else:
                    mention = (value.code or value.name or "").strip()
                    if not mention:
                        raise ValueError("each completed course needs a code or name")
                    courses.append(mention)
            answer, agent_state = run_audit(
                current(),
                cohort=request.cohort,
                major=request.major,
                completed_courses=tuple(courses),
                session_id=session_id,
            )
        return _public(
            answer,
            agent_state,
            debug=False,
            dataset_version=str(current().repository.metadata().get("dataset_version") or "") or None,
        )

    return application


__all__ = ["AcademicAuditRequest", "AskRequest", "ErrorResponse", "create_app"]
