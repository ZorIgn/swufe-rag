"""Formal, reusable HTTP adapter for the frozen B/C public facade."""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.runtime import RAGRuntime, build_production_hybrid_runtime
from contracts import (
    ContractError,
    GenerationUnavailableError,
    KnowledgeBaseNotReadyError,
)


STATIC_DIR = Path(__file__).parent.parent / "static"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AskRequest(StrictModel):
    question: str = Field(min_length=1, max_length=1000)
    college: str | None = None
    cohort: str | None = None
    session_id: str | None = Field(default=None, min_length=1, max_length=128)


class CitationResponse(StrictModel):
    marker: int
    chunk_id: str
    doc_title: str
    article: str
    quote: str
    page_url: str
    file_url: str


class RetrievedSummaryResponse(StrictModel):
    chunk_id: str
    doc_title: str
    article: str
    college: str
    cohort: str
    score: float
    is_table: bool
    summary: str


class OfficialLinkResponse(StrictModel):
    source_id: str
    title: str
    page_url: str
    file_url: str


class AskResponse(StrictModel):
    mode: Literal["general_chat", "school_rag"]
    answer_md: str
    citations: list[CitationResponse]
    refused: bool
    retrieved: list[RetrievedSummaryResponse]
    official_links: list[OfficialLinkResponse]
    latency_ms: float


class SourceResponse(StrictModel):
    chunk_id: str
    text: str
    doc_title: str
    article: str
    level: str
    college: str
    cohort: str
    year: int
    status: str
    page_url: str
    file_url: str
    is_table: bool


def create_app(runtime: RAGRuntime | None = None):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse
    except ImportError as exc:
        raise RuntimeError(
            "web dependencies are missing; install requirements-web.txt"
        ) from exc

    state = {"runtime": runtime}
    state_lock = RLock()

    def get_runtime() -> RAGRuntime:
        with state_lock:
            if state["runtime"] is None:
                state["runtime"] = build_production_hybrid_runtime()
            return state["runtime"]

    product_app = FastAPI(
        title="swufe-rag API",
        version="0.1.0",
        redoc_url=None,
    )

    @product_app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "chat.html")

    @product_app.get("/assets/chat.css", include_in_schema=False)
    def css():
        return FileResponse(STATIC_DIR / "chat.css", media_type="text/css")

    @product_app.get("/assets/chat.js", include_in_schema=False)
    def js():
        return FileResponse(
            STATIC_DIR / "chat.js", media_type="application/javascript"
        )

    @product_app.get("/options")
    def options():
        try:
            return get_runtime().options()
        except (ContractError, KnowledgeBaseNotReadyError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @product_app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest):
        try:
            runtime_instance = get_runtime()
            if hasattr(runtime_instance, "handle_question"):
                return runtime_instance.handle_question(
                    request.question,
                    college=request.college,
                    cohort=request.cohort,
                    session_id=request.session_id,
                )
            legacy = runtime_instance.ask(
                request.question,
                college=request.college,
                cohort=request.cohort,
            )
            return {
                "mode": "school_rag",
                **legacy,
                "official_links": [],
            }
        except ContractError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (KnowledgeBaseNotReadyError, GenerationUnavailableError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @product_app.get("/source/{chunk_id}", response_model=SourceResponse)
    def source(chunk_id: str):
        try:
            chunk = get_runtime().source(chunk_id)
        except (ContractError, KnowledgeBaseNotReadyError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if chunk is None:
            raise HTTPException(status_code=404, detail="chunk_id not found")
        return chunk

    return product_app


try:
    app = create_app()
except RuntimeError:
    app = None


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn is missing; install requirements-web.txt"
        ) from exc
    uvicorn.run("app.server:app", host="127.0.0.1", port=8000, reload=False)


__all__ = [
    "AskRequest",
    "AskResponse",
    "OfficialLinkResponse",
    "SourceResponse",
    "app",
    "create_app",
    "main",
]
