"""Formal, reusable HTTP adapter for the frozen B/C public facade."""

from __future__ import annotations

from threading import RLock

from pydantic import BaseModel, ConfigDict, Field

from app.runtime import RAGRuntime, build_production_runtime
from contracts import (
    ContractError,
    GenerationUnavailableError,
    KnowledgeBaseNotReadyError,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AskRequest(StrictModel):
    question: str = Field(min_length=1, max_length=1000)
    college: str | None = None
    cohort: str | None = None


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


class AskResponse(StrictModel):
    answer_md: str
    citations: list[CitationResponse]
    refused: bool
    retrieved: list[RetrievedSummaryResponse]
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
    except ImportError as exc:
        raise RuntimeError(
            "web dependencies are missing; install requirements-web.txt"
        ) from exc

    state = {"runtime": runtime}
    state_lock = RLock()

    def get_runtime() -> RAGRuntime:
        with state_lock:
            if state["runtime"] is None:
                state["runtime"] = build_production_runtime()
            return state["runtime"]

    product_app = FastAPI(
        title="swufe-rag API",
        version="0.1.0",
        redoc_url=None,
    )

    @product_app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest):
        try:
            return get_runtime().ask(
                request.question,
                college=request.college,
                cohort=request.cohort,
            )
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
    "SourceResponse",
    "app",
    "create_app",
    "main",
]
