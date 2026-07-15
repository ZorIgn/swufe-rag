"""FastAPI debug surface isolated from the future product HTTP contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock

from pydantic import BaseModel, Field

from app.runtime import RAGRuntime, build_demo_runtime, build_review_runtime
from contracts import GenerationUnavailableError, KnowledgeBaseNotReadyError


STATIC_DIR = Path(__file__).parent.parent / "static"
DEMO_CASES = Path(__file__).parents[2] / "demo" / "queries.json"
_runtime: RAGRuntime | None = None
_runtime_lock = RLock()


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    college: str | None = None
    cohort: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


def configure_runtime(runtime: RAGRuntime | None) -> None:
    global _runtime
    with _runtime_lock:
        _runtime = runtime


def get_runtime() -> RAGRuntime:
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            mode = os.getenv("SWUFE_RAG_MODE", "demo").lower()
            if mode == "demo":
                _runtime = build_demo_runtime()
            elif mode == "review":
                chunks_path = os.getenv("SWUFE_RAG_CHUNKS", "data/chunks.jsonl")
                _runtime = build_review_runtime(chunks_path)
            else:
                raise KnowledgeBaseNotReadyError(
                    "debug server mode must be demo or review; use app.server for production"
                )
        return _runtime


def create_app(runtime: RAGRuntime | None = None):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse
    except ImportError as exc:
        raise RuntimeError(
            "debug web dependencies are missing; install requirements-web.txt"
        ) from exc

    if runtime is not None:
        configure_runtime(runtime)
    debug_app = FastAPI(
        title="swufe-rag debug console",
        version="0.2.0",
        docs_url="/api/debug/docs",
        redoc_url=None,
    )

    @debug_app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "debug.html")

    @debug_app.get("/assets/debug.css", include_in_schema=False)
    def css():
        return FileResponse(STATIC_DIR / "debug.css", media_type="text/css")

    @debug_app.get("/assets/debug.js", include_in_schema=False)
    def js():
        return FileResponse(
            STATIC_DIR / "debug.js", media_type="application/javascript"
        )

    @debug_app.get("/api/debug/health")
    def health():
        return {"status": "ok", **get_runtime().options()}

    @debug_app.get("/api/debug/options")
    def options():
        return get_runtime().options()

    @debug_app.get("/api/debug/examples")
    def examples():
        return json.loads(DEMO_CASES.read_text(encoding="utf-8"))

    @debug_app.post("/api/debug/retrieve")
    def retrieve(request: QueryRequest):
        try:
            chunks = get_runtime().retrieve(
                request.question,
                top_k=request.top_k,
                college=request.college,
                cohort=request.cohort,
            )
            return {"retrieved": chunks, "mode": get_runtime().mode}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except KnowledgeBaseNotReadyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @debug_app.post("/api/debug/ask")
    def ask(request: QueryRequest):
        try:
            return get_runtime().debug_ask(
                request.question,
                top_k=request.top_k,
                college=request.college,
                cohort=request.cohort,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (KnowledgeBaseNotReadyError, GenerationUnavailableError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @debug_app.get("/api/debug/source/{chunk_id}")
    def source(chunk_id: str):
        chunk = get_runtime().source(chunk_id)
        if chunk is None:
            raise HTTPException(status_code=404, detail="chunk_id not found")
        return chunk

    return debug_app


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
    uvicorn.run("app.debug_server:app", host="127.0.0.1", port=8000, reload=False)


__all__ = ["QueryRequest", "app", "configure_runtime", "create_app", "get_runtime", "main"]
