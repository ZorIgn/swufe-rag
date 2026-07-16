"""HTTP API exposing truthful QueryPlan/SQL/RAG/LLM execution telemetry."""

from __future__ import annotations

import os
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from academic_audit import CurriculumAuditService
from app.runtime_v2 import (
    build_local_query_plan_runtime,
    build_request_query_plan_runtime,
)
from contracts import (
    ContractError,
    GenerationUnavailableError,
    KnowledgeBaseNotReadyError,
)


STATIC_DIR = Path(__file__).parent / "static"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AskRequest(StrictModel):
    question: str = Field(min_length=1, max_length=2000)
    college: str | None = None
    cohort: str | None = None
    major: str | None = None
    session_id: str | None = Field(default=None, min_length=1, max_length=128)


class CitationResponse(StrictModel):
    marker: int
    chunk_id: str
    doc_title: str
    article: str
    quote: str
    page_url: str
    file_url: str


class RetrievedResponse(StrictModel):
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


class AskResponseV2(StrictModel):
    mode: Literal["general_chat", "school_rag"]
    answer_md: str
    citations: list[CitationResponse]
    refused: bool
    retrieved: list[RetrievedResponse]
    official_links: list[OfficialLinkResponse]
    latency_ms: float
    execution_path: str
    llm_called: bool
    llm_stages: dict[str, bool]
    model: str | None
    query_plan: dict[str, Any]
    sql_coverage: bool | None
    fallback: str | None
    answer_generation_error: str | None
    timings: dict[str, float]


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


class CompletedCourseRequest(StrictModel):
    code: str | None = None
    name: str | None = None
    credits: float | None = None


class AcademicAuditRequest(StrictModel):
    question: str | None = Field(default=None, min_length=1, max_length=2000)
    cohort: str | None = None
    major: str | None = None
    target_module: str | None = None
    current_semester: str | int | None = None
    completed_courses: list[str | CompletedCourseRequest] = Field(default_factory=list)


def create_app(runtime: Any | None = None):
    try:
        from fastapi import FastAPI, Header, HTTPException
        from fastapi.responses import FileResponse
    except ImportError as exc:
        raise RuntimeError("web dependencies are missing") from exc

    state: dict[str, Any] = {"runtime": runtime, "audit": None}
    lock = RLock()

    def get_runtime():
        with lock:
            if state["runtime"] is None:
                state["runtime"] = build_local_query_plan_runtime(
                    os.getenv("SWUFE_RAG_CHUNKS", "data/chunks.jsonl"),
                    sources_path=os.getenv("SWUFE_RAG_SOURCES", "data/sources.csv"),
                    metadata_path=os.getenv("SWUFE_RAG_METADATA", "data/metadata.sqlite3"),
                    config_path=os.getenv("SWUFE_RAG_CONFIG", "config.advanced.yaml"),
                    academic_database=os.getenv(
                        "SWUFE_RAG_ACADEMIC_DB", "data/academic_v2.sqlite3"
                    ),
                )
            return state["runtime"]

    def get_audit():
        with lock:
            if state["audit"] is None:
                state["audit"] = CurriculumAuditService(
                    os.getenv(
                        "SWUFE_RAG_CURRICULUM",
                        "data/curriculum_catalog_v2.json",
                    )
                )
            return state["audit"]

    product = FastAPI(title="swufe-rag query-plan API", version="0.3.0", redoc_url=None)

    @product.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "chat_v2.html")

    @product.get("/academic-audit-ui", include_in_schema=False)
    def audit_ui():
        return FileResponse(STATIC_DIR / "academic_audit.html")

    @product.get("/options")
    def options():
        try:
            return get_runtime().options()
        except (ContractError, KnowledgeBaseNotReadyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @product.post("/ask", response_model=AskResponseV2)
    def ask(
        request: AskRequest,
        x_llm_api_key: str | None = Header(
            default=None, alias="X-LLM-API-Key", max_length=512
        ),
    ):
        question = request.question
        if request.major and request.major not in question:
            question = f"{request.major} {question}"
        try:
            selected = get_runtime()
            if x_llm_api_key is not None:
                selected = build_request_query_plan_runtime(
                    selected,
                    x_llm_api_key,
                    config_path=os.getenv(
                        "SWUFE_RAG_CONFIG", "config.advanced.yaml"
                    ),
                )
            return selected.handle_question(
                question,
                college=request.college,
                cohort=request.cohort,
                session_id=request.session_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (ContractError, KnowledgeBaseNotReadyError, GenerationUnavailableError, FileNotFoundError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @product.get("/source/{chunk_id}", response_model=SourceResponse)
    def source(chunk_id: str):
        value = get_runtime().source(chunk_id)
        if value is None:
            raise HTTPException(status_code=404, detail="chunk_id not found")
        return value

    @product.get("/academic-audit/options")
    def audit_options():
        try:
            return get_audit().options()
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @product.post("/academic-audit")
    def academic_audit(request: AcademicAuditRequest):
        completed = [
            item if isinstance(item, str) else item.model_dump(exclude_none=True)
            for item in request.completed_courses
        ]
        try:
            service = get_audit()
            if request.question:
                return service.audit_question(
                    request.question,
                    cohort=request.cohort,
                    major=request.major,
                    completed_courses=completed,
                    target_module=request.target_module,
                    current_semester=request.current_semester,
                )
            if not request.cohort or not request.major:
                raise ValueError("cohort and major are required")
            return service.audit(
                cohort=request.cohort,
                major=request.major,
                completed_courses=completed,
                target_module=request.target_module,
                current_semester=request.current_semester,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return product


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("app.server_v2:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()


__all__ = ["AskRequest", "AskResponseV2", "app", "create_app", "main"]
