"""Replaceable runtime adapter shared by the debug API and future frontends."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from pathlib import Path
from typing import Any, Callable

from contracts import AnswerResult, CHUNK_FIELDS, KnowledgeChunk, RetrievedChunk
from generation.pipeline import AdvancedGenerationService
from retrieval.embed import HashingEncoder
from retrieval.index import IndexBundle, file_sha256, load_chunks
from retrieval.pipeline import AdvancedRetriever
from retrieval.retriever import HybridRetriever
from app.demo_llm import DemoGroundedClient


DEMO_CHUNKS = Path(__file__).parents[1] / "tests" / "fixtures" / "chunks.jsonl"

RetrieveFunction = Callable[
    [str, int, str | None, str | None], list[RetrievedChunk]
]
AnswerFunction = Callable[[str, list[dict[str, Any]]], AnswerResult]


@dataclass
class RAGRuntime:
    retrieve_function: RetrieveFunction
    answer_function: AnswerFunction
    chunks: list[KnowledgeChunk]
    mode: str
    _chunks_by_id: dict[str, KnowledgeChunk] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._chunks_by_id = {chunk["chunk_id"]: chunk for chunk in self.chunks}

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 5,
        college: str | None = None,
        cohort: str | None = None,
    ) -> list[RetrievedChunk]:
        return self.retrieve_function(question, top_k, college, cohort)

    def ask(
        self,
        question: str,
        *,
        top_k: int = 5,
        college: str | None = None,
        cohort: str | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        retrieved = self.retrieve(
            question, top_k=top_k, college=college, cohort=cohort
        )
        result = self.answer_function(question, retrieved)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return {
            **result,
            "retrieved": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "doc_title": chunk["doc_title"],
                    "article": chunk["article"],
                    "college": chunk["college"],
                    "cohort": chunk["cohort"],
                    "score": chunk["score"],
                    "is_table": chunk["is_table"],
                    "summary": chunk["text"][:260],
                }
                for chunk in retrieved
            ],
            "latency_ms": latency_ms,
        }

    def debug_ask(
        self,
        question: str,
        *,
        top_k: int = 5,
        college: str | None = None,
        cohort: str | None = None,
    ) -> dict[str, Any]:
        """Return the formal D payload plus debug-only runtime metadata."""

        return {
            **self.ask(
                question,
                top_k=top_k,
                college=college,
                cohort=cohort,
            ),
            "mode": self.mode,
        }

    def source(self, chunk_id: str) -> KnowledgeChunk | None:
        chunk = self._chunks_by_id.get(chunk_id)
        if chunk is None:
            return None
        return {key: chunk[key] for key in CHUNK_FIELDS}  # type: ignore[return-value]

    def options(self) -> dict[str, Any]:
        colleges = sorted(
            {chunk["college"] for chunk in self.chunks if chunk["level"] == "院级"}
        )
        cohorts = sorted(
            {chunk["cohort"] for chunk in self.chunks if chunk["cohort"] != "不限"}
        )
        return {
            "mode": self.mode,
            "colleges": colleges,
            "cohorts": cohorts,
            "chunk_count": len(self.chunks),
            "default_top_k": 5,
        }


def build_demo_runtime(path: str | Path = DEMO_CHUNKS) -> RAGRuntime:
    chunks_path = Path(path)
    chunks = load_chunks(chunks_path)
    encoder = HashingEncoder(128)
    embeddings = encoder.encode_documents([chunk["text"] for chunk in chunks])
    bundle = IndexBundle(
        chunks=chunks,
        embeddings=embeddings,
        model_name=encoder.model_name,
        source_hash=file_sha256(chunks_path),
        backend="demo-memory",
    )
    retriever = AdvancedRetriever(HybridRetriever(bundle, encoder))
    generation = AdvancedGenerationService(DemoGroundedClient())
    return RAGRuntime(retriever.retrieve, generation.answer, chunks, "demo")


def build_production_runtime(
    chunks_path: str | Path = "data/chunks.jsonl",
) -> RAGRuntime:
    """Build D's adapter around the frozen public B/C facade.

    This function never falls back to fixtures.  Missing production chunks or
    retrieval artifacts therefore remain visible as
    ``KnowledgeBaseNotReadyError`` at the HTTP boundary.
    """

    from swufe_rag.api import answer, retrieve

    chunks = load_chunks(chunks_path)
    return RAGRuntime(retrieve, answer, chunks, "production")
