"""Replaceable runtime adapter shared by the debug API and future frontends."""

from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any

from contracts import CHUNK_FIELDS, KnowledgeChunk, RetrievedChunk
from generation.pipeline import AdvancedGenerationService
from retrieval.embed import HashingEncoder
from retrieval.index import IndexBundle, file_sha256, load_chunks
from retrieval.pipeline import AdvancedRetriever
from retrieval.retriever import HybridRetriever
from app.demo_llm import DemoGroundedClient


DEMO_CHUNKS = Path(__file__).parents[1] / "tests" / "fixtures" / "chunks.jsonl"


@dataclass
class RAGRuntime:
    retriever: AdvancedRetriever
    generation: AdvancedGenerationService
    chunks: list[KnowledgeChunk]
    mode: str

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 5,
        college: str | None = None,
        cohort: str | None = None,
    ) -> list[RetrievedChunk]:
        return self.retriever.retrieve(question, top_k, college, cohort)

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
        result = self.generation.answer(question, retrieved)
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
            "mode": self.mode,
        }

    def source(self, chunk_id: str) -> KnowledgeChunk | None:
        for chunk in self.chunks:
            if chunk["chunk_id"] == chunk_id:
                return {key: chunk[key] for key in CHUNK_FIELDS}  # type: ignore[return-value]
        return None

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
    return RAGRuntime(retriever, generation, chunks, "demo")
