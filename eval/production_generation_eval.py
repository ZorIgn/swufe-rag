"""Opt-in live evaluation for the production BGE/FAISS and LLM pipeline.

This module never substitutes the extractive review client.  The CLI requires
``--live`` so a paid or local model call cannot be triggered accidentally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Protocol

from eval.real_data_eval import DEFAULT_CASES
from generation.pipeline import AdvancedGenerationService, service_from_config
from retrieval.pipeline import AdvancedRetriever


class RetrieverLike(Protocol):
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        college: str | None = None,
        cohort: str | None = None,
    ) -> list[dict[str, Any]]: ...


class GenerationLike(Protocol):
    def answer(
        self, query: str, chunks: list[dict[str, Any]]
    ) -> dict[str, Any]: ...


def _compact(text: str) -> str:
    return "".join(text.split())


def evaluate(
    cases_path: str | Path = DEFAULT_CASES,
    *,
    chunks_path: str | Path = "data/chunks.jsonl",
    artifacts_dir: str | Path = "artifacts",
    config_path: str | Path = "config.advanced.yaml",
    use_reranker: bool = False,
    limit: int | None = None,
    retriever: RetrieverLike | None = None,
    generation: GenerationLike | None = None,
) -> dict[str, Any]:
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        cases = cases[:limit]

    actual_retriever = retriever or AdvancedRetriever.from_artifacts(
        chunks_path,
        artifacts_dir,
        use_reranker=use_reranker,
    )
    actual_generation = generation or service_from_config(config_path)

    rows: list[dict[str, Any]] = []
    refusal_correct = 0
    support_correct = 0
    support_total = 0
    citation_correct = 0
    for case in cases:
        chunks = actual_retriever.retrieve(
            case["question"],
            5,
            case.get("college"),
            case.get("cohort"),
        )
        result = actual_generation.answer(case["question"], chunks)
        should_refuse = bool(case.get("should_refuse", False))
        refusal_ok = bool(result["refused"]) == should_refuse
        refusal_correct += int(refusal_ok)

        required_terms = case.get("answer_must_contain", [])
        compact_answer = _compact(result["answer_md"])
        support_ok = not required_terms or (
            not result["refused"]
            and all(_compact(term) in compact_answer for term in required_terms)
        )
        if required_terms:
            support_total += 1
            support_correct += int(support_ok)

        by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        citation_ok = all(
            citation["chunk_id"] in by_id
            and citation["quote"] in by_id[citation["chunk_id"]]["text"]
            and citation["page_url"]
            == by_id[citation["chunk_id"]]["page_url"]
            and citation["file_url"]
            == by_id[citation["chunk_id"]]["file_url"]
            for citation in result["citations"]
        )
        if not result["refused"] and not result["citations"]:
            citation_ok = False
        citation_correct += int(citation_ok)
        rows.append(
            {
                "id": case["id"],
                "category": case.get("category"),
                "should_refuse": should_refuse,
                "refused": result["refused"],
                "refusal_correct": refusal_ok,
                "answer_support_correct": support_ok,
                "citation_integrity": citation_ok,
                "answer_md": result["answer_md"],
                "citation_count": len(result["citations"]),
                "retrieved_chunk_ids": [chunk["chunk_id"] for chunk in chunks],
            }
        )

    client = getattr(actual_generation, "client", None)
    model_spec = getattr(client, "model_spec", type(client).__name__)
    return {
        "runtime": "production-bge-faiss-live-llm",
        "model": model_spec,
        "reranker": "bge-reranker-base" if use_reranker else "heuristic",
        "case_count": len(cases),
        "refusal_accuracy": refusal_correct / max(len(cases), 1),
        "answer_support_accuracy": support_correct / max(support_total, 1),
        "citation_integrity_accuracy": citation_correct / max(len(cases), 1),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live",
        action="store_true",
        help="acknowledge that this command will call the configured LLM",
    )
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--config", default="config.advanced.yaml")
    parser.add_argument("--reranker", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not args.live:
        parser.error("--live is required because this evaluation calls the configured LLM")
    print(
        json.dumps(
            evaluate(
                args.cases,
                chunks_path=args.chunks,
                artifacts_dir=args.artifacts,
                config_path=args.config,
                use_reranker=args.reranker,
                limit=args.limit,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
