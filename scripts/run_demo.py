"""Run a deterministic, synthetic end-to-end demo without external data or models."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import cast

from academic.database import build_database
from agent.factory import build_runtime

DEMO_QUESTIONS = (
    "2024级测试专业X专业选修最低要求多少学分？",
    "2024级测试专业X的测试算法多少学分，在哪个学期开设？",
    "2024级X专业选课系统实际开哪些课程？",
)


def _build_demo_database(target: Path) -> None:
    fixture = Path(__file__).resolve().parents[1] / "tests" / "canonical" / "data"
    build_database(
        target,
        catalog_path=fixture / "catalog.json",
        sources_path=fixture / "sources.csv",
        chunks_path=fixture / "chunks.jsonl",
        aliases_path=fixture / "aliases.json",
        source_review_path=fixture / "source_review.csv",
        evidence_review_path=fixture / "evidence_review.csv",
    )


def _ask_demo_questions(database: Path) -> list[dict[str, object]]:
    os.environ["SWUFE_RETRIEVAL_MODE"] = "lexical"
    os.environ["SWUFE_SESSION_BACKEND"] = "memory"
    runtime = build_runtime(database)
    try:
        results: list[dict[str, object]] = []
        for question in DEMO_QUESTIONS:
            answer, state = runtime.ask(question)
            results.append(
                {
                    "question": question,
                    "answer_md": answer.answer_md,
                    "refused": answer.refused,
                    "clarification": answer.clarification,
                    "citations": [
                        {
                            "chunk_id": citation.chunk_id,
                            "title": citation.title,
                            "page_url": citation.page_url,
                        }
                        for citation in answer.citations
                    ],
                    "output_statuses": [
                        contract.model_dump(mode="json") for contract in state.output_contracts
                    ],
                }
            )
        return results
    finally:
        runtime.repository.close()


def _run(database_out: Path | None = None) -> list[dict[str, object]]:
    if database_out is not None:
        database_out.parent.mkdir(parents=True, exist_ok=True)
        _build_demo_database(database_out)
        return _ask_demo_questions(database_out)
    with tempfile.TemporaryDirectory(prefix="swufe-rag-demo-") as directory:
        database = Path(directory) / "academic.sqlite3"
        _build_demo_database(database)
        return _ask_demo_questions(database)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit one machine-readable JSON document",
    )
    parser.add_argument(
        "--database-out",
        type=Path,
        help="also keep the synthetic fixture database at this new path",
    )
    args = parser.parse_args()
    if args.database_out is not None and args.database_out.exists():
        parser.error(f"--database-out already exists: {args.database_out}")
    results = _run(args.database_out)
    if args.json:
        payload = {"dataset_kind": "synthetic_test_fixture", "results": results}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print("SWUFE RAG synthetic demo (no official school data or external model calls)\n")
    for index, result in enumerate(results, start=1):
        print(f"[{index}] {result['question']}")
        print(result["answer_md"])
        citations = cast(list[dict[str, object]], result["citations"])
        if citations:
            print("citations:", ", ".join(str(item["chunk_id"]) for item in citations))
        print()


if __name__ == "__main__":
    main()
