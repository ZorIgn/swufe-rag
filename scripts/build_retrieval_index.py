"""Build the standalone versioned policy retrieval index from the canonical DB."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from academic.database import AcademicRepository
from retrieval.index import build_retrieval_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=Path("data/academic.sqlite3"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/retrieval"))
    parser.add_argument(
        "--mode",
        choices=("lexical", "hybrid"),
        default=os.getenv("SWUFE_RETRIEVAL_MODE", "hybrid"),
    )
    parser.add_argument(
        "--embedding-model", default=os.getenv("SWUFE_EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")
    )
    parser.add_argument(
        "--reranker-model", default=os.getenv("SWUFE_RERANKER_MODEL", "BAAI/bge-reranker-base")
    )
    args = parser.parse_args()
    repository = AcademicRepository(args.database)
    try:
        metadata = repository.metadata()
        result = build_retrieval_index(
            list(repository.retrieval_documents()),
            dataset_version=metadata.get("dataset_version", "unknown"),
            source_hash=metadata.get(
                "evidence_state_sha256", metadata.get("chunks_sha256", "")
            ),
            output_root=args.output_root,
            mode=args.mode,
            embedding_model=args.embedding_model,
            reranker_model=args.reranker_model,
        )
    finally:
        repository.close()
    print(result)


if __name__ == "__main__":
    main()
