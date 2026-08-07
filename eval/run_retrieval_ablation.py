"""Run transparent retrieval ablations over a validated JSON development set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.metrics import rank_metrics
from retrieval.service import ScopedRetriever


def _load_inputs(
    documents_path: Path, queries_path: Path
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    documents = [
        value
        for line in documents_path.read_text(encoding="utf-8").splitlines()
        if line and isinstance(value := json.loads(line), dict)
    ]
    if not documents:
        raise SystemExit("retrieval documents must contain at least one JSON object")
    document_ids = {str(item.get("chunk_id") or "") for item in documents}
    if "" in document_ids:
        raise SystemExit("every retrieval document needs a non-empty chunk_id")

    raw_queries = json.loads(queries_path.read_text(encoding="utf-8"))
    if not isinstance(raw_queries, list) or not raw_queries:
        raise SystemExit("retrieval queries must be a non-empty JSON array")
    queries: list[dict[str, str]] = []
    for index, item in enumerate(raw_queries, start=1):
        if not isinstance(item, dict):
            raise SystemExit(f"retrieval query {index} must be an object")
        question = str(item.get("question") or "").strip()
        relevant = str(item.get("relevant_chunk_id") or "").strip()
        if not question or not relevant:
            raise SystemExit(f"retrieval query {index} needs question and relevant_chunk_id")
        if relevant not in document_ids:
            raise SystemExit(
                f"retrieval query {index} references unknown chunk_id: {relevant}"
            )
        queries.append({"question": question, "relevant_chunk_id": relevant})
    return documents, queries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("eval/reports/retrieval_ablation.json")
    )
    args = parser.parse_args()
    documents, queries = _load_inputs(args.documents, args.queries)
    retriever = ScopedRetriever(documents, dataset_version="ablation")
    ranks: list[int | None] = []
    for query in queries:
        results = retriever.retrieve(query["question"], limit=10)
        relevant = query["relevant_chunk_id"]
        ranks.append(
            next(
                (
                    index
                    for index, item in enumerate(results, start=1)
                    if item.chunk_id == relevant
                ),
                None,
            )
        )
    lexical_metrics = rank_metrics(ranks)
    report = {
        "query_count": len(queries),
        "results": {"lexical": lexical_metrics},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = [
        "# Retrieval ablation",
        "",
        "Only configured variants are reported as measurements.",
        "",
        "| Variant | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    markdown.append(
        "| lexical | {recall_at_1:.3f} | {recall_at_5:.3f} | "
        "{recall_at_10:.3f} | {mrr:.3f} | {ndcg_at_10:.3f} |".format(
            **lexical_metrics
        )
    )
    markdown.append("")
    markdown.append(
        "Dense/RRF/reranker/MMR variants require a real hybrid artifact; "
        "no unmeasured values are included in this report."
    )
    args.output.with_suffix(".md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()