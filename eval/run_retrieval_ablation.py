"""Run transparent retrieval ablations over a JSON development set."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from retrieval.service import ScopedRetriever


def _metrics(ranks: list[int | None]) -> dict[str, float]:
    total = max(1, len(ranks))

    def recall_at(limit: int) -> float:
        return sum(rank is not None and rank <= limit for rank in ranks) / total

    return {
        "recall_at_1": recall_at(1),
        "recall_at_5": recall_at(5),
        "recall_at_10": recall_at(10),
        "mrr": sum(1 / rank for rank in ranks if rank is not None) / total,
        "ndcg_at_10": sum(
            1 / math.log2(rank + 1) for rank in ranks if rank is not None and rank <= 10
        ) / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("eval/reports/retrieval_ablation.json"))
    args = parser.parse_args()
    documents = [json.loads(line) for line in args.documents.read_text(encoding="utf-8").splitlines() if line]
    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    retriever = ScopedRetriever(documents, dataset_version="ablation")
    ranks: list[int | None] = []
    for query in queries:
        results = retriever.retrieve(query["question"], limit=10)
        relevant = query.get("relevant_chunk_id")
        ranks.append(next((index for index, item in enumerate(results, start=1) if item.chunk_id == relevant), None))
    report = {
        "query_count": len(queries),
        "available_variants": ["lexical"],
        "unavailable_variants": {
            "dense_only": "No configured dense encoder; this runner does not fabricate a dense score.",
            "dense_bm25_rrf": "Requires an independently configured dense encoder.",
            "reranker_mmr": "Requires a configured reranker and diversity selector.",
        },
        "results": {"lexical": _metrics(ranks)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lexical = _metrics(ranks)
    markdown = ["# Retrieval ablation", "", "Only configured variants are reported as measurements.", "", "| Variant | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    markdown.append("| lexical | {recall_at_1:.3f} | {recall_at_5:.3f} | {recall_at_10:.3f} | {mrr:.3f} | {ndcg_at_10:.3f} |".format(**lexical))
    markdown.append("")
    markdown.append("Unavailable variants are intentionally omitted from numeric comparisons: dense_only; dense_bm25_rrf; reranker_mmr.")
    args.output.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
