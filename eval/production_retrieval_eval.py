"""Evaluate the real BGE/FAISS retrieval stack on reviewed development cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

from eval.real_data_eval import DEFAULT_CASES
from generation.pipeline import EvidenceGate
from retrieval.pipeline import AdvancedRetriever


def _score_summary(scores: list[float]) -> dict[str, float | int | None]:
    if not scores:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": len(scores),
        "min": min(scores),
        "median": median(scores),
        "max": max(scores),
    }


def evaluate(
    cases_path: str | Path = DEFAULT_CASES,
    *,
    chunks_path: str | Path = "data/chunks.jsonl",
    artifacts_dir: str | Path = "artifacts",
    use_reranker: bool = False,
    refuse_th: float = 0.35,
    retriever: Any | None = None,
) -> dict[str, Any]:
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    actual_retriever = retriever or AdvancedRetriever.from_artifacts(
        chunks_path,
        artifacts_dir,
        use_reranker=use_reranker,
    )
    rows: list[dict[str, Any]] = []
    gate = EvidenceGate(dense_threshold=refuse_th)
    hits = 0
    total = 0
    pollution = 0
    refusal_correct = 0
    threshold_only_correct = 0
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    evidence_support_correct = 0
    evidence_support_total = 0
    false_accept_ids: list[str] = []
    false_refusal_ids: list[str] = []
    for case in cases:
        results = actual_retriever.retrieve(
            case["question"],
            5,
            case.get("college"),
            case.get("cohort"),
        )
        expected = set(case.get("expected_docs", []))
        docs = [item["doc_title"] for item in results]
        hit = not expected or bool(expected & set(docs))
        if expected:
            total += 1
            hits += int(hit)
        polluted = any(
            (
                item["level"] == "院级"
                and case.get("college")
                and item["college"] != case["college"]
            )
            or (
                case.get("cohort")
                and item["cohort"] not in {"不限", case["cohort"]}
            )
            for item in results
        )
        pollution += int(polluted)
        required_terms = case.get("answer_must_contain", [])
        joined_evidence = "".join(
            "".join(item["text"].split()) for item in results
        )
        evidence_support = not required_terms or all(
            "".join(term.split()) in joined_evidence for term in required_terms
        )
        if required_terms:
            evidence_support_total += 1
            evidence_support_correct += int(evidence_support)
        max_score = max((item["score"] for item in results), default=None)
        should_refuse = bool(case.get("should_refuse", False))
        predicted_refused = not gate.sufficient(case["question"], results)
        threshold_only_refused = max_score is None or max_score < refuse_th
        refusal_ok = predicted_refused == should_refuse
        threshold_only_ok = threshold_only_refused == should_refuse
        refusal_correct += int(refusal_ok)
        threshold_only_correct += int(threshold_only_ok)
        if should_refuse:
            if max_score is not None:
                negative_scores.append(max_score)
            if not predicted_refused:
                false_accept_ids.append(case["id"])
        else:
            if max_score is not None:
                positive_scores.append(max_score)
            if predicted_refused:
                false_refusal_ids.append(case["id"])
        rows.append(
            {
                "id": case["id"],
                "category": case.get("category"),
                "retrieval_hit": hit,
                "scope_pollution": polluted,
                "evidence_support_at_5": evidence_support,
                "should_refuse": should_refuse,
                "max_dense_score": max_score,
                "threshold_only_refused": threshold_only_refused,
                "predicted_refused": predicted_refused,
                "refusal_correct": refusal_ok,
                "retrieved": [
                    {
                        "chunk_id": item["chunk_id"],
                        "doc_title": item["doc_title"],
                        "article": item["article"],
                        "score": item["score"],
                    }
                    for item in results
                ],
            }
        )
    return {
        "backend": "bge-large-zh-v1.5/faiss",
        "reranker": "bge-reranker-base" if use_reranker else "heuristic",
        "case_count": len(cases),
        "retrieval_recall_at_5": hits / max(total, 1),
        "scope_pollution_count": pollution,
        "evidence_support_at_5": evidence_support_correct
        / max(evidence_support_total, 1),
        "refusal_threshold": refuse_th,
        "threshold_only_refusal_accuracy": threshold_only_correct
        / max(len(cases), 1),
        "refusal_accuracy": refusal_correct / max(len(cases), 1),
        "positive_max_score_summary": _score_summary(positive_scores),
        "negative_max_score_summary": _score_summary(negative_scores),
        "false_accept_ids": false_accept_ids,
        "false_refusal_ids": false_refusal_ids,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--reranker", action="store_true")
    parser.add_argument("--refuse-th", type=float, default=0.35)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                args.cases,
                chunks_path=args.chunks,
                artifacts_dir=args.artifacts,
                use_reranker=args.reranker,
                refuse_th=args.refuse_th,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
