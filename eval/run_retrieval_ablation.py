"""Measure lexical and artifact-backed hybrid retrieval without inventing results."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from eval.metrics import rank_metrics
from retrieval.dense import DenseFaissIndex
from retrieval.hybrid import HybridPolicyRetriever
from retrieval.index import load_manifest
from retrieval.models import PolicyRetrievalRequest
from retrieval.reranker import CrossEncoderReranker
from retrieval.service import ScopedRetriever

METRIC_KEYS = ("recall_at_1", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10")


class HybridArtifactUnavailable(RuntimeError):
    """The requested hybrid run cannot be measured from local, valid artifacts."""


@dataclass(frozen=True)
class EvaluationQuery:
    question: str
    relevant_chunk_id: str
    cohort: int | None = None
    program_ids: tuple[str, ...] = ()
    college_ids: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    as_of: str | None = None


def _probability(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("threshold must be a number in [0, 1]") from exc
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("threshold must be in [0, 1]")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if not 1 <= parsed <= 100:
        raise argparse.ArgumentTypeError("value must be in [1, 100]")
    return parsed


def _string_tuple(value: object, field: str, index: int) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise SystemExit(f"retrieval query {index}: {field} must be a list of non-empty strings")
    return tuple(item.strip() for item in value)


def _load_inputs(
    documents_path: Path, queries_path: Path
) -> tuple[list[dict[str, object]], list[EvaluationQuery]]:
    documents: list[dict[str, object]] = []
    for index, line in enumerate(documents_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise SystemExit(f"retrieval document {index} must be a JSON object")
        documents.append(dict(value))
    if not documents:
        raise SystemExit("retrieval documents must contain at least one JSON object")
    document_ids = tuple(str(item.get("chunk_id") or "").strip() for item in documents)
    if any(not identifier for identifier in document_ids):
        raise SystemExit("every retrieval document needs a non-empty chunk_id")
    if len(set(document_ids)) != len(document_ids):
        raise SystemExit("retrieval document chunk_id values must be unique")

    raw_queries = json.loads(queries_path.read_text(encoding="utf-8"))
    if not isinstance(raw_queries, list) or not raw_queries:
        raise SystemExit("retrieval queries must be a non-empty JSON array")
    queries: list[EvaluationQuery] = []
    known_ids = set(document_ids)
    for index, item in enumerate(raw_queries, start=1):
        if not isinstance(item, dict):
            raise SystemExit(f"retrieval query {index} must be an object")
        question = item.get("question")
        relevant = item.get("relevant_chunk_id")
        if not isinstance(question, str) or not question.strip():
            raise SystemExit(f"retrieval query {index} needs a non-empty question")
        if not isinstance(relevant, str) or not relevant.strip():
            raise SystemExit(f"retrieval query {index} needs relevant_chunk_id")
        if relevant not in known_ids:
            raise SystemExit(f"retrieval query {index} references unknown chunk_id: {relevant}")
        cohort = item.get("cohort")
        if cohort is not None and (isinstance(cohort, bool) or not isinstance(cohort, int)):
            raise SystemExit(f"retrieval query {index}: cohort must be an integer or null")
        as_of = item.get("as_of")
        if as_of is not None and (not isinstance(as_of, str) or not as_of.strip()):
            raise SystemExit(f"retrieval query {index}: as_of must be a non-empty string or null")
        queries.append(
            EvaluationQuery(
                question=question.strip(),
                relevant_chunk_id=relevant.strip(),
                cohort=cohort,
                program_ids=_string_tuple(item.get("program_ids"), "program_ids", index),
                college_ids=_string_tuple(item.get("college_ids"), "college_ids", index),
                topics=_string_tuple(item.get("topics"), "topics", index),
                as_of=as_of.strip() if isinstance(as_of, str) else None,
            )
        )
    return documents, queries


def _load_hybrid_retriever(
    documents: list[dict[str, object]], *, artifact_root: Path, dataset_version: str | None
) -> HybridPolicyRetriever:
    if not dataset_version:
        raise HybridArtifactUnavailable("hybrid evaluation requires --dataset-version")
    try:
        directory, manifest = load_manifest(artifact_root, dataset_version)
        if str(manifest.get("retrieval_mode") or "") != "hybrid":
            raise HybridArtifactUnavailable("retrieval manifest does not declare hybrid mode")
        if str(manifest.get("dataset_version") or "") != dataset_version:
            raise HybridArtifactUnavailable("retrieval manifest dataset version does not match --dataset-version")
        model_name = str(manifest.get("embedding_model") or "")
        reranker_name = str(manifest.get("reranker_model") or "")
        dimension = int(str(manifest.get("embedding_dimension") or 0))
        if not model_name or not reranker_name or dimension <= 0:
            raise HybridArtifactUnavailable("retrieval manifest lacks hybrid model metadata")
        raw_chunk_ids = json.loads((directory / "doc_ids.json").read_text(encoding="utf-8"))
        if not isinstance(raw_chunk_ids, list) or not raw_chunk_ids:
            raise HybridArtifactUnavailable("hybrid artifact doc_ids.json is missing or invalid")
        expected_chunk_ids = tuple(str(value) for value in raw_chunk_ids)
        actual_chunk_ids = tuple(str(document["chunk_id"]) for document in documents)
        if set(expected_chunk_ids) != set(actual_chunk_ids) or len(expected_chunk_ids) != len(actual_chunk_ids):
            raise HybridArtifactUnavailable(
                "hybrid artifact chunk IDs do not match the evaluated documents"
            )
        dense = DenseFaissIndex.load(directory, model_name=model_name, dimension=dimension)
        reranker = CrossEncoderReranker(reranker_name)
    except HybridArtifactUnavailable:
        raise
    except Exception as exc:
        raise HybridArtifactUnavailable(
            f"unable to load hybrid retrieval artifact ({type(exc).__name__}: {exc})"
        ) from exc
    retriever = HybridPolicyRetriever(
        documents,
        mode="hybrid",
        dense_index=dense,
        reranker=reranker,
        dataset_version=dataset_version,
        index_version=str(manifest.get("dataset_version") or "") or None,
        expected_chunk_ids=expected_chunk_ids,
        expected_dimension=dimension,
    )
    ready, reasons = retriever.readiness()
    if not ready:
        raise HybridArtifactUnavailable("hybrid retriever is not ready: " + ", ".join(reasons))
    return retriever


def _hybrid_ranks(
    retriever: HybridPolicyRetriever, queries: list[EvaluationQuery], *, limit: int
) -> list[int | None]:
    ranks: list[int | None] = []
    for query in queries:
        result = retriever.retrieve(
            PolicyRetrievalRequest(
                query=query.question,
                cohort=query.cohort,
                program_ids=query.program_ids,
                college_ids=query.college_ids,
                topics=query.topics,
                as_of=query.as_of,
                top_k=limit,
            )
        )
        ranks.append(
            next(
                (
                    index
                    for index, item in enumerate(result.candidates, start=1)
                    if item.chunk_id == query.relevant_chunk_id
                ),
                None,
            )
        )
    return ranks


def _lexical_ranks(
    documents: list[dict[str, object]], queries: list[EvaluationQuery], *, limit: int
) -> list[int | None]:
    """Keep the lexical baseline independent from policy relevance gating."""

    retriever = ScopedRetriever(documents, dataset_version="ablation")
    ranks: list[int | None] = []
    for query in queries:
        results = retriever.retrieve(query.question, limit=limit)
        ranks.append(
            next(
                (
                    index
                    for index, item in enumerate(results, start=1)
                    if item.chunk_id == query.relevant_chunk_id
                ),
                None,
            )
        )
    return ranks


def _metric_gates(metrics: dict[str, float], thresholds: dict[str, float]) -> dict[str, dict[str, object]]:
    return {
        key: {
            "operator": ">=",
            "threshold": thresholds[key],
            "actual": metrics[key],
            "passed": metrics[key] >= thresholds[key],
        }
        for key in METRIC_KEYS
    }


def _write_markdown(output: Path, report: dict[str, object]) -> None:
    results = report["results"]
    assert isinstance(results, dict)
    lines = [
        "# Retrieval ablation",
        "",
        "Only measured variants receive metric values. `skipped` and `unavailable`",
        "variants are deliberately shown as N/A rather than being imputed.",
        "",
        "| Variant | Status | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | Gate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for variant, value in results.items():
        assert isinstance(variant, str) and isinstance(value, dict)
        status = str(value.get("status") or "unknown")
        if status == "measured":
            metrics = {key: float(str(value[key])) for key in METRIC_KEYS}
            lines.append(
                "| {variant} | measured | {recall_at_1:.3f} | {recall_at_5:.3f} | "
                "{recall_at_10:.3f} | {mrr:.3f} | {ndcg_at_10:.3f} | {passed} |".format(
                    variant=variant, passed="pass" if value.get("passed") else "FAIL", **metrics
                )
            )
        else:
            lines.append(
                f"| {variant} | {status} | N/A | N/A | N/A | N/A | N/A | {value.get('reason', '')} |"
            )
    lines.append("")
    lines.append(f"Overall status: **{report['status']}**")
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("eval/reports/retrieval_ablation.json")
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("lexical", "hybrid"),
        default=("lexical", "hybrid"),
    )
    parser.add_argument("--limit", type=_positive_int, default=10)
    parser.add_argument("--hybrid-artifact-root", type=Path, default=Path("artifacts/retrieval"))
    parser.add_argument("--dataset-version")
    parser.add_argument(
        "--hybrid-missing",
        choices=("fail", "skip"),
        default="fail",
        help="behavior when a requested hybrid artifact or local model is unavailable",
    )
    parser.add_argument("--min-recall-at-1", type=_probability, default=0.0)
    parser.add_argument("--min-recall-at-5", type=_probability, default=0.0)
    parser.add_argument("--min-recall-at-10", type=_probability, default=0.8)
    parser.add_argument("--min-mrr", type=_probability, default=0.5)
    parser.add_argument("--min-ndcg-at-10", type=_probability, default=0.5)
    args = parser.parse_args()

    documents, queries = _load_inputs(args.documents, args.queries)
    variants = tuple(dict.fromkeys(args.variants))
    thresholds = {
        "recall_at_1": args.min_recall_at_1,
        "recall_at_5": args.min_recall_at_5,
        "recall_at_10": args.min_recall_at_10,
        "mrr": args.min_mrr,
        "ndcg_at_10": args.min_ndcg_at_10,
    }
    results: dict[str, dict[str, object]] = {}
    unavailable: list[str] = []
    for variant in variants:
        if variant == "lexical":
            metrics = rank_metrics(_lexical_ranks(documents, queries, limit=args.limit), cutoff=args.limit)
        else:
            try:
                retriever = _load_hybrid_retriever(
                    documents,
                    artifact_root=args.hybrid_artifact_root,
                    dataset_version=args.dataset_version,
                )
            except HybridArtifactUnavailable as exc:
                reason = str(exc)
                status = "skipped" if args.hybrid_missing == "skip" else "unavailable"
                results[variant] = {"status": status, "reason": reason, "passed": False}
                unavailable.append(reason)
                continue
            metrics = rank_metrics(_hybrid_ranks(retriever, queries, limit=args.limit), cutoff=args.limit)
        gates = _metric_gates(metrics, thresholds)
        results[variant] = {
            "status": "measured",
            **metrics,
            "gates": gates,
            "passed": all(bool(gate["passed"]) for gate in gates.values()),
        }

    measured = [value for value in results.values() if value.get("status") == "measured"]
    quality_failed = any(not bool(value.get("passed")) for value in measured)
    artifact_failed = bool(unavailable) and args.hybrid_missing == "fail"
    status = (
        "failed"
        if quality_failed or artifact_failed
        else ("passed_with_skips" if unavailable else "passed")
    )
    report = {
        "query_count": len(queries),
        "variants": list(variants),
        "thresholds": thresholds,
        "hybrid_missing": args.hybrid_missing,
        "results": results,
        "status": status,
        "passed": status in {"passed", "passed_with_skips"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(args.output, report)
    print(json.dumps(report, ensure_ascii=False))
    if status == "failed":
        if artifact_failed:
            raise SystemExit("hybrid retrieval artifact is unavailable: " + "; ".join(unavailable))
        raise SystemExit("retrieval ablation quality gate failed")


if __name__ == "__main__":
    main()
