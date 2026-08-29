"""Measure lexical and artifact-backed hybrid retrieval without inventing results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from eval.candidate_release import (
    CandidateEvaluationError,
    load_candidate_evaluation_context,
)
from eval.holdout import HoldoutContractError, load_holdout_manifest
from eval.metrics import graded_retrieval_metrics, metric_gate
from eval.promotion_policy import (
    PROMOTION_POLICY_SHA256,
    PROMOTION_POLICY_VERSION,
    REPORT_CONTRACT_VERSION,
    RETRIEVAL_EVALUATION_CONFIG,
    PromotionPolicyError,
    validate_retrieval_report,
)
from retrieval.dense import DenseFaissIndex
from retrieval.hybrid import CandidateReranker, DenseIndex, HybridPolicyRetriever
from retrieval.index import artifact_directory, load_manifest
from retrieval.lexical import tokenize
from retrieval.models import PolicyRetrievalRequest
from retrieval.reranker import CrossEncoderReranker
from retrieval.scope import policy_scope_matches
from retrieval.scoring import RetrievedCandidate

METRIC_KEYS = ("recall_at_1", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10")


class HybridArtifactUnavailable(RuntimeError):
    """The requested hybrid run cannot be measured from local, valid artifacts."""


@dataclass(frozen=True)
class EvaluationQuery:
    identifier: str
    question: str
    relevance: tuple[tuple[str, float], ...]
    cohort: int | None = None
    program_ids: tuple[str, ...] = ()
    college_ids: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    as_of: str | None = None
    scope_label: str = ""
    hard_negative_chunk_ids: tuple[str, ...] = ()

    @property
    def qrel(self) -> dict[str, float]:
        return dict(self.relevance)


class _DeterministicArtifactDenseIndex:
    """Artifact-backed dense double used only by the checked-in test fixture.

    This is deliberately explicit and local: vectors and query vectors are
    loaded from the fixture artifact, so CI exercises artifact validation,
    scoped dense ranking and hybrid fusion without downloading model weights.
    """

    def __init__(
        self,
        *,
        chunk_ids: tuple[str, ...],
        vectors: Mapping[str, Sequence[float]],
        query_vectors: Mapping[str, Sequence[float]],
    ) -> None:
        if not chunk_ids or len(set(chunk_ids)) != len(chunk_ids):
            raise HybridArtifactUnavailable("deterministic dense artifact ids are empty or duplicated")
        matrix: dict[str, np.ndarray] = {}
        dimension: int | None = None
        for identifier in chunk_ids:
            raw = np.asarray(vectors.get(identifier), dtype="float32")
            if raw.ndim != 1 or not len(raw) or not np.isfinite(raw).all():
                raise HybridArtifactUnavailable(
                    f"deterministic dense vector is invalid: {identifier}"
                )
            norm = float(np.linalg.norm(raw))
            if not np.isfinite(norm) or norm <= 0.0:
                raise HybridArtifactUnavailable(
                    f"deterministic dense vector is zero or non-finite: {identifier}"
                )
            if dimension is None:
                dimension = int(raw.shape[0])
            if raw.shape != (dimension,):
                raise HybridArtifactUnavailable("deterministic dense vectors have mixed dimensions")
            matrix[identifier] = raw / norm
        assert dimension is not None
        self.chunk_ids = chunk_ids
        self.dimension = dimension
        self._vectors = matrix
        self._query_vectors = {
            str(query): self._normalize(value, dimension, f"query:{query}")
            for query, value in query_vectors.items()
        }

    @staticmethod
    def _normalize(value: Sequence[float], dimension: int, label: str) -> np.ndarray:
        raw = np.asarray(value, dtype="float32")
        if raw.shape != (dimension,) or not np.isfinite(raw).all():
            raise HybridArtifactUnavailable(f"deterministic vector has invalid shape: {label}")
        norm = float(np.linalg.norm(raw))
        if not np.isfinite(norm) or norm <= 0.0:
            raise HybridArtifactUnavailable(f"deterministic vector is zero: {label}")
        return raw / norm

    def vector_for(self, chunk_id: str) -> np.ndarray | None:
        return self._vectors.get(chunk_id)

    def _query_vector(self, query: str) -> np.ndarray:
        known = self._query_vectors.get(query)
        if known is not None:
            return known
        # Unknown queries still have deterministic behavior, but a fixture
        # should list every evaluated query so a missing vector is visible.
        vector = np.zeros(self.dimension, dtype="float32")
        for token in tokenize(query):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dimension] += 1.0
        return self._normalize(vector, self.dimension, f"query:{query}")

    def rank(
        self,
        query: str,
        documents: dict[str, dict[str, object]],
        candidate_ids: set[str],
        *,
        limit: int,
    ) -> tuple[RetrievedCandidate, ...]:
        if limit <= 0:
            return ()
        query_vector = self._query_vector(query)
        values = sorted(
            (
                identifier,
                float(self._vectors[identifier] @ query_vector),
            )
            for identifier in candidate_ids
            if identifier in self._vectors and identifier in documents
        )
        values.sort(key=lambda item: (-item[1], item[0]))
        return tuple(
            RetrievedCandidate(
                chunk_id=identifier,
                text=str(documents[identifier].get("text", "")),
                metadata=documents[identifier],
                dense_score=score,
                dense_rank=rank,
            )
            for rank, (identifier, score) in enumerate(values[:limit], start=1)
        )


class _DeterministicArtifactReranker:
    """Local deterministic reranker for the explicitly marked test fixture."""

    def __init__(self, scores: Mapping[str, Mapping[str, float]]) -> None:
        self._scores = {
            str(query): {str(identifier): float(score) for identifier, score in values.items()}
            for query, values in scores.items()
        }

    def rerank(
        self, query: str, candidates: Iterable[RetrievedCandidate]
    ) -> tuple[RetrievedCandidate, ...]:
        configured = self._scores.get(query, {})
        values = tuple(
            candidate.model_copy(
                update={
                    "reranker_score": float(
                        configured.get(candidate.chunk_id, candidate.dense_score or 0.0)
                    )
                }
            )
            for candidate in candidates
        )
        return tuple(
            sorted(values, key=lambda item: (-(item.reranker_score or 0.0), item.chunk_id))
        )


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


def _required_scope_label(item: dict[str, object], index: int) -> str:
    value = item.get("scope_label")
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"retrieval query {index}: scope_label is required")
    return value.strip()


def _relevance_labels(
    item: dict[str, object], *, known_ids: set[str], index: int
) -> tuple[tuple[str, float], ...]:
    raw = item.get("relevance")
    if not isinstance(raw, dict) or not raw:
        raise SystemExit(
            f"retrieval query {index}: relevance must be a non-empty object of chunk_id to grade"
        )
    values: list[tuple[str, float]] = []
    for raw_identifier, raw_grade in raw.items():
        if not isinstance(raw_identifier, str) or not raw_identifier.strip():
            raise SystemExit(f"retrieval query {index}: relevance ids must be non-empty strings")
        identifier = raw_identifier.strip()
        if identifier not in known_ids:
            raise SystemExit(f"retrieval query {index} references unknown chunk_id: {identifier}")
        if isinstance(raw_grade, bool) or not isinstance(raw_grade, (int, float)):
            raise SystemExit(f"retrieval query {index}: relevance grades must be numeric")
        grade = float(raw_grade)
        if not np.isfinite(grade) or grade <= 0.0 or grade > 3.0:
            raise SystemExit(f"retrieval query {index}: relevance grades must be in (0, 3]")
        values.append((identifier, grade))
    if not values:
        raise SystemExit(f"retrieval query {index}: relevance needs positive grades")
    return tuple(sorted(values))


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
    required_document_labels = (
        "scope_label",
        "review_status",
        "status",
        "cohort",
        "college_id",
        "program_ids",
        "topics",
        "effective_from",
    )
    for document_index, document in enumerate(documents, start=1):
        missing = [field for field in required_document_labels if field not in document]
        if missing:
            raise SystemExit(
                f"retrieval document {document_index} is missing scope/trust labels: {missing}"
            )
        if not isinstance(document["scope_label"], str) or not str(document["scope_label"]).strip():
            raise SystemExit(f"retrieval document {document_index}: scope_label is required")
        if not isinstance(document["program_ids"], (list, tuple)):
            raise SystemExit(f"retrieval document {document_index}: program_ids must be a list")
        if not isinstance(document["topics"], (list, tuple)):
            raise SystemExit(f"retrieval document {document_index}: topics must be a list")

    raw_queries = json.loads(queries_path.read_text(encoding="utf-8"))
    if not isinstance(raw_queries, list) or not raw_queries:
        raise SystemExit("retrieval queries must be a non-empty JSON array")
    queries: list[EvaluationQuery] = []
    known_ids = set(document_ids)
    for index, item in enumerate(raw_queries, start=1):
        if not isinstance(item, dict):
            raise SystemExit(f"retrieval query {index} must be an object")
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise SystemExit(f"retrieval query {index}: id is required")
        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise SystemExit(f"retrieval query {index} needs a non-empty question")
        relevance = _relevance_labels(item, known_ids=known_ids, index=index)
        scope = item.get("scope")
        if not isinstance(scope, dict):
            raise SystemExit(f"retrieval query {index}: scope is required and must be an object")
        cohort = scope.get("cohort")
        if cohort is not None and (isinstance(cohort, bool) or not isinstance(cohort, int)):
            raise SystemExit(f"retrieval query {index}: cohort must be an integer or null")
        as_of = scope.get("as_of")
        if as_of is not None and (not isinstance(as_of, str) or not as_of.strip()):
            raise SystemExit(f"retrieval query {index}: as_of must be a non-empty string or null")
        hard_negatives = item.get("hard_negative_chunk_ids")
        if not isinstance(hard_negatives, list) or any(
            not isinstance(value, str) or not value.strip() for value in hard_negatives
        ):
            raise SystemExit(
                f"retrieval query {index}: hard_negative_chunk_ids must be a list of strings"
            )
        hard_negative_values = tuple(str(value).strip() for value in hard_negatives)
        if len(set(hard_negative_values)) != len(hard_negative_values):
            raise SystemExit(f"retrieval query {index}: hard-negative ids must be unique")
        if not set(hard_negative_values).issubset(known_ids):
            raise SystemExit(f"retrieval query {index}: hard-negative id is unknown")
        if set(hard_negative_values).intersection(identifier for identifier, _grade in relevance):
            raise SystemExit(f"retrieval query {index}: a document cannot be both relevant and hard negative")
        queries.append(
            EvaluationQuery(
                identifier=identifier.strip(),
                question=question.strip(),
                relevance=relevance,
                cohort=cohort,
                program_ids=_string_tuple(scope.get("program_ids"), "scope.program_ids", index),
                college_ids=_string_tuple(scope.get("college_ids"), "scope.college_ids", index),
                topics=_string_tuple(scope.get("topics"), "scope.topics", index),
                as_of=as_of.strip() if isinstance(as_of, str) else None,
                scope_label=_required_scope_label(item, index),
                hard_negative_chunk_ids=hard_negative_values,
            )
        )
    return documents, queries


def _artifact_hash(directory: Path, manifest: Mapping[str, object], filename: str, field: str) -> None:
    path = directory / filename
    expected = manifest.get(field)
    if not path.is_file():
        raise HybridArtifactUnavailable(f"hybrid artifact file is missing: {filename}")
    if not isinstance(expected, str) or len(expected) != 64:
        raise HybridArtifactUnavailable(f"hybrid manifest lacks {field}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise HybridArtifactUnavailable(f"hybrid artifact {filename} hash mismatch")


def _load_json_mapping(directory: Path, filename: str) -> dict[str, object]:
    try:
        value = json.loads((directory / filename).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HybridArtifactUnavailable(f"hybrid artifact JSON is invalid: {filename}") from exc
    if not isinstance(value, dict):
        raise HybridArtifactUnavailable(f"hybrid artifact JSON must be an object: {filename}")
    return dict(value)


def _load_eval_manifest(
    artifact_root: Path, dataset_version: str
) -> tuple[Path, dict[str, object]]:
    """Load regular artifacts or the explicitly marked deterministic fixture."""

    try:
        return load_manifest(artifact_root, dataset_version)
    except Exception as exc:
        directory = artifact_directory(artifact_root, dataset_version)
        path = directory / "retrieval_manifest.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as load_exc:
            raise HybridArtifactUnavailable(str(exc)) from load_exc
        if not isinstance(value, dict) or not bool(value.get("test_fixture")):
            raise HybridArtifactUnavailable(str(exc)) from exc
        return directory, dict(value)


def _load_hybrid_retriever(
    documents: list[dict[str, object]], *, artifact_root: Path, dataset_version: str | None
) -> HybridPolicyRetriever:
    if not dataset_version:
        raise HybridArtifactUnavailable("hybrid evaluation requires --dataset-version")
    try:
        directory, manifest = _load_eval_manifest(artifact_root, dataset_version)
        if str(manifest.get("retrieval_mode") or "") != "hybrid":
            raise HybridArtifactUnavailable("retrieval manifest does not declare hybrid mode")
        if str(manifest.get("dataset_version") or "") != dataset_version:
            raise HybridArtifactUnavailable("retrieval manifest dataset version does not match --dataset-version")
        model_name = str(manifest.get("embedding_model") or "")
        reranker_name = str(manifest.get("reranker_model") or "")
        dimension = int(str(manifest.get("embedding_dimension") or 0))
        if not model_name or not reranker_name or dimension <= 0:
            raise HybridArtifactUnavailable("retrieval manifest lacks hybrid model metadata")
        _artifact_hash(directory, manifest, "documents.jsonl", "documents_sha256")
        _artifact_hash(directory, manifest, "doc_ids.json", "doc_ids_sha256")
        raw_chunk_ids = json.loads((directory / "doc_ids.json").read_text(encoding="utf-8"))
        if not isinstance(raw_chunk_ids, list) or not raw_chunk_ids:
            raise HybridArtifactUnavailable("hybrid artifact doc_ids.json is missing or invalid")
        expected_chunk_ids = tuple(str(value) for value in raw_chunk_ids)
        actual_chunk_ids = tuple(str(document["chunk_id"]) for document in documents)
        if expected_chunk_ids != actual_chunk_ids:
            raise HybridArtifactUnavailable(
                "hybrid artifact chunk IDs/order do not match the evaluated documents"
            )
        dense: DenseIndex
        reranker: CandidateReranker
        if bool(manifest.get("test_fixture")):
            if model_name != "deterministic-test-encoder" or reranker_name != "deterministic-test-reranker":
                raise HybridArtifactUnavailable(
                    "test fixture hybrid artifacts must declare deterministic local models"
                )
            _artifact_hash(directory, manifest, "vectors.json", "vectors_sha256")
            _artifact_hash(directory, manifest, "query_vectors.json", "query_vectors_sha256")
            _artifact_hash(directory, manifest, "reranker_scores.json", "reranker_scores_sha256")
            raw_vectors = _load_json_mapping(directory, "vectors.json")
            raw_query_vectors = _load_json_mapping(directory, "query_vectors.json")
            raw_scores = _load_json_mapping(directory, "reranker_scores.json")
            vectors = {
                str(identifier): cast(Sequence[float], value)
                for identifier, value in raw_vectors.items()
                if isinstance(value, list)
            }
            query_vectors = {
                str(query): cast(Sequence[float], value)
                for query, value in raw_query_vectors.items()
                if isinstance(value, list)
            }
            scores = {
                str(query): cast(Mapping[str, float], value)
                for query, value in raw_scores.items()
                if isinstance(value, dict)
            }
            if len(vectors) != len(raw_vectors) or len(query_vectors) != len(raw_query_vectors):
                raise HybridArtifactUnavailable("deterministic vector artifact values must be arrays")
            if len(scores) != len(raw_scores) or any(
                not all(isinstance(score, (int, float)) and not isinstance(score, bool) for score in value.values())
                for value in raw_scores.values()
                if isinstance(value, dict)
            ):
                raise HybridArtifactUnavailable("deterministic reranker scores are invalid")
            dense = _DeterministicArtifactDenseIndex(
                chunk_ids=expected_chunk_ids,
                vectors=vectors,
                query_vectors=query_vectors,
            )
            reranker = _DeterministicArtifactReranker(scores)
        else:
            _artifact_hash(directory, manifest, "vectors.npy", "vectors_sha256")
            _artifact_hash(directory, manifest, "faiss.index", "index_sha256")
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


def _request(query: EvaluationQuery, *, limit: int) -> PolicyRetrievalRequest:
    return PolicyRetrievalRequest(
        query=query.question,
        cohort=query.cohort,
        program_ids=query.program_ids,
        college_ids=query.college_ids,
        topics=query.topics,
        as_of=query.as_of,
        top_k=limit,
    )


def _rankings(
    retriever: HybridPolicyRetriever, queries: list[EvaluationQuery], *, limit: int
) -> list[tuple[str, ...]]:
    rankings: list[tuple[str, ...]] = []
    for query in queries:
        result = retriever.retrieve(_request(query, limit=limit))
        rankings.append(tuple(item.chunk_id for item in result.candidates))
    return rankings


def _lexical_ranks(
    documents: list[dict[str, object]], queries: list[EvaluationQuery], *, limit: int
) -> list[tuple[str, ...]]:
    """Run lexical through the same policy scope gate as hybrid retrieval."""

    retriever = HybridPolicyRetriever(documents, mode="lexical", dataset_version="ablation")
    return _rankings(retriever, queries, limit=limit)


def _scope_violation(
    document: dict[str, object], query: EvaluationQuery
) -> bool:
    """Use the same fail-closed scope contract as runtime retrieval."""

    return not policy_scope_matches(
        document,
        cohort=query.cohort,
        program_ids=query.program_ids,
        college_ids=query.college_ids,
    )

def _scope_and_hard_negative_metrics(
    rankings: list[tuple[str, ...]],
    queries: list[EvaluationQuery],
    documents: Mapping[str, dict[str, object]],
    *,
    cutoff: int,
) -> dict[str, float]:
    if len(rankings) != len(queries) or not queries:
        raise ValueError("scope metrics require one ranking per labeled query")
    hard_negative_rates: list[float] = []
    scope_violation_rates: list[float] = []
    for ranking, query in zip(rankings, queries, strict=True):
        top = ranking[:cutoff]
        hard_negatives = set(query.hard_negative_chunk_ids)
        if hard_negatives:
            hard_negative_rates.append(
                sum(identifier in hard_negatives for identifier in top) / len(hard_negatives)
            )
        scope_violation_rates.append(
            sum(
                _scope_violation(documents[identifier], query)
                for identifier in top
                if identifier in documents
            )
            / max(1, len(top))
        )
    return {
        "hard_negative_rate_at_cutoff": (
            sum(hard_negative_rates) / len(hard_negative_rates)
            if hard_negative_rates
            else 0.0
        ),
        "scope_violation_rate": sum(scope_violation_rates) / len(scope_violation_rates),
    }


def _metric_gates(
    metrics: dict[str, float], thresholds: dict[str, float], *, sample_count: int
) -> dict[str, dict[str, object]]:
    return {
        key: metric_gate(
            key,
            metrics.get(key),
            thresholds[key],
            sample_count=sample_count,
            operator=">=",
        )
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
    parser.add_argument("--documents", type=Path)
    parser.add_argument("--queries", type=Path)
    parser.add_argument(
        "--candidate-release-manifest",
        type=Path,
        help="verified candidate entry; derives corpus/index/commit for promotion evidence",
    )
    parser.add_argument(
        "--holdout-manifest",
        type=Path,
        help="frozen manifest whose hashes and dataset labels must cover inputs",
    )
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
    parser.add_argument("--hybrid-artifact-root", type=Path)
    parser.add_argument("--dataset-version")
    parser.add_argument("--model-id", default="deterministic-test-encoder")
    parser.add_argument("--artifact-id", default="none")
    parser.add_argument("--commit", default=os.environ.get("GITHUB_SHA", "working-tree"))
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
    parser.add_argument("--max-hard-negative-rate", type=_probability, default=0.0)
    parser.add_argument("--max-scope-violation-rate", type=_probability, default=0.0)
    args = parser.parse_args()

    candidate = None
    promotion_eligible = args.candidate_release_manifest is not None
    if promotion_eligible:
        if args.holdout_manifest is None:
            raise SystemExit("candidate evaluation requires --holdout-manifest")
        if (
            args.documents is not None
            or args.queries is not None
            or args.hybrid_artifact_root is not None
            or args.dataset_version is not None
            or args.artifact_id != "none"
            or list(dict.fromkeys(args.variants)) != ["lexical", "hybrid"]
            or args.limit != 10
            or args.hybrid_missing != "fail"
        ):
            raise SystemExit(
                "candidate evaluation derives corpus/queries/artifact/version and uses "
                "the frozen promotion retrieval config"
            )
        try:
            candidate = load_candidate_evaluation_context(
                args.candidate_release_manifest,
                args.holdout_manifest,
            )
        except CandidateEvaluationError as exc:
            raise SystemExit(f"candidate evaluation contract failed: {exc}") from exc
        holdout = candidate.holdout
        documents_path = candidate.documents_path
        queries_path = holdout.root / holdout.inputs["retrieval_queries"].path
        artifact_root = candidate.retrieval_root
        dataset_version: str | None = holdout.dataset_version
    else:
        if args.documents is None or args.queries is None:
            raise SystemExit(
                "diagnostic evaluation requires --documents and --queries"
            )
        documents_path = args.documents
        queries_path = args.queries
        artifact_root = args.hybrid_artifact_root or Path("artifacts/retrieval")
        holdout = None
        if args.holdout_manifest is not None:
            try:
                holdout = load_holdout_manifest(args.holdout_manifest)
                holdout.verify_role("retrieval_documents", documents_path)
                holdout.verify_role("retrieval_queries", queries_path)
            except HoldoutContractError as exc:
                raise SystemExit(f"holdout contract failed: {exc}") from exc
            if args.dataset_version and args.dataset_version != holdout.dataset_version:
                raise SystemExit(
                    "--dataset-version does not match holdout manifest: "
                    f"{args.dataset_version!r} != {holdout.dataset_version!r}"
                )
        dataset_version = args.dataset_version or (
            holdout.dataset_version if holdout is not None else None
        )
    documents, queries = _load_inputs(documents_path, queries_path)
    if holdout is not None:
        holdout.verify_role_count("retrieval_documents", len(documents))
        holdout.verify_role_count("retrieval_queries", len(queries))
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
    document_map = {str(document["chunk_id"]): document for document in documents}
    for variant in variants:
        if variant == "lexical":
            rankings = _lexical_ranks(documents, queries, limit=args.limit)
        else:
            try:
                retriever = _load_hybrid_retriever(
                    documents,
                    artifact_root=artifact_root,
                    dataset_version=dataset_version,
                )
            except HybridArtifactUnavailable as exc:
                reason = str(exc)
                # ``skip`` is retained as a diagnostic display option, but a
                # requested hybrid run is never allowed to pass by omission.
                status = "unavailable"
                results[variant] = {
                    "status": status,
                    "reason": reason,
                    "passed": False,
                }
                unavailable.append(reason)
                continue
            rankings = _rankings(retriever, queries, limit=args.limit)
        metrics = graded_retrieval_metrics(
            rankings,
            [query.qrel for query in queries],
            cutoffs=(1, 5, 10),
        )
        scope_metrics = _scope_and_hard_negative_metrics(
            rankings,
            queries,
            document_map,
            cutoff=args.limit,
        )
        gates = _metric_gates(metrics, thresholds, sample_count=len(queries))
        hard_negative_samples = sum(bool(query.hard_negative_chunk_ids) for query in queries)
        gates["hard_negative_rate_at_cutoff"] = metric_gate(
            "hard_negative_rate_at_cutoff",
            scope_metrics["hard_negative_rate_at_cutoff"],
            args.max_hard_negative_rate,
            sample_count=hard_negative_samples,
            operator="<=",
        )
        gates["scope_violation_rate"] = metric_gate(
            "scope_violation_rate",
            scope_metrics["scope_violation_rate"],
            args.max_scope_violation_rate,
            sample_count=len(queries),
            operator="<=",
        )
        results[variant] = {
            "status": "measured",
            **metrics,
            **scope_metrics,
            "hard_negative_labeled_queries": hard_negative_samples,
            "gates": gates,
            "passed": all(bool(gate["passed"]) for gate in gates.values()),
        }

    measured = [value for value in results.values() if value.get("status") == "measured"]
    quality_failed = any(not bool(value.get("passed")) for value in measured)
    artifact_failed = bool(unavailable)
    status = "failed" if quality_failed or artifact_failed else "passed"
    input_hashes = {
        "documents_sha256": hashlib.sha256(documents_path.read_bytes()).hexdigest(),
        "queries_sha256": hashlib.sha256(queries_path.read_bytes()).hexdigest(),
    }
    if candidate is not None:
        evaluation_config = dict(RETRIEVAL_EVALUATION_CONFIG)
        provenance: dict[str, object] = {
            "release_subject": candidate.release_subject,
            "holdout": candidate.holdout.release_lock(),
            "runtime": candidate.runtime_provenance(),
            "evaluator_git": candidate.evaluator_git,
        }
    else:
        evaluation_config = {
            "variants": list(variants),
            "limit": args.limit,
            "hard_negative_cutoff": args.limit,
            "metric_cutoffs": [1, 5, 10],
            "hybrid_missing": args.hybrid_missing,
        }
        provenance = {
            "release_subject": None,
            "holdout": holdout.provenance() if holdout is not None else None,
            "runtime": {
                "dataset_version": dataset_version,
                "retrieval_mode": "hybrid" if "hybrid" in variants else "lexical",
                **input_hashes,
            },
            "evaluator_git": {
                "commit": args.commit,
                "dirty": None,
            },
            "diagnostic_input": {
                "documents": str(documents_path),
                "queries": str(queries_path),
            },
            "diagnostic_artifact_label": args.artifact_id,
        }
    report = {
        "report_contract_version": REPORT_CONTRACT_VERSION,
        "promotion_policy_version": PROMOTION_POLICY_VERSION,
        "promotion_policy_sha256": PROMOTION_POLICY_SHA256,
        "promotion_eligible": promotion_eligible,
        "evaluation_config": evaluation_config,
        "query_count": len(queries),
        "document_count": len(documents),
        "variants": list(variants),
        "thresholds": thresholds,
        "hybrid_missing": args.hybrid_missing,
        "results": results,
        "status": status,
        "passed": status == "passed",
        "provenance": provenance,
    }
    if promotion_eligible:
        try:
            validate_retrieval_report(
                report,
                expected_query_count=len(queries),
                expected_document_count=len(documents),
            )
        except PromotionPolicyError as exc:
            raise SystemExit(f"retrieval report is not promotion-eligible: {exc}") from exc
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
