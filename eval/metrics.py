"""Deterministic metrics shared by released evaluation runners."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


def rank_metrics(ranks: Iterable[int | None], *, cutoff: int = 10) -> dict[str, float]:
    values = tuple(ranks)
    total = max(1, len(values))
    return {
        "recall_at_1": sum(rank is not None and rank <= 1 for rank in values) / total,
        "recall_at_5": sum(rank is not None and rank <= 5 for rank in values) / total,
        "recall_at_10": sum(rank is not None and rank <= 10 for rank in values) / total,
        "mrr": sum(1.0 / rank for rank in values if rank is not None) / total,
        "ndcg_at_10": sum(
            1.0 / math.log2(rank + 1)
            for rank in values
            if rank is not None and rank <= cutoff
        )
        / total,
    }


def graded_retrieval_metrics(
    rankings: Iterable[Sequence[str]],
    qrels: Iterable[Mapping[str, float]],
    *,
    cutoffs: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """Score ranked document ids against multi-document graded relevance.

    ``qrels`` is deliberately a mapping rather than a single gold id: a query
    may have several relevant source sections and their values are gains (for
    example ``3`` for a direct answer and ``1`` for useful supporting context).
    Every query must have at least one positive-gain judgment.  That invariant
    prevents an empty or malformed holdout from turning into a deceptively
    perfect metric.
    """

    normalized_cutoffs = tuple(sorted(set(int(value) for value in cutoffs)))
    if not normalized_cutoffs or any(value < 1 for value in normalized_cutoffs):
        raise ValueError("graded retrieval cutoffs must contain positive integers")
    pairs = tuple(zip(rankings, qrels, strict=True))
    if not pairs:
        raise ValueError("graded retrieval metrics require at least one query")

    recall_values: dict[int, list[float]] = {cutoff: [] for cutoff in normalized_cutoffs}
    reciprocal_ranks: list[float] = []
    ndcg_values: list[float] = []
    for raw_ranking, raw_qrel in pairs:
        ranking = tuple(str(identifier) for identifier in raw_ranking)
        if len(set(ranking)) != len(ranking):
            raise ValueError("retrieval rankings must not contain duplicate document ids")
        qrel = {str(identifier): float(gain) for identifier, gain in raw_qrel.items()}
        if not qrel or not any(gain > 0 for gain in qrel.values()):
            raise ValueError("each retrieval query needs at least one positive relevance label")
        relevant_count = sum(gain > 0 for gain in qrel.values())
        for cutoff in normalized_cutoffs:
            found = sum(qrel.get(identifier, 0.0) > 0 for identifier in ranking[:cutoff])
            recall_values[cutoff].append(found / relevant_count)

        first_rank = next(
            (rank for rank, identifier in enumerate(ranking, start=1) if qrel.get(identifier, 0.0) > 0),
            None,
        )
        reciprocal_ranks.append(1.0 / first_rank if first_rank is not None else 0.0)

        cutoff = normalized_cutoffs[-1]
        dcg = sum(
            (2.0 ** qrel[identifier] - 1.0) / math.log2(rank + 1)
            for rank, identifier in enumerate(ranking[:cutoff], start=1)
            if qrel.get(identifier, 0.0) > 0
        )
        ideal_gains = sorted((gain for gain in qrel.values() if gain > 0), reverse=True)[:cutoff]
        ideal_dcg = sum(
            (2.0 ** gain - 1.0) / math.log2(rank + 1)
            for rank, gain in enumerate(ideal_gains, start=1)
        )
        ndcg_values.append(dcg / ideal_dcg if ideal_dcg else 0.0)

    result = {
        f"recall_at_{cutoff}": sum(values) / len(values)
        for cutoff, values in recall_values.items()
    }
    result["mrr"] = sum(reciprocal_ranks) / len(reciprocal_ranks)
    result[f"ndcg_at_{normalized_cutoffs[-1]}"] = sum(ndcg_values) / len(ndcg_values)
    return result


def metric_gate(
    name: str,
    actual: float | None,
    threshold: float,
    *,
    sample_count: int,
    operator: str = ">=",
) -> dict[str, object]:
    """Return a fail-closed quality gate for one measured metric.

    Missing values and zero-sample metrics are ``missing_data`` failures.  In
    particular, they are never represented as an automatically passing N/A
    gate; a release must either provide the required labels or fail loudly.
    """

    if operator not in {">=", "<="}:
        raise ValueError(f"unsupported metric gate operator: {operator}")
    if actual is None or sample_count <= 0:
        measured = False
        passed = False
    else:
        measured = True
        passed = actual >= threshold if operator == ">=" else actual <= threshold
    return {
        "name": name,
        "operator": operator,
        "threshold": threshold,
        "actual": actual,
        "sample_count": sample_count,
        "status": "measured" if measured else "missing_data",
        "passed": passed,
    }


def binary_classification_metrics(
    expected: Iterable[bool], observed: Iterable[bool]
) -> dict[str, float]:
    pairs = tuple(zip(expected, observed, strict=True))
    tp = sum(actual and prediction for actual, prediction in pairs)
    fp = sum(not actual and prediction for actual, prediction in pairs)
    fn = sum(actual and not prediction for actual, prediction in pairs)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
    }


__all__ = [
    "binary_classification_metrics",
    "graded_retrieval_metrics",
    "metric_gate",
    "rank_metrics",
]
