"""Deterministic metrics shared by released evaluation runners."""

from __future__ import annotations

import math
from collections.abc import Iterable


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


__all__ = ["binary_classification_metrics", "rank_metrics"]