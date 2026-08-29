"""Versioned, non-configurable minimum policy for production promotion reports."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass

from storage.json_contract import canonical_json

REPORT_CONTRACT_VERSION = "2"
PROMOTION_POLICY_VERSION = "1"

AGENT_EVALUATION_CONFIG: dict[str, object] = {
    "retrieval_mode": "hybrid",
    "reranker_min_score": 0.5,
    "session_backend": "memory",
}
RETRIEVAL_EVALUATION_CONFIG: dict[str, object] = {
    "variants": ["lexical", "hybrid"],
    "limit": 10,
    "hard_negative_cutoff": 10,
    "metric_cutoffs": [1, 5, 10],
    "hybrid_missing": "fail",
}


class PromotionPolicyError(ValueError):
    """Raised when a report is incomplete, self-inconsistent, or too permissive."""


@dataclass(frozen=True)
class GateRule:
    operator: str
    threshold: float
    metric_path: tuple[str, ...]


AGENT_GATE_RULES = {
    "intent_accuracy": GateRule(">=", 1.0, ("intent_accuracy",)),
    "plan_exact_match": GateRule(">=", 1.0, ("plan_exact_match",)),
    "tool_precision": GateRule(">=", 1.0, ("tool_precision",)),
    "tool_recall": GateRule(">=", 1.0, ("tool_recall",)),
    "answer_containment": GateRule(">=", 1.0, ("answer_containment",)),
    "safe_rejection_f1": GateRule(">=", 1.0, ("safe_rejection", "f1")),
    "scope_pollution_rate": GateRule("<=", 0.0, ("scope_pollution_rate",)),
}

RETRIEVAL_GATE_RULES = {
    "recall_at_1": GateRule(">=", 0.0, ("recall_at_1",)),
    "recall_at_5": GateRule(">=", 0.0, ("recall_at_5",)),
    "recall_at_10": GateRule(">=", 0.8, ("recall_at_10",)),
    "mrr": GateRule(">=", 0.5, ("mrr",)),
    "ndcg_at_10": GateRule(">=", 0.5, ("ndcg_at_10",)),
    "hard_negative_rate_at_cutoff": GateRule(
        "<=", 0.0, ("hard_negative_rate_at_cutoff",)
    ),
    "scope_violation_rate": GateRule("<=", 0.0, ("scope_violation_rate",)),
}

_GATE_KEYS = frozenset(
    {"name", "operator", "threshold", "actual", "sample_count", "status", "passed"}
)


def _policy_payload() -> dict[str, object]:
    def values(rules: Mapping[str, GateRule]) -> dict[str, object]:
        return {
            name: {
                "operator": rule.operator,
                "threshold": rule.threshold,
                "metric_path": list(rule.metric_path),
            }
            for name, rule in sorted(rules.items())
        }

    return {
        "promotion_policy_version": PROMOTION_POLICY_VERSION,
        "report_contract_version": REPORT_CONTRACT_VERSION,
        "agent": values(AGENT_GATE_RULES),
        "retrieval": values(RETRIEVAL_GATE_RULES),
        "agent_evaluation_config": AGENT_EVALUATION_CONFIG,
        "retrieval_evaluation_config": RETRIEVAL_EVALUATION_CONFIG,
    }


PROMOTION_POLICY_SHA256 = hashlib.sha256(canonical_json(_policy_payload())).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PromotionPolicyError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, object], expected: frozenset[str], label: str) -> None:
    observed = frozenset(str(key) for key in value)
    if observed != expected:
        raise PromotionPolicyError(
            f"{label} fields mismatch; missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)}"
        )


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PromotionPolicyError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise PromotionPolicyError(f"{label} must be a finite number")
    if number < 0.0 or number > 1.0:
        raise PromotionPolicyError(f"{label} must be between 0 and 1")
    return number


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PromotionPolicyError(f"{label} must be a positive integer")
    return value


def _metric(report: Mapping[str, object], path: tuple[str, ...], label: str) -> float:
    value: object = report
    for part in path:
        value = _mapping(value, label).get(part)
    return _finite(value, label)


def _validate_gate_set(
    value: object,
    *,
    rules: Mapping[str, GateRule],
    metrics: Mapping[str, object],
    sample_counts: Mapping[str, int],
    label: str,
) -> None:
    gates = _mapping(value, label)
    expected_names = frozenset(rules)
    observed_names = frozenset(str(key) for key in gates)
    if observed_names != expected_names:
        raise PromotionPolicyError(
            f"{label} names mismatch; missing={sorted(expected_names - observed_names)}, "
            f"unknown={sorted(observed_names - expected_names)}"
        )
    for name, rule in rules.items():
        gate = _mapping(gates[name], f"{label}.{name}")
        _exact_keys(gate, _GATE_KEYS, f"{label}.{name}")
        if gate.get("name") != name:
            raise PromotionPolicyError(f"{label}.{name}.name does not match its key")
        if gate.get("operator") != rule.operator:
            raise PromotionPolicyError(f"{label}.{name} uses the wrong operator")
        threshold = _finite(gate.get("threshold"), f"{label}.{name}.threshold")
        if rule.operator == ">=" and threshold < rule.threshold:
            raise PromotionPolicyError(f"{label}.{name} weakens the promotion threshold")
        if rule.operator == "<=" and threshold > rule.threshold:
            raise PromotionPolicyError(f"{label}.{name} weakens the promotion threshold")
        actual = _finite(gate.get("actual"), f"{label}.{name}.actual")
        expected_actual = _metric(metrics, rule.metric_path, f"{label}.{name} metric")
        if actual != expected_actual:
            raise PromotionPolicyError(f"{label}.{name}.actual differs from report metric")
        if gate.get("status") != "measured":
            raise PromotionPolicyError(f"{label}.{name} is not measured")
        sample_count = _positive_int(
            gate.get("sample_count"), f"{label}.{name}.sample_count"
        )
        if sample_count != sample_counts[name]:
            raise PromotionPolicyError(f"{label}.{name} sample_count is not frozen")
        recomputed = actual >= threshold if rule.operator == ">=" else actual <= threshold
        if gate.get("passed") is not recomputed or not recomputed:
            raise PromotionPolicyError(f"{label}.{name} did not pass recomputation")


def validate_agent_report(
    report: Mapping[str, object],
    *,
    expected_question_count: int,
) -> None:
    if report.get("report_contract_version") != REPORT_CONTRACT_VERSION:
        raise PromotionPolicyError("agent report contract version is unsupported")
    if report.get("promotion_policy_version") != PROMOTION_POLICY_VERSION:
        raise PromotionPolicyError("agent promotion policy version is unsupported")
    if report.get("promotion_policy_sha256") != PROMOTION_POLICY_SHA256:
        raise PromotionPolicyError("agent promotion policy digest is invalid")
    if report.get("promotion_eligible") is not True:
        raise PromotionPolicyError("agent report is diagnostic-only")
    if report.get("evaluation_config") != AGENT_EVALUATION_CONFIG:
        raise PromotionPolicyError("agent evaluation config is not promotion policy v1")
    question_count = _positive_int(report.get("question_count"), "agent question_count")
    if question_count != expected_question_count:
        raise PromotionPolicyError("agent question_count differs from restricted holdout")
    sample_counts = {name: question_count for name in AGENT_GATE_RULES}
    _validate_gate_set(
        report.get("gates"),
        rules=AGENT_GATE_RULES,
        metrics=report,
        sample_counts=sample_counts,
        label="agent gates",
    )
    if report.get("passed") is not True:
        raise PromotionPolicyError("agent report did not pass")


def _validate_retrieval_variant(
    name: str,
    value: object,
    *,
    query_count: int,
) -> None:
    variant = _mapping(value, f"retrieval results.{name}")
    if variant.get("status") != "measured":
        raise PromotionPolicyError(f"retrieval variant {name} is not measured")
    hard_negative_count = _positive_int(
        variant.get("hard_negative_labeled_queries"),
        f"retrieval results.{name}.hard_negative_labeled_queries",
    )
    if hard_negative_count > query_count:
        raise PromotionPolicyError(f"retrieval results.{name} has impossible sample counts")
    sample_counts = {
        gate_name: (
            hard_negative_count
            if gate_name == "hard_negative_rate_at_cutoff"
            else query_count
        )
        for gate_name in RETRIEVAL_GATE_RULES
    }
    _validate_gate_set(
        variant.get("gates"),
        rules=RETRIEVAL_GATE_RULES,
        metrics=variant,
        sample_counts=sample_counts,
        label=f"retrieval results.{name}.gates",
    )
    if variant.get("passed") is not True:
        raise PromotionPolicyError(f"retrieval variant {name} did not pass")


def validate_retrieval_report(
    report: Mapping[str, object],
    *,
    expected_query_count: int,
    expected_document_count: int,
) -> None:
    if report.get("report_contract_version") != REPORT_CONTRACT_VERSION:
        raise PromotionPolicyError("retrieval report contract version is unsupported")
    if report.get("promotion_policy_version") != PROMOTION_POLICY_VERSION:
        raise PromotionPolicyError("retrieval promotion policy version is unsupported")
    if report.get("promotion_policy_sha256") != PROMOTION_POLICY_SHA256:
        raise PromotionPolicyError("retrieval promotion policy digest is invalid")
    if report.get("promotion_eligible") is not True:
        raise PromotionPolicyError("retrieval report is diagnostic-only")
    if report.get("evaluation_config") != RETRIEVAL_EVALUATION_CONFIG:
        raise PromotionPolicyError("retrieval evaluation config is not promotion policy v1")
    query_count = _positive_int(report.get("query_count"), "retrieval query_count")
    document_count = _positive_int(report.get("document_count"), "retrieval document_count")
    if query_count != expected_query_count or document_count != expected_document_count:
        raise PromotionPolicyError("retrieval counts differ from restricted holdout")
    variants = report.get("variants")
    if not isinstance(variants, list) or variants != ["lexical", "hybrid"]:
        raise PromotionPolicyError("promotion requires lexical and hybrid variants in order")
    if report.get("hybrid_missing") != "fail":
        raise PromotionPolicyError("promotion cannot skip a missing hybrid artifact")
    results = _mapping(report.get("results"), "retrieval results")
    if frozenset(str(key) for key in results) != frozenset(variants):
        raise PromotionPolicyError("retrieval results and variants differ")
    for name in variants:
        _validate_retrieval_variant(name, results[name], query_count=query_count)
    if report.get("status") != "passed" or report.get("passed") is not True:
        raise PromotionPolicyError("retrieval report did not pass recomputation")


__all__ = [
    "AGENT_EVALUATION_CONFIG",
    "AGENT_GATE_RULES",
    "PROMOTION_POLICY_SHA256",
    "PROMOTION_POLICY_VERSION",
    "REPORT_CONTRACT_VERSION",
    "RETRIEVAL_EVALUATION_CONFIG",
    "RETRIEVAL_GATE_RULES",
    "PromotionPolicyError",
    "validate_agent_report",
    "validate_retrieval_report",
]
