"""Fail-closed contracts for the frozen evaluation runners."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from eval.holdout import HoldoutContractError, load_holdout_manifest
from eval.metrics import metric_gate
from eval.promotion_policy import (
    AGENT_EVALUATION_CONFIG,
    AGENT_GATE_RULES,
    PROMOTION_POLICY_SHA256,
    PROMOTION_POLICY_VERSION,
    REPORT_CONTRACT_VERSION,
    RETRIEVAL_EVALUATION_CONFIG,
    RETRIEVAL_GATE_RULES,
    GateRule,
    PromotionPolicyError,
    validate_agent_report,
    validate_retrieval_report,
)
from eval.run_agent_eval import _expected_operations, _minimum_gate
from eval.run_retrieval_ablation import (
    HybridArtifactUnavailable,
    _load_hybrid_retriever,
    _load_inputs,
)


def test_missing_metric_is_a_failed_missing_data_gate() -> None:
    gate = metric_gate("recall", None, 0.5, sample_count=0)
    assert gate["status"] == "missing_data"
    assert gate["passed"] is False
    assert _minimum_gate("recall", None, 0.5, 0)["passed"] is False


def test_agent_eval_requires_typed_operation_labels() -> None:
    with pytest.raises(SystemExit, match="expected_operations is required"):
        _expected_operations(
            {
                "id": "legacy",
                "question": "q",
                "intent": "general",
                "expected_tools": [],
            },
            1,
        )


def test_agent_eval_compares_typed_args_and_dependency_types() -> None:
    expected = _expected_operations(
        {
            "expected_operations": [
                {
                    "type": "retrieve_policy",
                    "args": {
                        "question": "转专业资格",
                        "cohort": None,
                        "program_ids": [],
                        "college_ids": [],
                        "as_of": None,
                        "topics": ["转专业"],
                        "top_k": 8,
                    },
                    "depends_on": [],
                }
            ]
        },
        1,
    )
    operation = expected[0].as_json()
    args = operation["args"]
    assert operation["type"] == "retrieve_policy"
    assert isinstance(args, dict)
    assert args["topics"] == ["转专业"]


def test_retrieval_input_requires_scope_relevance_and_hard_negative_labels(
    tmp_path: Path,
) -> None:
    documents = tmp_path / "documents.jsonl"
    documents.write_text(
        json.dumps({"chunk_id": "doc-1", "text": "text"}) + "\n", encoding="utf-8"
    )
    queries = tmp_path / "queries.json"
    queries.write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit, match="scope/trust labels"):
        _load_inputs(documents, queries)


def test_holdout_manifest_rejects_tampering(tmp_path: Path) -> None:
    source = Path("eval/holdout/manifest.json")
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(source.read_bytes())
    manifest.with_suffix(".json.sha256").write_text(
        "0" * 64 + "  manifest.json\n", encoding="ascii"
    )
    with pytest.raises(HoldoutContractError, match="hash mismatch"):
        load_holdout_manifest(manifest)


def test_requested_hybrid_without_artifact_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(HybridArtifactUnavailable):
        _load_hybrid_retriever(
            [], artifact_root=tmp_path / "missing", dataset_version="missing-v1"
        )


def test_checked_in_fixture_loader_carries_multi_relevance_and_scope_labels() -> None:
    documents, queries = _load_inputs(
        Path("eval/holdout/retrieval_documents.jsonl"),
        Path("eval/holdout/retrieval_queries.json"),
    )
    assert len(documents) == 4
    assert len(queries[0].relevance) == 2
    assert queries[0].hard_negative_chunk_ids
    assert queries[0].scope_label == "global-2024"


def _policy_metric(value: Mapping[str, object], path: tuple[str, ...]) -> float:
    current: object = value
    for part in path:
        assert isinstance(current, Mapping)
        current = current[part]
    assert isinstance(current, (int, float)) and not isinstance(current, bool)
    return float(current)


def _policy_gates(
    rules: Mapping[str, GateRule],
    metrics: Mapping[str, object],
    *,
    sample_count: int,
    hard_negative_count: int | None = None,
) -> dict[str, object]:
    gates: dict[str, object] = {}
    for name, rule in rules.items():
        actual = _policy_metric(metrics, rule.metric_path)
        count = (
            hard_negative_count
            if name == "hard_negative_rate_at_cutoff"
            else sample_count
        )
        assert count is not None
        gates[name] = {
            "name": name,
            "operator": rule.operator,
            "threshold": rule.threshold,
            "actual": actual,
            "sample_count": count,
            "status": "measured",
            "passed": True,
        }
    return gates


def _agent_promotion_report() -> dict[str, object]:
    report: dict[str, object] = {
        "report_contract_version": REPORT_CONTRACT_VERSION,
        "promotion_policy_version": PROMOTION_POLICY_VERSION,
        "promotion_policy_sha256": PROMOTION_POLICY_SHA256,
        "promotion_eligible": True,
        "evaluation_config": AGENT_EVALUATION_CONFIG,
        "question_count": 20,
        "intent_accuracy": 1.0,
        "plan_exact_match": 1.0,
        "tool_precision": 1.0,
        "tool_recall": 1.0,
        "answer_containment": 1.0,
        "safe_rejection": {"f1": 1.0},
        "scope_pollution_rate": 0.0,
        "passed": True,
    }
    report["gates"] = _policy_gates(AGENT_GATE_RULES, report, sample_count=20)
    return report


def _retrieval_promotion_report() -> dict[str, object]:
    results: dict[str, object] = {}
    for name in ("lexical", "hybrid"):
        variant: dict[str, object] = {
            "status": "measured",
            "recall_at_1": 1.0,
            "recall_at_5": 1.0,
            "recall_at_10": 1.0,
            "mrr": 1.0,
            "ndcg_at_10": 1.0,
            "hard_negative_rate_at_cutoff": 0.0,
            "scope_violation_rate": 0.0,
            "hard_negative_labeled_queries": 5,
            "passed": True,
        }
        variant["gates"] = _policy_gates(
            RETRIEVAL_GATE_RULES,
            variant,
            sample_count=20,
            hard_negative_count=5,
        )
        results[name] = variant
    return {
        "report_contract_version": REPORT_CONTRACT_VERSION,
        "promotion_policy_version": PROMOTION_POLICY_VERSION,
        "promotion_policy_sha256": PROMOTION_POLICY_SHA256,
        "promotion_eligible": True,
        "evaluation_config": RETRIEVAL_EVALUATION_CONFIG,
        "query_count": 20,
        "document_count": 100,
        "variants": ["lexical", "hybrid"],
        "hybrid_missing": "fail",
        "results": results,
        "status": "passed",
        "passed": True,
    }


def test_promotion_policy_accepts_only_exact_recomputed_gate_sets() -> None:
    valid = _agent_promotion_report()
    validate_agent_report(valid, expected_question_count=20)

    missing = copy.deepcopy(valid)
    assert isinstance(missing["gates"], dict)
    del missing["gates"]["tool_recall"]
    with pytest.raises(PromotionPolicyError, match="names mismatch"):
        validate_agent_report(missing, expected_question_count=20)

    invented = copy.deepcopy(valid)
    assert isinstance(invented["gates"], dict)
    invented["gates"]["invented_accuracy"] = invented["gates"]["intent_accuracy"]
    with pytest.raises(PromotionPolicyError, match="unknown=.*invented_accuracy"):
        validate_agent_report(invented, expected_question_count=20)

    wrong_operator = copy.deepcopy(valid)
    assert isinstance(wrong_operator["gates"], dict)
    gate = wrong_operator["gates"]["intent_accuracy"]
    assert isinstance(gate, dict)
    gate["operator"] = "<="
    with pytest.raises(PromotionPolicyError, match="wrong operator"):
        validate_agent_report(wrong_operator, expected_question_count=20)

    weakened = copy.deepcopy(valid)
    assert isinstance(weakened["gates"], dict)
    gate = weakened["gates"]["intent_accuracy"]
    assert isinstance(gate, dict)
    gate["threshold"] = 0.9
    with pytest.raises(PromotionPolicyError, match="weakens"):
        validate_agent_report(weakened, expected_question_count=20)

    false_pass = copy.deepcopy(valid)
    false_pass["intent_accuracy"] = 0.9
    assert isinstance(false_pass["gates"], dict)
    gate = false_pass["gates"]["intent_accuracy"]
    assert isinstance(gate, dict)
    gate["actual"] = 0.9
    with pytest.raises(PromotionPolicyError, match="did not pass recomputation"):
        validate_agent_report(false_pass, expected_question_count=20)


def test_promotion_policy_rejects_invalid_metrics_and_sample_counts() -> None:
    valid = _agent_promotion_report()

    tampered_actual = copy.deepcopy(valid)
    assert isinstance(tampered_actual["gates"], dict)
    gate = tampered_actual["gates"]["scope_pollution_rate"]
    assert isinstance(gate, dict)
    gate["actual"] = 0.5
    with pytest.raises(PromotionPolicyError, match="differs from report metric"):
        validate_agent_report(tampered_actual, expected_question_count=20)

    nonfinite = copy.deepcopy(valid)
    assert isinstance(nonfinite["gates"], dict)
    gate = nonfinite["gates"]["intent_accuracy"]
    assert isinstance(gate, dict)
    gate["actual"] = float("nan")
    with pytest.raises(PromotionPolicyError, match="finite number"):
        validate_agent_report(nonfinite, expected_question_count=20)

    wrong_count = copy.deepcopy(valid)
    assert isinstance(wrong_count["gates"], dict)
    gate = wrong_count["gates"]["tool_precision"]
    assert isinstance(gate, dict)
    gate["sample_count"] = 19
    with pytest.raises(PromotionPolicyError, match="sample_count is not frozen"):
        validate_agent_report(wrong_count, expected_question_count=20)


def test_retrieval_promotion_requires_measured_lexical_and_hybrid_evidence() -> None:
    valid = _retrieval_promotion_report()
    validate_retrieval_report(
        valid,
        expected_query_count=20,
        expected_document_count=100,
    )

    missing_hybrid = copy.deepcopy(valid)
    assert isinstance(missing_hybrid["results"], dict)
    del missing_hybrid["results"]["hybrid"]
    with pytest.raises(PromotionPolicyError, match="results and variants differ"):
        validate_retrieval_report(
            missing_hybrid,
            expected_query_count=20,
            expected_document_count=100,
        )

    skipped_hybrid = copy.deepcopy(valid)
    skipped_hybrid["hybrid_missing"] = "skip"
    with pytest.raises(PromotionPolicyError, match="cannot skip"):
        validate_retrieval_report(
            skipped_hybrid,
            expected_query_count=20,
            expected_document_count=100,
        )

    no_hard_negatives = copy.deepcopy(valid)
    assert isinstance(no_hard_negatives["results"], dict)
    hybrid = no_hard_negatives["results"]["hybrid"]
    assert isinstance(hybrid, dict)
    hybrid["hard_negative_labeled_queries"] = 0
    with pytest.raises(PromotionPolicyError, match="positive integer"):
        validate_retrieval_report(
            no_hard_negatives,
            expected_query_count=20,
            expected_document_count=100,
        )

    weakened = copy.deepcopy(valid)
    assert isinstance(weakened["results"], dict)
    hybrid = weakened["results"]["hybrid"]
    assert isinstance(hybrid, dict)
    assert isinstance(hybrid["gates"], dict)
    gate = hybrid["gates"]["recall_at_10"]
    assert isinstance(gate, dict)
    gate["threshold"] = 0.7
    with pytest.raises(PromotionPolicyError, match="weakens"):
        validate_retrieval_report(
            weakened,
            expected_query_count=20,
            expected_document_count=100,
        )
