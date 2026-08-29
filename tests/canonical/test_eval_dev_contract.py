from __future__ import annotations

import json
from pathlib import Path

from eval.run_agent_eval import _validate_row_contract
from eval.run_retrieval_ablation import _load_inputs

DEV_DIR = Path("eval/dev")


def test_dev_queries_use_typed_fixture_contract() -> None:
    rows = json.loads((DEV_DIR / "queries.json").read_text(encoding="utf-8"))
    assert isinstance(rows, list) and rows
    for index, row in enumerate(rows, start=1):
        assert row["dataset_kind"] == "test_fixture"
        assert row["dataset_version"] == "fixture-1"
        assert "expected_tools" not in row
        _validate_row_contract(row, index)


def test_dev_retrieval_fixture_uses_graded_scope_and_hard_negative_labels() -> None:
    documents, queries = _load_inputs(
        DEV_DIR / "retrieval_documents.jsonl", DEV_DIR / "retrieval_queries.json"
    )
    assert len(documents) == 3
    assert len(queries) == 3
    assert all(query.scope_label == "global-2024" for query in queries)
    assert all(query.relevance and query.hard_negative_chunk_ids for query in queries)
