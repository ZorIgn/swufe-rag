from __future__ import annotations

from retrieval.hybrid import HybridPolicyRetriever
from retrieval.models import PolicyRetrievalRequest
from retrieval.scope import policy_scope_matches


def _document(
    chunk_id: str,
    *,
    cohort: str = "不限",
    college_id: str = "全校",
    program_ids: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "source_id": f"source-{chunk_id}",
        "title": "转专业规定",
        "article": "适用范围",
        "text": "学生申请转专业须按规定提交材料。",
        "status": "现行",
        "cohort": cohort,
        "college_id": college_id,
        "program_ids": program_ids,
        "review_status": "verified",
        "authority_level": 3,
    }


def test_missing_scope_dimensions_only_match_universal_documents() -> None:
    assert policy_scope_matches(
        _document("universal"),
        cohort=None,
        program_ids=(),
        college_ids=(),
    )
    assert not policy_scope_matches(
        _document("cohort", cohort="2024"),
        cohort=None,
        program_ids=(),
        college_ids=(),
    )
    assert not policy_scope_matches(
        _document("college", college_id="学院甲"),
        cohort=None,
        program_ids=(),
        college_ids=(),
    )
    assert not policy_scope_matches(
        _document("program", program_ids=("program-a",)),
        cohort=None,
        program_ids=(),
        college_ids=(),
    )


def test_exact_scope_matches_but_partial_or_cross_program_scope_does_not() -> None:
    scoped = _document(
        "scoped",
        cohort="2024",
        college_id="学院甲",
        program_ids=("program-a",),
    )

    assert policy_scope_matches(
        scoped,
        cohort=2024,
        program_ids=("program-a",),
        college_ids=("学院甲",),
    )
    assert not policy_scope_matches(
        scoped,
        cohort=None,
        program_ids=("program-a",),
        college_ids=("学院甲",),
    )
    assert not policy_scope_matches(
        scoped,
        cohort=2024,
        program_ids=(),
        college_ids=("学院甲",),
    )
    assert not policy_scope_matches(
        scoped,
        cohort=2024,
        program_ids=("program-a", "program-b"),
        college_ids=("学院甲",),
    )


def test_hybrid_retriever_applies_the_same_fail_closed_scope_contract() -> None:
    documents = (
        _document("universal"),
        _document("cohort", cohort="2024"),
        _document("college", college_id="学院甲"),
        _document(
            "program",
            cohort="2024",
            college_id="学院甲",
            program_ids=("program-a",),
        ),
    )
    retriever = HybridPolicyRetriever(documents, mode="lexical")

    unscoped = retriever.retrieve(
        PolicyRetrievalRequest(query="转专业提交材料", top_k=10)
    )
    exact = retriever.retrieve(
        PolicyRetrievalRequest(
            query="转专业提交材料",
            cohort=2024,
            college_ids=("学院甲",),
            program_ids=("program-a",),
            top_k=10,
        )
    )

    assert {item.chunk_id for item in unscoped.candidates} == {"universal"}
    assert {item.chunk_id for item in exact.candidates} == {
        "universal",
        "cohort",
        "college",
        "program",
    }
