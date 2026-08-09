from __future__ import annotations

from retrieval.hybrid import HybridPolicyRetriever
from retrieval.lexical import BM25LexicalIndex
from retrieval.models import PolicyRetrievalRequest


def _document() -> dict[str, object]:
    return {
        "chunk_id": "english-policy",
        "source_id": "source",
        "title": "大学英语免修规定",
        "article": "免修条件",
        "text": "大学英语达到规定考试成绩可以申请免修。",
        "status": "现行",
        "cohort": "不限",
        "college_id": "全校",
        "program_ids": (),
        "review_status": "verified",
        "authority_level": 3,
    }


def test_bm25_drops_zero_relevance_documents() -> None:
    index = BM25LexicalIndex([_document()])

    assert index.rank("火星移民住房补贴", ("english-policy",), limit=5) == ()
    assert index.rank("火星殖民规定", ("english-policy",), limit=5) == ()


def test_policy_retriever_does_not_turn_scope_priors_into_relevance() -> None:
    retriever = HybridPolicyRetriever([_document()], mode="lexical")

    result = retriever.retrieve(
        PolicyRetrievalRequest(query="火星移民住房补贴", top_k=5)
    )

    assert result.candidates == ()
    assert result.candidate_count == 0
    assert "policy_no_relevant_candidates" in result.warnings

    generic_overlap = retriever.retrieve(
        PolicyRetrievalRequest(query="火星殖民规定", top_k=5)
    )
    assert generic_overlap.candidates == ()


def test_policy_retriever_keeps_positive_lexical_evidence() -> None:
    retriever = HybridPolicyRetriever([_document()], mode="lexical")

    result = retriever.retrieve(PolicyRetrievalRequest(query="大学英语免修条件", top_k=5))

    assert len(result.candidates) == 1
    assert result.candidates[0].bm25_score is not None
    assert result.candidates[0].bm25_score != 0
    assert result.candidates[0].final_score > 0


def test_global_policy_survives_a_college_scoped_request() -> None:
    retriever = HybridPolicyRetriever([_document()], mode="lexical")

    result = retriever.retrieve(
        PolicyRetrievalRequest(
            query="大学英语免修条件", college_ids=("计算机与人工智能学院",), top_k=5
        )
    )

    assert [item.chunk_id for item in result.candidates] == ["english-policy"]
