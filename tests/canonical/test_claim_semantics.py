from __future__ import annotations

from datetime import datetime, timezone

from evidence.models import (
    ClaimSpan,
    ClaimValidation,
    Evidence,
    EvidencePacket,
    EvidenceTrust,
    Fact,
    FinalAnswer,
    Provenance,
)
from generation.claim_semantics import (
    BoundaryPolarity,
    PermissionPolarity,
    RequirementPolarity,
    TemporalPolarity,
    polarity_conflicts,
    text_signature,
)
from generation.validator import ClaimValidator


def _answer(text: str) -> FinalAnswer:
    return FinalAnswer(
        answer_md=text,
        claims=(
            ClaimSpan(
                text=text,
                fact_ids=("fact",),
                evidence_ids=("evidence",),
                validation=ClaimValidation(claim_id="claim", passed=False),
            ),
        ),
        citations=(),
    )


def _packet(
    *,
    quote: str,
    review_status: EvidenceTrust = EvidenceTrust.VERIFIED,
    predicate: str = "policy_rule",
) -> EvidencePacket:
    evidence = Evidence(
        evidence_id="evidence",
        source_id="source",
        chunk_id="chunk",
        title="测试规定",
        quote=quote,
        provenance=Provenance(
            record_id="record",
            source_id="source",
            chunk_id="chunk",
            physical_page=1,
            parser_version="test",
            extracted_at=datetime.now(timezone.utc),
            confidence=1.0,
            review_status=review_status,
        ),
    )
    return EvidencePacket(
        packet_id="packet",
        evidence=(evidence,),
        facts=(
            Fact(
                fact_id="fact",
                type="policy",
                subject="跨专业选修",
                predicate=predicate,
                value=quote,
                source_record_ids=("record",),
                evidence_ids=(evidence.evidence_id,),
            ),
        ),
    )


def test_signature_detects_permission_and_temporal_inversions() -> None:
    forbidden = text_signature("学生不得跨专业选修")
    allowed = text_signature("学生可以跨专业选修")
    assert PermissionPolarity.FORBIDDEN in forbidden.permissions
    assert "allowed_vs_forbidden" in polarity_conflicts(allowed, forbidden)

    required = text_signature("必须修读")
    optional = text_signature("可选修读")
    assert RequirementPolarity.REQUIRED in required.requirements
    assert "required_vs_optional" in polarity_conflicts(required, optional)

    before = text_signature("须在第6学期前完成")
    after = text_signature("可在第6学期后完成")
    assert TemporalPolarity.BEFORE in before.temporal
    assert "before_vs_after" in polarity_conflicts(after, before)


def test_validator_rejects_non_verified_evidence_for_school_fact() -> None:
    result = ClaimValidator().validate(
        _answer("课程为 3 学分。"),
        _packet(quote="课程为 3 学分。", review_status=EvidenceTrust.REVIEW_REQUIRED),
    )

    assert result.refused
    assert "school_fact_non_verified_evidence" in result.claims[0].validation.reasons


def test_validator_rejects_permission_polarity_conflict() -> None:
    result = ClaimValidator().validate(
        _answer("学生可以跨专业选修。"),
        _packet(quote="学生不得跨专业选修。"),
    )

    assert result.refused
    assert "claim_evidence_polarity_conflict:allowed_vs_forbidden" in result.claims[0].validation.reasons


def test_validator_rejects_required_credit_maximum_claim() -> None:
    result = ClaimValidator().validate(
        _answer("专业选修最多可修 3 学分。"),
        _packet(
            quote="专业选修最低要求为 3 学分。",
            predicate="required_credits",
        ),
    )

    assert result.refused
    assert BoundaryPolarity.MAXIMUM in text_signature("专业选修最多可修 3 学分。").boundaries
    assert any("minimum_vs_maximum" in reason for reason in result.claims[0].validation.reasons)
