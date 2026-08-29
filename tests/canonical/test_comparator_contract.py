from __future__ import annotations

from datetime import datetime, timezone

import pytest

from evidence.models import (
    ClaimAtom,
    ClaimComparator,
    ClaimSpan,
    ClaimValidation,
    Evidence,
    EvidencePacket,
    EvidenceTrust,
    Fact,
    FinalAnswer,
    Provenance,
)
from generation.validator import ClaimValidator


def _packet(
    *,
    comparator: ClaimComparator = "at_least",
    quote: str = "专业选修课最低要求为3学分。",
) -> EvidencePacket:
    evidence = Evidence(
        evidence_id="evidence",
        source_id="source",
        chunk_id="chunk",
        title="测试培养方案",
        quote=quote,
        provenance=Provenance(
            record_id="record",
            source_id="source",
            chunk_id="chunk",
            physical_page=1,
            parser_version="test",
            extracted_at=datetime.now(timezone.utc),
            confidence=1.0,
            review_status=EvidenceTrust.VERIFIED,
        ),
    )
    return EvidencePacket(
        packet_id="packet",
        evidence=(evidence,),
        facts=(
            Fact(
                fact_id="fact",
                type="requirement",
                subject="专业选修课",
                predicate="required_credits",
                value=3,
                comparator=comparator,
                unit="credits",
                source_record_ids=("record",),
                evidence_ids=("evidence",),
            ),
        ),
    )


def _answer(*, comparator: ClaimComparator, text: str) -> FinalAnswer:
    return FinalAnswer(
        answer_md=text,
        claims=(
            ClaimSpan(
                text=text,
                fact_ids=("fact",),
                evidence_ids=("evidence",),
                atoms=(
                    ClaimAtom(
                        subject="专业选修课",
                        predicate="required_credits",
                        comparator=comparator,
                        value=3,
                        unit="credits",
                        fact_ids=("fact",),
                        evidence_ids=("evidence",),
                    ),
                ),
                validation=ClaimValidation(claim_id="claim", passed=False),
            ),
        ),
        citations=(),
    )


def test_matching_minimum_comparator_passes() -> None:
    result = ClaimValidator().validate(
        _answer(comparator="at_least", text="专业选修课最低要求为3学分。"),
        _packet(),
    )

    assert not result.refused
    assert result.claims[0].validation.passed


@pytest.mark.parametrize("comparator", ["equals", "at_most"])
def test_same_value_with_changed_direction_is_rejected(
    comparator: ClaimComparator,
) -> None:
    text = (
        "专业选修课最高要求为3学分。"
        if comparator == "at_most"
        else "专业选修课要求为3学分。"
    )
    result = ClaimValidator().validate(
        _answer(comparator=comparator, text=text),
        _packet(),
    )

    assert result.refused
    assert "claim_atom_comparator_mismatch" in result.claims[0].validation.reasons


def test_minimum_comparator_must_be_visible_in_rendered_text() -> None:
    result = ClaimValidator().validate(
        _answer(comparator="at_least", text="专业选修课要求为3学分。"),
        _packet(),
    )

    assert result.refused
    assert "claim_text_atom_comparator_missing" in result.claims[0].validation.reasons


def test_minimum_comparator_must_be_supported_by_evidence_span() -> None:
    result = ClaimValidator().validate(
        _answer(comparator="at_least", text="专业选修课最低要求为3学分。"),
        _packet(quote="专业选修课要求为3学分。"),
    )

    assert result.refused
    assert "claim_atom_evidence_span_mismatch" in result.claims[0].validation.reasons


def test_temporal_direction_cannot_be_reversed() -> None:
    packet = _packet(comparator="before", quote="该课程须在第6学期之前完成。")
    packet = packet.model_copy(
        update={
            "facts": (
                packet.facts[0].model_copy(
                    update={
                        "predicate": "completion_deadline",
                        "value": "第6学期",
                        "comparator": "before",
                        "unit": None,
                    }
                ),
            )
        }
    )
    answer = FinalAnswer(
        answer_md="该课程须在第6学期之后完成。",
        claims=(
            ClaimSpan(
                text="该课程须在第6学期之后完成。",
                fact_ids=("fact",),
                evidence_ids=("evidence",),
                atoms=(
                    ClaimAtom(
                        subject="专业选修课",
                        predicate="completion_deadline",
                        comparator="after",
                        value="第6学期",
                        fact_ids=("fact",),
                        evidence_ids=("evidence",),
                    ),
                ),
                validation=ClaimValidation(claim_id="claim", passed=False),
            ),
        ),
        citations=(),
    )

    result = ClaimValidator().validate(answer, packet)

    assert result.refused
    assert "claim_atom_comparator_mismatch" in result.claims[0].validation.reasons


def test_numeric_comparator_rejects_non_numeric_fact_value() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        Fact(
            fact_id="invalid",
            type="requirement",
            subject="模块",
            predicate="required_credits",
            value="三",
            comparator="at_least",
        )
