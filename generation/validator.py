"""Programmatic claim/evidence validation; no subset-sum whitelist."""

from __future__ import annotations

import re

from evidence.models import (
    ClaimSpan,
    ClaimValidation,
    DerivedFact,
    EvidencePacket,
    Fact,
    FinalAnswer,
)

NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?")
COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,6}\d{2,4}\b", re.I)
SCHOOL_FACT_TYPES = frozenset({"course", "requirement", "progress", "comparison", "policy"})


def _fact_support(
    packet: EvidencePacket,
    fact_id: str,
    seen: set[str] | None = None,
) -> tuple[set[str], set[str]]:
    """Collect evidence through a derived-fact graph and reject broken edges."""

    path = set() if seen is None else set(seen)
    if fact_id in path:
        return set(), {f"cycle:{fact_id}"}
    path.add(fact_id)
    fact = packet.fact(fact_id)
    if fact is None:
        return set(), {fact_id}

    evidence_ids = set(fact.evidence_ids)
    missing_inputs: set[str] = set()
    if isinstance(fact, DerivedFact):
        if not fact.input_fact_ids:
            missing_inputs.add(fact_id)
        for input_id in fact.input_fact_ids:
            child_evidence, child_missing = _fact_support(packet, input_id, path)
            evidence_ids.update(child_evidence)
            missing_inputs.update(child_missing)
    return evidence_ids, missing_inputs


def _is_school_factual(facts: list[Fact | DerivedFact]) -> bool:
    return any(fact.type in SCHOOL_FACT_TYPES or bool(fact.source_record_ids) for fact in facts)


def _allowed_strings(facts: list[Fact | DerivedFact]) -> tuple[set[str], set[str]]:
    numbers: set[str] = set()
    codes: set[str] = set()
    for fact in facts:
        if isinstance(fact.value, (int, float)):
            numbers.add(f"{float(fact.value):g}")
        elif isinstance(fact.value, str):
            numbers.update(f"{float(value):g}" for value in NUMBER_RE.findall(fact.value))
            codes.update(value.upper() for value in COURSE_CODE_RE.findall(fact.value))
    return numbers, codes


def _entailment_terms(value: object) -> set[str]:
    """Return generic lexical units for a conservative claim/evidence check."""

    text = str(value or "")
    units = re.findall(r"[A-Za-z0-9]{2,}|[\u3400-\u9fff]{2,}", text.lower())
    return set(units)


def _claim_supported_by_text(
    claim_text: str, facts: list[Fact | DerivedFact], packet: EvidencePacket
) -> bool:
    claim_terms = _entailment_terms(claim_text)
    if not claim_terms:
        return False
    supporting_terms: set[str] = set()
    for fact in facts:
        supporting_terms.update(_entailment_terms(fact.subject))
        supporting_terms.update(_entailment_terms(fact.value))
        for evidence_id in _fact_support(packet, fact.fact_id)[0]:
            evidence = packet.evidence_by_id(evidence_id)
            if evidence is not None:
                supporting_terms.update(_entailment_terms(evidence.quote))
    return bool(claim_terms & supporting_terms)


def _citation_supports_claim(claim_text: str, evidence_id: str, packet: EvidencePacket) -> bool:
    evidence = packet.evidence_by_id(evidence_id)
    return evidence is not None and bool(
        _entailment_terms(claim_text) & _entailment_terms(evidence.quote)
    )


class ClaimValidator:
    """Validate claim-level fact and evidence bindings before rendering an answer."""

    @staticmethod
    def _conflict_refusal(answer: FinalAnswer, packet: EvidencePacket) -> FinalAnswer:
        claims = tuple(
            span.model_copy(
                update={
                    "validation": ClaimValidation(
                        claim_id=span.validation.claim_id,
                        passed=False,
                        reasons=tuple(
                            dict.fromkeys((*span.validation.reasons, "source_version_conflict"))
                        ),
                    )
                }
            )
            for span in answer.claims
        )
        details = "\n".join(f"- {str(value)[:300]}" for value in packet.conflicts)
        return answer.model_copy(
            update={
                "claims": claims,
                "citations": packet.evidence,
                "refused": True,
                "clarification": None,
                "answer_md": (
                    "检测到同等权威来源存在冲突，系统不会自动选择其中一个版本。"
                    f"\n请核对以下来源：\n{details}"
                ),
            }
        )

    def validate(self, answer: FinalAnswer, packet: EvidencePacket) -> FinalAnswer:
        if packet.conflicts:
            return self._conflict_refusal(answer, packet)

        claims: list[ClaimSpan] = []
        selected_evidence: dict[str, object] = {}
        for span in answer.claims:
            reasons: list[str] = []
            resolved = [packet.fact(fact_id) for fact_id in span.fact_ids]
            if not span.fact_ids or any(fact is None for fact in resolved):
                reasons.append("unknown_or_missing_fact")
            facts = [fact for fact in resolved if fact is not None]
            valid_evidence: set[str] = set()
            missing_inputs: set[str] = set()
            for fact in facts:
                fact_evidence, fact_missing = _fact_support(packet, fact.fact_id)
                valid_evidence.update(fact_evidence)
                missing_inputs.update(fact_missing)

            provided_evidence = set(span.evidence_ids)
            if not provided_evidence.issubset(valid_evidence):
                reasons.append("citation_not_linked_to_claim_fact")
            if any(packet.evidence_by_id(evidence_id) is None for evidence_id in provided_evidence):
                reasons.append("evidence_record_missing")

            if _is_school_factual(facts):
                if missing_inputs:
                    reasons.append("derived_fact_input_missing")
                if not valid_evidence:
                    reasons.append("school_fact_without_evidence")
                if not provided_evidence:
                    reasons.append("school_fact_missing_evidence_binding")

            numbers, codes = _allowed_strings(facts)
            mentioned_numbers = {f"{float(value):g}" for value in NUMBER_RE.findall(span.text)}
            mentioned_codes = {value.upper() for value in COURSE_CODE_RE.findall(span.text)}
            if not mentioned_numbers.issubset(numbers):
                reasons.append("number_not_bound_to_claim_fact")
            if not mentioned_codes.issubset(codes):
                reasons.append("course_code_not_bound_to_claim_fact")
            if _is_school_factual(facts) and not _claim_supported_by_text(span.text, facts, packet):
                reasons.append("claim_not_entailed_by_fact")
            if provided_evidence and any(
                not _citation_supports_claim(span.text, identifier, packet)
                for identifier in provided_evidence
            ):
                reasons.append("citation_not_supporting_claim")
            validation = ClaimValidation(
                claim_id=span.validation.claim_id,
                passed=not reasons,
                reasons=tuple(reasons),
            )
            claims.append(span.model_copy(update={"validation": validation}))
            for evidence_id in span.evidence_ids:
                evidence = packet.evidence_by_id(evidence_id)
                if evidence:
                    selected_evidence[evidence_id] = evidence

        passed = bool(claims) and all(claim.validation.passed for claim in claims)
        if answer.clarification:
            passed = True
        if not passed:
            return answer.model_copy(
                update={
                    "claims": tuple(claims),
                    "citations": tuple(selected_evidence.values()),
                    "refused": True,
                    "answer_md": "当前证据无法通过事实与引用校验，因此不返回未经验证的学校事实。",
                }
            )
        return answer.model_copy(
            update={
                "claims": tuple(claims),
                "citations": tuple(selected_evidence.values()),
            }
        )


__all__ = ["ClaimValidator"]
