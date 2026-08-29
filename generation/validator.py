"""Programmatic claim/evidence validation; no subset-sum whitelist."""

from __future__ import annotations

import re

from evidence.models import (
    ClaimAtom,
    ClaimComparator,
    ClaimSpan,
    ClaimValidation,
    DerivedFact,
    EvidencePacket,
    EvidenceTrust,
    Fact,
    FinalAnswer,
)
from generation.claim_semantics import fact_signature, polarity_conflicts, text_signature

NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?")
COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,6}\d{2,4}\b", re.I)
LATIN_TERM_RE = re.compile(r"[A-Za-z0-9]{2,}")
CHINESE_RUN_RE = re.compile(r"[\u3400-\u9fff]{2,}")
COMPARATOR_MARKERS: dict[ClaimComparator, tuple[str, ...]] = {
    "equals": (),
    "contains": ("包含", "包括", "含有", "涉及"),
    "at_least": ("至少", "不少于", "最低", "不低于"),
    "at_most": ("至多", "不超过", "最高", "不高于"),
    "before": ("之前", "以前", "早于", "不晚于"),
    "after": ("之后", "以后", "晚于", "不早于"),
    "satisfies": (),
}
PREDICATE_LANGUAGE = {
    "name": "课程名称",
    "code": "课程代码",
    "credits": "学分",
    "semester": "学期开设",
    "nature": "课程性质",
    "module": "所属模块",
    "required_credits": "最低要求学分",
    "remaining_credits": "尚差剩余学分",
    "completed_credits": "已修完成学分",
    "graduation_min_credits": "毕业最低学分",
    "sum_of_structured_module_minimums": (
        "结构化模块最低学分合计，不等同于已观测的毕业最低学分"
    ),
    "module_required_credits": "模块最低要求学分",
    "shared_courses": "两个专业共有课程",
    "courses_only_in_program": "该专业独有课程",
    "excerpt": "制度条款原文",
    "feasible": "学业规划可行不可行结论",
}


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


def _requires_evidence(facts: list[Fact | DerivedFact]) -> bool:
    """Return whether a claim makes an external factual assertion.

    Fact roles are opt-out rather than type allowlists.  A new fact type is
    therefore factual by default, and can bypass grounding only when the tool
    author explicitly marks it as process metadata/non-factual.
    """

    return any(fact.role == "factual" for fact in facts)


def _canonical_value(value: object) -> object:
    if isinstance(value, float):
        return float(f"{value:g}")
    if isinstance(value, list):
        return tuple(_canonical_value(item) for item in value)
    return value


def _comparator_mentioned(comparator: ClaimComparator, text: str) -> bool:
    markers = COMPARATOR_MARKERS[comparator]
    return not markers or any(marker in text for marker in markers)


def _atom_bound_facts(
    atom: ClaimAtom, facts_by_id: dict[str, Fact | DerivedFact]
) -> tuple[list[Fact | DerivedFact], list[str]]:
    """Resolve and verify the exact fact binding declared by an atom."""

    reasons: list[str] = []
    if not atom.fact_ids:
        return [], ["claim_atom_missing_fact_binding"]
    bound: list[Fact | DerivedFact] = []
    for fact_id in atom.fact_ids:
        fact = facts_by_id.get(fact_id)
        if fact is None:
            reasons.append("claim_atom_unknown_fact")
            continue
        bound.append(fact)
        if fact.subject != atom.subject:
            reasons.append("claim_atom_subject_mismatch")
        if fact.predicate != atom.predicate:
            reasons.append("claim_atom_predicate_mismatch")
        if fact.comparator != atom.comparator:
            reasons.append("claim_atom_comparator_mismatch")
        if _canonical_value(fact.value) != _canonical_value(atom.value):
            reasons.append("claim_atom_value_mismatch")
        if fact.unit != atom.unit:
            reasons.append("claim_atom_unit_mismatch")
        if fact.conditions != atom.conditions:
            reasons.append("claim_atom_conditions_mismatch")
        if fact.exceptions != atom.exceptions:
            reasons.append("claim_atom_exceptions_mismatch")
        if fact.scope != atom.scope:
            reasons.append("claim_atom_scope_mismatch")
        if fact.temporal != atom.temporal:
            reasons.append("claim_atom_temporal_mismatch")
    return bound, reasons


def _normalized_text(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "")).lower()


def _value_mentioned(text: str, value: object) -> bool:
    if isinstance(value, (int, float)):
        target = f"{float(value):g}"
        return target in {f"{float(item):g}" for item in NUMBER_RE.findall(text)}
    if isinstance(value, list):
        return all(_value_mentioned(text, item) for item in value)
    raw = str(value).strip()
    if not raw:
        return True
    if COURSE_CODE_RE.fullmatch(raw):
        return raw.upper() in {item.upper() for item in COURSE_CODE_RE.findall(text)}
    # A long excerpt may be rendered with an intentional display truncation.
    # For it, semantic coverage is checked by the source/evidence path below.
    return len(raw) > 120 or _normalized_text(raw) in _normalized_text(text)


def _atom_text_reasons(atom: ClaimAtom, text: str) -> list[str]:
    """Ensure rendered wording did not drop an atom's key qualifier."""

    reasons: list[str] = []
    if not _value_mentioned(text, atom.value):
        reasons.append("claim_text_atom_value_mismatch")
    if not _comparator_mentioned(atom.comparator, text):
        reasons.append("claim_text_atom_comparator_missing")
    rendered_unit = {"credits": "学分", "semester": "学期"}.get(atom.unit or "", atom.unit)
    if rendered_unit and _normalized_text(rendered_unit) not in _normalized_text(text):
        reasons.append("claim_text_atom_unit_missing")
    for condition in atom.conditions:
        if _normalized_text(condition) not in _normalized_text(text):
            reasons.append("claim_text_atom_condition_missing")
    for exception in atom.exceptions:
        if _normalized_text(exception) not in _normalized_text(text):
            reasons.append("claim_text_atom_exception_missing")
    if atom.scope and _normalized_text(atom.scope) not in _normalized_text(text):
        reasons.append("claim_text_atom_scope_missing")
    if atom.temporal and _normalized_text(atom.temporal) not in _normalized_text(text):
        reasons.append("claim_text_atom_temporal_missing")
    return reasons


def _evidence_supports_atom(atom: ClaimAtom, evidence_id: str, packet: EvidencePacket) -> bool:
    """Require one cited span to cover an atom's value and all qualifiers.

    This is a deterministic structural gate, not a claim that lexical matching
    is full natural-language entailment.  Catalog-field facts additionally
    rely on their field-verification lineage at build time.
    """

    evidence = packet.evidence_by_id(evidence_id)
    if evidence is None:
        return False
    quote = evidence.quote
    if not _value_mentioned(quote, atom.value):
        return False
    if not _comparator_mentioned(atom.comparator, quote):
        return False
    return all(
        _normalized_text(item) in _normalized_text(quote)
        for item in (*atom.conditions, *atom.exceptions)
    ) and (atom.scope is None or _normalized_text(atom.scope) in _normalized_text(quote)) and (
        atom.temporal is None or _normalized_text(atom.temporal) in _normalized_text(quote)
    )


def _allowed_strings(facts: list[Fact | DerivedFact]) -> tuple[set[str], set[str]]:
    numbers: set[str] = set()
    codes: set[str] = set()
    for fact in facts:
        if isinstance(fact.value, (int, float)):
            numbers.add(f"{float(fact.value):g}")
        else:
            values = fact.value if isinstance(fact.value, list) else [fact.value]
            for raw_value in values:
                if not isinstance(raw_value, str):
                    continue
                numbers.update(f"{float(value):g}" for value in NUMBER_RE.findall(raw_value))
                codes.update(value.upper() for value in COURSE_CODE_RE.findall(raw_value))
    return numbers, codes


def _entailment_terms(value: object) -> set[str]:
    """Return stable lexical units, including CJK bigrams.

    Treating an entire Chinese clause as one token made a single shared noun
    sufficient to pass the old entailment guard.  Character bigrams provide a
    deterministic coverage signal without pretending to be a general NLI model.
    """

    text = str(value or "").lower()
    terms = {item.lower() for item in LATIN_TERM_RE.findall(text)}
    for run in CHINESE_RUN_RE.findall(text):
        if len(run) == 2:
            terms.add(run)
        else:
            terms.update(run[index : index + 2] for index in range(len(run) - 1))
    return terms


def _fact_terms(fact: Fact | DerivedFact) -> set[str]:
    terms = _entailment_terms(fact.subject) | _entailment_terms(fact.value)
    terms.update(_entailment_terms(PREDICATE_LANGUAGE.get(fact.predicate, fact.predicate)))
    return terms


def _facts_share_record(facts: list[Fact | DerivedFact]) -> bool:
    """Require atomic, non-derived facts in one claim to share a record edge."""

    record_sets = [
        set(fact.source_record_ids)
        for fact in facts
        if not isinstance(fact, DerivedFact) and fact.source_record_ids
    ]
    if len(record_sets) <= 1:
        return True
    return bool(set.intersection(*record_sets))


def _claim_supported_by_text(
    claim_text: str, facts: list[Fact | DerivedFact], packet: EvidencePacket
) -> bool:
    claim_terms = _entailment_terms(claim_text)
    if not claim_terms:
        return False
    supporting_terms: set[str] = set()
    for fact in facts:
        supporting_terms.update(_fact_terms(fact))
        for evidence_id in _fact_support(packet, fact.fact_id)[0]:
            evidence = packet.evidence_by_id(evidence_id)
            if evidence is not None:
                supporting_terms.update(_entailment_terms(evidence.quote))
    return len(claim_terms & supporting_terms) / len(claim_terms) >= 0.60


def _citation_supports_claim(
    claim_text: str,
    evidence_id: str,
    packet: EvidencePacket,
    facts: list[Fact | DerivedFact],
) -> bool:
    evidence = packet.evidence_by_id(evidence_id)
    if evidence is None:
        return False
    # A rule-evaluation conclusion (for example, "可行") is deliberately not
    # expected to occur verbatim in a curriculum PDF.  Its atom and complete
    # input-fact graph are checked elsewhere in this validator; requiring a
    # lexical overlap here would turn that valid proof into a false refusal.
    # This exception is narrow: mixed/observed claims still need direct lexical
    # support from each cited evidence span.
    if facts and all(
        isinstance(fact, DerivedFact) and fact.operator == "rule_evaluation" for fact in facts
    ):
        return all(
            evidence_id in _fact_support(packet, fact.fact_id)[0]
            for fact in facts
        )
    return bool(_entailment_terms(claim_text) & _entailment_terms(evidence.quote))


def _untrusted_reachable_evidence(
    packet: EvidencePacket, evidence_ids: set[str]
) -> tuple[set[str], set[str]]:
    """Return missing and non-verified evidence IDs in a fact support graph."""

    missing: set[str] = set()
    non_verified: set[str] = set()
    for evidence_id in evidence_ids:
        evidence = packet.evidence_by_id(evidence_id)
        if evidence is None:
            missing.add(evidence_id)
        elif evidence.provenance.review_status is not EvidenceTrust.VERIFIED:
            non_verified.add(evidence_id)
    return missing, non_verified


def _polarity_conflict_reasons(
    claim_text: str,
    facts: list[Fact | DerivedFact],
    packet: EvidencePacket,
    evidence_ids: set[str],
) -> tuple[str, ...]:
    """Compare directional policy meaning of a claim with its bound support."""

    claim = text_signature(claim_text)
    reasons: list[str] = []
    for fact in facts:
        for conflict in polarity_conflicts(
            claim, fact_signature(fact.predicate, fact.subject, fact.value)
        ):
            reasons.append(f"claim_predicate_polarity_conflict:{conflict}")
    # A computed/tool-result fact can legitimately combine an optional course
    # with a mandatory minimum (for example: elective capacity covers a
    # required credit gap).  Comparing the final claim against every input
    # excerpt independently creates a false required-vs-optional inversion.
    # Observed/retrieved claims still receive the direct source polarity guard.
    if any(fact.derivation in {"observed", "retrieved"} for fact in facts):
        for evidence_id in sorted(evidence_ids):
            evidence = packet.evidence_by_id(evidence_id)
            if evidence is None:
                continue
            for conflict in polarity_conflicts(claim, text_signature(evidence.quote)):
                reasons.append(f"claim_evidence_polarity_conflict:{conflict}")
    return tuple(dict.fromkeys(reasons))


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
            facts_by_id = {fact.fact_id: fact for fact in facts}
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

            factual = _requires_evidence(facts)
            atoms_by_fact: dict[str, list[ClaimAtom]] = {}
            if factual and not span.atoms:
                reasons.append("claim_semantics_missing")
            for atom in span.atoms:
                if not set(atom.fact_ids).issubset(set(span.fact_ids)):
                    reasons.append("claim_atom_not_bound_to_claim_fact")
                bound_facts, atom_reasons = _atom_bound_facts(atom, facts_by_id)
                reasons.extend(atom_reasons)
                for fact in bound_facts:
                    atoms_by_fact.setdefault(fact.fact_id, []).append(atom)
                if factual:
                    if not atom.evidence_ids:
                        reasons.append("claim_atom_missing_evidence_binding")
                    if not set(atom.evidence_ids).issubset(provided_evidence):
                        reasons.append("claim_atom_evidence_not_bound_to_claim")
                    for fact in bound_facts:
                        reachable, _ = _fact_support(packet, fact.fact_id)
                        if not set(atom.evidence_ids).issubset(reachable):
                            reasons.append("claim_atom_evidence_not_reachable")
                    # A rule-evaluation atom describes a conclusion rather
                    # than verbatim source wording (for example "可行"). Its
                    # support is validated through the required input-fact
                    # graph; demanding the conclusion literal in a source PDF
                    # would reject every legitimate computation.  Observed
                    # and retrieved atoms still require one span to cover the
                    # atom itself.
                    if any(not isinstance(fact, DerivedFact) for fact in bound_facts) and not all(
                        _evidence_supports_atom(atom, evidence_id, packet)
                        for evidence_id in atom.evidence_ids
                    ):
                        reasons.append("claim_atom_evidence_span_mismatch")
                    reasons.extend(_atom_text_reasons(atom, span.text))
            if factual and any(
                fact.role == "factual" and fact.fact_id not in atoms_by_fact for fact in facts
            ):
                reasons.append("claim_fact_missing_semantic_atom")

            if factual:
                if missing_inputs:
                    reasons.append("derived_fact_input_missing")
                if not _facts_share_record(facts):
                    reasons.append("claim_facts_cross_record")
                if not valid_evidence:
                    reasons.append("school_fact_without_evidence")
                if not provided_evidence:
                    reasons.append("school_fact_missing_evidence_binding")
                elif any(
                    support and not provided_evidence.intersection(support)
                    for support in (_fact_support(packet, fact.fact_id)[0] for fact in facts)
                ):
                    reasons.append("claim_fact_missing_evidence_binding")
                missing_evidence, non_verified_evidence = _untrusted_reachable_evidence(
                    packet, valid_evidence
                )
                if missing_evidence:
                    reasons.append("school_fact_evidence_record_missing")
                if non_verified_evidence:
                    reasons.append("school_fact_non_verified_evidence")
                reasons.extend(_polarity_conflict_reasons(span.text, facts, packet, valid_evidence))

            numbers, codes = _allowed_strings(facts)
            mentioned_numbers = {f"{float(value):g}" for value in NUMBER_RE.findall(span.text)}
            mentioned_codes = {value.upper() for value in COURSE_CODE_RE.findall(span.text)}
            if not mentioned_numbers.issubset(numbers):
                reasons.append("number_not_bound_to_claim_fact")
            if not mentioned_codes.issubset(codes):
                reasons.append("course_code_not_bound_to_claim_fact")
            if factual and not _claim_supported_by_text(span.text, facts, packet):
                reasons.append("claim_not_entailed_by_fact")
            if provided_evidence and any(
                not _citation_supports_claim(span.text, identifier, packet, facts)
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
