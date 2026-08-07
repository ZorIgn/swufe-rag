"""Typed, read-only academic and policy tools backed by the canonical repository."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Literal

from academic.database import AcademicRepository, CourseRecord
from evidence.models import (
    CoverageComponent,
    CoverageReport,
    DerivedFact,
    Evidence,
    EvidencePacket,
    EvidenceTrust,
    Fact,
    Provenance,
)
from evidence.provenance import stable_id
from evidence.registry import EvidenceRegistry
from query.schemas import (
    AuditCompletedCoursesOperation,
    CompareProgramsOperation,
    GetCourseDetailOperation,
    GetGraduationRequirementsOperation,
    GetModuleRequirementsOperation,
    ListCoursesOperation,
    ResolveSourceOperation,
    RetrievePolicyOperation,
)
from retrieval.hybrid import HybridPolicyRetriever
from retrieval.models import PolicyRetrievalRequest, PolicyRetriever

ReviewStatus = Literal["verified", "review_required", "unverified"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        raise TypeError("boolean is not a numeric database value")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"invalid numeric database value: {value!r}")


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("boolean is not an integer database value")
    return int(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _review_status(value: object) -> ReviewStatus:
    status = str(value or "unverified")
    if status not in {"verified", "review_required", "unverified"}:
        raise ValueError(f"invalid review status: {status!r}")
    return status  # type: ignore[return-value]


def _provenance(record_id: str, stored: Mapping[str, object]) -> Provenance:
    extracted = str(stored.get("extracted_at") or _now().isoformat())
    try:
        extracted_at = datetime.fromisoformat(extracted)
    except ValueError:
        extracted_at = _now()
    return Provenance(
        record_id=record_id,
        source_id=str(stored["source_id"]),
        chunk_id=str(stored.get("chunk_id") or "") or None,
        physical_page=_optional_int(stored.get("physical_page")),
        parser_version=str(stored.get("parser_version") or "unknown"),
        source_sha256=_optional_str(stored.get("source_sha256")),
        extracted_at=extracted_at,
        effective_from=_optional_str(stored.get("effective_from")),
        effective_to=_optional_str(stored.get("effective_to")),
        confidence=float(stored.get("confidence") or 0.0),
        review_status=EvidenceTrust(_review_status(stored.get("review_status"))),
    )


def _evidence_from_stored(
    stored: Mapping[str, object], *, record_id: str, registry: EvidenceRegistry
) -> str:
    chunk_id = str(stored.get("chunk_id") or "") or None
    evidence_id = stable_id("ev", stored["source_id"], chunk_id or record_id)
    registry.add(
        Evidence(
            evidence_id=evidence_id,
            source_id=str(stored["source_id"]),
            chunk_id=chunk_id,
            title=str(stored.get("title") or ""),
            article=str(stored.get("article") or "") or None,
            quote=str(stored.get("text") or ""),
            page_url=str(stored.get("page_url") or "") or None,
            file_url=str(stored.get("file_url") or "") or None,
            provenance=_provenance(record_id, stored),
        )
    )
    return evidence_id


def _component(
    operation_id: str,
    tool_name: str,
    kind: Literal["course_set", "requirement", "policy", "comparison", "audit", "source"],
    *,
    complete: bool,
    expected_count: int | None = None,
    returned_count: int | None = None,
    truncated: bool = False,
    authoritative: bool | None = None,
    scope_matched: bool | None = None,
    version_resolved: bool | None = None,
    conflict_free: bool | None = None,
    trusted_evidence: bool | None = None,
    reasons: tuple[str, ...] = (),
) -> CoverageReport:
    return CoverageReport(
        components=(
            CoverageComponent(
                operation_id=operation_id,
                tool_name=tool_name,
                kind=kind,
                complete=complete,
                expected_count=expected_count,
                returned_count=returned_count,
                truncated=truncated,
                authoritative=authoritative,
                scope_matched=scope_matched,
                version_resolved=version_resolved,
                conflict_free=conflict_free,
                trusted_evidence=trusted_evidence,
                reasons=reasons,
            ),
        )
    )


class AcademicTools:
    """Every public method accepts one typed operation and returns one packet."""

    def __init__(
        self, repository: AcademicRepository, policy_retriever: PolicyRetriever | None = None
    ) -> None:
        self.repository = repository
        self.policy_retriever = policy_retriever or HybridPolicyRetriever(
            repository.retrieval_documents(),
            mode="lexical",
            dataset_version=repository.metadata().get("dataset_version", "unknown"),
            index_version=repository.metadata().get("dataset_version", "unknown"),
        )

    def _evidence_for_record(self, record: CourseRecord, registry: EvidenceRegistry) -> str | None:
        if not record.chunk_id:
            return None
        stored = self.repository.source(record.chunk_id)
        return (
            None
            if stored is None
            else _evidence_from_stored(stored, record_id=record.record_id, registry=registry)
        )

    @staticmethod
    def _course_facts(record: CourseRecord, evidence_id: str | None) -> list[Fact]:
        prefix = stable_id("fact", record.record_id)
        evidence_ids = (evidence_id,) if evidence_id else ()
        facts = [
            Fact(
                fact_id=f"{prefix}:name",
                type="course",
                subject=record.record_id,
                predicate="name",
                value=record.name,
                source_record_ids=(record.record_id,),
                evidence_ids=evidence_ids,
            ),
            Fact(
                fact_id=f"{prefix}:code",
                type="course",
                subject=record.record_id,
                predicate="code",
                value=record.code or "未标注",
                source_record_ids=(record.record_id,),
                evidence_ids=evidence_ids,
            ),
            Fact(
                fact_id=f"{prefix}:semester",
                type="course",
                subject=record.record_id,
                predicate="semester",
                value=record.semester,
                unit="semester",
                source_record_ids=(record.record_id,),
                evidence_ids=evidence_ids,
            ),
            Fact(
                fact_id=f"{prefix}:nature",
                type="course",
                subject=record.record_id,
                predicate="nature",
                value=record.nature or "未标注",
                source_record_ids=(record.record_id,),
                evidence_ids=evidence_ids,
            ),
            Fact(
                fact_id=f"{prefix}:module",
                type="course",
                subject=record.record_id,
                predicate="module",
                value=record.module_name,
                source_record_ids=(record.record_id,),
                evidence_ids=evidence_ids,
            ),
        ]
        if record.credits is not None:
            facts.append(
                Fact(
                    fact_id=f"{prefix}:credits",
                    type="course",
                    subject=record.record_id,
                    predicate="credits",
                    value=record.credits,
                    unit="credits",
                    source_record_ids=(record.record_id,),
                    evidence_ids=evidence_ids,
                )
            )
        return facts

    def _courses_packet(
        self,
        records: Iterable[CourseRecord],
        *,
        program_id: str,
        filters: tuple[str, ...],
        operation_id: str = "",
        tool_name: str = "academic.list_courses",
        kind: Literal["course_set", "audit"] = "course_set",
    ) -> EvidencePacket:
        values = tuple(records)
        registry = EvidenceRegistry()
        facts: list[Fact] = []
        trusted = True
        for record in values:
            evidence_id = self._evidence_for_record(record, registry)
            facts.extend(self._course_facts(record, evidence_id))
            if evidence_id:
                evidence = next(
                    item for item in registry.values() if item.evidence_id == evidence_id
                )
                trusted = trusted and evidence.provenance.review_status is EvidenceTrust.VERIFIED
            else:
                trusted = False
        component = _component(
            operation_id or stable_id("op", program_id, tool_name, *filters),
            tool_name,
            kind,
            complete=True,
            expected_count=len(values),
            returned_count=len(values),
            scope_matched=True,
            version_resolved=True,
            conflict_free=True,
            trusted_evidence=trusted,
            reasons=("empty_result",) if not values else (),
        )
        return EvidencePacket(
            packet_id=stable_id(
                "packet", operation_id or program_id, *[value.record_id for value in values]
            ),
            facts=tuple(facts),
            evidence=registry.values(),
            coverage=component,
        )

    def list_courses(self, operation: ListCoursesOperation) -> EvidencePacket:
        args = operation.args
        records = self.repository.list_courses(
            cohort=args.cohort,
            program_id=args.program_id,
            semesters=args.semesters,
            natures=args.course_natures,
            module_ids=args.module_ids,
            course_ids=args.course_ids,
        )
        filters = tuple(
            name
            for name, value in (
                ("semesters", args.semesters),
                ("course_natures", args.course_natures),
                ("module_ids", args.module_ids),
                ("course_ids", args.course_ids),
            )
            if value
        )
        return self._courses_packet(
            records,
            program_id=args.program_id,
            filters=filters,
            operation_id=operation.operation_id,
        )

    def get_course_detail(self, operation: GetCourseDetailOperation) -> EvidencePacket:
        args = operation.args
        records = self.repository.list_courses(
            cohort=args.cohort,
            program_id=args.program_id,
            course_ids=(args.course_id,) if args.course_id else (),
        )
        if args.course_code:
            records = tuple(
                record
                for record in records
                if (record.code or "").upper() == args.course_code.upper()
            )
        return self._courses_packet(
            records,
            program_id=args.program_id or "all-programs",
            filters=("course_detail",),
            operation_id=operation.operation_id,
            tool_name="academic.get_course",
        )

    def _requirements_packet(
        self,
        cohort: int,
        program_id: str,
        module_ids: tuple[str, ...],
        *,
        operation_id: str,
        tool_name: str,
    ) -> EvidencePacket:
        rows = self.repository.requirements(
            cohort=cohort, program_id=program_id, module_ids=module_ids
        )
        registry = EvidenceRegistry()
        facts: list[Fact] = []
        trusted = True
        for row in rows:
            evidence_ids: tuple[str, ...] = ()
            chunk_id = row.get("chunk_id")
            if chunk_id:
                stored = self.repository.source(str(chunk_id))
                if stored:
                    evidence_id = _evidence_from_stored(
                        stored, record_id=str(row["record_id"]), registry=registry
                    )
                    evidence_ids = (evidence_id,)
                    trusted = (
                        trusted
                        and registry.values()[-1].provenance.review_status is EvidenceTrust.VERIFIED
                    )
                else:
                    trusted = False
            else:
                trusted = False
            raw_value = row.get("required_credits")
            value: float | str = _as_float(raw_value) if raw_value is not None else "未结构化"
            facts.append(
                Fact(
                    fact_id=stable_id("fact", row["record_id"], "required_credits"),
                    type="requirement",
                    subject=str(row["module_name"]),
                    predicate="required_credits",
                    value=value,
                    unit="credits",
                    source_record_ids=(str(row["record_id"]),),
                    evidence_ids=evidence_ids,
                )
            )
        coverage = _component(
            operation_id,
            tool_name,
            "requirement",
            complete=bool(rows),
            expected_count=len(rows),
            returned_count=len(rows),
            authoritative=True,
            scope_matched=True,
            version_resolved=True,
            conflict_free=True,
            trusted_evidence=trusted and bool(rows),
            reasons=("no_structured_requirements",) if not rows else (),
        )
        return EvidencePacket(
            packet_id=stable_id("packet", operation_id),
            facts=tuple(facts),
            evidence=registry.values(),
            coverage=coverage,
        )

    def get_graduation_requirements(
        self, operation: GetGraduationRequirementsOperation
    ) -> EvidencePacket:
        return self._requirements_packet(
            operation.args.cohort,
            operation.args.program_id,
            (),
            operation_id=operation.operation_id,
            tool_name="academic.get_requirements",
        )

    def get_module_requirements(self, operation: GetModuleRequirementsOperation) -> EvidencePacket:
        return self._requirements_packet(
            operation.args.cohort,
            operation.args.program_id,
            operation.args.module_ids,
            operation_id=operation.operation_id,
            tool_name="academic.get_module_requirements",
        )

    def audit_completed_courses(
        self, operation: AuditCompletedCoursesOperation, *, context: object | None = None
    ) -> EvidencePacket:
        args = operation.args
        all_courses = self.repository.list_courses(cohort=args.cohort, program_id=args.program_id)
        completed = [row for row in all_courses if row.course_id in args.completed_course_ids]
        packet = self._courses_packet(
            completed,
            program_id=args.program_id,
            filters=("completed_courses",),
            operation_id=operation.operation_id,
            tool_name="academic.audit_progress",
            kind="audit",
        )
        requirements = self.repository.requirements(cohort=args.cohort, program_id=args.program_id)
        facts = list(packet.facts)
        evidence = {item.evidence_id: item for item in packet.evidence}
        for requirement in requirements:
            required_value = requirement.get("required_credits")
            if required_value is None:
                continue
            required = _as_float(required_value)
            requirement_fact_id = stable_id("fact", requirement["record_id"], "required")
            requirement_evidence_ids: tuple[str, ...] = ()
            chunk_id = requirement.get("chunk_id")
            if chunk_id:
                stored = self.repository.source(str(chunk_id))
                if stored:
                    evidence_id = _evidence_from_stored(
                        stored, record_id=str(requirement["record_id"]), registry=EvidenceRegistry()
                    )
                    # Re-register via a local deterministic map; EvidenceRegistry
                    # returns the same stable id for duplicate chunks.
                    stored_evidence = next(
                        (item for item in packet.evidence if item.evidence_id == evidence_id), None
                    )
                    if stored_evidence is None:
                        stored_evidence = Evidence(
                            evidence_id=evidence_id,
                            source_id=str(stored["source_id"]),
                            chunk_id=str(stored.get("chunk_id") or "") or None,
                            title=str(stored.get("title") or ""),
                            article=str(stored.get("article") or "") or None,
                            quote=str(stored.get("text") or ""),
                            page_url=str(stored.get("page_url") or "") or None,
                            file_url=str(stored.get("file_url") or "") or None,
                            provenance=_provenance(str(requirement["record_id"]), stored),
                        )
                        evidence[evidence_id] = stored_evidence
                    requirement_evidence_ids = (evidence_id,)
            requirement_fact = Fact(
                fact_id=requirement_fact_id,
                type="requirement",
                subject=str(requirement["module_name"]),
                predicate="required_credits",
                value=required,
                unit="credits",
                source_record_ids=(str(requirement["record_id"]),),
                evidence_ids=requirement_evidence_ids,
            )
            matching = [
                fact
                for fact in facts
                if fact.predicate == "credits"
                and any(
                    record.record_id == fact.subject
                    and record.module_id == requirement["module_id"]
                    for record in completed
                )
            ]
            total = sum(
                float(fact.value) for fact in matching if isinstance(fact.value, (int, float))
            )
            completed_fact = DerivedFact(
                fact_id=stable_id("fact", requirement["record_id"], "completed"),
                type="progress",
                subject=str(requirement["module_name"]),
                predicate="completed_credits",
                value=total,
                unit="credits",
                source_record_ids=(str(requirement["record_id"]),),
                evidence_ids=tuple(
                    dict.fromkeys(
                        evidence_id for fact in matching for evidence_id in fact.evidence_ids
                    )
                ),
                operator="sum",
                input_fact_ids=tuple(fact.fact_id for fact in matching),
            )
            remaining = DerivedFact(
                fact_id=stable_id("fact", requirement["record_id"], "remaining"),
                type="progress",
                subject=str(requirement["module_name"]),
                predicate="remaining_credits",
                value=max(required - total, 0),
                unit="credits",
                source_record_ids=(str(requirement["record_id"]),),
                evidence_ids=tuple(
                    dict.fromkeys((*requirement_evidence_ids, *completed_fact.evidence_ids))
                ),
                operator="difference",
                input_fact_ids=(requirement_fact.fact_id, completed_fact.fact_id),
            )
            facts.extend((requirement_fact, completed_fact, remaining))
        return packet.model_copy(
            update={"facts": tuple(facts), "evidence": tuple(evidence.values())}
        )

    def compare_programs(self, operation: CompareProgramsOperation) -> EvidencePacket:
        dimensions = operation.args.dimensions
        result = self.repository.compare_programs(
            cohort=operation.args.cohort,
            program_ids=operation.args.program_ids,
            dimensions=dimensions,
        )
        registry = EvidenceRegistry()
        facts: list[Fact] = []
        programs = {str(key): str(value) for key, value in dict(result.get("programs", {})).items()}
        if "module_requirements" in dimensions or "graduation_min_credits" in dimensions:
            for program_id in operation.args.program_ids:
                for row in self.repository.requirements(
                    cohort=operation.args.cohort, program_id=program_id
                ):
                    value = row.get("required_credits")
                    if value is None:
                        continue
                    evidence_id = None
                    if row.get("chunk_id"):
                        stored = self.repository.source(str(row["chunk_id"]))
                        if stored:
                            evidence_id = _evidence_from_stored(
                                stored, record_id=str(row["record_id"]), registry=registry
                            )
                    facts.append(
                        Fact(
                            fact_id=stable_id("fact", row["record_id"], "comparison"),
                            type="comparison",
                            subject=programs.get(program_id, program_id),
                            predicate="module_required_credits",
                            value=_as_float(value),
                            unit="credits",
                            source_record_ids=(str(row["record_id"]),),
                            evidence_ids=(evidence_id,) if evidence_id else (),
                        )
                    )
        if "graduation_min_credits" in dimensions:
            raw_totals = result.get("graduation_min_credits", {})
            canonical_totals = (
                {str(key): _as_float(value) for key, value in raw_totals.items()}
                if isinstance(raw_totals, Mapping)
                else {}
            )
            for program_id in operation.args.program_ids:
                subject = programs.get(program_id, program_id)
                inputs = tuple(fact for fact in facts if fact.subject == subject)
                if program_id in canonical_totals:
                    facts.append(
                        Fact(
                            fact_id=stable_id(
                                "fact", operation.operation_id, program_id, "graduation"
                            ),
                            type="comparison",
                            subject=subject,
                            predicate="graduation_min_credits",
                            value=canonical_totals[program_id],
                            unit="credits",
                            evidence_ids=tuple(
                                dict.fromkeys(item for fact in inputs for item in fact.evidence_ids)
                            ),
                            derivation="observed",
                        )
                    )
                elif inputs:
                    facts.append(
                        DerivedFact(
                            fact_id=stable_id("fact", operation.operation_id, program_id, "sum"),
                            type="comparison",
                            subject=subject,
                            predicate="sum_of_structured_module_minimums",
                            value=sum(
                                float(fact.value)
                                for fact in inputs
                                if isinstance(fact.value, (int, float))
                            ),
                            unit="credits",
                            evidence_ids=tuple(
                                dict.fromkeys(item for fact in inputs for item in fact.evidence_ids)
                            ),
                            operator="sum",
                            input_fact_ids=tuple(fact.fact_id for fact in inputs),
                        )
                    )
        if any(
            dimension in dimensions
            for dimension in ("course_sets", "required_courses", "practice_requirements")
        ):
            intersection = list(result.get("intersection", []))
            if "course_sets" in dimensions:
                facts.append(
                    Fact(
                        fact_id=stable_id("fact", operation.operation_id, "shared_courses"),
                        type="comparison",
                        subject="programs",
                        predicate="shared_courses",
                        value=intersection,
                    )
                )
            if "required_courses" in dimensions:
                facts.append(
                    Fact(
                        fact_id=stable_id(
                            "fact", operation.operation_id, "required_course_difference"
                        ),
                        type="comparison",
                        subject="programs",
                        predicate="required_course_difference",
                        value=[str(item) for item in result.get("required_course_difference", [])],
                    )
                )
            if "practice_requirements" in dimensions:
                facts.append(
                    Fact(
                        fact_id=stable_id("fact", operation.operation_id, "practice_courses"),
                        type="comparison",
                        subject="programs",
                        predicate="practice_requirements",
                        value=[str(item) for item in result.get("practice_requirements", [])],
                    )
                )
        trusted = all(
            evidence.provenance.review_status is EvidenceTrust.VERIFIED
            for evidence in registry.values()
        ) and bool(facts)
        return EvidencePacket(
            packet_id=stable_id("packet", operation.operation_id),
            facts=tuple(facts),
            evidence=registry.values(),
            coverage=_component(
                operation.operation_id,
                "academic.compare_programs",
                "comparison",
                complete=bool(facts),
                expected_count=len(operation.args.program_ids),
                returned_count=len({fact.subject for fact in facts}),
                authoritative=True,
                scope_matched=True,
                version_resolved=True,
                conflict_free=True,
                trusted_evidence=trusted,
            ),
        )

    def retrieve_policy(self, operation: RetrievePolicyOperation) -> EvidencePacket:
        args = operation.args
        result = self.policy_retriever.retrieve(
            PolicyRetrievalRequest(
                query=args.question,
                cohort=args.cohort,
                program_ids=args.program_ids,
                college_ids=args.college_ids,
                topics=args.topics,
                as_of=args.as_of,
                top_k=args.top_k,
            )
        )
        registry = EvidenceRegistry()
        facts: list[Fact] = []
        rows: list[dict[str, object]] = []
        for candidate in result.candidates:
            row = dict(candidate.metadata)
            rows.append(row)
            evidence_id = _evidence_from_stored(
                row, record_id=str(candidate.chunk_id), registry=registry
            )
            facts.append(
                Fact(
                    fact_id=stable_id("fact", candidate.chunk_id),
                    type="policy",
                    subject=str(row.get("title") or candidate.chunk_id),
                    predicate="excerpt",
                    value=str(candidate.text),
                    source_record_ids=(str(candidate.chunk_id),),
                    evidence_ids=(evidence_id,),
                    derivation="retrieved",
                )
            )
        conflicts = self.repository.policy_conflicts(rows) if rows else ()
        trusted = bool(rows) and all(str(row.get("review_status")) == "verified" for row in rows)
        coverage = _component(
            operation.operation_id,
            "policy.search",
            "policy",
            complete=bool(facts),
            expected_count=args.top_k,
            returned_count=len(facts),
            authoritative=bool(rows)
            and all(int(row.get("authority_level") or 0) >= 1 for row in rows),
            scope_matched=bool(rows),
            version_resolved=bool(rows) and not conflicts,
            conflict_free=not conflicts,
            trusted_evidence=trusted,
            reasons=tuple(
                dict.fromkeys(
                    (*result.warnings, "policy_support_insufficient")
                    if not facts
                    else result.warnings
                )
            ),
        )
        return EvidencePacket(
            packet_id=stable_id("packet", operation.operation_id),
            facts=tuple(facts),
            evidence=registry.values(),
            coverage=coverage,
            conflicts=tuple(conflicts),
            warnings=tuple(result.warnings),
        )

    def resolve_source(self, operation: ResolveSourceOperation) -> EvidencePacket:
        stored = self.repository.source(operation.args.chunk_id)
        if stored is None:
            return EvidencePacket(
                packet_id=stable_id("packet", operation.operation_id),
                coverage=_component(
                    operation.operation_id,
                    "source.resolve",
                    "source",
                    complete=False,
                    returned_count=0,
                    reasons=("source_not_found",),
                ),
                warnings=("source_not_found",),
            )
        registry = EvidenceRegistry()
        _evidence_from_stored(stored, record_id=operation.args.chunk_id, registry=registry)
        return EvidencePacket(
            packet_id=stable_id("packet", operation.operation_id),
            evidence=registry.values(),
            coverage=_component(
                operation.operation_id,
                "source.resolve",
                "source",
                complete=True,
                expected_count=1,
                returned_count=1,
                scope_matched=True,
                version_resolved=True,
                conflict_free=True,
                trusted_evidence=registry.values()[0].provenance.review_status
                is EvidenceTrust.VERIFIED,
            ),
        )


__all__ = ["AcademicTools"]
