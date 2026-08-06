"""Read-only typed academic and policy tools backed by :mod:`academic.database`."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Literal

from academic.database import AcademicRepository, CourseRecord
from evidence.models import (
    CourseSetCoverage,
    Coverage,
    DerivedFact,
    Evidence,
    EvidencePacket,
    Fact,
    FieldCoverage,
    PolicyCoverage,
    ProgramCoverage,
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

ReviewStatus = Literal["verified", "review_required", "unverified"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        raise TypeError("boolean is not a numeric database value")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise TypeError(f"invalid numeric database value: {value!r}") from exc
    raise TypeError(f"invalid numeric database value: {value!r}")


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("boolean is not an integer database value")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise TypeError(f"invalid integer database value: {value!r}") from exc
    raise TypeError(f"invalid integer database value: {value!r}")


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise TypeError(f"invalid string database value: {value!r}")


def _review_status(value: object) -> ReviewStatus:
    if value == "verified":
        return "verified"
    if value == "review_required":
        return "review_required"
    if value == "unverified":
        return "unverified"
    raise ValueError(f"invalid review status: {value!r}")


def _string_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("expected a string mapping")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise TypeError("expected a string mapping")
        result[key] = item
    return result


def _float_mapping(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError("expected a numeric mapping")
    return {str(key): _as_float(item) for key, item in value.items()}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise TypeError("expected a string list")
    return list(value)


def _provenance(record_id: str, stored: Mapping[str, object]) -> Provenance:
    return Provenance(
        record_id=record_id,
        source_id=str(stored["source_id"]),
        chunk_id=str(stored["chunk_id"]),
        physical_page=_optional_int(stored.get("physical_page")),
        parser_version=str(stored["parser_version"]),
        source_sha256=_optional_str(stored.get("source_sha256")),
        extracted_at=datetime.fromisoformat(str(stored["extracted_at"])),
        effective_from=_optional_str(stored.get("effective_from")),
        effective_to=_optional_str(stored.get("effective_to")),
        confidence=_as_float(stored["confidence"]),
        review_status=_review_status(stored["review_status"]),
    )


class AcademicTools:
    """Each public method has one strongly typed operation input and packet output."""

    def __init__(self, repository: AcademicRepository) -> None:
        self.repository = repository

    def _evidence_for_record(self, record: CourseRecord, registry: EvidenceRegistry) -> str | None:
        if not record.chunk_id:
            return None
        stored = self.repository.source(record.chunk_id)
        if stored is None:
            return None
        evidence_id = stable_id("ev", stored["source_id"], stored["chunk_id"])
        registry.add(
            Evidence(
                evidence_id=evidence_id,
                source_id=str(stored["source_id"]),
                chunk_id=str(stored["chunk_id"]),
                title=str(stored["title"]),
                article=str(stored.get("article") or ""),
                quote=str(stored["text"]),
                page_url=str(stored.get("page_url") or "") or None,
                file_url=str(stored.get("file_url") or "") or None,
                provenance=_provenance(record.record_id, stored),
            )
        )
        return evidence_id

    @staticmethod
    def _course_facts(record: CourseRecord, evidence_id: str | None) -> list[Fact]:
        prefix = stable_id("fact", record.record_id)
        evidence_ids = (evidence_id,) if evidence_id else ()
        values: list[Fact] = [
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
            values.append(
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
        return values

    def _courses_packet(
        self, records: Iterable[CourseRecord], *, program_id: str, filters: tuple[str, ...]
    ) -> EvidencePacket:
        values = tuple(records)
        registry = EvidenceRegistry()
        facts: list[Fact] = []
        for record in values:
            facts.extend(self._course_facts(record, self._evidence_for_record(record, registry)))
        coverage = Coverage(
            program=ProgramCoverage(
                requested_program_id=program_id, resolved=True, dataset_complete=True
            ),
            fields=(
                FieldCoverage(
                    field="course_records", covered=True, source_record_count=len(values)
                ),
            ),
            course_set=CourseSetCoverage(
                database_match_count=len(values),
                returned_count=len(values),
                filters_applied=filters,
                dataset_complete=True,
            ),
        )
        return EvidencePacket(
            packet_id=stable_id("packet", program_id, *[value.record_id for value in values]),
            facts=tuple(facts),
            evidence=registry.values(),
            coverage=coverage,
            tool_results=("academic.list_courses",),
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
        return self._courses_packet(records, program_id=args.program_id, filters=filters)

    def get_course_detail(self, operation: GetCourseDetailOperation) -> EvidencePacket:
        args = operation.args
        program_id = args.program_id or "all-programs"
        course_ids = (args.course_id,) if args.course_id else ()
        records = self.repository.list_courses(
            cohort=args.cohort, program_id=args.program_id, course_ids=course_ids
        )
        if args.course_code:
            records = tuple(
                record
                for record in records
                if (record.code or "").upper() == args.course_code.upper()
            )
        return self._courses_packet(records, program_id=program_id, filters=("course_detail",))

    def _requirements_packet(
        self, cohort: int, program_id: str, module_ids: tuple[str, ...] = ()
    ) -> EvidencePacket:
        rows = self.repository.requirements(
            cohort=cohort, program_id=program_id, module_ids=module_ids
        )
        registry = EvidenceRegistry()
        facts: list[Fact] = []
        for row in rows:
            evidence_ids: tuple[str, ...] = ()
            chunk_id = row.get("chunk_id")
            if chunk_id:
                stored = self.repository.source(str(chunk_id))
                if stored:
                    evidence_id = stable_id("ev", stored["source_id"], stored["chunk_id"])
                    registry.add(
                        Evidence(
                            evidence_id=evidence_id,
                            source_id=str(stored["source_id"]),
                            chunk_id=str(stored["chunk_id"]),
                            title=str(stored["title"]),
                            article=str(stored.get("article") or ""),
                            quote=str(stored["text"]),
                            page_url=str(stored.get("page_url") or "") or None,
                            file_url=str(stored.get("file_url") or "") or None,
                            provenance=_provenance(str(row["record_id"]), stored),
                        )
                    )
                    evidence_ids = (evidence_id,)
            facts.append(
                Fact(
                    fact_id=stable_id("fact", row["record_id"], "required_credits"),
                    type="requirement",
                    subject=str(row["module_name"]),
                    predicate="required_credits",
                    value=_as_float(row["required_credits"])
                    if row["required_credits"] is not None
                    else "未结构化",
                    unit="credits",
                    source_record_ids=(str(row["record_id"]),),
                    evidence_ids=evidence_ids,
                )
            )
        return EvidencePacket(
            packet_id=stable_id("packet", program_id, "requirements", *module_ids),
            facts=tuple(facts),
            evidence=registry.values(),
            coverage=Coverage(
                program=ProgramCoverage(
                    requested_program_id=program_id, resolved=True, dataset_complete=True
                ),
                fields=(
                    FieldCoverage(
                        field="requirements", covered=bool(rows), source_record_count=len(rows)
                    ),
                ),
            ),
            tool_results=("academic.get_requirements",),
        )

    def get_graduation_requirements(
        self, operation: GetGraduationRequirementsOperation
    ) -> EvidencePacket:
        return self._requirements_packet(operation.args.cohort, operation.args.program_id)

    def get_module_requirements(self, operation: GetModuleRequirementsOperation) -> EvidencePacket:
        return self._requirements_packet(
            operation.args.cohort, operation.args.program_id, operation.args.module_ids
        )

    def audit_completed_courses(self, operation: AuditCompletedCoursesOperation) -> EvidencePacket:
        args = operation.args
        all_courses = self.repository.list_courses(cohort=args.cohort, program_id=args.program_id)
        completed_codes = {item.upper() for item in args.completed_course_codes}
        completed = [
            row
            for row in all_courses
            if row.course_id in args.completed_course_ids
            or (row.code or "").upper() in completed_codes
        ]
        packet = self._courses_packet(
            completed, program_id=args.program_id, filters=("completed_courses",)
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
                    evidence_id = stable_id("ev", stored["source_id"], stored["chunk_id"])
                    evidence[evidence_id] = Evidence(
                        evidence_id=evidence_id,
                        source_id=str(stored["source_id"]),
                        chunk_id=str(stored["chunk_id"]),
                        title=str(stored["title"]),
                        article=str(stored.get("article") or ""),
                        quote=str(stored["text"]),
                        page_url=str(stored.get("page_url") or "") or None,
                        file_url=str(stored.get("file_url") or "") or None,
                        provenance=_provenance(str(requirement["record_id"]), stored),
                    )
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
                    record.module_id == requirement["module_id"]
                    and record.record_id == fact.subject
                    for record in completed
                )
            ]
            total = sum(
                float(fact.value) for fact in matching if isinstance(fact.value, (int, float))
            )
            completed_evidence_ids = tuple(
                dict.fromkeys(evidence_id for fact in matching for evidence_id in fact.evidence_ids)
            )
            completed_fact = DerivedFact(
                fact_id=stable_id("fact", requirement["record_id"], "completed"),
                type="progress",
                subject=str(requirement["module_name"]),
                predicate="completed_credits",
                value=total,
                unit="credits",
                source_record_ids=(str(requirement["record_id"]),),
                evidence_ids=completed_evidence_ids,
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
                    dict.fromkeys((*requirement_evidence_ids, *completed_evidence_ids))
                ),
                operator="difference",
                input_fact_ids=(requirement_fact.fact_id, completed_fact.fact_id),
            )
            facts.extend((requirement_fact, completed_fact, remaining))
        return packet.model_copy(
            update={
                "facts": tuple(facts),
                "evidence": tuple(evidence.values()),
                "tool_results": ("academic.audit_progress",),
            }
        )

    def compare_programs(self, operation: CompareProgramsOperation) -> EvidencePacket:
        result = self.repository.compare_programs(
            cohort=operation.args.cohort, program_ids=operation.args.program_ids
        )
        registry = EvidenceRegistry()
        facts: list[Fact] = []
        numeric_difference = _float_mapping(result["numeric_difference"])
        programs = _string_mapping(result["programs"])
        intersection = _string_list(result["intersection"])
        for program_id, total in numeric_difference.items():
            records = self.repository.list_courses(
                cohort=operation.args.cohort, program_id=program_id
            )
            evidence_id = self._evidence_for_record(records[0], registry) if records else None
            facts.append(
                Fact(
                    fact_id=stable_id("fact", program_id, "minimum"),
                    type="comparison",
                    subject=programs.get(program_id, program_id),
                    predicate="graduation_min_credits",
                    value=total,
                    unit="credits",
                    evidence_ids=(evidence_id,) if evidence_id else (),
                )
            )
        facts.append(
            Fact(
                fact_id=stable_id("fact", operation.operation_id, "intersection"),
                type="comparison",
                subject="programs",
                predicate="shared_courses",
                value=intersection,
            )
        )
        return EvidencePacket(
            packet_id=stable_id("packet", operation.operation_id),
            facts=tuple(facts),
            evidence=registry.values(),
            coverage=Coverage(program=ProgramCoverage(resolved=True, dataset_complete=True)),
            tool_results=("academic.compare_programs",),
        )

    def retrieve_policy(self, operation: RetrievePolicyOperation) -> EvidencePacket:
        rows = self.repository.policy_candidates(
            operation.args.question,
            operation.args.cohort,
            as_of=operation.args.as_of,
        )
        conflicts = self.repository.policy_conflicts(rows)
        registry = EvidenceRegistry()
        facts: list[Fact] = []
        for row in rows:
            evidence_id = stable_id("ev", row["source_id"], row["chunk_id"])
            registry.add(
                Evidence(
                    evidence_id=evidence_id,
                    source_id=str(row["source_id"]),
                    chunk_id=str(row["chunk_id"]),
                    title=str(row["title"]),
                    article=str(row.get("article") or ""),
                    quote=str(row["text"]),
                    page_url=str(row.get("page_url") or "") or None,
                    file_url=str(row.get("file_url") or "") or None,
                    provenance=_provenance(str(row["chunk_id"]), row),
                )
            )
            facts.append(
                Fact(
                    fact_id=stable_id("fact", row["chunk_id"]),
                    type="policy",
                    subject=str(row["title"]),
                    predicate="excerpt",
                    value=str(row["text"]),
                    source_record_ids=(str(row["chunk_id"]),),
                    evidence_ids=(evidence_id,),
                    derivation="retrieved",
                )
            )
        return EvidencePacket(
            packet_id=stable_id("packet", operation.operation_id),
            facts=tuple(facts),
            evidence=registry.values(),
            coverage=Coverage(
                policy=PolicyCoverage(
                    support_sufficient=bool(rows) and not conflicts,
                    source_authoritative=any(
                        _as_float(row["authority_level"]) >= 1 for row in rows
                    ),
                    scope_matched=bool(rows),
                    version_resolved=bool(rows) and not conflicts,
                    conflict_free=not conflicts,
                )
            ),
            conflicts=conflicts,
            warnings=("policy_source_conflict",) if conflicts else (),
            tool_results=("policy.search",),
        )

    def resolve_source(self, operation: ResolveSourceOperation) -> EvidencePacket:
        stored = self.repository.source(operation.args.chunk_id)
        if stored is None:
            return EvidencePacket(
                packet_id=stable_id("packet", operation.operation_id),
                warnings=("source_not_found",),
                tool_results=("source.resolve",),
            )
        evidence_id = stable_id("ev", stored["source_id"], stored["chunk_id"])
        evidence = Evidence(
            evidence_id=evidence_id,
            source_id=str(stored["source_id"]),
            chunk_id=str(stored["chunk_id"]),
            title=str(stored["title"]),
            article=str(stored.get("article") or ""),
            quote=str(stored["text"]),
            page_url=str(stored.get("page_url") or "") or None,
            file_url=str(stored.get("file_url") or "") or None,
            provenance=_provenance(str(stored["chunk_id"]), stored),
        )
        return EvidencePacket(
            packet_id=stable_id("packet", operation.operation_id),
            evidence=(evidence,),
            tool_results=("source.resolve",),
        )


__all__ = ["AcademicTools"]
