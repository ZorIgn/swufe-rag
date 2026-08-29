from __future__ import annotations

from dataclasses import replace

from academic.tools import AcademicTools
from query.schemas import CompareProgramsArgs, CompareProgramsOperation


def test_comparison_dimensions_preserve_program_lineage(
    canonical_runtime, monkeypatch
) -> None:
    repository = canonical_runtime.repository
    program_x = repository.resolve_program("测试专业X", 2024)
    program_y = repository.resolve_program("测试专业Y", 2024)
    assert program_x is not None
    assert program_y is not None

    records = {
        program_x.canonical_id: tuple(
            replace(record, nature="必修", module_name="实践模块")
            for record in repository.list_courses(
                cohort=2024, program_id=program_x.canonical_id
            )
        ),
        program_y.canonical_id: tuple(
            replace(record, nature="必修", module_name="实践模块")
            for record in repository.list_courses(
                cohort=2024, program_id=program_y.canonical_id
            )
        ),
    }

    def scoped_courses(
        *, cohort: int, program_id: str, **_filters: object
    ):
        assert cohort == 2024
        return records[program_id]

    monkeypatch.setattr(repository, "list_courses", scoped_courses)
    tools = AcademicTools(repository)
    packet = tools.compare_programs(
        CompareProgramsOperation(
            operation_id="compare-lineage",
            args=CompareProgramsArgs(
                cohort=2024,
                program_ids=(program_x.canonical_id, program_y.canonical_id),
                dimensions=("required_courses", "practice_requirements"),
            ),
        )
    )

    facts = {
        (fact.subject, fact.predicate): fact
        for fact in packet.facts
        if fact.predicate in {"required_courses", "practice_requirements"}
    }
    assert facts[(program_x.canonical_name, "required_courses")].value == ["TST101"]
    assert facts[(program_y.canonical_name, "required_courses")].value == ["TST201"]
    assert facts[(program_x.canonical_name, "practice_requirements")].value == ["TST101"]
    assert facts[(program_y.canonical_name, "practice_requirements")].value == ["TST201"]
    assert all(fact.source_record_ids for fact in facts.values())
    assert all(fact.evidence_ids for fact in facts.values())

    component = packet.coverage.for_operation("compare-lineage")
    assert component is not None
    assert component.complete
    assert component.trusted_evidence is True
    assert component.expected_count == 4
    assert component.returned_count == 4
