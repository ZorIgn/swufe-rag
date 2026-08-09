from __future__ import annotations

from typing import Any

import pytest

from academic.database import CourseRecord
from academic.tools import AcademicTools
from agent.policies import RuntimePolicy
from agent.tools import PlanExecutor, standard_registry
from evidence.models import DerivedFact, EvidencePacket
from generation.synthesizer import DeterministicSynthesizer
from generation.validator import ClaimValidator
from query.planner import build_plan
from query.schemas import NormalizedQuery


class _PlanningRepository:
    def __init__(self) -> None:
        self.list_courses_calls = 0
        self.requirements_calls = 0
        self._courses = (
            CourseRecord(
                record_id="offering-c1",
                course_id="C1",
                code="C1",
                name="基础选修一",
                credits=2.0,
                semester="1",
                nature="选修",
                module_id="M1",
                module_name="选修模块",
                department="测试学院",
                source_id="source-plan",
                source_page=1,
                chunk_id="chunk-plan",
            ),
            CourseRecord(
                record_id="offering-c2",
                course_id="C2",
                code="C2",
                name="基础选修二",
                credits=2.0,
                semester="6",
                nature="选修",
                module_id="M1",
                module_name="选修模块",
                department="测试学院",
                source_id="source-plan",
                source_page=1,
                chunk_id="chunk-plan",
            ),
            CourseRecord(
                record_id="offering-c3",
                course_id="C3",
                code="C3",
                name="毕业实践",
                credits=2.0,
                semester="7",
                nature="必修",
                module_id="M2",
                module_name="实践模块",
                department="测试学院",
                source_id="source-plan",
                source_page=1,
                chunk_id="chunk-plan",
            ),
        )
        self._requirements = [
            {
                "record_id": "requirement-m1",
                "module_id": "M1",
                "module_name": "选修模块",
                "required_credits": 4.0,
                "chunk_id": "chunk-plan",
            },
            {
                "record_id": "requirement-m2",
                "module_id": "M2",
                "module_name": "实践模块",
                "required_credits": 2.0,
                "chunk_id": "chunk-plan",
            },
        ]

    def list_courses(self, **_: Any) -> tuple[CourseRecord, ...]:
        self.list_courses_calls += 1
        return self._courses

    def requirements(self, **_: Any) -> list[dict[str, object]]:
        self.requirements_calls += 1
        return list(self._requirements)

    @staticmethod
    def source(chunk_id: str) -> dict[str, object] | None:
        if chunk_id != "chunk-plan":
            return None
        return {
            "source_id": "source-plan",
            "chunk_id": "chunk-plan",
            "title": "测试培养方案",
            "article": "课程与模块要求",
            "text": (
                "测试培养方案：选修模块最低4学分，实践模块最低2学分；"
                "基础选修一2学分第1学期，基础选修二2学分第6学期，"
                "毕业实践2学分第7学期必修。"
            ),
            "physical_page": 1,
            "parser_version": "fixture-1",
            "source_sha256": "fixture-sha256",
            "extracted_at": "2026-01-01T00:00:00+00:00",
            "confidence": 1.0,
            "review_status": "verified",
            "page_url": "https://example.test/plan#page=1",
            "file_url": "https://example.test/plan",
        }

    @staticmethod
    def retrieval_documents() -> tuple[object, ...]:
        return ()

    @staticmethod
    def metadata() -> dict[str, str]:
        return {"dataset_version": "fixture-1"}


def _execute(
    completed: tuple[str, ...],
) -> tuple[_PlanningRepository, NormalizedQuery, EvidencePacket]:
    repository = _PlanningRepository()
    academic = AcademicTools(repository)  # type: ignore[arg-type]
    executor = PlanExecutor(
        standard_registry(academic, RuntimePolicy()),
        RuntimePolicy(tool_timeout_seconds=5.0),
    )
    query = NormalizedQuery(
        raw_question="能否在大四开始前完成培养方案？",
        intent="curriculum_feasibility",
        requested_outputs=("feasibility",),
        cohort=2024,
        program_ids=("P1",),
        program_names=("测试规划专业",),
        completed_course_ids=completed,
        deadline_semester=7,
        information_scope="curriculum",
    )
    return repository, query, executor.execute(build_plan(query))


def _status(packet: EvidencePacket) -> str:
    value = next(
        fact.value
        for fact in packet.facts
        if fact.predicate == "feasibility_status"
    )
    return str(value)


def test_zero_completed_courses_is_a_valid_zero_sum() -> None:
    _, _, packet = _execute(())
    completed = [fact for fact in packet.facts if fact.predicate == "completed_credits"]
    assert completed
    assert all(fact.value == 0 for fact in completed)
    assert all(not isinstance(fact, DerivedFact) for fact in completed)
    fact_ids = {fact.fact_id for fact in packet.facts}
    for remaining in (
        fact for fact in packet.facts if isinstance(fact, DerivedFact)
    ):
        assert remaining.input_fact_ids
        assert set(remaining.input_fact_ids) <= fact_ids


def test_partial_completion_can_be_feasible_from_remaining_capacity() -> None:
    _, query, packet = _execute(("C1", "C3"))
    assert _status(packet).startswith("可行")
    answer = DeterministicSynthesizer().synthesize(query, packet)
    answer = ClaimValidator().validate(answer, packet)
    assert answer.refused is False
    assert "结论：可行" in answer.answer_md
    assert "选修模块尚差2学分" in answer.answer_md
    assert "可覆盖" in answer.answer_md


def test_all_completed_is_feasible() -> None:
    _, query, packet = _execute(("C1", "C2", "C3"))
    assert _status(packet).startswith("可行")
    answer = DeterministicSynthesizer().synthesize(query, packet)
    answer = ClaimValidator().validate(answer, packet)
    assert answer.refused is False
    assert "所有已结构化模块的最低学分差额均为0" in answer.answer_md


def test_deadline_semester_is_exclusive_and_blocks_same_semester_course() -> None:
    _, query, packet = _execute(("C1", "C2"))
    assert _status(packet).startswith("不可行")
    answer = DeterministicSynthesizer().synthesize(query, packet)
    answer = ClaimValidator().validate(answer, packet)
    assert answer.refused is False
    assert "毕业实践安排在第7学期" in answer.answer_md
    assert "第7学期开始前" in answer.answer_md


@pytest.mark.parametrize("completed", [(), ("C1", "C3"), ("C1", "C2", "C3")])
def test_planning_dag_consumes_dependency_packets_without_requery(
    completed: tuple[str, ...],
) -> None:
    repository, _, packet = _execute(completed)
    assert repository.list_courses_calls == 1
    assert repository.requirements_calls == 1
    assert all(result.status == "success" for result in packet.execution_results)
    assert len(packet.execution_results) == 6
