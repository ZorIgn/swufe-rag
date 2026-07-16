from pathlib import Path

from academic_audit.database import AcademicDatabase
from academic_audit.structured_executor_v4 import execute
from storage.metadata_db import MetadataDB
from swufe_rag.query_plan_catalog_v7 import CourseAwareQuestionPlanner


ROOT = Path(__file__).parents[2]


def _planner() -> tuple[CourseAwareQuestionPlanner, AcademicDatabase]:
    database = AcademicDatabase(ROOT / "data" / "academic_v2.sqlite3")
    return CourseAwareQuestionPlanner(database), database


def test_subject_word_variant_resolves_to_exact_course_sql() -> None:
    planner, _ = _planner()
    plan = planner.plan(
        "计算机科学与技术专业2023级计算机科学与技术导论有多少学分？"
    )
    assert plan.tool == "sql"
    assert plan.intent == "course_detail"
    assert plan.course_name == "计算机科学与技术学科 导论"
    assert plan.missing_fields == ()


def test_full_school_major_uses_sql_not_coverage_refusal() -> None:
    planner, _ = _planner()
    plan = planner.plan("金融学专业2023级第一学期有哪些课程？")
    assert plan.tool == "sql"
    assert plan.cohort == 2023
    assert plan.major == "金融学"
    assert plan.semester == (1,)


def test_exact_course_answer_is_bound_to_original_pdf_page() -> None:
    planner, database = _planner()
    plan = planner.plan(
        "计算机科学与技术专业2023级计算机科学与技术导论有多少学分？"
    )
    metadata = MetadataDB(ROOT / "data" / "metadata.sqlite3")
    result = execute(plan, plan.normalized_query, metadata_db=metadata, db=database)
    assert result is not None
    assert "1学分" in result.answer["answer_md"]
    assert result.answer["citations"]
    assert all("第452页" in item["article"] for item in result.answer["citations"])
    metadata.close()
