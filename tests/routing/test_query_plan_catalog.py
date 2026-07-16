from __future__ import annotations

import sqlite3

from academic_audit.database import AcademicDatabase
from swufe_rag.query_plan_catalog import CatalogAwareQuestionPlanner


def _database(tmp_path):
    path = tmp_path / "catalog.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE course_offerings(
            cohort INTEGER, major TEXT, is_primary INTEGER
        );
        CREATE TABLE major_aliases(
            alias TEXT, canonical_major TEXT, cohort TEXT
        );
        INSERT INTO course_offerings VALUES
            (2023, '人工智能专业', 1),
            (2023, '会计学（注册会计师方向）专业', 1),
            (2024, '金融学专业', 1);
        INSERT INTO major_aliases VALUES('AI专业','人工智能专业',NULL);
        """
    )
    connection.commit()
    connection.close()
    return AcademicDatabase(path)


def test_natural_course_scope_becomes_sql_plan(tmp_path):
    planner = CatalogAwareQuestionPlanner(_database(tmp_path))
    plan = planner.plan("23级人工智能大三下有什么选修课？")
    assert plan.domain == "school"
    assert plan.intent == "course_list"
    assert plan.cohort == 2023
    assert plan.semester == (6,)
    assert plan.course_nature == ("选修",)
    assert plan.tool == "sql"


def test_schoolwide_requirement_does_not_require_major(tmp_path):
    planner = CatalogAwareQuestionPlanner(_database(tmp_path))
    plan = planner.plan("2023级公共外语一共要修多少学分？")
    assert plan.intent == "school_requirement"
    assert plan.tool == "rag"
    assert "major" not in plan.missing_fields


def test_genuinely_missing_scope_clarifies(tmp_path):
    planner = CatalogAwareQuestionPlanner(_database(tmp_path))
    plan = planner.plan("大三下有哪些课？")
    assert plan.tool == "clarify"
    assert set(plan.missing_fields) == {"cohort", "major"}


def test_unrelated_chat_stays_general(tmp_path):
    planner = CatalogAwareQuestionPlanner(_database(tmp_path))
    assert planner.plan("你好，今天怎么样？").tool == "general_llm"
