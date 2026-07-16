from __future__ import annotations

import pytest

from swufe_rag.query_plan_catalog_v3 import normalize_model_mapping


def test_scalar_course_nature_is_normalized_to_schema_list():
    raw = {
        "domain": "school",
        "intent": "course_list",
        "college": None,
        "major": "人工智能",
        "cohort": 2023,
        "semester": [6],
        "course_nature": "选修",
        "course_name": None,
        "requires_sql": True,
        "requires_rag": False,
        "missing_fields": [],
        "normalized_query": "查询第6学期选修课",
        "confidence": 0.95,
    }
    assert normalize_model_mapping(raw)["course_nature"] == ["选修"]


def test_nature_alias_and_short_cohort_are_normalized():
    raw = {
        "domain": "school",
        "intent": "course_list",
        "college": None,
        "major": "人工智能",
        "cohort": "23",
        "semester": 6,
        "course_nature": "专业方向课",
        "course_name": None,
        "requires_sql": True,
        "requires_rag": False,
        "missing_fields": [],
        "normalized_query": "查询",
    }
    value = normalize_model_mapping(raw)
    assert value["cohort"] == 2023
    assert value["semester"] == [6]
    assert value["course_nature"] == ["专业方向课程"]


def test_model_may_not_smuggle_sql_or_urls_into_plan():
    with pytest.raises(ValueError, match="forbidden fields"):
        normalize_model_mapping({"domain": "school", "sql": "DROP TABLE x"})
