"""Production runtime with policy-topic and authority-aware retrieval."""

from __future__ import annotations

from pathlib import Path
import re
from types import MethodType
from typing import Any

from app.runtime_v10 import (
    build_local_query_plan_runtime as _build_local,
    build_request_query_plan_runtime as _build_request,
)
from swufe_rag.orchestration import HybridRuntime
from swufe_rag.query_plan import QueryPlan


TOPICS = (
    (re.compile(r"推免|保研|推荐免试"), "promotion"),
    (re.compile(r"选课操作|选课指南|怎么选课"), "course_selection"),
    (re.compile(r"缓考|考试|考核"), "assessment"),
    (re.compile(r"学籍|重修|休学|复学|转学"), "academic_status"),
    (re.compile(r"学分认定|课程免修|辅修"), "credit"),
    (re.compile(r"毕业论文|毕业设计"), "thesis"),
    (re.compile(r"转专业|专业分流"), "transfer"),
)
EXPLICIT_COHORT_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}\s*(?:级|届)")
COLLEGE_RE = re.compile(r"学院|计智|计算机与人工智能|经济信息工程")


def _topic(question: str) -> str | None:
    return next((topic for pattern, topic in TOPICS if pattern.search(question)), None)


def _install_policy_retrieval(runtime: Any) -> Any:
    def retrieve_rag(
        self: Any,
        plan: QueryPlan,
        question: str,
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        terms = [plan.normalized_query, question, plan.major or ""]
        terms.extend(plan.course_nature)
        query = " ".join(dict.fromkeys(value for value in terms if value))
        topic = _topic(question)
        broad_school_promotion = bool(
            topic == "promotion"
            and not COLLEGE_RE.search(question)
            and not EXPLICIT_COHORT_RE.search(question)
        )
        return self.school_retrieve(
            query,
            top_k=max(top_k, 12),
            college="全校" if broad_school_promotion else None,
            cohort=(
                None
                if broad_school_promotion
                else str(plan.cohort) if plan.cohort is not None else None
            ),
            policy_year=None,
            topic=topic,
        )

    runtime._retrieve_rag = MethodType(retrieve_rag, runtime)
    return runtime


def build_local_query_plan_runtime(
    chunks_path: str | Path = "data/chunks.jsonl",
    *,
    sources_path: str | Path = "data/sources.csv",
    metadata_path: str | Path = "data/metadata.sqlite3",
    config_path: str | Path = "config.advanced.yaml",
    academic_database: str | Path = "data/academic_v2.sqlite3",
):
    return _install_policy_retrieval(
        _build_local(
            chunks_path,
            sources_path=sources_path,
            metadata_path=metadata_path,
            config_path=config_path,
            academic_database=academic_database,
        )
    )


def build_request_query_plan_runtime(
    base_runtime: HybridRuntime,
    api_key: str,
    *,
    config_path: str | Path = "config.advanced.yaml",
):
    return _install_policy_retrieval(
        _build_request(base_runtime, api_key, config_path=config_path)
    )


__all__ = ["build_local_query_plan_runtime", "build_request_query_plan_runtime"]
