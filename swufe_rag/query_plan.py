"""Strict question understanding for safe school-tool orchestration.

The language model may only fill this schema.  It never writes SQL, chooses
table names, or answers school facts.  A deterministic parser remains
available for offline operation and as a fail-closed fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Literal, Mapping

from generation.llm import LLMClient
from retrieval.query import normalize_query


Domain = Literal["school", "general"]
Intent = Literal[
    "course_list",
    "course_detail",
    "progress_audit",
    "school_requirement",
    "policy",
    "campus_service",
    "general_chat",
]
ToolName = Literal["sql", "rag", "sql+rag", "general_llm", "clarify"]

ALLOWED_INTENTS = {
    "course_list",
    "course_detail",
    "progress_audit",
    "school_requirement",
    "policy",
    "campus_service",
    "general_chat",
}
ALLOWED_NATURES = {"必修", "选修", "专业方向课程", "自由选修"}
ALLOWED_MISSING = {"college", "major", "cohort", "semester", "completed_courses"}

COHORT_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*(?:级|届)")
SHORT_COHORT_RE = re.compile(r"(?<!\d)(\d{2})\s*级")
COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,5}\d{3}\b", re.I)
SCHOOL_RE = re.compile(
    r"西南财(?:经大学|大)|SWUFE|教务|培养方案|课程|学分|学期|选课|免修|"
    r"推免|保研|缓考|重修|考试|学籍|转专业|辅修|毕业|学位|论文|学院"
)
COURSE_LIST_RE = re.compile(
    r"有什么.*课|有哪些.*课|哪些课|修什么课|课程列表|课程安排|课表|开设.*课"
)
COURSE_DETAIL_RE = re.compile(
    r"学分|学时|代码|课号|第几学期|哪个学期|什么时候开|必修|选修"
)
PROGRESS_RE = re.compile(r"已修|已经修|还差|差多少|怎么安排|如何安排|够不够")
SCHOOLWIDE_RE = re.compile(
    r"公共外语|大学英语|英语免修|通识教育核心|跨专业选修|教学周|"
    r"暑期学期|春季学期|秋季学期|艺术类课程|学校统一"
)
POLICY_RE = re.compile(
    r"推免|保研|缓考|重修|考试|考核|学籍|休学|复学|转专业|转学|"
    r"辅修|学分认定|毕业论文|学位授予|规定|办法|细则|条件"
)
MAJOR_ALIASES = {
    "计科": "计算机科学与技术",
    "计算机科学": "计算机科学与技术",
    "计算机科学与技术": "计算机科学与技术",
    "AI专业": "人工智能",
    "智能专业": "人工智能",
    "人工智能": "人工智能",
}

PLANNER_SYSTEM_PROMPT = """你是西南财经大学教务问答系统的问题理解器。
只输出一个 JSON 对象，不回答用户问题，不生成 SQL，不选择表名。
允许字段且必须全部输出：
domain, intent, college, major, cohort, semester, course_nature, course_name,
requires_sql, requires_rag, missing_fields, normalized_query, confidence。
domain 只能是 school/general。intent 只能是 course_list/course_detail/
progress_audit/school_requirement/policy/campus_service/general_chat。
semester 是 1-8 的整数数组。course_nature 只能使用 必修、选修、
专业方向课程、自由选修。cohort 是四位整数或 null。
公共外语、大学英语、教学周等校级统一要求不需要专业。
只有课程表精确事实需要 SQL；制度文字需要 RAG；修读进度和复杂建议通常 SQL+RAG。
缺少真正必需的信息才写入 missing_fields。不要因为数据库可能没覆盖而声称用户漏填。
"""


def _json_object(value: str) -> Mapping[str, Any]:
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", value.strip(), flags=re.I)
    raw = json.loads(clean)
    if not isinstance(raw, dict):
        raise ValueError("query plan must be a JSON object")
    return raw


def _semesters(question: str) -> tuple[int, ...]:
    values: list[int] = []
    zh = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8}
    for match in re.finditer(r"第([一二三四五六七八1-8])学期", question):
        token = match.group(1)
        values.append(int(token) if token.isdigit() else zh[token])
    stage = re.search(r"大([一二三四])([上下])?", question)
    if stage:
        first = (zh[stage.group(1)] - 1) * 2 + 1
        values.extend(
            [first]
            if stage.group(2) == "上"
            else [first + 1]
            if stage.group(2) == "下"
            else [first, first + 1]
        )
    return tuple(dict.fromkeys(value for value in values if 1 <= value <= 8))


def _cohort(question: str) -> int | None:
    match = COHORT_RE.search(question)
    if match:
        return int(match.group(1))
    short = SHORT_COHORT_RE.search(question)
    if short:
        value = int(short.group(1))
        return 2000 + value if value <= 80 else 1900 + value
    return None


def _major(question: str) -> str | None:
    compact = question.replace(" ", "")
    for alias, canonical in sorted(MAJOR_ALIASES.items(), key=lambda item: -len(item[0])):
        if alias.lower() in compact.lower():
            return canonical
    match = re.search(
        r"([\u4e00-\u9fffA-Za-z“”()（）]{2,36}?)(?:专业)(?=20\d{2}|\d{2}级|第|大[一二三四]|的|有|要|课|$)",
        compact,
    )
    if match:
        return match.group(1).strip("的")
    return None


@dataclass(frozen=True)
class QueryPlan:
    domain: Domain
    intent: Intent
    college: str | None
    major: str | None
    cohort: int | None
    semester: tuple[int, ...]
    course_nature: tuple[str, ...]
    course_name: str | None
    requires_sql: bool
    requires_rag: bool
    missing_fields: tuple[str, ...]
    normalized_query: str
    confidence: float
    parser: Literal["llm", "deterministic"] = "deterministic"

    @property
    def tool(self) -> ToolName:
        if self.missing_fields:
            return "clarify"
        if self.domain == "general":
            return "general_llm"
        if self.requires_sql and self.requires_rag:
            return "sql+rag"
        if self.requires_sql:
            return "sql"
        return "rag"

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        question: str,
        parser: Literal["llm", "deterministic"] = "llm",
    ) -> "QueryPlan":
        domain = raw.get("domain")
        if domain not in {"school", "general"}:
            raise ValueError("domain must be school or general")
        intent = raw.get("intent")
        if intent not in ALLOWED_INTENTS:
            raise ValueError("unsupported query-plan intent")
        college = raw.get("college")
        major = raw.get("major")
        for name, value in (("college", college), ("major", major)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be null or a non-empty string")
        cohort = raw.get("cohort")
        if isinstance(cohort, str) and cohort.isdigit():
            cohort = int(cohort)
        if cohort is not None and (
            isinstance(cohort, bool) or not isinstance(cohort, int) or not 1900 <= cohort <= 2100
        ):
            raise ValueError("cohort must be null or a four-digit year")
        semesters = raw.get("semester", [])
        if not isinstance(semesters, list):
            raise ValueError("semester must be a list")
        semester_values = tuple(
            dict.fromkeys(int(value) for value in semesters if str(value).isdigit())
        )
        if any(value < 1 or value > 8 for value in semester_values):
            raise ValueError("semester values must be between 1 and 8")
        natures = raw.get("course_nature", [])
        if not isinstance(natures, list) or any(value not in ALLOWED_NATURES for value in natures):
            raise ValueError("invalid course_nature")
        course_name = raw.get("course_name")
        if course_name is not None and not isinstance(course_name, str):
            raise ValueError("course_name must be null or a string")
        requires_sql = raw.get("requires_sql")
        requires_rag = raw.get("requires_rag")
        if not isinstance(requires_sql, bool) or not isinstance(requires_rag, bool):
            raise ValueError("requires_sql and requires_rag must be booleans")
        missing = raw.get("missing_fields", [])
        if not isinstance(missing, list) or any(value not in ALLOWED_MISSING for value in missing):
            raise ValueError("invalid missing_fields")
        normalized = raw.get("normalized_query")
        if not isinstance(normalized, str) or not normalized.strip():
            normalized = question
        confidence = raw.get("confidence", 0.8)
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("confidence must be numeric")
        return cls(
            domain=domain,
            intent=intent,
            college=college.strip() if isinstance(college, str) else None,
            major=major.strip().removesuffix("专业") if isinstance(major, str) else None,
            cohort=cohort,
            semester=semester_values,
            course_nature=tuple(dict.fromkeys(natures)),
            course_name=course_name.strip() if isinstance(course_name, str) and course_name.strip() else None,
            requires_sql=requires_sql,
            requires_rag=requires_rag,
            missing_fields=tuple(dict.fromkeys(missing)),
            normalized_query=normalized.strip()[:2000],
            confidence=max(0.0, min(1.0, float(confidence))),
            parser=parser,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "intent": self.intent,
            "college": self.college,
            "major": self.major,
            "cohort": self.cohort,
            "semester": list(self.semester),
            "course_nature": list(self.course_nature),
            "course_name": self.course_name,
            "requires_sql": self.requires_sql,
            "requires_rag": self.requires_rag,
            "missing_fields": list(self.missing_fields),
            "normalized_query": self.normalized_query,
            "confidence": self.confidence,
            "parser": self.parser,
            "tool": self.tool,
        }


def deterministic_plan(
    question: str,
    *,
    college: str | None = None,
    cohort: str | int | None = None,
    inherited_major: str | None = None,
    inherited_cohort: str | int | None = None,
) -> QueryPlan:
    clean = normalize_query(question)
    is_school = bool(SCHOOL_RE.search(clean))
    if not is_school:
        return QueryPlan(
            domain="general",
            intent="general_chat",
            college=None,
            major=None,
            cohort=None,
            semester=(),
            course_nature=(),
            course_name=None,
            requires_sql=False,
            requires_rag=False,
            missing_fields=(),
            normalized_query=clean,
            confidence=0.9,
        )
    resolved_cohort = _cohort(clean)
    if resolved_cohort is None and cohort is not None and str(cohort).isdigit():
        resolved_cohort = int(cohort)
    if resolved_cohort is None and inherited_cohort is not None and str(inherited_cohort).isdigit():
        resolved_cohort = int(inherited_cohort)
    resolved_major = _major(clean) or inherited_major
    semesters = _semesters(clean)
    schoolwide = bool(SCHOOLWIDE_RE.search(clean))
    if PROGRESS_RE.search(clean):
        intent: Intent = "progress_audit"
    elif COURSE_LIST_RE.search(clean):
        intent = "course_list"
    elif COURSE_CODE_RE.search(clean) or (COURSE_DETAIL_RE.search(clean) and resolved_major):
        intent = "course_detail"
    elif schoolwide:
        intent = "school_requirement"
    elif POLICY_RE.search(clean):
        intent = "policy"
    else:
        intent = "school_requirement"
    requires_sql = intent in {"course_list", "course_detail", "progress_audit"}
    requires_rag = intent in {"school_requirement", "policy", "campus_service", "progress_audit"}
    natures: list[str] = []
    if "必修" in clean:
        natures.append("必修")
    if re.search(r"专业方向|专业选修", clean):
        natures.extend(("选修", "专业方向课程"))
    elif "自由选修" in clean or "自选课" in clean:
        natures.append("自由选修")
    elif "选修" in clean:
        natures.append("选修")
    code = COURSE_CODE_RE.search(clean)
    missing: list[str] = []
    if requires_sql:
        if resolved_cohort is None:
            missing.append("cohort")
        if resolved_major is None:
            missing.append("major")
        if intent == "course_list" and not semesters:
            missing.append("semester")
    normalized = clean
    if resolved_major and resolved_cohort:
        normalized = f"查询{resolved_major}专业{resolved_cohort}级"
        if semesters:
            normalized += "第" + "、".join(str(value) for value in semesters) + "学期"
        normalized += "的" + ("课程列表" if intent == "course_list" else "课程与培养要求")
    return QueryPlan(
        domain="school",
        intent=intent,
        college=college,
        major=resolved_major,
        cohort=resolved_cohort,
        semester=semesters,
        course_nature=tuple(dict.fromkeys(natures)),
        course_name=code.group(0).upper() if code else None,
        requires_sql=requires_sql,
        requires_rag=requires_rag,
        missing_fields=tuple(missing),
        normalized_query=normalized,
        confidence=0.86,
    )


class QuestionPlanner:
    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client

    def plan(self, question: str, **scope: Any) -> QueryPlan:
        fallback = deterministic_plan(question, **scope)
        if self.client is None:
            return fallback
        context = {
            "question": question,
            "explicit_college": scope.get("college"),
            "explicit_cohort": scope.get("cohort"),
            "inherited_major": scope.get("inherited_major"),
            "inherited_cohort": scope.get("inherited_cohort"),
        }
        try:
            raw = _json_object(
                self.client.generate(
                    PLANNER_SYSTEM_PROMPT,
                    json.dumps(context, ensure_ascii=False),
                )
            )
            plan = QueryPlan.from_mapping(raw, question=question, parser="llm")
        except Exception:
            return fallback
        # Explicit API scope is authoritative; the model may not erase it.
        values = plan.to_dict()
        values.pop("parser", None)
        values.pop("tool", None)
        if scope.get("college"):
            values["college"] = scope["college"]
        if scope.get("cohort") and str(scope["cohort"]).isdigit():
            values["cohort"] = int(scope["cohort"])
        return QueryPlan.from_mapping(values, question=question, parser="llm")


__all__ = [
    "PLANNER_SYSTEM_PROMPT",
    "QueryPlan",
    "QuestionPlanner",
    "deterministic_plan",
]
