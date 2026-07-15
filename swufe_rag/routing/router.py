"""High-precision mixed-dialogue router with a fail-open general fallback."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Protocol

from generation.llm import LLMClient
from retrieval.query import normalize_query
from swufe_rag.routing.prompts import ROUTER_SYSTEM_PROMPT, build_router_prompt
from swufe_rag.routing.schemas import RouteContext, RouteDecision


class RouteClassifier(Protocol):
    def classify(self, question: str, context: RouteContext) -> Mapping[str, Any]: ...


class LLMRouteClassifier:
    """Converts an LLM response to strict routing JSON; it never executes it."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def classify(self, question: str, context: RouteContext) -> Mapping[str, Any]:
        response = self.client.generate(
            ROUTER_SYSTEM_PROMPT,
            build_router_prompt(
                question,
                last_mode=context.last_mode,
                last_intent=context.last_intent,
                last_college=context.last_college,
                last_cohort=context.last_cohort,
                last_rewritten_query=context.last_rewritten_query,
            ),
        ).strip()
        response = re.sub(r"^```(?:json)?\s*|\s*```$", "", response, flags=re.I)
        raw = json.loads(response)
        if not isinstance(raw, dict):
            raise ValueError("route classifier must return a JSON object")
        return raw


COHORT_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*级")
POLICY_YEAR_RE = re.compile(
    r"(?<!\d)((?:19|20)\d{2})\s*年(?:的)?(?:政策|规定|办法|通知|推免|保研|细则|方案)"
)
FOLLOW_UP_RE = re.compile(
    r"^(?:那|那么|然后|还有|还|这个|这种情况|重修通过后|以后|之后)"
    r"[^。！？!?]{0,30}[？?]?$"
)
SCHOOL_ENTITY_RE = re.compile(
    r"西南财经大学|西南财大|西财|SWUFE|教务处|学院教务办|计算机与人工智能学院|金融学院"
)
SCHOOL_POLICY_RE = re.compile(
    r"培养方案|推免|保研|推荐免试|选课|学分认定|毕业学分|专业选修|必修课|"
    r"修多少学分|多少学分|学分够不够|学分要求|"
    r"挂科|不及格|重修|考试|考核|缓考|休学|转学|转专业|专业分流|辅修|免修|学籍|"
    r"课程代码|教务系统|校内通知|官方通知|教务处通知"
    r"|毕业论文|毕业设计|学位授予|奖学金|数字课程"
)
CAMPUS_FACT_RE = re.compile(
    r"校园网|食堂|宿舍|校车|图书馆|一卡通|校内邮箱|校内网站|校内网址"
)
FACT_REQUEST_RE = re.compile(
    r"多少|几分|几门|条件|要求|规定|办法|政策|时间|什么时候|几点|"
    r"怎么办|还能|可以|是否|哪里|在哪|网址|链接|原文|通知|怎么选|应该选|"
    r"忘了|丢了|补办"
)
GENERAL_TASK_RE = re.compile(
    r"写代码|改代码|调试|翻译|润色|改写|写一封|写封|写邮件|写作文|"
    r"写策划|讲个笑话|编个故事|陪我聊|压力很大|心情不好|解释注意力机制|"
    r"什么是注意力机制|排序算法|Python|Java|SQL语法"
)
GENERAL_SWITCH_RE = re.compile(
    r"换个话题|不说这个了|别说这个了|给我写代码|讲个笑话|帮我翻译|帮我润色"
)

INTENT_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"推免|保研|推荐免试|竞赛.*加分|比赛.*加分"), "promotion"),
    (re.compile(r"培养方案|毕业学分|专业选修|必修课|修读|修满"), "curriculum"),
    (re.compile(r"选课|应该选|怎么选"), "course_selection"),
    (re.compile(r"转专业|专业分流"), "transfer"),
    (re.compile(r"考试|缓考|考核"), "assessment"),
    (re.compile(r"挂科|不及格|重修|休学|学籍|转学"), "academic_status"),
    (re.compile(r"学分认定|免修|辅修"), "credit"),
    (CAMPUS_FACT_RE, "campus_service"),
)


def _intent(question: str) -> str:
    if re.search(r"[A-Za-z]{2,}\d{2,4}", question) and re.search(
        r"课程|什么课|学分|选修|必修", question
    ):
        return "curriculum"
    for pattern, value in INTENT_RULES:
        if pattern.search(question):
            return value
    return "school_general"


def _search_terms(question: str, intent: str) -> tuple[str, ...]:
    additions = {
        "promotion": ("推荐免试", "推免资格", "综合成绩"),
        "curriculum": ("培养方案", "课程设置", "毕业要求"),
        "course_selection": ("选课", "课程设置"),
        "transfer": ("转专业", "申请条件"),
        "assessment": ("课程考核", "考试规定"),
        "academic_status": ("学籍管理", "重修", "不及格"),
        "credit": ("学分认定",),
        "campus_service": ("校内服务",),
    }.get(intent, ())
    entities = re.findall(r"[A-Za-z]{2,}\d{2,4}|[\u4e00-\u9fff]{2,12}", question)
    return tuple(dict.fromkeys([*entities[:6], *additions]))[:10]


class HybridRouter:
    def __init__(
        self,
        classifier: RouteClassifier | None = None,
        *,
        known_colleges: tuple[str, ...] = (),
    ) -> None:
        self.classifier = classifier
        self.known_colleges = known_colleges

    def _college(self, question: str, explicit: str | None) -> str | None:
        if explicit and explicit.strip():
            return explicit.strip()
        for college in self.known_colleges:
            if college in question:
                return college
        for common in ("计算机与人工智能学院", "金融学院"):
            if common in question:
                return common
        return None

    @staticmethod
    def _cohort(question: str, explicit: str | None) -> str | None:
        if explicit and explicit.strip():
            return explicit.strip()
        match = COHORT_RE.search(question)
        return match.group(1) if match else None

    @staticmethod
    def _policy_year(question: str) -> int | None:
        match = POLICY_YEAR_RE.search(question)
        return int(match.group(1)) if match else None

    @staticmethod
    def _explicit_general(question: str) -> bool:
        if GENERAL_SWITCH_RE.search(question):
            return True
        if GENERAL_TASK_RE.search(question):
            if SCHOOL_POLICY_RE.search(question):
                return False
            school_fact = SCHOOL_ENTITY_RE.search(question)
            writing_request = re.search(r"写|翻译|润色|改写|代码|笑话|故事", question)
            return bool(writing_request or not school_fact)
        return False

    @staticmethod
    def _definite_school_fact(question: str) -> bool:
        if SCHOOL_POLICY_RE.search(question):
            return True
        if re.search(r"[A-Za-z]{2,}\d{2,4}", question) and re.search(
            r"学分|课程|什么课|选修|必修", question
        ):
            return True
        if CAMPUS_FACT_RE.search(question) and FACT_REQUEST_RE.search(question):
            return True
        return bool(SCHOOL_ENTITY_RE.search(question) and FACT_REQUEST_RE.search(question))

    def _general(self, question: str, confidence: float = 0.9) -> RouteDecision:
        return RouteDecision(
            mode="general_chat",
            requires_school_facts=False,
            intent="general_chat",
            college=None,
            cohort=None,
            policy_year=None,
            rewritten_query=question,
            search_terms=(),
            confidence=confidence,
        )

    def _school(
        self,
        question: str,
        context: RouteContext,
        *,
        college: str | None,
        cohort: str | None,
        confidence: float,
    ) -> RouteDecision:
        intent = _intent(question)
        rewritten = question
        inherited = context.last_mode == "school_rag" and FOLLOW_UP_RE.search(question)
        if inherited and context.last_rewritten_query:
            rewritten = f"{context.last_rewritten_query}；用户追问：{question}"
            if context.last_intent:
                intent = context.last_intent
            college = college or context.last_college
            cohort = cohort or context.last_cohort
        return RouteDecision(
            mode="school_rag",
            requires_school_facts=True,
            intent=intent,
            college=college,
            cohort=cohort,
            policy_year=self._policy_year(question),
            rewritten_query=rewritten,
            search_terms=_search_terms(rewritten, intent),
            confidence=confidence,
        )

    def route(
        self,
        question: str,
        *,
        context: RouteContext | None = None,
        college: str | None = None,
        cohort: str | None = None,
    ) -> RouteDecision:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must not be blank")
        clean = normalize_query(question)
        context = context or RouteContext()
        resolved_college = self._college(clean, college)
        resolved_cohort = self._cohort(clean, cohort)

        if self._explicit_general(clean):
            return self._general(clean, 0.99)
        if self._definite_school_fact(clean):
            return self._school(
                clean,
                context,
                college=resolved_college,
                cohort=resolved_cohort,
                confidence=0.99,
            )
        if (
            context.last_mode == "school_rag"
            and FOLLOW_UP_RE.search(clean)
            and not GENERAL_SWITCH_RE.search(clean)
        ):
            return self._school(
                clean,
                context,
                college=resolved_college,
                cohort=resolved_cohort,
                confidence=0.94,
            )

        if self.classifier is not None:
            try:
                decision = RouteDecision.from_mapping(
                    self.classifier.classify(clean, context),
                    question=clean,
                    known_colleges=self.known_colleges,
                )
            except Exception:
                decision = None
            if decision is not None:
                if decision.mode == "school_rag":
                    return RouteDecision(
                        mode="school_rag",
                        requires_school_facts=True,
                        intent=decision.intent,
                        college=resolved_college or decision.college,
                        cohort=resolved_cohort or decision.cohort,
                        policy_year=decision.policy_year,
                        rewritten_query=decision.rewritten_query,
                        search_terms=decision.search_terms,
                        confidence=decision.confidence,
                    )
                return decision

        # Classifier failure and genuinely ambiguous input are normal chat.
        # The high-precision checks above already prevent explicit school facts
        # from escaping to a general model.
        return self._general(clean, 0.7)


_default_router = HybridRouter()


def route_question(
    question: str,
    *,
    context: RouteContext | None = None,
    college: str | None = None,
    cohort: str | None = None,
) -> RouteDecision:
    return _default_router.route(
        question, context=context, college=college, cohort=cohort
    )


__all__ = [
    "HybridRouter",
    "LLMRouteClassifier",
    "RouteClassifier",
    "route_question",
]
