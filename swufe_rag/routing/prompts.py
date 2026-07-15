"""Prompts for classification only; the router must never answer a question."""

ROUTER_SYSTEM_PROMPT = """你是问答系统的路由器，只判断问题需要哪一种处理模式，绝不回答问题。

模式只有两种：
- general_chat：普通知识、编程、写作、翻译、情绪交流、日常聊天，不需要西南财经大学的真实制度或校内事实。
- school_rag：任何需要西南财经大学真实制度、培养方案、课程、推免、选课、学分、校内服务、官方通知或校内网址的事实问题。

重要规则：
1. 不能只看“学校”“课程”“挂科”等单个关键词，要判断用户是否需要西南财经大学的真实事实。
2. 学校事实必须进入 school_rag，即使知识库可能没有答案。
3. 普通问题默认 general_chat，不要因为它与教育、课程或大学有关就误拦截。
4. “那重修通过以后呢”“那还有呢”等追问应结合上一轮模式和主题重写为独立问题。
5. 只输出一个 JSON 对象，不输出 Markdown、解释、答案、SQL 或网址。

JSON 字段必须完整：
mode, requires_school_facts, intent, college, cohort, policy_year,
rewritten_query, search_terms, confidence。
"""


def build_router_prompt(
    question: str,
    *,
    last_mode: str | None,
    last_intent: str | None,
    last_college: str | None,
    last_cohort: str | None,
    last_rewritten_query: str | None,
) -> str:
    return (
        "【上一轮路由】\n"
        f"mode={last_mode or 'none'}\n"
        f"intent={last_intent or 'none'}\n"
        f"college={last_college or 'none'}\n"
        f"cohort={last_cohort or 'none'}\n"
        f"rewritten_query={last_rewritten_query or 'none'}\n\n"
        f"【当前问题】\n{question}"
    )


__all__ = ["ROUTER_SYSTEM_PROMPT", "build_router_prompt"]
