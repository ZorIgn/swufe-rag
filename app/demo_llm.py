"""Deterministic extractive LLM substitute for the isolated debug runtime."""

from __future__ import annotations

import re

from generation.prompts import REFUSAL_TEXT
from retrieval.query import analyze_query, lexical_tokens, normalize_query


SOURCE_RE = re.compile(
    r'<source id="(\d+)">.*?原文：\s*(.*?)\s*</source>', re.S
)
QUESTION_RE = re.compile(r"【(?:原)?问题】\s*(.*?)(?:\n\n|\Z)", re.S)
PASSAGE_RE = re.compile(r"[^\n。！？!?；;]+[。！？!?；;]?|\|[^\n]+\|")


class DemoGroundedClient:
    """Extracts one grounded passage; never used by the production runtime."""

    @staticmethod
    def _question(prompt: str) -> str:
        matches = QUESTION_RE.findall(prompt)
        return matches[-1].strip() if matches else ""

    @staticmethod
    def _score(question: str, passage: str) -> float:
        analysis = analyze_query(question)
        tokens = lexical_tokens(passage)
        overlap = len(analysis.tokens & tokens)
        coverage = overlap / len(analysis.tokens) if analysis.tokens else 0.0
        normalized = normalize_query(passage).upper().replace("-", "").replace(" ", "")
        exact = sum(code in normalized for code in analysis.course_codes) * 2.5
        exact += sum(article in passage for article in analysis.article_refs) * 1.5
        exact += sum(
            number in passage
            for number in analysis.numbers
            if not re.fullmatch(r"(?:19|20)\d{2}", number)
        ) * 0.3
        intent = 0.0
        if re.search(r"最迟|什么时候|多久|时间", question) and re.search(
            r"\d|小时|工作日|日前|月前|周前|学期前|开考前", passage
        ):
            intent += 1.0
        if re.search(r"多少|几分|比例|学分|分数|怎么计算|怎么算", question) and re.search(
            r"\d|%|％|×|=|学分|分", passage
        ):
            intent += 0.8
        if re.search(r"怎么计算|怎么算", question) and re.search(
            r"%|％|×|=|构成|占比", passage
        ):
            intent += 1.0
        if re.search(r"进考场|进入考场|迟到", question) and re.search(
            r"未进入考场|取消.*考试资格", passage
        ):
            intent += 1.2
        if "什么时候" in question and re.search(r"期中|期末|学期|大一|大二", passage):
            intent += 1.0
        if re.search(r"多少门|几门", question) and re.search(r"\d+\s*门", passage):
            intent += 1.2
        if "最长学习年限" in question and re.search(r"最长为|最长学习年限.*\d", passage):
            intent += 1.5
        header_only = len(passage) < 150 and (
            re.fullmatch(r"《[^》]+》[^。！？；;]*", passage)
            or re.fullmatch(r"下表为.{0,120}的原表[:：]", passage)
        )
        header_penalty = 2.0 if header_only else 0.0
        return coverage + exact + intent - header_penalty

    @staticmethod
    def _cite(passage: str, marker: int) -> str:
        text = passage.strip()
        if not text:
            return REFUSAL_TEXT
        text = re.sub(r"[；;。]\s*\n", "，", text)
        text = re.sub(r"[:：]\s*\n", "：", text)
        text = re.sub(r"\s*\n\s*", "", text)
        if text[-1] in "。！？!?；;":
            return f"{text[:-1]}[{marker}]{text[-1]}"
        return f"{text}[{marker}]。"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        question = self._question(user_prompt)
        candidates: list[tuple[float, int, str]] = []
        for marker_text, source in SOURCE_RE.findall(user_prompt):
            marker = int(marker_text)
            passages = [
                match.group(0).strip()
                for match in PASSAGE_RE.finditer(source)
                if match.group(0).strip()
            ] or [source.strip()]
            expanded_passages = list(passages)
            for index, passage in enumerate(passages):
                if passage.endswith(("：", ":")) and index + 1 < len(passages):
                    expanded_passages.append(
                        passage + "\n" + "\n".join(passages[index + 1 : index + 7])
                    )
            for passage in expanded_passages:
                candidates.append((self._score(question, passage), marker, passage))
        if not candidates:
            return REFUSAL_TEXT
        score, marker, passage = max(
            candidates, key=lambda item: (item[0], -item[1], len(item[2]))
        )
        if score < 0.12:
            return REFUSAL_TEXT
        return self._cite(passage, marker)


class DemoGeneralClient:
    """Deterministic ordinary-chat substitute used only in local tests/demo."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        question = user_prompt.rsplit("【当前问题】", 1)[-1].strip()
        if "注意力机制" in question:
            return "注意力机制会根据当前任务，为输入中的不同信息分配不同权重。"
        if "快速排序" in question:
            return "快速排序通过选取基准值，把序列划分为较小和较大的两部分后递归处理。"
        if "压力" in question or "心情" in question:
            return "听起来你最近有些辛苦。可以先把最急的一件事拆成很小的一步，我们慢慢来。"
        return f"当然可以。你问的是：{question}"
