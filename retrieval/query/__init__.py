"""Domain-aware query analysis for SWUFE policy retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


COURSE_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z]{2,}[- ]?\d{2,4})(?![A-Za-z0-9])"
)
ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百零〇0-9]+[条章节款]")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")

DOMAIN_EXPANSIONS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("挂科", "挂过", "没及格"), ("不及格", "重修")),
    (("保研",), ("推免", "推荐免试")),
    (("专选",), ("专业选修",)),
    (("必修课",), ("必修课程",)),
    (("学分够不够", "学分够吗", "能毕业吗"), ("毕业要求", "学分")),
    (("换专业",), ("转专业",)),
    (("休学一年",), ("休学", "学籍")),
)


def normalize_query(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", normalized).strip()


def lexical_tokens(text: str) -> set[str]:
    normalized = normalize_query(text).lower()
    tokens = set(re.findall(r"[a-z]+\d*|\d+(?:\.\d+)?%?", normalized))
    for run in re.findall(r"[\u4e00-\u9fff]+", normalized):
        try:
            import jieba
        except ImportError:
            tokens.update(run[index : index + 2] for index in range(len(run) - 1))
        else:
            tokens.update(token.strip() for token in jieba.lcut(run) if token.strip())
    return {token for token in tokens if token}


def required_query_entities(query: str) -> tuple[str, ...]:
    """Extract course-like entities that must occur in at least one source."""

    normalized = normalize_query(query)
    codes = [
        match.replace("-", "").replace(" ", "").upper()
        for match in COURSE_CODE_RE.findall(normalized)
    ]
    if codes:
        return tuple(dict.fromkeys(codes))

    entities: list[str] = []
    for match in re.finditer(
        r"([^，。！？!?；;]{2,24}?)(?:算不算|是不是|属于|是|算(?=专选|专业选修|选修|必修))(?:什么|哪类|专业|课程|专选|选修|必修)?",
        normalized,
    ):
        candidate = match.group(1).strip()
        candidate = re.split(r"的", candidate)[-1]
        candidate = re.sub(r"^(请问|想问|那么|这个)", "", candidate)
        candidate = re.sub(r"\d{4}级$", "", candidate)
        if 2 <= len(candidate) <= 10 and candidate not in {
            "课程",
            "这门课",
            "要求",
            "条件",
            "学分",
        }:
            entities.append(candidate)
    return tuple(dict.fromkeys(entities))


@dataclass(frozen=True)
class QueryAnalysis:
    original: str
    normalized: str
    expanded: str
    course_codes: tuple[str, ...]
    article_refs: tuple[str, ...]
    numbers: tuple[str, ...]
    tokens: frozenset[str]
    required_entities: tuple[str, ...]


def analyze_query(query: str) -> QueryAnalysis:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be blank")
    normalized = normalize_query(query)
    additions: list[str] = []
    for triggers, expansions in DOMAIN_EXPANSIONS:
        if any(trigger in normalized for trigger in triggers):
            additions.extend(term for term in expansions if term not in normalized)
    expanded = normalized
    if additions:
        expanded += " " + " ".join(dict.fromkeys(additions))

    codes = tuple(
        dict.fromkeys(
            match.replace("-", "").replace(" ", "").upper()
            for match in COURSE_CODE_RE.findall(normalized)
        )
    )
    return QueryAnalysis(
        original=query,
        normalized=normalized,
        expanded=expanded,
        course_codes=codes,
        article_refs=tuple(dict.fromkeys(ARTICLE_RE.findall(normalized))),
        numbers=tuple(dict.fromkeys(NUMBER_RE.findall(normalized))),
        tokens=frozenset(lexical_tokens(expanded)),
        required_entities=required_query_entities(normalized),
    )


def chunk_search_text(chunk: dict) -> str:
    return "\n".join(
        [
            str(chunk["doc_title"]),
            str(chunk["doc_title"]),
            str(chunk["article"]),
            str(chunk["article"]),
            str(chunk["article"]),
            str(chunk["text"]),
        ]
    )


def entity_coverage(analysis: QueryAnalysis, chunks: list[dict]) -> bool:
    if not analysis.required_entities:
        return True
    searchable = normalize_query("\n".join(chunk_search_text(chunk) for chunk in chunks))
    compact = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]", "", searchable).upper()
    return all(
        re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]", "", entity).upper() in compact
        for entity in analysis.required_entities
    )


def exact_signal_score(analysis: QueryAnalysis, chunk: dict) -> float:
    searchable = normalize_query(chunk_search_text(chunk)).upper()
    compact = searchable.replace("-", "").replace(" ", "")
    score = 0.0
    if analysis.course_codes:
        matched = sum(code in compact for code in analysis.course_codes)
        score += 0.45 * matched / len(analysis.course_codes)
    if analysis.article_refs:
        matched = sum(article.upper() in searchable for article in analysis.article_refs)
        score += 0.25 * matched / len(analysis.article_refs)
    if analysis.numbers:
        source_numbers = set(NUMBER_RE.findall(searchable))
        matched = sum(number in source_numbers for number in analysis.numbers)
        score += 0.2 * matched / len(analysis.numbers)
    title_tokens = lexical_tokens(str(chunk["doc_title"]) + " " + str(chunk["article"]))
    if analysis.tokens:
        score += 0.1 * len(analysis.tokens & title_tokens) / len(analysis.tokens)
    return min(score, 1.0)


__all__ = [
    "QueryAnalysis",
    "analyze_query",
    "chunk_search_text",
    "entity_coverage",
    "exact_signal_score",
    "lexical_tokens",
    "normalize_query",
    "required_query_entities",
]
