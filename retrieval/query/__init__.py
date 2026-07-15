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
COHORT_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*级")
LATIN_ENTITY_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9+._-]{2,})(?![A-Za-z0-9])"
)
TEMPORAL_QUESTION_RE = re.compile(
    r"([^，。！？!?；;]{2,30}?)(?:什么时候|何时|几点)"
)
ASSISTANCE_QUESTION_RE = re.compile(
    r"([^，。！？!?；;]{2,24}?)(?:忘了|丢了|遗失了|找不到了|打不开|登录不了)(?:该)?怎么办"
)

DOMAIN_EXPANSIONS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("挂科", "挂过", "没及格"), ("不及格", "重修")),
    (("保研",), ("推免", "推荐免试")),
    (("专选",), ("专业选修",)),
    (("必修课",), ("必修课程",)),
    (("学分够不够", "学分够吗", "能毕业吗"), ("毕业要求", "学分")),
    (("换专业",), ("转专业",)),
    (("申请转专业前", "转专业前必须"), ("第一学年", "全部必修课程")),
    (("休学一年",), ("休学", "学籍")),
    (("考试必须带哪些证件", "考试带什么证件", "参加考试带哪些证件"), ("有效身份证件", "准考证")),
)

COHORT_SENSITIVE_TERMS = (
    "培养方案",
    "专业选修",
    "跨专业选修",
    "毕业最低",
    "毕业学分",
    "推免",
    "保研",
    "竞赛",
    "比赛",
    "加分",
)
REQUIRED_DOMAIN_PHRASES = (
    "专业选修课",
    "跨专业选修课",
    "毕业最低学分",
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
    for entity in LATIN_ENTITY_RE.findall(normalized):
        if entity.upper() not in {"HTTP", "HTTPS", "PDF", "RAG", "API"}:
            entities.append(entity)
    entities.extend(
        phrase for phrase in REQUIRED_DOMAIN_PHRASES if phrase in normalized
    )
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

    # Dense similarity is not an evidence guarantee.  Temporal questions often
    # receive a deceptively high score from an unrelated policy that merely
    # contains words such as "考核" or "进行".  Preserve the concrete subject
    # before the time interrogative as a must-match phrase.  Only conversational
    # and time-position modifiers are removed; the policy topic itself remains.
    for match in TEMPORAL_QUESTION_RE.finditer(normalized):
        candidate = match.group(1).strip()
        candidate = re.sub(r"^(请问|想问|我想知道|那么|这个)", "", candidate)
        candidate = re.sub(r"^(?:我(?:的)?|本人(?:的)?)", "", candidate)
        candidate = re.sub(r"(?:最迟|最晚|通常|一般|大概|大约)$", "", candidate)
        candidate = re.sub(
            r"(?:在)?大[一二三四](?:上|下)?(?:学期)?$", "", candidate
        )
        candidate = re.sub(r"(?:早上|上午|中午|下午|晚上)$", "", candidate)
        candidate = candidate.strip().rstrip("的")
        if 2 <= len(candidate) <= 24 and candidate not in {
            "考试",
            "课程",
            "规定",
            "政策",
            "学校",
        }:
            entities.append(candidate)

    # The object of a concrete help request is also evidence-bearing.  This
    # catches high-similarity but out-of-corpus requests such as account or
    # campus-network support without maintaining a blacklist of services.
    for match in ASSISTANCE_QUESTION_RE.finditer(normalized):
        candidate = match.group(1).strip()
        candidate = re.sub(r"^(请问|想问|我想知道|那么|这个)", "", candidate)
        candidate = re.sub(r"^(?:我(?:的)?|本人(?:的)?)", "", candidate)
        candidate = candidate.strip().rstrip("的")
        if 2 <= len(candidate) <= 24:
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
    def is_covered(entity: str) -> bool:
        entity_compact = re.sub(
            r"[^A-Za-z0-9\u4e00-\u9fff]", "", entity
        ).upper()
        if entity_compact in compact:
            return True
        # Word order may differ between the question and the policy text.  In
        # that case all subject tokens must still be present; a partial match
        # such as "博士/研究生/考核" without "中期" remains insufficient.
        tokens = {
            re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]", "", token).upper()
            for token in lexical_tokens(entity)
        }
        tokens.discard("")
        return bool(tokens) and all(token in compact for token in tokens)

    return all(is_covered(entity) for entity in analysis.required_entities)


def cohort_specific_coverage(analysis: QueryAnalysis, chunks: list[dict]) -> bool:
    """Require year-specific evidence for program and recommendation advice.

    School-wide timeless policies may use ``cohort=不限``.  Training plans,
    elective requirements, recommendation rules and competition points are
    versioned by entry cohort, so an explicitly named cohort must be supported
    by evidence carrying that exact cohort.
    """

    cohorts = tuple(dict.fromkeys(COHORT_RE.findall(analysis.normalized)))
    if not cohorts or not any(
        term in analysis.normalized for term in COHORT_SENSITIVE_TERMS
    ):
        return True
    for cohort in cohorts:
        scoped = [chunk for chunk in chunks if chunk.get("cohort") == cohort]
        if not scoped or not entity_coverage(analysis, scoped):
            return False
    return True


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
    "cohort_specific_coverage",
    "exact_signal_score",
    "lexical_tokens",
    "normalize_query",
    "required_query_entities",
]
