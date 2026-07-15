"""Strict, model-independent routing structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping


RouteMode = Literal["general_chat", "school_rag"]


@dataclass(frozen=True)
class RouteContext:
    last_mode: RouteMode | None = None
    last_intent: str | None = None
    last_college: str | None = None
    last_cohort: str | None = None
    last_rewritten_query: str | None = None
    recent_messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class RouteDecision:
    mode: RouteMode
    requires_school_facts: bool
    intent: str
    college: str | None
    cohort: str | None
    policy_year: int | None
    rewritten_query: str
    search_terms: tuple[str, ...]
    confidence: float

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        question: str,
        known_colleges: tuple[str, ...] = (),
    ) -> "RouteDecision":
        mode = raw.get("mode")
        if mode not in {"general_chat", "school_rag"}:
            raise ValueError("route mode must be general_chat or school_rag")
        requires = raw.get("requires_school_facts")
        if not isinstance(requires, bool):
            raise ValueError("requires_school_facts must be a boolean")
        if mode == "school_rag" and not requires:
            raise ValueError("school_rag must require school facts")
        if mode == "general_chat" and requires:
            raise ValueError("general_chat cannot require school facts")

        intent = raw.get("intent")
        if not isinstance(intent, str) or not intent.strip():
            raise ValueError("route intent must be a non-empty string")
        intent = intent.strip()[:80]

        college = raw.get("college")
        if college is not None:
            if not isinstance(college, str) or not college.strip():
                raise ValueError("college must be null or a non-empty string")
            college = college.strip()
            if known_colleges and college not in known_colleges:
                college = None

        cohort = raw.get("cohort")
        if cohort is not None:
            if not isinstance(cohort, str) or not (
                len(cohort.strip()) == 4 and cohort.strip().isdigit()
            ):
                raise ValueError("cohort must be null or a four-digit string")
            cohort = cohort.strip()

        policy_year = raw.get("policy_year")
        if policy_year is not None:
            if isinstance(policy_year, str) and policy_year.isdigit():
                policy_year = int(policy_year)
            if (
                isinstance(policy_year, bool)
                or not isinstance(policy_year, int)
                or not 1900 <= policy_year <= 2100
            ):
                raise ValueError("policy_year must be null or a valid year")

        rewritten = raw.get("rewritten_query")
        if not isinstance(rewritten, str) or not rewritten.strip():
            rewritten = question
        rewritten = rewritten.strip()[:2000]

        terms = raw.get("search_terms", [])
        if not isinstance(terms, list) or any(
            not isinstance(term, str) for term in terms
        ):
            raise ValueError("search_terms must be a list of strings")
        search_terms = tuple(
            dict.fromkeys(
                term.strip()[:80]
                for term in terms[:12]
                if term.strip()
            )
        )

        confidence = raw.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("confidence must be numeric")
        if not 0 <= float(confidence) <= 1:
            raise ValueError("confidence must be between 0 and 1")
        return cls(
            mode=mode,
            requires_school_facts=requires,
            intent=intent,
            college=college,
            cohort=cohort,
            policy_year=policy_year,
            rewritten_query=rewritten,
            search_terms=search_terms,
            confidence=float(confidence),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "requires_school_facts": self.requires_school_facts,
            "intent": self.intent,
            "college": self.college,
            "cohort": self.cohort,
            "policy_year": self.policy_year,
            "rewritten_query": self.rewritten_query,
            "search_terms": list(self.search_terms),
            "confidence": self.confidence,
        }


__all__ = ["RouteContext", "RouteDecision", "RouteMode"]
