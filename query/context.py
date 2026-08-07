"""Explicit request-scoped constraints carried separately from question text."""

from __future__ import annotations

from pydantic import Field

from query.schemas import StrictModel


class RequestContext(StrictModel):
    """Trusted transport scope; never synthesized by concatenating text."""

    cohort: int | None = Field(default=None, ge=2010, le=2100)
    college: str | None = None
    major: str | None = None
    as_of: str | None = None
    session_id: str | None = None
    completed_course_mentions: tuple[str, ...] = ()


__all__ = ["RequestContext"]
