"""Shared fail-closed policy scope rules for repositories and retrievers."""

from __future__ import annotations

from collections.abc import Mapping

_UNIVERSAL_COHORTS = frozenset({"", "不限"})
_UNIVERSAL_COLLEGES = frozenset({"", "校级", "全校"})


def _string_values(value: object) -> frozenset[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return frozenset({stripped}) if stripped else frozenset()
    if isinstance(value, (list, tuple, set, frozenset)):
        return frozenset(
            stripped
            for item in value
            if isinstance(item, str) and (stripped := item.strip())
        )
    return frozenset()


def policy_scope_matches(
    document: Mapping[str, object],
    *,
    cohort: int | None,
    program_ids: tuple[str, ...],
    college_ids: tuple[str, ...],
) -> bool:
    """Return whether one document can safely answer the entire request scope.

    Missing request dimensions match only documents explicitly marked
    universal in that dimension.  For a request spanning multiple programs or
    colleges, a scoped document must cover the whole requested set; a policy
    for one member cannot be generalized to its siblings.
    """

    document_cohort = str(document.get("cohort") or "不限").strip()
    if cohort is None:
        if document_cohort not in _UNIVERSAL_COHORTS:
            return False
    elif document_cohort not in {*_UNIVERSAL_COHORTS, str(cohort)}:
        return False

    requested_colleges = _string_values(college_ids)
    document_college = str(document.get("college_id") or "").strip()
    if not requested_colleges:
        if document_college not in _UNIVERSAL_COLLEGES:
            return False
    elif (
        document_college not in _UNIVERSAL_COLLEGES
        and (
            len(requested_colleges) != 1
            or document_college not in requested_colleges
        )
    ):
        return False

    requested_programs = _string_values(program_ids)
    document_programs = _string_values(document.get("program_ids", ()))
    if not requested_programs:
        if document_programs:
            return False
    elif document_programs and not requested_programs.issubset(document_programs):
        return False

    return True


__all__ = ["policy_scope_matches"]
