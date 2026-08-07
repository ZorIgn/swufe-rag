"""Deterministic polarity signatures for policy and factual claims.

The validator needs a small semantic guard that does not depend on an LLM.  It
does not attempt full natural-language entailment; instead it recognizes the
directional policy predicates that are unsafe to invert while preserving the
same number or entity: permission, obligation, numeric boundary, and temporal
boundary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class PermissionPolarity(str, Enum):
    """Whether text permits or forbids the described action."""

    ALLOWED = "allowed"
    FORBIDDEN = "forbidden"


class RequirementPolarity(str, Enum):
    """Whether text makes the described action compulsory or optional."""

    REQUIRED = "required"
    OPTIONAL = "optional"


class BoundaryPolarity(str, Enum):
    """Whether a numeric policy statement is a lower or upper bound."""

    MINIMUM = "minimum"
    MAXIMUM = "maximum"


class TemporalPolarity(str, Enum):
    """Whether a temporal policy statement constrains an earlier or later point."""

    BEFORE = "before"
    AFTER = "after"


@dataclass(frozen=True)
class PolaritySignature:
    """Directional predicates observed in one claim, fact, or source excerpt."""

    permissions: frozenset[PermissionPolarity] = field(default_factory=frozenset)
    requirements: frozenset[RequirementPolarity] = field(default_factory=frozenset)
    boundaries: frozenset[BoundaryPolarity] = field(default_factory=frozenset)
    temporal: frozenset[TemporalPolarity] = field(default_factory=frozenset)

    def merge(self, *others: PolaritySignature) -> PolaritySignature:
        """Return the union of this signature and the supplied signatures."""

        return PolaritySignature(
            permissions=frozenset().union(self.permissions, *(item.permissions for item in others)),
            requirements=frozenset().union(
                self.requirements, *(item.requirements for item in others)
            ),
            boundaries=frozenset().union(self.boundaries, *(item.boundaries for item in others)),
            temporal=frozenset().union(self.temporal, *(item.temporal for item in others)),
        )


# ``不得少于``/``不得超过`` are numeric constraints, not an action-level ban.
_FORBIDDEN_RE = re.compile(
    r"(?:禁止|严禁|不允许|不可以|不准|不得(?!少于|低于|超过|晚于|早于)|不可(?!少于|低于|超过))"
)
_ALLOWED_RE = re.compile(r"(?:允许|可以|可申请|可选|准予|获准|有权)")
_REQUIRED_RE = re.compile(r"(?:必须|应当|须(?:要)?|需(?:要)?|必修|不得少于|不低于|至少|最低|最少)")
_OPTIONAL_RE = re.compile(r"(?:可选|任选|选修|自愿|非必修|无需|不要求|可不)")
_MINIMUM_RE = re.compile(r"(?:最低|最少|至少|不少于|不低于|下限|不得少于)")
_MAXIMUM_RE = re.compile(r"(?:最高|最多|至多|不超过|不得超过|上限)")
_BEFORE_RE = re.compile(
    r"(?:之前|以前|截至|截止(?:于)?|不得晚于|前(?=(?:[，。；、\s]|完成|修完|申请|提交|毕业|方可|$)))"
)
_AFTER_RE = re.compile(
    r"(?:之后|以后|自[^。；，]{0,32}起|后(?=(?:[，。；、\s]|开始|执行|生效|完成|修完|申请|提交|毕业|方可|$)))"
)


def text_signature(text: object) -> PolaritySignature:
    """Extract a deterministic policy signature from natural-language text."""

    value = str(text or "")
    permissions: set[PermissionPolarity] = set()
    requirements: set[RequirementPolarity] = set()
    boundaries: set[BoundaryPolarity] = set()
    temporal: set[TemporalPolarity] = set()
    if _ALLOWED_RE.search(value):
        permissions.add(PermissionPolarity.ALLOWED)
    if _FORBIDDEN_RE.search(value):
        permissions.add(PermissionPolarity.FORBIDDEN)
    if _REQUIRED_RE.search(value):
        requirements.add(RequirementPolarity.REQUIRED)
    if _OPTIONAL_RE.search(value):
        requirements.add(RequirementPolarity.OPTIONAL)
    if _MINIMUM_RE.search(value):
        boundaries.add(BoundaryPolarity.MINIMUM)
    if _MAXIMUM_RE.search(value):
        boundaries.add(BoundaryPolarity.MAXIMUM)
    if _BEFORE_RE.search(value):
        temporal.add(TemporalPolarity.BEFORE)
    if _AFTER_RE.search(value):
        temporal.add(TemporalPolarity.AFTER)
    return PolaritySignature(
        permissions=frozenset(permissions),
        requirements=frozenset(requirements),
        boundaries=frozenset(boundaries),
        temporal=frozenset(temporal),
    )


def predicate_signature(predicate: str) -> PolaritySignature:
    """Extract directional semantics encoded by a normalized fact predicate."""

    value = predicate.lower().replace("-", "_")
    permissions: set[PermissionPolarity] = set()
    requirements: set[RequirementPolarity] = set()
    boundaries: set[BoundaryPolarity] = set()
    temporal: set[TemporalPolarity] = set()
    if any(token in value for token in ("forbidden", "prohibited", "ban")):
        permissions.add(PermissionPolarity.FORBIDDEN)
    if any(token in value for token in ("allowed", "permitted", "permission")):
        permissions.add(PermissionPolarity.ALLOWED)
    if any(token in value for token in ("optional", "elective")):
        requirements.add(RequirementPolarity.OPTIONAL)
    if any(token in value for token in ("required", "requirement", "mandatory")):
        requirements.add(RequirementPolarity.REQUIRED)
    if any(
        token in value for token in ("minimum", "min_", "_min", "lower_bound", "required_credits")
    ):
        boundaries.add(BoundaryPolarity.MINIMUM)
    if any(token in value for token in ("maximum", "max_", "_max", "upper_bound")):
        boundaries.add(BoundaryPolarity.MAXIMUM)
    if "before" in value:
        temporal.add(TemporalPolarity.BEFORE)
    if "after" in value:
        temporal.add(TemporalPolarity.AFTER)
    return PolaritySignature(
        permissions=frozenset(permissions),
        requirements=frozenset(requirements),
        boundaries=frozenset(boundaries),
        temporal=frozenset(temporal),
    )


def fact_signature(predicate: str, subject: object, value: object) -> PolaritySignature:
    """Combine the machine predicate with its human-readable subject and value."""

    return predicate_signature(predicate).merge(text_signature(subject), text_signature(value))


def polarity_conflicts(claim: PolaritySignature, support: PolaritySignature) -> tuple[str, ...]:
    """Return stable conflict labels where claim direction inverts clear support."""

    conflicts: list[str] = []
    if (
        PermissionPolarity.ALLOWED in claim.permissions
        and PermissionPolarity.FORBIDDEN in support.permissions
        and PermissionPolarity.ALLOWED not in support.permissions
    ) or (
        PermissionPolarity.FORBIDDEN in claim.permissions
        and PermissionPolarity.ALLOWED in support.permissions
        and PermissionPolarity.FORBIDDEN not in support.permissions
    ):
        conflicts.append("allowed_vs_forbidden")
    if (
        RequirementPolarity.REQUIRED in claim.requirements
        and RequirementPolarity.OPTIONAL in support.requirements
        and RequirementPolarity.REQUIRED not in support.requirements
    ) or (
        RequirementPolarity.OPTIONAL in claim.requirements
        and RequirementPolarity.REQUIRED in support.requirements
        and RequirementPolarity.OPTIONAL not in support.requirements
    ):
        conflicts.append("required_vs_optional")
    if (
        BoundaryPolarity.MINIMUM in claim.boundaries
        and BoundaryPolarity.MAXIMUM in support.boundaries
        and BoundaryPolarity.MINIMUM not in support.boundaries
    ) or (
        BoundaryPolarity.MAXIMUM in claim.boundaries
        and BoundaryPolarity.MINIMUM in support.boundaries
        and BoundaryPolarity.MAXIMUM not in support.boundaries
    ):
        conflicts.append("minimum_vs_maximum")
    if (
        TemporalPolarity.BEFORE in claim.temporal
        and TemporalPolarity.AFTER in support.temporal
        and TemporalPolarity.BEFORE not in support.temporal
    ) or (
        TemporalPolarity.AFTER in claim.temporal
        and TemporalPolarity.BEFORE in support.temporal
        and TemporalPolarity.AFTER not in support.temporal
    ):
        conflicts.append("before_vs_after")
    return tuple(conflicts)


__all__ = [
    "BoundaryPolarity",
    "PermissionPolarity",
    "PolaritySignature",
    "RequirementPolarity",
    "TemporalPolarity",
    "fact_signature",
    "polarity_conflicts",
    "predicate_signature",
    "text_signature",
]
