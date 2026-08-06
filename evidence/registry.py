"""Deduplicated evidence registration for a single agent request."""

from __future__ import annotations

from dataclasses import dataclass, field

from evidence.models import Evidence


@dataclass
class EvidenceRegistry:
    _values: dict[tuple[str, str | None], Evidence] = field(default_factory=dict)

    def add(self, evidence: Evidence) -> str:
        key = (evidence.source_id, evidence.chunk_id)
        current = self._values.get(key)
        if current is None:
            self._values[key] = evidence
            return evidence.evidence_id
        return current.evidence_id

    def values(self) -> tuple[Evidence, ...]:
        return tuple(self._values.values())
