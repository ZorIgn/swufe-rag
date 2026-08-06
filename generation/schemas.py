"""Constrained output schemas for answer synthesis."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ClaimOutput(StrictModel):
    text: str = Field(min_length=1)
    fact_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
