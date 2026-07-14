"""Canonical stable facade for the refined B and C pipelines."""

from generation.pipeline import answer
from retrieval.pipeline import retrieve

__all__ = ["retrieve", "answer"]
