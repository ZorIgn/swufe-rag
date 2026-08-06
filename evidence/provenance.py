"""Helpers that make provenance identifiers deterministic across rebuilds."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256

PARSER_VERSION = "canonical-1"


def stable_id(prefix: str, *parts: object) -> str:
    material = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{sha256(material.encode('utf-8')).hexdigest()[:20]}"


def extracted_now() -> datetime:
    return datetime.now(timezone.utc)
