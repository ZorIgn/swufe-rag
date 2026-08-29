"""Strict JSON primitives for signed manifests, reports, and active pointers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class StrictJSONError(ValueError):
    """Raised when a trust-boundary JSON document is ambiguous or non-standard."""


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise StrictJSONError(f"duplicate JSON key is not allowed: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise StrictJSONError(f"non-finite JSON number is not allowed: {value}")


def loads_strict_json(value: str | bytes, *, label: str = "JSON") -> object:
    """Parse RFC-compatible JSON while rejecting duplicates and non-finite numbers."""

    try:
        return json.loads(
            value,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except StrictJSONError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StrictJSONError(f"{label} is invalid JSON") from exc


def load_strict_json_file(path: str | Path, *, label: str = "JSON") -> object:
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise StrictJSONError(f"{label} is unreadable: {source}") from exc
    return loads_strict_json(raw, label=label)


def load_strict_json_snapshot(
    path: str | Path, *, label: str = "JSON"
) -> tuple[object, str, bytes]:
    """Read trust-boundary bytes once, then return parsed value, digest, and bytes."""

    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise StrictJSONError(f"{label} is unreadable: {source}") from exc
    return loads_strict_json(raw, label=label), hashlib.sha256(raw).hexdigest(), raw


def strict_json_object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise StrictJSONError(f"{label} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise StrictJSONError(f"{label} keys must be strings")
    return dict(value)


def canonical_json(value: object) -> bytes:
    """Serialize finite JSON deterministically for hashing and signatures."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise StrictJSONError("value is not finite canonical JSON") from exc
    return (encoded + "\n").encode("utf-8")


__all__ = [
    "StrictJSONError",
    "canonical_json",
    "load_strict_json_file",
    "load_strict_json_snapshot",
    "loads_strict_json",
    "strict_json_object",
]
