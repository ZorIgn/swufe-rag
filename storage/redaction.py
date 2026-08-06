"""Redact request-scoped credentials before logs, errors, or traces escape."""

from __future__ import annotations

import re

SECRET_RE = re.compile(r"(?i)(?:sk|api[_-]?key|bearer)[_-]?[A-Za-z0-9._-]{8,}")


def redact_secrets(value: object) -> str:
    return SECRET_RE.sub("[REDACTED]", str(value))
