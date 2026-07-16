"""Run the real-provider smoke test against the final v5 runtime."""

from __future__ import annotations

import scripts.live_query_plan_smoke as base
from app.runtime_v5 import (
    build_local_query_plan_runtime,
    build_request_query_plan_runtime,
)


def main() -> None:
    base.build_local_query_plan_runtime = build_local_query_plan_runtime
    base.build_request_query_plan_runtime = build_request_query_plan_runtime
    base.main()


if __name__ == "__main__":
    main()
