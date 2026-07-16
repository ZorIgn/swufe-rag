"""Current production server entry point."""

from __future__ import annotations

import app.server_v2 as base
from app.runtime_v11 import build_local_query_plan_runtime, build_request_query_plan_runtime


base.build_local_query_plan_runtime = build_local_query_plan_runtime
base.build_request_query_plan_runtime = build_request_query_plan_runtime
app = base.create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("app.server_v11:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()


__all__ = ["app", "main"]
