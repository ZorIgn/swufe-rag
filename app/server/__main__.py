"""Run the one supported production HTTP entry point."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run("app.server:app", host=os.getenv("SWUFE_HOST", "0.0.0.0"), port=int(os.getenv("SWUFE_PORT", "8000")), reload=False)


if __name__ == "__main__":
    main()
