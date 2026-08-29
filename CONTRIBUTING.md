# Contributing

Use Python 3.10–3.12 and the checked-in uv lock:

~~~powershell
uv sync --locked --extra dev
~~~

Before submitting a change, run:

~~~powershell
uv run python -m ruff check agent academic app evidence generation ingest query retrieval storage scripts eval tests/canonical
uv run python -m mypy agent academic app evidence generation ingest query retrieval storage scripts eval
uv run python -m pytest -q tests/canonical
uv run python -m scripts.run_demo --json
~~~

Changes to facts, claims, scope, coverage, release identity or promotion policy require both a positive case and an adversarial false-pass case. Keep public fixtures synthetic. Do not commit official school sources, restricted holdouts, student data, credentials, model weights or generated release artifacts.

Public behavior and trust-boundary changes should update the relevant document under <code>docs/</code>.
