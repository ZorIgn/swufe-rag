# syntax=docker/dockerfile:1.7
# Keep the package resolver pinned and consume the checked-in uv.lock rather
# than resolving or upgrading dependencies during an image build.
FROM ghcr.io/astral-sh/uv:0.11.33 AS uv

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/models \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    SWUFE_RETRIEVAL_MODE=hybrid \
    SWUFE_RETRIEVAL_ARTIFACT_ROOT=/app/artifacts/retrieval

WORKDIR /app

COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock README.md ./

# The production image intentionally includes the real dense-index and
# cross-encoder dependencies.  Model weights and versioned artifacts are
# mounted at runtime, never downloaded while building the image.
RUN uv sync --locked --no-dev --extra retrieval --no-install-project

COPY agent/ ./agent/
COPY academic/ ./academic/
COPY app/ ./app/
COPY evidence/ ./evidence/
COPY generation/ ./generation/
COPY ingest/ ./ingest/
COPY query/ ./query/
COPY retrieval/ ./retrieval/
COPY scripts/ ./scripts/
COPY storage/ ./storage/
COPY config/ ./config/
RUN uv sync --locked --no-dev --extra retrieval

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data /app/artifacts/retrieval /app/artifacts/manifests /models \
    && chown -R appuser:appuser /app /models
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health/ready', timeout=3)"
CMD ["python", "-m", "app.server"]
