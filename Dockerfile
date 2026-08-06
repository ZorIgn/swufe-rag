FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
RUN pip install --upgrade pip
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
RUN pip install .
RUN useradd --create-home --uid 10001 appuser && mkdir -p /app/data /app/artifacts && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health/live', timeout=3)"
CMD ["python", "-m", "app.server"]
