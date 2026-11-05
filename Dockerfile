FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app

FROM base AS deps
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

FROM base AS runtime
ENV PATH="/app/.venv/bin:$PATH" \
    UV_NO_SYNC=1 \
    UV_OFFLINE=1
COPY --from=deps /app/.venv /app/.venv
COPY pyproject.toml uv.lock ./
COPY app/ ./app/
COPY job_scraper/ ./job_scraper/
COPY data/ ./data/
COPY cli.py main.py ./
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/output \
    && chown -R appuser:appuser /app /home/appuser
USER appuser
EXPOSE 8010
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/healthz')"
CMD ["uvicorn", "app.api.__main__:app", "--host", "0.0.0.0", "--port", "8010"]
