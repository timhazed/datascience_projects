# Developer Memory — Multi-stage Dockerfile
# Python 3.13.8 | Apple Silicon (linux/arm64) compatible
#
# Build stages:
#   builder  — installs Poetry + all production deps into /app/.venv
#   production — slim runtime image; copies /app/.venv from builder

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.13.8-slim AS builder

WORKDIR /app

# System packages needed at build time only (git for GitPython, build-essential for some C exts)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pin Poetry version — never install an unpinned tool in CI/CD
ENV POETRY_VERSION=1.8.5 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

# Copy dependency manifests first — layer cache invalidates only when these change
COPY pyproject.toml poetry.lock* ./

# Install production deps into .venv inside /app; skip root package (no src yet)
RUN poetry install --only=main --no-root && rm -rf "${POETRY_CACHE_DIR}"

# Download spaCy NLP model for Presidio PII detection at build time
# en_core_web_lg: ~560 MB, better Organization/Location PII accuracy on M4 Max (TBD-02 resolved)
RUN .venv/bin/python -m spacy download en_core_web_lg

# ── Stage 2: production ───────────────────────────────────────────────────────
FROM python:3.13.8-slim AS production

WORKDIR /app

# Runtime system packages only (git for GitPython, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built virtualenv from builder — no Poetry in the runtime image
COPY --from=builder /app/.venv /app/.venv

# Activate venv for all subsequent RUN/CMD/ENTRYPOINT calls
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy application source — keep src/ and ui/ separate from system packages
COPY src/ ./src/
COPY ui/ ./ui/

# Non-root user for container security
RUN groupadd --system devmem && useradd --system --gid devmem devmem \
    && chown -R devmem:devmem /app \
    && mkdir -p /app/chroma_data && chown devmem:devmem /app/chroma_data
USER devmem

# MCP server HTTP transport port
EXPOSE 9000

# Streamlit port (used when this image runs as the streamlit service)
EXPOSE 8501

# Default: run MCP server over HTTP so Claude Code / Cursor can connect via
# the MCP config pointing at http://localhost:9000
CMD ["python", "-m", "src.server"]
