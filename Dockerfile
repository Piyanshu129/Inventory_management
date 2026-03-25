# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – Dependency resolver
# Uses the official uv image to produce a fully resolved virtualenv
# ─────────────────────────────────────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

WORKDIR /app

# Copy dependency manifests first (layer-cache friendly)
COPY pyproject.toml uv.lock ./

# Install all runtime deps into /app/.venv (no torch / cuda in base image)
RUN uv sync --frozen --no-dev

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – Runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

# System deps (SQLite is built-in; only need CA certs for HTTPS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copy the resolved virtualenv from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY . .

# Make sure the venv Python is on PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Ensure data directories exist
RUN mkdir -p data/chroma_db

# Expose the FastAPI port
EXPOSE 8000

# Default command: start the API server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
