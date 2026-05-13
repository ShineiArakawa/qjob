FROM python:3.14-slim

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# ── Layer 1: Install Python dependencies ─────────────────────────────────────
# Copy only the package definition first so this layer is rebuilt only when
# dependencies change, not on every source code change.
# README.md is required because pyproject.toml references it as the readme field.
COPY pyproject.toml README.md ./
# Create a minimal stub so hatchling can resolve the package without full source.
RUN mkdir -p qjob && touch qjob/__init__.py && \
    uv pip install --system --no-cache -e . && \
    rm -rf qjob/

# ── Layer 2: Copy application source ─────────────────────────────────────────
COPY qjob/ ./qjob/

# Copy Alembic config so migrations can be run from inside the container.
COPY alembic.ini ./

ENTRYPOINT ["qjob"]
