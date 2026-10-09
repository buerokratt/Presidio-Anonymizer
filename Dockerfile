FROM python:3.12.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_APP=app.py \
    FLASK_ENV=production \
    PORT=8000 \
    HOST=0.0.0.0

# Create non-root user FIRST
RUN groupadd -r presidio && \
    useradd -r -g presidio -m presidio

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set ownership of /app before copying files
RUN chown presidio:presidio /app

# Switch to non-root user
USER presidio

# Copy dependency files with correct ownership
COPY --chown=presidio:presidio pyproject.toml uv.lock ./

# Install everything in one layer as root to avoid duplication
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    && su presidio -c "uv sync --frozen --no-cache --no-dev" \
    && apt-get remove -y gcc g++ \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to presidio
USER presidio

# Copy application code
COPY --chown=presidio:presidio . .

# Create necessary directories
RUN mkdir -p /app/config /app/logs /app/models

EXPOSE $PORT

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["uv", "run", "python", "app.py"]