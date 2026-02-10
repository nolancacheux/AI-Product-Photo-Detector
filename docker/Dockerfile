# Multi-stage Dockerfile for AI Product Photo Detector
# Optimized for Google Cloud Run (port 8080) and local docker-compose

# ============================================================
# Stage 1: Builder - install dependencies
# ============================================================
FROM python:3.11-slim AS builder

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY pyproject.toml .

# Install production deps only (no dev/ui extras)
# Use CPU-only PyTorch to drastically reduce image size
RUN pip install --no-cache-dir --target=/install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    . \
    && find /install -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
    && find /install -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true \
    && find /install -type d -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true \
    && find /install -name "*.pyc" -delete 2>/dev/null || true

COPY src/ ./src/

# ============================================================
# Stage 2: Runtime - minimal production image
# ============================================================
FROM python:3.11-slim AS runtime

LABEL maintainer="Nolan Cacheux <cachnolan@gmail.com>"
LABEL version="1.0.0"
LABEL description="AI Product Photo Detector - Cloud Run optimized"

# Install only runtime system deps (libgomp for torch, libglib for cv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local/lib/python3.11/site-packages/

# Copy application source
COPY --from=builder /build/src/ ./src/

# Copy configs
COPY configs/ ./configs/

# Copy model checkpoint if it exists (use a glob-safe pattern)
# For Cloud Run, the model is typically downloaded at startup or baked in
COPY models/checkpoints/best_model.pt* ./models/checkpoints/

# Create required directories and set ownership
RUN mkdir -p models/checkpoints logs \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    MODEL_PATH=/app/models/checkpoints/best_model.pt \
    LOG_LEVEL=INFO

EXPOSE ${PORT}

# Health check using curl (lighter than Python import)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Entrypoint: uvicorn with shell form to expand $PORT
CMD uvicorn src.inference.api:app --host 0.0.0.0 --port ${PORT}
