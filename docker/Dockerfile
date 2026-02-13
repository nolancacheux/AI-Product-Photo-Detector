# Inference API Dockerfile — Multi-stage Build
# AI Product Photo Detector
#
# Stage 1: Build dependencies in a throw-away image
# Stage 2: Lean runtime with non-root user

# ── Build Stage ──────────────────────────────────────────────────────────────
FROM python:3.11.9-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install CPU-only PyTorch + project deps
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    ".[ui]" \
    gunicorn==22.0.0

# ── Runtime Stage ────────────────────────────────────────────────────────────
FROM python:3.11.9-slim

LABEL maintainer="Nolan Cacheux <cachnolan@gmail.com>"
LABEL version="1.0.0"

# Runtime-only system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/

# Prepare model directory (model injected via volume in dev, baked in prod)
RUN mkdir -p models/checkpoints logs && chown -R appuser:appuser /app

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    AIDETECT_MODEL_PATH=/app/models/checkpoints/best_model.pt \
    AIDETECT_LOG_LEVEL=INFO

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

CMD ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8080"]
