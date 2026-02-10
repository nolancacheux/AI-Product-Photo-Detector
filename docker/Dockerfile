# Dockerfile for AI Product Photo Detector
# Optimized for Google Cloud Run

FROM python:3.11-slim

LABEL maintainer="Nolan Cacheux <cachnolan@gmail.com>"
LABEL version="1.0.0"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 curl libgl1 libglib2.0-0 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy full project for pip install
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/checkpoints/best_model.pt* ./models/checkpoints/

# Install deps (CPU-only PyTorch to save ~2GB)
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "." \
    && apt-get purge -y build-essential && apt-get autoremove -y \
    && rm -rf /root/.cache/pip

# Ensure dirs exist
RUN mkdir -p models/checkpoints logs && chown -R appuser:appuser /app

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    AIDETECT_MODEL_PATH=/app/models/checkpoints/best_model.pt \
    AIDETECT_LOG_LEVEL=INFO

EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/healthz || exit 1

CMD python -m uvicorn src.inference.api:app --host 0.0.0.0 --port ${PORT}
