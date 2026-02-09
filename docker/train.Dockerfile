# Training Dockerfile
# AI Product Photo Detector

FROM python:3.11-slim AS builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install uv for faster dependency resolution
RUN pip install --no-cache-dir uv

# Install Python dependencies
COPY pyproject.toml .
RUN uv pip install --no-cache-dir . --system

# Production stage
FROM python:3.11-slim

# Labels
LABEL maintainer="Nolan Cacheux <cachnolan@gmail.com>"
LABEL version="1.0.0"
LABEL description="Training container for AI Product Photo Detector"

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/

# Create directories for data and models
RUN mkdir -p data/processed models/checkpoints && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
CMD ["python", "-m", "src.training.train", "--config", "configs/train_config.yaml"]
