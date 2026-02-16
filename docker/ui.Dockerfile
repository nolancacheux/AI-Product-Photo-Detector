# Streamlit UI Dockerfile — Multi-stage Build
# AI Product Photo Detector

# ── Build Stage ──────────────────────────────────────────────────────────────
FROM python:3.14.3-slim AS builder

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY requirements-ui.txt ./
RUN pip install --no-cache-dir -r requirements-ui.txt

# ── Runtime Stage ────────────────────────────────────────────────────────────
FROM python:3.14.3-slim

LABEL maintainer="Nolan Cacheux <cachnolan@gmail.com>"
LABEL version="1.0.0"

RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY src/ui/ ./src/ui/

RUN chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    API_URL=http://localhost:8080

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8501/_stcore/health').raise_for_status()" || exit 1

CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
