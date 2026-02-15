<h1 align="center">AI Product Photo Detector</h1>

<p align="center">
  <strong>Production-grade MLOps system for detecting AI-generated product photos in e-commerce</strong>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="https://docker.com/"><img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white" alt="Docker"></a>
  <a href="https://dvc.org/"><img src="https://img.shields.io/badge/DVC-945DD6?logo=dvc&logoColor=white" alt="DVC"></a>
  <a href="https://www.terraform.io/"><img src="https://img.shields.io/badge/Terraform-7B42BC?logo=terraform&logoColor=white" alt="Terraform"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
</p>

---

<p align="center">
  <a href="https://github.com/nolancacheux">Nolan Cacheux</a> · <a href="https://github.com/nolancacheux">GitHub</a> · <a href="https://www.linkedin.com/in/nolan-cacheux/">LinkedIn</a>
</p>

---

<p align="center"><img src="docs/architecture.svg" alt="Architecture" width="100%"></p>

## Overview

End-to-end machine learning system that classifies product photos as real or AI-generated using an EfficientNet-B0 model with Grad-CAM explainability. The project covers the full MLOps lifecycle: from DVC-managed data pipelines and GPU training (local, Colab, or Vertex AI) to a FastAPI serving layer with authentication, rate limiting, and Prometheus monitoring. Infrastructure is provisioned with Terraform and deployed serverlessly via Docker and GitHub Actions CI/CD.

## Features

| Category | Feature | Description |
|----------|---------|-------------|
| ML Model | EfficientNet-B0 + Grad-CAM | Transfer learning with ImageNet weights via `timm`, visual heatmap explainability |
| API | FastAPI with auth & rate limiting | Single, batch (up to 10), and explain endpoints with Pydantic v2 schemas |
| Training | 3 training modes | Local (Docker/CPU), Google Colab (free T4 GPU), Vertex AI (production T4 GPU) |
| Monitoring | Prometheus + Grafana + drift | 12+ custom metrics, auto-provisioned dashboards, real-time drift detection |
| Infrastructure | Terraform + Docker + Cloud Run | Modular IaC, full Docker Compose stack, serverless deployment |
| CI/CD | GitHub Actions (4 workflows) | Lint, test, build, deploy with quality gates (accuracy ≥ 0.85, F1 ≥ 0.80) |

## Training Options

<details>
<summary>Local Training: Docker-based, for development</summary>

**When to use:** Development, debugging, quick iterations on CPU.

**Prerequisites:** Python 3.11+, Docker & Docker Compose, Make

```bash
# Download the CIFAKE dataset
make data

# Train with default config
make train

# Or run the full DVC pipeline: download → validate → train
make dvc-repro

# Start MLflow UI to view experiments
make mlflow  # → http://localhost:5000
```

Training takes ~1–2 hours on CPU. Edit `configs/train_config.yaml` to adjust hyperparameters.

</details>

<details>
<summary>Google Colab: Free T4 GPU, one-click notebook</summary>

**When to use:** Quick experiments with free GPU, no local setup needed.

**Prerequisites:** Google account

The notebook (`notebooks/train_colab.ipynb`) handles everything automatically:

1. Installs dependencies and clones the repository
2. Downloads the CIFAKE dataset from HuggingFace
3. Trains EfficientNet-B0 with progress tracking
4. Evaluates the model and exports the checkpoint
5. Optionally uploads the trained model to GCS

Open in Colab → set runtime to **T4 GPU** → Run all cells. Training takes ~20 minutes.

</details>

<details>
<summary>Vertex AI Pipeline: Production training on GCP</summary>

**When to use:** Production retraining, CI/CD-triggered training, reproducible GPU runs.

**Prerequisites:** GCP project with Vertex AI enabled, `gcloud` CLI configured, GCS bucket with data

```bash
# Trigger via GitHub Actions
gh workflow run model-training.yml \
  -f epochs=15 \
  -f batch_size=64 \
  -f auto_deploy=true

# Or submit directly
python -m src.training.vertex_submit --epochs 15 --batch-size 64 --sync
```

Pipeline stages: Verify Data → Build Image → GPU Training (T4) → Evaluate → Quality Gate → Deploy

Training takes ~25 minutes. The quality gate blocks deployment if accuracy < 0.85 or F1 < 0.80.

</details>

## Quick Start

**Prerequisites:**

- Python 3.11+
- Docker & Docker Compose
- Make

**Installation:**

```bash
git clone https://github.com/nolancacheux/AI-Product-Photo-Detector.git
cd AI-Product-Photo-Detector
make dev  # Install dependencies + pre-commit hooks
```

**Run locally:**

```bash
docker compose up -d  # API + UI + MLflow + Prometheus + Grafana
```

| Service | URL |
|---------|-----|
| API | `http://localhost:8080` |
| Streamlit UI | `http://localhost:8501` |
| MLflow | `http://localhost:5000` |
| Prometheus | `http://localhost:9090` |
| Grafana | `http://localhost:3000` |

**Test:**

```bash
make test  # pytest with coverage
make lint  # ruff + mypy
```

<details>
<summary><strong>API Reference</strong></summary>

### Endpoints

| Method | Endpoint | Description | Rate Limit |
|--------|----------|-------------|------------|
| `POST` | `/predict` | Single image classification | 30/min |
| `POST` | `/predict/batch` | Batch classification (up to 10 images) | 5/min |
| `POST` | `/predict/explain` | Prediction + Grad-CAM heatmap | 10/min |
| `GET` | `/health` | Readiness probe (model status, uptime, drift) | - |
| `GET` | `/metrics` | Prometheus metrics (text format) | - |

### Authentication

Authentication is optional in development and enforced in production via environment variables.

| Variable | Description |
|----------|-------------|
| `API_KEYS` | Comma-separated list of valid API keys |
| `REQUIRE_AUTH` | Set to `true` to enforce authentication |

Pass the key via header: `X-API-Key: YOUR_KEY`

### Response Format

```json
{
  "prediction": "ai_generated",
  "probability": 0.87,
  "confidence": "high",
  "inference_time_ms": 45.2,
  "model_version": "1.0.0"
}
```

</details>

<details>
<summary><strong>Monitoring</strong></summary>

### Prometheus Metrics

All exposed at `GET /metrics` in Prometheus text format:

| Metric | Type | Description |
|--------|------|-------------|
| `aidetect_predictions_total` | Counter | Total predictions by status, class, confidence |
| `aidetect_prediction_latency_seconds` | Histogram | Per-prediction latency distribution |
| `aidetect_prediction_probability` | Histogram | Probability score distribution |
| `aidetect_batch_predictions_total` | Counter | Batch request count |
| `aidetect_batch_size` | Histogram | Images per batch request |
| `aidetect_model_loaded` | Gauge | Model load status (0/1) |
| `http_request_duration_seconds` | Histogram | HTTP latency by endpoint |
| `http_requests_total` | Counter | HTTP requests by method, endpoint, status |

### Grafana Dashboards

Pre-configured and auto-provisioned via `configs/grafana/provisioning/`:

- **Request throughput** - Requests/sec by endpoint
- **Latency percentiles** - p50, p90, p99 per endpoint
- **Prediction distribution** - Real vs AI-generated ratio over time
- **Model health** - Load status, drift alerts, error rates

Default credentials: `admin` / `admin`

### Drift Detection

Real-time monitoring of prediction distribution shifts using a sliding window over the last 1,000 predictions. Tracks mean probability, confidence distribution, and class ratios. Configurable alert thresholds with status available at `GET /drift`.

</details>

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| ML | PyTorch 2.0+, torchvision, timm (EfficientNet-B0), Grad-CAM |
| API | FastAPI, Uvicorn, Pydantic v2, slowapi |
| MLOps | DVC (pipelines + versioning), MLflow (experiment tracking), HuggingFace Datasets |
| Monitoring | Prometheus, Grafana, structlog (JSON logging), custom drift detection |
| Infrastructure | Docker, Docker Compose, Terraform (modular), Cloud Run, Artifact Registry |
| CI/CD | GitHub Actions (CI, CD, Model Training, PR Preview) |
| Cloud | Google Cloud Platform (Vertex AI, Cloud Run, GCS, Artifact Registry, Secret Manager) |

<details>
<summary><strong>Project Structure</strong></summary>

```
AI-Product-Photo-Detector/
├── .github/workflows/
│   ├── ci.yml                        # Lint + type-check + test (3.11, 3.12) + security
│   ├── cd.yml                        # Build → push → deploy → smoke test
│   ├── model-training.yml            # Vertex AI GPU training pipeline
│   └── pr-preview.yml                # PR preview deployments
├── configs/
│   ├── grafana/                      # Dashboard definitions + provisioning
│   ├── prometheus/                   # Alerting rules
│   ├── inference_config.yaml         # API server configuration
│   ├── pipeline_config.yaml          # Vertex AI pipeline parameters
│   ├── prometheus.yml                # Prometheus scrape targets
│   └── train_config.yaml             # Training hyperparameters
├── docker/
│   ├── Dockerfile                    # Production API image (non-root)
│   ├── Dockerfile.training           # Vertex AI GPU training image
│   └── ui.Dockerfile                 # Streamlit UI image
├── docs/
│   ├── architecture.svg              # System architecture diagram
│   ├── ARCHITECTURE.md               # Design decisions
│   ├── CICD.md                       # CI/CD pipeline docs
│   ├── CONTRIBUTING.md               # Contribution guidelines
│   ├── COSTS.md                      # Cloud cost analysis
│   ├── DEPLOYMENT.md                 # Deployment guide
│   ├── INFRASTRUCTURE.md             # Infrastructure docs
│   ├── MONITORING.md                 # Monitoring guide
│   └── TRAINING.md                   # Training pipeline docs
├── notebooks/
│   └── train_colab.ipynb             # Colab notebook (free T4 GPU)
├── scripts/                          # Dataset download & sample data utilities
├── src/
│   ├── data/
│   │   └── validate.py               # Dataset validation & integrity checks
│   ├── inference/
│   │   ├── api.py                    # FastAPI application & routes
│   │   ├── auth.py                   # API key auth (HMAC, constant-time)
│   │   ├── explainer.py              # Grad-CAM heatmap generation
│   │   ├── predictor.py              # Model inference engine
│   │   ├── rate_limit.py             # Rate limiting configuration
│   │   ├── routes/                   # Modular API routes
│   │   ├── schemas.py                # Pydantic request/response models
│   │   ├── shadow.py                 # Shadow model A/B testing
│   │   ├── state.py                  # Application state management
│   │   └── validation.py             # Image validation utilities
│   ├── monitoring/
│   │   ├── drift.py                  # Real-time drift detection
│   │   └── metrics.py                # Prometheus metric definitions
│   ├── pipelines/
│   │   ├── evaluate.py               # Model evaluation stage
│   │   └── training_pipeline.py      # End-to-end training orchestrator
│   ├── training/
│   │   ├── augmentation.py           # Data augmentation transforms
│   │   ├── dataset.py                # PyTorch Dataset implementation
│   │   ├── gcs.py                    # GCS upload/download helpers
│   │   ├── model.py                  # EfficientNet-B0 architecture
│   │   ├── train.py                  # Training loop with MLflow tracking
│   │   └── vertex_submit.py          # Vertex AI job submission CLI
│   ├── ui/
│   │   └── app.py                    # Streamlit web interface
│   └── utils/
│       ├── config.py                 # Settings management (Pydantic Settings)
│       ├── logger.py                 # Structured logging setup
│       └── model_loader.py           # Model loading utilities
├── terraform/
│   ├── environments/
│   │   ├── dev/                      # Development environment
│   │   └── prod/                     # Production environment
│   ├── modules/
│   │   ├── cloud-run/                # Cloud Run service module
│   │   ├── iam/                      # IAM bindings module
│   │   ├── monitoring/               # Monitoring module
│   │   ├── registry/                 # Artifact Registry module
│   │   └── storage/                  # GCS bucket module
│   ├── backend.tf                    # Terraform state backend (GCS)
│   └── versions.tf                   # Provider version constraints
├── tests/
│   ├── load/                         # Locust + k6 load tests
│   ├── conftest.py                   # Shared test fixtures
│   └── test_*.py                     # 20+ test modules (API, auth, model, training, ...)
├── docker-compose.yml                # Full stack: API + UI + MLflow + Prometheus + Grafana
├── dvc.yaml                          # DVC pipeline: download → validate → train
├── Makefile                          # Development commands
├── pyproject.toml                    # Dependencies & tool config
└── LICENSE                           # MIT License
```

</details>

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | System architecture and design decisions |
| [Training Guide](docs/TRAINING.md) | Training pipeline documentation (all 3 modes) |
| [Deployment](docs/DEPLOYMENT.md) | Deployment guide |
| [Monitoring](docs/MONITORING.md) | Monitoring and observability guide |
| [CI/CD](docs/CICD.md) | CI/CD pipeline documentation |
| [Infrastructure](docs/INFRASTRUCTURE.md) | Infrastructure and Terraform documentation |
| [Costs](docs/COSTS.md) | Cloud cost analysis |
| [Contributing](docs/CONTRIBUTING.md) | Contribution guidelines |

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">Made by <a href="https://github.com/nolancacheux">Nolan Cacheux</a></p>
