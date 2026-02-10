# AI Product Photo Detector

[![CI](https://github.com/nolancacheux/AI-Product-Photo-Detector/actions/workflows/ci.yml/badge.svg)](https://github.com/nolancacheux/AI-Product-Photo-Detector/actions/workflows/ci.yml)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20|%203.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://docs.docker.com/compose/)
[![Terraform](https://img.shields.io/badge/IaC-Terraform-purple.svg)](https://www.terraform.io/)

A MLOps system that classifies product images as **real** or **AI-generated**, helping e-commerce platforms detect fraudulent listings.

**Version:** 1.0.0

## Problem

E-commerce platforms face a growing threat: **AI-generated fake product images**. Scammers use generative models to create convincing product photos for items that don't exist. This project provides an API to detect these fake images.

## Features

- Binary classification: real vs AI-generated product images
- REST API with single and batch prediction endpoints
- Drift detection endpoint for monitoring model degradation
- Web UI (Streamlit) for interactive testing
- MLflow experiment tracking and model versioning
- Prometheus-compatible metrics endpoint
- Docker Compose deployment (API + MLflow + Streamlit)

## Architecture

<p align="center">
  <img src="docs/images/architecture.svg" alt="MLOps Pipeline Architecture" width="100%"/>
</p>

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed component documentation.

## Tech Stack

| Category | Technology |
|----------|------------|
| ML Framework | PyTorch + timm (EfficientNet-B0) |
| API | FastAPI + Uvicorn |
| Web UI | Streamlit |
| Experiment Tracking | MLflow |
| Data Versioning | DVC (Data Version Control) |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Cloud | GCP (Cloud Run, Artifact Registry, GCS) |
| IaC | Terraform |
| Monitoring | prometheus_client + structlog |
| Code Quality | Ruff (lint + format) + MyPy + pre-commit |

## Quick Start

### Installation

```bash
# Clone
git clone https://github.com/nolancacheux/AI-Product-Photo-Detector.git
cd AI-Product-Photo-Detector

# With uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,ui]"

# Or with pip
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ui]"
```

### Training

You can train locally (CPU) or on Google Colab (GPU, recommended).

#### Option A: Train on Google Colab (GPU) — Recommended

Train on a free T4 GPU in ~1 minute instead of ~15 minutes on CPU.

1. **Open the notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nolancacheux/AI-Product-Photo-Detector/blob/main/notebooks/train_colab.ipynb)
2. **Select GPU runtime:** `Runtime > Change runtime type > T4 GPU`
3. **Run all cells** — the notebook will clone the repo, download data, train, and offer file downloads
4. **Download the outputs** when prompted:
   - `best_model.pt` — trained model weights (~54 MB)
   - `mlflow_artifacts.zip` — experiment tracking data
5. **Place files in your local project:**

```
AI-Product-Photo-Detector/
├── models/
│   └── checkpoints/
│       └── best_model.pt        ← put the model here
└── mlruns/                       ← unzip mlflow_artifacts.zip here
```

#### Option B: Train locally (CPU)

```bash
# Download dataset (CIFAKE - CIFAR-10 real vs Stable Diffusion AI)
python scripts/download_cifake.py --max-per-class 2500

# Train model (~15 min on CPU)
python -m src.training.train --config configs/train_config.yaml
```

The trained model is automatically saved to `models/checkpoints/best_model.pt`.

### Run API

Once you have a trained model in `models/checkpoints/best_model.pt`:

```bash
# Start server
make serve
# or: uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
```

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/predict` | POST | Single image prediction |
| `/predict/batch` | POST | Batch prediction (up to 10 images) |
| `/metrics` | GET | Prometheus metrics |
| `/drift` | GET | Drift detection status |
| `/privacy` | GET | Privacy policy summary |
| `/docs` | GET | Swagger UI (interactive API docs) |

#### curl Examples

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -F "file=@tests/data/sample_real.jpg"

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -F "files=@tests/data/sample_real.jpg" \
  -F "files=@tests/data/sample_ai.png"

# Prometheus metrics
curl http://localhost:8000/metrics

# Drift status
curl http://localhost:8000/drift

# Privacy info
curl http://localhost:8000/privacy

# With API key (when auth is enabled)
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: your-key-here" \
  -F "file=@image.jpg"
```

#### Quick Test with Make

```bash
# Requires running API (make serve)
make predict        # Test single image prediction
make predict-batch  # Test batch prediction
```

### Run UI

```bash
streamlit run src/ui/app.py
```

### Docker

```bash
docker compose up -d
# API:        http://localhost:8080
# UI:         http://localhost:8501
# MLflow:     http://localhost:5000
```

## Data Version Control (DVC)

Dataset files are versioned with [DVC](https://dvc.org/). The actual data is **not** stored in Git — only lightweight `.dvc` pointer files are committed.

### Pull the Data

After cloning the repository, retrieve the dataset:

```bash
# Install DVC (included in dev dependencies)
pip install dvc

# Pull the tracked data
dvc pull
```

### Reproduce the Pipeline

DVC defines a reproducible pipeline (`dvc.yaml`) with two stages:

1. **download** — Downloads the CIFAKE dataset
2. **train** — Trains the EfficientNet-B0 model

```bash
# Run the full pipeline
dvc repro

# Run a specific stage
dvc repro download
dvc repro train
```

### Configure a Remote Storage

To share data with your team, set up a DVC remote:

```bash
# Example with S3
dvc remote add -d myremote s3://my-bucket/dvc-store
dvc push
```

## MLflow

Training runs are tracked with MLflow. Experiment data is stored in the `mlruns/` directory.

```bash
# View training experiments locally
mlflow ui --backend-store-uri mlruns --port 5000
# Then open http://localhost:5000
```

The `docker-compose.yml` includes an MLflow tracking server (port 5000) backed by SQLite for persistent experiment storage.

## API Reference

### POST /predict

Classify a single image.

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@image.jpg"
```

Response:
```json
{
  "prediction": "ai_generated",
  "probability": 0.87,
  "confidence": "high",
  "inference_time_ms": 45.2,
  "model_version": "1.0.0"
}
```

### POST /predict/batch

Classify multiple images (max 20).

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -F "files=@img1.jpg" -F "files=@img2.jpg"
```

### GET /health

Returns service health status.

### GET /metrics

Prometheus-formatted metrics.

### GET /drift

Returns drift detection status (mean probability, low confidence ratio, drift score).

## Project Structure

```
mlops_project/
├── .github/workflows/    # CI/CD (ci.yml, deploy.yml)
├── src/
│   ├── data/             # Data processing
│   ├── inference/        # FastAPI server & prediction
│   │   ├── api.py        # Endpoints
│   │   ├── predictor.py  # Model loading & inference
│   │   ├── schemas.py    # Pydantic models
│   │   ├── auth.py       # Optional API key auth
│   │   └── validation.py # Input validation
│   ├── monitoring/       # Metrics & drift detection
│   │   ├── metrics.py    # Prometheus-compatible metrics
│   │   └── drift.py      # Drift detector
│   ├── training/         # Training pipeline
│   │   ├── train.py      # Training loop (MLflow tracked)
│   │   ├── model.py      # EfficientNet-B0 architecture
│   │   ├── dataset.py    # PyTorch dataset
│   │   └── augmentation.py
│   ├── ui/               # Streamlit app
│   └── utils/            # Config & logging
├── tests/                # Unit tests
├── configs/              # Train/inference configs
├── docker/               # Dockerfiles (serve, train)
├── docs/                 # Architecture, contributing, incident scenario
├── models/               # Saved model checkpoints
├── notebooks/            # Colab training notebook
├── scripts/              # Data download utilities
├── terraform/            # GCP infrastructure as code
├── docker-compose.yml    # Local dev stack (API + UI + MLflow)
├── dvc.yaml              # DVC pipeline (download → train)
├── pyproject.toml        # Python project config
└── Makefile              # Dev commands
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) -- System design and component details
- [Contributing](docs/CONTRIBUTING.md) -- Development setup and workflow
- [Incident Scenario](docs/INCIDENT_SCENARIO.md) -- Data drift incident response scenario

## Testing

```bash
# Run all tests
pytest tests/ -v --cov=src

# Run specific test file
pytest tests/test_api.py -v
```

## Model Performance

| Metric | Value |
|--------|-------|
| Train Accuracy | 93.0% |
| Validation Accuracy | 82.9% |
| Training Time (T4 GPU) | ~45s |
| Training Time (CPU) | ~15 min |
| Inference | ~50ms/image |

## CI/CD Pipeline

### Workflows

| Workflow | Trigger | Description |
|----------|---------|-------------|
| **CI** (`ci.yml`) | Push/PR to `main` | Lint (ruff), type check (mypy), tests (pytest + coverage), security scan, Docker build (PR), deploy (push main) |
| **Manual Deploy** (`deploy.yml`) | Manual (`workflow_dispatch`) | Deploy any image tag (rollback support), dry run, health check |
| **DVC Pipeline** (`dvc.yml`) | Manual (`workflow_dispatch`) | Reproduce DVC pipeline (download data, train model) |

### CI Pipeline Details

On every PR and push to `main`:
1. **Lint & Format** — `ruff check` + `ruff format --check`
2. **Type Check** — `mypy` with strict mode
3. **Tests** — `pytest` with coverage on Python 3.11 & 3.12 (matrix)
4. **Security** — `pip-audit` (dependency CVEs) + `bandit` (code scan)
5. **Docker Build** — Validates Dockerfile on PRs (with GitHub Actions cache)
6. **Deploy** — Auto-deploys to Cloud Run on `main` push (after all checks pass)

### Required GitHub Secrets

Configure these in **Settings → Secrets and variables → Actions**:

| Secret | Description | Required for |
|--------|-------------|-------------|
| `GCP_PROJECT_ID` | Google Cloud project ID (e.g., `my-project-123456`) | Deploy |
| `GCP_SA_KEY` | Service account JSON key with roles: Cloud Run Admin, Artifact Registry Writer, Service Account User | Deploy |
| `DVC_REMOTE_URL` | DVC remote storage URL (e.g., `s3://bucket/dvc` or `gs://bucket/dvc`) | DVC pipeline (optional) |

#### GCP Service Account Setup

```bash
# Create service account
gcloud iam service-accounts create github-actions \
  --display-name="GitHub Actions CI/CD"

# Grant roles
for ROLE in run.admin artifactregistry.writer iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/$ROLE"
done

# Create JSON key
gcloud iam service-accounts keys create key.json \
  --iam-account=github-actions@$PROJECT_ID.iam.gserviceaccount.com

# Copy key.json content to GitHub secret GCP_SA_KEY, then delete local file
rm key.json
```

### Branch Protection (Recommended)

Configure in **Settings → Branches → Add rule** for `main`:

- ✅ Require pull request reviews before merging
- ✅ Require status checks to pass (select: `Lint & Format`, `Tests (Python 3.11)`, `Type Checking`)
- ✅ Require branches to be up to date before merging
- ✅ Do not allow bypassing the above settings

## Privacy & Data Handling

This API is designed with privacy-by-design principles:

- **No image storage** — Uploaded images are processed in-memory only and never saved to disk
- **No user tracking** — No cookies, sessions, or user identifiers
- **No personal data in logs** — Only operational metadata (prediction result, latency)
- **No personal data in metrics** — Prometheus metrics contain only aggregate counters
- **GDPR compliant** — Fully stateless service, no data retention

See [PRIVACY.md](PRIVACY.md) for the full privacy policy, or query the `/privacy` endpoint.

## Author

**Nolan Cacheux**
- GitHub: [nolancacheux](https://github.com/nolancacheux)
- LinkedIn: [nolancacheux](https://linkedin.com/in/nolancacheux)

## License

MIT License -- see [LICENSE](LICENSE)

---
*M2 MLOps -- JUNIA 2026*
