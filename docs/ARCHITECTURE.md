# Architecture Documentation

## System Overview

The AI Product Photo Detector is a full MLOps system for detecting AI-generated product images. It covers the entire lifecycle: data versioning, model training with experiment tracking, CI/CD with automated deployment to Google Cloud Run, API serving, and production monitoring.

<p align="center">
  <img src="images/architecture.svg" alt="MLOps Pipeline Architecture" width="100%"/>
</p>

## Pipeline Stages

### 1. Data Pipeline

| Component | Technology | Purpose |
|-----------|------------|---------|
| Dataset | CIFAKE (CIFAR-10 + Stable Diffusion v1.4) | Real vs AI-generated image pairs |
| Version Control | DVC | Track dataset versions independently of Git |
| Remote Storage | Google Cloud Storage | Shared data/model store for the team |
| Pipeline | `dvc.yaml` (download → train) | Reproducible, cacheable pipeline stages |

**Data flow:**
```
HuggingFace/Kaggle → download_cifake.py → data/processed/ → DVC → GCS Bucket
```

### 2. Training Pipeline

| Component | Technology | Purpose |
|-----------|------------|---------|
| Framework | PyTorch + timm | EfficientNet-B0 with ImageNet transfer learning |
| Augmentation | torchvision transforms | Horizontal flip, rotation, color jitter, random crop |
| Scheduler | Cosine annealing + warmup | Learning rate schedule |
| Tracking | MLflow | Log params, metrics, artifacts per run |
| Checkpointing | `models/checkpoints/best_model.pt` | Save best model by validation accuracy |
| Configuration | `configs/train_config.yaml` | Centralized hyperparameters |

**Training flow:**
```
data/processed/ → PyTorch Dataset → EfficientNet-B0 → MLflow logs → best_model.pt
```

**Key files:**
- `src/training/train.py` — Training loop with MLflow integration
- `src/training/model.py` — EfficientNet-B0 architecture (binary classifier)
- `src/training/dataset.py` — PyTorch dataset with lazy loading
- `src/training/augmentation.py` — Data augmentation transforms

### 3. CI/CD Pipeline

| Component | Technology | Purpose |
|-----------|------------|---------|
| CI | GitHub Actions (`ci.yml`) | Lint (ruff) + Tests (pytest) on every push/PR |
| Docker Build | Multi-stage Dockerfile | Minimal production image (python:3.11-slim) |
| Registry | GCP Artifact Registry | Store tagged Docker images (`:sha` + `:latest`) |
| CD | GitHub Actions (`deploy.yml`) | Auto-deploy to Cloud Run on push to `main` |
| IaC | Terraform | Provision GCS, Artifact Registry, Cloud Run, IAM |

**CI/CD flow:**
```
git push → GitHub Actions → ruff + pytest → Docker build → Artifact Registry → Cloud Run deploy
```

**GitHub Actions workflows:**
- `.github/workflows/ci.yml` — Lint + Test + Docker build (PR) + Deploy (main)
- `.github/workflows/deploy.yml` — Manual deploy with custom image tag

**Terraform resources** (`terraform/`):
- `google_storage_bucket` — DVC data and model storage
- `google_artifact_registry_repository` — Docker image registry
- `google_cloud_run_v2_service` — Serverless API deployment
- `google_project_service` — Enable required GCP APIs
- IAM bindings for public access

### 4. Inference / Serving

| Component | Technology | Purpose |
|-----------|------------|---------|
| Framework | FastAPI + Uvicorn | Async REST API |
| Deployment | Google Cloud Run | Serverless, auto-scaling (0→3 instances) |
| Local | Docker Compose | Multi-service local development |
| UI | Streamlit | Interactive web interface for testing |
| Auth | Optional API key (`API_KEYS` env var) | Protect endpoints in production |
| Rate Limiting | slowapi | Prevent abuse (60/min predict, 10/min batch) |

**Inference flow:**
```
Client (cURL/Streamlit) → FastAPI → Validate → Preprocess (224×224) → EfficientNet-B0 → JSON response
```

**API files:**
- `src/inference/api.py` — FastAPI application and endpoint definitions
- `src/inference/predictor.py` — Model loading, preprocessing, inference
- `src/inference/schemas.py` — Pydantic request/response models
- `src/inference/auth.py` — Optional API key authentication
- `src/inference/validation.py` — Input validation (file type, size)

### 5. Monitoring

| Component | Technology | Purpose |
|-----------|------------|---------|
| Metrics | prometheus_client | Request latency, count, prediction distributions |
| Logging | structlog (JSON) | Structured, machine-readable logs |
| Drift Detection | KS Test + PSI | Detect input distribution shifts over time |

**Monitoring files:**
- `src/monitoring/metrics.py` — Prometheus-compatible metrics (exposed via `/metrics`)
- `src/monitoring/drift.py` — Statistical drift detection (exposed via `/drift`)

## API Endpoints

| Endpoint | Method | Description | Rate Limit |
|----------|--------|-------------|------------|
| `/predict` | POST | Classify a single image | 60/min |
| `/predict/batch` | POST | Classify up to 20 images | 10/min |
| `/health` | GET | Service health check | — |
| `/metrics` | GET | Prometheus-formatted metrics | — |
| `/drift` | GET | Drift detection status | — |

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_KEYS` | Comma-separated API keys | (auth disabled) |
| `LOG_LEVEL` | Logging level | `INFO` |
| `MODEL_PATH` | Path to model checkpoint | `models/checkpoints/best_model.pt` |
| `PORT` | Server port | `8000` |
| `ENVIRONMENT` | Deployment environment | `dev` |
| `GCS_BUCKET` | GCS bucket name (set by Terraform) | — |

### Training Configuration (`configs/train_config.yaml`)

| Parameter | Value |
|-----------|-------|
| Model | EfficientNet-B0 (pretrained) |
| Image size | 128×128 |
| Batch size | 64 |
| Epochs | 3 (increase for production) |
| Learning rate | 0.001 |
| Scheduler | Cosine with 2-epoch warmup |
| Early stopping | 5 epochs patience |

## Deployment Modes

### Local Development

```bash
# API only
uvicorn src.inference.api:app --host 0.0.0.0 --port 8000 --reload

# Full stack (API + UI + MLflow)
docker compose up -d
# API:    http://localhost:8080
# UI:     http://localhost:8501
# MLflow: http://localhost:5000
```

### Production (Cloud Run)

```bash
# Automated: push to main triggers deploy via GitHub Actions

# Manual deploy
gh workflow run deploy.yml -f image_tag=latest

# Terraform provisioning
cd terraform
terraform init
terraform plan
terraform apply
```

## Docker Architecture

```
docker/
├── Dockerfile           # Production API image (used by CI/CD + compose)
├── serve.Dockerfile     # Standalone inference server
└── train.Dockerfile     # Training environment

docker-compose.yml
├── api (FastAPI)        → :8080
├── ui (Streamlit)       → :8501 (depends on api)
└── mlflow (MLflow)      → :5000 (SQLite backend)
```

All services share a `detector-network` bridge network. The API container mounts `models/` and `configs/` as read-only volumes.

## Infrastructure (Terraform)

```
terraform/
├── main.tf                  # GCS, Artifact Registry, Cloud Run, IAM
├── variables.tf             # Configurable inputs
├── outputs.tf               # Cloud Run URL, bucket name
├── terraform.tfvars.example # Example variable values
└── README.md                # Terraform-specific docs
```

**Provisioned resources:**
- GCS bucket with versioning (data + models)
- Artifact Registry repository (Docker images)
- Cloud Run service (auto-scaling, health probes)
- IAM bindings (public API access)
- Required GCP API enablement
