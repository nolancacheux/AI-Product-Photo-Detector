# 🔍 AI Product Photo Detector

[![CI](https://github.com/nolancacheux/mlops_project/actions/workflows/ci.yml/badge.svg)](https://github.com/nolancacheux/mlops_project/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Detect AI-generated product photos in e-commerce listings with production-grade MLOps.**

An end-to-end MLOps project that classifies product images as **real** or **AI-generated**, helping e-commerce platforms fight fraudulent listings.

## 🎯 Problem Statement

E-commerce platforms face a growing threat: **AI-generated fake product images**. Scammers use tools like Stable Diffusion and Flux to create convincing product photos for items that don't exist, leading to:
- Customer fraud and chargebacks
- Platform reputation damage
- Regulatory compliance issues

This project provides a **production-ready API** to detect these fake images in real-time.

## ✨ Features

### Core Capabilities
- **Binary Classification**: Real vs AI-generated product images
- **Probability Score**: Calibrated confidence score (0.0 - 1.0)
- **Multi-Generator Detection**: Trained on Stable Diffusion & Flux outputs
- **Explainability**: GradCAM heatmaps for prediction interpretation

### API Features
- **REST API**: Production-ready FastAPI with OpenAPI docs
- **Batch Processing**: Process up to 20 images in one request
- **Rate Limiting**: Configurable per-endpoint limits
- **Authentication**: API key and JWT token support
- **Response Caching**: Redis or in-memory caching
- **Input Validation**: Comprehensive security checks

### MLOps Features
- **Model Versioning**: MLflow experiment tracking
- **Hyperparameter Tuning**: Optuna optimization
- **Data Augmentation**: CutMix, MixUp, RandAugment
- **Model Calibration**: Temperature scaling for reliable probabilities
- **Drift Detection**: Monitor input distribution shifts

### Deployment
- **Docker**: Multi-stage builds for API and UI
- **Kubernetes**: Helm charts with HPA autoscaling
- **CI/CD**: GitHub Actions for testing and deployment
- **Observability**: Prometheus metrics, structured logging

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Clients                              │
│  Web UI (Streamlit)  │  REST API  │  Batch Jobs             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                       │
│  Rate Limiting → Auth → Validation → Cache → Inference      │
├─────────────────────────────────────────────────────────────┤
│  Endpoints:                                                  │
│  • POST /predict         - Single image                     │
│  • POST /predict/batch   - Batch (up to 20)                 │
│  • POST /explain         - GradCAM visualization            │
│  • GET  /health          - Health check                     │
│  • GET  /metrics         - Prometheus metrics               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    ML Inference Engine                       │
│  EfficientNet-B0 → Calibration → Explainability             │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| **ML Framework** | PyTorch + timm (EfficientNet-B0) |
| **API** | FastAPI + Uvicorn |
| **Web UI** | Streamlit |
| **MLOps** | MLflow + DVC + Optuna |
| **Containerization** | Docker + Kubernetes |
| **CI/CD** | GitHub Actions |
| **Observability** | Prometheus + Grafana + structlog |

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (optional)
- CUDA-capable GPU (optional, for faster inference)

### Installation

```bash
# Clone the repository
git clone https://github.com/nolancacheux/mlops_project.git
cd mlops_project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -e ".[dev,ui]"

# Install pre-commit hooks
pre-commit install
```

### Training

```bash
# Create sample dataset
python scripts/create_sample_data.py --output data/processed

# Train model
python -m src.training.train --config configs/train_config.yaml

# Run hyperparameter optimization (optional)
python -m src.training.hyperopt --config configs/train_config.yaml --n-trials 50
```

### Inference

```bash
# Start API server
uvicorn src.inference.api:app --host 0.0.0.0 --port 8000 --reload

# Test prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"

# Get explanation (GradCAM)
curl -X POST "http://localhost:8000/explain?alpha=0.5" \
  -F "file=@test_image.jpg" \
  --output explanation.png
```

### Web UI

```bash
# Start Streamlit interface
streamlit run src/ui/app.py --server.port 8501
```

### Docker

```bash
# Build and start all services
docker-compose up -d

# Access:
# - API: http://localhost:8000
# - UI: http://localhost:8501
# - MLflow: http://localhost:5000
```

### Kubernetes (Helm)

```bash
# Deploy with Helm
helm install ai-detector ./helm/ai-detector \
  --namespace ai-detector \
  --create-namespace

# Or with Kustomize
kubectl apply -k k8s/
```

## 📊 API Reference

### POST /predict
Classify a single image.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "X-API-Key: your-api-key" \
  -F "file=@image.jpg"
```

**Response:**
```json
{
  "prediction": "ai_generated",
  "probability": 0.87,
  "confidence": "high",
  "inference_time_ms": 45.2,
  "model_version": "1.0.5"
}
```

### POST /predict/batch
Classify multiple images (max 20).

**Response:**
```json
{
  "results": [
    {"filename": "img1.jpg", "prediction": "real", "probability": 0.12},
    {"filename": "img2.jpg", "prediction": "ai_generated", "probability": 0.94}
  ],
  "total": 2,
  "successful": 2,
  "failed": 0,
  "total_inference_time_ms": 89.5
}
```

### POST /explain
Get GradCAM heatmap overlay.

**Query Parameters:**
- `alpha` (float): Heatmap transparency (0.1-0.9, default 0.5)

**Response:** PNG image with heatmap overlay

## 📁 Project Structure

```
mlops_project/
├── .github/workflows/     # CI/CD pipelines
├── configs/               # Training & deployment configs
├── data/                  # Datasets (DVC tracked)
├── docker/                # Dockerfiles
├── docs/                  # Documentation
├── helm/                  # Helm charts
├── k8s/                   # Kubernetes manifests
├── notebooks/             # Exploration notebooks
├── src/
│   ├── data/             # Data processing
│   ├── inference/        # API & prediction
│   ├── monitoring/       # Metrics & drift
│   ├── training/         # Training pipeline
│   ├── ui/               # Streamlit app
│   └── utils/            # Shared utilities
└── tests/                 # Unit & integration tests
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_KEYS` | Comma-separated API keys | (auth disabled) |
| `JWT_SECRET` | JWT signing secret | (auto-generated) |
| `REDIS_URL` | Redis connection URL | (memory cache) |
| `LOG_LEVEL` | Logging level | INFO |
| `REQUIRE_AUTH` | Force authentication | false |

### Rate Limits

| Endpoint | Limit |
|----------|-------|
| `/predict` | 60/minute |
| `/predict/batch` | 10/minute |
| `/explain` | 30/minute |

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 83% |
| Precision | 82% |
| Recall | 84% |
| F1-Score | 83% |
| Inference Latency | ~50ms |

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

## 📖 Documentation

- [Architecture Documentation](docs/ARCHITECTURE.md)
- [Product Requirements (PRD)](docs/PRD.md)
- [API Documentation](http://localhost:8000/docs) (when running)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/amazing-feature`)
3. Commit changes (`git commit -m 'feat: add amazing feature'`)
4. Push to branch (`git push origin feat/amazing-feature`)
5. Open a Pull Request

## 👤 Author

**Nolan Cacheux**
- LinkedIn: [nolancacheux](https://linkedin.com/in/nolancacheux)
- GitHub: [nolancacheux](https://github.com/nolancacheux)
- Email: cachnolan@gmail.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Built as part of M2 MLOps course - JUNIA 2026*
