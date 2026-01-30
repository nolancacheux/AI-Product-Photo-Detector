# 🔍 AI Product Photo Detector

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

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      GitHub Repository                       │
├─────────────────────────────────────────────────────────────┤
│  src/                                                        │
│  ├── training/      → Training pipeline                     │
│  ├── inference/     → FastAPI inference server              │
│  └── ui/            → Streamlit web interface               │
│  docker/                                                     │
│  ├── train.Dockerfile                                       │
│  └── serve.Dockerfile                                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    CI/CD (GitHub Actions)                    │
│  • Lint & Test → Build Images → Push to Registry            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Production Server                         │
├───────────────┬───────────────┬─────────────────────────────┤
│   MLflow      │   FastAPI     │   Streamlit                 │
│   :5000       │   :8000       │   :8501                     │
│               │   /predict    │   Web UI                    │
│               │   /health     │                             │
├───────────────┴───────────────┴─────────────────────────────┤
│              Prometheus + Grafana (Observability)            │
└─────────────────────────────────────────────────────────────┘
```

## ✨ Features

- **Binary Classification**: Real vs AI-generated product images
- **Probability Score**: Confidence score (0.0 - 1.0)
- **Multi-Generator Detection**: Trained on Stable Diffusion & Flux outputs
- **REST API**: Production-ready FastAPI with `/predict` and `/health`
- **Web UI**: Streamlit interface for easy testing
- **Drift Detection**: Monitors for distribution shift in incoming data
- **Observability**: Structured logging, Prometheus metrics, Grafana dashboards

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| **ML Framework** | PyTorch + timm (EfficientNet-B0) |
| **API** | FastAPI + Uvicorn |
| **Web UI** | Streamlit |
| **MLOps** | MLflow + DVC |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |
| **Observability** | Prometheus + Grafana + structlog |

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- (Optional) CUDA-capable GPU

### Installation

```bash
# Clone the repository
git clone https://github.com/nolancacheux/mlops_project.git
cd mlops_project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies (with dev tools)
pip install -e ".[dev,ui]"

# Install pre-commit hooks
pre-commit install
```

### Create Sample Dataset

```bash
# Generate synthetic sample data for testing
python scripts/create_sample_data.py --output data/processed

# Validate dataset
python -m src.data.prepare validate --data-dir data/processed
```

### Training

```bash
# Train model (uses configs/train_config.yaml)
make train

# Or manually:
python -m src.training.train --config configs/train_config.yaml

# Model is automatically logged to MLflow
```

### Inference (Local)

```bash
# Start API server
make serve

# Or manually:
uvicorn src.inference.api:app --host 0.0.0.0 --port 8000 --reload

# Test prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

### Web UI

```bash
# Start Streamlit interface
make ui

# Or manually:
streamlit run src/ui/app.py --server.port 8501
```

### Docker (Full Stack)

```bash
# Build all images
make docker-build

# Start all services (API, UI, MLflow, Prometheus, Grafana)
make docker-up

# Stop all services
make docker-down
```

### Available Makefile Commands

```bash
make help      # Show all available commands
make install   # Install production dependencies
make dev       # Install dev dependencies + pre-commit
make lint      # Run linting (ruff + mypy)
make format    # Format code
make test      # Run tests with coverage
```

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | TBD |
| Precision | TBD |
| Recall | TBD |
| F1-Score | TBD |
| Inference Latency | TBD |

## 📁 Project Structure

```
mlops_project/
├── .github/
│   └── workflows/          # CI/CD pipelines
├── configs/                # Training & deployment configs
├── data/
│   ├── raw/               # Original images (DVC tracked)
│   └── processed/         # Preprocessed data
├── docker/
│   ├── train.Dockerfile   # Training container
│   └── serve.Dockerfile   # Inference container
├── docs/
│   └── PRD.md            # Product Requirements Document
├── notebooks/             # Exploration notebooks
├── src/
│   ├── training/         # Training pipeline
│   ├── inference/        # FastAPI server
│   ├── ui/               # Streamlit app
│   └── utils/            # Shared utilities
├── tests/                 # Unit & integration tests
├── .gitignore
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 🔄 MLOps Pipeline

1. **Data Validation**: Schema checks, distribution analysis
2. **Training**: EfficientNet-B0 fine-tuning with MLflow tracking
3. **Evaluation**: Metrics computation, threshold optimization
4. **Model Registry**: Version control with MLflow Model Registry
5. **Deployment**: Docker packaging, API deployment
6. **Monitoring**: Drift detection, performance dashboards

## 🚨 Incident Scenario

This project includes a simulated incident scenario:

> **Scenario**: A new AI generator (e.g., Flux 2.0) produces images that bypass the detector, causing accuracy to drop from 95% to 70%.

**Response**:
1. Drift detection alerts trigger
2. Root cause analysis identifies new generator
3. Dataset augmented with new samples
4. Model retrained and redeployed
5. Post-mortem documented

## 📖 Documentation

- [Product Requirements Document (PRD)](docs/PRD.md)
- [API Documentation](http://localhost:8000/docs) (when running)
- [MLflow Dashboard](http://localhost:5000) (when running)

## 👤 Author

**Nolan Cacheux**
- LinkedIn: [nolancacheux](https://linkedin.com/in/nolancacheux)
- GitHub: [nolancacheux](https://github.com/nolancacheux)
- Email: cachnolan@gmail.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Built as part of M2 MLOps course - JUNIA 2026*
