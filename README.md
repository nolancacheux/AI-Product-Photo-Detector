# 🔍 AI Product Photo Detector

[![CI](https://github.com/nolancacheux/AI-Product-Photo-Detector/actions/workflows/ci.yml/badge.svg)](https://github.com/nolancacheux/AI-Product-Photo-Detector/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Detect AI-generated product photos in e-commerce listings.**

A MLOps project that classifies product images as **real** or **AI-generated**, helping e-commerce platforms fight fraudulent listings.

## 🎯 Problem

E-commerce platforms face a growing threat: **AI-generated fake product images**. Scammers use tools like Stable Diffusion to create convincing product photos for items that don't exist.

This project provides an **API to detect these fake images**.

## ✨ Features

- **Binary Classification**: Real vs AI-generated product images
- **REST API**: FastAPI with `/predict` and `/predict/batch` endpoints
- **Web UI**: Streamlit interface for easy testing
- **MLflow Tracking**: Experiment tracking and model versioning
- **Docker**: Ready for deployment
- **CI/CD**: GitHub Actions for automated testing

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Clients                              │
│         Web UI (Streamlit)  │  REST API                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                       │
│  Rate Limiting → Validation → Inference                     │
├─────────────────────────────────────────────────────────────┤
│  Endpoints:                                                  │
│  • POST /predict         - Single image                     │
│  • POST /predict/batch   - Batch (up to 20)                 │
│  • GET  /health          - Health check                     │
│  • GET  /metrics         - Prometheus metrics               │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| **ML Framework** | PyTorch + timm (EfficientNet-B0) |
| **API** | FastAPI + Uvicorn |
| **Web UI** | Streamlit |
| **MLOps** | MLflow + DVC |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |

## 🚀 Quick Start

### Installation

```bash
# Clone
git clone https://github.com/nolancacheux/mlops_project.git
cd mlops_project

# Create venv
python -m venv .venv
source .venv/bin/activate

# Install
pip install -e ".[dev,ui]"
```

### Training

```bash
# Create sample dataset
python scripts/create_sample_data.py --output data/processed

# Train model
python -m src.training.train --config configs/train_config.yaml
```

### Run API

```bash
# Start server
uvicorn src.inference.api:app --host 0.0.0.0 --port 8000

# Test
curl -X POST "http://localhost:8000/predict" -F "file=@image.jpg"
```

### Run UI

```bash
streamlit run src/ui/app.py
```

### Docker

```bash
docker-compose up -d
# API: http://localhost:8000
# UI: http://localhost:8501
```

## 📊 API Reference

### POST /predict

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

Process multiple images (max 20):
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -F "files=@img1.jpg" -F "files=@img2.jpg"
```

## 📁 Project Structure

```
mlops_project/
├── src/
│   ├── data/             # Data processing
│   ├── inference/        # API & prediction
│   ├── monitoring/       # Metrics & drift
│   ├── training/         # Training pipeline
│   ├── ui/               # Streamlit app
│   └── utils/            # Shared utilities
├── tests/                # Unit tests
├── docker/               # Dockerfiles
├── configs/              # Configuration files
└── docs/                 # Documentation
```

## 📦 Data Versioning (DVC)

This project uses [DVC](https://dvc.org/) for data versioning. The local remote is configured to store data artifacts in `./dvc-storage` within the project directory.

```bash
# Initialize DVC (already done)
dvc init

# Track a data file
dvc add data/processed/dataset.csv

# Push data to local storage
dvc push

# Pull data from storage
dvc pull
```

To switch to a cloud remote (S3, GCS, Azure), update `.dvc/config`:
```bash
dvc remote modify local url s3://your-bucket/dvc-storage
```

## 📋 Documentation

- [Incident Scenario](docs/INCIDENT_SCENARIO.md) — Drift detection incident response scenario documenting a realistic data drift event, root cause analysis, remediation steps, and prevention measures.

## 🧪 Testing

```bash
pytest tests/ -v --cov=src
```

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | ~83% |
| Inference | ~50ms |

## 👤 Author

**Nolan Cacheux**
- GitHub: [nolancacheux](https://github.com/nolancacheux)
- LinkedIn: [nolancacheux](https://linkedin.com/in/nolancacheux)

## 📄 License

MIT License - see [LICENSE](LICENSE)

---
*M2 MLOps - JUNIA 2026*
