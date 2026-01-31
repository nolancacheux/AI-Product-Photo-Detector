# Architecture Documentation

## System Overview

The AI Product Photo Detector is a MLOps system for detecting AI-generated product images.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Client Layer                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│         Web UI (Streamlit)       │       REST API Clients                   │
└──────────┬───────────────────────────────────┬──────────────────────────────┘
           │                                   │
           ▼                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FastAPI Application                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Rate Limiting (slowapi)                                                   │
│  • Optional API Key Auth                                                     │
│  • Input Validation                                                          │
│  • Response Caching (in-memory)                                             │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ML Inference Engine                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  Predictor                                                                   │
│  ├── Model Loading (EfficientNet-B0 from checkpoint)                        │
│  ├── Image Preprocessing (resize, normalize)                                │
│  └── Binary Classification (real vs AI-generated)                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. API Layer (`src/inference/`)

| File | Purpose |
|------|---------|
| `api.py` | FastAPI endpoints (/predict, /predict/batch, /health, /metrics) |
| `predictor.py` | Model loading and inference |
| `schemas.py` | Pydantic models for request/response |
| `auth.py` | Optional API key authentication |
| `validation.py` | Input validation (file type, size) |
| `cache.py` | In-memory response caching |

### 2. Training Pipeline (`src/training/`)

| File | Purpose |
|------|---------|
| `train.py` | Training loop with MLflow tracking |
| `model.py` | EfficientNet-B0 architecture |
| `dataset.py` | PyTorch dataset for images |
| `augmentation.py` | Data augmentation transforms |

### 3. Monitoring (`src/monitoring/`)

| File | Purpose |
|------|---------|
| `metrics.py` | Prometheus metrics |
| `drift.py` | Input distribution monitoring |

### 4. User Interface (`src/ui/`)

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web interface |

## Data Flow

### Prediction Flow

```
1. Client uploads image
2. Rate limiter checks request count
3. (Optional) API key validated
4. Image validated (format, size)
5. Check cache for existing prediction
6. If cache miss:
   a. Preprocess image (resize to 224x224, normalize)
   b. Run EfficientNet-B0 inference
   c. Cache result
7. Return prediction with confidence
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Classify single image |
| `/predict/batch` | POST | Classify up to 20 images |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_KEYS` | Comma-separated API keys | (auth disabled) |
| `LOG_LEVEL` | Logging level | INFO |

### Rate Limits

| Endpoint | Limit |
|----------|-------|
| `/predict` | 60/minute |
| `/predict/batch` | 10/minute |

## Deployment

### Docker

```bash
docker-compose up -d
```

Services:
- API: port 8000
- UI: port 8501
- MLflow: port 5000

### Manual

```bash
# API
uvicorn src.inference.api:app --host 0.0.0.0 --port 8000

# UI
streamlit run src/ui/app.py --server.port 8501
```
