# Architecture Documentation

## System Overview

The AI Product Photo Detector is a production-grade MLOps system for detecting AI-generated product images in e-commerce contexts.

## Components

### 1. Training Pipeline (`src/training/`)

**Purpose**: Train and evaluate the detection model.

**Components**:
- `model.py`: EfficientNet-B0 based binary classifier
- `dataset.py`: PyTorch Dataset and DataLoader utilities
- `train.py`: Main training script with MLflow integration

**Flow**:
```
Raw Images → Preprocessing → Augmentation → Model Training → MLflow Logging → Checkpoint
```

### 2. Inference API (`src/inference/`)

**Purpose**: Serve predictions via REST API.

**Components**:
- `api.py`: FastAPI application with endpoints
- `predictor.py`: Model loading and prediction logic
- `schemas.py`: Pydantic request/response models

**Endpoints**:
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Classify image |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

### 3. Web UI (`src/ui/`)

**Purpose**: User-friendly interface for testing.

**Components**:
- `app.py`: Streamlit application

### 4. Utilities (`src/utils/`)

**Purpose**: Shared functionality.

**Components**:
- `config.py`: Configuration management
- `logger.py`: Structured logging setup

## Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│   FastAPI   │────▶│  Predictor  │
│  (Upload)   │     │   (API)     │     │  (Model)    │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  Response   │
                    │  (JSON)     │
                    └─────────────┘
```

## Model Architecture

```
Input (224x224x3)
       │
       ▼
┌──────────────────┐
│  EfficientNet-B0 │ (Pretrained ImageNet)
│  Feature Extract │
└────────┬─────────┘
         │ (1280 features)
         ▼
┌──────────────────┐
│  Linear(1280→512)│
│  ReLU + Dropout  │
│  Linear(512→1)   │
│  Sigmoid         │
└────────┬─────────┘
         │
         ▼
   Probability [0,1]
```

## Deployment Architecture

### Docker Compose Stack

```
┌─────────────────────────────────────────────────────┐
│                   Docker Network                     │
├─────────────┬─────────────┬─────────────────────────┤
│    API      │     UI      │        MLflow           │
│   :8000     │   :8501     │        :5000            │
├─────────────┴─────────────┴─────────────────────────┤
│              Prometheus :9090                        │
├─────────────────────────────────────────────────────┤
│              Grafana :3000                           │
└─────────────────────────────────────────────────────┘
```

## Security Considerations

1. **Input Validation**: All inputs validated via Pydantic
2. **File Size Limits**: Max 10MB per image
3. **Content Type Checks**: Only JPEG, PNG, WebP accepted
4. **Non-root Containers**: Docker images run as `appuser`
5. **No Secrets in Images**: Configuration via environment variables

## Observability

### Metrics (Prometheus)
- `predictions_total`: Counter by status and result
- `prediction_latency_seconds`: Histogram of inference times

### Logging (structlog)
- JSON format for production
- Structured fields: timestamp, level, message, context

### Dashboards (Grafana)
- Request rate and latency
- Error rates
- Model performance trends

## Scalability

- **Horizontal**: Stateless API allows multiple replicas
- **Vertical**: GPU support for faster inference
- **Caching**: Consider Redis for repeated predictions

## Future Improvements

1. Model versioning with A/B testing
2. Automated retraining pipeline
3. Drift detection dashboard
4. Multi-region deployment
