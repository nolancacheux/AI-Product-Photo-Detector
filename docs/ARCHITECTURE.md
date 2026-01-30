# Architecture Documentation

## System Overview

The AI Product Photo Detector is a production-grade MLOps system for detecting AI-generated product images in e-commerce listings.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Client Layer                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Web UI (Streamlit)  │  REST API Clients  │  Batch Processing Jobs          │
└──────────┬───────────────────┬────────────────────────┬────────────────────┘
           │                   │                        │
           ▼                   ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            API Gateway Layer                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Rate Limiting (slowapi)                                                   │
│  • Authentication (API Key / JWT)                                           │
│  • Request Validation                                                        │
│  • Response Caching (Redis/Memory)                                          │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Application Layer                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  FastAPI Application                                                         │
│  ├── /predict      - Single image classification                            │
│  ├── /predict/batch - Batch processing (up to 20 images)                    │
│  ├── /explain      - GradCAM visualization                                  │
│  ├── /health       - Health check                                           │
│  └── /metrics      - Prometheus metrics                                     │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ML Inference Layer                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  Predictor                                                                   │
│  ├── Model Loading (EfficientNet-B0)                                        │
│  ├── Image Preprocessing                                                     │
│  ├── Inference                                                               │
│  ├── Probability Calibration                                                │
│  └── Explainability (GradCAM)                                               │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Observability Layer                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Structured Logging (structlog)                                           │
│  • Metrics (Prometheus)                                                      │
│  • Distributed Tracing (Request IDs)                                        │
│  • Drift Detection                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. API Layer (`src/inference/`)

#### API Server (`api.py`)
- FastAPI application with lifespan management
- CORS middleware for cross-origin requests
- Rate limiting per endpoint
- Authentication middleware

#### Authentication (`auth.py`)
- API Key authentication (X-API-Key header)
- JWT Bearer token authentication
- Scope-based permissions (predict, batch, explain)
- Optional: disabled when no API_KEYS configured

#### Input Validation (`validation.py`)
- Magic bytes detection for MIME type verification
- Image dimension validation
- File size limits
- Path traversal prevention
- Content hash generation

#### Response Caching (`cache.py`)
- In-memory LRU cache with TTL
- Redis backend (optional)
- Content-hash based cache keys
- Model version aware caching

### 2. Inference Engine (`src/inference/`)

#### Predictor (`predictor.py`)
- Model loading from checkpoints
- Auto-device selection (CUDA/MPS/CPU)
- Configurable thresholds
- Confidence level calculation

#### Explainability (`explainability.py`)
- GradCAM heatmap generation
- Customizable overlay transparency
- Supports multiple colormap options

### 3. Training Pipeline (`src/training/`)

#### Data Augmentation (`augmentation.py`)
- CutMix: Spatial region mixing
- MixUp: Linear interpolation
- RandAugment: Automated augmentation
- GridMask: Grid-based dropout

#### Hyperparameter Optimization (`hyperopt.py`)
- Optuna integration
- TPE sampler with median pruner
- MLflow experiment tracking
- Auto-config generation

#### Model Calibration (`calibration.py`)
- Temperature scaling
- Platt scaling
- Isotonic regression
- ECE/MCE metrics

### 4. Monitoring (`src/monitoring/`)

#### Metrics (`metrics.py`)
- Prediction counters by class/confidence
- Latency histograms
- Batch size tracking
- Cache hit rates
- Error counting

#### Drift Detection (`drift.py`)
- Input distribution monitoring
- Prediction distribution tracking
- Alert thresholds

### 5. User Interface (`src/ui/`)

#### Streamlit App (`app.py`)
- Single image analysis
- Batch upload processing
- GradCAM visualization
- Prediction history
- Export functionality

## Data Flow

### Single Prediction Flow
```
1. Client uploads image
2. Rate limiter checks request count
3. Authenticator validates credentials
4. Validator checks image format/size
5. Cache checks for existing prediction
6. If cache miss:
   a. Predictor preprocesses image
   b. Model runs inference
   c. Calibrator adjusts probability
   d. Result cached
7. Response returned with metrics
```

### Batch Prediction Flow
```
1. Client uploads multiple images
2. Rate limiter (stricter limits)
3. Authenticator validates
4. For each image:
   a. Validate independently
   b. Check cache
   c. Run prediction if needed
   d. Collect result
5. Aggregate results returned
```

## Deployment Architecture

### Kubernetes Deployment
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Kubernetes Cluster                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                       │
│  │   API Pod    │  │   API Pod    │  │   API Pod    │  (HPA: 2-10)         │
│  │ Port: 8000   │  │ Port: 8000   │  │ Port: 8000   │                       │
│  └──────────────┘  └──────────────┘  └──────────────┘                       │
│         │                 │                 │                                │
│         └─────────────────┼─────────────────┘                                │
│                           │                                                  │
│                    ┌──────┴──────┐                                          │
│                    │   Service   │                                          │
│                    │  ClusterIP  │                                          │
│                    └──────┬──────┘                                          │
│                           │                                                  │
│  ┌──────────────┐  ┌──────┴──────┐  ┌──────────────┐                       │
│  │   UI Pod     │  │   Ingress   │  │    Redis     │  (Optional)           │
│  │ Port: 8501   │  │   nginx     │  │  Cache Pod   │                       │
│  └──────────────┘  └─────────────┘  └──────────────┘                       │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                              PVC                                      │   │
│  │                         Model Storage                                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### CI/CD Pipeline
```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  Push   │───▶│  Lint   │───▶│  Test   │───▶│  Build  │───▶│  Push   │
│  Code   │    │  Ruff   │    │ Pytest  │    │ Docker  │    │  GHCR   │
└─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
                                                                 │
                                                                 ▼
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Notify  │◀───│ Verify  │◀───│ Deploy  │◀───│  Helm   │◀───│ Staging │
│ Slack   │    │ Health  │    │  Prod   │    │ Upgrade │    │  First  │
└─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
```

## Security Considerations

1. **Input Validation**: All uploads verified via magic bytes
2. **Authentication**: Optional API key/JWT for production
3. **Rate Limiting**: Per-IP limits prevent abuse
4. **CORS**: Configurable allowed origins
5. **Secrets**: Environment variables, never in code
6. **Container Security**: Non-root user, minimal base image

## Performance Optimizations

1. **Caching**: SHA256-based content deduplication
2. **Batch Processing**: Up to 20 images per request
3. **Model Loading**: Singleton pattern, loaded once
4. **Async I/O**: FastAPI async endpoints
5. **Connection Pooling**: Redis connection pool
6. **HPA**: Auto-scaling based on CPU/memory

## Future Improvements

- [ ] GPU inference support (NVIDIA Triton)
- [ ] A/B testing framework
- [ ] Online learning pipeline
- [ ] Feature store integration
- [ ] Model ensemble serving
