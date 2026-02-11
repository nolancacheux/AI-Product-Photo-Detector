---
marp: true
theme: default
paginate: true
backgroundColor: #ffffff
style: |
  section {
    font-family: 'Segoe UI', sans-serif;
  }
  h1 {
    color: #1a1a2e;
  }
  h2 {
    color: #16213e;
  }
  code {
    background: #f0f0f0;
    padding: 2px 6px;
    border-radius: 4px;
  }
  table {
    font-size: 0.8em;
  }
  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }
  pre {
    font-size: 0.7em;
  }
  footer {
    font-size: 0.6em;
    color: #888;
  }
---

<!-- _paginate: false -->
<!-- _backgroundColor: #1a1a2e -->
<!-- _color: #ffffff -->

# AI Product Photo Detector

### Detecting AI-Generated Images in E-Commerce

**Nolan Cacheux**
Master 2 Data Science – MLOps
2025

---

## The Problem: Trust in E-Commerce

Online marketplaces are flooded with AI-generated product images that mislead consumers and erode trust.

**Business impact:**
- Consumers receive products that look nothing like the listing photo
- Marketplace platforms face increased return rates and complaints
- Legitimate sellers lose visibility to AI-enhanced fake listings
- Regulatory pressure is growing (EU AI Act, FTC guidelines)

**Our goal:** Build a production-grade ML system that classifies product images as **real** or **AI-generated**, with full MLOps lifecycle management.

<!-- Speaker notes: Start with why this matters. Mention concrete examples: Amazon, AliExpress listings with AI photos that mislead buyers. This is not just an academic exercise. -->

---

## Dataset: CIFAKE

| Split | Real | AI-Generated | Total |
|-------|------|--------------|-------|
| Train | 2,500 | 2,500 | 5,000 |
| Val | 250 | 250 | 500 |
| Test | 250 | 250 | 500 |

**Data pipeline with DVC:**

```yaml
stages:
  download:
    cmd: python scripts/download_cifake.py
    outs:
      - data/processed
  validate:
    cmd: python -m src.data.validate --data-dir data/processed
    deps: [src/data/validate.py, data/processed]
    outs: [reports/data_validation.json]
```

Data is versioned with **DVC** and stored on **Google Cloud Storage**, ensuring full reproducibility across local and cloud environments.

<!-- Speaker notes: Explain that CIFAKE is a well-known benchmark. DVC tracks data versions alongside git commits. -->

---

## Data Validation Pipeline

Automated integrity checks run before every training cycle:

```python
def validate_dataset(data_dir: str) -> dict:
    """Validates structure, integrity, and statistics."""
    # 1. Check split directories exist (train/val/test)
    # 2. Verify class subdirectories (real/ and ai/)
    # 3. Open every image with PIL to detect corruption
    # 4. Compute class balance ratio
    # 5. Flag warnings for unexpected formats
```

**Checks performed:**
- Directory structure validation (expected splits and classes)
- Image file integrity (PIL open test on every file)
- Class balance verification (max imbalance ratio threshold)
- Supported format enforcement (JPEG, PNG, WebP)
- Corruption detection with detailed error reporting

Output: `reports/data_validation.json` with full audit trail.

<!-- Speaker notes: This runs as a DVC stage and as a Kubeflow pipeline component. Any corruption or imbalance blocks training. -->

---

## Model Architecture

**EfficientNet-B0** via `timm` with custom binary classification head:

```python
class AIImageDetector(nn.Module):
    def __init__(self, model_name="efficientnet_b0",
                 pretrained=True, dropout=0.3):
        super().__init__()
        # Pretrained backbone (ImageNet weights)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1),    # Raw logits -> BCEWithLogitsLoss
        )
```

| Parameter | Value |
|-----------|-------|
| Backbone | EfficientNet-B0 (5.3M params) |
| Classifier | 512-unit hidden layer + BatchNorm + Dropout |
| Loss | BCEWithLogitsLoss |
| Total params | ~5.6M |

<!-- Speaker notes: Why EfficientNet? Best accuracy/efficiency tradeoff. Transfer learning from ImageNet gives us strong feature extraction out of the box. -->

---

## Training Configuration

```yaml
seed: 42
data:
  image_size: 224
  batch_size: 64
  num_workers: 4
training:
  epochs: 15
  learning_rate: 0.001
  weight_decay: 0.0001
  scheduler: cosine
  warmup_epochs: 2
  early_stopping_patience: 5
augmentation:
  horizontal_flip: true
  rotation_degrees: 15
  color_jitter: {brightness: 0.2, contrast: 0.2}
```

**Experiment tracking with MLflow:**
- All hyperparameters logged automatically
- Metrics tracked per epoch (loss, accuracy, F1)
- Best model checkpoint saved based on `val_accuracy`
- Local tracking URI: `mlruns/` directory

<!-- Speaker notes: Cosine scheduler with warmup helps convergence. Early stopping prevents overfitting on this relatively small dataset. -->

---

## Training Pipeline: Local and Cloud

```
LOCAL (Development)                    CLOUD (Production)
===================                    ==================

  DVC Pipeline                         GitHub Actions Trigger
       |                                      |
   download_data                         upload-data (GCS)
       |                                      |
   validate_data                       build-training-image
       |                                      |
   train (MLflow)                      submit-training (Vertex AI)
       |                                  T4 GPU | n1-standard-4
   best_model.pt                              |
                                         evaluate (quality gate)
                                           acc >= 0.85, F1 >= 0.80
                                              |
                                         deploy (Cloud Run)
```

**Vertex AI training job:**
- Custom container with `docker/Dockerfile.training`
- T4 GPU on `n1-standard-4` machine
- Data synced from GCS, model uploaded back to GCS
- Fully automated via `model-training.yml` workflow

<!-- Speaker notes: Two paths to training. Local for experimentation with DVC+MLflow. Cloud for production training with Vertex AI. Both produce the same model format. -->

---

## API Design: FastAPI

Four prediction endpoints, each serving a different use case:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/predict` | POST | Single image classification |
| `/predict/batch` | POST | Up to 20 images in one request |
| `/predict/explain` | POST | Prediction + Grad-CAM heatmap |
| `/predict/compare` | POST | Shadow A/B model comparison |

**Infrastructure endpoints:**

| Endpoint | Purpose |
|----------|---------|
| `/health` | Readiness check (model status, drift, uptime) |
| `/healthz` | Liveness probe (Cloud Run) |
| `/metrics` | Prometheus metrics export |
| `/docs` | Interactive Swagger/OpenAPI documentation |

All prediction endpoints require `X-API-Key` header when `REQUIRE_AUTH=true`.

<!-- Speaker notes: The API is the backbone of the system. Every endpoint has Pydantic validation, structured logging, and Prometheus instrumentation. -->

---

## API: Authentication and Rate Limiting

**API Key Authentication:**

```python
class APIKeyManager:
    def validate_key(self, key: str) -> bool:
        """Constant-time comparison to prevent timing attacks."""
        key_hash = self._hash_key(key)
        found = False
        for stored in self._keys:
            if hmac.compare_digest(key_hash, stored):
                found = True
        return found
```

- Keys stored as SHA-256 hashes (never in plaintext)
- Constant-time comparison prevents timing side-channels
- `REQUIRE_AUTH` env var enforces auth even with no keys configured

**Rate Limiting** via `slowapi`:
- Per-endpoint rate limits to prevent abuse
- Returns `429 Too Many Requests` with retry-after header
- Rate limit events tracked in Prometheus metrics

<!-- Speaker notes: Security is not an afterthought. Constant-time comparison is a real-world best practice against timing attacks. -->

---

## Explainability: Grad-CAM

**Why explainability matters:** A prediction without explanation is a black box. Grad-CAM shows *where* the model is looking.

```python
class GradCAMExplainer:
    def __init__(self, model_path, device="cpu"):
        # Target: last conv block before global pooling
        target_layer = self.model.backbone.bn2
        self.cam = GradCAM(
            model=self.model,
            target_layers=[target_layer]
        )

    def explain(self, image_bytes):
        grayscale_cam = self.cam(input_tensor=input_tensor)
        visualization = show_cam_on_image(
            img_array, grayscale_cam[0, :], use_rgb=True
        )
        return {
            "prediction": prediction,
            "probability": probability,
            "heatmap_base64": base64_encode(visualization),
        }
```

The heatmap overlay is returned as base64-encoded JPEG, ready for display in the Streamlit UI or any client.

<!-- Speaker notes: Grad-CAM targets the last convolutional layer of EfficientNet's backbone. The heatmap highlights which regions influenced the decision most. -->

---

## Shadow A/B Testing

The `/predict/compare` endpoint enables safe model comparison in production:

```
                  +---> Primary Model ---> PredictResponse
                  |                                  |
  Image Upload ---+                           CompareResponse
                  |                                  |
                  +---> Shadow Model  ---> PredictResponse
```

```python
class CompareResponse(BaseModel):
    primary: PredictResponse    # Current production model
    shadow: PredictResponse     # Candidate model
    agreement: bool             # Do they agree on the label?
    difference: float           # |prob_primary - prob_shadow|
```

**Use case:** Deploy a new model as shadow, route real traffic through both, compare results *without* affecting users. If the shadow model outperforms, promote it.

Configured via `SHADOW_MODEL_PATH` environment variable. Lazy-loaded on first request.

<!-- Speaker notes: This is how you do safe model updates in production. The shadow model is invisible to the end user but gives you real production data to evaluate against. -->

---

## Streamlit Web Interface

A drag-and-drop interface for non-technical users:

**Features:**
- Image upload with automatic compression (max 4.5MB for API limits)
- Large percentage display with color-coded result cards
- Confidence bar visualization
- Grad-CAM heatmap display when available
- Prediction history in session state

**Architecture:**
```
User Browser --> Streamlit (port 8501)
                     |
                     | HTTP (httpx)
                     v
              FastAPI API (port 8080)
                     |
                     v
              EfficientNet Model
```

The UI is deployed as a separate Cloud Run service, communicating with the API via internal networking.

<!-- Speaker notes: The UI makes the project accessible to anyone. The auto-compression handles large phone photos gracefully. -->

---

## CI/CD Pipeline: GitHub Actions

Three workflows orchestrate the full lifecycle:

```
+-------+     +--------+     +---------------------+
|  CI   |     |   CD   |     | Model Training      |
+-------+     +--------+     +---------------------+
| Lint  |     | Wait   |     | Upload data (GCS)   |
| ruff  |---->| for CI |     | Build training image|
|       |     |   |    |     | Vertex AI job (T4)  |
| Type  |     | Build  |     | Evaluate model      |
| mypy  |     | Docker |     | Quality gate check  |
|       |     |   |    |     | Conditional deploy  |
| Tests |     | Push   |     +---------------------+
| 3.11  |     | AR     |
| 3.12  |     |   |    |
|       |     | Deploy |
| Sec.  |     | CR     |
| Audit |     +--------+
+-------+
```

**Quality gates:**
- Lint + type-check must pass before CD runs
- Tests run on Python 3.11 and 3.12 (matrix strategy)
- Model must achieve accuracy >= 0.85 and F1 >= 0.80 to deploy
- Security scan: `pip-audit` + `bandit`

<!-- Speaker notes: CI runs on every push/PR. CD only on main after CI passes. Model training is triggered manually or on data changes. -->

---

## Cloud Architecture

```
GitHub Actions
     |
     +---> Artifact Registry (Docker images)
     |         |
     |         +---> Cloud Run: API (8080)
     |         |         - EfficientNet model
     |         |         - FastAPI + Prometheus metrics
     |         |
     |         +---> Cloud Run: UI (8501)
     |                   - Streamlit interface
     |
     +---> GCS Bucket
     |         - Training data (DVC remote)
     |         - Model checkpoints
     |         - Pipeline artifacts
     |
     +---> Vertex AI
              - Custom training jobs
              - T4 GPU acceleration
```

| Service | Purpose | Config |
|---------|---------|--------|
| Cloud Run (API) | Model serving | 1Gi RAM, auto-scale |
| Cloud Run (UI) | Web interface | 512Mi RAM |
| Artifact Registry | Docker images | europe-west1 |
| GCS | Data + models | Multi-region |
| Vertex AI | GPU training | n1-standard-4 + T4 |

<!-- Speaker notes: Everything runs in europe-west1 for GDPR compliance and low latency. Cloud Run auto-scales to zero when idle, keeping costs minimal. -->

---

## Monitoring and Observability

**Prometheus** – 20+ custom metrics across 5 categories:

| Category | Metrics |
|----------|---------|
| Predictions | Total count, latency histogram, probability distribution |
| Batch | Batch size, batch latency, success/failure rate |
| Images | Upload size, dimensions, validation errors |
| HTTP | Request count by endpoint/status, request duration |
| System | Active requests, concurrent max, rate limit events |

**Grafana** – 16 dashboard panels for real-time visualization

**Structured Logging** via `structlog`:
- JSON-formatted logs with correlation IDs
- Every prediction logged with probability, confidence, latency

**Drift Detection:**
- Sliding window of 1000 predictions
- Tracks mean probability shift, confidence distribution, class ratios
- Alerts when metrics deviate beyond configurable thresholds

<!-- Speaker notes: Monitoring is what separates a prototype from production. We track everything needed to detect problems before users report them. -->

---

## Drift Detection: Deep Dive

```python
class DriftDetector:
    def __init__(self, window_size=1000,
                 drift_threshold=0.15):
        self.predictions = deque(maxlen=window_size)

    def check_drift(self) -> DriftMetrics:
        # Compare current window to baseline:
        # 1. Mean probability shift
        # 2. Low-confidence ratio change
        # 3. Class prediction ratio drift
        # Alert if any metric exceeds threshold
```

**Three drift signals:**

| Signal | Baseline comparison | Threshold |
|--------|-------------------|-----------|
| Probability mean shift | Current vs. training distribution | 0.15 |
| Low confidence ratio | Predictions near 0.5 decision boundary | 0.30 |
| Class ratio imbalance | Real vs. AI-generated distribution | 0.20 |

Exposed via `/health` endpoint: `drift_detected: true/false` with `drift_score`.

<!-- Speaker notes: Drift detection is thread-safe with a sliding window. The baseline can be saved from initial deployment and compared against ongoing predictions. -->

---

## Testing Strategy

**183+ tests** across multiple levels:

| Level | Tests | Framework | What it covers |
|-------|-------|-----------|----------------|
| Unit | Model, predictor, auth, schemas, drift, config, logger | pytest | Core business logic |
| Integration | API endpoints, batch flow, explainer pipeline | pytest + httpx | End-to-end request handling |
| Load | Concurrent users, sustained throughput | Locust + k6 | Performance under stress |

**Test matrix:** Python 3.11 and 3.12 (GitHub Actions matrix strategy)

**Coverage tracking:**
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

**Load testing with Locust:**
```python
class PredictUser(HttpUser):
    wait_time = between(1, 3)
    @task
    def predict(self):
        self.client.post("/predict", files={"file": image})
```

<!-- Speaker notes: 183+ tests may sound like a lot, but each module has dedicated test coverage. The load tests validate Cloud Run auto-scaling behavior. -->

---

## Security

| Layer | Implementation |
|-------|---------------|
| Authentication | API key via `X-API-Key` header, SHA-256 hashed storage |
| Timing attack prevention | `hmac.compare_digest` with constant-time iteration |
| Rate limiting | `slowapi` per-endpoint limits, 429 responses |
| Input validation | File type check, size limits, PIL integrity test |
| CORS | Configurable origin whitelist |
| Secrets management | Environment variables, GitHub Secrets for CI/CD |
| Docker | Non-root user, minimal base image, `.dockerignore` |
| Dependency scanning | `pip-audit` in CI pipeline |
| Code scanning | `bandit` for Python security anti-patterns |

**Image validation pipeline:**
```
Upload -> Size check (<10MB) -> Extension check -> PIL.open()
  -> Format verify -> Resize to 224x224 -> Normalize -> Model
```

Every rejected image increments `aidetect_image_validation_errors_total` in Prometheus.

<!-- Speaker notes: Defense in depth. Every layer adds protection. The constant-time key comparison is a detail that matters in real deployments. -->

---

## Project Structure

```
ai-product-photo-detector/
|-- src/
|   |-- training/         # Model, dataset, augmentation, train loop
|   |   |-- model.py          AIImageDetector (EfficientNet-B0)
|   |   |-- train.py          Training loop with MLflow tracking
|   |   |-- dataset.py        DataLoader creation
|   |   |-- augmentation.py   Torchvision transforms
|   |   |-- vertex_submit.py  Vertex AI job submission
|   |-- inference/        # FastAPI application
|   |   |-- api.py             App factory + middleware
|   |   |-- predictor.py       Model loading + prediction
|   |   |-- explainer.py       Grad-CAM heatmap generation
|   |   |-- shadow.py          A/B comparison logic
|   |   |-- auth.py            API key management
|   |   |-- schemas.py         Pydantic request/response models
|   |-- monitoring/       # Prometheus + drift detection
|   |-- pipelines/        # Kubeflow pipeline definition
|   |-- ui/               # Streamlit web interface
|   |-- data/             # Dataset validation
|   |-- utils/            # Config, structured logging
|-- tests/                # 183+ tests (unit, integration, load)
|-- configs/              # Training config, Prometheus, Grafana
|-- docker/               # Dockerfile (API) + Dockerfile.training
|-- .github/workflows/    # CI, CD, Model Training
```

<!-- Speaker notes: Clean separation of concerns. Each module is independently testable. The project follows standard Python packaging with pyproject.toml. -->

---

## Results and Metrics

**Model Performance:**

| Metric | Value |
|--------|-------|
| Accuracy | > 0.85 (quality gate threshold) |
| F1 Score | > 0.80 (quality gate threshold) |
| Inference time (CPU) | ~45ms per image |
| Batch throughput | ~20 images/second |

**System Performance:**

| Metric | Value |
|--------|-------|
| Cold start (Cloud Run) | ~3-5 seconds |
| API latency (p50) | < 100ms |
| API latency (p99) | < 500ms |
| Docker image size | Optimized multi-stage build |
| Uptime | Cloud Run managed (99.95% SLA) |

**Live deployment:**
- API: `ai-product-detector-714127049161.europe-west1.run.app`
- UI: `ai-product-detector-ui-714127049161.europe-west1.run.app`

<!-- Speaker notes: The quality gate in CI/CD ensures no model below these thresholds reaches production. Latency is measured with Prometheus histograms. -->

---

## Challenges and Lessons Learned

**State dict compatibility:**
Old checkpoints lacked `BatchNorm1d` in the classifier. Solution: dynamic classifier rebuilding in the explainer that inspects checkpoint keys.

**Docker build complexity:**
Training image needs GPU libraries (CUDA), inference image needs to stay lean. Solution: separate Dockerfiles (`Dockerfile` for serving, `Dockerfile.training` for GPU training).

**Coverage gaps:**
Some modules were hard to test without a real model file. Solution: conftest fixtures that create minimal mock models, plus integration tests with real checkpoint loading.

**GCS model synchronization:**
CD pipeline must always use the latest trained model. Solution: the CD workflow compares local and GCS model sizes before building, downloading only when they differ.

**Thread safety in monitoring:**
Drift detector and concurrent request tracking need thread-safe access. Solution: `threading.Lock` around shared state in `DriftDetector` and `metrics.py`.

<!-- Speaker notes: These are real engineering challenges, not theoretical. Each one taught something about production ML systems. -->

---

## Future Improvements

**Model enhancements:**
- Train for more epochs with larger dataset (full CIFAKE or custom dataset)
- Experiment with EfficientNet-B3/B4 for higher accuracy
- Add multi-class detection (GAN vs. diffusion vs. real)

**Infrastructure:**
- Canary deployments on Cloud Run (gradual traffic shift)
- Model versioning registry with promotion workflow
- Automated retraining triggered by drift detection alerts
- Feature store for image embeddings

**Monitoring:**
- Alerting rules in Grafana (PagerDuty/Slack integration)
- A/B test analysis dashboard with statistical significance
- Cost tracking per prediction

**Product:**
- Browser extension for real-time marketplace image scanning
- Batch processing API for marketplace platform integration
- Confidence calibration with Platt scaling

<!-- Speaker notes: The architecture supports all of these extensions. The shadow A/B system is already in place for safe model rollouts. -->

---

<!-- _backgroundColor: #1a1a2e -->
<!-- _color: #ffffff -->

# Live Demo

### API (Swagger UI)
`https://ai-product-detector-714127049161.europe-west1.run.app/docs`

### Web Interface
`https://ai-product-detector-ui-714127049161.europe-west1.run.app`

### Test with curl:
```bash
curl -X POST \
  "https://ai-product-detector-714127049161.europe-west1.run.app/predict" \
  -H "X-API-Key: YOUR_KEY" \
  -F "file=@product_photo.jpg"
```

### GitHub Repository
`github.com/nolancacheux/AI-Product-Photo-Detector`

---

<!-- _backgroundColor: #1a1a2e -->
<!-- _color: #ffffff -->

# Thank You

### Questions?

**Nolan Cacheux**
Master 2 Data Science – MLOps

Key technologies: PyTorch, FastAPI, Streamlit, DVC, MLflow,
Vertex AI, Cloud Run, Prometheus, Grafana, GitHub Actions
