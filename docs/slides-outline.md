# AI Product Photo Detector – Oral Presentation Cheat Sheet

**Nolan Cacheux | Master 2 Data Science – MLOps | 2025**

---

## Slide-by-Slide Key Points

### 1. Title
- AI Product Photo Detector, binary classification (real vs AI-generated)

### 2. Problem Statement
- AI-generated product images mislead consumers in e-commerce
- Increased returns, trust erosion, regulatory pressure (EU AI Act)
- Goal: production-grade ML detection system with full MLOps lifecycle

### 3. Dataset (CIFAKE + HuggingFace)
- Two sources: CIFAKE (local/DVC, 5k train) and HuggingFace `date3k2/raw_real_fake_images` (Colab, up to 8k)
- DVC for versioning, GCS as remote storage (local/cloud paths)
- Colab path loads directly from HuggingFace Hub, zero-setup
- Pipeline: download -> validate -> train

### 4. Data Validation
- Automated checks: structure, integrity (PIL open), class balance, format
- Runs as a DVC stage before every training cycle
- Output: JSON report, blocks training on failure

### 5. Model Architecture
- EfficientNet-B0 via timm, pretrained ImageNet
- Custom head: Linear(1280->512) + BatchNorm + ReLU + Dropout(0.3) + Linear(512->1)
- BCEWithLogitsLoss, ~5.6M parameters

### 6. Training Config
- 15 epochs, lr=0.001, cosine scheduler, 2 warmup epochs, early stopping (patience 5)
- Augmentation: flip, rotation 15deg, color jitter
- MLflow tracking (all params, metrics, checkpoints)

### 7. Three Training Paths (NEW)
- Local (DVC): CPU/local GPU, CIFAKE data, MLflow tracking, `dvc repro`
- Google Colab: Free T4 GPU, HuggingFace data, self-contained notebook, zero setup
- Vertex AI: Production, T4 on n1-standard-4, GCS data, triggered by GitHub Actions
- Same model architecture across all three paths
- Colab notebook: `notebooks/train_colab.ipynb`

### 8. Training Pipeline (Local + Cloud)
- Local: DVC pipeline (download -> validate -> train)
- Cloud: GitHub Actions -> upload GCS -> build training image -> Vertex AI (T4 GPU) -> evaluate -> deploy
- Quality gate: accuracy >= 0.85, F1 >= 0.80

### 9. API Design
- 4 predict endpoints: /predict, /predict/batch, /predict/explain, /predict/compare
- Infrastructure: /health, /healthz, /metrics, /docs
- Pydantic schemas, structured responses

### 10. Auth + Rate Limiting
- X-API-Key header, SHA-256 hashed, constant-time comparison (hmac.compare_digest)
- REQUIRE_AUTH env var, slowapi rate limiting, 429 responses

### 11. Explainability (Grad-CAM)
- Targets last conv block (backbone.bn2) of EfficientNet
- Returns base64 JPEG heatmap overlay
- Shows WHERE the model is looking, builds trust

### 12. Shadow A/B Testing
- /predict/compare runs primary + shadow model on same image
- Returns agreement (bool) and probability difference
- SHADOW_MODEL_PATH env var, lazy-loaded
- Safe model updates without affecting users

### 13. Streamlit UI
- Drag & drop upload, auto-compression (4.5MB limit)
- Color-coded result cards, large percentage display
- Grad-CAM heatmap integration, prediction history
- Separate Cloud Run service, calls API via HTTP

### 14. CI/CD Pipeline
- CI: ruff lint + mypy type-check + pytest (3.11/3.12 matrix) + pip-audit + bandit
- CD: wait for CI -> build Docker -> push Artifact Registry -> deploy Cloud Run
- Model Training: upload data -> build training image -> Vertex AI -> evaluate -> conditional deploy

### 15. Infrastructure as Code (NEW)
- Terraform provisions full GCP: GCS bucket, Artifact Registry, Cloud Run, Service Account, Budget Alerts
- 6 GCP APIs enabled automatically
- Budget alerts at 50/80/100% thresholds (critical for student projects)
- Docker Compose: 5-service local stack (API, UI, MLflow, Prometheus, Grafana)
- One command `docker compose up -d` for full local MLOps environment

### 16. Cloud Architecture
- Cloud Run (API + UI), Artifact Registry, GCS, Vertex AI
- All in europe-west1, auto-scale to zero
- CD syncs latest model from GCS before building

### 17. Monitoring + Observability
- Prometheus: 20+ metrics (predictions, batch, images, HTTP, system)
- Grafana: 16 dashboard panels
- structlog: JSON-formatted structured logging
- Drift detection: sliding window 1000 predictions, 3 drift signals

### 18. Drift Detection
- Mean probability shift, low confidence ratio, class ratio imbalance
- Thread-safe (threading.Lock), configurable thresholds
- Exposed via /health endpoint (drift_detected, drift_score)

### 19. Testing
- 183+ tests: unit, integration, load (Locust + k6)
- pytest with coverage tracking (--cov=src)
- Matrix: Python 3.11 + 3.12

### 20. Security
- Auth (API keys, SHA-256, constant-time), rate limiting, input validation
- CORS, Docker non-root, dependency scanning, code scanning (bandit)
- Image validation pipeline: size -> extension -> PIL -> format -> resize

### 21. Results
- Accuracy > 0.85, F1 > 0.80, inference ~45ms, cold start ~3-5s
- Live: API + UI on Cloud Run

### 22. Challenges
- State dict compatibility (dynamic classifier rebuild)
- Separate Dockerfiles (serving vs training/GPU)
- GCS model sync in CD (size comparison)
- Thread safety in monitoring

### 23. Future
- Larger dataset, bigger model (B3/B4), multi-class
- Canary deploys, model registry, auto-retrain on drift
- Grafana alerting, cost tracking
- Browser extension for marketplace scanning

### 24. Demo + Links
- API Swagger: https://ai-product-detector-714127049161.europe-west1.run.app/docs
- UI: https://ai-product-detector-ui-714127049161.europe-west1.run.app
- GitHub: github.com/nolancacheux/AI-Product-Photo-Detector

---

## Key Numbers to Remember
- **5,000** training images (balanced) -- up to 8,000 via HuggingFace
- **5.6M** model parameters
- **183+** tests
- **20+** Prometheus metrics
- **16** Grafana panels
- **3** GitHub Actions workflows (CI, CD, Training)
- **3** training paths (Local, Colab, Vertex AI)
- **5** Docker Compose services (API, UI, MLflow, Prometheus, Grafana)
- **3** DVC pipeline stages
- **6** Terraform-managed GCP resources
- **< 100ms** API latency (p50)
