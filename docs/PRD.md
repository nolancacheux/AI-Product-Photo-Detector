# Product Requirements Document (PRD)

## AI Product Photo Detector

**Version:** 1.0  
**Author:** Nolan Cacheux  
**Date:** 2025-01-30  
**Status:** Draft

---

## 1. Executive Summary

### 1.1 Problem Statement

E-commerce platforms are experiencing a surge in fraudulent listings featuring AI-generated product images. These synthetic images, created using tools like Stable Diffusion and Flux, are used to advertise products that don't exist, leading to customer fraud and platform trust erosion.

### 1.2 Solution

An ML-powered detection system that analyzes product images and returns a probability score indicating whether the image is AI-generated or real. Deployed as a production-grade API with full MLOps infrastructure.

### 1.3 Success Criteria

| Metric | Target |
|--------|--------|
| Model Accuracy | ≥ 90% |
| API Latency (p95) | < 200ms |
| System Uptime | 99% |
| Drift Detection | < 24h response time |

---

## 2. Stakeholders

| Role | Responsibility |
|------|----------------|
| Nolan Cacheux | Project Owner, Developer |
| M2 MLOps Professor | Evaluator |
| E-commerce Platforms | Target Users (hypothetical) |

---

## 3. Functional Requirements

### 3.1 Core Features

#### FR-1: Image Classification API
- **Description**: REST API endpoint that accepts an image and returns classification result
- **Input**: Image file (JPEG, PNG, WebP) - max 10MB
- **Output**: JSON response with probability score
- **Priority**: P0 (Critical)

```json
{
  "prediction": "ai_generated",
  "probability": 0.87,
  "confidence": "high",
  "inference_time_ms": 45,
  "model_version": "1.0.0"
}
```

#### FR-2: Health Check Endpoint
- **Description**: Endpoint to verify API availability and model status
- **Endpoint**: `GET /health`
- **Priority**: P0 (Critical)

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "uptime_seconds": 3600
}
```

#### FR-3: Web User Interface
- **Description**: Streamlit-based UI for manual image testing
- **Features**:
  - Drag & drop image upload
  - Real-time prediction display
  - Visualization of model confidence
- **Priority**: P1 (High)

#### FR-4: Batch Prediction
- **Description**: Endpoint for bulk image classification
- **Input**: ZIP file or list of image URLs
- **Output**: CSV/JSON with results
- **Priority**: P2 (Medium)

### 3.2 MLOps Features

#### FR-5: Model Training Pipeline
- **Description**: Reproducible training with fixed seeds
- **Components**:
  - Data validation
  - Feature extraction
  - Model training
  - Evaluation metrics
  - Model registration
- **Priority**: P0 (Critical)

#### FR-6: Model Versioning
- **Description**: All models tracked in MLflow Model Registry
- **Metadata**:
  - Training date
  - Dataset version
  - Hyperparameters
  - Performance metrics
- **Priority**: P0 (Critical)

#### FR-7: Drift Detection
- **Description**: Monitor incoming data for distribution shift
- **Metrics**:
  - Feature distribution comparison
  - Prediction confidence trends
  - Accuracy on labeled samples
- **Alert**: Trigger when drift score > threshold
- **Priority**: P1 (High)

---

## 4. Non-Functional Requirements

### 4.1 Performance

| Requirement | Specification |
|-------------|---------------|
| Inference Latency | p50 < 100ms, p95 < 200ms |
| Throughput | 100 requests/second |
| Model Load Time | < 10 seconds |
| Image Processing | < 50ms preprocessing |

### 4.2 Scalability

- Stateless API design for horizontal scaling
- Container-based deployment
- Load balancer ready

### 4.3 Reliability

| Requirement | Specification |
|-------------|---------------|
| Uptime | 99% availability |
| Error Rate | < 1% 5xx errors |
| Graceful Degradation | Return cached results on model failure |

### 4.4 Security

- Input validation (file type, size limits)
- No sensitive data in Docker images
- Environment-based configuration
- Rate limiting on API endpoints
- Non-root container execution

### 4.5 Observability

| Component | Tool |
|-----------|------|
| Logging | structlog (JSON format) |
| Metrics | Prometheus |
| Dashboards | Grafana |
| Tracing | OpenTelemetry (optional) |

**Key Metrics to Track:**
- Request latency (histogram)
- Request count (by status code)
- Model inference time
- Prediction distribution
- Error rates

---

## 5. Technical Architecture

### 5.1 System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                             │
├─────────────────────────────────────────────────────────────────┤
│  Web UI (Streamlit)  │  REST Client  │  Batch Processor          │
└──────────┬───────────┴───────┬───────┴────────────┬─────────────┘
           │                   │                    │
           ▼                   ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                         API Layer                                │
├─────────────────────────────────────────────────────────────────┤
│                    FastAPI Application                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  /predict   │  │  /health    │  │  /metrics (prometheus)  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ML Layer                                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐                     │
│  │  Preprocessor    │  │  EfficientNet-B0 │                     │
│  │  (transforms)    │→ │  (classifier)    │                     │
│  └──────────────────┘  └──────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Observability Layer                           │
├─────────────────────────────────────────────────────────────────┤
│  Prometheus  │  Grafana  │  structlog  │  MLflow                │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Technology Stack

| Layer | Technology | Justification |
|-------|------------|---------------|
| **ML Framework** | PyTorch 2.0+ | Industry standard, great ecosystem |
| **Model** | EfficientNet-B0 (timm) | Good accuracy/speed tradeoff |
| **API** | FastAPI | Async, auto-docs, type hints |
| **Web UI** | Streamlit | Rapid prototyping, free hosting |
| **Experiment Tracking** | MLflow | Industry standard, free |
| **Data Versioning** | DVC | Git-like for data |
| **Containerization** | Docker | Standard, lightweight images |
| **CI/CD** | GitHub Actions | Free tier, integrated |
| **Metrics** | Prometheus | Standard, Grafana compatible |
| **Logging** | structlog | Structured JSON logs |

### 5.3 Model Architecture

```
Input Image (224x224x3)
         │
         ▼
┌─────────────────────────┐
│  Preprocessing          │
│  - Resize (224x224)     │
│  - Normalize            │
│  - ToTensor             │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  EfficientNet-B0        │
│  (pretrained ImageNet)  │
│  - Frozen backbone      │
│  - Custom head          │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Classification Head    │
│  - Linear(1280, 512)    │
│  - ReLU + Dropout(0.3)  │
│  - Linear(512, 1)       │
│  - Sigmoid              │
└───────────┬─────────────┘
            │
            ▼
    Probability [0, 1]
    (1 = AI-generated)
```

---

## 6. Data Requirements

### 6.1 Dataset Composition

| Category | Source | Count | Notes |
|----------|--------|-------|-------|
| **Real Products** | Amazon Berkeley Objects | 2,500 | Diverse categories |
| **AI (Stable Diffusion)** | Self-generated | 1,250 | SD 1.5 & SDXL |
| **AI (Flux)** | Self-generated | 1,250 | Flux.1 dev |

**Total: 5,000 images**

### 6.2 Data Split

| Split | Percentage | Count |
|-------|------------|-------|
| Train | 70% | 3,500 |
| Validation | 15% | 750 |
| Test | 15% | 750 |

### 6.3 Data Augmentation

- Random horizontal flip
- Random rotation (±15°)
- Color jitter (brightness, contrast)
- Random crop & resize

---

## 7. API Specification

### 7.1 Endpoints

#### POST /predict

```yaml
Request:
  Content-Type: multipart/form-data
  Body:
    file: binary (image file)

Response (200):
  {
    "prediction": "ai_generated" | "real",
    "probability": float (0.0 - 1.0),
    "confidence": "low" | "medium" | "high",
    "inference_time_ms": int,
    "model_version": string
  }

Response (400):
  {
    "error": "Invalid image format",
    "detail": "Supported formats: JPEG, PNG, WebP"
  }

Response (413):
  {
    "error": "File too large",
    "detail": "Maximum file size: 10MB"
  }
```

#### GET /health

```yaml
Response (200):
  {
    "status": "healthy",
    "model_loaded": true,
    "model_version": "1.0.0",
    "uptime_seconds": int
  }

Response (503):
  {
    "status": "unhealthy",
    "model_loaded": false,
    "error": "Model failed to load"
  }
```

#### GET /metrics

```yaml
Response (200):
  Content-Type: text/plain
  Body: Prometheus metrics format
```

---

## 8. Deployment

### 8.1 Docker Images

| Image | Purpose | Base |
|-------|---------|------|
| `ai-detector-train:v1.0.0` | Training | python:3.11-slim |
| `ai-detector-serve:v1.0.0` | Inference | python:3.11-slim |

**Image Requirements:**
- No `latest` tag (explicit versioning)
- Non-root user execution
- No sensitive data baked in
- Multi-stage builds for minimal size

### 8.2 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MODEL_PATH` | Yes | Path to model weights |
| `LOG_LEVEL` | No | Logging level (default: INFO) |
| `MLFLOW_TRACKING_URI` | No | MLflow server URL |
| `PROMETHEUS_PORT` | No | Metrics port (default: 9090) |

### 8.3 Infrastructure

- **Host**: Hetzner VPS (existing)
- **Container Runtime**: Docker
- **Reverse Proxy**: Nginx (optional)
- **SSL**: Let's Encrypt (optional)

---

## 9. Incident Scenario

### 9.1 Scenario Description

> **Incident**: New AI image generator "Flux 2.0" releases, producing hyper-realistic product images that bypass the current detector.

### 9.2 Detection

1. Drift detection system notices:
   - Prediction confidence dropping
   - Unusual feature distribution in incoming images
   - User reports of false negatives increasing

2. Alert triggered: `DRIFT_DETECTED: confidence_drop > 15%`

### 9.3 Response Playbook

1. **Acknowledge** (< 1 hour)
   - Confirm drift alert
   - Initial assessment

2. **Investigate** (< 4 hours)
   - Identify new generator
   - Collect sample images
   - Quantify impact

3. **Mitigate** (< 24 hours)
   - Generate new training data with Flux 2.0
   - Augment dataset
   - Retrain model

4. **Deploy** (< 48 hours)
   - A/B test new model
   - Gradual rollout
   - Monitor metrics

5. **Post-mortem**
   - Document incident
   - Update runbook
   - Improve detection

### 9.4 Metrics During Incident

| Metric | Before | During | After |
|--------|--------|--------|-------|
| Accuracy | 95% | 70% | 93% |
| False Negative Rate | 5% | 30% | 7% |
| Detection Latency | - | 2 hours | - |

---

## 10. Timeline

### Phase 1: Foundation (Week 1)
- [ ] Project setup (repo, CI/CD, Docker)
- [ ] Dataset collection & generation
- [ ] Data validation pipeline

### Phase 2: Model Development (Week 2)
- [ ] Model training pipeline
- [ ] MLflow integration
- [ ] Evaluation & optimization

### Phase 3: Deployment (Week 3)
- [ ] FastAPI inference server
- [ ] Streamlit UI
- [ ] Docker images

### Phase 4: Observability (Week 4)
- [ ] Prometheus metrics
- [ ] Grafana dashboards
- [ ] Drift detection

### Phase 5: Incident Simulation (Week 5)
- [ ] Simulate drift scenario
- [ ] Document response
- [ ] Final documentation

---

## 11. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Dataset bias | Medium | High | Diverse data sources |
| Model overfitting | Medium | Medium | Proper validation split |
| Inference latency | Low | Medium | Model optimization |
| Drift undetected | Medium | High | Multiple detection signals |

---

## 12. Success Metrics

| Metric | Measurement | Target |
|--------|-------------|--------|
| Code Quality | SonarCloud score | A rating |
| Test Coverage | pytest-cov | > 80% |
| Documentation | README completeness | 100% |
| MLOps Maturity | Checklist completion | All items |

---

## 13. Competitive Landscape

### 13.1 Existing Solutions

| Solution | Type | Limitations |
|----------|------|-------------|
| **Google SynthID** | Watermark-based | Only detects Google Imagen images |
| **C2PA Protocol** | Metadata-based | Easily stripped, requires adoption |
| **Hive Moderation** | Detection API | Paid, closed-source |
| **Illuminarty** | Detection API | Limited accuracy on new generators |

### 13.2 Why SynthID Doesn't Solve the Problem

Google's SynthID embeds invisible watermarks in AI-generated images, but:

1. **Only works on Google Imagen**: Watermarks are embedded at generation time by Google's Imagen model. Images from Stable Diffusion, Midjourney, DALL-E, or Flux have NO watermark.

2. **Requires generator cooperation**: The AI tool must actively embed the watermark. Scammers use open-source tools (SD, Flux) that don't watermark.

3. **Watermarks can be removed**: While robust to cropping/resizing, determined attackers can attempt removal.

> *"SynthID is rolling out first in a Google-centric way: Google Cloud customers who use the company's Vertex AI platform and the Imagen image generator will be able to embed and detect the watermark."*  
> — The Verge, 2023

### 13.3 Our Differentiation

| Aspect | SynthID | Our Detector |
|--------|---------|--------------|
| Detection method | Embedded watermark | Visual artifact analysis |
| Works on SD/Flux/MJ | No | Yes |
| Works without cooperation | No | Yes |
| Open source | No | Yes |
| Self-hostable | No | Yes |

**Our value proposition**: Unlike watermark-based approaches, our detector works on ANY AI-generated image regardless of source, making it effective against real-world e-commerce fraud where scammers use open-source generators.

---

## 14. Appendix

### 14.1 Reference Materials

- [MLOps Course PDF](./Projet_MLOps_M2.pdf)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

### B. Glossary

| Term | Definition |
|------|------------|
| **Drift** | Change in data distribution over time |
| **Inference** | Using trained model to make predictions |
| **MLOps** | ML + DevOps practices for production ML |

---

*Document maintained by Nolan Cacheux - Last updated: 2025-01-30*
