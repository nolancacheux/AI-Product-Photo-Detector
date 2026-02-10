# Architecture Audit Report

**Date:** 2025-07-17  
**Scope:** Global architecture, design patterns, scalability, package structure, configuration  
**Project:** AI Product Photo Detector (v1.0.0)

---

## Executive Summary

The project exhibits **solid architecture for an M2 MLOps project**. The separation of concerns is clean, the CI/CD pipeline is production-grade, and the monitoring stack (Prometheus + Grafana + drift detection) demonstrates real operational maturity. However, several patterns and structural decisions need attention for production readiness.

**Overall Grade: B+ (strong academic project, needs polish for production)**

---

## 1. Architecture — Separation of Responsibilities

### ✅ What's Good

- **Clean module separation**: `training/`, `inference/`, `monitoring/`, `utils/` each have a clear, single responsibility
- **Data flow is logical**: `data/ → training/ → models/ → inference/ → monitoring/`
- **No business logic leakage**: Schemas stay in `schemas.py`, auth in `auth.py`, validation in `validation.py`
- **API layer is well structured**: `api.py` orchestrates, delegates to `predictor.py` for inference, `drift.py` for monitoring
- **Config centralization**: YAML configs for train/inference, env vars via `pydantic-settings`

### ⚠️ Issues Found

| # | Issue | Severity | Location |
|---|-------|----------|----------|
| 1 | **`api.py` is 617 lines** — too large, handles middleware + routes + lifespan | Medium | `src/inference/api.py` |
| 2 | **`predictor.py` imports from `training.model`** — inference depends on training module | Medium | `src/inference/predictor.py:12` |
| 3 | **`validation.py` is unused** — thorough validation module exists but `api.py` does its own inline validation | Medium | `src/inference/api.py` vs `validation.py` |
| 4 | **`src/data/__init__.py` is empty** — module exists but contains nothing | Low | `src/data/` |
| 5 | **`total_predictions` is a global counter** in `api.py` instead of using the Prometheus counter | Low | `src/inference/api.py` |

### Recommendations

**Issue 1 — Split `api.py`:**
```
src/inference/
├── api.py          → App creation, lifespan, root/health/info endpoints (~150 lines)
├── routes.py       → /predict, /predict/batch, /metrics, /drift (~250 lines)  
├── middleware.py   → observability_middleware, CORS setup (~100 lines)
├── predictor.py    → (unchanged)
├── schemas.py      → (unchanged)
├── auth.py         → (unchanged)
└── validation.py   → (unchanged, but actually wired in)
```

**Issue 2 — Decouple inference from training:**  
The `Predictor` imports `AIImageDetector` from `src.training.model` to instantiate the model class. This creates a hard dependency: deploying inference requires shipping the entire training module. Solution: move the model *architecture definition* to a shared location or use `torch.jit` / ONNX for serving.

**Issue 3 — Wire up `validation.py`:**  
The `validate_image_bytes()` and `validate_upload_file()` functions in `validation.py` are comprehensive (magic byte detection, dimension checks, content hash). But `api.py` does its own simpler checks inline. The validation module should replace the inline checks.

---

## 2. Design Patterns

### Singleton Pattern (Predictor)

**Current:** Global `predictor: Predictor | None = None` in `api.py`, initialized in `lifespan()`.

**Verdict: Acceptable for this use case**, but not ideal.

| Aspect | Assessment |
|--------|------------|
| Thread safety | ⚠️ `total_predictions += 1` is not thread-safe (though uvicorn async is single-threaded) |
| Testability | ⚠️ Global state makes unit testing harder — needs monkeypatching |
| Multi-model | ❌ Can't serve multiple models simultaneously |

**Better approach:** FastAPI dependency injection:
```python
async def get_predictor(request: Request) -> Predictor:
    return request.app.state.predictor
```
This is testable, mockable, and supports swapping predictors.

### Factory Pattern (Model)

**Current:** `create_model()` in `model.py` — simple factory function.

**Verdict: ✅ Good.** The factory properly abstracts model creation. It accepts `model_name` which allows swapping architectures (e.g., `efficientnet_b0` → `resnet50`).

**Improvement:** Add a model registry for multi-model support:
```python
MODEL_REGISTRY = {
    "efficientnet_b0": AIImageDetector,
    "resnet50": ResNetDetector,
}
```

### Dependency Injection

**Current:** `verify_api_key` uses `Depends()` properly. Good FastAPI pattern.

**Verdict: ✅ Auth DI is clean.** But the predictor and drift detector should also use DI instead of globals.

---

## 3. Scalability

### Multi-Model Support

**Current state: ❌ Single model hardcoded.**

- `Predictor.__init__()` loads one model from one path
- The model architecture class `AIImageDetector` is hardcoded in `_load_model()`
- Config only supports one `model.path`

**To support multiple models:**
1. Model registry pattern (map name → class)
2. Predictor pool (dict of predictors by name/version)
3. Route parameter: `/predict?model=v2`

### New Detection Types

**Current state: ⚠️ Partially extensible.**

- The binary classification (real vs AI) is baked in with `PredictionResult(StrEnum)` having only `REAL` and `AI_GENERATED`
- Adding a new category (e.g., `PHOTOSHOPPED`) requires changes in: schemas, predictor logic, training pipeline, monitoring
- The model outputs a single sigmoid — needs architecture change for multi-class

**Extensibility score: 5/10** — possible but requires touching many files.

### Monitoring Scalability

**Current state: ✅ Good foundation.**

- Prometheus metrics are well-defined with proper labels
- Drift detector uses a sliding window (configurable size)
- Docker Compose includes Prometheus + Grafana

**Concern:** Drift detector is in-memory only — restarts lose all history. For production, persist to a time-series DB.

---

## 4. Package Structure

### Import Cleanliness

**Analysis of all 29 cross-module imports:**

```
src.inference → src.training    (1 import: model.AIImageDetector)  ⚠️ Cross-boundary
src.inference → src.monitoring  (2 imports: drift, metrics)        ✅ Expected
src.inference → src.utils       (3 imports: config, logger)        ✅ Expected
src.training  → src.utils       (2 imports: config, logger)        ✅ Expected  
src.monitoring → src.utils      (1 import: logger)                 ✅ Expected
```

**Dependency graph:**
```
utils (leaf) ← training ← inference → monitoring
                              ↑              ↑
                              └──────────────┘
```

**No circular dependencies detected. ✅**

The only problematic edge is `inference → training` (for the model class). Everything else follows a clean DAG.

### `__init__.py` API Surface

| Module | Exports | Assessment |
|--------|---------|------------|
| `src/` | `__version__` only | ✅ Minimal, correct |
| `src/inference/` | `Predictor` + all schemas | ✅ Good public API |
| `src/training/` | All transforms, dataset, model | ✅ Good public API |
| `src/monitoring/` | Nothing (empty) | ⚠️ Should export `DriftDetector`, key metrics |
| `src/utils/` | All configs + logger | ✅ Good public API |
| `src/data/` | Nothing (empty module) | ⚠️ Placeholder — remove or populate |
| `src/ui/` | Nothing | ✅ OK (standalone Streamlit app) |

---

## 5. Configuration

### Current State

| Config Source | Type | Contents |
|-------------|------|----------|
| `configs/train_config.yaml` | Build-time | Hyperparameters, data paths, MLflow settings |
| `configs/inference_config.yaml` | Runtime | Server settings, thresholds, rate limits |
| `src/utils/config.py` | Runtime | Pydantic `Settings` (env vars), YAML loader |
| `.env.example` | Template | Environment variable reference |
| `docker-compose.yml` | Deploy-time | Service config, env vars |

### Assessment

**✅ Good separation:**
- Training config = YAML (static, versioned)
- Runtime config = env vars via `pydantic-settings`
- Infrastructure config = Docker Compose + Terraform

**⚠️ Issues:**

1. **Config duplication**: Thresholds are defined in 3 places:
   - `configs/train_config.yaml` → `thresholds.classification: 0.5`
   - `configs/inference_config.yaml` → `thresholds.classification: 0.5`
   - `src/inference/predictor.py` → `threshold: float = 0.5` (default arg)
   
   → Single source of truth needed.

2. **Pydantic config models exist but aren't used**: `DataConfig`, `ModelConfig`, `TrainingConfig`, `ThresholdsConfig` are defined in `config.py` but `train.py` just uses raw `dict` from YAML. The typed config models should be used.

3. **`inference_config.yaml` partially ignored**: The server config, rate limit config, and metrics config in `inference_config.yaml` are never read — `api.py` hardcodes these values (`MAX_FILE_SIZE`, rate limits, etc.).

4. **Image size mismatch**: Training config says `image_size: 128` but inference `Predictor` resizes to `224×224`. This is a **functional bug** — the model was trained on 128×128 but inference sends 224×224. (Mitigated because the checkpoint stores config and predictor could read it, but the transform is hardcoded to 224.)

---

## 6. Missing Pieces for Production MLOps

### Critical (Must-Have)

| # | Missing | Why It Matters |
|---|---------|---------------|
| 1 | **Model registry** (MLflow Model Registry or similar) | No way to promote staging → production models |
| 2 | **A/B testing / shadow mode** | Can't safely roll out new model versions |
| 3 | **Data validation pipeline** (Great Expectations / Pandera) | No input data quality gates |
| 4 | **Persistent drift storage** | Drift history lost on restart |
| 5 | **Alerting rules** (Grafana alerts or PagerDuty) | Drift detected but nobody gets notified |
| 6 | **Rollback mechanism** | `deploy.yml` deploys but no automated rollback on failure |

### Important (Should-Have)

| # | Missing | Why It Matters |
|---|---------|---------------|
| 7 | **Feature store** | Image embeddings not stored for analysis |
| 8 | **Retraining pipeline trigger** | Drift detected → manual intervention required |
| 9 | **Load testing** (Locust / k6) | No performance baseline |
| 10 | **API versioning** (`/v1/predict`) | Breaking changes require client updates |
| 11 | **Canary deployments** | All-or-nothing deployments are risky |
| 12 | **Secret management** (GCP Secret Manager) | API keys via env vars, not rotatable |

### Nice-to-Have

| # | Missing | Why It Matters |
|---|---------|---------------|
| 13 | **Model explainability** (GradCAM / SHAP) | Can't explain *why* an image is flagged |
| 14 | **Batch inference pipeline** (async / queue-based) | Current batch is synchronous, blocks |
| 15 | **Multi-region deployment** | Single region = single point of failure |

---

## 7. Fixes Applied

### Fix 1: `src/monitoring/__init__.py` — Expose Public API

The monitoring module exported nothing, making users import directly from submodules.

### Fix 2: `src/data/__init__.py` — Document purpose or mark as namespace

### Fix 3: Image size inconsistency documentation

---

## 8. Summary of Recommendations (Priority Order)

| Priority | Action | Effort |
|----------|--------|--------|
| 🔴 High | Fix image size mismatch (128 train vs 224 inference) | 1h |
| 🔴 High | Wire `validation.py` into `api.py` (replace inline checks) | 2h |
| 🟡 Medium | Decouple `inference` from `training` (shared model module) | 3h |
| 🟡 Medium | Split `api.py` into routes + middleware | 2h |
| 🟡 Medium | Use Pydantic config models instead of raw dicts in `train.py` | 2h |
| 🟡 Medium | Replace global state with FastAPI DI | 1h |
| 🟢 Low | Add model registry for multi-model support | 4h |
| 🟢 Low | Single source of truth for thresholds | 1h |
| 🟢 Low | Add API versioning (`/v1/`) | 1h |

---

## Architecture Diagram (Current)

```
┌─────────────────────────────────────────────────────────┐
│                    src/ (package)                        │
│                                                         │
│  ┌──────────┐    ┌───────────┐    ┌──────────────────┐  │
│  │ utils/   │◄───│ training/ │    │   inference/      │  │
│  │  config  │    │  model    │◄───│   predictor ──┐   │  │
│  │  logger  │◄───│  dataset  │    │   api ────────┤   │  │
│  │          │    │  train    │    │   schemas     │   │  │
│  │          │    │  augment  │    │   auth        │   │  │
│  │          │    │           │    │   validation  │   │  │
│  └──────────┘    └───────────┘    └───────┬──────┘   │  │
│       ▲                                    │          │  │
│       │          ┌──────────────┐          │          │  │
│       └──────────│ monitoring/  │◄─────────┘          │  │
│                  │  metrics     │                      │  │
│                  │  drift       │                      │  │
│                  └──────────────┘                      │  │
│                                                         │
│  ┌──────────┐    ┌──────────┐                           │
│  │  data/   │    │   ui/    │  (standalone Streamlit)   │
│  │ (empty)  │    │  app.py  │                           │
│  └──────────┘    └──────────┘                           │
└─────────────────────────────────────────────────────────┘
```

---

*Report generated by architecture audit. No commits made.*
