# Deployment Guide

This guide covers all deployment methods for the AI Product Photo Detector: local Docker Compose for development and production, and Google Cloud Run for cloud deployment.

---

## Table of Contents

1. [Local Deployment (Docker Compose)](#local-deployment-docker-compose)
2. [Cloud Run Deployment](#cloud-run-deployment)
3. [Environment Variables Reference](#environment-variables-reference)
4. [Scaling Configuration](#scaling-configuration)
5. [Health Checks and Monitoring](#health-checks-and-monitoring)
6. [Rollback Procedures](#rollback-procedures)
7. [Troubleshooting](#troubleshooting)

---

## Local Deployment (Docker Compose)

The project uses a **base + override** pattern with three Compose files:

| File | Purpose |
|---|---|
| `docker-compose.yml` | Base service definitions (ports, networks, build context) |
| `docker-compose.dev.yml` | Development override (hot reload, debug logging, named volumes) |
| `docker-compose.prod.yml` | Production override (gunicorn, resource limits, strict health checks) |

### Services

| Service | Dockerfile / Image | Port | Description |
|---|---|---|---|
| `api` | `docker/Dockerfile` | 8080 | FastAPI inference API (uvicorn in dev, gunicorn in prod) |
| `ui` | `docker/ui.Dockerfile` | 8501 | Streamlit web interface |
| `mlflow` | `python:3.11-slim` | 5000 | MLflow tracking server (installs mlflow 2.16.0 at runtime) |
| `prometheus` | `prom/prometheus:v2.53.0` | 9090 | Metrics collection (15-day retention) |
| `grafana` | `grafana/grafana:11.1.0` | 3000 | Dashboards and alerting |

The API and UI are built from separate Dockerfiles. The API image does not include the Streamlit UI.

### Prerequisites

- Docker and Docker Compose installed
- A trained model checkpoint at `models/checkpoints/best_model.pt`

### Quick Start (Development)

```bash
# Build and start all services with dev overrides
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build

# Verify services are healthy
docker compose ps

# View API logs
docker compose logs -f api
```

### Quick Start (Production)

```bash
# Requires GF_ADMIN_PASSWORD to be set
export GF_ADMIN_PASSWORD="your-secure-password"

docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

### Access Points

| Service | URL |
|---|---|
| Inference API | http://localhost:8080 |
| API Docs (Swagger) | http://localhost:8080/docs |
| Streamlit UI | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

All ports are configurable via environment variables (`API_PORT`, `UI_PORT`, `MLFLOW_PORT`, `PROMETHEUS_PORT`, `GRAFANA_PORT`).

### Service Dependencies

```
grafana --> prometheus --> api (healthy)
                ui -----> api (healthy)
                mlflow (independent)
```

The `api` service includes a Docker health check. The `ui` and `prometheus` services wait for the API to become healthy before starting.

### Volumes

The base Compose file defines only bind mounts for configuration. Named volumes for data persistence are added by the override files.

**Development override (`docker-compose.dev.yml`):**

| Volume | Mount | Purpose |
|---|---|---|
| `./src` (bind) | `/app/src:ro` | Source code (hot reload) |
| `./configs` (bind) | `/app/configs:ro` | Configuration files |
| `./models` (bind) | `/app/models:ro` | Model checkpoint |
| `mlflow-data` | `/mlflow` | MLflow database and artifacts |
| `prometheus-data` | `/prometheus` | Prometheus time-series data |
| `grafana-data` | `/var/lib/grafana` | Grafana dashboards and state |

**Production override (`docker-compose.prod.yml`):**

| Volume | Mount | Purpose |
|---|---|---|
| `mlflow-data` | `/mlflow` | MLflow database and artifacts |
| `prometheus-data` | `/prometheus` | Prometheus time-series data |
| `grafana-data` | `/var/lib/grafana` | Grafana dashboards and state |

Production images are self-contained (no source bind mounts). The model checkpoint is baked into the Docker image at build time.

### Resource Limits (Production Override)

| Service | CPU Limit | Memory Limit | CPU Reserved | Memory Reserved |
|---|---|---|---|---|
| `api` | 2.0 | 2 GB | 0.5 | 512 MB |
| `ui` | 1.0 | 512 MB | 0.25 | 128 MB |
| `mlflow` | 1.0 | 1 GB | 0.25 | 256 MB |
| `prometheus` | 0.5 | 512 MB | 0.1 | 128 MB |
| `grafana` | 0.5 | 512 MB | 0.1 | 128 MB |

### Stopping and Cleaning Up

```bash
# Stop all services
docker compose -f docker-compose.yml -f docker-compose.dev.yml down

# Stop and remove volumes (deletes MLflow/Prometheus/Grafana data)
docker compose -f docker-compose.yml -f docker-compose.dev.yml down -v
```

---

## Cloud Run Deployment

### Production URLs

| Service | URL |
|---|---|
| API | https://ai-product-detector-714127049161.europe-west1.run.app |
| UI | https://ai-product-detector-ui-714127049161.europe-west1.run.app |

- **GCP Project:** `ai-product-detector-487013`
- **Region:** `europe-west1`
- **Artifact Registry:** `europe-west1-docker.pkg.dev/ai-product-detector-487013/ai-product-detector/api`
- **GCS Bucket:** `ai-product-detector-487013-mlops-data`
- **Service Account:** `714127049161-compute@developer.gserviceaccount.com`

### Automated Deployment (via CD Pipeline)

The recommended approach is to let the CD workflow handle deployment automatically. See [CICD.md](CICD.md) for details.

Every push to `main` that passes CI triggers:

1. Model checkpoint download from GCS (fallback to DVC pull).
2. Docker image build and push to Artifact Registry.
3. Deployment to Cloud Run with `REQUIRE_AUTH=false` and `ENVIRONMENT=production`.
4. Smoke tests (health, docs, predict endpoints).
5. Automatic rollback if smoke tests fail.

The CD pipeline sets `REQUIRE_AUTH=false` for the production deployment. API key authentication is not currently enforced in production.

### Manual Deployment (via gcloud)

For cases where manual deployment is needed (debugging, hotfixes, custom configuration).

#### Prerequisites

```bash
# Authenticate
gcloud auth login
gcloud config set project ai-product-detector-487013

# Configure Docker for Artifact Registry
gcloud auth configure-docker europe-west1-docker.pkg.dev --quiet
```

#### Build and Push

```bash
# Build the image
docker build -f docker/Dockerfile \
  -t europe-west1-docker.pkg.dev/ai-product-detector-487013/ai-product-detector/api:manual \
  .

# Push to Artifact Registry
docker push europe-west1-docker.pkg.dev/ai-product-detector-487013/ai-product-detector/api:manual
```

#### Deploy

```bash
gcloud run deploy ai-product-detector \
  --image=europe-west1-docker.pkg.dev/ai-product-detector-487013/ai-product-detector/api:manual \
  --region=europe-west1 \
  --port=8080 \
  --memory=1Gi \
  --allow-unauthenticated \
  --set-env-vars="REQUIRE_AUTH=false,ENVIRONMENT=production" \
  --quiet
```

#### Verify

```bash
# Health check
curl "https://ai-product-detector-714127049161.europe-west1.run.app/health"

# Test prediction
curl -X POST "https://ai-product-detector-714127049161.europe-west1.run.app/predict" \
  -F "file=@test_image.jpg"
```

### Manual Deployment via GitHub Actions

Use the CD workflow dispatch to deploy a specific image tag or rebuild:

1. Go to **Actions > CD > Run workflow**.
2. Set `image_tag` to a previous commit SHA for rollback, or leave as `latest` to build fresh.
3. Optionally adjust memory allocation (512Mi, 1Gi, or 2Gi).

---

## Environment Variables Reference

### Inference API (Cloud Run / Docker)

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8080` | Server port |
| `AIDETECT_MODEL_PATH` | `/app/models/checkpoints/best_model.pt` | Path to model checkpoint |
| `AIDETECT_LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `API_KEYS` | (none) | Comma-separated list of valid API keys |
| `REQUIRE_AUTH` | `false` | Enable API key authentication |
| `ENVIRONMENT` | (none) | Deployment environment label |
| `MLFLOW_TRACKING_URI` | (none) | MLflow server URL (set in Docker Compose overrides) |

### Streamlit UI (Docker Compose / Cloud Run)

| Variable | Default | Description |
|---|---|---|
| `API_URL` | `http://api:8080` (Compose) / `http://localhost:8080` (Dockerfile default) | URL of the inference API |

### MLflow (Docker Compose)

| Variable | Default | Description |
|---|---|---|
| Backend store | `sqlite:///mlflow/mlflow.db` | Local SQLite database |
| Artifact root | `/mlflow/artifacts` | Local artifact storage |

### Grafana (Docker Compose)

| Variable | Dev Default | Prod Default | Description |
|---|---|---|---|
| `GF_SECURITY_ADMIN_USER` | `admin` | `${GF_ADMIN_USER:-admin}` | Grafana admin username |
| `GF_SECURITY_ADMIN_PASSWORD` | `admin` | Required (`GF_ADMIN_PASSWORD`) | Grafana admin password |
| `GF_USERS_ALLOW_SIGN_UP` | `false` | `false` | Disable public sign-up |
| `GF_AUTH_ANONYMOUS_ENABLED` | `true` | `false` | Anonymous access |

---

## Scaling Configuration

### Cloud Run Scaling

Managed by Terraform (in `terraform/environments/prod/main.tf`) or `gcloud` flags:

| Parameter | Terraform Variable | Prod Value | Module Default |
|---|---|---|---|
| Min instances | `min_instances` | 0 | 0 |
| Max instances | `max_instances` | 3 | 2 |
| CPU | `cpu` | 1 | 1000m |
| Memory | `memory` | 1Gi | 512Mi |

The Terraform Cloud Run module is at `terraform/modules/cloud-run/`.

#### Cold Start Optimization

With `min_instances = 0`, the first request after a period of inactivity incurs a cold start (model must be loaded from disk into memory).

To reduce cold start latency:
- Set `min_instances = 1` (keeps one instance warm; incurs ongoing cost).
- The Docker image already uses CPU-only PyTorch to minimize image size.
- The startup probe allows up to 240 seconds for the container to become ready.

#### Adjusting via gcloud

```bash
# Scale up for a demo or load test
gcloud run services update ai-product-detector \
  --region=europe-west1 \
  --min-instances=1 \
  --max-instances=5 \
  --memory=2Gi

# Scale back down
gcloud run services update ai-product-detector \
  --region=europe-west1 \
  --min-instances=0 \
  --max-instances=3 \
  --memory=1Gi
```

---

## Health Checks and Monitoring

### API Endpoints

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/health` | GET | No | Basic health check (HTTP 200 if healthy) |
| `/healthz` | GET | No | Kubernetes-style health check |
| `/metrics` | GET | No | Prometheus metrics endpoint |

### Docker Health Check (Local)

Defined in `docker/Dockerfile`:

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1
```

The UI Dockerfile (`docker/ui.Dockerfile`) uses a Python-based health check:

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8501/_stcore/health').raise_for_status()" || exit 1
```

### Cloud Run Probes

Defined in the Terraform Cloud Run module (`terraform/modules/cloud-run/main.tf`):

- **Startup probe:** TCP socket on port 8080, 240-second timeout, 240-second period, 1 failure threshold. This generous timeout accommodates model loading time.
- **No liveness probe** is configured in Terraform. Cloud Run uses its built-in health management.

### Cloud Monitoring

The Terraform monitoring module (`terraform/modules/monitoring/`) provisions:

- **Uptime check:** HTTPS GET on `/health` (port 443, SSL validated).
- **Uptime alert:** Fires when the health check fails for more than 60 seconds.
- **Error rate alert:** Fires when 5xx responses exceed the configured threshold.
- **Notification channel:** Email (configured via `notification_email` variable in `terraform.tfvars`).

### Prometheus Metrics (Local)

The API exposes Prometheus metrics at `/metrics`. The local Docker Compose stack includes a pre-configured Prometheus instance that scrapes these metrics.

**Prometheus configuration:** `configs/prometheus.yml`

Scraped targets:
- `prometheus:9090` (self-monitoring)
- `api:8080` (inference API)

### Grafana Dashboards (Local)

Grafana is accessible at http://localhost:3000. In development mode, anonymous access is enabled. In production mode, authentication is required (`GF_ADMIN_PASSWORD` must be set).

Provisioning configuration is mounted from `configs/grafana/provisioning/`.

---

## Rollback Procedures

### Automatic Rollback (CD Pipeline)

The CD pipeline includes automatic rollback. If production smoke tests fail after deployment, the pipeline routes 100% of traffic back to the previous revision.

### Rollback via CD Workflow Dispatch

1. Identify the commit SHA of the last known good deployment.
2. Go to **Actions > CD > Run workflow**.
3. Set `image_tag` to the commit SHA.
4. The workflow skips building and deploys the existing image from Artifact Registry.

### Rollback via gcloud

```bash
# List recent revisions
gcloud run revisions list \
  --service=ai-product-detector \
  --region=europe-west1

# Route traffic to a specific revision
gcloud run services update-traffic ai-product-detector \
  --region=europe-west1 \
  --to-revisions=ai-product-detector-<REVISION_SUFFIX>=100

# Alternatively, redeploy a previous image
gcloud run deploy ai-product-detector \
  --image=europe-west1-docker.pkg.dev/ai-product-detector-487013/ai-product-detector/api:<PREVIOUS_SHA> \
  --region=europe-west1 \
  --quiet
```

### Rollback a Model

If a newly trained model causes issues:

1. Identify the previous model on GCS:
   ```bash
   gsutil ls -l gs://ai-product-detector-487013-mlops-data/models/
   ```
2. Restore the previous model:
   ```bash
   gsutil cp gs://ai-product-detector-487013-mlops-data/models/training-<OLD_SHA>/best_model.pt \
     gs://ai-product-detector-487013-mlops-data/models/best_model.pt
   ```
3. Trigger a CD deployment to rebuild the image with the restored model.

---

## Troubleshooting

### Container fails to start

**Symptom:** Cloud Run deployment succeeds but the service returns 503.

**Checks:**
```bash
# View Cloud Run logs
gcloud run services logs read ai-product-detector \
  --region=europe-west1 \
  --limit=50

# Check if the model file exists in the image
docker run --rm -it <IMAGE> ls -lh /app/models/checkpoints/
```

**Common causes:**
- Missing model checkpoint (`best_model.pt` not included in the image).
- Insufficient memory (increase to 1Gi or 2Gi).
- Port mismatch (ensure the app listens on the port specified by `PORT`).

### Health check failures

**Symptom:** Startup probe fails, service never becomes healthy.

**Checks:**
```bash
# Test locally
docker compose -f docker-compose.yml -f docker-compose.dev.yml up api
curl http://localhost:8080/health
```

**Common causes:**
- Model loading takes longer than the startup probe timeout (240 seconds). Increase `startup_probe_timeout` in Terraform.
- Application crash on startup (check logs for Python tracebacks).

### Docker build fails in CI

**Symptom:** The Docker Build Validation job fails on a pull request.

**Checks:**
- Verify `docker/Dockerfile` syntax.
- Check that all `COPY` paths exist and are not in `.dockerignore`.
- Review the build log for missing system dependencies.

### Model not found during CD build

**Symptom:** CD workflow fails with "ERROR: No model checkpoint available!"

**Causes:**
- No model has been uploaded to GCS yet.
- DVC remote is not configured or accessible.

**Fix:**
```bash
# Upload a model manually
gsutil cp models/checkpoints/best_model.pt \
  gs://ai-product-detector-487013-mlops-data/models/best_model.pt
```

### Cloud Run cold start too slow

**Symptom:** First request after idle period takes 10-20 seconds.

**Mitigations:**
1. Set `min_instances = 1` (keeps one instance warm).
2. Reduce Docker image size (already optimized with CPU-only PyTorch).
3. Use a lighter model if latency is critical.

### Prometheus not scraping metrics

**Symptom:** No data in Grafana dashboards.

**Checks:**
```bash
# Verify the API exposes metrics
curl http://localhost:8080/metrics

# Check Prometheus targets
# Open http://localhost:9090/targets in a browser
```

**Common causes:**
- The `api` service is not healthy (Prometheus depends on it).
- Network name mismatch in `configs/prometheus.yml`.

### Permission denied errors in CI/CD

**Symptom:** `gcloud` commands fail with 403 or permission denied.

**Checks:**
- Verify the `GCP_SA_KEY` secret is a valid JSON service account key.
- Verify the service account has the required IAM roles (see [INFRASTRUCTURE.md](INFRASTRUCTURE.md#service-account-permissions)).
- Check that the required APIs are enabled in the GCP project.
