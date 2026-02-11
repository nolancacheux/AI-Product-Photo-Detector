# Deployment Guide

This guide covers all deployment methods for the AI Product Photo Detector: local
Docker Compose for development, and Google Cloud Run for production.

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

The `docker-compose.yml` at the project root defines a complete local
development stack with five services.

### Services

| Service | Image | Port | Description |
|---|---|---|---|
| `api` | `ai-product-detector:1.0.0` | 8080 | FastAPI inference API |
| `ui` | `ai-product-detector:1.0.0` | 8501 | Streamlit web interface |
| `mlflow` | `python:3.11-slim` | 5000 | MLflow tracking server |
| `prometheus` | `prom/prometheus:v2.53.0` | 9090 | Metrics collection |
| `grafana` | `grafana/grafana:11.1.0` | 3000 | Dashboards and alerting |

### Prerequisites

- Docker and Docker Compose installed
- A trained model checkpoint at `models/checkpoints/best_model.pt`

### Quick Start

```bash
# Build and start all services
docker compose up -d

# Verify services are healthy
docker compose ps

# View API logs
docker compose logs -f api
```

### Access Points

| Service | URL |
|---|---|
| Inference API | http://localhost:8080 |
| API Docs (Swagger) | http://localhost:8080/docs |
| Streamlit UI | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |

### Service Dependencies

```
grafana --> prometheus --> api (healthy)
                ui -----> api (healthy)
                mlflow (independent)
```

The `api` service includes a Docker health check. The `ui` and `prometheus`
services wait for the API to become healthy before starting.

### Volumes

| Volume | Mount | Purpose |
|---|---|---|
| `./models` (bind) | `/app/models:ro` | Model checkpoint (read-only) |
| `./configs` (bind) | `/app/configs:ro` | Configuration files (read-only) |
| `mlflow-data` | `/mlflow` | MLflow database and artifacts |
| `prometheus-data` | `/prometheus` | Prometheus time-series data (15-day retention) |
| `grafana-data` | `/var/lib/grafana` | Grafana dashboards and configuration |

### Stopping and Cleaning Up

```bash
# Stop all services
docker compose down

# Stop and remove volumes (deletes MLflow/Prometheus/Grafana data)
docker compose down -v
```

---

## Cloud Run Deployment

### Automated Deployment (via CD Pipeline)

The recommended approach is to let the CD workflow handle deployment
automatically. See [CICD.md](CICD.md) for details.

Every push to `main` that passes CI triggers:

1. Docker image build with the latest model checkpoint.
2. Push to Artifact Registry.
3. Deployment to Cloud Run.
4. Smoke test.

### Manual Deployment (via gcloud)

For cases where manual deployment is needed (debugging, hotfixes, custom
configuration).

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
  --set-env-vars="API_KEYS=<your-api-key>,REQUIRE_AUTH=true" \
  --quiet
```

#### Verify

```bash
# Get the service URL
URL=$(gcloud run services describe ai-product-detector \
  --region=europe-west1 \
  --format='value(status.url)')

# Health check
curl "${URL}/health"

# Test prediction (with API key)
curl -X POST "${URL}/predict" \
  -H "X-API-Key: <your-api-key>" \
  -F "file=@test_image.jpg"
```

### Manual Deployment via GitHub Actions

Use the CD workflow dispatch to deploy a specific image tag or rebuild:

1. Go to **Actions > CD > Run workflow**.
2. Set `image_tag` to a previous commit SHA for rollback, or leave as `latest`
   to build fresh.
3. Optionally adjust memory allocation.

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
| `GCS_BUCKET` | (none) | GCS bucket name (set by Terraform) |

### Streamlit UI (Docker Compose)

| Variable | Default | Description |
|---|---|---|
| `API_URL` | `http://api:8080` | URL of the inference API |

### MLflow (Docker Compose)

| Variable | Default | Description |
|---|---|---|
| Backend store | `sqlite:///mlflow.db` | Local SQLite database |
| Artifact root | `/mlflow/artifacts` | Local artifact storage |

### Grafana (Docker Compose)

| Variable | Default | Description |
|---|---|---|
| `GF_SECURITY_ADMIN_USER` | `admin` | Grafana admin username |
| `GF_SECURITY_ADMIN_PASSWORD` | `admin` | Grafana admin password |
| `GF_USERS_ALLOW_SIGN_UP` | `false` | Disable public sign-up |

---

## Scaling Configuration

### Cloud Run Scaling

Managed by Terraform variables or `gcloud` flags:

| Parameter | Terraform Variable | gcloud Flag | Default | Recommendation |
|---|---|---|---|---|
| Min instances | `cloud_run_min_instances` | `--min-instances` | 0 | 0 for cost savings; 1 to avoid cold starts |
| Max instances | `cloud_run_max_instances` | `--max-instances` | 2 | 2--5 for moderate traffic |
| CPU | `cloud_run_cpu` | `--cpu` | 1000m | 1 vCPU is sufficient for inference |
| Memory | `cloud_run_memory` | `--memory` | 512Mi | 1Gi recommended (model loading) |

#### Cold Start Optimization

With `min_instances = 0`, the first request after a period of inactivity incurs
a cold start (5--15 seconds). The model must be loaded from disk into memory.

To reduce cold start latency:
- Set `min_instances = 1` (keeps one instance warm; costs ~$10/month).
- Optimize the Docker image size (CPU-only PyTorch is already used).
- The startup probe allows 5s initial delay with 3 retries.

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
  --max-instances=2 \
  --memory=1Gi
```

---

## Health Checks and Monitoring

### Endpoints

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/health` | GET | No | Basic health check (HTTP 200 if healthy) |
| `/healthz` | GET | No | Kubernetes-style health check |
| `/metrics` | GET | No | Prometheus metrics endpoint |

### Docker Health Check (local)

Defined in `docker/Dockerfile`:

```
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3
    CMD curl -f http://localhost:${PORT}/healthz || exit 1
```

### Cloud Run Probes

Defined in Terraform (`terraform/main.tf`):

- **Startup probe:** `GET /health`, 5s initial delay, 10s period, 3 failures
  before marking unhealthy.
- **Liveness probe:** `GET /health`, 30s period.

### Prometheus Metrics

The API exposes Prometheus metrics at `/metrics`. The local Docker Compose stack
includes a pre-configured Prometheus instance that scrapes these metrics every
10 seconds.

**Prometheus configuration:** `configs/prometheus.yml`

Scraped targets:
- `prometheus:9090` (self-monitoring)
- `api:8080` (inference API)

### Grafana Dashboards

Access Grafana at http://localhost:3000 (admin/admin). Provisioning
configuration is mounted from `configs/grafana/provisioning/`.

---

## Rollback Procedures

### Rollback via CD Workflow Dispatch

The quickest rollback method:

1. Identify the commit SHA of the last known good deployment.
2. Go to **Actions > CD > Run workflow**.
3. Set `image_tag` to the commit SHA.
4. The workflow skips building and deploys the existing image from Artifact
   Registry.

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
   gsutil ls -l gs://ai-product-detector-487013/models/
   ```
2. Restore the previous model:
   ```bash
   gsutil cp gs://ai-product-detector-487013/models/training-<OLD_SHA>/best_model.pt \
     gs://ai-product-detector-487013/models/best_model.pt
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
docker compose up api
curl http://localhost:8080/health
```

**Common causes:**
- Model loading takes longer than the startup probe timeout. Increase
  `initial_delay_seconds` in `terraform/main.tf`.
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
  gs://ai-product-detector-487013/models/best_model.pt
```

### Cloud Run cold start too slow

**Symptom:** First request after idle period takes 10--20 seconds.

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
- Verify the service account has the required IAM roles (see
  [INFRASTRUCTURE.md](INFRASTRUCTURE.md#service-account-permissions)).
- Check that the required APIs are enabled in the GCP project.
