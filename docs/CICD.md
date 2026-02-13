# CI/CD Pipeline Documentation

This project uses four GitHub Actions workflows to automate code quality checks,
application deployment, model training, and PR previews. All workflow definitions
live in `.github/workflows/`.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Workflow: CI](#workflow-ci)
3. [Workflow: CD](#workflow-cd)
4. [Workflow: Model Training](#workflow-model-training)
5. [Workflow: PR Preview](#workflow-pr-preview)
6. [Pipeline Flow Diagram](#pipeline-flow-diagram)
7. [Required Secrets](#required-secrets)
8. [Triggering Workflows](#triggering-workflows)
9. [Modifying Workflows](#modifying-workflows)
10. [Branch Protection Rules](#branch-protection-rules)

---

## Pipeline Overview

| Workflow | File | Trigger | Purpose |
|---|---|---|---|
| CI | `.github/workflows/ci.yml` | Push/PR to `main` | Lint, type check, test, security scan, Docker build validation |
| CD | `.github/workflows/cd.yml` | Push to `main` or manual dispatch | Build Docker image, push to Artifact Registry, deploy to Cloud Run |
| Model Training | `.github/workflows/model-training.yml` | Manual dispatch or data changes on `main` | Train on Vertex AI, evaluate, conditionally deploy |
| PR Preview | `.github/workflows/pr-preview.yml` | PR open/update | Deploy preview environment for testing |

All workflows use concurrency groups to prevent duplicate runs. CI cancels
in-progress runs on the same branch; CD and Model Training do not cancel
(to avoid interrupted deployments).

---

## Workflow: CI

**File:** `.github/workflows/ci.yml`

The CI workflow runs on every push and pull request targeting `main`. It
validates code quality, correctness, and security across multiple dimensions.

### Jobs

#### 1. Lint and Format Check

- **Runner:** `ubuntu-latest`, Python 3.11
- **Tools:** [ruff](https://docs.astral.sh/ruff/)
- **Steps:**
  1. Install project with dev dependencies (`pip install -e ".[dev]"`)
  2. Run `ruff check src/ tests/` (linting rules)
  3. Run `ruff format --check src/ tests/` (formatting validation)

#### 2. Type Checking

- **Runner:** `ubuntu-latest`, Python 3.11
- **Tool:** [mypy](https://mypy.readthedocs.io/)
- **Command:** `mypy src/ --ignore-missing-imports`

#### 3. Tests (Matrix)

- **Runner:** `ubuntu-latest`
- **Matrix:** Python 3.11 and Python 3.12 (`fail-fast: false`)
- **Tool:** [pytest](https://docs.pytest.org/) with coverage
- **Command:**
  ```bash
  pytest tests/ -v --tb=short \
    --cov=src \
    --cov-report=term-missing \
    --cov-report=xml:coverage.xml \
    --cov-report=html:htmlcov/
  ```
- **Artifacts:** Coverage report (XML + HTML) uploaded for the Python 3.11 run,
  retained for 14 days.
- **Summary:** A Markdown coverage table is posted to the GitHub Actions job
  summary (Python 3.11 only).

#### 4. Security Scan

- **Runner:** `ubuntu-latest`, Python 3.11
- **Tools:**
  - [pip-audit](https://pypi.org/project/pip-audit/) -- checks installed
    dependencies for known vulnerabilities.
  - [bandit](https://bandit.readthedocs.io/) -- static analysis for common
    security issues in Python source code. Skips `B101` (assert) and `B601`
    (shell injection in parameterized calls).
- **Note:** Both steps use `continue-on-error: true` so they report findings
  without blocking the pipeline.

#### 5. Docker Build Validation (PR only)

- **Runner:** `ubuntu-latest`
- **Depends on:** `lint`, `test`
- **Condition:** Only runs on pull requests (`github.event_name == 'pull_request'`).
- **Action:** Builds the Docker image from `docker/Dockerfile` without pushing.
  Uses GitHub Actions cache (`type=gha`) for layer caching.

---

## Workflow: CD

**File:** `.github/workflows/cd.yml`

The CD workflow builds the application Docker image, pushes it to Google
Artifact Registry, deploys to Cloud Run, and runs a smoke test.

### Environment Variables

| Variable | Value |
|---|---|
| `REGION` | `europe-west1` |
| `SERVICE` | `ai-product-detector` |
| `REGISTRY` | `europe-west1-docker.pkg.dev` |
| `IMAGE` | `europe-west1-docker.pkg.dev/ai-product-detector-487013/ai-product-detector/api` |

### Jobs

#### 1. Wait for CI

- **Condition:** Only on `push` events (skipped for manual dispatch).
- Uses `lewagon/wait-on-check-action` to wait for CI jobs (Lint, Tests, Type
  Checking) to complete successfully before proceeding.

#### 2. Build and Push Docker Image

- **Depends on:** `ci-check` (success) or `workflow_dispatch`.
- **Steps:**
  1. Authenticate to GCP using the `GCP_SA_KEY` secret.
  2. Configure Docker to authenticate with Artifact Registry.
  3. Fetch the model checkpoint:
     - **Strategy 1:** Direct download from `gs://<GCS_BUCKET>/models/best_model.pt`.
     - **Strategy 2 (fallback):** `dvc pull` from DVC remote.
     - Fails the build if no model checkpoint is available.
  4. Determine image tag (commit SHA for new builds, or a user-specified tag for
     rollbacks via manual dispatch).
  5. Build and push the image with both the SHA tag and `latest`.
- **Output:** `image_tag` (the tag used for deployment).

#### 3. Deploy to Cloud Run

- **Depends on:** `build`.
- **Steps:**
  1. Deploy using `gcloud run deploy` with the built image.
  2. Configuration: port 8080, configurable memory (default 1Gi),
     unauthenticated access enabled.
  3. Environment variables set: `API_KEYS`, `REQUIRE_AUTH=true`.
- **Environment:** `production` (creates a GitHub Environments entry with the
  deployment URL).

#### 4. Smoke Test

- **Depends on:** `deploy`.
- Waits 15 seconds for the service to stabilize, then sends a `GET` request to
  `/health`. Fails the workflow if the response is not HTTP 200.

### Manual Dispatch Inputs

| Input | Description | Default |
|---|---|---|
| `image_tag` | Image tag to deploy (commit SHA or `latest` to build fresh) | `latest` |
| `memory` | Cloud Run memory allocation (`512Mi`, `1Gi`, `2Gi`) | `1Gi` |

---

## Workflow: Model Training

**File:** `.github/workflows/model-training.yml`

This workflow orchestrates end-to-end model training on Vertex AI, evaluates
the resulting model, and conditionally deploys it to production.

### Environment Variables

| Variable | Value |
|---|---|
| `GCP_PROJECT` | `ai-product-detector-487013` |
| `REGION` | `europe-west1` |
| `GCS_BUCKET` | `ai-product-detector-487013` |
| `MACHINE_TYPE` | `n1-standard-4` |
| `ACCELERATOR_TYPE` | `NVIDIA_TESLA_T4` |
| `ACCELERATOR_COUNT` | `1` |

### Jobs

#### 1. Upload Training Data

- Verifies that training data exists on GCS under
  `gs://<GCS_BUCKET>/data/processed/{train,val,test}/`.
- Posts a file count summary to the job summary.

#### 2. Build Training Image

- Builds `docker/Dockerfile.training` (GPU-enabled PyTorch base image).
- Pushes to `<REPO>/training:latest` and `<REPO>/training:<SHA>`.
- **Output:** `image_uri` for use by the training job.

#### 3. Submit Vertex AI Training Job

- **Depends on:** `upload-data`, `build-training-image`.
- **Timeout:** 180 minutes.
- Submits a `CustomContainerTrainingJob` to Vertex AI using the Python SDK.
- Configuration:
  - Machine: `n1-standard-4` with 1x NVIDIA Tesla T4 GPU.
  - Command: `python -m src.training.train --config configs/train_config.yaml`.
  - Environment variables passed: `EPOCHS`, `BATCH_SIZE`, `GCS_DATA_PATH`,
    `GCS_MODEL_OUTPUT`.
- After training completes:
  1. Downloads `best_model.pt` from the training output directory.
  2. Copies it to the canonical location `gs://<GCS_BUCKET>/models/best_model.pt`.
  3. Uploads the model as a GitHub Actions artifact (30-day retention).
- **Output:** `model_gcs_path`.

#### 4. Evaluate Model

- **Depends on:** `submit-training`.
- Downloads the trained model artifact and test data from GCS.
- Runs evaluation on CPU using the project's own model and dataset code.
- Computes: accuracy, precision, recall, F1 score.
- **Quality gate:** accuracy >= 0.85 AND F1 >= 0.80.
- **Outputs:** `accuracy`, `f1_score`, `passed` (true/false).
- Uploads `reports/metrics.json` as an artifact.

#### 5. Deploy to Cloud Run (Conditional)

- **Depends on:** `evaluate`.
- **Condition:** Deploys only when:
  - Manual dispatch with `auto_deploy: true` AND quality gate passed, OR
  - Push trigger (data change) AND quality gate passed.
- Builds a new inference image with the trained model baked in, pushes it,
  deploys to Cloud Run, and runs a smoke test.

### Manual Dispatch Inputs

| Input | Type | Default | Description |
|---|---|---|---|
| `epochs` | string | `15` | Number of training epochs |
| `batch_size` | string | `64` | Training batch size |
| `auto_deploy` | boolean | `false` | Deploy automatically if evaluation passes |

---

## Pipeline Flow Diagram

```
                          +------------------+
                          |   Push to main   |
                          |   or PR to main  |
                          +--------+---------+
                                   |
              +--------------------+--------------------+
              |                                         |
              v                                         v
    +-------------------+                     +-------------------+
    |     CI Workflow    |                     |     CD Workflow   |
    |  (push + PR)      |                     |  (push only)      |
    +-------------------+                     +-------------------+
    |                   |                     |                   |
    | +---------+       |                     | Wait for CI       |
    | | Lint    |       |                     |       |           |
    | +---------+       |                     |       v           |
    |                   |                     | Build & Push      |
    | +---------+       |                     | Docker Image      |
    | | Type    |       |                     |       |           |
    | | Check   |       |                     |       v           |
    | +---------+       |                     | Deploy to         |
    |                   |                     | Cloud Run         |
    | +---------+       |                     |       |           |
    | | Tests   |       |                     |       v           |
    | | 3.11    |       |                     | Smoke Test        |
    | | 3.12    |       |                     +-------------------+
    | +---------+       |
    |                   |
    | +---------+       |      +----------------------------+
    | |Security |       |      | Model Training Workflow    |
    | | Scan   |       |      | (manual or data change)    |
    | +---------+       |      +----------------------------+
    |                   |      |                            |
    | +---------+       |      | Upload Data Verification   |
    | | Docker  | (PR)  |      |       |                    |
    | | Build   |       |      | Build Training Image       |
    | +---------+       |      |       |                    |
    +-------------------+      |       v                    |
                               | Vertex AI Training Job     |
                               |       |                    |
                               |       v                    |
                               | Evaluate Model             |
                               |       |                    |
                               |   [quality gate]           |
                               |    /         \             |
                               |  pass       fail           |
                               |   |           |            |
                               |   v           v            |
                               | Deploy      Stop           |
                               +----------------------------+
```

---

## Required Secrets

Configure these in **Settings > Secrets and variables > Actions** in your
GitHub repository.

| Secret | Description | Example |
|---|---|---|
| `GCP_SA_KEY` | GCP service account key (JSON). Must have permissions for Cloud Run, Artifact Registry, GCS, and Vertex AI. | `{"type": "service_account", ...}` |
| `GCP_PROJECT_ID` | GCP project ID. | `ai-product-detector-487013` |
| `GCS_BUCKET` | GCS bucket name for data and model storage. | `ai-product-detector-487013` |
| `API_KEY` | API key(s) for the deployed inference service. Set as `API_KEYS` environment variable on Cloud Run. | `sk-abc123` |

### Service Account Permissions

The service account referenced by `GCP_SA_KEY` requires the following IAM roles:

- `roles/run.admin` -- deploy and manage Cloud Run services
- `roles/artifactregistry.writer` -- push Docker images
- `roles/storage.objectAdmin` -- read/write GCS objects (data, models)
- `roles/aiplatform.user` -- submit Vertex AI training jobs
- `roles/iam.serviceAccountUser` -- act as the Cloud Run service account

---

## Workflow: PR Preview

**File:** `.github/workflows/pr-preview.yml`

The PR Preview workflow deploys a temporary preview environment for each pull
request, allowing reviewers to test changes before merging.

### Trigger

- Runs on pull request open, synchronize (new commits), and reopen events.

### Features

- **Ephemeral environment:** Each PR gets its own Cloud Run revision.
- **Automatic cleanup:** Preview environments are deleted when the PR is closed.
- **Comment integration:** Posts the preview URL as a PR comment.

---

## Triggering Workflows

### CI (automatic)

Triggers automatically on every push or pull request to `main`. No manual
action required.

### CD (automatic + manual)

- **Automatic:** Triggers on push to `main` (after CI passes).
- **Manual:** Go to **Actions > CD > Run workflow**. Optionally specify:
  - An image tag for rollback (e.g., a previous commit SHA).
  - A memory allocation override.

### Model Training (manual + automatic)

- **Manual:** Go to **Actions > Model Training (Vertex AI) > Run workflow**.
  Configure epochs, batch size, and whether to auto-deploy.
- **Automatic:** Triggers on push to `main` when files under `data/**` are
  modified.

### PR Preview (automatic)

Triggers automatically when a pull request is opened or updated. The preview
URL is posted as a comment on the PR.

---

## Modifying Workflows

### Adding a new CI check

1. Add a new job in `.github/workflows/ci.yml`.
2. If the CD workflow should wait for it, update the `check-regexp` pattern in
   the `ci-check` job of `cd.yml`.

### Changing deployment configuration

- **Region/service name:** Update the `env` block at the top of `cd.yml` and
  `model-training.yml`.
- **Cloud Run settings:** Modify the `gcloud run deploy` command arguments in
  the `deploy` job.

### Changing training configuration

- **Machine type/GPU:** Update the `MACHINE_TYPE`, `ACCELERATOR_TYPE`, and
  `ACCELERATOR_COUNT` environment variables in `model-training.yml`.
- **Hyperparameters:** Modify `configs/train_config.yaml` or override via
  workflow dispatch inputs.

### Adding environment-specific deployments

To add staging/production environments, duplicate the `deploy` job with
different Cloud Run service names and create separate GitHub Environments with
approval rules.

---

## Branch Protection Rules

The following branch protection settings are recommended for the `main` branch:

| Setting | Recommended Value |
|---|---|
| Require pull request before merging | Yes |
| Required approvals | 1 (minimum) |
| Require status checks to pass | Yes |
| Required status checks | `Lint & Format Check`, `Tests (Python 3.11)`, `Tests (Python 3.12)`, `Type Checking (mypy)` |
| Require branches to be up to date | Yes |
| Require conversation resolution | Yes |
| Do not allow bypassing the above settings | Project preference |

### Setup

1. Go to **Settings > Branches > Add rule**.
2. Set branch name pattern to `main`.
3. Enable the settings listed above.
4. Select the required status checks from the list (they appear after the first
   CI run).
