# CI/CD Pipeline Documentation

This project uses four GitHub Actions workflows to automate code quality checks, application deployment, model training, and PR previews. All workflow definitions live in `.github/workflows/`.

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
| CI | `.github/workflows/ci.yml` | Push/PR to `main` | Lint, type check, test, security scan, CodeQL analysis |
| CD | `.github/workflows/cd.yml` | Push to `main` or manual dispatch | Build Docker image, push to Artifact Registry, deploy to Cloud Run, smoke test, automatic rollback |
| Model Training | `.github/workflows/model-training.yml` | Manual dispatch or data changes on `main` | Train on Vertex AI, evaluate, conditionally deploy |
| PR Preview | `.github/workflows/pr-preview.yml` | PR open/update | Deploy preview environment for testing |

All workflows use concurrency groups to prevent duplicate runs. None cancel in-progress runs, to avoid interrupted deployments or inconsistent results.

### Paths-Ignore (Docs-Only Changes)

Both CI and CD workflows use `paths-ignore` to skip runs when only non-code files are modified. This avoids unnecessary pipeline runs for documentation, image, or media changes. The ignored patterns include:

- Markdown files (`*.md`, `**/*.md`)
- Images and media (`*.gif`, `*.mp4`, `*.png`, `*.jpg`, `*.jpeg`, `*.svg`)
- Documents (`*.pptx`, `*.docx`, `*.pdf`, `*.txt`)
- The `docs/` and `images/` directories
- The `LICENSE` file

When a push contains only these files, neither CI nor CD will trigger.

---

## Workflow: CI

**File:** `.github/workflows/ci.yml`

The CI workflow runs on every push and pull request targeting `main` (subject to `paths-ignore` filtering). It validates code quality, correctness, and security.

### Permissions

- `contents: read` -- checkout code.
- `security-events: write` -- required by CodeQL to upload SARIF results.

### Jobs

#### 1. Lint, Type Check and Tests (`ci`)

A single consolidated job that runs linting, type checking, and tests sequentially for fast feedback.

- **Runner:** `ubuntu-latest`, Python 3.11
- **Caching:** pip dependencies cached with a key based on `pyproject.toml`.
- **Steps:**
  1. Install project with dev dependencies (`pip install -e ".[dev]"`).
  2. **Lint and format** -- `ruff check src/ tests/` and `ruff format --check src/ tests/`.
  3. **Type check** -- `mypy src/ --ignore-missing-imports`.
  4. **Tests with coverage:**
     ```bash
     pytest tests/ -v --tb=short \
       --cov=src \
       --cov-report=term-missing \
       --cov-report=xml:coverage.xml \
       --junitxml=test-results.xml
     ```
     The `RATE_LIMIT_ENABLED` environment variable is set to `false` during tests.
  5. **Upload test artifacts** -- `coverage.xml` and `test-results.xml` (JUnit XML), retained for 14 days. Runs even if tests fail (`if: always()`).
  6. **Coverage summary** -- A coverage badge and Markdown coverage table are posted to the GitHub Actions job summary.

#### 2. Security Scan (`security`)

- **Runner:** `ubuntu-latest`, Python 3.11
- **Job-level `continue-on-error: true`** -- findings are reported but never block the pipeline.
- **Tools:**
  - [pip-audit](https://pypi.org/project/pip-audit/) -- checks installed dependencies for known vulnerabilities.
  - [bandit](https://bandit.readthedocs.io/) -- static analysis for common security issues. Skips rules: `B101` (assert), `B601` (shell injection), `B104` (binding all interfaces), `B614`, `B404` (subprocess import), `B607` (partial path), `B603` (subprocess call), `B108` (hardcoded tmp).
- Both steps use `|| true` so individual tool failures do not fail the job.

#### 3. CodeQL Analysis (`codeql`)

- **Runner:** `ubuntu-latest`
- **Permissions:** `security-events: write`
- **Tool:** [GitHub CodeQL](https://codeql.github.com/)
- **Steps:**
  1. Initialize CodeQL for Python with `security-and-quality` query suite.
  2. Run CodeQL analysis and upload results under category `/language:python`.
- Results appear in the repository's **Security > Code scanning alerts** tab.

---

## Workflow: CD

**File:** `.github/workflows/cd.yml`

The CD workflow builds the application Docker image, pushes it to Google Artifact Registry, deploys to Cloud Run, runs a multi-endpoint smoke test, and performs automatic rollback if the smoke test fails.

Like CI, the CD workflow uses `paths-ignore` to skip docs-only changes.

### Environment Variables

| Variable | Value | Description |
|---|---|---|
| `REGION` | `europe-west1` | GCP region for deployment |
| `SERVICE` | `ai-product-detector` | Cloud Run service name |
| `REGISTRY` | `europe-west1-docker.pkg.dev` | Artifact Registry hostname |
| `IMAGE` | `europe-west1-docker.pkg.dev/ai-product-detector-487013/ai-product-detector/api` | Full image path |

### Jobs

#### 1. Wait for CI (`ci-check`)

- **Condition:** Only on `push` events (skipped for manual dispatch).
- **Smart skip logic:** Before waiting, checks whether the CI workflow was actually triggered for this commit. If CI was not triggered (e.g., docs-only change that passed `paths-ignore`), the wait is skipped entirely. This prevents the CD pipeline from hanging indefinitely when CI did not run.
- Uses `lewagon/wait-on-check-action@v1.3.4` to wait for the CI job `Lint, Type Check & Tests` to complete.
- **Configuration:** `check-regexp: "^Lint, Type Check & Tests"`, polling every 15 seconds. Accepts `success` or `skipped` conclusions.

#### 2. Build and Push Docker Image (`build`)

- **Depends on:** `ci-check` (success) or `workflow_dispatch`.
- **Condition:** `always() && (needs.ci-check.result == 'success' || github.event_name == 'workflow_dispatch')`.
- **Steps:**
  1. Authenticate to GCP using the `GCP_SA_KEY` secret.
  2. Configure Docker to authenticate with Artifact Registry.
  3. Fetch the model checkpoint:
     - **Strategy 1:** Direct download from GCS.
     - **Strategy 2 (fallback):** `dvc pull` from DVC remote.
     - Fails the build if no model checkpoint is available.
  4. Determine image tag: commit SHA for new builds, or a user-specified tag for rollbacks via manual dispatch. If a specific tag is provided, the Docker build is skipped (existing image is reused).
  5. Build and push the image with both the SHA tag and `latest`.
- **Output:** `image_tag` (the tag used for deployment).

#### 3. Deploy to Production (`deploy-production`)

- **Depends on:** `build`.
- **Condition:** `always() && needs.build.result == 'success'`.
- **Steps:**
  1. **Capture current revision** -- records the currently deployed revision name for potential rollback.
  2. **Deploy** using `gcloud run deploy` with the built image.
  3. Configuration: port 8080, configurable memory (default 1Gi), unauthenticated access enabled.
  4. Environment variables set on Cloud Run: `REQUIRE_AUTH=false`, `ENVIRONMENT=production`.
  5. **Get production URL** -- retrieves the service URL after deployment.
- **Environment:** `production` (creates a GitHub Environments entry with the deployment URL).
- **Outputs:** `url`, `previous_revision`.

#### 4. Smoke Test (`smoke-test-production`)

- **Depends on:** `deploy-production`.
- Waits 20 seconds for the service to stabilize, then runs three tests:
  1. `GET /health` -- must return HTTP 200.
  2. `GET /docs` -- must return HTTP 200.
  3. `POST /predict` -- must not return a 5xx error (client errors like 422 are acceptable).
- **Output:** `status` (success/failure).
- If any test fails, the job exits with a non-zero code, triggering the rollback job.

#### 5. Rollback Production (`rollback`)

- **Depends on:** `deploy-production`, `smoke-test-production`.
- **Condition:** `failure() && needs.smoke-test-production.result == 'failure'`.
- **Steps:**
  1. Authenticate to GCP.
  2. Route 100% of traffic back to the previous revision using `gcloud run services update-traffic`.
  3. Wait 10 seconds, then verify the rollback by checking `/health`.
- If no previous revision exists (first deployment), the rollback is skipped with an error.

#### 6. Deployment Notification (`notify`)

- **Depends on:** `build`, `deploy-production`, `smoke-test-production`.
- **Condition:** `always()` (runs regardless of prior job outcomes).
- Posts a summary table to the GitHub Actions job summary with the result of each pipeline stage (build, deploy, smoke test), the commit SHA, the actor, and the image tag.

### Manual Dispatch Inputs

| Input | Type | Default | Description |
|---|---|---|---|
| `image_tag` | string | `latest` | Image tag to deploy (commit SHA or `latest` to build fresh) |
| `memory` | choice | `1Gi` | Cloud Run memory allocation (`512Mi`, `1Gi`, `2Gi`) |

---

## Workflow: Model Training

**File:** `.github/workflows/model-training.yml`

This workflow orchestrates end-to-end model training on Vertex AI, evaluates the resulting model, and conditionally deploys it to production.

### Environment Variables

| Variable | Description |
|---|---|
| `GCP_PROJECT` | GCP project ID |
| `REGION` | GCP region |
| `GCS_BUCKET` | GCS bucket for data and models |
| `MACHINE_TYPE` | Vertex AI machine type |
| `ACCELERATOR_TYPE` | GPU type |
| `ACCELERATOR_COUNT` | Number of GPUs |

### Jobs

#### 1. Upload Training Data

- Verifies that training data exists on GCS.
- Posts a file count summary to the job summary.

#### 2. Build Training Image

- Builds `docker/Dockerfile.training` (GPU-enabled PyTorch base image).
- Pushes to Artifact Registry with `latest` and `<SHA>` tags.
- **Output:** `image_uri` for use by the training job.

#### 3. Submit Vertex AI Training Job

- **Depends on:** `upload-data`, `build-training-image`.
- **Timeout:** 180 minutes.
- Submits a `CustomContainerTrainingJob` to Vertex AI using the Python SDK.
- Configuration:
  - Machine: `n1-standard-4` with 1x NVIDIA Tesla T4 GPU.
  - Command: `python -m src.training.train --config configs/train_config.yaml`.
  - Environment variables passed: `EPOCHS`, `BATCH_SIZE`, `GCS_DATA_PATH`, `GCS_MODEL_OUTPUT`.
- After training completes:
  1. Downloads `best_model.pt` from the training output directory.
  2. Copies it to the canonical GCS location.
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
- Builds a new inference image with the trained model baked in, pushes it, deploys to Cloud Run, and runs a smoke test.

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
                          (paths-ignore filter)
                                   |
              +--------------------+--------------------+
              |                                         |
              v                                         v
    +-------------------+                     +-------------------+
    |     CI Workflow    |                     |     CD Workflow   |
    |  (push + PR)      |                     |  (push only)      |
    +-------------------+                     +-------------------+
    |                   |                     |                   |
    | +-------------+   |                     | Wait for CI       |
    | | ci          |   |                     | (skip if CI not   |
    | | - lint      |   |                     |  triggered)       |
    | | - typecheck |   |                     |       |           |
    | | - tests     |   |                     |       v           |
    | +-------------+   |                     | Build & Push      |
    |                   |                     | Docker Image      |
    | +-------------+   |                     |       |           |
    | | security    |   |                     |       v           |
    | | (non-block) |   |                     | Deploy to         |
    | +-------------+   |                     | Production        |
    |                   |                     |       |           |
    | +-------------+   |                     |       v           |
    | | codeql      |   |                     | Smoke Test        |
    | | (background)|   |                     | (health/docs/     |
    | +-------------+   |                     |  predict)         |
    +-------------------+                     |       |           |
                                              |   fail?---> Rollback
                                              |       |           |
                                              |       v           |
                                              |    Notify         |
                                              +-------------------+

    +----------------------------+
    | Model Training Workflow    |
    | (manual or data change)    |
    +----------------------------+
    |                            |
    | Upload Data Verification   |
    |       |                    |
    | Build Training Image       |
    |       |                    |
    |       v                    |
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

Configure these in **Settings > Secrets and variables > Actions** in your GitHub repository.

| Secret | Description |
|---|---|
| `GCP_SA_KEY` | GCP service account key (JSON). Must have permissions for Cloud Run, Artifact Registry, GCS, and Vertex AI. |
| `GCP_PROJECT_ID` | GCP project ID. |
| `GCS_BUCKET` | GCS bucket name for data and model storage. |
| `API_KEY` | API key(s) for the deployed inference service. |

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

The PR Preview workflow deploys a temporary preview environment for each pull request, allowing reviewers to test changes before merging.

### Trigger

- Runs on pull request open, synchronize (new commits), and reopen events.

### Features

- **Ephemeral environment:** Each PR gets its own Cloud Run revision.
- **Automatic cleanup:** Preview environments are deleted when the PR is closed.
- **Comment integration:** Posts the preview URL as a PR comment.

---

## Triggering Workflows

### CI (automatic)

Triggers automatically on every push or pull request to `main`, unless the change only modifies files matched by `paths-ignore` (documentation, images, media). No manual action required.

### CD (automatic + manual)

- **Automatic:** Triggers on push to `main` (subject to `paths-ignore`). Waits for CI to pass before deploying, or skips the CI wait if CI was not triggered.
- **Manual:** Go to **Actions > CD > Run workflow**. Optionally specify:
  - An image tag for rollback (e.g., a previous commit SHA).
  - A memory allocation override.

### Model Training (manual + automatic)

- **Manual:** Go to **Actions > Model Training (Vertex AI) > Run workflow**. Configure epochs, batch size, and whether to auto-deploy.
- **Automatic:** Triggers on push to `main` when files under `data/**` are modified.

### PR Preview (automatic)

Triggers automatically when a pull request is opened or updated. The preview URL is posted as a comment on the PR.

---

## Modifying Workflows

### Adding a new CI check

1. Add a new job in `.github/workflows/ci.yml`.
2. If the CD workflow should wait for it, update the `check-regexp` pattern in the `ci-check` job of `cd.yml`. The current pattern is `^Lint, Type Check & Tests`.

### Changing deployment configuration

- **Region/service name:** Update the `env` block at the top of `cd.yml` and `model-training.yml`.
- **Cloud Run settings:** Modify the `gcloud run deploy` command arguments in the `deploy-production` job.

### Changing training configuration

- **Machine type/GPU:** Update the `MACHINE_TYPE`, `ACCELERATOR_TYPE`, and `ACCELERATOR_COUNT` environment variables in `model-training.yml`.
- **Hyperparameters:** Modify `configs/train_config.yaml` or override via workflow dispatch inputs.

### Adding environment-specific deployments

To add staging/production environments, duplicate the `deploy-production` job with different Cloud Run service names and create separate GitHub Environments with approval rules.

---

## Branch Protection Rules

The following branch protection settings are recommended for the `main` branch:

| Setting | Recommended Value |
|---|---|
| Require pull request before merging | Yes |
| Required approvals | 1 (minimum) |
| Require status checks to pass | Yes |
| Required status checks | `Lint, Type Check & Tests` |
| Require branches to be up to date | Yes |
| Require conversation resolution | Yes |
| Do not allow bypassing the above settings | Project preference |

**Note:** The CI pipeline uses a single consolidated job named `Lint, Type Check & Tests`. The `Security Scan` and `CodeQL Analysis` jobs are informational and should not be required status checks.

### Setup

1. Go to **Settings > Branches > Add rule**.
2. Set branch name pattern to `main`.
3. Enable the settings listed above.
4. Select the required status check `Lint, Type Check & Tests` from the list (it appears after the first CI run).
