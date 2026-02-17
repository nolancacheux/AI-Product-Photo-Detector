# Training Guide

This document covers all three methods for training the AI Product Photo Detector model.

---

## Table of Contents

1. [Training Modes Overview](#training-modes-overview)
2. [Model Architecture](#model-architecture)
3. [Dataset](#dataset)
4. [Mode 1: Local Training](#mode-1-local-training)
5. [Mode 2: Google Colab](#mode-2-google-colab)
6. [Mode 3: Vertex AI](#mode-3-vertex-ai)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Evaluation](#evaluation)
9. [Updating the Deployed Model](#updating-the-deployed-model)
10. [Troubleshooting](#troubleshooting)

---

## Training Modes Overview

| Mode | GPU | Cost | Time | Best For |
|------|-----|------|------|----------|
| **Local Training** | CPU (or local GPU) | Free | 1-2h | Development, debugging, quick tests |
| **Google Colab** | Free T4/A100 | Free | ~20 min | Experiments, prototyping |
| **Vertex AI** | Configurable (T4 or CPU fallback) | ~$0.10-0.50/run | ~25 min | Production training, CI/CD |

### Decision Tree

```
                    Which training mode?
                              |
                              v
              Need GPU for fast training?
                     |                 |
                    Yes               No
                     |                 |
                     v                 v
        Production model?       LOCAL TRAINING
           |           |        make train
          Yes         No
           |           |
           v           v
       VERTEX AI    GOOGLE COLAB
       CI/CD        Free T4/A100
```

---

## Model Architecture

The detector uses an **EfficientNet-B0** backbone with a custom binary classification head.

**Source:** `src/training/model.py`

```
Input Image (3 x 224 x 224)
        |
        v
+-------------------+
| EfficientNet-B0   |   Pretrained on ImageNet (via timm)
| (backbone)        |   num_classes=0 removes original head
| Feature dim: 1280 |   Global average pooling built-in
+--------+----------+
         |
         v  [1280]
+-------------------+
| Linear(1280, 512) |
| BatchNorm1d(512)  |
| ReLU              |
| Dropout(0.3)      |
| Linear(512, 1)    |   Raw logit output
+-------------------+
         |
         v  [1]
   BCEWithLogitsLoss     (training)
   Sigmoid               (inference via predict_proba)
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| EfficientNet-B0 | Good accuracy-to-size ratio; small enough for Cloud Run |
| Pretrained backbone (timm) | Transfer learning from ImageNet reduces training time |
| BatchNorm in classifier | Stabilizes training, especially with small batch sizes |
| BCEWithLogitsLoss | Numerically stable; model outputs raw logits |
| Dropout 0.3 | Regularization to prevent overfitting on small datasets |

### Parameters

- **Total parameters:** ~5.3M (EfficientNet-B0 backbone + classifier head)
- **Trainable parameters:** ~5.3M (full fine-tuning by default)
- **Optional:** Set `freeze_backbone=True` in `create_model()` to freeze the backbone and train only the classifier head (~660K trainable parameters). Note: currently not exposed as a config option -- requires code change.

---

## Dataset

### CIFAKE Dataset

The primary dataset is [CIFAKE: Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images), containing real photographs and AI-generated counterparts.

### Directory Structure

```
data/processed/
|-- train/
|   |-- real/
|   +-- ai_generated/
|-- val/
|   |-- real/
|   +-- ai_generated/
+-- test/
    |-- real/
    +-- ai_generated/
```

- **Labels:** `real/` = 0, `ai_generated/` = 1
- **Supported formats:** `.jpg`, `.jpeg`, `.png`, `.webp`

### HuggingFace Alternatives

- `emirhanbilgic/cifake-real-and-ai-generated-synthetic-images`
- `jlbaker361/CIFake`

### Data Augmentation

Defined in `src/training/augmentation.py`:

**Training transforms** (applied in order):

| Transform | Details |
|-----------|---------|
| Resize | To (256, 256) -- slightly larger than target |
| Random crop | 224 x 224 |
| Horizontal flip | p=0.5 |
| Random rotation | +/-15 degrees |
| Color jitter | brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05 |
| ToTensor + Normalize | ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) |
| Random erasing | p=0.1 (applied after normalization) |

**Validation/test transforms:**

| Transform | Details |
|-----------|---------|
| Resize | Directly to (224, 224) |
| ToTensor + Normalize | ImageNet statistics |

No augmentation is applied to validation or test sets -- only resize and normalization.

---

## Mode 1: Local Training

**When to use:** Development, debugging, quick experiments, and iterating on model changes without cloud costs.

### Prerequisites

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- GPU optional (CUDA or Apple MPS); CPU works but is slower
- Docker & Docker Compose (for full stack)

### Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/nolancacheux/AI-Product-Photo-Detector.git
cd AI-Product-Photo-Detector

# 2. Install dependencies
make dev

# 3. Download dataset
make data

# 4. Train (CPU)
make train
```

### Training Commands

```bash
# Default configuration
python -m src.training.train --config configs/train_config.yaml

# Override epochs and batch size via CLI
python -m src.training.train --config configs/train_config.yaml \
  --epochs 10 \
  --batch-size 32

# With GCS integration (downloads data from GCS if missing locally,
# uploads model + MLflow artifacts to GCS after training)
python -m src.training.train --config configs/train_config.yaml \
  --gcs-bucket ai-product-detector-487013
```

**Available CLI flags for `src.training.train`:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | str | `configs/train_config.yaml` | Path to training config |
| `--gcs-bucket` | str | None | GCS bucket for remote data/model storage |
| `--epochs` | int | None | Override epochs from config |
| `--batch-size` | int | None | Override batch size from config |

Note: learning rate and other hyperparameters must be changed in `configs/train_config.yaml`.

### DVC Pipeline

The DVC pipeline defines three stages: `download`, `validate`, and `train`.

**Source:** `dvc.yaml`

```bash
# Run full pipeline (download -> validate -> train)
make dvc-repro

# Or with DVC directly
dvc repro

# Run specific stage
dvc repro train

# Check pipeline status
dvc status
```

**DVC stages:**

| Stage | Command | Outputs |
|-------|---------|---------|
| `download` | `python scripts/download_cifake.py` | `data/processed/` |
| `validate` | `python -m src.data.validate` | `reports/data_validation.json` |
| `train` | `python -m src.training.train --config configs/train_config.yaml` | `models/checkpoints/best_model.pt` |

### MLflow Tracking

Training metrics are logged to MLflow. The config file (`configs/train_config.yaml`) sets `tracking_uri: "mlruns"` for local file-based storage. To view results in the MLflow UI:

```bash
# Start MLflow UI (reads from local mlruns/ directory)
make mlflow
# Open http://localhost:5000
```

**Logged Parameters:**

All training configuration values are logged at the start of each run, including model name, learning rate, weight decay, batch size, image size, epochs, seed, dropout, optimizer (AdamW), scheduler (CosineAnnealingLR), device, and GCS bucket.

**Logged Metrics (per epoch):**

| Metric | Description |
|--------|-------------|
| `train_loss` | Training loss |
| `train_accuracy` | Training accuracy |
| `val_loss` | Validation loss |
| `val_accuracy` | Validation accuracy |
| `val_precision` | Validation precision |
| `val_recall` | Validation recall |
| `val_f1` | Validation F1 score |
| `learning_rate` | Current learning rate (from scheduler) |

**Logged Artifacts:**

- `best_model.pt` -- best model checkpoint (logged on each improvement)
- Full PyTorch model via `mlflow.pytorch.log_model()` (if `mlflow.log_models: true` in config)
- Training config YAML file

### Full Stack Development

```bash
# Start all services (API + UI + MLflow + Prometheus + Grafana)
make docker-up

# Service URLs:
# API:        http://localhost:8080
# Streamlit:  http://localhost:8501
# MLflow:     http://localhost:5000
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000

# Watch logs
make docker-logs

# Stop
make docker-down
```

### Code Quality Checks

```bash
make lint          # ruff + mypy
make format        # Auto-format with ruff
make test          # pytest with coverage
```

### Output

The best model checkpoint is saved to `models/checkpoints/best_model.pt`:

```python
{
    "epoch": 12,                     # Epoch number (0-indexed)
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "val_accuracy": 0.92,
    "best_val_accuracy": 0.92,
    "config": {...},                 # Full training config dict
}
```

---

## Mode 2: Google Colab

**When to use:** Free GPU training for experiments and prototyping without local GPU hardware or cloud costs.

### Prerequisites

- Google account
- (Optional) GCS bucket for model storage

### Quick Start

1. **Open the notebook:**
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nolancacheux/AI-Product-Photo-Detector/blob/main/notebooks/train_colab.ipynb)

2. **Select GPU runtime:**
   - Go to **Runtime -> Change runtime type -> T4 GPU** (or A100)

3. **Run all cells:**
   - The notebook handles setup, data loading, training, and export

4. **Export model:**
   - Download from Colab file browser, or
   - Auto-upload to GCS bucket

### Notebook Structure

**File:** `notebooks/train_colab.ipynb`

| Section | Description |
|---------|-------------|
| **1. Environment Setup** | Install PyTorch, timm, dependencies |
| **2. GCS Authentication** | Optional: authenticate for data/model storage |
| **3. Data Loading** | Download from HuggingFace or mount GCS |
| **4. Model Definition** | EfficientNet-B0 with custom head |
| **5. Training Loop** | Training with progress bars and metrics |
| **6. Evaluation** | Test set metrics and confusion matrix |
| **7. Export** | Save checkpoint, upload to GCS |

### Configuration

```python
CONFIG = {
    "epochs": 15,
    "batch_size": 64,        # T4 handles 64; reduce to 32 if OOM
    "learning_rate": 0.001,
    "image_size": 224,
    "num_workers": 2,

    # GCS Integration (optional)
    "gcs_bucket": "<YOUR-GCS-BUCKET>",
    "gcs_data_path": "data/processed/",
    "gcs_model_path": "models/colab_trained.pt",
}
```

### Data Loading Options

**Option 1: HuggingFace Datasets (recommended)**
```python
from datasets import load_dataset
dataset = load_dataset("emirhanbilgic/cifake-real-and-ai-generated-synthetic-images")
```

**Option 2: Google Cloud Storage**
```python
from google.colab import auth
auth.authenticate_user()
!gsutil -m cp -r gs://<YOUR-GCS-BUCKET>/data/processed/ ./data/
```

**Option 3: Google Drive Mount**
```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/AI-Product-Photo-Detector/data/processed/ ./data/
```

### Export Options

**Option 1: Download to local machine**
```python
from google.colab import files
files.download('models/checkpoints/best_model.pt')
```

**Option 2: Upload to GCS**
```python
!gsutil cp models/checkpoints/best_model.pt gs://<YOUR-GCS-BUCKET>/models/
```

**Option 3: Save to Google Drive**
```python
!cp models/checkpoints/best_model.pt /content/drive/MyDrive/AI-Product-Photo-Detector/models/
```

### Expected Performance

| GPU | Batch Size | Time/Epoch | Total (15 epochs) |
|-----|------------|------------|-------------------|
| T4 | 64 | ~1.5 min | ~20-25 min |
| A100 | 64 | ~0.5 min | ~8-10 min |

### Tips and Troubleshooting

| Issue | Solution |
|-------|----------|
| **Session timeout** | Save checkpoints to Google Drive periodically |
| **OOM errors** | Reduce `batch_size` to 32 or 16 |
| **Slow data loading** | Use HuggingFace datasets (pre-cached) |
| **Need more GPU time** | Use Colab Pro for longer sessions |
| **Dataset not found** | Try alternative HuggingFace datasets |

---

## Mode 3: Vertex AI

There are two approaches for Vertex AI training, depending on the use case:

1. **GitHub Actions workflow** (`model-training.yml`) -- the primary production method, triggered manually or on data changes.
2. **Kubeflow Pipelines** (`src/pipelines/training_pipeline.py`) -- a full KFP pipeline with data validation, training, evaluation, model comparison, registration, and deployment stages.

Both share the same core training code (`src/training/train.py`).

### Prerequisites

- GCP project with billing enabled (`ai-product-detector-487013`)
- Service account with Vertex AI, GCS, Artifact Registry, and Cloud Run permissions
- Training data uploaded to GCS (`gs://ai-product-detector-487013/data/processed/`)
- GitHub Actions secrets configured (see [CICD.md](CICD.md)): `GCP_SA_KEY`, `GCP_PROJECT_ID`, `GCS_BUCKET`, `API_KEY`

---

### Approach 1: GitHub Actions Workflow

This is the recommended production training method. The workflow verifies data on GCS, builds a training Docker image, submits a Vertex AI job, evaluates the result, and optionally deploys.

#### Architecture

```
+------------------------------------------------------------------+
|                    GitHub Actions Workflow                        |
+------------------------------------------------------------------+
|                                                                  |
|  +--------------+   +--------------+   +----------------------+  |
|  | Verify Data  | > | Build Image  | > | Submit Vertex AI Job |  |
|  | (GCS bucket) |   | (Artifact    |   | (T4 GPU or CPU       |  |
|  +--------------+   |  Registry)   |   |  fallback)           |  |
|                     +--------------+   +----------------------+  |
|                                                   |              |
|  +--------------+   +--------------+   +----------v-----------+  |
|  | Auto Deploy  | < | Quality Gate | < | Evaluate Model       |  |
|  | (Cloud Run)  |   | acc>=0.85    |   | (on GH runner, CPU)  |  |
|  +--------------+   | F1>=0.80     |   +----------------------+  |
|                     +--------------+                             |
+------------------------------------------------------------------+
```

#### Trigger Training

**Option 1: GitHub Actions UI**
1. Go to **Actions -> Model Training (Vertex AI) -> Run workflow**
2. Configure inputs:
   - `epochs`: 15 (default)
   - `batch_size`: 64 (default)
   - `auto_deploy`: false (set to true for automatic deployment)
   - `use_gpu`: true (falls back to CPU if GPU quota unavailable)
   - `region`: us-central1 (default, best GPU quota) or europe-west1/europe-west4/asia-east1

**Option 2: GitHub CLI**
```bash
gh workflow run model-training.yml \
  -f epochs=15 \
  -f batch_size=64 \
  -f auto_deploy=true \
  -f use_gpu=true \
  -f region=us-central1
```

#### Automatic Triggers

The workflow also runs on pushes to `main` that modify files under `data/**`:

```yaml
on:
  push:
    branches: [main]
    paths:
      - 'data/**'
  workflow_dispatch:
```

#### Workflow Stages

| Stage | Duration | Description |
|-------|----------|-------------|
| **Verify Data** | ~30s | Check GCS bucket for train/val/test data |
| **Build Image** | ~3-5 min | Build and push training image to Artifact Registry |
| **Submit Job** | ~20-30 min | Vertex AI CustomContainerTrainingJob (GPU or CPU) |
| **Evaluate** | ~2 min | Run evaluation on GitHub runner (CPU) |
| **Quality Gate** | ~10s | Check accuracy >= 0.85 and F1 >= 0.80 |
| **Deploy** | ~3 min | Deploy to Cloud Run (if auto_deploy=true and gate passes) |

#### Machine Configuration (GitHub Actions)

The workflow attempts GPU first, then falls back to CPU if quota is exhausted:

| Priority | Machine Type | GPU | Region |
|----------|-------------|-----|--------|
| 1st | `n1-standard-4` (4 vCPUs, 15 GB) | 1x NVIDIA Tesla T4 | Configured region |
| Fallback | `n1-standard-8` (8 vCPUs, 30 GB) | None (CPU-only) | Same region |

- **Training image:** `docker/Dockerfile.training` (base: `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`)
- **Timeout:** 180 minutes

---

### Approach 2: Kubeflow Pipeline (KFP)

A full Kubeflow Pipeline that includes data validation, training, evaluation, comparison against the production model, model registration in Vertex AI Model Registry, and conditional Cloud Run deployment.

**Source:** `src/pipelines/training_pipeline.py`

#### Pipeline Stages

```
validate_data -> train_model -> evaluate_model -> compare_models -> register_model -> deploy_model
```

| Stage | Description |
|-------|-------------|
| **validate_data** | Check class directories exist, minimum samples per class, class balance, image integrity |
| **train_model** | Launch Vertex AI CustomContainerTrainingJob (n1-standard-4 + T4 GPU) |
| **evaluate_model** | Download model + test data from GCS, compute accuracy/precision/recall/F1/AUC-ROC |
| **compare_models** | Compare new metrics against production model; gate on min accuracy and min F1 |
| **register_model** | Register in Vertex AI Model Registry if comparison passes |
| **deploy_model** | Update Cloud Run service (gated by `auto_deploy` flag) |

#### Usage

```bash
# Compile the pipeline to YAML
python -m src.pipelines.training_pipeline compile --output pipeline.yaml

# Submit a pipeline run to Vertex AI Pipelines
python -m src.pipelines.training_pipeline run \
  --config configs/pipeline_config.yaml \
  --epochs 15 \
  --batch-size 64 \
  --min-accuracy 0.85 \
  --auto-deploy
```

#### Pipeline Configuration

All pipeline parameters are in `configs/pipeline_config.yaml`:

| Section | Key Parameters |
|---------|---------------|
| **pipeline** | project_id, region (europe-west1), data/output GCS paths, training/serving images |
| **training** | epochs (15), batch_size (64), learning_rate (0.001), image_size (224) |
| **evaluation** | min_accuracy (0.85), min_f1 (0.80), min_samples_per_class (100) |
| **deployment** | auto_deploy (false), service_name, Cloud Run settings |

---

### Direct Vertex AI Submission (Simple)

For quick one-off training without the full KFP pipeline:

**Source:** `src/training/vertex_submit.py`

```bash
python -m src.training.vertex_submit \
  --epochs 15 \
  --batch-size 64 \
  --sync  # Wait for completion and download model
```

This script uploads data to GCS, builds and pushes the Docker image, and submits a `CustomContainerTrainingJob` using `e2-standard-2` (CPU-only). Use the GitHub Actions workflow or KFP pipeline for GPU training.

**Available flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 15 | Training epochs |
| `--batch-size` | 64 | Batch size |
| `--config` | `configs/train_config.yaml` | Config path inside container |
| `--sync` | false | Wait for completion and download model |
| `--skip-upload` | false | Skip uploading data to GCS |
| `--skip-build` | false | Skip building Docker image |

---

### Monitor Training

```bash
# List recent training jobs
gcloud ai custom-jobs list \
  --project=ai-product-detector-487013 \
  --region=europe-west1

# View job details
gcloud ai custom-jobs describe <JOB_ID> \
  --project=ai-product-detector-487013 \
  --region=europe-west1

# Stream logs
gcloud ai custom-jobs stream-logs <JOB_ID> \
  --project=ai-product-detector-487013 \
  --region=europe-west1
```

### Quality Gate

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| Accuracy | >= 0.85 | Overall correctness |
| F1 Score | >= 0.80 | Balance of precision and recall |

- **Pass:** Model is deployed to Cloud Run (if `auto_deploy` is enabled)
- **Fail:** Deployment is blocked; metrics saved to `reports/metrics.json`

For the KFP pipeline, an additional comparison against the production model is performed: the new model must also match or exceed the production model's accuracy and F1 to be registered.

### Cost Estimate

| Resource | Cost |
|----------|------|
| Vertex AI (T4, ~30 min) | ~$0.10-0.15 |
| Vertex AI (CPU fallback, ~60 min) | ~$0.05-0.10 |
| Artifact Registry (image push) | ~$0.01 |
| GCS (data transfer) | ~$0.01 |
| Cloud Run (deployment) | ~$0.00-0.05 |
| **Total per training run** | **~$0.10-0.25** |

---

## Hyperparameter Tuning

All hyperparameters are configured in `configs/train_config.yaml`.

### Default Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 0.001 | AdamW optimizer |
| Weight decay | 0.0001 | L2 regularization |
| Batch size | 64 | Reduce to 32 for smaller GPUs |
| Image size | 224 x 224 | EfficientNet-B0 native resolution |
| Epochs | 15 | With early stopping |
| Early stopping patience | 5 | Epochs without improvement |
| Scheduler | CosineAnnealingLR | Decays LR over T_max=epochs |
| Warmup epochs | 2 | Defined in config but not yet implemented in training loop |
| Dropout | 0.3 | In the classifier head |
| Seed | 42 | For reproducibility |
| Gradient clipping | max_norm=1.0 | Applied every training step |

### Tuning Recommendations

**If overfitting** (val loss increasing while train loss decreases):
- Increase dropout: 0.3 -> 0.5
- Increase weight decay: 0.0001 -> 0.001
- Add more aggressive augmentation
- Freeze the backbone (requires code change in `create_model()`)

**If underfitting** (both losses remain high):
- Increase learning rate: 0.001 -> 0.003
- Increase epochs: 15 -> 30
- Unfreeze the backbone if frozen
- Use a larger model: `efficientnet_b1` or `efficientnet_b2`

**If training is unstable** (loss spikes):
- Reduce learning rate: 0.001 -> 0.0003
- Gradient clipping is already applied (max norm = 1.0)
- Reduce batch size

**For faster iteration:**
- Freeze the backbone (code change in `create_model()`, train only classifier)
- Reduce epochs to 5-10 for quick experiments

---

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions / total |
| **Precision** | True positives / (true positives + false positives) |
| **Recall** | True positives / (true positives + false negatives) |
| **F1 Score** | Harmonic mean of precision and recall |
| **AUC-ROC** | Area under the ROC curve (KFP pipeline evaluation only) |

### Manual Evaluation

```bash
python -c "
import torch
from src.training.model import create_model
from src.training.dataset import create_dataloaders

model = create_model(pretrained=False)
checkpoint = torch.load('models/checkpoints/best_model.pt', map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

_, test_loader = create_dataloaders(
    train_dir='data/processed/train',
    val_dir='data/processed/test',
    batch_size=32,
    num_workers=2,
)

correct = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        labels = labels.float().unsqueeze(1)
        outputs = model(images)
        predicted = (outputs > 0.0).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f'Test accuracy: {correct / total:.4f}')
"
```

### Classification Thresholds

Configured in `configs/train_config.yaml` under `thresholds`:

| Threshold | Value | Usage |
|-----------|-------|-------|
| `classification` | 0.5 | Default decision boundary (applied after sigmoid) |
| `high_confidence` | 0.8 | High confidence predictions |
| `low_confidence` | 0.3 | Low confidence / uncertain |

Note: During training, the decision boundary is applied at 0.0 on raw logits, which is equivalent to 0.5 after sigmoid.

---

## Updating the Deployed Model

### Method 1: Vertex AI via GitHub Actions (Recommended)

Trigger the Model Training workflow with `auto_deploy: true`:

```bash
gh workflow run model-training.yml \
  -f epochs=15 \
  -f batch_size=64 \
  -f auto_deploy=true
```

If the quality gate passes, the pipeline automatically:
1. Builds a new inference Docker image with the model baked in
2. Deploys to Cloud Run
3. Runs a smoke test against the `/health` endpoint

### Method 2: KFP Pipeline with Auto-Deploy

```bash
python -m src.pipelines.training_pipeline run \
  --config configs/pipeline_config.yaml \
  --auto-deploy
```

If the new model passes both the quality gate and the comparison against the production model, it is registered in Vertex AI Model Registry and deployed to Cloud Run.

### Method 3: Manual Model Replacement

1. Train a model locally or in Colab
2. Upload to GCS:
   ```bash
   gsutil cp models/checkpoints/best_model.pt \
     gs://ai-product-detector-487013/models/best_model.pt
   ```
3. Trigger the CD workflow:
   ```bash
   gh workflow run cd.yml -f image_tag=latest
   ```

### Method 4: Direct Cloud Run Update

```bash
# Build and push new image
docker build -f docker/Dockerfile \
  -t europe-west1-docker.pkg.dev/ai-product-detector-487013/ai-product-detector/api:latest .
docker push europe-west1-docker.pkg.dev/ai-product-detector-487013/ai-product-detector/api:latest

# Deploy to Cloud Run
gcloud run deploy ai-product-detector \
  --image europe-west1-docker.pkg.dev/ai-product-detector-487013/ai-product-detector/api:latest \
  --region europe-west1
```

### Verifying the Update

```bash
# Health check
curl "https://ai-product-detector-714127049161.europe-west1.run.app/health"

# Test with a known image
curl -X POST "https://ai-product-detector-714127049161.europe-west1.run.app/predict" \
  -H "X-API-Key: <your-api-key>" \
  -F "file=@test_image.jpg"
```

---

## Troubleshooting

### Local Training Issues

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce `batch_size` to 32 or 16 in config |
| `Dataset not found` | Run `make data` to download CIFAKE |
| `MLflow connection error` | MLflow uses local `mlruns/` by default; start UI with `make mlflow` |
| `Import error` | Reinstall with `make dev` |
| `MPS backend error` (Apple Silicon) | Falls back to CPU automatically |

### Colab Issues

| Issue | Solution |
|-------|----------|
| Session disconnects | Save checkpoints to Drive periodically |
| GPU not available | Check runtime type is set to GPU |
| Package conflicts | Restart runtime after installing packages |

### Vertex AI Issues

| Issue | Solution |
|-------|----------|
| Job fails to start | Check service account permissions |
| GPU quota exceeded | Workflow falls back to CPU automatically; or try a different region |
| OOM during training | Reduce `batch_size` in workflow inputs |
| Quality gate fails | Lower thresholds or improve model |
| Deployment fails | Check Cloud Run quota and permissions |
| KFP pipeline fails | Check `configs/pipeline_config.yaml` paths match GCS layout |

### Common Checks

```bash
# Check GCS data
gsutil ls gs://ai-product-detector-487013/data/processed/

# Check Vertex AI quota
gcloud ai custom-jobs list --region=europe-west1 \
  --project=ai-product-detector-487013

# Check Cloud Run status
gcloud run services describe ai-product-detector \
  --region=europe-west1 \
  --project=ai-product-detector-487013
```
