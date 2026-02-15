# Training Guide

This document covers all three methods for training the AI Product Photo Detector model.

---

## Table of Contents

1. [Training Modes Overview](#training-modes-overview)
2. [Model Architecture](#model-architecture)
3. [Dataset](#dataset)
4. [Mode 1: Local Training](#mode-1-local-training)
5. [Mode 2: Google Colab](#mode-2-google-colab)
6. [Mode 3: Vertex AI Pipeline](#mode-3-vertex-ai-pipeline)
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
| **Vertex AI Pipeline** | Paid T4 | ~$0.10-0.50/run | ~25 min | Production training, CI/CD |

### Decision Tree

```
┌─────────────────────────────────────────────────────────────┐
│                    Which training mode?                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │   Need GPU for fast training? │
              └───────────────────────────────┘
                     │                 │
                    Yes               No
                     │                 │
                     ▼                 ▼
    ┌─────────────────────┐    ┌─────────────────────┐
    │  Production model?  │    │  LOCAL TRAINING     │
    └─────────────────────┘    │  make train         │
           │           │       └─────────────────────┘
          Yes         No
           │           │
           ▼           ▼
    ┌───────────┐  ┌───────────────┐
    │ VERTEX AI │  │ GOOGLE COLAB  │
    │ CI/CD     │  │ Free T4/A100  │
    └───────────┘  └───────────────┘
```

---

## Model Architecture

The detector uses an **EfficientNet-B0** backbone with a custom binary classification head.

**Source:** `src/training/model.py`

```
Input Image (3 x 224 x 224)
        │
        ▼
┌───────────────────┐
│ EfficientNet-B0   │   Pretrained on ImageNet
│ (backbone)        │   Global average pooling removes spatial dims
│ Feature dim: 1280 │
└────────┬──────────┘
         │
         ▼  [1280]
┌───────────────────┐
│ Linear(1280, 512) │
│ BatchNorm1d(512)  │
│ ReLU              │
│ Dropout(0.3)      │
│ Linear(512, 1)    │   Raw logit output
└───────────────────┘
         │
         ▼  [1]
   BCEWithLogitsLoss     (training)
   Sigmoid               (inference)
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| EfficientNet-B0 | Good accuracy-to-size ratio; small enough for Cloud Run |
| Pretrained backbone | Transfer learning from ImageNet reduces training time |
| BatchNorm in classifier | Stabilizes training, especially with small batch sizes |
| BCEWithLogitsLoss | Numerically stable; model outputs raw logits |
| Dropout 0.3 | Regularization to prevent overfitting on small datasets |

### Parameters

- **Total parameters:** ~5.3M (EfficientNet-B0 backbone + classifier head)
- **Trainable parameters:** ~5.3M (full fine-tuning by default)
- **Optional:** Set `freeze_backbone=True` to freeze the backbone and train only the classifier head (~660K trainable parameters).

---

## Dataset

### CIFAKE Dataset

The primary dataset is [CIFAKE: Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images), containing real photographs and AI-generated counterparts.

### Directory Structure

```
data/processed/
├── train/
│   ├── real/
│   └── ai_generated/
├── val/
│   ├── real/
│   └── ai_generated/
└── test/
    ├── real/
    └── ai_generated/
```

- **Labels:** `real/` = 0, `ai_generated/` = 1
- **Supported formats:** `.jpg`, `.jpeg`, `.png`, `.webp`

### HuggingFace Alternatives

- `emirhanbilgic/cifake-real-and-ai-generated-synthetic-images`
- `jlbaker361/CIFake`

### Data Augmentation

Defined in `src/training/augmentation.py`:

| Augmentation | Value |
|--------------|-------|
| Horizontal flip | Yes |
| Random rotation | ±15 degrees |
| Color jitter (brightness) | 0.2 |
| Color jitter (contrast) | 0.2 |
| Color jitter (saturation) | 0.1 |
| Color jitter (hue) | 0.05 |
| Random crop | Yes |

Validation and test sets use only resize + center crop + normalization (ImageNet statistics).

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

# Override epochs and batch size
python -m src.training.train --config configs/train_config.yaml \
  --epochs 10 \
  --batch-size 32 \
  --learning-rate 0.0005

# With GCS integration (uploads model after training)
python -m src.training.train --config configs/train_config.yaml \
  --gcs-bucket <YOUR-GCS-BUCKET>
```

### DVC Pipeline

```bash
# Run full pipeline (download → validate → train)
make dvc-repro

# Or with DVC directly
dvc repro

# Run specific stage
dvc repro train

# Check pipeline status
dvc status
```

### MLflow Tracking

```bash
# Start MLflow UI
make mlflow
# Open http://localhost:5000
```

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
| `learning_rate` | Current learning rate |

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
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "val_accuracy": 0.92,
    "best_val_accuracy": 0.92,
    "config": {...}
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
   - Go to **Runtime → Change runtime type → T4 GPU** (or A100)

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

### Tips & Troubleshooting

| Issue | Solution |
|-------|----------|
| **Session timeout** | Save checkpoints to Google Drive periodically |
| **OOM errors** | Reduce `batch_size` to 32 or 16 |
| **Slow data loading** | Use HuggingFace datasets (pre-cached) |
| **Need more GPU time** | Use Colab Pro for longer sessions |
| **Dataset not found** | Try alternative HuggingFace datasets |

---

## Mode 3: Vertex AI Pipeline

**When to use:** Production-grade GPU training with CI/CD integration and automated quality gates. Recommended for production model updates.

### Prerequisites

- GCP project with billing enabled
- Service account with Vertex AI and GCS permissions
- Training data uploaded to GCS
- GitHub Actions secrets configured (see [CICD.md](CICD.md))

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GitHub Actions Workflow                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │ Verify Data  │ → │ Build Image  │ → │ Submit Vertex AI Job │ │
│  │ (GCS bucket) │   │ (Artifact    │   │ (n1-standard-4 + T4) │ │
│  └──────────────┘   │  Registry)   │   └──────────────────────┘ │
│                     └──────────────┘              │              │
│                                                   ▼              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │ Auto Deploy  │ ← │ Quality Gate │ ← │ Evaluate Model       │ │
│  │ (Cloud Run)  │   │ acc≥0.85     │   │ (on GH runner)       │ │
│  └──────────────┘   │ F1≥0.80      │   └──────────────────────┘ │
│                     └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

### Trigger Training

**Option 1: GitHub Actions UI**
1. Go to **Actions → Model Training (Vertex AI) → Run workflow**
2. Configure inputs:
   - `epochs`: 15 (default)
   - `batch_size`: 64 (default)
   - `auto_deploy`: false (set to true for automatic deployment)

**Option 2: GitHub CLI**
```bash
gh workflow run model-training.yml \
  -f epochs=15 \
  -f batch_size=64 \
  -f auto_deploy=true
```

**Option 3: Direct Vertex AI Submission**
```bash
python -m src.training.vertex_submit \
  --epochs 15 \
  --batch-size 64 \
  --sync  # Wait for completion
```

### Pipeline Stages

| Stage | Duration | Description |
|-------|----------|-------------|
| **Verify Data** | ~30s | Check GCS bucket for training data |
| **Build Image** | ~3-5 min | Build and push training image to Artifact Registry |
| **Submit Job** | ~1 min | Submit CustomContainerTrainingJob to Vertex AI |
| **GPU Training** | ~20-25 min | Training on n1-standard-4 + T4 GPU |
| **Evaluate** | ~2 min | Run evaluation on GitHub runner |
| **Quality Gate** | ~10s | Check acc ≥ 0.85, F1 ≥ 0.80 |
| **Deploy** | ~3 min | Deploy to Cloud Run (if auto_deploy=true) |

### Machine Configuration

| Component | Specification |
|-----------|---------------|
| Machine type | `n1-standard-4` (4 vCPUs, 15 GB RAM) |
| GPU | 1x NVIDIA Tesla T4 (16 GB VRAM) |
| Region | `europe-west1` |
| Training image | `docker/Dockerfile.training` |
| Base image | `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime` |
| Timeout | 180 minutes |

### Monitor Training

```bash
# List recent training jobs
gcloud ai custom-jobs list \
  --project=<YOUR-PROJECT-ID> \
  --region=europe-west1

# View job details
gcloud ai custom-jobs describe <JOB_ID> \
  --project=<YOUR-PROJECT-ID> \
  --region=europe-west1

# Stream logs
gcloud ai custom-jobs stream-logs <JOB_ID> \
  --project=<YOUR-PROJECT-ID> \
  --region=europe-west1
```

### Quality Gate

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| Accuracy | ≥ 0.85 | Overall correctness |
| F1 Score | ≥ 0.80 | Balance of precision/recall |

- **Pass:** Model is deployed to Cloud Run (if `auto_deploy=true`)
- **Fail:** Deployment is blocked; metrics saved to `reports/metrics.json`

### Automatic Triggers

```yaml
on:
  push:
    branches: [main]
    paths:
      - 'data/**'  # Data changes trigger retraining
  workflow_dispatch:  # Manual trigger
```

### Cost Estimate

| Resource | Cost |
|----------|------|
| Vertex AI (T4, ~30 min) | ~$0.10-0.15 |
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
| Image size | 224 × 224 | EfficientNet-B0 native resolution |
| Epochs | 15 | With early stopping |
| Early stopping patience | 5 | Epochs without improvement |
| Scheduler | Cosine annealing | Decays LR to near zero |
| Warmup epochs | 2 | Gradual LR increase at start |
| Dropout | 0.3 | In the classifier head |
| Seed | 42 | For reproducibility |

### Tuning Recommendations

**If overfitting** (val loss increasing while train loss decreases):
- Increase dropout: 0.3 → 0.5
- Increase weight decay: 0.0001 → 0.001
- Add more aggressive augmentation
- Freeze the backbone: `freeze_backbone: true`

**If underfitting** (both losses remain high):
- Increase learning rate: 0.001 → 0.003
- Increase epochs: 15 → 30
- Unfreeze the backbone if frozen
- Use a larger model: `efficientnet_b1` or `efficientnet_b2`

**If training is unstable** (loss spikes):
- Reduce learning rate: 0.001 → 0.0003
- Gradient clipping is already applied (max norm = 1.0)
- Reduce batch size

**For faster iteration:**
- Use `freeze_backbone: true` (train only classifier)
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
| `classification` | 0.5 | Default decision boundary |
| `high_confidence` | 0.8 | High confidence predictions |
| `low_confidence` | 0.3 | Low confidence / uncertain |

---

## Updating the Deployed Model

### Method 1: Vertex AI (Recommended)

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
3. Runs a smoke test

### Method 2: Manual Model Replacement

1. Train a model locally or in Colab
2. Upload to GCS:
   ```bash
   gsutil cp models/checkpoints/best_model.pt \
     gs://<YOUR-GCS-BUCKET>/models/best_model.pt
   ```
3. Trigger the CD workflow:
   ```bash
   gh workflow run cd.yml -f image_tag=latest
   ```

### Method 3: Direct Cloud Run Update

```bash
# Build and push new image
docker build -f docker/Dockerfile \
  -t <REGION>-docker.pkg.dev/<YOUR-PROJECT-ID>/<REPO>/api:latest .
docker push <REGION>-docker.pkg.dev/<YOUR-PROJECT-ID>/<REPO>/api:latest

# Deploy to Cloud Run
gcloud run deploy <SERVICE-NAME> \
  --image <REGION>-docker.pkg.dev/<YOUR-PROJECT-ID>/<REPO>/api:latest \
  --region <REGION>
```

### Verifying the Update

```bash
# Health check
curl "https://<YOUR-SERVICE-URL>/health"

# Test with a known image
curl -X POST "https://<YOUR-SERVICE-URL>/predict" \
  -H "X-API-Key: <your-api-key>" \
  -F "file=@test_image.jpg"
```

---

## Troubleshooting

### Local Training Issues

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce `batch_size` to 32 or 16 |
| `Dataset not found` | Run `make data` to download CIFAKE |
| `MLflow connection error` | Start MLflow with `make mlflow` |
| `Import error` | Reinstall with `make dev` |

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
| OOM during training | Reduce `batch_size` in workflow inputs |
| Quality gate fails | Lower thresholds or improve model |
| Deployment fails | Check Cloud Run quota and permissions |

### Common Checks

```bash
# Check GCS permissions
gsutil ls gs://<YOUR-GCS-BUCKET>/

# Check Vertex AI quota
gcloud ai custom-jobs list --region=europe-west1

# Check Cloud Run status
gcloud run services describe <SERVICE-NAME> --region=europe-west1
```
