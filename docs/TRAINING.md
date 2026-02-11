# Training Guide

This document covers all three methods for training the AI Product Photo
Detector model: local training, Google Colab, and Vertex AI.

---

## Table of Contents

1. [Model Architecture](#model-architecture)
2. [Dataset](#dataset)
3. [Training Option 1: Local](#training-option-1-local)
4. [Training Option 2: Google Colab](#training-option-2-google-colab)
5. [Training Option 3: Vertex AI](#training-option-3-vertex-ai)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Evaluation](#evaluation)
8. [Updating the Deployed Model](#updating-the-deployed-model)

---

## Model Architecture

The detector uses an **EfficientNet-B0** backbone with a custom binary
classification head.

**Source:** `src/training/model.py`

### Architecture

```
Input Image (3 x 224 x 224)
        |
        v
+-------------------+
| EfficientNet-B0   |   Pretrained on ImageNet
| (backbone)        |   Global average pooling removes spatial dims
| Feature dim: 1280 |
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
   Sigmoid               (inference)
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| EfficientNet-B0 | Good accuracy-to-size ratio; small enough for Cloud Run |
| Pretrained backbone | Transfer learning from ImageNet reduces training time |
| BatchNorm in classifier | Stabilizes training, especially with small batch sizes |
| BCEWithLogitsLoss | Numerically stable; model outputs raw logits |
| Dropout 0.3 | Regularization to prevent overfitting on small datasets |

### Parameters

- **Total parameters:** ~5.3M (EfficientNet-B0 backbone + classifier head)
- **Trainable parameters:** ~5.3M (full fine-tuning by default)
- **Optional:** Set `freeze_backbone=True` to freeze the backbone and train only
  the classifier head (~660K trainable parameters).

---

## Dataset

### CIFAKE Dataset

The primary dataset is [CIFAKE: Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images),
containing real photographs and AI-generated counterparts.

### Directory Structure

```
data/processed/
├── train/
│   ├── real/
│   │   ├── image_0001.jpg
│   │   └── ...
│   └── ai_generated/
│       ├── image_0001.jpg
│       └── ...
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

If the original CIFAKE dataset is unavailable on Kaggle, equivalent datasets
can be found on HuggingFace:

- `emirhanbilgic/cifake-real-and-ai-generated-synthetic-images`
- `jlbaker361/CIFake`

Download and organize the images into the directory structure shown above.

### Data Augmentation

Training augmentations are defined in `src/training/augmentation.py` and
configured in `configs/train_config.yaml`:

| Augmentation | Value |
|---|---|
| Horizontal flip | Yes |
| Random rotation | +/- 15 degrees |
| Color jitter (brightness) | 0.2 |
| Color jitter (contrast) | 0.2 |
| Color jitter (saturation) | 0.1 |
| Color jitter (hue) | 0.05 |
| Random crop | Yes |

Validation and test sets use only resize + center crop + normalization (ImageNet
statistics).

---

## Training Option 1: Local

Local training is suitable for development, debugging, and quick experiments.

### Prerequisites

- Python 3.11 or 3.12
- GPU recommended (CUDA or Apple MPS); CPU works but is slow
- Training data in `data/processed/` (see [Dataset](#dataset))

### Setup

```bash
# Install project with dev dependencies
pip install -e ".[dev]"

# If data is tracked with DVC, pull it
pip install "dvc[gs]"
dvc pull
```

### DVC Configuration

If using DVC for data versioning:

```bash
# Initialize DVC (if not already done)
dvc init

# Add GCS remote
dvc remote add -d gcs gs://ai-product-detector-487013/dvc
dvc remote modify gcs credentialpath /path/to/sa-key.json

# Track data
dvc add data/processed

# Pull data
dvc pull
```

### Run Training

```bash
# Default configuration
python -m src.training.train --config configs/train_config.yaml

# Override epochs and batch size
python -m src.training.train --config configs/train_config.yaml \
  --epochs 10 --batch-size 32

# With GCS integration (uploads model after training)
python -m src.training.train --config configs/train_config.yaml \
  --gcs-bucket ai-product-detector-487013
```

### MLflow Tracking

Training metrics are automatically logged to MLflow.

**Using the local Docker Compose MLflow server:**

```bash
# Start MLflow server
docker compose up -d mlflow

# The training script will connect to http://localhost:5000 if configured
# Update configs/train_config.yaml:
#   mlflow.tracking_uri: "http://localhost:5000"
```

**Using local file storage (default):**

The default configuration stores MLflow runs in the `mlruns/` directory. View
them with:

```bash
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000
```

### Logged Metrics (per epoch)

| Metric | Description |
|---|---|
| `train_loss` | Training loss |
| `train_accuracy` | Training accuracy |
| `val_loss` | Validation loss |
| `val_accuracy` | Validation accuracy |
| `val_precision` | Validation precision |
| `val_recall` | Validation recall |
| `val_f1` | Validation F1 score |
| `learning_rate` | Current learning rate |

### Output

The best model checkpoint is saved to `models/checkpoints/best_model.pt`
containing:
- `model_state_dict`
- `optimizer_state_dict`
- `scheduler_state_dict`
- `val_accuracy` (accuracy at the time of saving)
- `best_val_accuracy`
- `config` (full training configuration)

---

## Training Option 2: Google Colab

A pre-built notebook is available for training on Google Colab with free GPU
access.

**Notebook:** `notebooks/train_colab.ipynb`

### Steps

1. **Open the notebook** in Google Colab:
   - Upload `notebooks/train_colab.ipynb` to Colab, or
   - Open directly from GitHub if the repository is public.

2. **Select a GPU runtime:**
   - Go to **Runtime > Change runtime type > T4 GPU**.

3. **Configure the notebook:**
   - Set your GCS bucket name (if uploading the model to GCS).
   - Adjust hyperparameters (epochs, batch size, learning rate).
   - Ensure the dataset is accessible (upload to Colab or mount Google Drive).

4. **Run all cells** sequentially.

5. **Download the trained model:**
   - The notebook saves the model to `models/checkpoints/best_model.pt`.
   - Download it from the Colab file browser, or let the notebook upload it
     to GCS.

### Tips for Colab

- Free Colab sessions have a ~12-hour limit and may disconnect.
- Save checkpoints periodically to Google Drive to avoid losing progress.
- Use a smaller batch size (32) if running out of GPU memory on a free T4.

---

## Training Option 3: Vertex AI

Vertex AI provides managed GPU training with automatic scaling and integration
with the CI/CD pipeline. This is the recommended approach for production model
updates.

### Via GitHub Actions (recommended)

1. Go to **Actions > Model Training (Vertex AI) > Run workflow**.
2. Configure inputs:

   | Input | Default | Description |
   |---|---|---|
   | `epochs` | 15 | Number of training epochs |
   | `batch_size` | 64 | Training batch size |
   | `auto_deploy` | false | Automatically deploy if quality gate passes |

3. The workflow executes the following pipeline:

```
Verify GCS Data --> Build Training Image --> Submit Vertex AI Job
                                                    |
                                              (n1-standard-4 + T4 GPU)
                                                    |
                                                    v
                                            Download Model
                                                    |
                                                    v
                                            Evaluate Model
                                              (CPU, on runner)
                                                    |
                                            Quality Gate:
                                            accuracy >= 0.85
                                            F1 >= 0.80
                                                    |
                                         +----------+----------+
                                         |                     |
                                       PASS                  FAIL
                                         |                     |
                                    [if auto_deploy]       Pipeline
                                    Deploy to              stops
                                    Cloud Run
```

### Monitoring a Training Job

```bash
# List recent Vertex AI training jobs
gcloud ai custom-jobs list \
  --project=ai-product-detector-487013 \
  --region=europe-west1

# View logs for a specific job
gcloud ai custom-jobs describe <JOB_ID> \
  --project=ai-product-detector-487013 \
  --region=europe-west1

# Stream logs
gcloud ai custom-jobs stream-logs <JOB_ID> \
  --project=ai-product-detector-487013 \
  --region=europe-west1
```

### Via GitHub Actions (automatic trigger)

The training workflow also triggers automatically when files under `data/**` are
modified on the `main` branch. This enables a data-driven retraining flow:

1. Update or add training data.
2. Push to `main`.
3. The workflow verifies data, trains, evaluates, and deploys if the quality gate
   passes.

### Machine Configuration

| Component | Specification |
|---|---|
| Machine type | `n1-standard-4` (4 vCPUs, 15 GB RAM) |
| GPU | 1x NVIDIA Tesla T4 (16 GB VRAM) |
| Training image | `docker/Dockerfile.training` (PyTorch 2.2.0 + CUDA 12.1) |
| Timeout | 180 minutes |

### Training Container

The training Docker image (`docker/Dockerfile.training`) is built on
`pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime` and includes:

- Full project source code and dependencies.
- The training configuration from `configs/train_config.yaml`.
- No data or model baked in -- pulled from GCS at runtime.

**Default command:**
```
python -m src.training.train --config configs/train_config.yaml
```

---

## Hyperparameter Tuning

All hyperparameters are configured in `configs/train_config.yaml`.

### Default Configuration

| Parameter | Value | Notes |
|---|---|---|
| Learning rate | 0.001 | AdamW optimizer |
| Weight decay | 0.0001 | L2 regularization |
| Batch size | 64 | Reduce to 32 for smaller GPUs |
| Image size | 224 x 224 | EfficientNet-B0 native resolution |
| Epochs | 15 | With early stopping |
| Early stopping patience | 5 | Epochs without improvement |
| Scheduler | Cosine annealing | Decays LR to near zero over training |
| Warmup epochs | 2 | Gradual LR increase at start |
| Dropout | 0.3 | In the classifier head |
| Seed | 42 | For reproducibility |

### Tuning Recommendations

**If overfitting (val loss increasing while train loss decreases):**
- Increase dropout (0.3 -> 0.5).
- Increase weight decay (0.0001 -> 0.001).
- Add more aggressive augmentation.
- Freeze the backbone (`freeze_backbone: true`).

**If underfitting (both losses remain high):**
- Increase learning rate (0.001 -> 0.003).
- Increase epochs (15 -> 30).
- Unfreeze the backbone if frozen.
- Use a larger model (`efficientnet_b1` or `efficientnet_b2`).

**If training is unstable (loss spikes):**
- Reduce learning rate (0.001 -> 0.0003).
- Gradient clipping is already applied (max norm = 1.0).
- Reduce batch size.

**For faster iteration:**
- Use `freeze_backbone: true` and train only the classifier head.
- Reduce epochs to 5--10 for quick experiments.

---

## Evaluation

### Quality Gate (CI/CD)

The Model Training workflow applies an automatic quality gate:

| Metric | Threshold |
|---|---|
| Accuracy | >= 0.85 |
| F1 Score | >= 0.80 |

If the model fails the quality gate, it is not deployed. Metrics are saved to
`reports/metrics.json` and uploaded as a GitHub Actions artifact.

### Manual Evaluation

```bash
# Run evaluation on local test data
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

### Metrics Tracked

| Metric | Description |
|---|---|
| Accuracy | Overall correct predictions / total |
| Precision | True positives / (true positives + false positives) |
| Recall | True positives / (true positives + false negatives) |
| F1 Score | Harmonic mean of precision and recall |

### Classification Thresholds

Configured in `configs/train_config.yaml` under `thresholds`:

| Threshold | Value | Usage |
|---|---|---|
| `classification` | 0.5 | Default decision boundary |
| `high_confidence` | 0.8 | High confidence predictions |
| `low_confidence` | 0.3 | Low confidence / uncertain |

During inference, the model outputs a raw logit. Applying sigmoid converts it
to a probability in [0, 1]. The classification threshold determines the binary
decision.

---

## Updating the Deployed Model

### Method 1: Vertex AI (recommended)

1. Trigger the Model Training workflow with `auto_deploy: true`.
2. If the quality gate passes, the pipeline automatically:
   - Builds a new inference Docker image with the model baked in.
   - Deploys to Cloud Run.
   - Runs a smoke test.

### Method 2: Manual model replacement

1. Train a model locally or download from another source.
2. Upload to GCS:
   ```bash
   gsutil cp models/checkpoints/best_model.pt \
     gs://ai-product-detector-487013/models/best_model.pt
   ```
3. Trigger the CD workflow (push to `main` or manual dispatch) to rebuild the
   Docker image with the new model.

### Method 3: Direct Cloud Run update

For immediate updates without rebuilding the image:

1. Upload the model to GCS.
2. If the application supports runtime model loading from GCS (check
   implementation), the new model will be picked up on the next cold start.
3. Otherwise, trigger a redeployment to bake the model into a new image.

### Verifying the Update

```bash
# Check the deployed service
URL=$(gcloud run services describe ai-product-detector \
  --region=europe-west1 \
  --format='value(status.url)')

# Health check
curl "${URL}/health"

# Test with a known image
curl -X POST "${URL}/predict" \
  -H "X-API-Key: <your-api-key>" \
  -F "file=@test_image.jpg"
```
