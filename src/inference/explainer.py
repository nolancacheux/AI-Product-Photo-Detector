"""Grad-CAM explainability for model predictions."""

import base64
import io
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

from src.training.model import AIImageDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _rebuild_classifier(state_dict: dict, feature_dim: int, dropout: float) -> nn.Sequential:
    """Rebuild classifier head to match checkpoint structure.

    Inspects state_dict keys to determine the original classifier layout,
    handling older checkpoints that lack BatchNorm1d.
    """
    cls_keys = sorted(k for k in state_dict if k.startswith("classifier."))
    weight_indices = sorted({int(k.split(".")[1]) for k in cls_keys})

    # Current model: [Linear, BatchNorm, ReLU, Dropout, Linear] → indices 0,1,2,3,4
    # Old model:     [Linear, ReLU, Dropout, Linear]             → indices 0,1,2,3
    has_batchnorm = any(
        k for k in cls_keys if ".running_mean" in k or ".running_var" in k
    )

    if has_batchnorm:
        return nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1),
        )
    return nn.Sequential(
        nn.Linear(feature_dim, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(512, 1),
    )


class GradCAMExplainer:
    """Generate Grad-CAM heatmaps explaining model predictions."""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.model_path = model_path
        self.device = torch.device(device)
        self.model: AIImageDetector | None = None
        self.cam: GradCAM | None = None
        self.model_version: str = "1.0.0"
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._load_model()

    def _load_model(self) -> None:
        try:
            checkpoint = torch.load(
                self.model_path, map_location=self.device, weights_only=False,
            )

            config = checkpoint.get("config", {})
            model_config = config.get("model", {})

            self.model = AIImageDetector(
                model_name=model_config.get("name", "efficientnet_b0"),
                pretrained=False,
                dropout=model_config.get("dropout", 0.3),
            )

            # Rebuild classifier to match checkpoint structure (handles old checkpoints)
            state_dict = checkpoint["model_state_dict"]
            self.model.classifier = _rebuild_classifier(
                state_dict, self.model.feature_dim, model_config.get("dropout", 0.3),
            )

            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            self.model_version = f"1.0.{checkpoint.get('epoch', 0)}"

            # Target the last conv block before global pooling (timm EfficientNet)
            target_layer = self.model.backbone.bn2
            self.cam = GradCAM(model=self.model, target_layers=[target_layer])
            logger.info("GradCAM explainer loaded", model_path=self.model_path)
        except Exception as e:
            logger.error("Failed to load explainer", error=str(e))
            self.model = None
            self.cam = None

    def explain(self, image_bytes: bytes) -> dict:
        """Generate prediction with Grad-CAM heatmap overlay.

        Args:
            image_bytes: Raw JPEG/PNG image bytes.

        Returns:
            Dict with prediction, probability, confidence, heatmap_base64, inference_time_ms.

        Raises:
            RuntimeError: If explainer is not ready.
        """
        if not self.is_ready():
            raise RuntimeError("Explainer not loaded")

        start = time.monotonic()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_resized = image.resize((224, 224))

        # Normalized float array for heatmap overlay
        img_array = np.array(image_resized).astype(np.float32) / 255.0

        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Get prediction
        with torch.no_grad():
            logit = self.model(input_tensor).squeeze()
            probability = torch.sigmoid(logit).item()

        # Generate heatmap (needs gradients, so outside no_grad)
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]

        # Overlay heatmap on original image
        visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)

        # Encode to base64 JPEG
        vis_image = Image.fromarray(visualization)
        buf = io.BytesIO()
        vis_image.save(buf, format="JPEG", quality=90)
        heatmap_b64 = base64.b64encode(buf.getvalue()).decode()

        image.close()

        elapsed = (time.monotonic() - start) * 1000

        prediction = "ai_generated" if probability > 0.5 else "real"
        distance = abs(probability - 0.5)
        if distance > 0.3:
            confidence = "high"
        elif distance > 0.1:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "confidence": confidence,
            "heatmap_base64": heatmap_b64,
            "inference_time_ms": round(elapsed, 2),
        }

    def is_ready(self) -> bool:
        """Check if explainer is loaded and ready."""
        return self.model is not None and self.cam is not None
