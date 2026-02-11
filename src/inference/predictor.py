"""Model predictor for inference."""

import io
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.inference.schemas import ConfidenceLevel, PredictionResult, PredictResponse
from src.training.model import AIImageDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    """Handles model loading and prediction."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "auto",
        threshold: float = 0.5,
        high_confidence_threshold: float = 0.8,
        low_confidence_threshold: float = 0.3,
    ) -> None:
        """Initialize predictor.

        Args:
            model_path: Path to model checkpoint.
            device: Device to use (auto, cpu, cuda, mps).
            threshold: Classification threshold.
            high_confidence_threshold: Threshold for high confidence.
            low_confidence_threshold: Threshold for low confidence.
        """
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.high_confidence_threshold = high_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold

        # Set device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Load model
        self.model: AIImageDetector | None = None
        self.model_version: str = "unknown"
        self._load_model()

        # Setup transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _load_model(self) -> None:
        """Load model from checkpoint."""
        if not self.model_path.exists():
            logger.warning("Model not found", model_path=str(self.model_path))
            return

        try:
            checkpoint = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=False,
            )

            # Get config from checkpoint
            config = checkpoint.get("config", {})
            model_config = config.get("model", {})

            # Create model
            self.model = AIImageDetector(
                model_name=model_config.get("name", "efficientnet_b0"),
                pretrained=False,
                dropout=model_config.get("dropout", 0.3),
            )

            # Handle old checkpoints without BatchNorm in classifier
            state_dict = checkpoint["model_state_dict"]
            try:
                self.model.load_state_dict(state_dict)
            except RuntimeError:
                # Rebuild classifier to match checkpoint format
                import torch.nn as nn

                if "classifier.1.weight" not in state_dict and "classifier.3.weight" in state_dict:
                    # Old format: Linear(in, 512) -> ReLU -> Dropout -> Linear(512, 1)
                    in_features = getattr(self.model.classifier[0], "in_features", 1280)
                    dropout = model_config.get("dropout", 0.3)
                    self.model.classifier = nn.Sequential(
                        nn.Linear(int(in_features), 512),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=dropout),
                        nn.Linear(512, 1),
                    )
                    self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            self.model.eval()

            # Set version
            self.model_version = f"1.0.{checkpoint.get('epoch', 0)}"

            logger.info(
                "Model loaded successfully",
                model_path=str(self.model_path),
                device=str(self.device),
                version=self.model_version,
            )
        except Exception:
            logger.exception("Failed to load model")
            self.model = None

    def is_ready(self) -> bool:
        """Check if predictor is ready.

        Returns:
            True if model is loaded and ready.
        """
        return self.model is not None

    def _get_confidence_level(self, probability: float) -> ConfidenceLevel:
        """Determine confidence level from probability.

        Args:
            probability: Prediction probability.

        Returns:
            Confidence level.
        """
        # Distance from decision boundary (0.5)
        distance = abs(probability - 0.5)

        if distance > (self.high_confidence_threshold - 0.5):
            return ConfidenceLevel.HIGH
        elif distance < (0.5 - self.low_confidence_threshold):
            return ConfidenceLevel.LOW
        return ConfidenceLevel.MEDIUM

    def predict_from_bytes(self, image_bytes: bytes) -> PredictResponse:
        """Predict from image bytes.

        Args:
            image_bytes: Raw image bytes.

        Returns:
            Prediction response.

        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If image cannot be processed.
        """
        if not self.is_ready():
            raise RuntimeError("Model not loaded")

        start_time = time.time()

        try:
            # Load image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Transform then close PIL image to free memory
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            image.close()

            # Predict (model outputs raw logits, apply sigmoid for probability)
            assert self.model is not None  # guaranteed by is_ready() check above
            with torch.no_grad():
                output = self.model(tensor)
                probability = torch.sigmoid(output).item()

            # Calculate inference time
            inference_time_ms = (time.time() - start_time) * 1000

            # Determine prediction
            prediction = (
                PredictionResult.AI_GENERATED
                if probability > self.threshold
                else PredictionResult.REAL
            )

            return PredictResponse(
                prediction=prediction,
                probability=round(probability, 4),
                confidence=self._get_confidence_level(probability),
                inference_time_ms=round(inference_time_ms, 2),
                model_version=self.model_version,
            )

        except Exception as e:
            logger.exception("Prediction failed")
            raise ValueError("Failed to process image") from e

    def predict_from_path(self, image_path: str | Path) -> PredictResponse:
        """Predict from image file path.

        Args:
            image_path: Path to image file.

        Returns:
            Prediction response.
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(image_path, "rb") as f:
            return self.predict_from_bytes(f.read())
