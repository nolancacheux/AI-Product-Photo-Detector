"""Model calibration for reliable probability estimates."""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import DataLoader

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TemperatureScaling:
    """Temperature scaling for neural network calibration.

    Learns a single temperature parameter T to scale logits:
    calibrated_prob = sigmoid(logit / T)
    """

    def __init__(self, temperature: float = 1.0) -> None:
        """Initialize temperature scaling.

        Args:
            temperature: Initial temperature value.
        """
        self.temperature = temperature

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> "TemperatureScaling":
        """Fit temperature parameter using NLL loss.

        Args:
            logits: Model logits (before sigmoid).
            labels: Ground truth labels.

        Returns:
            Self with fitted temperature.
        """

        def nll_loss(t: float) -> float:
            """Negative log-likelihood with temperature."""
            scaled_logits = logits / t
            probs = 1 / (1 + np.exp(-scaled_logits))
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            loss = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
            return loss

        # Optimize temperature
        result = minimize_scalar(
            nll_loss,
            bounds=(0.1, 10.0),
            method="bounded",
        )

        self.temperature = result.x
        logger.info(f"Fitted temperature: {self.temperature:.4f}")

        return self

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits.

        Args:
            logits: Uncalibrated logits.

        Returns:
            Calibrated probabilities.
        """
        scaled_logits = logits / self.temperature
        return 1 / (1 + np.exp(-scaled_logits))

    def save(self, path: str | Path) -> None:
        """Save calibrator to file."""
        with open(path, "wb") as f:
            pickle.dump({"temperature": self.temperature}, f)

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureScaling":
        """Load calibrator from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls(temperature=data["temperature"])
        return instance


class PlattScaling:
    """Platt scaling (logistic calibration).

    Fits a logistic regression on the logits: p = 1 / (1 + exp(-(a*logit + b)))
    """

    def __init__(self) -> None:
        """Initialize Platt scaling."""
        self.a = 1.0
        self.b = 0.0

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        max_iter: int = 100,
    ) -> "PlattScaling":
        """Fit Platt scaling parameters.

        Args:
            logits: Model logits.
            labels: Ground truth labels.
            max_iter: Maximum iterations for optimization.

        Returns:
            Self with fitted parameters.
        """
        from sklearn.linear_model import LogisticRegression

        # Fit logistic regression
        lr = LogisticRegression(max_iter=max_iter, solver="lbfgs")
        lr.fit(logits.reshape(-1, 1), labels)

        self.a = lr.coef_[0, 0]
        self.b = lr.intercept_[0]

        logger.info(f"Fitted Platt parameters: a={self.a:.4f}, b={self.b:.4f}")

        return self

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply Platt scaling to logits.

        Args:
            logits: Uncalibrated logits.

        Returns:
            Calibrated probabilities.
        """
        return 1 / (1 + np.exp(-(self.a * logits + self.b)))

    def save(self, path: str | Path) -> None:
        """Save calibrator to file."""
        with open(path, "wb") as f:
            pickle.dump({"a": self.a, "b": self.b}, f)

    @classmethod
    def load(cls, path: str | Path) -> "PlattScaling":
        """Load calibrator from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls()
        instance.a = data["a"]
        instance.b = data["b"]
        return instance


class IsotonicCalibration:
    """Isotonic regression calibration.

    Non-parametric calibration that preserves probability ordering.
    """

    def __init__(self) -> None:
        """Initialize isotonic calibration."""
        self.ir = IsotonicRegression(out_of_bounds="clip")

    def fit(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> "IsotonicCalibration":
        """Fit isotonic regression.

        Args:
            probs: Uncalibrated probabilities.
            labels: Ground truth labels.

        Returns:
            Self with fitted isotonic regression.
        """
        self.ir.fit(probs, labels)
        logger.info("Fitted isotonic calibration")
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration.

        Args:
            probs: Uncalibrated probabilities.

        Returns:
            Calibrated probabilities.
        """
        return self.ir.predict(probs)

    def save(self, path: str | Path) -> None:
        """Save calibrator to file."""
        with open(path, "wb") as f:
            pickle.dump(self.ir, f)

    @classmethod
    def load(cls, path: str | Path) -> "IsotonicCalibration":
        """Load calibrator from file."""
        with open(path, "rb") as f:
            ir = pickle.load(f)
        instance = cls()
        instance.ir = ir
        return instance


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Args:
        probs: Predicted probabilities.
        labels: Ground truth labels.
        n_bins: Number of bins for ECE.

    Returns:
        ECE value.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(probs)

    for i in range(n_bins):
        mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_accuracy = labels[mask].mean()
            bin_confidence = probs[mask].mean()
            bin_size = mask.sum()
            ece += (bin_size / total) * abs(bin_accuracy - bin_confidence)

    return ece


def compute_calibration_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute comprehensive calibration metrics.

    Args:
        probs: Predicted probabilities.
        labels: Ground truth labels.
        n_bins: Number of bins for calibration curve.

    Returns:
        Dictionary of calibration metrics.
    """
    # ECE
    ece = compute_ece(probs, labels, n_bins=15)

    # Calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        labels, probs, n_bins=n_bins, strategy="uniform"
    )

    # Brier score
    brier_score = np.mean((probs - labels) ** 2)

    # Maximum Calibration Error
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    max_error = 0.0

    for i in range(n_bins):
        mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_accuracy = labels[mask].mean()
            bin_confidence = probs[mask].mean()
            error = abs(bin_accuracy - bin_confidence)
            max_error = max(max_error, error)

    return {
        "ece": ece,
        "mce": max_error,
        "brier_score": brier_score,
        "calibration_curve": {
            "fraction_of_positives": fraction_of_positives.tolist(),
            "mean_predicted_value": mean_predicted_value.tolist(),
        },
    }


def calibrate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    method: str = "temperature",
) -> TemperatureScaling | PlattScaling | IsotonicCalibration:
    """Calibrate a model on validation data.

    Args:
        model: Trained model.
        val_loader: Validation data loader.
        device: Torch device.
        method: Calibration method ('temperature', 'platt', 'isotonic').

    Returns:
        Fitted calibrator.
    """
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            # Get logits (before sigmoid)
            outputs = model(images, return_logits=True)
            all_logits.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.numpy())

    logits = np.array(all_logits)
    labels = np.array(all_labels)
    probs = 1 / (1 + np.exp(-logits))

    # Pre-calibration metrics
    pre_metrics = compute_calibration_metrics(probs, labels)
    logger.info(
        "Pre-calibration metrics",
        ece=f"{pre_metrics['ece']:.4f}",
        brier=f"{pre_metrics['brier_score']:.4f}",
    )

    # Fit calibrator
    if method == "temperature":
        calibrator = TemperatureScaling().fit(logits, labels)
        calibrated_probs = calibrator.calibrate(logits)
    elif method == "platt":
        calibrator = PlattScaling().fit(logits, labels)
        calibrated_probs = calibrator.calibrate(logits)
    elif method == "isotonic":
        calibrator = IsotonicCalibration().fit(probs, labels)
        calibrated_probs = calibrator.calibrate(probs)
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    # Post-calibration metrics
    post_metrics = compute_calibration_metrics(calibrated_probs, labels)
    logger.info(
        "Post-calibration metrics",
        ece=f"{post_metrics['ece']:.4f}",
        brier=f"{post_metrics['brier_score']:.4f}",
    )

    return calibrator
