"""GradCAM explainability for model predictions."""

import io
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class GradCAM:
    """Gradient-weighted Class Activation Mapping for model explainability."""

    def __init__(self, model: torch.nn.Module, target_layer: str = "features") -> None:
        """Initialize GradCAM.

        Args:
            model: The model to explain.
            target_layer: Name of the target layer for activation maps.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target layer."""

        def forward_hook(
            module: torch.nn.Module, input: Any, output: torch.Tensor
        ) -> None:
            self.activations = output.detach()

        def backward_hook(
            module: torch.nn.Module, grad_input: Any, grad_output: tuple[torch.Tensor, ...]
        ) -> None:
            self.gradients = grad_output[0].detach()

        # Find and hook the target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                return

        raise ValueError(f"Layer '{self.target_layer}' not found in model")

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """Generate GradCAM heatmap.

        Args:
            input_tensor: Input image tensor (1, C, H, W).
            target_class: Target class index. None for predicted class.

        Returns:
            Heatmap as numpy array (H, W) with values 0-1.
        """
        self.model.eval()
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = 1 if output.item() > 0.5 else 0

        # Create target for binary classification
        target = output if target_class == 1 else (1 - output)

        # Backward pass
        self.model.zero_grad()
        target.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not capture gradients/activations")

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to 0-1
        if cam.max() != cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam

    def generate_overlay(
        self,
        image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: str = "jet",
    ) -> Image.Image:
        """Overlay heatmap on original image.

        Args:
            image: Original PIL image.
            heatmap: GradCAM heatmap array.
            alpha: Overlay transparency (0-1).
            colormap: Matplotlib colormap name.

        Returns:
            PIL Image with heatmap overlay.
        """
        import matplotlib.pyplot as plt

        # Resize heatmap to image size
        heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_resized = heatmap_resized.resize(image.size, Image.Resampling.BILINEAR)
        heatmap_array = np.array(heatmap_resized) / 255.0

        # Apply colormap
        cmap = plt.get_cmap(colormap)
        heatmap_colored = cmap(heatmap_array)[:, :, :3]  # Remove alpha channel
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

        # Convert original image to RGB
        image_rgb = image.convert("RGB")
        image_array = np.array(image_rgb)

        # Blend images
        blended = (1 - alpha) * image_array + alpha * heatmap_colored
        blended = blended.astype(np.uint8)

        return Image.fromarray(blended)


class ExplainablePredictor:
    """Predictor with GradCAM explainability."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        transform: transforms.Compose,
    ) -> None:
        """Initialize explainable predictor.

        Args:
            model: The loaded model.
            device: Torch device.
            transform: Image transform pipeline.
        """
        self.model = model
        self.device = device
        self.transform = transform
        self.gradcam = GradCAM(model, target_layer="features")

    def explain(
        self,
        image_bytes: bytes,
        alpha: float = 0.5,
    ) -> tuple[Image.Image, np.ndarray, float]:
        """Generate explanation for prediction.

        Args:
            image_bytes: Raw image bytes.
            alpha: Heatmap overlay transparency.

        Returns:
            Tuple of (overlay image, raw heatmap, probability).
        """
        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Transform for model
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Get prediction
        with torch.no_grad():
            output = self.model(tensor.clone())
            probability = output.item()

        # Generate heatmap (needs gradients, so no no_grad)
        tensor_for_grad = self.transform(image).unsqueeze(0).to(self.device)
        heatmap = self.gradcam.generate(tensor_for_grad)

        # Create overlay
        overlay = self.gradcam.generate_overlay(image, heatmap, alpha=alpha)

        return overlay, heatmap, probability

    def explain_to_bytes(
        self,
        image_bytes: bytes,
        alpha: float = 0.5,
        format: str = "PNG",
    ) -> tuple[bytes, float]:
        """Generate explanation and return as bytes.

        Args:
            image_bytes: Raw image bytes.
            alpha: Heatmap overlay transparency.
            format: Output image format.

        Returns:
            Tuple of (overlay image bytes, probability).
        """
        overlay, _, probability = self.explain(image_bytes, alpha)

        # Convert to bytes
        buffer = io.BytesIO()
        overlay.save(buffer, format=format)
        buffer.seek(0)

        return buffer.getvalue(), probability
