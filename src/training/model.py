"""Model architecture for AI image detection."""

import timm
import torch
import torch.nn as nn


class AIImageDetector(nn.Module):
    """EfficientNet-based binary classifier for AI image detection.

    Architecture:
        - Pretrained EfficientNet-B0 backbone (frozen or fine-tuned)
        - Custom classification head with dropout
        - Sigmoid output for probability score
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ) -> None:
        """Initialize the detector.

        Args:
            model_name: Name of the timm model to use.
            pretrained: Whether to use pretrained weights.
            dropout: Dropout rate for classification head.
            freeze_backbone: Whether to freeze backbone weights.
        """
        super().__init__()

        # Load pretrained backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
        )

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Get feature dimension from backbone
        # Spatial size doesn't matter here — global avg pool produces same feature_dim
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]

        # Custom classification head (outputs raw logits — use BCEWithLogitsLoss)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1),
        )

        self.feature_dim = feature_dim
        self.model_name = model_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224).

        Returns:
            Raw logits tensor of shape (batch_size, 1).
        """
        features = self.backbone(x)
        return self.classifier(features)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions.

        Args:
            x: Input tensor.

        Returns:
            Probability of being AI-generated.
        """
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters.

        Returns:
            Count of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_total_params(self) -> int:
        """Get total number of parameters.

        Returns:
            Total parameter count.
        """
        return sum(p.numel() for p in self.parameters())


def create_model(
    model_name: str = "efficientnet_b0",
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
) -> AIImageDetector:
    """Factory function to create the detector model.

    Args:
        model_name: Name of the backbone model.
        pretrained: Whether to use pretrained weights.
        dropout: Dropout rate.
        freeze_backbone: Whether to freeze backbone.

    Returns:
        Initialized AIImageDetector.
    """
    return AIImageDetector(
        model_name=model_name,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )
