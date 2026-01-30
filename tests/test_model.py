"""Tests for the model module."""

import torch

from src.training.model import AIImageDetector, create_model


class TestAIImageDetector:
    """Tests for AIImageDetector class."""

    def test_model_creation(self) -> None:
        """Test model can be created."""
        model = create_model(pretrained=False)
        assert isinstance(model, AIImageDetector)

    def test_model_forward_pass(self) -> None:
        """Test forward pass produces correct output shape."""
        model = create_model(pretrained=False)
        model.eval()

        # Create dummy input
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)

        # Forward pass
        with torch.no_grad():
            output = model(x)

        # Check output shape
        assert output.shape == (batch_size, 1)

    def test_model_output_range(self) -> None:
        """Test output is in valid probability range [0, 1]."""
        model = create_model(pretrained=False)
        model.eval()

        x = torch.randn(8, 3, 224, 224)

        with torch.no_grad():
            output = model(x)

        # Check range
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)

    def test_model_param_count(self) -> None:
        """Test parameter counting methods."""
        model = create_model(pretrained=False)

        total_params = model.get_num_total_params()
        trainable_params = model.get_num_trainable_params()

        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params

    def test_model_freeze_backbone(self) -> None:
        """Test backbone freezing works."""
        model = create_model(pretrained=False, freeze_backbone=True)

        # Check backbone is frozen
        for param in model.backbone.parameters():
            assert not param.requires_grad

        # Check classifier is trainable
        for param in model.classifier.parameters():
            assert param.requires_grad

    def test_predict_proba(self) -> None:
        """Test predict_proba method."""
        model = create_model(pretrained=False)
        x = torch.randn(2, 3, 224, 224)

        proba = model.predict_proba(x)

        assert proba.shape == (2, 1)
        assert torch.all(proba >= 0)
        assert torch.all(proba <= 1)
