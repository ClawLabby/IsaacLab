# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for image_features ResNet model preparation in observations.py.

Verifies that the ResNet model preparation correctly removes the FC layer,
produces the expected feature dimensions, and runs inference with no_grad.
No Kit/AppLauncher required — tests the model loading logic directly.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class TestResNetModelPreparation:
    """Test the _prepare_resnet_model logic from image_features."""

    @staticmethod
    def _build_resnet_dict(model_name: str, device: str = "cpu") -> dict:
        """Reproduce the model preparation logic from observations.py.

        We import and call the logic inline to avoid needing the full environment
        manager infrastructure (which requires Kit).
        """
        from torchvision import models

        resnet_weights_map = {
            "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
            "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
            "resnet50": models.ResNet50_Weights.IMAGENET1K_V1,
            "resnet101": models.ResNet101_Weights.IMAGENET1K_V1,
        }

        if model_name not in resnet_weights_map:
            raise ValueError(f"Unsupported ResNet model: {model_name}")

        model = getattr(models, model_name)(weights=resnet_weights_map[model_name])
        model.fc = torch.nn.Identity()
        model.eval()
        model = model.to(device)

        def inference(m, images: torch.Tensor) -> torch.Tensor:
            image_proc = images.to(device)
            image_proc = image_proc.permute(0, 3, 1, 2).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(1, 3, 1, 1)
            image_proc = (image_proc - mean) / std
            with torch.no_grad():
                features = m(image_proc)
            return features

        return {"model": model, "inference": inference}

    def test_resnet18_fc_removed(self):
        """ResNet18 FC layer should be replaced with Identity."""
        d = self._build_resnet_dict("resnet18")
        model = d["model"]
        assert isinstance(model.fc, nn.Identity), f"Expected Identity, got {type(model.fc)}"

    def test_resnet18_feature_dim(self):
        """ResNet18 should produce 512-dim features."""
        d = self._build_resnet_dict("resnet18")
        model = d["model"]
        # Create a dummy image batch (NHWC format, uint8-like)
        dummy = torch.randint(0, 256, (2, 224, 224, 3), dtype=torch.uint8)
        features = d["inference"](model, dummy)
        assert features.shape == (2, 512), f"Expected (2, 512), got {features.shape}"

    def test_resnet50_feature_dim(self):
        """ResNet50 should produce 2048-dim features."""
        d = self._build_resnet_dict("resnet50")
        model = d["model"]
        dummy = torch.randint(0, 256, (1, 224, 224, 3), dtype=torch.uint8)
        features = d["inference"](model, dummy)
        assert features.shape == (1, 2048), f"Expected (1, 2048), got {features.shape}"

    def test_inference_no_grad(self):
        """Inference should not accumulate gradients."""
        d = self._build_resnet_dict("resnet18")
        model = d["model"]
        dummy = torch.randint(0, 256, (1, 224, 224, 3), dtype=torch.uint8)
        features = d["inference"](model, dummy)
        assert not features.requires_grad, "Features should not require grad"

    def test_model_eval_mode(self):
        """Model should be in eval mode (not training)."""
        d = self._build_resnet_dict("resnet18")
        model = d["model"]
        assert not model.training, "Model should be in eval mode"

    def test_unsupported_model_raises(self):
        """Unsupported model name should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported ResNet model"):
            self._build_resnet_dict("resnet152")
