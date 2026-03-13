# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ResNet18 encoder for vision-based RL policies."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


class ResNet18Encoder(nn.Module):
    """ResNet18 encoder with ImageNet pretrained weights for vision-based RL.

    This encoder uses a pretrained ResNet18 (ImageNet weights) as a feature extractor.
    The final fully-connected layer is removed, and a linear projection layer is added
    to map the 512-dimensional ResNet features to the desired output dimension.

    The encoder supports:
    - Frozen mode: ResNet weights are frozen (eval mode, no gradients)
    - Finetuning mode: ResNet weights can be trained (train mode, gradients enabled)

    Image normalization uses ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    which matches the pretrained weights.

    Args:
        output_dim: The output dimension of the encoder (after linear projection). Defaults to 32.
        device: The device to run the encoder on. Defaults to "cuda".
        train_resnet: Whether ResNet weights should be trainable. Defaults to False (frozen).
    """

    # ImageNet normalization constants
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, output_dim: int = 32, device: str = "cuda", train_resnet: bool = False):
        super().__init__()
        self.device = device
        self.train_resnet = train_resnet
        self.output_dim = output_dim

        # Load pretrained ResNet18 (torchvision automatically downloads and caches weights)
        self.resnet18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

        # Remove the final fully-connected layer (keep features only)
        self.resnet18.fc = nn.Identity()

        # Set ResNet to appropriate mode
        if train_resnet:
            self.resnet18.train()
        else:
            self.resnet18.eval()

        # Move to device
        self.resnet18 = self.resnet18.to(device)

        # Linear projection layer (512 ResNet features -> output_dim)
        self.linear = nn.Linear(512, output_dim).to(device)

        # ImageNet normalization transform
        # Note: Images should already be in [0, 1] range and in NCHW format
        self.normalize = transforms.Normalize(
            mean=self.IMAGENET_MEAN,
            std=self.IMAGENET_STD,
        )

    def forward(self, x: torch.Tensor, train_encoder: bool | None = None) -> torch.Tensor:
        """Forward pass through the ResNet18 encoder.

        Args:
            x: Input images in NCHW format, normalized to [0, 1] range.
                Shape: (batch_size, 3, height, width)
            train_encoder: Whether to enable gradients. If None, uses self.train_resnet.
                When False, uses torch.no_grad() to prevent gradient computation.

        Returns:
            Encoded features. Shape: (batch_size, output_dim)
        """
        # Determine if we should compute gradients
        if train_encoder is None:
            train_encoder = self.train_resnet

        # Apply ImageNet normalization
        x = self.normalize(x)

        # Forward through ResNet
        if train_encoder:
            # Enable gradients
            resnet_out = self.resnet18(x)
        else:
            # Disable gradients (frozen mode)
            with torch.no_grad():
                resnet_out = self.resnet18(x)

        # Project to output dimension
        out = self.linear(resnet_out)
        return out

    def set_train_mode(self, train: bool):
        """Set whether ResNet should be in training mode.

        Args:
            train: If True, ResNet is set to train mode and gradients are enabled.
                If False, ResNet is set to eval mode and gradients are disabled.
        """
        self.train_resnet = train
        if train:
            self.resnet18.train()
        else:
            self.resnet18.eval()
