"""
unet.py
───────
Defines the wildfire spread prediction model.

We use segmentation_models_pytorch (smp) which provides a U-Net with a
pretrained ResNet18 encoder. This gives us:
  - Skip connections between encoder and decoder (important for spatial detail)
  - Pretrained feature extraction from ImageNet (helps with limited fire data)
  - A clean 1-channel sigmoid output → probability map per pixel [web:30]

Architecture: ResNet18 encoder → U-Net decoder → 1x1 conv → sigmoid
"""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_model(
    encoder_name: str = "resnet18",
    encoder_weights: str = "imagenet",
    in_channels: int = 12,
    out_channels: int = 1,
):
    """
    Builds a U-Net model for wildfire spread segmentation.

    Args:
        encoder_name    : Backbone encoder (e.g., 'resnet18', 'resnet34')
        encoder_weights : Use 'imagenet' for pretrained weights, None for random init
        in_channels     : Number of input feature channels (12 for this dataset)
        out_channels    : 1 for binary spread probability map

    Returns:
        model: nn.Module ready for training
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
        activation=None,  # We apply sigmoid manually so we can use BCEWithLogitsLoss
    )
    return model


def count_parameters(model: nn.Module) -> int:
    """Utility: count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)