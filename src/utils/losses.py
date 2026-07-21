"""
losses.py
─────────
Custom loss functions for wildfire spread prediction.

Wildfire spread is highly imbalanced: fire pixels are rare compared to
non-fire pixels. Standard BCE loss will push the model to predict "no fire"
everywhere and still get low loss.

We use two strategies:
  1. Weighted BCE: manually upweight fire pixels via pos_weight
  2. Focal Loss: downweight easy negatives so the model focuses on hard positives
     Focal loss paper: https://arxiv.org/abs/1708.02002 [web:30]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy with Logits.
    pos_weight > 1 increases the penalty for missing fire pixels.
    """
    def __init__(self, pos_weight: float = 10.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            logits  : Raw model output [B, 1, H, W] (before sigmoid)
            targets : Binary fire labels [B, 1, H, W]
            mask    : Valid pixel mask [B, 1, H, W] — ignore uncertain pixels
        """
        pw = torch.tensor([self.pos_weight], device=logits.device)
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction="none"
        )
        if mask is not None:
            # Zero out loss for uncertain/invalid pixels
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-6)
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for binary segmentation.
    gamma=2.0 is standard; higher gamma focuses more on hard examples.
    Combined with pos_weight for the imbalanced spread problem. [web:30]
    """
    def __init__(self, gamma: float = 2.0, pos_weight: float = 10.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=torch.tensor([self.pos_weight], device=logits.device),
            reduction="none"
        )
        # pt = probability of the true class (how "easy" is this sample?)
        pt = torch.exp(-bce)
        # Focal weight: down-weight easy examples where pt is high
        focal_loss = ((1 - pt) ** self.gamma) * bce

        if mask is not None:
            focal_loss = focal_loss * mask
            return focal_loss.sum() / (mask.sum() + 1e-6)
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combines Focal Loss + Dice Loss for best segmentation performance.
    Dice loss directly optimizes the F1 score, complementing focal loss. [web:30]
    """
    def __init__(self, gamma: float = 2.0, pos_weight: float = 10.0, alpha: float = 0.5):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, pos_weight=pos_weight)
        self.alpha = alpha  # weight between focal and dice

    def dice_loss(self, probs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        """Soft Dice loss computed on sigmoid probabilities."""
        if mask is not None:
            probs   = probs   * mask
            targets = targets * mask
        num = 2 * (probs * targets).sum()
        den = probs.sum() + targets.sum() + 1e-6
        return 1 - num / den

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        focal = self.focal(logits, targets, mask)
        probs = torch.sigmoid(logits)
        dice  = self.dice_loss(probs, targets, mask)
        return self.alpha * focal + (1 - self.alpha) * dice