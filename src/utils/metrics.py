"""
metrics.py
──────────
Evaluation metrics for wildfire spread prediction.

Because fire pixels are rare, accuracy is misleading.
We use:
  - F1 Score  : balance of precision and recall
  - AUROC     : area under ROC curve (threshold-independent)
  - AUPRC     : area under Precision-Recall curve (best for imbalanced data)
  - IoU       : intersection over union of fire pixels
"""

import numpy as np
from sklearn.metrics import (
    f1_score, roc_auc_score,
    average_precision_score,
    jaccard_score
)


def compute_metrics(
    preds_prob: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray = None,
    threshold: float = 0.5,
) -> dict:
    """
    Compute all evaluation metrics.

    Args:
        preds_prob : Predicted probabilities, shape [N] (flattened)
        targets    : Binary ground truth,    shape [N] (flattened)
        mask       : Valid pixel mask,        shape [N] (1=valid, 0=ignore)
        threshold  : Threshold to convert probs → binary predictions

    Returns:
        dict with keys: f1, auroc, auprc, iou
    """
    # Only evaluate on valid (non-uncertain) pixels
    if mask is not None:
        valid = mask.astype(bool)
        preds_prob = preds_prob[valid]
        targets    = targets[valid]

    preds_binary = (preds_prob >= threshold).astype(np.int32)
    targets_int  = targets.astype(np.int32)

    results = {}

    # F1 score on binary predictions
    results["f1"] = f1_score(targets_int, preds_binary, zero_division=0)

    # AUROC and AUPRC require both classes to be present
    if targets_int.max() > 0:
        results["auroc"] = roc_auc_score(targets_int, preds_prob)
        results["auprc"] = average_precision_score(targets_int, preds_prob)
    else:
        results["auroc"] = float("nan")
        results["auprc"] = float("nan")

    # IoU (Jaccard) for fire pixels only
    results["iou"] = jaccard_score(targets_int, preds_binary, zero_division=0)

    return results