"""
eval.py
───────
Evaluates the best saved model checkpoint on the test set.
Generates:
  - Printed metrics table (F1, AUROC, AUPRC, IoU)
  - PNG visualizations: input fire mask, prediction, ground truth side by side

Usage:
    python src/eval.py
"""

import os
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from datasets.wildfire_dataset import WildfireDataset
from models.unet import build_model
from utils.metrics import compute_metrics

with open("configs/base.yaml", "r") as f:
    cfg = yaml.safe_load(f)

PROC   = cfg["data"]["processed_dir"]
MCFG   = cfg["model"]
ECFG   = cfg["eval"]
TCFG   = cfg["training"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_prediction(x, y_true, y_pred_prob, idx, out_dir, threshold=0.5):
    """
    Saves a 3-panel plot:
      Left:   Previous fire mask (input channel 11)
      Center: Predicted spread probability map
      Right:  Ground truth next-day fire mask
    """
    prev_fire = x[11]   # channel index 11 = PrevFireMask
    pred_bin  = (y_pred_prob[0] >= threshold).astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(prev_fire,         cmap="Reds",  vmin=0, vmax=1)
    axes[0].set_title("Previous Fire Mask (Input)")
    axes[1].imshow(y_pred_prob[0],    cmap="hot",   vmin=0, vmax=1)
    axes[1].set_title("Predicted Spread Probability")
    axes[2].imshow(y_true[0],         cmap="Reds",  vmin=0, vmax=1)
    axes[2].set_title("Ground Truth (Next Day)")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"sample_{idx:04d}.png"), dpi=150)
    plt.close()


def main():
    # Load test dataset
    test_ds = WildfireDataset(
        x_path=os.path.join(PROC, "test_x.npy"),
        y_path=os.path.join(PROC, "test_y.npy"),
        split="test",
        crop_size=TCFG["random_crop_size"],
    )
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    # Load model and checkpoint
    model = build_model(
        encoder_name=MCFG["encoder"],
        encoder_weights=None,        # Don't load ImageNet weights — we'll load our checkpoint
        in_channels=MCFG["in_channels"],
        out_channels=MCFG["out_channels"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(ECFG["checkpoint"], map_location=DEVICE))
    model.eval()

    all_probs, all_targets, all_masks = [], [], []
    all_pred_maps = []   # full (32, 32) maps — saved as predictions.npy
    vis_samples = []  # Store a few samples for visualization

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            x    = batch["x"].to(DEVICE)
            y    = batch["y"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            logits = model(x)
            probs  = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy().flatten())
            all_targets.append(y.cpu().numpy().flatten())
            all_masks.append(mask.cpu().numpy().flatten())
            # Collect spatial probability maps for downstream use (visualize_3d.py)
            all_pred_maps.append(probs.squeeze(1).cpu().numpy())  # (B, 32, 32)

            # Save the first 5 batches worth of individual samples for visualization
            if len(vis_samples) < 10:
                for j in range(x.shape[0]):
                    vis_samples.append({
                        "x":    x[j].cpu().numpy(),
                        "y":    y[j].cpu().numpy(),
                        "pred": probs[j].cpu().numpy(),
                    })

    # Compute metrics
    all_probs   = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    all_masks   = np.concatenate(all_masks)
    metrics = compute_metrics(all_probs, all_targets, all_masks, threshold=ECFG["threshold"])

    print("\n─── Test Set Results ───────────────────────────────")
    for name, val in metrics.items():
        print(f"  {name.upper():<8}: {val:.4f}")
    print("────────────────────────────────────────────────────\n")

    # Save full prediction maps so visualize_3d.py can load them without re-running inference
    pred_maps = np.concatenate(all_pred_maps, axis=0)  # (N, 32, 32)
    pred_path = os.path.join(PROC, "predictions.npy")
    np.save(pred_path, pred_maps)
    print(f"Saved predictions.npy ({pred_maps.shape}) → {pred_path}")

    # Save visualizations for the first few test samples
    for idx, s in enumerate(vis_samples[:10]):
        visualize_prediction(
            x=s["x"], y_true=s["y"], y_pred_prob=s["pred"],
            idx=idx, out_dir=ECFG["output_dir"], threshold=ECFG["threshold"]
        )
    print(f"Saved {len(vis_samples[:10])} visualizations → {ECFG['output_dir']}")


if __name__ == "__main__":
    main()