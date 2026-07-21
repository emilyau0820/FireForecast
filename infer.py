"""
infer.py
────────
Run inference on a single sample by index and display a visualization.
Useful for demos and debugging.

Usage:
    python infer.py --idx 42
"""

import argparse
import os
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.datasets.wildfire_dataset import WildfireDataset
from src.models.unet import build_model

with open("configs/base.yaml", "r") as f:
    cfg = yaml.safe_load(f)

PROC   = cfg["data"]["processed_dir"]
MCFG   = cfg["model"]
ECFG   = cfg["eval"]
TCFG   = cfg["training"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_NAMES = cfg["features"]["input_features"]


def run_inference(idx: int):
    # Load test dataset
    test_ds = WildfireDataset(
        x_path=os.path.join(PROC, "test_x.npy"),
        y_path=os.path.join(PROC, "test_y.npy"),
        split="test",
        crop_size=TCFG["random_crop_size"],
    )
    sample = test_ds[idx]
    x    = sample["x"].unsqueeze(0).to(DEVICE)  # add batch dim → [1, 12, 32, 32]
    y    = sample["y"].numpy()                  # [1, 32, 32]

    # Load model
    model = build_model(
        encoder_name=MCFG["encoder"],
        encoder_weights=None,
        in_channels=MCFG["in_channels"],
        out_channels=MCFG["out_channels"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(ECFG["checkpoint"], map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        prob = torch.sigmoid(model(x)).squeeze().cpu().numpy()  # [32, 32]

    # ─── Plot: all input channels + prediction + ground truth ─────────────────
    n_channels = x.shape[1]
    n_cols = 5
    n_rows = (n_channels + 2 + n_cols - 1) // n_cols   # +2 for pred and GT

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = axes.flatten()

    # Show each input channel
    for c in range(n_channels):
        axes[c].imshow(x[0, c].cpu().numpy(), cmap="viridis")
        axes[c].set_title(FEATURE_NAMES[c], fontsize=8)
        axes[c].axis("off")

    # Show prediction and ground truth
    axes[n_channels].imshow(prob,    cmap="hot", vmin=0, vmax=1)
    axes[n_channels].set_title("Predicted Probability", fontsize=8)
    axes[n_channels].axis("off")

    axes[n_channels + 1].imshow(y[0], cmap="Reds", vmin=0, vmax=1)
    axes[n_channels + 1].set_title("Ground Truth", fontsize=8)
    axes[n_channels + 1].axis("off")

    # Hide any unused axes
    for ax in axes[n_channels + 2:]:
        ax.axis("off")

    plt.suptitle(f"Test Sample #{idx}", fontsize=12)
    plt.tight_layout()

    out_path = os.path.join(ECFG["output_dir"], f"infer_sample_{idx}.png")
    os.makedirs(ECFG["output_dir"], exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved inference visualization → {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0, help="Test sample index to visualize")
    args = parser.parse_args()
    run_inference(args.idx)