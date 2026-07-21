"""
train.py
────────
Full training loop for the wildfire spread U-Net.

Usage:
    python src/train.py

Logs metrics and prediction visualizations to Weights & Biases.
Saves best checkpoint (by val AUPRC) to checkpoints/best_model.pth.
"""

import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import wandb

from datasets.wildfire_dataset import WildfireDataset
from models.unet import build_model, count_parameters
from utils.losses import CombinedLoss
from utils.metrics import compute_metrics

# ─── Load Config ──────────────────────────────────────────────────────────────
with open("configs/base.yaml", "r") as f:
    cfg = yaml.safe_load(f)

PROC  = cfg["data"]["processed_dir"]
TCFG  = cfg["training"]
MCFG  = cfg["model"]
LCFG  = cfg["logging"]
DEVICE = torch.device(TCFG["device"] if torch.cuda.is_available() else "cpu")
torch.manual_seed(TCFG["seed"])


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    """Run one full pass over the training set and return mean loss."""
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="  train", leave=False):
        x      = batch["x"].to(device)      # [B, 12, 32, 32]
        y      = batch["y"].to(device)      # [B,  1, 32, 32]
        mask   = batch["mask"].to(device)   # [B,  1, 32, 32]

        optimizer.zero_grad()
        logits = model(x)                   # [B,  1, 32, 32]
        loss   = criterion(logits, y, mask)
        loss.backward()

        # Gradient clipping prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, threshold=0.5):
    """Run one pass over the validation set. Returns loss and metrics dict."""
    model.eval()
    total_loss = 0.0
    all_probs, all_targets, all_masks = [], [], []

    for batch in tqdm(loader, desc="    val", leave=False):
        x    = batch["x"].to(device)
        y    = batch["y"].to(device)
        mask = batch["mask"].to(device)

        logits = model(x)
        loss   = criterion(logits, y, mask)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)

        # Collect for metric computation (move to CPU/numpy)
        all_probs.append(probs.cpu().numpy().flatten())
        all_targets.append(y.cpu().numpy().flatten())
        all_masks.append(mask.cpu().numpy().flatten())

    all_probs   = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    all_masks   = np.concatenate(all_masks)

    metrics = compute_metrics(all_probs, all_targets, all_masks, threshold)
    return total_loss / len(loader), metrics


def main():
    # ─── W&B Init ─────────────────────────────────────────────────────────────
    wandb.init(
        project=LCFG["project"],
        entity=LCFG.get("entity"),
        config={**TCFG, **MCFG},
    )

    # ─── Datasets & Loaders ───────────────────────────────────────────────────
    train_ds = WildfireDataset(
        x_path=os.path.join(PROC, "train_x.npy"),
        y_path=os.path.join(PROC, "train_y.npy"),
        split="train",
        crop_size=TCFG["random_crop_size"],
    )
    val_ds = WildfireDataset(
        x_path=os.path.join(PROC, "val_x.npy"),
        y_path=os.path.join(PROC, "val_y.npy"),
        split="val",
        crop_size=TCFG["random_crop_size"],
    )

    train_loader = DataLoader(train_ds, batch_size=TCFG["batch_size"], shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=TCFG["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # ─── Model, Loss, Optimizer, Scheduler ────────────────────────────────────
    model = build_model(
        encoder_name=MCFG["encoder"],
        encoder_weights=MCFG["encoder_weights"],
        in_channels=MCFG["in_channels"],
        out_channels=MCFG["out_channels"],
    ).to(DEVICE)
    print(f"Model parameters: {count_parameters(model):,}")

    criterion = CombinedLoss(
        gamma=TCFG["focal_gamma"],
        pos_weight=TCFG["pos_weight"],
    )
    optimizer = Adam(model.parameters(), lr=TCFG["learning_rate"], weight_decay=TCFG["weight_decay"])

    # OneCycleLR: warm up then cosine anneal — works well with ResNet encoders [web:30]
    scheduler = OneCycleLR(
        optimizer,
        max_lr=TCFG["learning_rate"],
        steps_per_epoch=len(train_loader),
        epochs=TCFG["epochs"],
    )

    os.makedirs(LCFG["save_dir"], exist_ok=True)
    best_auprc = 0.0

    # ─── Training Loop ────────────────────────────────────────────────────────
    for epoch in range(1, TCFG["epochs"] + 1):
        print(f"\nEpoch {epoch}/{TCFG['epochs']}")

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, DEVICE)
        val_loss, val_metrics = validate(model, val_loader, criterion, DEVICE)

        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val   loss: {val_loss:.4f} | F1: {val_metrics['f1']:.3f} | "
              f"AUROC: {val_metrics['auroc']:.3f} | AUPRC: {val_metrics['auprc']:.3f} | "
              f"IoU: {val_metrics['iou']:.3f}")

        # Log everything to W&B
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            **{f"val/{k}": v for k, v in val_metrics.items()},
        })

        # Save best model by AUPRC (best metric for imbalanced binary classification)
        if val_metrics["auprc"] > best_auprc:
            best_auprc = val_metrics["auprc"]
            ckpt_path = os.path.join(LCFG["save_dir"], "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved new best model (AUPRC={best_auprc:.4f}) → {ckpt_path}")

    wandb.finish()
    print("\nTraining complete.")


if __name__ == "__main__":
    main()