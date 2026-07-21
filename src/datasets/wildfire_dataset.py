"""
wildfire_dataset.py
───────────────────
PyTorch Dataset for the Next Day Wildfire Spread dataset.

Handles:
  - Loading preprocessed .npy arrays (already NaN/inf-free from preprocess.py)
  - Per-channel normalization (z-score) processed one channel at a time
  - Random crop augmentation for training (32x32 from 64x64)
  - Center crop for val/test (fixed 32x32 center window per benchmark paper)
  - Ignoring uncertain fire mask pixels (value = -1 in the original dataset)
"""

import numpy as np
import torch
from torch.utils.data import Dataset


# Per-channel mean and std — approximate values from the benchmark paper
# Index order matches INPUT_FEATURES order in configs/base.yaml exactly
CHANNEL_MEANS = np.array([
    1657.0,   # elevation (m)
    190.0,    # wind direction (degrees)
    3.5,      # wind speed (m/s)
    281.0,    # min temp (K)
    298.0,    # max temp (K)
    0.007,    # specific humidity (kg/kg)
    0.5,      # precipitation (mm)
    -1.0,     # PDSI (drought index)
    0.45,     # NDVI
    50.0,     # ERC
    100.0,    # population density
    0.05,     # previous fire mask
], dtype=np.float32)

CHANNEL_STDS = np.array([
    649.0,   # elevation
    100.0,   # wind direction
    2.5,     # wind speed
    15.0,    # min temp
    16.0,    # max temp
    0.006,   # specific humidity
    2.0,     # precipitation
    3.5,     # PDSI
    0.25,    # NDVI
    30.0,    # ERC
    800.0,   # population density
    0.22,    # previous fire mask
], dtype=np.float32)


class WildfireDataset(Dataset):
    """
    Loads preprocessed wildfire spread samples.

    Args:
        x_path    : Path to input .npy file, shape [N, 12, 64, 64]
        y_path    : Path to label .npy file, shape [N,  1, 64, 64]
        split     : "train" | "val" | "test"
        crop_size : Spatial crop size (default 32, matches benchmark paper)
    """

    def __init__(self, x_path: str, y_path: str, split: str = "train", crop_size: int = 32):
        self.x = np.load(x_path)  # [N, 12, 64, 64]
        self.y = np.load(y_path)  # [N,  1, 64, 64]
        self.split = split
        self.crop_size = crop_size

        # Normalize and clip each channel independently to avoid allocating
        # large temporary arrays over the full [N, 12, H, W] tensor at once.
        # Each iteration operates on shape [N, 64, 64] (~58 MB max).
        for c in range(self.x.shape[1]):
            self.x[:, c, :, :] = (
                self.x[:, c, :, :] - CHANNEL_MEANS[c]
            ) / (CHANNEL_STDS[c] + 1e-6)
            # Clamp to [-5, 5] to suppress extreme outliers
            self.x[:, c, :, :] = np.clip(self.x[:, c, :, :], -5.0, 5.0)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]  # [12, 64, 64]
        y = self.y[idx]  # [ 1, 64, 64]

        # Spatially crop the sample
        x, y = self._crop(x, y)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        # Valid pixel mask: uncertain pixels are stored as -1 in the raw label.
        # Mark them as invalid (0) so they are excluded from loss computation.
        valid_mask = (y >= 0).float()

        # Clamp label to binary [0, 1] — suppresses residual -1 uncertainty values
        y = torch.clamp(y, 0.0, 1.0)

        return {"x": x, "y": y, "mask": valid_mask}

    def _crop(self, x: np.ndarray, y: np.ndarray):
        """
        Training : random 32x32 crop anywhere in the 64x64 grid (augmentation).
        Val/Test : fixed center 32x32 crop (deterministic, matches benchmark eval).
        """
        H, W = x.shape[1], x.shape[2]
        cs   = self.crop_size

        if self.split == "train":
            top  = np.random.randint(0, H - cs)
            left = np.random.randint(0, W - cs)
        else:
            top  = (H - cs) // 2
            left = (W - cs) // 2

        return x[:, top:top+cs, left:left+cs], y[:, top:top+cs, left:left+cs]


def compute_stats(x_path: str):
    """
    Utility: recompute per-channel mean and std from a training .npy file.
    Call this once if you want to recalibrate the CHANNEL_MEANS/STDS constants.
    Processes one channel at a time to stay within memory limits.
    """
    x = np.load(x_path)  # [N, C, H, W]
    means, stds = [], []
    for c in range(x.shape[1]):
        channel = x[:, c, :, :]
        means.append(channel.mean())
        stds.append(channel.std())
    means = np.array(means)
    stds  = np.array(stds)
    print("Means:", means)
    print("Stds :", stds)
    return means, stds