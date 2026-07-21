"""
visualize_3d.py
---------------
Interactive 3D wildfire spread visualization using Plotly.

Renders a two-panel HTML page:
  Left  -- 3D terrain surface (elevation = Z-axis, fire probability = color)
  Right -- Geographic map (OpenStreetMap tiles, no API key required)

The output is a self-contained HTML file you can open in any browser, share
by email, or host as a static file.

Usage
-----
    python visualize_3d.py                               # sample 0, default coords
    python visualize_3d.py --idx 5                       # sample 5
    python visualize_3d.py --idx 3 --lat 34.1 --lon -118.2   # custom location
    python visualize_3d.py --idx 0 --output my_viz.html       # custom output path
    python visualize_3d.py --idx 0 --no-browser               # save only

Arguments
---------
    --idx         Test sample index (default: 0)
    --config      YAML config path  (default: configs/base.yaml)
    --predictions Path to predictions.npy; auto-generated from checkpoint if absent
    --lat         Patch center latitude  -- overrides geo.lat_center in config
    --lon         Patch center longitude -- overrides geo.lon_center in config
    --output      HTML output path (default: outputs/viz_3d_sample_<idx>.html)
    --no-browser  Write HTML without launching a browser tab
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import torch
import yaml
from plotly.subplots import make_subplots

# Allow `from models.unet import ...` when running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from models.unet import build_model  # noqa: E402


# -- Per-channel stats (must match wildfire_dataset.py exactly) ----------------

CHANNEL_MEANS = np.array(
    [1657.0, 190.0, 3.5, 281.0, 298.0, 0.007, 0.5, -1.0, 0.45, 50.0, 100.0, 0.05],
    dtype=np.float32,
)
CHANNEL_STDS = np.array(
    [649.0, 100.0, 2.5, 15.0, 16.0, 0.006, 2.0, 3.5, 0.25, 30.0, 800.0, 0.22],
    dtype=np.float32,
)

CROP_SIZE = 32  # center-crop size used by WildfireDataset for val/test

# Dark fire colorscale: near-black -> deep orange -> bright red -> hot yellow
FIRE_COLORSCALE = [
    [0.00, "rgb(8,   8,   8)"],
    [0.10, "rgb(40,  15,   5)"],
    [0.25, "rgb(120, 40,   0)"],
    [0.45, "rgb(200, 80,   0)"],
    [0.65, "rgb(230, 50,   0)"],
    [0.82, "rgb(255, 20,   0)"],
    [1.00, "rgb(255, 220,  30)"],
]


# -- Config --------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# -- Data helpers --------------------------------------------------------------

def center_crop_2d(arr: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
    """Center-crop a (H, W) array to (size, size)."""
    h, w = arr.shape
    r = (h - size) // 2
    c = (w - size) // 2
    return arr[r : r + size, c : c + size]


def normalize_and_crop(raw_x: np.ndarray) -> np.ndarray:
    """
    Replicate the WildfireDataset preprocessing for a single sample.

    Input : raw_x  shape (12, 64, 64)  -- values from test_x.npy on disk
    Output: x_norm shape (12, 32, 32)  -- z-score normalised, center-cropped
    """
    x = raw_x.copy().astype(np.float32)
    for c in range(x.shape[0]):
        x[c] = (x[c] - CHANNEL_MEANS[c]) / (CHANNEL_STDS[c] + 1e-6)
        x[c] = np.clip(x[c], -5.0, 5.0)
    h, w = x.shape[1], x.shape[2]
    r = (h - CROP_SIZE) // 2
    c_off = (w - CROP_SIZE) // 2
    return x[:, r : r + CROP_SIZE, c_off : c_off + CROP_SIZE]


def load_checkpoint(model: torch.nn.Module, path: str, device: torch.device) -> torch.nn.Module:
    """Load a checkpoint saved as a raw state-dict or as {'model_state_dict': ...}."""
    raw = torch.load(path, map_location=device, weights_only=False)
    state = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    model.load_state_dict(state)
    return model


def run_inference(config: dict, test_x: np.ndarray) -> np.ndarray:
    """
    Run the trained model over all N test samples.
    Returns predictions array of shape (N, 32, 32), values in [0, 1].
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Running inference on {device} ...")

    model = build_model(
        encoder_name=config["model"]["encoder"],
        encoder_weights=None,
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
    )
    model = load_checkpoint(model, config["eval"]["checkpoint"], device)
    model.to(device).eval()

    preds = []
    with torch.no_grad():
        for i in range(len(test_x)):
            x_norm = normalize_and_crop(test_x[i])          # (12, 32, 32)
            tensor = torch.tensor(x_norm).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()
            preds.append(prob)

    return np.stack(preds)  # (N, 32, 32)


def get_data(
    config: dict,
    pred_arg: Optional[str],
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Return (test_x, predictions, test_y_or_None).

    Predictions are loaded from file if available, otherwise generated from the
    checkpoint and cached to data/processed/predictions.npy for future runs.
    """
    processed_dir = config["data"]["processed_dir"]
    test_x = np.load(os.path.join(processed_dir, "test_x.npy"))

    # Load ground-truth labels if present
    test_y_path = os.path.join(processed_dir, "test_y.npy")
    test_y = np.load(test_y_path) if os.path.exists(test_y_path) else None

    # Resolve predictions
    candidates = [pred_arg, os.path.join(processed_dir, "predictions.npy")]
    for path in candidates:
        if path and os.path.exists(path):
            print(f"Loading predictions from {path}")
            return test_x, np.load(path), test_y

    print("No predictions.npy found -- generating from checkpoint ...")
    predictions = run_inference(config, test_x)
    save_path = os.path.join(processed_dir, "predictions.npy")
    np.save(save_path, predictions)
    print(f"  Cached predictions -> {save_path}")
    return test_x, predictions, test_y


# -- Geographic grid -----------------------------------------------------------

def make_latlon_grid(
    lat_center: float,
    lon_center: float,
    grid_size: int = CROP_SIZE,
    meters_per_pixel: float = 375.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (lat_grid, lon_grid), each shape (grid_size, grid_size).

    Uses a flat-earth approximation valid for the ~12 km patches in this dataset
    (error < 0.1 % at mid-latitudes).  Coordinates are illustrative unless the
    caller supplies the true patch centre from real metadata.
    """
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat_center))

    deg_lat = meters_per_pixel / m_per_deg_lat
    deg_lon = meters_per_pixel / m_per_deg_lon

    half = grid_size / 2.0
    lats = np.linspace(lat_center + half * deg_lat, lat_center - half * deg_lat, grid_size)
    lons = np.linspace(lon_center - half * deg_lon, lon_center + half * deg_lon, grid_size)

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return lat_grid, lon_grid


# -- Figure builder ------------------------------------------------------------

def build_figure(
    elevation: np.ndarray,             # (32, 32)  metres
    fire_prob: np.ndarray,             # (32, 32)  0-1
    prev_fire: np.ndarray,             # (32, 32)  binary-ish
    ground_truth: Optional[np.ndarray],# (32, 32) or None
    lat_grid: np.ndarray,              # (32, 32)
    lon_grid: np.ndarray,              # (32, 32)
    sample_idx: int,
) -> go.Figure:
    """Assemble the two-panel Plotly figure."""

    fire_rows, fire_cols = np.where(prev_fire > 0.5)
    lat_f  = lat_grid.flatten()
    lon_f  = lon_grid.flatten()
    prob_f = fire_prob.flatten()
    visible = prob_f > 0.05   # only render pixels with meaningful probability

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.56, 0.44],
        specs=[[{"type": "scene"}, {"type": "mapbox"}]],
        subplot_titles=[
            "3D Terrain  .  Fire Probability Overlay",
            "Geographic Placement",
        ],
        horizontal_spacing=0.02,
    )

    # -- Left panel: 3D surface ------------------------------------------------

    fig.add_trace(
        go.Surface(
            z=elevation,
            surfacecolor=fire_prob,
            colorscale=FIRE_COLORSCALE,
            cmin=0.0, cmax=1.0,
            colorbar=dict(
                title=dict(text="Fire<br>Probability", font=dict(size=12)),
                # x=0.0 / xanchor="left" places the bar along the LEFT edge of
                # the figure, well clear of the mapbox panel on the right.
                x=0.0, xanchor="left",
                len=0.70, thickness=14,
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=["0 %", "25 %", "50 %", "75 %", "100 %"],
                tickfont=dict(size=10),
            ),
            lighting=dict(ambient=0.65, diffuse=0.85, specular=0.2, roughness=0.6),
            lightposition=dict(x=100, y=-200, z=500),
            hovertemplate="Elevation: %{z:.0f} m<br>Row %{y} . Col %{x}<extra></extra>",
            name="Terrain",
            showlegend=True,
        ),
        row=1, col=1,
    )

    # Active fire locations floating 100 m above the surface
    if len(fire_rows) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=fire_cols.tolist(),
                y=fire_rows.tolist(),
                z=(elevation[fire_rows, fire_cols] + 100).tolist(),
                mode="markers",
                marker=dict(size=4, color="#FF7700", opacity=0.9, symbol="circle"),
                name="Active Fire (t)",
                hovertemplate="Active fire<br>Row %{y} . Col %{x}<extra></extra>",
            ),
            row=1, col=1,
        )

    # Ground-truth next-day fire, 60 m above surface (cyan, distinguishable from prediction)
    if ground_truth is not None:
        gt_rows, gt_cols = np.where(ground_truth > 0.5)
        if len(gt_rows) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=gt_cols.tolist(),
                    y=gt_rows.tolist(),
                    z=(elevation[gt_rows, gt_cols] + 60).tolist(),
                    mode="markers",
                    marker=dict(size=3, color="#00CCFF", opacity=0.7, symbol="circle"),
                    name="Ground Truth (t+1)",
                    hovertemplate="Ground truth<br>Row %{y} . Col %{x}<extra></extra>",
                ),
                row=1, col=1,
            )

    # -- Right panel: map ------------------------------------------------------

    fig.add_trace(
        go.Densitymapbox(
            lat=lat_f[visible].tolist(),
            lon=lon_f[visible].tolist(),
            z=prob_f[visible].tolist(),
            radius=12,
            colorscale=FIRE_COLORSCALE,
            zmin=0.0, zmax=1.0,
            showscale=False,
            opacity=0.80,
            name="Fire Probability",
            hovertemplate="Prob: %{z:.1%}<extra></extra>",
        ),
        row=1, col=2,
    )

    if len(fire_rows) > 0:
        fig.add_trace(
            go.Scattermapbox(
                lat=lat_grid[fire_rows, fire_cols].tolist(),
                lon=lon_grid[fire_rows, fire_cols].tolist(),
                mode="markers",
                marker=dict(size=7, color="#FF7700", opacity=0.9),
                name="Active Fire (t)",
                hovertemplate="Active fire<extra></extra>",
                showlegend=False,
            ),
            row=1, col=2,
        )

    # -- Layout ----------------------------------------------------------------

    fig.update_layout(
        # Title is intentionally blank — the Streamlit app supplies its own heading
        # above the chart, so we don't want a duplicate title inside the figure.
        title=dict(text="", x=0.5),
        height=700,
        paper_bgcolor="#0d0d0d",
        font=dict(color="#dddddd", family="Inter, Segoe UI, Arial, sans-serif"),
        scene=dict(
            xaxis=dict(title="Col (W->E)", showgrid=False, zeroline=False, showbackground=False),
            yaxis=dict(title="Row (N->S)", showgrid=False, zeroline=False, showbackground=False),
            zaxis=dict(
                title="Elevation (m)",
                gridcolor="rgba(255,255,255,0.06)",
                zerolinecolor="rgba(255,255,255,0.10)",
            ),
            bgcolor="#0d0d0d",
            camera=dict(eye=dict(x=1.55, y=-1.55, z=0.85)),
            aspectratio=dict(x=1.0, y=1.0, z=0.32),
        ),
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=float(lat_grid.mean()), lon=float(lon_grid.mean())),
            zoom=9,
        ),
        legend=dict(
            bgcolor="rgba(20,20,20,0.85)",
            bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1,
            x=0.01, y=0.99,
            font=dict(size=11),
        ),
        margin=dict(t=55, b=5, l=5, r=5),
    )

    return fig


# -- Entry point ---------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive 3D wildfire spread visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--idx",         type=int,   default=0,
                        help="Test sample index (default: 0)")
    parser.add_argument("--config",      default="configs/base.yaml",
                        help="YAML config path")
    parser.add_argument("--predictions", default=None,
                        help="Path to predictions.npy (auto-generated if absent)")
    parser.add_argument("--lat",         type=float, default=None,
                        help="Patch centre latitude  (overrides geo.lat_center in config)")
    parser.add_argument("--lon",         type=float, default=None,
                        help="Patch centre longitude (overrides geo.lon_center in config)")
    parser.add_argument("--output",      default=None,
                        help="HTML output path (default: outputs/viz_3d_sample_<idx>.html)")
    parser.add_argument("--no-browser",  action="store_true",
                        help="Save HTML without opening a browser tab")
    args = parser.parse_args()

    config = load_config(args.config)
    geo    = config.get("geo", {})

    lat_center       = args.lat if args.lat is not None else geo.get("lat_center", 37.5)
    lon_center       = args.lon if args.lon is not None else geo.get("lon_center", -119.5)
    meters_per_pixel = geo.get("meters_per_pixel", 375.0)

    test_x, predictions, test_y = get_data(config, args.predictions)

    n   = len(test_x)
    idx = max(0, min(args.idx, n - 1))
    if idx != args.idx:
        print(f"Index {args.idx} out of range (0-{n - 1}), using {idx}.")

    # Raw elevation in metres (channel 0), center-cropped to 32×32 to align
    # with the 32×32 predictions produced by the model.
    elevation = center_crop_2d(test_x[idx, 0]).astype(np.float32)
    prev_fire = center_crop_2d(test_x[idx, 11]).astype(np.float32)
    fire_prob = predictions[idx].astype(np.float32)              # (32, 32)

    ground_truth = None
    if test_y is not None:
        gt_raw = center_crop_2d(test_y[idx, 0])
        ground_truth = np.where(gt_raw > 0.5, 1.0, 0.0).astype(np.float32)

    lat_grid, lon_grid = make_latlon_grid(
        lat_center, lon_center,
        grid_size=CROP_SIZE,
        meters_per_pixel=meters_per_pixel,
    )

    fig = build_figure(elevation, fire_prob, prev_fire, ground_truth, lat_grid, lon_grid, idx)

    out_dir = os.path.dirname(args.output) if args.output else "outputs"
    out_path = args.output or os.path.join("outputs", f"viz_3d_sample_{idx}.html")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Saved -> {out_path}")

    if not args.no_browser:
        fig.show()


if __name__ == "__main__":
    main()
